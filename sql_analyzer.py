"""
SQLAnalyzer — Multi-step reasoning engine for complex analytical questions.

Architecture:
  1. Decompose: Analyzer LLM breaks question into simple sub-questions
  2. Execute:   Each sub-question runs through the existing pipeline (process_query)
  3. Inspect:   Analyzer reviews results and can modify remaining steps
  4. Synthesize: Analyzer combines results into a final SQL query
  5. Review:    Opus Review validates the final SQL

The Analyzer LLM is Claude Sonnet. Sub-queries reuse process_query() from
flow_router.py — no pipeline reimplementation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnalyzerResult:
    """Complete result returned by SQLAnalyzer.analyze()."""
    plan: Dict                  # Decomposition plan from Step 1
    sub_queries: List[Dict]     # [{step_id, question, sql, results_summary, tokens}]
    synthesis_sql: str          # Final combined SQL
    final_results: Any          # Executed query results
    opus_verdict: str           # CORRECT / INCORRECT / UNCERTAIN / NOT_REVIEWED
    total_tokens: Dict          # Breakdown by stage
    trace: List[Dict]           # Full trace for debugging


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """You are an expert SQL analyst. Your job is to break down complex analytical questions into simple sub-questions, where each sub-question can be answered by a single SQL query with no CTEs.

Rules for decomposition:
1. PRESERVE INTENT — If the user asks for "highest/top/best," the sub-question must include "ordered by [metric] descending, returning the top result." If "lowest/worst," order ascending. If "why/explain," include a dimensional breakdown step. Never neutralize a ranking question into a plain GROUP BY.
2. MINIMIZE STEPS — Target 2–3 steps, max 5. Combine multiple breakdowns of the same entity from the same table into one step. Only separate steps when different tables are needed or a step's result is required as a filter for the next step.
3. ANTICIPATE DATA — Open/pending records have future dates; closed/completed have past dates. Do not apply past-date filters to open records. If comparing actuals vs pipeline/forecast, plan for different source tables.
4. EXACT REFERENCES — Use exact schema-prefixed table names and exact column names from the schema. Never paraphrase.
5. NO HARDCODED VALUES — Describe filter intent in plain language, never hardcode literals. Exception: values returned by a previous step CAN be used literally since they came from the actual database.
6. APPLY BUSINESS RULES — Provided business rules are mandatory constraints.
7. SPECIFY RESULTS — State exact columns, aggregates, and sort order each step returns. If identifying a top/bottom entity, results must be sorted and limited so the entity is unambiguous.
8. PLAN SYNTHESIS — If sub-query results directly answer the question, note that synthesis should narrate results (no new SQL needed). If a combining query is needed, plan steps to gather its inputs. Synthesis must never contradict sub-query findings.

Return a JSON object ONLY (no markdown, no explanation outside the JSON)."""


def _build_decompose_prompt(
    question: str,
    bare_schema: str,
    rules_compressed: str,
    dialect_info: Dict,
) -> str:
    from flow_router import get_dialect_syntax_rules
    dialect = dialect_info.get("dialect", "postgresql") if dialect_info else "postgresql"
    quote_char = dialect_info.get("quote_char", '"') if dialect_info else '"'
    dialect_rules = get_dialect_syntax_rules(dialect)
    rules_text = rules_compressed if rules_compressed and rules_compressed != "[]" else "None"

    # Build a concrete example of good vs bad sub_question using the actual quoting style
    if dialect == "postgresql":
        good_example = 'What is the total "sales_amount" from "public"."CRM" grouped by "region"?'
        bad_example = "What is the total sales from CRM by region?"
    elif dialect == "mysql":
        good_example = "What is the total `sales_amount` from `mydb`.`CRM` grouped by `region`?"
        bad_example = "What is the total sales from CRM by region?"
    elif dialect in ("mssql", "sqlserver"):
        good_example = "What is the total [sales_amount] from [dbo].[CRM] grouped by [region]?"
        bad_example = "What is the total sales from CRM by region?"
    else:
        good_example = 'What is the total "sales_amount" from "public"."CRM" grouped by "region"?'
        bad_example = "What is the total sales from CRM by region?"

    return f"""Question: {question}

{dialect_rules}

IMPORTANT — table and column naming in sub_questions:
- Use EXACT schema-qualified table names as shown in the schema below (e.g. if schema shows "public"."CRM", the sub_question must say "public"."CRM" — never bare "CRM").
- Use EXACT column names as shown in the schema — never paraphrase or abbreviate them.
- Good sub_question: "{good_example}"
- Bad sub_question:  "{bad_example}"  ← missing schema prefix and exact column names

IMPORTANT — filter values in sub_questions (applies to ALL steps equally):
- NEVER hardcode specific filter values such as status names, stage names, entity names, category values, or any string literal that represents data stored in the database.
- Instead, describe the INTENT of the filter in plain English. The pipeline's Entity Resolver will automatically match your intent to the actual values stored in this specific database.
- The principle: if a filter value is a quoted string you invented or guessed, replace it with a plain-English description of what that value means.
- Generic examples of the principle (apply the same logic to whatever tables/columns are in your schema):
    ✗ Bad (hardcodes literals): "Get records where the status is 'Active' and the priority is 'High'"
    ✓ Good (describes intent): "Get records that are currently active and flagged as high priority"
    ✗ Bad (hardcodes literals): "Count rows where category is not 'Trial' and region equals 'West'"
    ✓ Good (describes intent): "Count non-trial rows in the western region"

Schema (exact table and column names to use):
{bare_schema}

Business rules:
{rules_text}

Break the question into simple sub-questions. Each sub-question must:
• Be answerable with a single SQL query (no CTEs needed)
• Name the exact schema-qualified table(s) involved
• Name the exact column(s) involved
• Describe any filters by INTENT in plain English — never by hardcoded string values

Return this exact JSON structure:
{{
  "analysis_type": "what_if_simulation | trend_analysis | comparative_ranking | impact_analysis | multi_metric",
  "reasoning": "Why this needs multi-step analysis",
  "steps": [
    {{
      "step_id": 1,
      "description": "What this step computes",
      "sub_question": "A question naming exact tables/columns with filters described by intent, not by hardcoded values",
      "depends_on": [],
      "result_usage": "How this result will be used in later steps"
    }}
  ],
  "synthesis_approach": "How to combine sub-query results into the final answer"
}}"""


def _build_inspect_prompt(
    question: str,
    plan: Dict,
    completed: List[Dict],
    remaining_steps: List[Dict],
) -> str:
    completed_text = ""
    for item in completed:
        completed_text += f"\nStep {item['step_id']}: {item['sub_question']}\n"
        completed_text += f"  SQL: {item.get('sql', 'N/A')}\n"
        completed_text += f"  Results: {item.get('results_summary', 'N/A')}\n"
        if item.get('identified_entity'):
            completed_text += f"  Identified entity: {item['identified_entity']}\n"

    remaining_text = json.dumps(remaining_steps, indent=2)

    return f"""Original question: {question}

Analysis type: {plan.get('analysis_type', 'unknown')}

Completed steps so far:
{completed_text}

Remaining steps in plan:
{remaining_text}

Based on the results so far, decide what to do next.

Return a JSON object ONLY:
{{
  "decision": "proceed | modify | synthesize | abort",
  "reason": "Brief explanation",
  "updated_remaining_steps": [
    {{
      "step_id": <int>,
      "description": "...",
      "sub_question": "...",
      "depends_on": [],
      "result_usage": "..."
    }}
  ],
  "abort_message": "Message to user if aborting (e.g. no data found)"
}}

RULES FOR MODIFYING SUB-QUESTIONS (applies when decision = "modify"):
- If a completed step has an "Identified entity" shown above, UPDATE remaining sub-questions
  to use that exact value as a filter — it came from the database and is safe to use literally.
- Modified sub-questions must NOT hardcode values that were NOT discovered by a prior step —
  describe intent in plain language instead.
- If a completed step returned 0 rows and its filters involved dates, check whether
  the date filter was compatible with the record type (e.g., open/pipeline records have
  future dates — don't filter them with < CURRENT_DATE). Modify the step to fix the filter
  rather than aborting.

CRITICAL: You must NOT use "synthesize" just because you think you have enough data.
The decomposition plan was carefully designed — ALL planned steps are needed for a complete and analytically sound answer.

Use "synthesize" ONLY in these specific cases:
  (a) All planned steps are already complete (the remaining_steps list above is empty), OR
  (b) A completed step returned 0 rows, making all remaining steps logically impossible (nothing to join/filter against), OR
  (c) A completed step FAILED with an error AND the remaining steps explicitly depend on its output.

Use "abort" if the data shows the question cannot be answered at all (e.g. the primary dataset returned 0 rows).
Use "modify" if remaining steps need to change based on what you learned (e.g. a column name or value differs from expectation, or an identified entity should be substituted in).
Use "proceed" to continue with remaining steps as-is. This is the DEFAULT — use it unless one of the synthesize/abort/modify conditions above strictly applies. Do NOT skip steps just because you feel the data collected so far is sufficient."""


def _build_synthesis_prompt(
    question: str,
    completed: List[Dict],
    synthesis_approach: str,
    dialect_info: Dict,
) -> str:
    from flow_router import get_dialect_syntax_rules
    dialect = dialect_info.get("dialect", "postgresql") if dialect_info else "postgresql"
    dialect_rules = get_dialect_syntax_rules(dialect)
    steps_text = ""
    for item in completed:
        steps_text += f"\nStep {item['step_id']}: {item['sub_question']}\n"
        steps_text += f"  SQL used: {item.get('sql', 'N/A')}\n"
        steps_text += f"  Results summary: {item.get('results_summary', 'N/A')}\n"
        if item.get('identified_entity'):
            steps_text += f"  Identified entity: {item['identified_entity']}\n"

    return f"""Original question: {question}

{dialect_rules}

Synthesis approach: {synthesis_approach}

Sub-queries already validated and executed:
{steps_text}

STEP 1 — Choose synthesis type:
- "narrate": Sub-query results already completely answer the question. No new SQL needed.
  Choose when all steps hit the same table and the data returned is complete and unambiguous.
- "cte": A new SQL query is needed to combine data from multiple tables or perform
  cross-step calculations that the sub-queries alone cannot express.

STEP 2 — Produce the result based on synthesis type:

For "narrate":
  Write a structured summary of the sub-query results that answers the original question.
  Reference specific numbers and entities from the results shown above.
  Do NOT write any SQL.

For "cte":
  Write the COMPLETE final SQL query directly — NOT a sub-question for another pipeline.
  Rules for the SQL:
  - Use exact table names and column names from the sub-query SQL shown above (copy them exactly)
  - Must be consistent with sub-query findings (if sub-queries found Entity X as top, this SQL must also produce Entity X)
  - Exactly ONE SQL statement — never multiple SELECTs separated by semicolons or otherwise
  - Copy working WHERE clauses from the sub-query SQL; do not rewrite date logic differently
  - For simulation/what-if queries, use this exact CTE pattern:
      BASELINE CTE: Current metric per group with RANK()
      ITEMS CTE: Items being simulated
      SIMULATION CTE: CROSS JOIN baseline groups with items.
        For each item, recalculate EVERY group's metric using CASE
        (only the matching group gets modified, others stay at baseline).
        Re-rank ALL groups per item: RANK() OVER (PARTITION BY item_id ORDER BY simulated_metric DESC)
      COMPARISON: Filter to rows where the item's OWN group changed rank.
        Use WHERE item.group_key = baseline.group_key AND simulated_rank <> baseline_rank
        to avoid returning side-effect rows for other groups.

Return ONLY a JSON object:
{{
  "synthesis_type": "narrate | cte",
  "result": "...narration text OR complete SQL depending on synthesis_type...",
  "reasoning": "Why this synthesis approach works"
}}"""


# ---------------------------------------------------------------------------
# Helper: summarise QueryResult rows for the Analyzer LLM
# ---------------------------------------------------------------------------

def _summarise_results(query_result) -> str:
    """Return a compact text summary of a QueryResult for LLM consumption."""
    if not query_result or not query_result.success:
        err = getattr(query_result, "error", "unknown error") if query_result else "no result"
        return f"FAILED: {err}"

    results = query_result.results
    if results is None:
        return "Query succeeded but returned no data."

    # results may be a list of dicts (rows), a list of tuples, or a DataFrame
    try:
        import pandas as pd
        if isinstance(results, pd.DataFrame):
            row_count = len(results)
            cols = list(results.columns)
            if row_count <= 50:
                all_rows = results.to_dict(orient="records")
                return (
                    f"Returned {row_count} row(s). Columns: {cols}. "
                    f"All rows: {all_rows}"
                )
            else:
                sample_rows = results.head(5).to_dict(orient="records")
                return (
                    f"Returned {row_count} row(s). Columns: {cols}. "
                    f"First 5 of {row_count} rows: {sample_rows}"
                )
    except ImportError:
        pass

    if isinstance(results, list):
        row_count = len(results)
        if row_count <= 50:
            return f"Returned {row_count} row(s). All rows: {results}"
        else:
            sample = results[:5]
            return f"Returned {row_count} row(s). First 5 of {row_count}: {sample}"

    return f"Returned: {str(results)[:500]}"


def _extract_tokens(query_result) -> Dict:
    """Pull token counts from a QueryResult."""
    if not query_result:
        return {"input": 0, "output": 0}
    try:
        tok = query_result.tokens
        total = tok.total()
        return {
            "input": total.get("input", 0),
            "output": total.get("output", 0),
        }
    except Exception:
        return {"input": 0, "output": 0}


# ---------------------------------------------------------------------------
# SQLAnalyzer
# ---------------------------------------------------------------------------

class SQLAnalyzer:
    """
    Multi-step SQL reasoning engine.

    Usage:
        analyzer = SQLAnalyzer(engine, vector_engine, selected_tables,
                               config, schema_text, bare_schema,
                               rules_compressed, dialect_info)
        result = analyzer.analyze("Which product line had the highest growth last quarter?")
    """

    MAX_STEPS = 5

    def __init__(
        self,
        engine,
        vector_engine,
        selected_tables: List[str],
        config,
        schema_text: str,
        bare_schema: str,
        rules_compressed: str,
        dialect_info: Dict,
    ):
        self.engine = engine
        self.vector_engine = vector_engine
        self.selected_tables = selected_tables
        self.config = config
        self.schema_text = schema_text
        self.bare_schema = bare_schema
        self.rules_compressed = rules_compressed
        self.dialect_info = dialect_info

        # Analyzer always uses Sonnet
        self._analyzer_provider = "claude_sonnet"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, question: str) -> AnalyzerResult:
        """
        Main entry point. Returns AnalyzerResult with final SQL + results + trace.

        Implements the 5-step reasoning loop:
          1. Decompose question into sub-questions (Analyzer LLM)
          2. Execute sub-queries via process_query()
          3. Inspect results & decide next action (Analyzer LLM)
          4. Synthesize final SQL
          5. Opus Review
        """
        from llm_v2 import call_llm
        from flow_router import process_query, FlowConfig, _run_opus_review

        trace: List[Dict] = []
        sub_queries: List[Dict] = []
        total_tokens: Dict[str, int] = {
            "decompose": {"input": 0, "output": 0},
            "inspect": {"input": 0, "output": 0},
            "synthesis_plan": {"input": 0, "output": 0},
            "sub_queries": {"input": 0, "output": 0},
            "synthesis_execution": {"input": 0, "output": 0},
            "opus_review": {"input": 0, "output": 0},
        }

        # ── Step 1: Decompose ──────────────────────────────────────────────
        plan = self._decompose(question, call_llm, trace, total_tokens)

        remaining_steps: List[Dict] = list(plan.get("steps", []))
        synthesis_approach: str = plan.get("synthesis_approach", "Combine sub-queries into a CTE.")
        completed: List[Dict] = []

        # ── Steps 2 + 3: Execute → Inspect loop ───────────────────────────
        iterations = 0
        abort_message: Optional[str] = None

        while remaining_steps and iterations < self.MAX_STEPS:
            iterations += 1

            # Pick next executable step (all dependencies satisfied)
            next_step = self._pick_next_step(remaining_steps, completed)
            if next_step is None:
                # No runnable step found — dependency cycle or empty
                break

            remaining_steps = [s for s in remaining_steps if s["step_id"] != next_step["step_id"]]

            # Build sub_question with context from prior steps if needed
            sub_question = self._enrich_sub_question(next_step, completed)

            trace.append({
                "stage": "execute_sub_query",
                "step_id": next_step["step_id"],
                "sub_question": sub_question,
            })

            # Execute via existing pipeline (Pass1 → Context Agent → Pass2 → SQL Coder)
            sub_config = self._make_sub_config()
            qr = process_query(
                question=sub_question,
                engine=self.engine,
                vector_engine=self.vector_engine,
                selected_tables=self.selected_tables,
                config=sub_config,
            )

            results_summary = _summarise_results(qr)
            step_tokens = _extract_tokens(qr)

            total_tokens["sub_queries"]["input"] += step_tokens["input"]
            total_tokens["sub_queries"]["output"] += step_tokens["output"]

            completed_item = {
                "step_id": next_step["step_id"],
                "description": next_step.get("description", ""),
                "sub_question": sub_question,
                "sql": qr.sql if qr else "",
                "results_summary": results_summary,
                "success": qr.success if qr else False,
                "tokens": step_tokens,
            }

            # Detect ranking queries and extract the identified entity for inspect context
            identified_entity = self._identify_entity(sub_question, qr)
            if identified_entity:
                completed_item["identified_entity"] = identified_entity

            # Store sub-query LLM trace for the Analyzer debug tab
            if qr and hasattr(qr, 'llm_trace'):
                completed_item["llm_trace"] = qr.llm_trace

            completed.append(completed_item)
            sub_queries.append(completed_item)

            trace.append({
                "stage": "sub_query_result",
                "step_id": next_step["step_id"],
                "sql": completed_item["sql"],
                "results_summary": results_summary,
                "success": completed_item["success"],
            })

            # ── Step 3: Inspect results ────────────────────────────────────
            if remaining_steps:
                decision_obj = self._inspect(
                    question, plan, completed, remaining_steps,
                    call_llm, trace, total_tokens,
                )
                decision = decision_obj.get("decision", "proceed")

                if decision == "abort":
                    abort_message = decision_obj.get("abort_message", "No data found to answer the question.")
                    trace.append({"stage": "abort", "reason": abort_message})
                    break

                if decision == "synthesize":
                    trace.append({"stage": "early_synthesize", "reason": decision_obj.get("reason", "")})
                    break

                if decision == "modify":
                    updated = decision_obj.get("updated_remaining_steps", remaining_steps)
                    remaining_steps = updated
                    trace.append({"stage": "modify_plan", "updated_steps": remaining_steps})

                # "proceed" — continue loop with remaining steps as-is

        # ── Step 4: Synthesis ──────────────────────────────────────────────
        synthesis_sql = ""
        final_results = None

        # If every completed sub-query failed, synthesizing is pointless —
        # the LLM would only see "FAILED: ..." summaries and produce garbage.
        # This covers single-step plans (where _inspect is never called because
        # remaining_steps is empty after the only step runs) and multi-step plans
        # where the last step was the final one and all happened to fail.
        if completed and all(not c.get("success") for c in completed) and not abort_message:
            first_err = completed[0].get("results_summary", "All sub-queries failed.")
            return AnalyzerResult(
                plan=plan,
                sub_queries=sub_queries,
                synthesis_sql="",
                final_results=None,
                opus_verdict="NOT_REVIEWED",
                total_tokens=total_tokens,
                trace=trace + [{"stage": "aborted_all_failed", "message": first_err}],
            )

        if abort_message:
            # Nothing to synthesize
            return AnalyzerResult(
                plan=plan,
                sub_queries=sub_queries,
                synthesis_sql="",
                final_results=None,
                opus_verdict="NOT_REVIEWED",
                total_tokens=total_tokens,
                trace=trace + [{"stage": "aborted", "message": abort_message}],
            )

        synthesis_sql, final_results = self._synthesize(
            question=question,
            completed=completed,
            synthesis_approach=synthesis_approach,
            call_llm=call_llm,
            process_query_fn=process_query,
            trace=trace,
            total_tokens=total_tokens,
        )

        # ── Step 5: Opus Review ────────────────────────────────────────────
        opus_verdict = "NOT_REVIEWED"
        _opus_raw = getattr(self.config, "enable_opus", False)
        # Explicitly normalize all valid values — do NOT rely on bool("auto") == True.
        # "auto" means run Opus for analysis queries (they are inherently complex and
        # the Analyzer is their designated reviewer; the outer Stage 7 skips them).
        if isinstance(_opus_raw, str):
            _opus_enabled = _opus_raw.lower() in ("auto", "true")
        else:
            _opus_enabled = bool(_opus_raw)
        _has_narration = isinstance(final_results, str) and bool(final_results)
        if (synthesis_sql or _has_narration) and _opus_enabled:
            try:
                # For narrate synthesis: pass narration text as the content to verify.
                # _run_opus_review accepts sql="" with results=narration_text for this case.
                opus_out = _run_opus_review(
                    question=question,
                    sql=synthesis_sql,
                    results=final_results,
                    schema_text=self.schema_text,
                    rules_compressed=self.rules_compressed,
                    config=self.config,
                    engine=self.engine,
                    use_opus_refinement=False,
                )
                opus_verdict = opus_out.get("verdict", "NOT_REVIEWED")
                opus_tok = opus_out.get("tokens", {"input": 0, "output": 0})
                total_tokens["opus_review"]["input"] += opus_tok.get("input", 0)
                total_tokens["opus_review"]["output"] += opus_tok.get("output", 0)
                trace.append({
                    "stage": "opus_review",
                    "verdict": opus_verdict,
                    "reasoning": opus_out.get("final_review", {}).get("reasoning", ""),
                    "input_prompt": opus_out.get("trace_opus_input", ""),
                    "raw_response": opus_out.get("trace_opus_output", ""),
                })
            except Exception as e:
                trace.append({"stage": "opus_review_error", "error": str(e)})

        return AnalyzerResult(
            plan=plan,
            sub_queries=sub_queries,
            synthesis_sql=synthesis_sql,
            final_results=final_results,
            opus_verdict=opus_verdict,
            total_tokens=total_tokens,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decompose(
        self,
        question: str,
        call_llm,
        trace: List[Dict],
        total_tokens: Dict,
    ) -> Dict:
        """Step 1: Call Analyzer LLM to decompose question into sub-questions."""
        prompt = _build_decompose_prompt(
            question=question,
            bare_schema=self.bare_schema,
            rules_compressed=self.rules_compressed,
            dialect_info=self.dialect_info,
        )

        trace.append({"stage": "decompose_input", "prompt": prompt, "system_prompt": _DECOMPOSE_SYSTEM})

        response, tok = call_llm(
            prompt=prompt,
            provider=self._analyzer_provider,
            system_prompt=_DECOMPOSE_SYSTEM,
        )

        total_tokens["decompose"]["input"] += tok.get("input", 0)
        total_tokens["decompose"]["output"] += tok.get("output", 0)

        trace.append({"stage": "decompose_output", "response": response})

        plan = self._parse_json_response(response, default={"steps": [], "synthesis_approach": ""})
        if not plan.get("steps"):
            # Fallback: treat the entire question as a single step
            plan = {
                "analysis_type": "multi_metric",
                "reasoning": "Could not decompose — treating as single step.",
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Answer the original question directly",
                        "sub_question": question,
                        "depends_on": [],
                        "result_usage": "Direct answer",
                    }
                ],
                "synthesis_approach": "Use the single sub-query result directly.",
            }

        return plan

    def _inspect(
        self,
        question: str,
        plan: Dict,
        completed: List[Dict],
        remaining_steps: List[Dict],
        call_llm,
        trace: List[Dict],
        total_tokens: Dict,
    ) -> Dict:
        """Step 3: Analyzer reviews completed results and decides what to do next."""
        prompt = _build_inspect_prompt(
            question=question,
            plan=plan,
            completed=completed,
            remaining_steps=remaining_steps,
        )

        trace.append({"stage": "inspect_input", "prompt": prompt})

        response, tok = call_llm(
            prompt=prompt,
            provider=self._analyzer_provider,
        )

        total_tokens["inspect"]["input"] += tok.get("input", 0)
        total_tokens["inspect"]["output"] += tok.get("output", 0)

        trace.append({"stage": "inspect_output", "response": response})

        return self._parse_json_response(response, default={"decision": "proceed", "updated_remaining_steps": remaining_steps})

    def _synthesize(
        self,
        question: str,
        completed: List[Dict],
        synthesis_approach: str,
        call_llm,
        process_query_fn,
        trace: List[Dict],
        total_tokens: Dict,
    ):
        """Step 4: Build and execute the final synthesis query."""
        if not completed:
            return "", None

        # If only one step and it succeeded, use it directly
        if len(completed) == 1 and completed[0].get("success"):
            only = completed[0]
            trace.append({"stage": "synthesis", "type": "direct_single_step"})
            # Re-execute via pipeline to get live results as objects
            sub_config = self._make_sub_config()
            qr = process_query_fn(
                question=only["sub_question"],
                engine=self.engine,
                vector_engine=self.vector_engine,
                selected_tables=self.selected_tables,
                config=sub_config,
            )
            tok = _extract_tokens(qr)
            total_tokens["synthesis_execution"]["input"] += tok["input"]
            total_tokens["synthesis_execution"]["output"] += tok["output"]
            return qr.sql if qr else "", qr.results if qr else None

        # Ask Analyzer to produce synthesis output (narrate or cte SQL)
        prompt = _build_synthesis_prompt(
            question=question,
            completed=completed,
            synthesis_approach=synthesis_approach,
            dialect_info=self.dialect_info,
        )

        trace.append({"stage": "synthesis_plan_input", "prompt": prompt})

        raw_synthesis, tok = call_llm(
            prompt=prompt,
            provider=self._analyzer_provider,
        )

        total_tokens["synthesis_plan"]["input"] += tok.get("input", 0)
        total_tokens["synthesis_plan"]["output"] += tok.get("output", 0)

        trace.append({"stage": "synthesis_plan_output", "response": raw_synthesis})

        synthesis_obj = self._parse_json_response(raw_synthesis, default=None)
        synthesis_type = (synthesis_obj or {}).get("synthesis_type", "cte")

        # ── Narrate path: sub-query results already answer the question ──────
        if synthesis_type == "narrate":
            narration = (synthesis_obj or {}).get("result", "")
            trace.append({"stage": "synthesis_narrate", "narration": narration})
            return "", narration

        # ── CTE path: synthesis LLM wrote a complete SQL query ────────────────
        from sql_static_validator import validate_sql_static, build_tier1_mini_retry_prompt
        from db import run_sql

        dialect = self.dialect_info.get("dialect", "postgresql") if self.dialect_info else "postgresql"
        quote_char = self.dialect_info.get("quote_char", '"') if self.dialect_info else '"'

        synthesis_sql = self._extract_sql_from_response(raw_synthesis, synthesis_obj)

        if not synthesis_sql:
            # Fallback: nothing usable from synthesis LLM — return empty
            trace.append({"stage": "synthesis_cte_no_sql"})
            return "", None

        trace.append({"stage": "synthesis_cte_sql", "sql": synthesis_sql})

        # Tier 1 static validation + mini-retry
        tier1_issues = validate_sql_static(sql=synthesis_sql, column_metadata={}, dialect=dialect)
        if tier1_issues:
            retry_prompt = build_tier1_mini_retry_prompt(synthesis_sql, tier1_issues, dialect, quote_char)
            fixed_text, fix_tok = call_llm(prompt=retry_prompt, provider=self._analyzer_provider)
            total_tokens["synthesis_plan"]["input"] += fix_tok.get("input", 0)
            total_tokens["synthesis_plan"]["output"] += fix_tok.get("output", 0)
            fixed_candidate = self._extract_sql_from_response(fixed_text, None)
            if fixed_candidate:
                synthesis_sql = fixed_candidate
            trace.append({"stage": "synthesis_cte_tier1_fix", "sql": synthesis_sql})

        # Execute directly — no medium pipeline re-routing
        def _try_execute(sql: str):
            return run_sql(self.engine, sql)

        try:
            df = _try_execute(synthesis_sql)
            trace.append({"stage": "synthesis_result", "sql": synthesis_sql, "success": True})
            return synthesis_sql, df
        except Exception as exec_err:
            trace.append({
                "stage": "synthesis_cte_exec_error",
                "error": str(exec_err),
                "sql": synthesis_sql,
            })

            # ONE LLM retry on execution failure
            retry_fix_prompt = (
                f"The following {dialect.upper()} SQL failed with error:\n{exec_err}\n\n"
                f"SQL:\n{synthesis_sql}\n\n"
                "Fix the SQL to resolve the error. Return ONLY the corrected SQL. No explanation."
            )
            fixed_text, retry_tok = call_llm(prompt=retry_fix_prompt, provider=self._analyzer_provider)
            total_tokens["synthesis_plan"]["input"] += retry_tok.get("input", 0)
            total_tokens["synthesis_plan"]["output"] += retry_tok.get("output", 0)

            fixed_sql = self._extract_sql_from_response(fixed_text, None) or synthesis_sql

            # Re-validate before retry execute
            tier1_retry = validate_sql_static(sql=fixed_sql, column_metadata={}, dialect=dialect)
            if tier1_retry:
                mini = build_tier1_mini_retry_prompt(fixed_sql, tier1_retry, dialect, quote_char)
                fixed_text2, mini_tok = call_llm(prompt=mini, provider=self._analyzer_provider)
                total_tokens["synthesis_plan"]["input"] += mini_tok.get("input", 0)
                total_tokens["synthesis_plan"]["output"] += mini_tok.get("output", 0)
                candidate2 = self._extract_sql_from_response(fixed_text2, None)
                if candidate2:
                    fixed_sql = candidate2

            try:
                df = _try_execute(fixed_sql)
                synthesis_sql = fixed_sql
                trace.append({"stage": "synthesis_cte_retry_success", "sql": synthesis_sql})
                return synthesis_sql, df
            except Exception as retry_err:
                trace.append({
                    "stage": "synthesis_cte_retry_failed",
                    "error": str(retry_err),
                    "sql": fixed_sql,
                })
                return synthesis_sql, None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _make_sub_config(self):
        """
        Return a FlowConfig for sub-queries.

        Sub-queries must NEVER re-enter the Analyzer — that would cause infinite
        recursion.  Enforce this with two independent guards:
          - enable_analyzer = False  : hard block even if complexity == "analysis"
          - enable_classification = False : skip the LLM classifier entirely;
                process_query defaults to "medium" → analytical flow
                (Pass 1 → Context Agent → Pass 2 → SQL Coder)

        Also disables query cache (each sub-query is bespoke) and Opus review
        (a single Opus review runs on the final synthesis SQL instead).
        """
        from flow_router import FlowConfig
        import copy

        cfg = copy.copy(self.config) if self.config else FlowConfig()
        cfg.enable_cache = False
        cfg.enable_opus = False
        cfg.enable_analyzer = False        # prevent Analyzer re-entry
        cfg.enable_classification = False  # skip re-classification; use medium flow
        cfg.initial_classification = None  # clear any pre-computed result from the UI
        # enable_opus_descriptions is intentionally inherited from the parent config
        # (via copy.copy above) so sub-queries benefit from column descriptions if
        # the parent config had them enabled.
        return cfg

    def _pick_next_step(
        self,
        remaining: List[Dict],
        completed: List[Dict],
    ) -> Optional[Dict]:
        """Return the first remaining step whose dependencies are all satisfied."""
        completed_ids = {s["step_id"] for s in completed}
        for step in remaining:
            deps = set(step.get("depends_on", []))
            if deps.issubset(completed_ids):
                return step
        return None

    def _enrich_sub_question(self, step: Dict, completed: List[Dict]) -> str:
        """
        Inject summaries of dependency results into the sub_question so the
        pipeline has the context it needs (e.g. specific values to filter on).
        """
        deps = step.get("depends_on", [])
        if not deps:
            return step["sub_question"]

        dep_summaries = []
        for dep_id in deps:
            for c in completed:
                if c["step_id"] == dep_id:
                    dep_summaries.append(
                        f"[Context from step {dep_id}: {c['results_summary']}]"
                    )
                    break

        if not dep_summaries:
            return step["sub_question"]

        context_block = " ".join(dep_summaries)
        return f"{step['sub_question']} (Additional context: {context_block})"

    @staticmethod
    def _extract_sql_from_response(raw_response: str, parsed_obj: Optional[Dict]) -> str:
        """
        Extract complete SQL from a synthesis LLM response.

        Clean path: parsed_obj["result"] if available.
        Dirty path (broken JSON from unescaped double-quotes in SQL):
          Scan raw_response for the first WITH or SELECT keyword at word boundary
          and extract from that point to end, stripping trailing JSON artifacts.
        Also strips markdown fences in both paths.
        """
        import re as _re

        def _strip_fences(text: str) -> str:
            text = text.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                # Drop opening fence line
                inner = lines[1:] if lines[0].startswith("```") else lines
                # Drop closing fence line
                if inner and inner[-1].strip() == "```":
                    inner = inner[:-1]
                text = "\n".join(inner).strip()
            return text

        # Clean path
        if parsed_obj and isinstance(parsed_obj.get("result"), str):
            candidate = parsed_obj["result"].strip()
            if candidate:
                return _strip_fences(candidate)

        # Dirty path — scan raw response for SQL start
        m = _re.search(r'\b(WITH|SELECT)\b', raw_response, _re.IGNORECASE)
        if m:
            sql_candidate = raw_response[m.start():]
            # Strip trailing JSON artifacts: "}", "})", '"', newlines
            sql_candidate = _re.sub(r'[\s"}\)]+$', '', sql_candidate).strip()
            if sql_candidate:
                return _strip_fences(sql_candidate)

        return ""

    @staticmethod
    def _identify_entity(sub_question: str, query_result) -> Optional[str]:
        """
        If a sub-question is a ranking/top/bottom query and results are non-empty,
        extract the identified entity from the first result row.

        Returns a string like "KAM Name = Arjun Vt" or None if not applicable.
        """
        if not query_result or not query_result.success:
            return None

        import re as _re
        ranking_pattern = _re.compile(
            r'\b(highest|top|best|lowest|worst|ordered by.*descending|returning the top)\b',
            _re.IGNORECASE,
        )
        if not ranking_pattern.search(sub_question):
            return None

        results = query_result.results
        if results is None:
            return None

        # Get first row as a dict
        first_row: Optional[Dict] = None
        try:
            import pandas as pd
            if isinstance(results, pd.DataFrame) and not results.empty:
                first_row = results.iloc[0].to_dict()
        except ImportError:
            pass

        if first_row is None and isinstance(results, list) and results:
            row = results[0]
            if isinstance(row, dict):
                first_row = row
            elif hasattr(row, '_asdict'):
                first_row = row._asdict()

        if not first_row:
            return None

        # Find the first column whose value is a non-numeric, non-None string
        for col, val in first_row.items():
            if val is None:
                continue
            if isinstance(val, str) and val.strip():
                return f"{col} = {val}"
            # Skip numeric types
            if isinstance(val, (int, float)):
                continue

        return None

    @staticmethod
    def _parse_json_response(response: str, default: Any = None) -> Any:
        """Parse JSON from LLM response, stripping markdown fences if present."""
        text = response.strip()
        # Strip ```json ... ``` or ``` ... ```
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first and last lines (fences)
            inner = lines[1:] if lines[0].startswith("```") else lines
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            text = "\n".join(inner).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the first JSON object/array
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            return default if default is not None else {}
