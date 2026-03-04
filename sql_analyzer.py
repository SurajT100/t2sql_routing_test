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
1. Each sub_question must be answerable with a single, non-nested SQL query.
2. Do NOT combine multiple aggregations or multi-hop logic into one sub-question.
3. Reference specific column names from the schema provided.
4. Apply the business rules provided when deciding what data to retrieve.
5. Do not hardcode table names, column names, or specific values in your plan — describe them generically using the schema.
6. Specify exactly what columns/aggregates each step returns so downstream steps can use them.
7. Limit to at most 5 steps.

Return a JSON object ONLY (no markdown, no explanation outside the JSON)."""


def _build_decompose_prompt(
    question: str,
    bare_schema: str,
    rules_compressed: str,
    dialect_info: Dict,
) -> str:
    dialect = dialect_info.get("dialect", "postgresql") if dialect_info else "postgresql"
    rules_text = rules_compressed if rules_compressed and rules_compressed != "[]" else "None"
    return f"""Question: {question}

Database dialect: {dialect}

Schema (column names and types):
{bare_schema}

Business rules:
{rules_text}

Break this question into simple sub-questions. Each sub-question must be answerable with a single SQL query.

Return this exact JSON structure:
{{
  "analysis_type": "what_if_simulation | trend_analysis | comparative_ranking | impact_analysis | multi_metric",
  "reasoning": "Why this needs multi-step analysis",
  "steps": [
    {{
      "step_id": 1,
      "description": "What this step computes",
      "sub_question": "A simple question that a SQL generator can handle in one query",
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

Use "synthesize" if you have enough data to answer the original question.
Use "abort" if the data shows the question cannot be answered (e.g. Step returned 0 rows).
Use "modify" if remaining steps need to change based on what you learned.
Use "proceed" to continue with remaining steps as-is."""


def _build_synthesis_prompt(
    question: str,
    completed: List[Dict],
    synthesis_approach: str,
    dialect_info: Dict,
) -> str:
    dialect = dialect_info.get("dialect", "postgresql") if dialect_info else "postgresql"
    steps_text = ""
    for item in completed:
        steps_text += f"\nStep {item['step_id']}: {item['sub_question']}\n"
        steps_text += f"  SQL used: {item.get('sql', 'N/A')}\n"
        steps_text += f"  Results summary: {item.get('results_summary', 'N/A')}\n"

    return f"""Original question: {question}

Database dialect: {dialect}

Synthesis approach: {synthesis_approach}

Sub-queries already validated and executed:
{steps_text}

Write a final SQL query that answers the original question by combining the above sub-queries.
Options:
  A) A single query incorporating knowledge from the results (e.g. using IN lists, hard-coded values from results, or derived logic).
  B) A CTE-based query where each CTE corresponds to one of the sub-queries above.

Choose whichever approach produces a correct, readable query.

Return ONLY a JSON object:
{{
  "synthesis_type": "single_query | cte",
  "sub_question": "A clear question that describes what the final SQL should answer",
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
            sample_rows = results.head(5).to_dict(orient="records")
            return (
                f"Returned {row_count} row(s). Columns: {cols}. "
                f"First {min(5, row_count)} rows: {sample_rows}"
            )
    except ImportError:
        pass

    if isinstance(results, list):
        row_count = len(results)
        sample = results[:5]
        return f"Returned {row_count} row(s). Sample: {sample}"

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
        if synthesis_sql and final_results is not None:
            try:
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

        trace.append({"stage": "decompose_input", "prompt": prompt})

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

        # Ask Analyzer to produce a synthesis sub-question
        prompt = _build_synthesis_prompt(
            question=question,
            completed=completed,
            synthesis_approach=synthesis_approach,
            dialect_info=self.dialect_info,
        )

        trace.append({"stage": "synthesis_plan_input", "prompt": prompt})

        response, tok = call_llm(
            prompt=prompt,
            provider=self._analyzer_provider,
        )

        total_tokens["synthesis_plan"]["input"] += tok.get("input", 0)
        total_tokens["synthesis_plan"]["output"] += tok.get("output", 0)

        trace.append({"stage": "synthesis_plan_output", "response": response})

        synthesis_obj = self._parse_json_response(
            response,
            default={"synthesis_type": "single_query", "sub_question": question},
        )

        final_question = synthesis_obj.get("sub_question", question)
        trace.append({"stage": "synthesis_execute", "final_question": final_question})

        # Execute the synthesis question through the pipeline
        sub_config = self._make_sub_config()
        qr = process_query_fn(
            question=final_question,
            engine=self.engine,
            vector_engine=self.vector_engine,
            selected_tables=self.selected_tables,
            config=sub_config,
        )

        tok = _extract_tokens(qr)
        total_tokens["synthesis_execution"]["input"] += tok["input"]
        total_tokens["synthesis_execution"]["output"] += tok["output"]

        trace.append({
            "stage": "synthesis_result",
            "sql": qr.sql if qr else "",
            "success": qr.success if qr else False,
        })

        return (qr.sql if qr else ""), (qr.results if qr else None)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _make_sub_config(self):
        """
        Return a FlowConfig for sub-queries.
        Disables query cache (each sub-query is bespoke) and Opus review
        (we do a single Opus review on the final synthesis SQL instead).
        """
        from flow_router import FlowConfig
        import copy

        cfg = copy.copy(self.config) if self.config else FlowConfig()
        cfg.enable_cache = False
        cfg.enable_opus = False
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
