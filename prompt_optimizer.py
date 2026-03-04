"""
Prompt Optimizer
=================
Optimizes prompts for token efficiency while maintaining accuracy.
Includes compressed rule formats and optimized prompt templates.

Usage:
    from prompt_optimizer import (
        compress_rules_for_llm,
        create_easy_query_prompt,
        create_medium_query_prompt,
        create_opus_review_prompt_optimized
    )
"""

import json
from typing import List, Dict, Any, Optional
from decimal import Decimal


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Decimal types from PostgreSQL."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def safe_json_dumps(obj, **kwargs) -> str:
    """JSON dumps that handles Decimal and other PostgreSQL types."""
    return json.dumps(obj, cls=DecimalEncoder, **kwargs)


# =============================================================================
# RULE COMPRESSION
# =============================================================================

def compress_rules_for_llm(rules: List[Dict]) -> str:
    """
    Quality-first rule compression.

    Goal: reduce token size while preserving business logic fidelity.
    We keep critical semantics (formula/sql_pattern/conditions/filters/mappings)
    and only trim verbose narrative text.
    """
    compressed = []

    for rule in rules:
        rule_type = rule.get("rule_type", "other")
        rule_name = rule.get("rule_name", "Unknown")
        rule_desc = rule.get("rule_description", "")
        rule_data = rule.get("rule_data", {})

        if isinstance(rule_data, str):
            try:
                rule_data = json.loads(rule_data)
            except Exception:
                rule_data = {}

        compact_rule = {
            "type": rule_type,
            "name": rule_name,
        }

        # Preserve lightweight metadata that helps ranking/applicability
        if rule.get("priority") is not None:
            compact_rule["priority"] = rule.get("priority")
        if rule.get("is_mandatory") is not None:
            compact_rule["mandatory"] = bool(rule.get("is_mandatory"))
        if rule.get("keywords"):
            compact_rule["keywords"] = rule.get("keywords")[:12]
        if rule.get("tables"):
            compact_rule["tables"] = rule.get("tables")[:8]

        if rule_type == "metric":
            # Keep core metric semantics intact
            for key, out_key in [
                ("formula", "formula"),
                ("aggregation", "aggregation"),
                ("table", "table"),
                ("column", "column"),
                ("transform", "transform"),
                ("display_unit", "unit"),
                ("condition", "filter"),
            ]:
                val = rule_data.get(key)
                if val not in (None, "", [], {}):
                    compact_rule[out_key] = val

            # Preserve mandatory filters/business-term aliases (high-value for accuracy)
            if rule_data.get("mandatory_filters"):
                compact_rule["mandatory_filters"] = rule_data["mandatory_filters"]
            if rule_data.get("user_terms"):
                compact_rule["user_terms"] = rule_data["user_terms"][:12]
            if rule_data.get("description"):
                compact_rule["desc"] = str(rule_data["description"])[:220]
            elif rule_desc:
                compact_rule["desc"] = str(rule_desc)[:220]

        elif rule_type == "join":
            # Join condition must remain exact
            compact_rule["on"] = rule_data.get("join_condition", "")
            if rule_data.get("tables"):
                compact_rule["tables"] = rule_data.get("tables", [])[:8]
            if rule_data.get("join_type"):
                compact_rule["join_type"] = rule_data.get("join_type")
            if rule_data.get("description"):
                compact_rule["desc"] = str(rule_data["description"])[:180]
            elif rule_desc:
                compact_rule["desc"] = str(rule_desc)[:180]

        elif rule_type == "filter":
            # Preserve full filter logic; this is accuracy-critical
            if rule_data.get("sql_pattern"):
                compact_rule["apply"] = rule_data.get("sql_pattern")
            for key in ["table", "column", "operator", "values", "filter_name", "user_terms"]:
                val = rule_data.get(key)
                if val not in (None, "", [], {}):
                    compact_rule[key] = val
            if rule_data.get("description"):
                compact_rule["desc"] = str(rule_data["description"])[:180]
            elif rule_desc:
                compact_rule["desc"] = str(rule_desc)[:180]

        elif rule_type in ("mapping", "value_mapping", "term_alias"):
            # Mapping structure is usually compact and highly informative
            mappings = rule_data.get("mappings", rule_data.get("values", {}))
            if mappings:
                compact_rule["maps"] = mappings
            for key in ["source", "target", "table", "column", "description_type", "description", "user_terms"]:
                val = rule_data.get(key)
                if val not in (None, "", [], {}):
                    compact_rule[key] = val
            if rule_desc and "description" not in compact_rule:
                compact_rule["desc"] = str(rule_desc)[:200]

        elif rule_type in ("dialect", "default"):
            if rule_type == "dialect":
                # Keep dialect directives exact and complete
                compact_rule = {
                    "type": "dialect",
                    "name": rule_name,
                    "db": rule_data.get("dialect", ""),
                    "quote": rule_data.get("quote_char", '"'),
                    "str_quote": rule_data.get("string_quote", "'"),
                    "schema_hint": rule_data.get("schema_qualification", ""),
                    "date_functions": rule_data.get("date_functions", {}),
                    "limit_syntax": rule_data.get("limit_syntax", ""),
                }
            else:
                # Default/critical business rules should retain full logic
                for key, out_key in [
                    ("sql_pattern", "apply"),
                    ("condition", "condition"),
                    ("auto_apply", "auto_apply"),
                    ("rule_type", "subtype"),
                    ("applies_to_queries", "applies_to_queries"),
                ]:
                    val = rule_data.get(key)
                    if val not in (None, "", [], {}):
                        compact_rule[out_key] = val
                if rule_data.get("description"):
                    compact_rule["desc"] = str(rule_data["description"])[:260]
                elif rule_desc:
                    compact_rule["desc"] = str(rule_desc)[:260]

        elif rule_type in ("example", "query_example"):
            # Keep concise examples (still useful but token-capped)
            compact_rule.update({
                "q": rule_data.get("question", rule_name)[:160],
                "sql": rule_data.get("sql", "")[:500],
            })

        elif rule_type in ("column", "table_column"):
            compact_rule.update({
                "col": rule_data.get("column_name", rule_name),
                "table": rule_data.get("table_name", rule_data.get("table", "")),
                "desc": rule_data.get("description", rule_desc)[:220],
            })
            if rule_data.get("business_terms"):
                compact_rule["terms"] = rule_data["business_terms"][:12]

        else:
            # Generic safe fallback
            if rule_data:
                compact_rule["data"] = rule_data
            if rule_desc:
                compact_rule["desc"] = str(rule_desc)[:200]

        compact_rule = {k: v for k, v in compact_rule.items() if v not in (None, "", [], {})}
        if len(compact_rule) > 1:
            compressed.append(compact_rule)

    return safe_json_dumps(compressed, separators=(',', ':'))


def decompress_rules_for_display(compressed_json: str) -> List[Dict]:
    """
    Convert compressed rules back to readable format for UI display.
    """
    try:
        rules = json.loads(compressed_json)
        return rules
    except:
        return []


# =============================================================================
# PROMPT TEMPLATES - EASY QUERIES
# =============================================================================

def create_easy_query_prompt(
    question: str,
    schema: str,
    dialect_info: Dict,
    rules: str = None
) -> str:
    """
    Minimal prompt for EASY queries - direct to SQL LLM.
    No reasoning phase needed.
    """
    quote_char = dialect_info.get("quote_char", '"')
    string_quote = dialect_info.get("string_quote", "'")
    dialect_name = dialect_info.get("dialect", "SQL").upper()
    
    if dialect_name == "POSTGRESQL":
        table_example = '"public"."table_name"'
        col_example = '"column_name"'
        full_example = 'SELECT "Amount" FROM "public"."orders" WHERE "Status" = \'Active\''
        wrong_example = 'FROM public.orders or FROM "public.orders"'
    elif dialect_name == "MYSQL":
        table_example = '`database`.`table_name`'
        col_example = '`column_name`'
        full_example = 'SELECT `Amount` FROM `mydb`.`orders` WHERE `Status` = \'Active\''
        wrong_example = 'FROM mydb.orders or FROM `mydb.orders`'
    elif dialect_name == "MSSQL":
        table_example = '[schema].[table_name]'
        col_example = '[column_name]'
        full_example = 'SELECT [Amount] FROM [dbo].[orders] WHERE [Status] = \'Active\''
        wrong_example = 'FROM dbo.orders or FROM [dbo.orders]'
    else:
        table_example = '"schema"."table_name"'
        col_example = '"column_name"'
        full_example = 'SELECT "Amount" FROM "public"."orders" WHERE "Status" = \'Active\''
        wrong_example = 'FROM public.orders'
    
    prompt = f"""Generate a {dialect_name} SQL query for this question.

SYNTAX RULES:
- Tables: {table_example} (quote schema and table SEPARATELY)
- Columns: {col_example}
- Strings: {string_quote}value{string_quote}
- Example: {full_example}
- WRONG: {wrong_example}

SCHEMA:
{schema}
"""
    
    if rules:
        prompt += f"""
BUSINESS RULES:
{rules}
"""
    
    prompt += f"""
QUESTION: {question}

Return ONLY the SQL query. No explanation, no markdown, just SQL."""
    
    return prompt


# =============================================================================
# PROMPT TEMPLATES - MEDIUM QUERIES
# =============================================================================

def create_medium_query_prompt(
    question: str,
    schema: str,
    rules: str,
    examples: List[Dict] = None,
    dialect_info: Dict = None,
    output_type: str = "full"
) -> str:
    """
    Standard prompt for MEDIUM queries with reasoning.
    """
    dialect_name = dialect_info.get("dialect", "SQL").upper() if dialect_info else "SQL"
    quote_char = dialect_info.get("quote_char", '"') if dialect_info else '"'
    string_quote = dialect_info.get("string_quote", "'") if dialect_info else "'"
    
    if dialect_name == "POSTGRESQL":
        example_table = '"schema_name"."table_name"'
        example_col = '"column_name"'
        example_filter = '"status" = \'active\''
        table_format = '"schema"."table"'
        wrong_format = '"schema.table"'
    elif dialect_name == "MYSQL":
        example_table = '`db_name`.`table_name`'
        example_col = '`column_name`'
        example_filter = '`status` = \'active\''
        table_format = '`database`.`table`'
        wrong_format = '`database.table`'
    elif dialect_name == "MSSQL":
        example_table = '[schema_name].[table_name]'
        example_col = '[column_name]'
        example_filter = '[status] = \'active\''
        table_format = '[schema].[table]'
        wrong_format = '[schema.table]'
    elif dialect_name == "ORACLE":
        example_table = '"SCHEMA_NAME"."TABLE_NAME"'
        example_col = '"COLUMN_NAME"'
        example_filter = '"STATUS" = \'ACTIVE\''
        table_format = '"SCHEMA"."TABLE"'
        wrong_format = '"SCHEMA.TABLE"'
    else:
        example_table = '"schema_name"."table_name"'
        example_col = '"column_name"'
        example_filter = '"status" = \'active\''
        table_format = '"schema"."table"'
        wrong_format = '"schema.table"'
    
    if output_type == "analysis":
        prompt = f"""You are a SQL query planner for {dialect_name} database.
Analyze this question and plan the query structure.

DATABASE: {dialect_name}
TABLE FORMAT: {table_format} (quote schema and table SEPARATELY)
COLUMN FORMAT: {quote_char}column_name{quote_char}
STRING VALUES: {string_quote}value{string_quote}

SCHEMA:
{schema}

BUSINESS RULES:
{rules}

QUESTION: {question}

TASK: Analyze what's needed to answer this question.
1. Which tables are needed?
2. Which columns are needed?
3. What filters should be applied (write as valid {dialect_name} WHERE conditions)?
4. What aggregations are needed?
5. Which business rules apply?

CRITICAL QUOTING RULES FOR {dialect_name}:
- CORRECT table format: {table_format} → e.g., {example_table}
- WRONG table format: {wrong_format} (do NOT quote schema.table together)
- Column format: {example_col}
- Filter example: {example_filter}

OUTPUT (JSON only):
{{
  "tables": ["{example_table}"],
  "columns": ["{example_col}"],
  "filters": ["{example_filter}"],
  "aggregations": ["SUM({example_col})"],
  "group_by": [],
  "business_rules_applied": ["rule_name"]
}}"""
    else:
        prompt = f"""Analyze this question and generate SQL.

DATABASE: {dialect_name}
QUOTE IDENTIFIERS WITH: {quote_char}

SCHEMA:
{schema}

BUSINESS RULES:
{rules}
"""
        if examples:
            prompt += "\nEXAMPLES TO FOLLOW:\n"
            for ex in examples[:3]:
                q = ex.get("q", ex.get("question", ""))
                sql = ex.get("sql", "")
                if q and sql:
                    prompt += f"Q: {q}\nSQL: {sql}\n\n"
        
        prompt += f"""
QUESTION: {question}

TASK:
1. Identify which columns and tables are needed
2. Apply any relevant business rules
3. Generate the correct SQL query

OUTPUT FORMAT (JSON):
{{
  "analysis": "Brief explanation of approach",
  "tables": ["table1", "table2"],
  "columns": ["col1", "col2"],
  "filters": ["condition1", "condition2"],
  "sql": "SELECT ... FROM ... WHERE ..."
}}"""
    
    return prompt


# =============================================================================
# PROMPT TEMPLATES - HARD QUERIES
# =============================================================================

def create_hard_query_prompt(
    question: str,
    schema: str,
    rules: str,
    examples: List[Dict] = None,
    dialect_info: Dict = None,
    output_type: str = "full"
) -> str:
    """
    Comprehensive prompt for HARD queries requiring detailed reasoning.
    """
    dialect_name = dialect_info.get("dialect", "SQL").upper() if dialect_info else "SQL"
    quote_char = dialect_info.get("quote_char", '"') if dialect_info else '"'
    string_quote = dialect_info.get("string_quote", "'") if dialect_info else "'"
    
    if dialect_name == "POSTGRESQL":
        example_table = '"schema_name"."table_name"'
        example_col = '"column_name"'
        example_filter = '"is_active" = true'
        wrong_format = '"schema_name.table_name"'
        table_format = '"schema"."table"'
    elif dialect_name == "MYSQL":
        example_table = '`db_name`.`table_name`'
        example_col = '`column_name`'
        example_filter = '`is_active` = 1'
        wrong_format = '`db_name.table_name`'
        table_format = '`database`.`table`'
    elif dialect_name == "MSSQL":
        example_table = '[schema_name].[table_name]'
        example_col = '[column_name]'
        example_filter = '[is_active] = 1'
        wrong_format = '[schema_name.table_name]'
        table_format = '[schema].[table]'
    elif dialect_name == "ORACLE":
        example_table = '"SCHEMA_NAME"."TABLE_NAME"'
        example_col = '"COLUMN_NAME"'
        example_filter = '"IS_ACTIVE" = 1'
        wrong_format = '"SCHEMA_NAME.TABLE_NAME"'
        table_format = '"SCHEMA"."TABLE"'
    else:
        example_table = '"schema_name"."table_name"'
        example_col = '"column_name"'
        example_filter = '"is_active" = true'
        wrong_format = '"schema_name.table_name"'
        table_format = '"schema"."table"'
    
    if output_type == "analysis":
        prompt = f"""You are an expert SQL query planner for {dialect_name}. This is a COMPLEX query requiring careful analysis.

DATABASE: {dialect_name}
TABLE FORMAT: {table_format} (quote schema and table SEPARATELY)
COLUMN FORMAT: {example_col}
STRING VALUES: {string_quote}value{string_quote}

AVAILABLE SCHEMA:
{schema}

BUSINESS RULES (MUST APPLY):
{rules}

USER QUESTION: {question}

ANALYSIS STEPS:
1. UNDERSTAND: What exactly is being asked? Break down into components.
2. IDENTIFY: Which tables/columns are needed? Are there any JOINs required?
3. RULES: Which business rules apply? (metrics, filters, mappings)
4. CALCULATE: Any aggregations, calculations, or transformations?

CRITICAL QUOTING FOR {dialect_name}:
- CORRECT: {example_table}
- WRONG: {wrong_format} (never quote schema.table together)
- Columns: {example_col}
- Filters: {example_filter}

OUTPUT (JSON only, no SQL):
{{
  "understanding": "What the user wants",
  "tables_needed": ["{example_table}"],
  "columns_needed": {{"table_name": [{example_col}]}},
  "joins": ["{example_table}.{example_col} = {quote_char}other_schema{quote_char}.{quote_char}other_table{quote_char}.{quote_char}col{quote_char}"],
  "rules_applied": ["rule1", "rule2"],
  "aggregations": ["SUM({example_col})"],
  "filters": ["{example_filter}"],
  "group_by": [],
  "order_by": []
}}"""
    else:
        prompt = f"""You are an expert SQL analyst for {dialect_name}. This is a COMPLEX query requiring careful analysis.

DATABASE: {dialect_name}
IDENTIFIER QUOTING: Use {quote_char} for column/table names
STRING QUOTING: Use {string_quote} for string values

AVAILABLE SCHEMA:
{schema}

BUSINESS RULES (MUST APPLY):
{rules}
"""
        if examples:
            prompt += "\nSIMILAR SOLVED EXAMPLES:\n"
            for i, ex in enumerate(examples[:3], 1):
                q = ex.get("q", ex.get("question", ""))
                sql = ex.get("sql", "")
                if q and sql:
                    prompt += f"\nExample {i}:\nQ: {q}\nSQL:\n{sql}\n"
        
        prompt += f"""
USER QUESTION: {question}

ANALYSIS STEPS:
1. UNDERSTAND: What exactly is being asked? Break down into components.
2. IDENTIFY: Which tables/columns are needed? Are there any JOINs required?
3. RULES: Which business rules apply? (metrics, filters, mappings)
4. CALCULATE: Any aggregations, calculations, or transformations?
5. VALIDATE: Does the query logic match the question?

OUTPUT (JSON):
{{
  "understanding": "What the user wants in plain terms",
  "tables_needed": ["table1", "table2"],
  "columns_needed": {{"table1": ["col1", "col2"], "table2": ["col3"]}},
  "joins": ["table1.col = table2.col"],
  "rules_applied": ["rule1", "rule2"],
  "aggregations": ["SUM(col)", "COUNT(*)"],
  "filters": ["condition1", "condition2"],
  "group_by": ["col1", "col2"],
  "order_by": ["col1 DESC"],
  "sql": "The complete SQL query"
}}"""
    
    return prompt


# =============================================================================
# PROMPT TEMPLATES - SQL GENERATION (From Reasoning Output)
# =============================================================================

def create_sql_from_reasoning_prompt(
    reasoning_output: str,
    schema: str,
    dialect_info: Dict
) -> str:
    """
    Generate SQL from reasoning output.
    """
    dialect_name = dialect_info.get("dialect", "SQL").upper() if dialect_info else "SQL"
    quote_char = dialect_info.get("quote_char", '"') if dialect_info else '"'
    string_quote = dialect_info.get("string_quote", "'") if dialect_info else "'"
    
    return f"""Convert this analysis to SQL.

DATABASE: {dialect_name}
- Identifiers: {quote_char}column_name{quote_char}
- Strings: {string_quote}value{string_quote}

SCHEMA:
{schema}

ANALYSIS:
{reasoning_output}

OUTPUT: Return ONLY the SQL query. No markdown, no explanation."""


# =============================================================================
# PROMPT TEMPLATES - OPUS REVIEW (Strict Verification)
# =============================================================================

def create_opus_review_prompt_optimized(
    question: str,
    sql: str,
    results_preview: str,
    columns_used: List[str] = None,
    rules_applied: List[str] = None,
    error: str = None,
    schema_text: str = "",
    rules_compressed: str = "[]"
) -> str:
    """
    Strict Opus review prompt.
    
    Opus verifies that business rules were applied CORRECTLY,
    not just that a matching rule exists. It checks actual
    column names, filter values, and aggregation logic against
    both the schema and the rules.
    
    Args:
        question: Original user question
        sql: Generated SQL query
        results_preview: Preview of query results
        columns_used: List of columns referenced in SQL
        rules_applied: List of business rules that should have been applied
        error: Error message if SQL failed
        schema_text: Full schema with column types and descriptions
        rules_compressed: Compressed business rules
    
    Returns:
        Strict review prompt for Opus
    """
    if error:
        return f"""Review this failed SQL query.

QUESTION: {question}

SQL (FAILED):
```sql
{sql}
```

ERROR: {error}

SCHEMA:
{schema_text}

BUSINESS RULES:
{rules_compressed}

TASK: Identify the error and suggest a fix.

OUTPUT (JSON):
{{"verdict": "INCORRECT", "error_type": "syntax|logic|schema|other", "issue": "specific problem", "fix": "suggested correction"}}"""
    
    # SQL succeeded — strict correctness verification
    prompt = f"""You are an independent SQL auditor. Your job is to verify this SQL is ACTUALLY CORRECT,
not just plausible. Be critical. Wrong column = wrong answer even if the query runs.

## SCHEMA (ground truth — check every column name and data type)
{schema_text}

## BUSINESS RULES (verify each rule was applied correctly, not just referenced)
{rules_compressed}

## USER QUESTION
{question}

## SQL TO AUDIT
```sql
{sql}
```
"""
    
    if columns_used:
        prompt += f"\nCOLUMNS USED IN SQL: {', '.join(columns_used)}"
    
    prompt += f"""

## RESULTS PREVIEW
{results_preview}

## INTENT VERIFICATION — Answer these THREE questions first:

**Question 1 — WHAT is the user actually asking for?**
Restate the user's question as a precise analytical requirement.
What computation, comparison, or insight does the user expect?
What would a correct result set look like — what columns, what relationships between rows,
what would make the user say 'yes this answers my question'?

**Question 2 — DOES this SQL produce that?**
Trace the SQL logic step by step. What does each CTE/subquery compute?
What does the final SELECT actually return?
Describe the result set this SQL would produce in plain English.
Would a business user looking at these results get the answer, or would they say
'this isn't what I asked for'?

**Question 3 — Is there a GAP between Question 1 and Question 2?**
- If the SQL produces exactly what the user asked for → intent_match = PASS
- If the SQL produces something related but simpler or different → intent_match = FAIL.
  Note: 'User asked for [X]. SQL produces [Y]. Missing: [specific gap].'
- If the SQL produces the right thing but with a logical flaw in how it computes it → intent_match = FAIL.
  Note the specific flaw.

---

## AUDIT CHECKLIST — verify each point against schema and rules above:

0. INTENT MATCH ← MOST IMPORTANT CHECK
   Answer Question 3 above.
   If intent does not match, mark INCORRECT immediately — nothing else matters.

1. COLUMN VERIFICATION
   - Do all column names exist exactly in the schema? (check spelling, casing, quoting)
   - Is the correct column used for each metric?
     Example: if rules say "sales = SUM(Margin)" but SQL uses SUM(Revenue), that is INCORRECT
   - Are data types compatible with the operations applied?
     Example: comparing a TEXT column to a DATE using >= will fail or give wrong results

2. FILTER VERIFICATION
   - Are all mandatory filters from business rules present?
     Example: if a rule says "always exclude Rebate" but WHERE clause is missing this, it is INCORRECT
   - Are filter values exactly correct? (case-sensitive string matching matters)
   - Are date ranges correct per any fiscal year / quarter mapping rules?

3. AGGREGATION VERIFICATION
   - Does the aggregation match what the question asks for?
   - Is GROUP BY consistent with SELECT columns?
   - Are there missing or extra grouping columns?

4. JOIN VERIFICATION (if applicable)
   - Are join conditions using the correct columns from both tables?
   - Would the join produce duplicates or missing rows?

5. RESULTS SANITY
   - Do the results look reasonable for the question asked?
   - Are there unexpected NULLs, zeros, or extreme values?

## VERDICT CRITERIA
- CORRECT: Intent matches AND all other checks pass.
- INCORRECT: Intent does NOT match → always INCORRECT regardless of other checks.
  OR: Intent matches but another check fails.
  In either case, state: 'User asked for [X]. SQL produces [Y]. Missing/wrong: [gap].'
- UNCERTAIN: Intent plausibly matches but you cannot fully verify without data samples.

## OUTPUT (JSON only, no markdown)
{{"verdict": "CORRECT|INCORRECT|UNCERTAIN", "confidence": 0.0-1.0, "checks": {{"intent_match": "pass|fail|unknown", "columns": "pass|fail|unknown", "filters": "pass|fail|unknown", "aggregations": "pass|fail|unknown", "joins": "pass|fail|unknown|na", "results": "pass|fail|unknown"}}, "issues": ["specific problem if any — lead with intent gap if present"], "reasoning": "Start with intent verification result (Q1→Q2→Q3), then column/filter details. If intent fails, stop here."}}"""
    
    return prompt


# =============================================================================
# PROMPT TEMPLATES - REFINEMENT (After Opus Rejection)
# =============================================================================

def create_refinement_prompt(
    question: str,
    previous_sql: str,
    opus_feedback: Dict,
    schema: str,
    rules: str
) -> str:
    """
    Create prompt for SQL refinement after Opus rejection.
    Fixes only what Opus flagged, preserves everything else.
    """
    issues = opus_feedback.get("issues", [])
    reasoning = opus_feedback.get("reasoning", "")
    fix_suggestion = opus_feedback.get("fix", opus_feedback.get("guidance_for_regeneration", ""))
    checks = opus_feedback.get("checks", {})
    
    # Identify which checks failed to give focused fix instructions
    failed_checks = [k for k, v in checks.items() if v == "fail"]
    
    issues_text = "\n".join(f"• {issue}" for issue in issues) if issues else "• Unknown issues"
    failed_checks_text = ", ".join(failed_checks) if failed_checks else "unspecified"
    
    return f"""Fix a SQL query rejected by the auditor.

## WHAT FAILED
Failed checks: {failed_checks_text}

## SPECIFIC ISSUES FOUND
{issues_text}

## AUDITOR REASONING
{reasoning}

{f"## SUGGESTED FIX{chr(10)}{fix_suggestion}" if fix_suggestion else ""}

## BUSINESS RULES (must be applied correctly)
{rules}

## SCHEMA (use exact column names and compatible data types)
{schema}

## ORIGINAL QUESTION
{question}

## REJECTED SQL
```sql
{previous_sql}
```

## YOUR TASK
1. Produce SQL that ANSWERS the original question end-to-end.
2. Resolve every failed check above; do not keep logic that makes the query incomplete.
3. Add missing joins/metrics/filters when required by question or business rules.
4. If a type mismatch is possible (e.g. DATE/TEXT), cast safely.
5. Return a materially corrected query (not a cosmetic rewrite).

## OUTPUT (JSON only)
{{"analysis": "what you changed and why, referencing specific columns/rules", "sql": "corrected SQL query"}}"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL from LLM response (handles JSON and raw SQL).
    """
    import re
    
    # First, try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if "sql" in data:
                return data["sql"].strip()
        except:
            pass
    
    # Try to parse as raw JSON
    try:
        cleaned = response.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json|sql)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        
        if cleaned.startswith('{'):
            data = json.loads(cleaned)
            if "sql" in data:
                return data["sql"].strip()
    except:
        pass
    
    # Try to extract from SQL markdown code block
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try generic code block
    code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        if content.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
            return content
    
    # Last resort: if response looks like SQL
    cleaned = response.strip()
    if cleaned.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
        return cleaned
    
    return response.strip()


def extract_columns_from_sql(sql: str) -> List[str]:
    """
    Extract column names referenced in SQL.
    """
    import re
    
    columns = set()
    
    patterns = [
        r'"([^"]+)"',
        r'`([^`]+)`',
        r'\[([^\]]+)\]',
        r'\.([a-zA-Z_][a-zA-Z0-9_]*)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, sql)
        columns.update(matches)
    
    keywords = {
        'select', 'from', 'where', 'and', 'or', 'join', 'on', 'group', 
        'order', 'by', 'having', 'limit', 'as', 'inner', 'left', 'right',
        'outer', 'distinct', 'count', 'sum', 'avg', 'min', 'max', 'case',
        'when', 'then', 'else', 'end', 'null', 'not', 'in', 'like', 
        'between', 'is', 'asc', 'desc', 'true', 'false', 'cast', 'coalesce'
    }
    
    columns = {c for c in columns if c.lower() not in keywords}
    
    return list(columns)


def estimate_prompt_tokens(prompt: str) -> int:
    """
    Estimate token count for a prompt.
    Rough estimate: ~4 characters per token for English text.
    """
    return len(prompt) // 4


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PROMPT OPTIMIZER TEST")
    print("=" * 70)
    
    test_rules = [
        {
            "rule_name": "Revenue Metric",
            "rule_type": "metric",
            "rule_description": "Calculate revenue by summing amount excluding cancelled orders",
            "rule_data": {
                "formula": "SUM(Amount)",
                "condition": "Status <> 'Cancelled'",
                "note": "This is the primary revenue metric used across all reports"
            }
        },
        {
            "rule_name": "PostgreSQL Dialect",
            "rule_type": "dialect",
            "rule_data": {
                "dialect": "postgresql",
                "quote_char": '"',
                "string_quote": "'"
            }
        },
        {
            "rule_name": "Status Filter",
            "rule_type": "filter",
            "rule_data": {
                "sql_pattern": "Status IN ('Active', 'Completed', 'Pending')",
                "description": "Filter for valid order statuses"
            }
        }
    ]
    
    print("\nOriginal rules (verbose):")
    print("-" * 40)
    for rule in test_rules:
        print(f"  {rule['rule_name']}: {rule.get('rule_description', '')[:50]}...")
    
    compressed = compress_rules_for_llm(test_rules)
    print(f"\nCompressed rules ({len(compressed)} chars):")
    print("-" * 40)
    print(compressed)
    
    original_tokens = sum(len(str(r)) for r in test_rules) // 4
    compressed_tokens = len(compressed) // 4
    print(f"\nToken estimate: {original_tokens} -> {compressed_tokens} ({(1-compressed_tokens/original_tokens)*100:.0f}% reduction)")
    
    print("\n" + "=" * 70)
    print("SQL EXTRACTION TEST")
    print("=" * 70)
    
    test_responses = [
        '{"analysis": "Simple query", "sql": "SELECT * FROM users"}',
        '```sql\nSELECT * FROM orders\nWHERE status = \'active\'\n```',
        'SELECT COUNT(*) FROM products',
    ]
    
    for resp in test_responses:
        sql = extract_sql_from_response(resp)
        print(f"\nInput: {resp[:50]}...")
        print(f"Extracted: {sql}")
    
    print("\n" + "=" * 70)
