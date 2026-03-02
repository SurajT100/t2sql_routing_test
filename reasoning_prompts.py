"""
Two-Pass Reasoning Prompts + Metadata Fetch
=============================================
Implements the two-pass reasoning architecture:

Pass 1 — Column Identification (cheap, short output)
  Input:  question + schema + rules
  Output: tables + columns needed ONLY

Metadata Fetch — NO LLM
  Input:  Pass 1 column list
  Output: sample values, data types from schema_columns table

Pass 2 — Full Plan with Metadata Awareness (no schema/rules repeated)
  Input:  question + Pass 1 output + metadata
  Output: complete query plan (NO SQL — intent only)

SQL Coder receives Pass 2 plan only.

Error Retry — reuses Pass 2 plan + metadata + error (no re-fetch)
  Input:  Pass 2 plan + metadata + error message
  Output: corrected plan → back to SQL Coder

Usage:
    from reasoning_prompts import (
        create_pass1_prompt,
        fetch_column_metadata,
        create_pass2_prompt,
        create_error_retry_prompt,
        create_opus_complex_prompt
    )
"""

import json
from typing import Dict, List, Any, Optional


# =============================================================================
# PASS 1 — COLUMN IDENTIFICATION ONLY
# =============================================================================

def create_pass1_prompt(
    question: str,
    schema: str,
    rules: str,
    dialect_info: Dict = None
) -> str:
    """
    Pass 1: Identify tables and columns needed.
    Short output — just column/table list, no plan yet.
    Schema + rules included here ONLY (not repeated in Pass 2).

    Args:
        question: User question
        schema: Full schema text
        rules: Compressed business rules
        dialect_info: Dialect config

    Returns:
        Prompt string for Pass 1
    """
    dialect_name = dialect_info.get("dialect", "postgresql").upper() if dialect_info else "POSTGRESQL"
    quote_char = dialect_info.get("quote_char", '"') if dialect_info else '"'

    return f"""You are a SQL query planner. Your ONLY task right now is to identify 
which tables and columns are needed to answer this question.

DATABASE: {dialect_name}
COLUMN FORMAT: {quote_char}column_name{quote_char}

SCHEMA:
{schema}

BUSINESS RULES:
{rules}

QUESTION: {question}

TASK:
1. Identify which tables are needed
2. For each table, list ALL columns needed (SELECT, WHERE, GROUP BY, ORDER BY)
3. Flag any column where the user provided a string value that needs matching
   (names of people, categories, codes — anything user typed as a filter value)

OUTPUT (JSON only):
{{
  "tables": ["schema.table1", "schema.table2"],
  "columns": {{
    "table1": ["col1", "col2", "col3"],
    "table2": ["col4"]
  }},
  "string_filter_columns": [
    {{
      "table": "table1",
      "column": "person_name_column",
      "user_value": "partial_value_user_typed",
      "filter_type": "exclude"
    }}
  ],
  "joins_needed": true
}}

RULES:
- List every column that will be used anywhere in the query
- string_filter_columns = columns where user provided a human-readable value
  that may not match exactly (names, partial codes, categories)
- Do NOT write SQL or plan aggregations yet — just column identification"""


# =============================================================================
# METADATA FETCH — NO LLM
# =============================================================================

def fetch_column_metadata(
    vector_engine,
    columns_by_table: Dict[str, List[str]],
    string_filter_columns: List[Dict] = None
) -> Dict[str, Any]:
    """
    Fetch sample values, data types, and descriptions from schema_columns.
    NO LLM — direct DB query.

    Args:
        vector_engine: Supabase connection
        columns_by_table: {"table1": ["col1", "col2"], "table2": ["col3"]}
        string_filter_columns: From Pass 1 output — columns needing lookup

    Returns:
        {
          "table1": {
            "col1": {
              "data_type": "text",
              "sample_values": ["value1", "value2", "value3"],
              "description": "column description from profiling",
              "needs_partial_match": true
            },
            "col2": {
              "data_type": "text",
              "sample_values": ["cat1", "cat2", "cat3"],
              "description": "another column description"
            }
          }
        }
    """
    metadata = {}

    # Build set of string filter columns for quick lookup
    string_filter_set = set()
    if string_filter_columns:
        for sf in string_filter_columns:
            string_filter_set.add((sf.get("table", ""), sf.get("column", "")))

    try:
        for table_name, columns in columns_by_table.items():
            metadata[table_name] = {}

            for col in columns:
                try:
                    # Query schema_columns table (already populated during DB profiling)
                    result = vector_engine.table("schema_columns").select(
                        "column_name, data_type, sample_values, opus_description"
                    ).eq("table_name", table_name).eq("column_name", col).execute()

                    if result.data:
                        row = result.data[0]
                        sample_values = row.get("sample_values") or []

                        # Parse if stored as JSON string
                        if isinstance(sample_values, str):
                            try:
                                sample_values = json.loads(sample_values)
                            except Exception:
                                sample_values = []

                        # Limit to 10 samples — enough for context, not too many tokens
                        sample_values = sample_values[:10]

                        # Flag if this column needs partial match consideration
                        needs_partial = (table_name, col) in string_filter_set

                        metadata[table_name][col] = {
                            "data_type": row.get("data_type", "unknown"),
                            "sample_values": sample_values,
                            "description": row.get("opus_description", ""),
                            "needs_partial_match": needs_partial
                        }
                    else:
                        # Column not found in schema_columns — store minimal info
                        metadata[table_name][col] = {
                            "data_type": "unknown",
                            "sample_values": [],
                            "description": "",
                            "needs_partial_match": (table_name, col) in string_filter_set
                        }

                except Exception as col_err:
                    print(f"[METADATA] Could not fetch {table_name}.{col}: {col_err}")
                    metadata[table_name][col] = {
                        "data_type": "unknown",
                        "sample_values": [],
                        "description": "",
                        "needs_partial_match": False
                    }

    except Exception as e:
        print(f"[METADATA] Fetch failed: {e}")

    return metadata


def format_metadata_for_prompt(metadata: Dict[str, Any]) -> str:
    """
    Format metadata dict into compact prompt section.
    Keeps tokens low — only essential info.

    Returns:
        Formatted string for injection into Pass 2 prompt
    """
    if not metadata:
        return "No metadata available."

    lines = []
    for table, columns in metadata.items():
        lines.append(f"Table: {table}")
        for col, info in columns.items():
            data_type = info.get("data_type", "unknown")
            samples = info.get("sample_values", [])
            desc = info.get("description", "")
            needs_partial = info.get("needs_partial_match", False)

            # Compact format
            sample_str = f"samples={samples}" if samples else "no samples"
            partial_flag = " ⚠️ PARTIAL MATCH LIKELY NEEDED" if needs_partial else ""
            desc_str = f" | {desc[:80]}" if desc else ""

            lines.append(
                f"  • {col} ({data_type}): {sample_str}{desc_str}{partial_flag}"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# PASS 2 — FULL PLAN WITH METADATA AWARENESS
# =============================================================================

def create_pass2_prompt(
    question: str,
    pass1_output: str,
    metadata: Dict[str, Any],
    dialect_info: Dict = None,
    resolver_text: str = "",
    rules: str = "[]"
) -> str:
    """
    Pass 2: Build complete query plan using Pass 1 columns + metadata.
    NO schema repeated — already seen in Pass 1.
    Rules NOW come from Context Agent (focused retrieval after Pass 1).
    Only new information: metadata with actual stored values + resolver results + rules.

    Args:
        question: Original user question
        pass1_output: JSON output from Pass 1 (tables + columns identified)
        metadata: From Context Agent (descriptions + sample values + data types)
        dialect_info: Dialect config
        resolver_text: Formatted entity resolutions from live DB (optional)
        rules: Business rules from Context Agent (focused on identified columns)

    Returns:
        Prompt string for Pass 2
    """
    dialect_name = dialect_info.get("dialect", "postgresql").upper() if dialect_info else "POSTGRESQL"
    quote_char = dialect_info.get("quote_char", '"') if dialect_info else '"'
    string_quote = dialect_info.get("string_quote", "'") if dialect_info else "'"

    metadata_text = format_metadata_for_prompt(metadata)

    # Build resolver section if available
    resolver_section = ""
    if resolver_text:
        resolver_section = f"""

{resolver_text}
"""

    # Build rules section if available
    rules_section = ""
    if rules and rules != "[]":
        rules_section = f"""
BUSINESS RULES (relevant to identified columns):
{rules}
"""

    return f"""You are completing a SQL query plan. You already identified the tables 
and columns needed. Now use the actual stored values to finalize all filters correctly.

DATABASE: {dialect_name}
COLUMN FORMAT: {quote_char}column_name{quote_char}
STRING FORMAT: {string_quote}value{string_quote}

ORIGINAL QUESTION: {question}

YOUR PASS 1 OUTPUT (tables + columns already identified):
{pass1_output}

ACTUAL COLUMN METADATA (real stored values from database):
{metadata_text}
{resolver_section}{rules_section}
TASK: Build the complete query plan using metadata awareness.

CRITICAL FILTER RULES:
- If ENTITY RESOLUTIONS are provided above, use those resolved conditions EXACTLY
  — they come from the live database and override any guessing from samples
- If column data_type is timestamp/date but filter needs date comparison
  → Note the casting needed
- If sample values show the exact value exists as-is
  → Exact match is fine (e.g. user said "Active", samples contain "Active")
- If ⚠️ PARTIAL MATCH LIKELY NEEDED is flagged AND no entity resolution is available
  → You MUST use wildcards, never exact match

OUTPUT (JSON only, NO SQL — intent and plan only):
{{
  "understanding": "What user wants in plain terms",
  "tables": ["schema.table"],
  "columns": {{"table": ["col1", "col2"]}},
  "joins": [],
  "filters": [
    {{
      "column": "status_column",
      "condition": " = 'Open'",
      "reason": "show open status"
    }},
    {{
      "column": "category_column",
      "condition": "<> 'ExactValue'",
      "reason": "exact match confirmed from samples"
    }}
  ],
  "aggregations": ["SUM(col)"],
  "group_by": ["col"],
  "order_by": ["col DESC"],
  "notes": "any edge cases or casting needed"
}}

IMPORTANT: Output is a PLAN — no SQL syntax in filters, write conditions as strings only."""


# =============================================================================
# ERROR RETRY — REUSES PASS 2 PLAN + METADATA + ERROR
# =============================================================================

def create_error_retry_prompt(
    question: str,
    pass2_plan: str,
    metadata: Dict[str, Any],
    failed_sql: str,
    error_message: str,
    dialect_info: Dict = None,
    use_opus: bool = False
) -> str:
    """
    Error retry: reuses Pass 2 plan + metadata.
    No re-fetch, no schema/rules repetition.

    Args:
        question: Original user question
        pass2_plan: JSON output from Pass 2
        metadata: Already fetched metadata (reused, no DB call)
        failed_sql: The SQL that failed
        error_message: Database error message
        dialect_info: Dialect config
        use_opus: If True, add stronger instructions (for Opus retry)

    Returns:
        Prompt string for error retry
    """
    dialect_name = dialect_info.get("dialect", "postgresql").upper() if dialect_info else "POSTGRESQL"
    quote_char = dialect_info.get("quote_char", '"') if dialect_info else '"'
    string_quote = dialect_info.get("string_quote", "'") if dialect_info else "'"

    metadata_text = format_metadata_for_prompt(metadata)

    opus_instruction = """
You are Opus — fix this with maximum accuracy. Check every column name,
every data type, every filter value against the metadata below.""" if use_opus else ""

    return f"""Fix a failed SQL query using the original plan and column metadata.{opus_instruction}

DATABASE: {dialect_name}
COLUMN FORMAT: {quote_char}column_name{quote_char}
STRING FORMAT: {string_quote}value{string_quote}

ORIGINAL QUESTION: {question}

ORIGINAL PLAN (what was intended):
{pass2_plan}

COLUMN METADATA (actual stored values — use this to verify every filter):
{metadata_text}

FAILED SQL:
{failed_sql}

ERROR:
{error_message}

TASK:
1. Identify what caused the error (type mismatch, wrong column name, syntax, etc.)
2. Cross-check every column and filter value against the metadata above
3. Fix ONLY the specific error — preserve all tables, JOINs, and query structure exactly as in the failed SQL

Return ONLY the corrected SQL query. No explanation, no markdown, no JSON."""


# =============================================================================
# OPUS COMPLEX — SINGLE CALL FOR COMPLEX QUERIES
# =============================================================================

def create_opus_complex_prompt(
    question: str,
    schema: str,
    rules: str,
    metadata: Dict[str, Any],
    dialect_info: Dict = None,
    resolver_text: str = ""
) -> str:
    """
    Opus single-call prompt for COMPLEX queries.
    Opus handles reasoning + SQL writing in one call.
    Used when classifier routes to COMPLEX level.

    Args:
        question: User question
        schema: Full schema
        rules: Compressed business rules
        metadata: Pre-fetched column metadata (after quick column scan)
        dialect_info: Dialect config
        resolver_text: Formatted entity resolutions from live DB (optional)

    Returns:
        Prompt for Opus to reason + write SQL in one pass
    """
    dialect_name = dialect_info.get("dialect", "postgresql").upper() if dialect_info else "POSTGRESQL"
    quote_char = dialect_info.get("quote_char", '"') if dialect_info else '"'
    string_quote = dialect_info.get("string_quote", "'") if dialect_info else "'"

    metadata_text = format_metadata_for_prompt(metadata) if metadata else "Not available."

    # Build resolver section if available
    resolver_section = ""
    if resolver_text:
        resolver_section = f"""

{resolver_text}
"""

    return f"""You are an expert SQL engineer. Generate a precise SQL query for this complex question.
This query has been flagged as COMPLEX — it likely involves name exclusions, 
partial matching, or custom date ranges that require careful handling.

DATABASE: {dialect_name}
COLUMN FORMAT: {quote_char}column_name{quote_char}  
STRING FORMAT: {string_quote}value{string_quote}

SCHEMA:
{schema}

BUSINESS RULES:
{rules}

COLUMN METADATA (actual stored values — use this for all filter decisions):
{metadata_text}
{resolver_section}
QUESTION: {question}

COMPLEX QUERY RULES:
- If ENTITY RESOLUTIONS are provided above, use those resolved conditions EXACTLY
- Date ranges: if user specifies custom period default date filters
- NULL handling: for exclusions, always add OR col IS NULL

OUTPUT (JSON only):
{{
  "reasoning": "how you interpreted the question and handled edge cases",
  "filters_decided": [
    {{"column": "col", "condition": " = 'condition'", "why": "partial name match"}}
  ],
  "sql": "complete SQL query"
}}"""


# =============================================================================
# UTILITY — Parse Pass 1 output safely
# =============================================================================

def parse_pass1_output(response: str) -> Dict:
    """
    Safely parse Pass 1 JSON output.
    Returns empty dict on failure.
    """
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            import re
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        return json.loads(cleaned)
    except Exception as e:
        print(f"[PASS1 PARSE] Failed: {e}")
        return {
            "tables": [],
            "columns": {},
            "string_filter_columns": [],
            "joins_needed": False
        }


def parse_pass2_output(response: str) -> Dict:
    """
    Safely parse Pass 2 JSON output.
    Returns empty dict on failure.
    """
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            import re
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        return json.loads(cleaned)
    except Exception as e:
        print(f"[PASS2 PARSE] Failed: {e}")
        return {}
