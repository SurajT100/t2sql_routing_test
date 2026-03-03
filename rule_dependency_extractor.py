"""
Rule Dependency Extractor
=========================
Parses business rule JSON structures to extract mandatory column dependencies
and upserts them into the rule_column_dependencies table.

Principle: LLMs reason about user intent. Backend enforces business policy.
Mandatory filter columns (e.g. a status column that must always be filtered)
must be deterministically injected — not left to LLM memory.

Detection strategy (generic — no hardcoded column names):
  - metric rules:   mandatory_filters[].column  +  the metric column itself
  - filter rules:   rule_data.table / rule_data.column  +  sql_pattern parsing
  - default rules:  sql_pattern / apply field parsing for table.column refs
  - join rules:     column1 / column2 from the join definition

Each extracted column is cross-referenced against schema_columns to avoid
false positives before being upserted into rule_column_dependencies.
"""

import re
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_rule_dependencies(
    vector_engine,
    rule_name: str,
    rule_data: dict,
    rule_type: str,
    verify_schema: bool = True
) -> int:
    """
    Extract column dependencies from a rule and upsert into
    rule_column_dependencies.

    Works for any database — parses the rule structure, not hardcoded names.

    Args:
        vector_engine:  SQLAlchemy engine connected to Supabase / metadata DB.
        rule_name:      Unique rule identifier (matches business_rules_v2.rule_name).
        rule_data:      Parsed rule JSON dict (rule_data column from business_rules_v2).
        rule_type:      One of 'metric', 'filter', 'default', 'join', 'critical', etc.
        verify_schema:  Cross-reference extracted columns against schema_columns.
                        Set False to skip verification (e.g. if schema not yet profiled).

    Returns:
        Number of dependency rows upserted.
    """
    deps = extract_columns_from_rule(rule_name, rule_data, rule_type)

    if not deps:
        return 0

    if verify_schema and vector_engine is not None:
        deps = _verify_columns_against_schema(vector_engine, deps)

    if not deps:
        return 0

    from sqlalchemy import text

    upserted = 0
    try:
        with vector_engine.connect() as conn:
            for table, column, dep_type, reason in deps:
                conn.execute(
                    text("""
                        INSERT INTO rule_column_dependencies
                            (table_name, column_name, rule_name,
                             reason, dependency_type, auto_apply)
                        VALUES
                            (:table, :col, :rule,
                             :reason, :dep_type, TRUE)
                        ON CONFLICT (table_name, column_name, rule_name)
                        DO UPDATE SET
                            reason          = EXCLUDED.reason,
                            dependency_type = EXCLUDED.dependency_type,
                            auto_apply      = EXCLUDED.auto_apply
                    """),
                    {
                        "table":    table,
                        "col":      column,
                        "rule":     rule_name,
                        "reason":   reason,
                        "dep_type": dep_type,
                    }
                )
                upserted += 1
            conn.commit()

        print(f"[RULE DEPS] Upserted {upserted} dependencies for rule '{rule_name}'")

    except Exception as e:
        print(f"[RULE DEPS] Error upserting dependencies for '{rule_name}': {e}")

    return upserted


# ---------------------------------------------------------------------------
# Extraction logic (pure — no DB calls)
# ---------------------------------------------------------------------------

def extract_columns_from_rule(
    rule_name: str,
    rule_data: dict,
    rule_type: str
) -> List[Tuple[str, str, str, str]]:
    """
    Parse a rule dict and return a list of
    (table_name, column_name, dependency_type, reason) tuples.

    Supports all rule types; gracefully handles missing fields.
    Does NOT cross-reference schema_columns (see _verify_columns_against_schema).
    """
    deps: List[Tuple[str, str, str, str]] = []

    if not isinstance(rule_data, dict):
        return deps

    # ------------------------------------------------------------------
    # Metric rules
    # ------------------------------------------------------------------
    if rule_type == "metric":
        table  = rule_data.get("table")
        column = rule_data.get("column")

        if table and column:
            deps.append((table, column, "metric",
                         f"Metric aggregation column for rule: {rule_name}"))

        # mandatory_filters are the primary injection targets for metric rules:
        # these columns MUST be present in every query that uses the metric.
        mandatory_filters = rule_data.get("mandatory_filters", [])
        if isinstance(mandatory_filters, list):
            for mf in mandatory_filters:
                if not isinstance(mf, dict):
                    continue
                mf_col = mf.get("column")
                if mf_col and table:
                    condition = mf.get("condition", "").lower()
                    dep_type = (
                        "date"
                        if any(k in condition for k in
                               ["fy", "date", "year", "month", "week", "quarter"])
                        else "filter"
                    )
                    deps.append((table, mf_col, dep_type,
                                 f"Mandatory filter for metric rule: {rule_name}"))

    # ------------------------------------------------------------------
    # Filter rules
    # ------------------------------------------------------------------
    elif rule_type == "filter":
        table  = rule_data.get("table")
        column = rule_data.get("column")

        if table and column:
            deps.append((table, column, "filter",
                         f"Filter column for rule: {rule_name}"))

        # Also parse the SQL pattern for additional column references
        sql_pat = rule_data.get("sql_pattern") or rule_data.get("apply", "")
        if sql_pat:
            for t, c in _extract_columns_from_sql(sql_pat):
                if not any(d[0] == t and d[1] == c for d in deps):
                    deps.append((t, c, "filter",
                                 f"Column extracted from SQL pattern: {rule_name}"))

    # ------------------------------------------------------------------
    # Default / critical rules — parse sql_pattern / apply field
    # ------------------------------------------------------------------
    elif rule_type in ("default", "critical"):
        sql_pat = rule_data.get("sql_pattern") or rule_data.get("apply", "")
        if sql_pat:
            for t, c in _extract_columns_from_sql(sql_pat):
                deps.append((t, c, "filter",
                             f"Column in default rule SQL pattern: {rule_name}"))

    # ------------------------------------------------------------------
    # Join rules — both join columns become dependencies
    # ------------------------------------------------------------------
    elif rule_type == "join":
        table1 = rule_data.get("table1")
        col1   = rule_data.get("column1")
        table2 = rule_data.get("table2")
        col2   = rule_data.get("column2")

        if table1 and col1:
            deps.append((table1, col1, "join",
                         f"Join key column for rule: {rule_name}"))
        if table2 and col2:
            deps.append((table2, col2, "join",
                         f"Join key column for rule: {rule_name}"))

        # Optionally parse join_condition for additional columns
        cond = rule_data.get("join_condition", "")
        if cond:
            for t, c in _extract_columns_from_sql(cond):
                if not any(d[0] == t and d[1] == c for d in deps):
                    deps.append((t, c, "join",
                                 f"Column in join condition: {rule_name}"))

    return deps


# ---------------------------------------------------------------------------
# SQL column reference parser
# ---------------------------------------------------------------------------

def _extract_columns_from_sql(sql_text: str) -> List[Tuple[str, str]]:
    """
    Extract (table, column) pairs from SQL text.

    Supports four quoting styles:
      1. "Table"."Column"       — PostgreSQL / standard SQL double-quotes
      2. `Table`.`Column`       — MySQL backticks
      3. [Table].[Column]       — SQL Server brackets
      4. Table.Column           — Unquoted (fallback, only when no quoted found)
    """
    results: List[Tuple[str, str]] = []

    # 1. Double-quoted: "Table"."Column"
    for m in re.finditer(r'"([^"]+)"\."([^"]+)"', sql_text):
        results.append((m.group(1), m.group(2)))

    # 2. Backtick-quoted: `Table`.`Column`
    if not results:
        for m in re.finditer(r'`([^`]+)`\.`([^`]+)`', sql_text):
            results.append((m.group(1), m.group(2)))

    # 3. Bracket-quoted: [Table].[Column]
    if not results:
        for m in re.finditer(r'\[([^\]]+)\]\.\[([^\]]+)\]', sql_text):
            results.append((m.group(1), m.group(2)))

    # 4. Unquoted fallback: Table.Column
    if not results:
        for m in re.finditer(r'\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\b',
                              sql_text):
            results.append((m.group(1), m.group(2)))

    return results


# ---------------------------------------------------------------------------
# Schema cross-reference
# ---------------------------------------------------------------------------

def _verify_columns_against_schema(
    vector_engine,
    deps: List[Tuple[str, str, str, str]]
) -> List[Tuple[str, str, str, str]]:
    """
    Filter deps to only those where (table_name, column_name) exist in
    schema_columns.  Checks both the full name and the bare name (without
    schema prefix) to handle cases where Pass 1 uses short names.

    On any DB error for a specific row, that row is kept (fail-open) to avoid
    silently dropping genuine dependencies due to transient errors.
    """
    from sqlalchemy import text

    verified: List[Tuple[str, str, str, str]] = []

    for table, column, dep_type, reason in deps:
        try:
            with vector_engine.connect() as conn:
                row = conn.execute(
                    text("""
                        SELECT 1 FROM schema_columns
                        WHERE object_name = :table
                          AND column_name  = :col
                        LIMIT 1
                    """),
                    {"table": table, "col": column}
                ).fetchone()

                # Retry with bare name if schema-prefixed name not found
                if row is None and "." in table:
                    bare = table.split(".")[-1]
                    row = conn.execute(
                        text("""
                            SELECT 1 FROM schema_columns
                            WHERE object_name = :table
                              AND column_name  = :col
                            LIMIT 1
                        """),
                        {"table": bare, "col": column}
                    ).fetchone()

                if row is not None:
                    verified.append((table, column, dep_type, reason))
                else:
                    print(f"[RULE DEPS] Skipping {table}.{column} "
                          f"— not found in schema_columns")

        except Exception as e:
            # Fail-open: keep the dep on transient error
            print(f"[RULE DEPS] Schema check error for {table}.{column}: {e}")
            verified.append((table, column, dep_type, reason))

    return verified
