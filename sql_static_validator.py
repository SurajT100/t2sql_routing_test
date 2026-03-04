"""
SQL Static Validator - Tier 1 Pre-execution Check
==================================================
Validates SQL before execution using column metadata and dialect rules.
No LLM calls — pure pattern matching.

Input:  SQL string + column metadata (from schema_columns) + dialect info
Output: List of structured issues found (empty = clean)

Issue types:
  type_mismatch         - Aggregation on wrong type, or date vs text comparison
  group_by_incomplete   - Non-aggregated SELECT column missing from GROUP BY
  missing_mandatory_filter - Required filter column absent from WHERE/HAVING
  quote_style           - Identifier quoting doesn't match the dialect
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class StaticIssue:
    """A single static validation issue found in the SQL."""
    issue_type: str   # type_mismatch | group_by_incomplete | missing_mandatory_filter | quote_style
    location: str     # Which part of SQL (e.g., "SELECT / GROUP BY", "WHERE clause")
    column: str       # Affected column name (empty if N/A)
    column_type: str  # Actual type from metadata (empty if N/A)
    detail: str       # Human-readable description of the problem

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Dialect configuration
# ─────────────────────────────────────────────────────────────────────────────

_DIALECT_QUOTE_RULES: Dict[str, Dict] = {
    "postgresql": {"correct": '"', "wrong_chars": ["`"], "wrong_patterns": [r"\[[\w\s]+\]"]},
    "mysql":      {"correct": "`", "wrong_chars": ['"'], "wrong_patterns": [r"\[[\w\s]+\]"]},
    "mssql":      {"correct": "[", "wrong_chars": ["`"], "wrong_patterns": []},
    "oracle":     {"correct": '"', "wrong_chars": ["`"], "wrong_patterns": [r"\[[\w\s]+\]"]},
    "sqlite":     {"correct": '"', "wrong_chars": ["`"], "wrong_patterns": [r"\[[\w\s]+\]"]},
}

# Column data types considered text (not numeric)
_TEXT_TYPES = frozenset({
    "text", "varchar", "character varying", "char", "character",
    "string", "nvarchar", "nchar", "clob", "nclob",
    "mediumtext", "longtext", "tinytext", "enum", "set",
})

# Numeric aggregation functions that require numeric columns
_NUMERIC_AGGS = frozenset({"SUM", "AVG", "STDDEV", "VARIANCE", "STDEV", "STDDEV_POP", "STDDEV_SAMP"})

# All aggregation functions (for GROUP BY check)
_ALL_AGGS = frozenset({
    "SUM", "AVG", "COUNT", "MIN", "MAX", "STDDEV", "VARIANCE", "STDEV",
    "GROUP_CONCAT", "STRING_AGG", "ARRAY_AGG", "LISTAGG", "MEDIAN",
    "PERCENTILE_CONT", "PERCENTILE_DISC", "STDDEV_POP", "STDDEV_SAMP",
    "FIRST_VALUE", "LAST_VALUE",
})

# Date/time comparison patterns
_DATE_COMPARISON_PATTERNS = [
    r"\bCURRENT_DATE\b",
    r"\bCURRENT_TIMESTAMP\b",
    r"\bNOW\s*\(\s*\)",
    r"\bGETDATE\s*\(\s*\)",
    r"\bSYSDATE\b",
    r"\bTODAY\s*\(\s*\)",
    r"DATE\s*'[\d\-]+'",
    r"TIMESTAMP\s*'[\d\-\s:\.]+[\+\-\d:]*'",
    r"'\d{4}-\d{2}-\d{2}'",   # bare date literal
]
_DATE_COMPARISON_RE = re.compile(
    "(" + "|".join(_DATE_COMPARISON_PATTERNS) + ")", re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_sql_static(
    sql: str,
    column_metadata: Dict[str, Any],
    dialect: str = "postgresql",
    mandatory_columns: Optional[List[str]] = None,
) -> List[StaticIssue]:
    """
    Run all static pre-execution checks on *sql*.

    Args:
        sql:               Raw SQL string to check.
        column_metadata:   Dict of {column_name: {"data_type": ..., ...}} from
                           context agent / schema_columns table.  May be empty.
        dialect:           Database dialect string.
        mandatory_columns: Column names that MUST appear in WHERE / HAVING.

    Returns:
        List of StaticIssue.  Empty list means no issues found.
    """
    if not sql or not sql.strip():
        return []

    issues: List[StaticIssue] = []

    # 1. Quote-style check (always run)
    issues.extend(_check_quote_style(sql, dialect))

    if column_metadata:
        # 2. Numeric aggregation on text columns
        issues.extend(_check_aggregation_types(sql, column_metadata))

        # 3. Date/timestamp comparison against text-typed column
        issues.extend(_check_date_comparisons(sql, column_metadata))

    # 4. GROUP BY completeness
    issues.extend(_check_group_by(sql))

    # 5. Mandatory filter columns
    if mandatory_columns:
        issues.extend(_check_mandatory_filters(sql, mandatory_columns))

    return issues


def build_tier1_mini_retry_prompt(
    original_sql: str,
    issues: List[StaticIssue],
    dialect: str,
    quote_char: str,
) -> str:
    """
    Build the mini-retry prompt sent to SQL Coder when Tier 1 finds issues.
    Small prompt: SQL + diagnosis + dialect only.  No schema, rules, or plan.
    """
    issue_lines = "\n".join(
        "  [" + i.issue_type.upper() + "] " + i.detail + "  (location: " + i.location + ")"
        for i in issues
    )
    return (
        "Fix the following SQL issues in this " + dialect.upper() + " query.\n"
        "Only fix the listed issues — do not change anything else.\n\n"
        "DIALECT: " + dialect.upper() + " "
        "(use " + quote_char + " for identifiers, single quotes for strings)\n\n"
        "FAILED SQL:\n" + original_sql + "\n\n"
        "ISSUES TO FIX:\n" + issue_lines + "\n\n"
        "Return ONLY the corrected SQL. No explanation. Start with SELECT or WITH."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Check implementations
# ─────────────────────────────────────────────────────────────────────────────

def _strip_string_literals(sql: str) -> str:
    """Replace single-quoted string literals with '' to avoid false matches."""
    return re.sub(r"'(?:[^'\\]|\\.)*'", "''", sql)


def _check_quote_style(sql: str, dialect: str) -> List[StaticIssue]:
    """Detect identifier quoting that doesn't match the expected dialect style."""
    rules = _DIALECT_QUOTE_RULES.get(dialect, _DIALECT_QUOTE_RULES["postgresql"])
    sql_no_strings = _strip_string_literals(sql)
    issues: List[StaticIssue] = []

    for bad_char in rules["wrong_chars"]:
        if bad_char in sql_no_strings:
            issues.append(StaticIssue(
                issue_type="quote_style",
                location="identifiers",
                column="",
                column_type="",
                detail=(
                    "Found '" + bad_char + "' identifier quoting, but dialect '"
                    + dialect + "' expects '" + rules["correct"] + "' for identifiers."
                ),
            ))
            break

    for pattern in rules.get("wrong_patterns", []):
        if re.search(pattern, sql_no_strings):
            issues.append(StaticIssue(
                issue_type="quote_style",
                location="identifiers",
                column="",
                column_type="",
                detail=(
                    "Found square-bracket [identifier] quoting but dialect '"
                    + dialect + "' expects '" + rules["correct"] + "' for identifiers."
                ),
            ))
            break

    return issues


def _check_aggregation_types(
    sql: str,
    column_metadata: Dict[str, Any],
) -> List[StaticIssue]:
    """Flag SUM/AVG/STDDEV/VARIANCE applied to text-typed columns."""
    issues: List[StaticIssue] = []

    agg_re = re.compile(
        r'\b(' + "|".join(_NUMERIC_AGGS) + r')\s*\(\s*'
        r'(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([A-Za-z_]\w*))\s*\)',
        re.IGNORECASE,
    )

    for m in agg_re.finditer(sql):
        func = m.group(1).upper()
        col_name = m.group(2) or m.group(3) or m.group(4) or m.group(5)
        if not col_name:
            continue

        meta = _find_column_meta(col_name, column_metadata)
        if meta:
            col_type = _get_type(meta)
            if _is_text_type(col_type):
                issues.append(StaticIssue(
                    issue_type="type_mismatch",
                    location=func + "(" + col_name + ")",
                    column=col_name,
                    column_type=col_type,
                    detail=(
                        func + "() applied to column '" + col_name + "' which has type '"
                        + col_type + "'. Numeric aggregation on a text column will fail."
                    ),
                ))

    return issues


def _check_date_comparisons(
    sql: str,
    column_metadata: Dict[str, Any],
) -> List[StaticIssue]:
    """Flag WHERE/HAVING comparisons of text-typed columns against date/timestamp values."""
    issues: List[StaticIssue] = []

    where_text = _extract_where_having(sql)
    if not where_text:
        return issues

    if not _DATE_COMPARISON_RE.search(where_text):
        return issues

    col_re = re.compile(
        r'(?:"([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([A-Za-z_]\w*))\s*'
        r'[><=!]+\s*'
        r'(' + "|".join(_DATE_COMPARISON_PATTERNS) + r')',
        re.IGNORECASE,
    )

    for m in col_re.finditer(where_text):
        col_name = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        if not col_name:
            continue

        meta = _find_column_meta(col_name, column_metadata)
        if meta:
            col_type = _get_type(meta)
            if _is_text_type(col_type):
                issues.append(StaticIssue(
                    issue_type="type_mismatch",
                    location="WHERE / HAVING clause",
                    column=col_name,
                    column_type=col_type,
                    detail=(
                        "Column '" + col_name + "' (stored as '" + col_type + "') is compared "
                        "to a date/timestamp expression. The comparison will fail or produce "
                        "wrong results — cast the column or use string literals."
                    ),
                ))

    return issues


def _check_group_by(sql: str) -> List[StaticIssue]:
    """
    Detect non-aggregated SELECT columns that are absent from GROUP BY.
    Deliberately conservative: only flag when we are confident.
    """
    sql_flat = " ".join(sql.split())
    sql_upper = sql_flat.upper()

    if "GROUP BY" not in sql_upper:
        return []

    select_m = re.search(r"\bSELECT\s+(.*?)\s+FROM\b", sql_flat, re.IGNORECASE | re.DOTALL)
    group_m = re.search(
        r"\bGROUP\s+BY\s+(.*?)(?:\s+HAVING\b|\s+ORDER\s+BY\b|\s+LIMIT\b|\s+UNION\b|\s*$)",
        sql_flat, re.IGNORECASE | re.DOTALL,
    )
    if not select_m or not group_m:
        return []

    select_clause = select_m.group(1).strip()
    group_by_clause = group_m.group(1).strip()

    group_by_exprs: set = set()
    for g in _split_csv(group_by_clause):
        group_by_exprs.add(_normalise_expr(g))

    missing: List[str] = []

    for col_expr in _split_csv(select_clause):
        col_expr = col_expr.strip()
        if not col_expr or col_expr == "*":
            continue

        col_no_alias = re.sub(r"\s+AS\s+\w+\s*$", "", col_expr, flags=re.IGNORECASE).strip()

        if _contains_aggregate(col_no_alias):
            continue

        if re.match(r"^[\d\.]+$", col_no_alias) or re.match(r"^'[^']*'$", col_no_alias):
            continue

        normalised = _normalise_expr(col_no_alias)
        if not normalised:
            continue

        bare_col = _bare_identifier(col_no_alias)
        if normalised not in group_by_exprs and (bare_col not in group_by_exprs):
            missing.append(bare_col or normalised)

    if missing:
        return [StaticIssue(
            issue_type="group_by_incomplete",
            location="SELECT / GROUP BY",
            column=", ".join(missing[:3]),
            column_type="",
            detail=(
                "Non-aggregated SELECT column(s) not found in GROUP BY: "
                + ", ".join(missing[:3]) + ". "
                "Add them to GROUP BY or wrap in an aggregate function."
            ),
        )]
    return []


def _check_mandatory_filters(
    sql: str,
    mandatory_columns: List[str],
) -> List[StaticIssue]:
    """Check that each mandatory column appears in WHERE or HAVING."""
    issues: List[StaticIssue] = []
    where_text = _extract_where_having(sql) or ""

    for col in mandatory_columns:
        if not col:
            continue
        patterns = [
            r'"(' + re.escape(col) + r')"',
            r'`(' + re.escape(col) + r')`',
            r'\[(' + re.escape(col) + r')\]',
            r'\b(' + re.escape(col) + r')\b',
        ]
        if not any(re.search(p, where_text, re.IGNORECASE) for p in patterns):
            issues.append(StaticIssue(
                issue_type="missing_mandatory_filter",
                location="WHERE / HAVING clause",
                column=col,
                column_type="",
                detail=(
                    "Mandatory filter column '" + col + "' is missing from WHERE / HAVING. "
                    "This query may return unfiltered data across all '" + col + "' values."
                ),
            ))

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_column_meta(
    col_name: str, column_metadata: Dict[str, Any]
) -> Optional[Dict]:
    col_lower = col_name.lower()
    for key, val in column_metadata.items():
        if key.lower() == col_lower:
            return val
    return None


def _get_type(meta: Dict) -> str:
    return str(meta.get("data_type", meta.get("type", ""))).lower().strip()


def _is_text_type(col_type: str) -> bool:
    return any(t in col_type for t in _TEXT_TYPES)


def _extract_where_having(sql: str) -> Optional[str]:
    m = re.search(
        r"\bWHERE\b(.*?)(?=\bORDER\s+BY\b|\bLIMIT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|$)",
        sql, re.IGNORECASE | re.DOTALL,
    )
    return m.group(1) if m else None


def _split_csv(clause: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    for ch in clause:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _contains_aggregate(expr: str) -> bool:
    return bool(re.search(
        r'\b(' + "|".join(_ALL_AGGS) + r')\s*\(', expr, re.IGNORECASE
    ))


def _normalise_expr(expr: str) -> str:
    s = re.sub(r'["`\[\]]', "", expr).strip().lower()
    return s


def _bare_identifier(expr: str) -> str:
    s = re.sub(r'["`\[\]]', "", expr).strip()
    parts = s.rsplit(".", 1)
    return parts[-1].strip().lower()
