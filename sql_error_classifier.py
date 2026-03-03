"""
SQL Error Classifier - Tier 2 Post-execution Error Classification
=================================================================
Classifies database execution errors into structured diagnoses.
Uses per-dialect regex patterns.  No LLM calls.

Input:  error string from database + SQL + dialect string
Output: ErrorDiagnosis (or None if unclassifiable)

Error categories:
  alias_error       - table/subquery alias referenced incorrectly
  missing_column    - column or table/relation does not exist
  type_mismatch     - operator applied to incompatible types
  invalid_syntax    - date format wrong, unexpected token, bad cast
  division_by_zero  - divide-by-zero encountered at runtime
  group_by_error    - column must appear in GROUP BY or aggregate
  ambiguous_column  - column name matches multiple tables

Confidence levels:
  high   -> mini-retry with SQL Coder; if still fails -> Tier 3
  medium -> mini-retry with SQL Coder; if still fails -> Tier 3
  low    -> skip mini-retry, go directly to Tier 3
"""

import re
from typing import Optional, Dict, Callable
from dataclasses import dataclass, asdict


@dataclass
class ErrorDiagnosis:
    """Structured diagnosis of a SQL execution error."""
    error_category: str
    affected_expression: str
    suggested_fix_description: str
    confidence: str
    raw_error: str

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify_sql_error(
    error_message: str,
    sql: str,
    dialect: str = "postgresql",
) -> Optional[ErrorDiagnosis]:
    """
    Classify a SQL execution error.

    Tries dialect-specific patterns first, then generic cross-dialect patterns.
    Returns ErrorDiagnosis if classifiable, None otherwise.
    """
    if not error_message:
        return None

    error_lower = error_message.lower()

    _dialect_classifiers: Dict[str, Callable] = {
        "postgresql": _classify_postgresql,
        "mysql":      _classify_mysql,
        "mssql":      _classify_mssql,
        "oracle":     _classify_oracle,
        "sqlite":     _classify_sqlite,
    }

    specific = _dialect_classifiers.get(dialect)
    if specific:
        result = specific(error_message, error_lower, sql)
        if result:
            return result

    for name, clf in _dialect_classifiers.items():
        if name == dialect:
            continue
        result = clf(error_message, error_lower, sql)
        if result:
            return result

    return _classify_generic(error_message, error_lower, sql)


def build_tier2_mini_retry_prompt(
    original_sql: str,
    diagnosis: "ErrorDiagnosis",
    dialect: str,
    quote_char: str,
) -> str:
    """
    Build the mini-retry prompt sent to SQL Coder when Tier 2 classifies an error.
    Small prompt: SQL + diagnosis + dialect only.  No schema, rules, or plan.
    """
    return (
        "Fix this failed " + dialect.upper() + " SQL query based on the error diagnosis below.\n"
        "Only fix the diagnosed issue — do not change anything else.\n\n"
        "DIALECT: " + dialect.upper() + " "
        "(use " + quote_char + " for identifiers, single quotes for strings)\n\n"
        "FAILED SQL:\n" + original_sql + "\n\n"
        "ERROR DIAGNOSIS:\n"
        "  Error type:   " + diagnosis.error_category + "\n"
        "  Expression:   " + diagnosis.affected_expression + "\n"
        "  Fix to apply: " + diagnosis.suggested_fix_description + "\n\n"
        "Return ONLY the corrected SQL. No explanation. Start with SELECT or WITH."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_token(error_msg: str) -> str:
    for pat in [r'"([^"]+)"', r"'([^']+)'", r"`([^`]+)`"]:
        m = re.search(pat, error_msg)
        if m:
            return m.group(1)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def _classify_postgresql(error_msg, error_lower, sql):
    m = re.search(r'column "([^"]+)" does not exist', error_lower)
    if m:
        col = m.group(1)
        return ErrorDiagnosis("missing_column", '"' + col + '"',
            "Column '" + col + "' does not exist. Check column name spelling, schema prefix, and quoting.",
            "high", error_msg)

    m = re.search(r"operator does not exist:\s*(\w[\w\s]*?)\s*([\>\<=!]+)\s*(\w+)", error_lower)
    if m:
        left_type, op, right_type = m.group(1).strip(), m.group(2), m.group(3)
        token = _extract_token(error_msg)
        return ErrorDiagnosis("type_mismatch", token or (left_type + " " + op + " " + right_type),
            "Type mismatch: '" + left_type + "' vs '" + right_type + "'. "
            "Cast the column — e.g., col::date, CAST(col AS DATE), or CAST(col AS INTEGER).",
            "high", error_msg)

    m = re.search(r"invalid input syntax for (?:type )?(\w+):\s*\"([^\"]+)\"", error_lower)
    if m:
        target_type, value = m.group(1), m.group(2)
        return ErrorDiagnosis("invalid_syntax", '"' + value + '"',
            "Invalid " + target_type + " literal: '" + value + "'. "
            "Use the correct format (e.g., 'YYYY-MM-DD' for dates).",
            "high", error_msg)

    m = re.search(r'column "([^"]+)" must appear in the group by clause', error_lower)
    if m:
        col = m.group(1)
        return ErrorDiagnosis("group_by_error", '"' + col + '"',
            "Add column '" + col + "' to GROUP BY or wrap it in an aggregate function.",
            "high", error_msg)

    m = re.search(r'column reference "([^"]+)" is ambiguous', error_lower)
    if m:
        col = m.group(1)
        return ErrorDiagnosis("ambiguous_column", '"' + col + '"',
            "Column '" + col + "' exists in multiple tables. Qualify with a table alias.",
            "high", error_msg)

    if "division by zero" in error_lower:
        return ErrorDiagnosis("division_by_zero", "division expression",
            "Use NULLIF(divisor, 0) to handle zero denominators, or add WHERE to exclude zeros.",
            "high", error_msg)

    m = re.search(r'relation "([^"]+)" does not exist', error_lower)
    if m:
        table = m.group(1)
        return ErrorDiagnosis("missing_column", '"' + table + '"',
            "Table or view '" + table + "' does not exist. Check the table name, schema prefix, and spelling.",
            "medium", error_msg)

    m = re.search(r'syntax error at or near "([^"]+)"', error_lower)
    if m:
        token = m.group(1)
        return ErrorDiagnosis("invalid_syntax", token,
            "SQL syntax error near '" + token + "'. "
            "Check for missing keywords, extra commas, or unbalanced parentheses.",
            "medium", error_msg)

    m = re.search(r'function "([^"]+)".*does not exist', error_lower)
    if m:
        func = m.group(1)
        return ErrorDiagnosis("invalid_syntax", '"' + func + '"()',
            "Function '" + func + "' does not exist. Check the function name and argument types.",
            "medium", error_msg)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MySQL
# ─────────────────────────────────────────────────────────────────────────────

def _classify_mysql(error_msg, error_lower, sql):
    m = re.search(r"unknown column '([^']+)' in '([^']+)'", error_lower)
    if m:
        col, location = m.group(1), m.group(2)
        return ErrorDiagnosis("missing_column", col,
            "Column '" + col + "' not found in " + location + ". Check column name spelling and table alias.",
            "high", error_msg)

    m = re.search(r"incorrect (\w+) value: '([^']+)' for column '([^']+)'", error_lower)
    if m:
        val_type, value, col = m.group(1), m.group(2), m.group(3)
        return ErrorDiagnosis("type_mismatch", col + " = '" + value + "'",
            "Invalid " + val_type + " value '" + value + "' for column '" + col + "'. Cast or format correctly.",
            "high", error_msg)

    m = re.search(r"expression #(\d+) of select list is not in group by clause", error_lower)
    if m:
        pos = m.group(1)
        return ErrorDiagnosis("group_by_error", "SELECT column #" + pos,
            "Column at SELECT position " + pos + " must be in GROUP BY or wrapped in an aggregate function.",
            "high", error_msg)

    if "division by zero" in error_lower:
        return ErrorDiagnosis("division_by_zero", "division expression",
            "Use NULLIF(divisor, 0) to handle zero denominators.", "high", error_msg)

    m = re.search(r"column '([^']+)' in (?:field list|order clause|group statement|having clause) is ambiguous", error_lower)
    if m:
        col = m.group(1)
        return ErrorDiagnosis("ambiguous_column", col,
            "Column '" + col + "' is ambiguous. Add a table alias prefix.", "high", error_msg)

    m = re.search(r"truncated incorrect (\w+) value: '([^']+)'", error_lower)
    if m:
        val_type, value = m.group(1), m.group(2)
        return ErrorDiagnosis("type_mismatch", "'" + value + "'",
            "Value '" + value + "' cannot be converted to " + val_type + ". Check data type compatibility.",
            "medium", error_msg)

    m = re.search(r"table '([^']+)' doesn'?t exist", error_lower)
    if m:
        table = m.group(1)
        return ErrorDiagnosis("missing_column", table,
            "Table '" + table + "' does not exist. Check the table name and database.",
            "high", error_msg)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SQL Server / MSSQL
# ─────────────────────────────────────────────────────────────────────────────

def _classify_mssql(error_msg, error_lower, sql):
    m = re.search(r"invalid column name '([^']+)'", error_lower)
    if m:
        col = m.group(1)
        return ErrorDiagnosis("missing_column", "[" + col + "]",
            "Column '" + col + "' does not exist. Check column name spelling and table.",
            "high", error_msg)

    m = re.search(r"conversion failed when converting (.*?) to data type (\w+)", error_lower)
    if m:
        from_val, to_type = m.group(1).strip(), m.group(2)
        return ErrorDiagnosis("type_mismatch", from_val,
            "Type conversion failed: cannot convert to '" + to_type + "'. "
            "Use CAST(expr AS " + to_type.upper() + ") or TRY_CAST().",
            "high", error_msg)

    m = re.search(
        r"column '([^']+)' is invalid in the select list because it is not contained in "
        r"either an aggregate function or the group by clause", error_lower)
    if m:
        col = m.group(1)
        return ErrorDiagnosis("group_by_error", "[" + col + "]",
            "Column '" + col + "' must be added to GROUP BY or wrapped in an aggregate function.",
            "high", error_msg)

    if "divide by zero error encountered" in error_lower or "division by zero" in error_lower:
        return ErrorDiagnosis("division_by_zero", "division expression",
            "Use NULLIF(divisor, 0) to handle zero denominators.", "high", error_msg)

    m = re.search(r"ambiguous column name '([^']+)'", error_lower)
    if m:
        col = m.group(1)
        return ErrorDiagnosis("ambiguous_column", "[" + col + "]",
            "Column '" + col + "' is ambiguous. Add a table alias prefix.", "high", error_msg)

    m = re.search(r"incorrect syntax near '([^']+)'", error_lower)
    if m:
        token = m.group(1)
        return ErrorDiagnosis("invalid_syntax", token,
            "Syntax error near '" + token + "'. Check for missing keywords or misplaced operators.",
            "medium", error_msg)

    m = re.search(r"invalid object name '([^']+)'", error_lower)
    if m:
        obj = m.group(1)
        return ErrorDiagnosis("missing_column", "[" + obj + "]",
            "Table or view '" + obj + "' does not exist. Check the object name and schema prefix.",
            "high", error_msg)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Oracle
# ─────────────────────────────────────────────────────────────────────────────

def _classify_oracle(error_msg, error_lower, sql):
    m = re.search(r'ora-00904:\s*"?([^":]+)"?:\s*invalid identifier', error_lower)
    if not m:
        m = re.search(r'ora-00904', error_lower)
    if m:
        col = _extract_token(error_msg) or (m.group(1).strip() if m.lastindex else "identifier")
        return ErrorDiagnosis("missing_column", '"' + col + '"',
            "Invalid identifier '" + col + "'. Column does not exist or the name is misspelled.",
            "high", error_msg)

    if "ora-00979" in error_lower or "not a group by expression" in error_lower:
        col = _extract_token(error_msg)
        return ErrorDiagnosis("group_by_error", col or "SELECT expression",
            "Non-aggregated SELECT column not in GROUP BY. Add it or use an aggregate function.",
            "high", error_msg)

    if "ora-01722" in error_lower or "invalid number" in error_lower:
        return ErrorDiagnosis("type_mismatch", "numeric expression",
            "Invalid number conversion. Use TO_NUMBER() or cast explicitly.", "high", error_msg)

    if "ora-01476" in error_lower or "divisor is equal to zero" in error_lower:
        return ErrorDiagnosis("division_by_zero", "division expression",
            "Use NULLIF(divisor, 0) or DECODE(divisor, 0, NULL, divisor) to handle zero denominators.",
            "high", error_msg)

    if "ora-00918" in error_lower or "column ambiguously defined" in error_lower:
        col = _extract_token(error_msg)
        return ErrorDiagnosis("ambiguous_column", col or "column",
            "Ambiguous column reference. Qualify with the table alias.", "high", error_msg)

    if "ora-00942" in error_lower or "table or view does not exist" in error_lower:
        return ErrorDiagnosis("missing_column", _extract_token(error_msg) or "table/view",
            "Table or view does not exist. Check the name, schema prefix, and access permissions.",
            "high", error_msg)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SQLite
# ─────────────────────────────────────────────────────────────────────────────

def _classify_sqlite(error_msg, error_lower, sql):
    m = re.search(r"no such column:\s*([A-Za-z_\"'\[\]`][A-Za-z0-9_\"'\[\]`\.]*)", error_lower)
    if m:
        col = m.group(1).strip("\"'`[]")
        return ErrorDiagnosis("missing_column", col,
            "Column '" + col + "' does not exist. Check column name spelling.", "high", error_msg)

    m = re.search(r"no such table:\s*([A-Za-z_\"'\[\]`][A-Za-z0-9_\"'\[\]`\.]*)", error_lower)
    if m:
        table = m.group(1)
        return ErrorDiagnosis("missing_column", table,
            "Table '" + table + "' does not exist. Check the table name.", "high", error_msg)

    if "misuse of aggregate" in error_lower:
        return ErrorDiagnosis("group_by_error", "aggregate expression",
            "Aggregate function misused. Check GROUP BY clause.", "medium", error_msg)

    m = re.search(r"ambiguous column name:\s*(\S+)", error_lower)
    if m:
        col = m.group(1)
        return ErrorDiagnosis("ambiguous_column", col,
            "Column '" + col + "' is ambiguous. Add a table alias prefix.", "high", error_msg)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Generic (cross-dialect fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_generic(error_msg, error_lower, sql):
    token = _extract_token(error_msg)

    if re.search(r"divis\w+\s+by\s+zero", error_lower):
        return ErrorDiagnosis("division_by_zero", "division expression",
            "Division by zero. Use NULLIF(divisor, 0) or add a WHERE condition to exclude zeros.",
            "high", error_msg)

    if re.search(r"(type mismatch|cannot convert|conversion failed|invalid.*type|type.*invalid)", error_lower):
        return ErrorDiagnosis("type_mismatch", token or "expression",
            "Data type mismatch. Cast the column to the appropriate type.", "medium", error_msg)

    if re.search(r"(column.*not found|no such column|invalid column|unknown column|does not exist)", error_lower):
        return ErrorDiagnosis("missing_column", token or "column/table",
            ("Object '" + token + "' not found. Check name spelling and schema prefix.")
            if token else "Column or table not found. Check spelling and schema prefix.",
            "medium", error_msg)

    if re.search(r"(group by|aggregate|not in group)", error_lower):
        return ErrorDiagnosis("group_by_error", token or "expression",
            "GROUP BY error. Add the missing column to GROUP BY or wrap it in an aggregate function.",
            "medium", error_msg)

    if re.search(r"ambig\w+", error_lower):
        return ErrorDiagnosis("ambiguous_column", token or "column",
            "Ambiguous column name. Qualify with a table alias.", "medium", error_msg)

    if re.search(r"syntax error", error_lower):
        return ErrorDiagnosis("invalid_syntax", token or "SQL",
            "SQL syntax error. Check for missing keywords or malformed expressions.", "low", error_msg)

    return None
