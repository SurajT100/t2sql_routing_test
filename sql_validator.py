"""
SQL Validator
==============
Pre-execution validation to catch errors before running SQL.
Saves tokens by catching obvious mistakes without LLM calls.

Usage:
    from sql_validator import validate_sql, SQLValidationResult
    
    result = validate_sql(sql, "postgresql", available_tables, available_columns)
    if not result.is_valid:
        print(result.issues)
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SQLValidationResult:
    """Result of SQL validation."""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    severity: str = "ok"  # "ok", "warning", "error", "critical"


# =============================================================================
# DIALECT-SPECIFIC RULES
# =============================================================================

DIALECT_RULES = {
    "postgresql": {
        "identifier_quote": '"',
        "string_quote": "'",
        "wrong_quotes": ['`', '['],
        "case_sensitive": True,
        "limit_syntax": "LIMIT n",
    },
    "mysql": {
        "identifier_quote": '`',
        "string_quote": "'",
        "wrong_quotes": ['['],
        "case_sensitive": False,
        "limit_syntax": "LIMIT n",
    },
    "mssql": {
        "identifier_quote": '[',
        "identifier_quote_end": ']',
        "string_quote": "'",
        "wrong_quotes": ['`'],
        "case_sensitive": False,
        "limit_syntax": "TOP n",
    },
    "oracle": {
        "identifier_quote": '"',
        "string_quote": "'",
        "wrong_quotes": ['`', '['],
        "case_sensitive": True,
        "limit_syntax": "ROWNUM",
    },
    "sqlite": {
        "identifier_quote": '"',
        "string_quote": "'",
        "wrong_quotes": ['['],
        "case_sensitive": False,
        "limit_syntax": "LIMIT n",
    },
}

# Common SQL typos
COMMON_TYPOS = {
    "SLECT": "SELECT",
    "FORM": "FROM",
    "FOMR": "FROM",
    "WHER": "WHERE",
    "WEHRE": "WHERE",
    "GRUOP": "GROUP",
    "GROPU": "GROUP",
    "ORDE": "ORDER",
    "ORDR": "ORDER",
    "ORDERY": "ORDER",
    "DISTICT": "DISTINCT",
    "DINSTINCT": "DISTINCT",
    "SEELCT": "SELECT",
    "SELCT": "SELECT",
    "JOINN": "JOIN",
    "JION": "JOIN",
    "INNNER": "INNER",
    "LEFFT": "LEFT",
    "RIGTH": "RIGHT",
    "HAVNG": "HAVING",
    "HAVIG": "HAVING",
}

# Dangerous SQL patterns
DANGEROUS_PATTERNS = [
    (r'\bDROP\s+', "DROP statement detected - will delete data"),
    (r'\bDELETE\s+', "DELETE statement detected - will remove data"),
    (r'\bTRUNCATE\s+', "TRUNCATE statement detected - will remove all data"),
    (r'\bUPDATE\s+.*\bSET\b', "UPDATE statement detected - will modify data"),
    (r'\bINSERT\s+INTO\b', "INSERT statement detected - will add data"),
    (r'\bALTER\s+', "ALTER statement detected - will modify schema"),
    (r'\bCREATE\s+', "CREATE statement detected - will create objects"),
    (r'\bGRANT\s+', "GRANT statement detected - security operation"),
    (r'\bREVOKE\s+', "REVOKE statement detected - security operation"),
    (r'\bEXEC\s*\(', "EXEC detected - potential SQL injection"),
    (r';\s*--', "Comment after semicolon - potential injection"),
    (r'\bxp_', "Extended stored procedure detected"),
]


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_sql(
    sql: str,
    dialect: str,
    available_tables: List[str] = None,
    available_columns: List[str] = None,
    strict: bool = False
) -> SQLValidationResult:
    """
    Validate SQL before execution.
    
    Args:
        sql: SQL query string
        dialect: Database dialect (postgresql, mysql, mssql, oracle, sqlite)
        available_tables: List of valid table names
        available_columns: List of valid column names
        strict: If True, warnings become errors
    
    Returns:
        SQLValidationResult with validation details
    """
    result = SQLValidationResult(is_valid=True)
    sql_upper = sql.upper()
    
    # Get dialect rules
    rules = DIALECT_RULES.get(dialect.lower(), DIALECT_RULES["postgresql"])
    
    # 1. Check for dangerous patterns
    _check_dangerous_patterns(sql, sql_upper, result)
    
    # 2. Check dialect-specific quoting
    _check_quoting(sql, dialect, rules, result)
    
    # 3. Check for common SQL errors
    _check_common_errors(sql, sql_upper, result)
    
    # 4. Check for typos
    _check_typos(sql_upper, result)
    
    # 5. Check balanced quotes and parentheses
    _check_balance(sql, result)
    
    # 6. Check table references
    if available_tables:
        _check_tables(sql, available_tables, result)
    
    # 7. Basic syntax structure
    _check_basic_structure(sql_upper, result)
    
    # Determine overall validity and severity
    if result.issues:
        result.is_valid = False
        result.severity = "error"
    elif result.warnings:
        result.severity = "warning"
        if strict:
            result.is_valid = False
            result.issues = result.warnings
            result.severity = "error"
    
    return result


# =============================================================================
# VALIDATION CHECKS
# =============================================================================

def _check_dangerous_patterns(sql: str, sql_upper: str, result: SQLValidationResult):
    """Check for dangerous SQL patterns."""
    for pattern, message in DANGEROUS_PATTERNS:
        if re.search(pattern, sql_upper):
            result.issues.append(f"DANGEROUS: {message}")
            result.severity = "critical"


def _check_quoting(sql: str, dialect: str, rules: Dict, result: SQLValidationResult):
    """Check dialect-specific quoting."""
    wrong_quotes = rules.get("wrong_quotes", [])
    correct_quote = rules.get("identifier_quote", '"')
    
    for wrong in wrong_quotes:
        if wrong in sql:
            result.issues.append(
                f"Wrong quote character '{wrong}' for {dialect}. Use '{correct_quote}' for identifiers."
            )
    
    # Check for double quotes used as string quotes (common mistake)
    string_quote = rules.get("string_quote", "'")
    
    # Pattern: = "value" or = "value" (double quotes around string literal)
    if string_quote == "'" and re.search(r'=\s*"[^"]+"\s*(?:AND|OR|$|;|\))', sql):
        result.warnings.append(
            f"Possible mistake: Use single quotes (') for string values in {dialect}, not double quotes."
        )


def _check_common_errors(sql: str, sql_upper: str, result: SQLValidationResult):
    """Check for common SQL errors."""
    
    # SELECT * with GROUP BY
    if "SELECT *" in sql_upper and "GROUP BY" in sql_upper:
        result.issues.append(
            "SELECT * with GROUP BY will fail. Specify columns explicitly."
        )
    
    # SELECT without FROM (except for constants)
    if sql_upper.strip().startswith("SELECT") and "FROM" not in sql_upper:
        # Allow SELECT without FROM for constants like SELECT 1, SELECT CURRENT_DATE
        if not re.search(r'SELECT\s+[\d\'\"]|SELECT\s+CURRENT|SELECT\s+NOW|SELECT\s+\@', sql_upper):
            result.warnings.append(
                "SELECT without FROM clause. Intended?"
            )
    
    # Missing GROUP BY with aggregates
    aggregates = ['SUM(', 'COUNT(', 'AVG(', 'MIN(', 'MAX(', 'GROUP_CONCAT(', 'STRING_AGG(']
    has_aggregate = any(agg in sql_upper for agg in aggregates)
    has_group_by = "GROUP BY" in sql_upper
    
    if has_aggregate and not has_group_by:
        # Check if it's a simple aggregate without other columns
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # If there are non-aggregate columns, GROUP BY is likely needed
            non_agg_cols = re.sub(r'(SUM|COUNT|AVG|MIN|MAX|GROUP_CONCAT|STRING_AGG)\s*\([^)]+\)', '', select_clause)
            non_agg_cols = re.sub(r'\s+AS\s+\w+', '', non_agg_cols)
            non_agg_cols = non_agg_cols.replace(',', '').strip()
            
            if non_agg_cols and non_agg_cols not in ['*', '']:
                result.warnings.append(
                    "Query has aggregate functions with non-aggregated columns but no GROUP BY. "
                    "This may fail or give unexpected results."
                )
    
    # ORDER BY with column not in SELECT (when DISTINCT is used)
    if "DISTINCT" in sql_upper and "ORDER BY" in sql_upper:
        # This is complex to validate fully, just warn
        result.warnings.append(
            "DISTINCT with ORDER BY - ensure ORDER BY columns are in SELECT list."
        )
    
    # Comparison with NULL using = instead of IS
    if re.search(r'=\s*NULL\b', sql_upper) or re.search(r'<>\s*NULL\b', sql_upper):
        result.issues.append(
            "Use 'IS NULL' or 'IS NOT NULL' instead of '= NULL' or '<> NULL'."
        )


def _check_typos(sql_upper: str, result: SQLValidationResult):
    """Check for common SQL typos."""
    for typo, correct in COMMON_TYPOS.items():
        # Match as whole word
        if re.search(rf'\b{typo}\b', sql_upper):
            result.issues.append(f"Possible typo: '{typo}' should be '{correct}'")


def _check_balance(sql: str, result: SQLValidationResult):
    """Check for balanced quotes and parentheses."""
    
    # Check parentheses
    open_parens = sql.count('(')
    close_parens = sql.count(')')
    if open_parens != close_parens:
        result.issues.append(
            f"Unbalanced parentheses: {open_parens} '(' and {close_parens} ')'"
        )
    
    # Check single quotes (simple check - doesn't handle escaped quotes perfectly)
    # Remove escaped quotes first
    sql_no_escaped = sql.replace("''", "").replace("\\'", "")
    if sql_no_escaped.count("'") % 2 != 0:
        result.issues.append("Unbalanced single quotes")
    
    # Check double quotes
    sql_no_escaped = sql.replace('""', "").replace('\\"', "")
    if sql_no_escaped.count('"') % 2 != 0:
        result.issues.append("Unbalanced double quotes")
    
    # Check square brackets (for SQL Server)
    if '[' in sql or ']' in sql:
        if sql.count('[') != sql.count(']'):
            result.issues.append("Unbalanced square brackets")


def _check_tables(sql: str, available_tables: List[str], result: SQLValidationResult):
    """Check if referenced tables exist."""
    # Extract table references from FROM and JOIN clauses
    # This is a simplified check - won't catch all cases
    
    # Normalize available tables for comparison
    normalized_tables = {t.lower().replace('"', '').replace('`', '').replace('[', '').replace(']', '') 
                        for t in available_tables}
    
    # Extract potential table names after FROM and JOIN
    patterns = [
        r'FROM\s+(["\[\`]?[\w\.]+["\]\`]?)',
        r'JOIN\s+(["\[\`]?[\w\.]+["\]\`]?)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, sql, re.IGNORECASE)
        for match in matches:
            # Clean the match
            clean_match = match.lower().replace('"', '').replace('`', '').replace('[', '').replace(']', '')
            
            # Check if table exists
            if clean_match not in normalized_tables:
                # Also check without schema prefix
                table_only = clean_match.split('.')[-1] if '.' in clean_match else clean_match
                schema_matches = [t for t in normalized_tables if t.endswith('.' + table_only) or t == table_only]
                
                if not schema_matches:
                    result.warnings.append(f"Table '{match}' not found in available tables")


def _check_basic_structure(sql_upper: str, result: SQLValidationResult):
    """Check basic SQL structure."""
    sql_stripped = sql_upper.strip()
    
    # Must start with a valid keyword
    valid_starts = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 
                    'ALTER', 'DROP', 'TRUNCATE', 'EXPLAIN', 'SHOW', 'DESCRIBE']
    
    if not any(sql_stripped.startswith(kw) for kw in valid_starts):
        result.issues.append(
            f"SQL doesn't start with a valid keyword. Starts with: '{sql_stripped[:20]}...'"
        )
    
    # Check for multiple statements (potential injection)
    # Count semicolons not inside quotes
    statements = sql_upper.split(';')
    non_empty = [s.strip() for s in statements if s.strip()]
    
    if len(non_empty) > 1:
        result.warnings.append(
            f"Multiple SQL statements detected ({len(non_empty)}). This may not be allowed."
        )


# =============================================================================
# QUICK VALIDATION FUNCTIONS
# =============================================================================

def is_select_query(sql: str) -> bool:
    """Check if SQL is a SELECT query (safe to execute)."""
    sql_upper = sql.strip().upper()
    
    # Allow WITH (CTE) followed by SELECT
    if sql_upper.startswith("WITH"):
        return "SELECT" in sql_upper and not any(
            kw in sql_upper for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        )
    
    return sql_upper.startswith("SELECT")


def is_safe_query(sql: str) -> Tuple[bool, str]:
    """
    Quick check if query is safe to execute.
    
    Returns:
        (is_safe, reason)
    """
    sql_upper = sql.upper()
    
    for pattern, message in DANGEROUS_PATTERNS:
        if re.search(pattern, sql_upper):
            return False, message
    
    if not is_select_query(sql):
        return False, "Not a SELECT query"
    
    return True, "Query appears safe"


def fix_common_issues(sql: str, dialect: str) -> Tuple[str, List[str]]:
    """
    Attempt to fix common SQL issues.
    
    Args:
        sql: SQL query string
        dialect: Database dialect
    
    Returns:
        (fixed_sql, list_of_fixes_applied)
    """
    fixes = []
    fixed_sql = sql
    
    # Fix NULL comparisons
    if re.search(r'=\s*NULL\b', fixed_sql, re.IGNORECASE):
        fixed_sql = re.sub(r'=\s*NULL\b', 'IS NULL', fixed_sql, flags=re.IGNORECASE)
        fixes.append("Changed '= NULL' to 'IS NULL'")
    
    if re.search(r'<>\s*NULL\b', fixed_sql, re.IGNORECASE):
        fixed_sql = re.sub(r'<>\s*NULL\b', 'IS NOT NULL', fixed_sql, flags=re.IGNORECASE)
        fixes.append("Changed '<> NULL' to 'IS NOT NULL'")
    
    # Fix common typos
    for typo, correct in COMMON_TYPOS.items():
        if re.search(rf'\b{typo}\b', fixed_sql, re.IGNORECASE):
            fixed_sql = re.sub(rf'\b{typo}\b', correct, fixed_sql, flags=re.IGNORECASE)
            fixes.append(f"Fixed typo: '{typo}' -> '{correct}'")
    
    # Fix quote issues for specific dialects
    rules = DIALECT_RULES.get(dialect.lower(), DIALECT_RULES["postgresql"])
    correct_quote = rules.get("identifier_quote", '"')
    
    if dialect.lower() == "postgresql" and '`' in fixed_sql:
        fixed_sql = fixed_sql.replace('`', '"')
        fixes.append("Changed backticks to double quotes for PostgreSQL")
    
    if dialect.lower() == "mysql" and fixed_sql.count('"') > fixed_sql.count('`'):
        # This is trickier - only change if clearly identifier quotes
        # For now, just warn
        pass
    
    return fixed_sql, fixes


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SQL VALIDATOR TEST")
    print("=" * 70)
    
    test_cases = [
        # Valid queries
        ('SELECT * FROM users WHERE id = 1', "postgresql", True),
        ('SELECT "Region", SUM("Margin") FROM "public"."SAP" GROUP BY "Region"', "postgresql", True),
        
        # Dialect issues
        ('SELECT `Region` FROM `SAP`', "postgresql", False),  # Wrong quotes
        ('SELECT [Region] FROM [SAP]', "mysql", False),  # Wrong quotes
        
        # Common errors
        ('SELECT * FROM users GROUP BY name', "postgresql", False),  # SELECT * with GROUP BY
        ("SELECT * FROM users WHERE status = NULL", "postgresql", False),  # NULL comparison
        
        # Typos
        ('SLECT * FROM users', "postgresql", False),
        ('SELECT * FORM users', "postgresql", False),
        
        # Balance issues
        ('SELECT * FROM users WHERE (id = 1', "postgresql", False),
        ("SELECT * FROM users WHERE name = 'test", "postgresql", False),
        
        # Dangerous
        ('DROP TABLE users', "postgresql", False),
        ('SELECT * FROM users; DELETE FROM users', "postgresql", False),
    ]
    
    print("\nValidation Results:")
    print("-" * 70)
    
    for sql, dialect, expected_valid in test_cases:
        result = validate_sql(sql, dialect)
        status = "✓" if result.is_valid == expected_valid else "✗"
        valid_str = "VALID" if result.is_valid else "INVALID"
        
        print(f"{status} [{valid_str:7}] {sql[:50]}...")
        if result.issues:
            for issue in result.issues[:2]:
                print(f"          Issue: {issue[:60]}")
        if result.warnings:
            for warn in result.warnings[:1]:
                print(f"          Warning: {warn[:60]}")
    
    print("\n" + "=" * 70)
    print("AUTO-FIX TEST")
    print("=" * 70)
    
    fix_tests = [
        "SELECT * FROM users WHERE status = NULL",
        "SLECT * FROM users",
        "SELECT `name` FROM `users`",  # PostgreSQL
    ]
    
    for sql in fix_tests:
        fixed, fixes = fix_common_issues(sql, "postgresql")
        print(f"\nOriginal: {sql}")
        print(f"Fixed:    {fixed}")
        print(f"Fixes:    {fixes}")
    
    print("\n" + "=" * 70)
