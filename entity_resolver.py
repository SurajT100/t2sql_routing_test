"""
Entity Resolver — Live Database Entity Resolution
====================================================
Resolves fuzzy entity references against ACTUAL database values at query time.

This is the critical gap between profiling (static samples) and query execution.
When a CxO types "sales for Dell", this module runs a quick SELECT DISTINCT
against the live database to find what "Dell" actually looks like in the data.

Architecture position:
    Pass 1 (column ID) → Metadata Fetch → **Entity Resolver** → Pass 2 (plan)

The resolver:
1. Takes string_filter_columns from Pass 1 (columns where user typed a value)
2. For each, runs a lightweight query against the user's LIVE database
3. Returns resolved matches with recommended filter strategies
4. Pass 2 receives deterministic values instead of guessing from samples

Usage:
    from entity_resolver import resolve_entities, format_resolutions_for_prompt

    resolutions = resolve_entities(
        user_engine=engine,
        string_filter_columns=pass1_data["string_filter_columns"],
        metadata=column_metadata,
        dialect_info=config.dialect_info
    )
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EntityResolution:
    """Result of resolving a single entity against the live database."""
    table: str
    column: str
    user_value: str                    # What the user typed (e.g., "Dell")
    filter_type: str = "include"       # "include" or "exclude"
    
    # Resolution results
    exact_match: Optional[str] = None  # Exact match found (e.g., "Dell")
    partial_matches: List[str] = field(default_factory=list)  # ILIKE matches
    match_count: int = 0
    
    # Recommended strategy
    strategy: str = "exact"            # "exact", "ilike", "in_list", "no_match"
    filter_condition: str = ""         # Ready-to-use condition string
    confidence: str = "high"           # "high", "medium", "low"
    
    # Diagnostics
    query_time_ms: int = 0
    query_used: str = ""               # The actual SELECT DISTINCT query run
    warning: str = ""                  # Any warning for Pass 2


@dataclass
class ResolverResult:
    """Complete result of entity resolution for all filter columns."""
    resolutions: List[EntityResolution] = field(default_factory=list)
    total_time_ms: int = 0
    queries_run: int = 0
    all_resolved: bool = True          # False if any entity had no_match


# =============================================================================
# CORE RESOLVER
# =============================================================================

def resolve_entities(
    user_engine,
    string_filter_columns: List[Dict],
    metadata: Dict[str, Any],
    dialect_info: Dict = None,
    max_results: int = 20,
    timeout_ms: int = 3000
) -> ResolverResult:
    """
    Resolve entity references against the LIVE database.
    
    For each string_filter_column from Pass 1, runs a quick SELECT DISTINCT
    to find actual matching values.
    
    Args:
        user_engine: SQLAlchemy engine connected to user's database
        string_filter_columns: From Pass 1 output — columns where user typed values
            [{"table": "SAP", "column": "Company", "user_value": "Dell", "filter_type": "include"}]
        metadata: From fetch_column_metadata — used for data_type awareness
        dialect_info: Database dialect configuration
        max_results: Max distinct values to return per column
        timeout_ms: Max time per query in milliseconds
    
    Returns:
        ResolverResult with all resolutions
    """
    from sqlalchemy import text
    
    if not string_filter_columns:
        return ResolverResult()
    
    dialect = (dialect_info or {}).get("dialect", "postgresql")
    quote_char = (dialect_info or {}).get("quote_char", '"')
    
    result = ResolverResult()
    start_time = time.time()
    
    for sf in string_filter_columns:
        table_name = sf.get("table", "")
        column_name = sf.get("column", "")
        user_value = sf.get("user_value", "").strip()
        filter_type = sf.get("filter_type", "include")
        
        if not table_name or not column_name or not user_value:
            continue
        
        resolution = EntityResolution(
            table=table_name,
            column=column_name,
            user_value=user_value,
            filter_type=filter_type
        )
        
        query_start = time.time()
        
        try:
            # Get column data type from metadata
            col_meta = metadata.get(table_name, {}).get(column_name, {})
            data_type = col_meta.get("data_type", "text").lower()
            
            # Only resolve string/text columns — skip numeric/date
            if any(t in data_type for t in ["int", "float", "decimal", "numeric", "double", "real"]):
                resolution.strategy = "exact"
                resolution.filter_condition = f"= {user_value}"
                resolution.confidence = "high"
                resolution.warning = "Numeric column — exact value used"
                result.resolutions.append(resolution)
                continue
            
            # ── Step 1: Try exact match first (cheapest) ──
            exact_query = _build_exact_query(
                table_name, column_name, user_value, dialect, quote_char
            )
            resolution.query_used = exact_query
            
            exact_matches = _run_resolve_query(user_engine, exact_query)
            result.queries_run += 1
            
            if exact_matches:
                # Perfect — exact match exists in data
                resolution.exact_match = exact_matches[0]
                resolution.match_count = 1
                resolution.strategy = "exact"
                resolution.filter_condition = f"= '{_escape_sql(exact_matches[0])}'"
                resolution.confidence = "high"
                resolution.query_time_ms = int((time.time() - query_start) * 1000)
                result.resolutions.append(resolution)
                print(f"[RESOLVER] ✅ Exact match: {user_value} → {exact_matches[0]}")
                continue
            
            # ── Step 2: Try case-insensitive exact match ──
            ci_query = _build_case_insensitive_query(
                table_name, column_name, user_value, dialect, quote_char
            )
            
            ci_matches = _run_resolve_query(user_engine, ci_query)
            result.queries_run += 1
            
            if ci_matches and len(ci_matches) == 1:
                # Single case-insensitive match — very confident
                resolution.exact_match = ci_matches[0]
                resolution.match_count = 1
                resolution.strategy = "exact"
                resolution.filter_condition = f"= '{_escape_sql(ci_matches[0])}'"
                resolution.confidence = "high"
                resolution.query_time_ms = int((time.time() - query_start) * 1000)
                resolution.query_used = ci_query
                result.resolutions.append(resolution)
                print(f"[RESOLVER] ✅ CI exact match: {user_value} → {ci_matches[0]}")
                continue
            elif ci_matches and len(ci_matches) > 1:
                # Multiple case-insensitive exact matches (rare but possible)
                resolution.partial_matches = ci_matches
                resolution.match_count = len(ci_matches)
                resolution.strategy = "in_list"
                escaped = [f"'{_escape_sql(v)}'" for v in ci_matches]
                resolution.filter_condition = f"IN ({', '.join(escaped)})"
                resolution.confidence = "high"
                resolution.query_time_ms = int((time.time() - query_start) * 1000)
                resolution.query_used = ci_query
                result.resolutions.append(resolution)
                print(f"[RESOLVER] ✅ CI multi match: {user_value} → {ci_matches}")
                continue
            
            # ── Step 3: Try ILIKE / partial match ──
            partial_query = _build_partial_query(
                table_name, column_name, user_value, dialect, quote_char,
                max_results=max_results
            )
            
            partial_matches = _run_resolve_query(user_engine, partial_query)
            result.queries_run += 1
            
            if partial_matches:
                resolution.partial_matches = partial_matches
                resolution.match_count = len(partial_matches)
                
                if len(partial_matches) == 1:
                    # Single partial match — use exact value
                    resolution.strategy = "exact"
                    resolution.filter_condition = f"= '{_escape_sql(partial_matches[0])}'"
                    resolution.confidence = "high"
                    print(f"[RESOLVER] ✅ Single partial: {user_value} → {partial_matches[0]}")
                    
                elif len(partial_matches) <= 5:
                    # Small set — use IN list
                    resolution.strategy = "in_list"
                    escaped = [f"'{_escape_sql(v)}'" for v in partial_matches]
                    resolution.filter_condition = f"IN ({', '.join(escaped)})"
                    resolution.confidence = "high"
                    print(f"[RESOLVER] ✅ IN list: {user_value} → {partial_matches}")
                    
                else:
                    # Many matches — use ILIKE for broader matching
                    resolution.strategy = "ilike"
                    resolution.filter_condition = _build_ilike_condition(
                        user_value, dialect
                    )
                    resolution.confidence = "medium"
                    resolution.warning = (
                        f"Found {len(partial_matches)} partial matches. "
                        f"Using ILIKE. First 5: {partial_matches[:5]}"
                    )
                    print(f"[RESOLVER] ⚠️ ILIKE ({len(partial_matches)} matches): {user_value}")
            else:
                # ── Step 4: No matches at all ──
                resolution.strategy = "no_match"
                resolution.match_count = 0
                resolution.confidence = "low"
                resolution.filter_condition = _build_ilike_condition(
                    user_value, dialect
                )
                resolution.warning = (
                    f"No matches found for '{user_value}' in {table_name}.{column_name}. "
                    f"Using ILIKE as fallback — results may be empty."
                )
                result.all_resolved = False
                print(f"[RESOLVER] ❌ No match: {user_value} in {table_name}.{column_name}")
            
            resolution.query_time_ms = int((time.time() - query_start) * 1000)
            resolution.query_used = partial_query
            result.resolutions.append(resolution)
            
        except Exception as e:
            print(f"[RESOLVER] Error resolving {user_value} in {table_name}.{column_name}: {e}")
            resolution.strategy = "ilike"
            resolution.filter_condition = _build_ilike_condition(user_value, dialect)
            resolution.confidence = "low"
            resolution.warning = f"Resolution failed: {str(e)[:100]}. Using ILIKE fallback."
            resolution.query_time_ms = int((time.time() - query_start) * 1000)
            result.resolutions.append(resolution)
            result.all_resolved = False
    
    result.total_time_ms = int((time.time() - start_time) * 1000)
    return result


# =============================================================================
# QUERY BUILDERS — Dialect-aware
# =============================================================================

def _build_exact_query(
    table: str, column: str, value: str,
    dialect: str, quote_char: str
) -> str:
    """Build exact match query. Checks for TRIM'd exact match."""
    q = quote_char
    qr = ']' if quote_char == '[' else quote_char
    
    # Handle schema.table format
    if "." in table:
        schema, tbl = table.split(".", 1)
        table_ref = f"{q}{schema}{qr}.{q}{tbl}{qr}"
    else:
        table_ref = f"{q}{table}{qr}"
    
    # MSSQL uses TOP instead of LIMIT
    if dialect == "mssql":
        return (
            f"SELECT DISTINCT TOP 5 RTRIM(LTRIM({q}{column}{qr})) AS val "
            f"FROM {table_ref} "
            f"WHERE RTRIM(LTRIM({q}{column}{qr})) = '{_escape_sql(value)}'"
        )
    
    return (
        f"SELECT DISTINCT TRIM({q}{column}{qr}) AS val "
        f"FROM {table_ref} "
        f"WHERE TRIM({q}{column}{qr}) = '{_escape_sql(value)}' "
        f"LIMIT 5"
    )


def _build_case_insensitive_query(
    table: str, column: str, value: str,
    dialect: str, quote_char: str
) -> str:
    """Build case-insensitive exact match query."""
    q = quote_char
    qr = ']' if quote_char == '[' else quote_char
    
    if "." in table:
        schema, tbl = table.split(".", 1)
        table_ref = f"{q}{schema}{qr}.{q}{tbl}{qr}"
    else:
        table_ref = f"{q}{table}{qr}"
    
    value_lower = value.lower()
    
    if dialect == "mssql":
        return (
            f"SELECT DISTINCT TOP 5 RTRIM(LTRIM({q}{column}{qr})) AS val "
            f"FROM {table_ref} "
            f"WHERE LOWER(RTRIM(LTRIM({q}{column}{qr}))) = '{_escape_sql(value_lower)}'"
        )
    
    return (
        f"SELECT DISTINCT TRIM({q}{column}{qr}) AS val "
        f"FROM {table_ref} "
        f"WHERE LOWER(TRIM({q}{column}{qr})) = '{_escape_sql(value_lower)}' "
        f"LIMIT 5"
    )


def _build_partial_query(
    table: str, column: str, value: str,
    dialect: str, quote_char: str,
    max_results: int = 20
) -> str:
    """Build partial/ILIKE match query."""
    q = quote_char
    qr = ']' if quote_char == '[' else quote_char
    
    if "." in table:
        schema, tbl = table.split(".", 1)
        table_ref = f"{q}{schema}{qr}.{q}{tbl}{qr}"
    else:
        table_ref = f"{q}{table}{qr}"
    
    value_lower = value.lower()
    
    if dialect == "mssql":
        return (
            f"SELECT DISTINCT TOP {max_results} RTRIM(LTRIM({q}{column}{qr})) AS val "
            f"FROM {table_ref} "
            f"WHERE LOWER(RTRIM(LTRIM({q}{column}{qr}))) LIKE '%{_escape_sql(value_lower)}%'"
        )
    elif dialect == "postgresql":
        # PostgreSQL supports ILIKE natively
        return (
            f"SELECT DISTINCT TRIM({q}{column}{qr}) AS val "
            f"FROM {table_ref} "
            f"WHERE TRIM({q}{column}{qr}) ILIKE '%{_escape_sql(value)}%' "
            f"LIMIT {max_results}"
        )
    else:
        # MySQL, Oracle, SQLite — use LOWER() + LIKE
        return (
            f"SELECT DISTINCT TRIM({q}{column}{qr}) AS val "
            f"FROM {table_ref} "
            f"WHERE LOWER(TRIM({q}{column}{qr})) LIKE '%{_escape_sql(value_lower)}%' "
            f"LIMIT {max_results}"
        )


def _build_ilike_condition(value: str, dialect: str) -> str:
    """Build the ILIKE/LIKE condition string for use in the final SQL."""
    if dialect == "postgresql":
        return f"ILIKE '%{_escape_sql(value)}%'"
    else:
        return f"LIKE '%{_escape_sql(value.lower())}%'"


# =============================================================================
# QUERY EXECUTION
# =============================================================================

def _run_resolve_query(engine, query: str) -> List[str]:
    """
    Run a resolver query and return list of distinct values.
    Lightweight — uses raw connection for speed.
    """
    from sqlalchemy import text
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            # Return non-null, non-empty values
            return [
                str(row[0]).strip() for row in rows
                if row[0] is not None and str(row[0]).strip()
            ]
    except Exception as e:
        print(f"[RESOLVER] Query failed: {str(e)[:100]}")
        print(f"[RESOLVER] Query was: {query[:200]}")
        return []


# =============================================================================
# FORMATTING FOR PASS 2
# =============================================================================

def format_resolutions_for_prompt(resolver_result: ResolverResult) -> str:
    """
    Format resolver results into a compact section for Pass 2 prompt.
    This replaces guesswork with facts.
    
    Args:
        resolver_result: Output from resolve_entities()
    
    Returns:
        Formatted string injected into Pass 2 prompt
    """
    if not resolver_result.resolutions:
        return "No entity resolution needed."
    
    lines = ["ENTITY RESOLUTIONS (from live database — use these EXACT values and conditions):"]
    lines.append(f"  Queries run: {resolver_result.queries_run} | "
                 f"Time: {resolver_result.total_time_ms}ms | "
                 f"All resolved: {'Yes' if resolver_result.all_resolved else 'No ⚠️'}")
    lines.append("")
    
    for res in resolver_result.resolutions:
        action = "EXCLUDE" if res.filter_type == "exclude" else "FILTER"
        lines.append(f"  {action}: {res.table}.{res.column}")
        lines.append(f"    User typed: \"{res.user_value}\"")
        lines.append(f"    Strategy: {res.strategy.upper()} (confidence: {res.confidence})")
        lines.append(f"    → Use condition: {res.filter_condition}")
        
        if res.exact_match:
            lines.append(f"    Exact value in DB: \"{res.exact_match}\"")
        elif res.partial_matches:
            shown = res.partial_matches[:5]
            lines.append(f"    Matches found ({res.match_count}): {shown}")
        
        if res.warning:
            lines.append(f"    ⚠️ {res.warning}")
        
        lines.append("")
    
    lines.append("CRITICAL: Use the resolved conditions above AS-IS. Do NOT change them.")
    
    return "\n".join(lines)


def merge_resolutions_into_metadata(
    metadata: Dict[str, Any],
    resolver_result: ResolverResult
) -> Dict[str, Any]:
    """
    Merge resolver results back into the metadata dict.
    This enriches metadata with live resolution data so Pass 2 
    sees it alongside sample values and descriptions.
    
    Args:
        metadata: Original metadata from fetch_column_metadata
        resolver_result: Output from resolve_entities
    
    Returns:
        Enriched metadata dict (mutated in place and returned)
    """
    for res in resolver_result.resolutions:
        table = res.table
        column = res.column
        
        if table in metadata and column in metadata[table]:
            col_meta = metadata[table][column]
            col_meta["resolved"] = True
            col_meta["resolver_strategy"] = res.strategy
            col_meta["resolver_condition"] = res.filter_condition
            col_meta["resolver_confidence"] = res.confidence
            col_meta["resolver_filter_type"] = res.filter_type
            
            if res.exact_match:
                col_meta["resolved_value"] = res.exact_match
            elif res.partial_matches:
                col_meta["resolved_matches"] = res.partial_matches[:10]
            
            if res.warning:
                col_meta["resolver_warning"] = res.warning
            
            # Override the partial match flag — resolver has definitive answer
            col_meta["needs_partial_match"] = res.strategy in ("ilike", "no_match")
    
    return metadata


# =============================================================================
# UTILITIES
# =============================================================================

def _escape_sql(value: str) -> str:
    """Escape single quotes in SQL values."""
    return value.replace("'", "''")


# =============================================================================
# MAIN — Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENTITY RESOLVER MODULE")
    print("=" * 70)
    print()
    print("Integration point: Between fetch_column_metadata and Pass 2")
    print()
    print("Resolution flow per entity:")
    print("  Step 1: Exact match (TRIM + =)")
    print("  Step 2: Case-insensitive exact (LOWER + TRIM + =)")
    print("  Step 3: Partial match (ILIKE / LOWER+LIKE)")
    print("  Step 4: No match → ILIKE fallback + warning")
    print()
    print("Strategies: exact | in_list | ilike | no_match")
    print("=" * 70)
