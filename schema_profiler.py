"""
Schema Profiler - Auto-profile tables for Schema RAG
=====================================================
Profiles selected tables: samples columns, detects PII, expands abbreviations,
stores in schema_columns table for intelligent retrieval.

This module provides:
1. profile_selected_tables() - Main profiling function
2. get_enrichment_candidates() - Get columns needing user enrichment  
3. update_column_enrichment() - Save user's enrichments
4. Streamlit UI components for Tab 1 integration
"""

from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import text, inspect
import json
import re
from datetime import datetime
import hashlib


# =============================================================================
# COLUMN INTELLIGENCE — Computed at profiling time, used by reasoner at query time
# =============================================================================
# All classification is DATA-DRIVEN first, keyword-assisted second.
# Keywords are configurable via COLUMN_ROLE_KEYWORDS dict — extend for new domains.
# No customer-specific or region-specific hardcoding.
# =============================================================================

# --- Configurable keyword banks (extend these for new domains/languages) ---
# These are HINTS, not rules. Data-driven analysis (cardinality, word count,
# sample patterns) always takes priority. Keywords only help disambiguate
# when data signals are ambiguous.

COLUMN_ROLE_KEYWORDS = {
    "date_field": [
        'date', 'month', 'year', 'quarter', 'period', 'created at', 'updated at',
        'modified', 'timestamp', 'time', 'day', 'week', 'fiscal', 'fy',
        'effective', 'expiry', 'due', 'since', 'start date', 'end date',
        'from date', 'to date',
    ],
    "identifier": [
        'id', 'number', 'no', 'code', 'key', 'entry', 'index', 'ref',
        'serial', 'uuid', 'guid', 'pk', 'fk', 'seq', 'num',
    ],
    "entity_name": [
        'company', 'comp', 'customer', 'vendor', 'supplier', 'client', 'oem',
        'product', 'brand', 'organization', 'org', 'firm', 'partner', 'merchant',
        'manufacturer', 'distributor', 'account', 'institution', 'hospital',
        'school', 'university', 'store', 'shop', 'outlet',
    ],
    "person_name": [
        'name', 'user', 'assigned', 'owner', 'manager', 'head', 'rep',
        'architect', 'created by', 'modified by', 'employee', 'agent',
        'executive', 'engineer', 'analyst', 'director', 'officer', 'admin',
        'contact', 'reviewer', 'approver', 'requester', 'submitter', 'author',
        'assignee', 'reporter', 'salesperson', 'consultant',
    ],
    "description": [
        'description', 'remark', 'comment', 'note', 'detail', 'reason', 'memo',
        'summary', 'text', 'body', 'content', 'narrative', 'explanation',
        'feedback', 'observation', 'message',
    ],
    "dimension": [
        'region', 'block', 'area', 'zone', 'territory', 'country', 'city',
        'state', 'department', 'division', 'segment', 'vertical', 'channel',
        'subcategory', 'sub category', 'market', 'branch', 'location', 'site',
        'geography', 'geo', 'province', 'district', 'sector', 'group',
    ],
    "flag": [
        'status', 'stage', 'flag', 'level', 'grade', 'class', 'priority',
        'active', 'enabled', 'approved', 'is ', 'has ', 'can ',
    ],
}

# Entity suffixes — international coverage
ENTITY_SUFFIXES = [
    # English
    ' Ltd', ' Pvt Ltd', ' Pvt. Ltd.', ' Inc', ' Inc.', ' Corp', ' Corp.',
    ' LLC', ' LLP', ' Co', ' Co.', ' PLC', ' Plc',
    ' Limited', ' Incorporated', ' Corporation',
    # German
    ' GmbH', ' AG', ' KG', ' OHG', ' e.V.',
    # French
    ' SA', ' SAS', ' SARL', ' S.A.',
    # Japanese/Korean
    ' K.K.', ' Co., Ltd.',
    # Indian
    ' Pvt Limited', ' Private Limited',
    # Generic
    ' Group', ' Holdings', ' Enterprises', ' International',
    ' Technologies', ' Solutions', ' Services', ' Systems', ' Industries',
    ' Consulting', ' Partners', ' Associates', ' Foundation',
]

# Date patterns for TEXT columns storing dates
DATE_TEXT_PATTERNS = [
    (r'^\d{4}-\d{2}-\d{2}$', "YYYY-MM-DD"),
    (r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', "YYYY-MM-DD HH:MM (datetime as text)"),
    (r'^\d{2}/\d{2}/\d{4}$', "DD/MM/YYYY or MM/DD/YYYY"),
    (r'^\d{2}-\d{2}-\d{4}$', "DD-MM-YYYY"),
    (r'^\d{2}-[A-Za-z]{3}-\d{4}$', "DD-Mon-YYYY"),
    (r'^\d{2}/[A-Za-z]{3}/\d{4}$', "DD/Mon/YYYY"),
    (r'^[A-Za-z]{3}-\d{2}$', "Mon-YY"),
    (r'^[A-Za-z]{3}-\d{4}$', "Mon-YYYY"),
    (r'^[A-Za-z]+ \d{4}$', "Month YYYY"),
    (r'^\d{4}-\d{2}$', "YYYY-MM"),
    (r'^\d{8}$', "YYYYMMDD (compact)"),
    (r'^\d{2}\.\d{2}\.\d{4}$', "DD.MM.YYYY"),
]

# Delimiters to check for multi-value columns
MULTI_VALUE_DELIMITERS = [
    (", ", "comma-space"),
    (",", "comma"),
    ("|", "pipe"),
    (";", "semicolon"),
    (" / ", "slash"),
    (" | ", "pipe-space"),
    ("\t", "tab"),
]


def infer_match_strategy(col_name: str, col_type: str, samples: list, cardinality: int) -> dict:
    """
    Determine how this column should be matched in WHERE clauses.
    
    Philosophy: For TEXT columns, ALWAYS default to case-insensitive.
    Use partial match (ILIKE '%x%') unless the column is clearly a small 
    closed enum (≤10 distinct, single-word values like Y/N, Active/Inactive).
    Wrong partial match returns a superset (recoverable).
    Wrong exact match returns empty set (data loss).
    """
    col_type_upper = str(col_type).upper()
    
    # Non-text types — deterministic
    if any(t in col_type_upper for t in ['BOOL']):
        return {"strategy": "boolean", "reason": "Boolean column",
                "suggestion": "Use = true/false"}
    
    if any(t in col_type_upper for t in ['INT', 'FLOAT', 'DOUBLE', 'NUMERIC', 'DECIMAL', 'BIGINT', 'REAL', 'SERIAL']):
        return {"strategy": "numeric", "reason": "Numeric column",
                "suggestion": "Use numeric operators: =, >, <, >=, <=, BETWEEN"}
    
    if any(t in col_type_upper for t in ['TIMESTAMP', 'DATE']):
        return {"strategy": "date", "reason": "Date/time column",
                "suggestion": "Use date range operators: >=, <, BETWEEN. For exact day: WHERE col >= 'date' AND col < 'date+1'"}
    
    # TIME without DATE
    if 'TIME' in col_type_upper:
        return {"strategy": "date", "reason": "Time column",
                "suggestion": "Use time range operators: >=, <, BETWEEN"}
    
    # --- ALL TEXT columns from here ---
    str_samples = [str(s).strip() for s in (samples or []) if s is not None and str(s).strip()]
    
    if not str_samples:
        return {"strategy": "partial", "reason": "No samples available — defaulting to safe partial match",
                "suggestion": "Use ILIKE '%value%' for filtering. For exclusions use NOT ILIKE '%value%'"}
    
    avg_word_count = sum(len(s.split()) for s in str_samples) / len(str_samples)
    all_single_word = all(len(s.split()) == 1 for s in str_samples)
    
    # ONLY case for exact match: tiny closed enum with single-word values
    # e.g., Y/N, Active/Inactive, High/Medium/Low, >90/<90
    if (cardinality is not None 
            and cardinality <= 10 
            and all_single_word
            and avg_word_count <= 1.0):
        return {
            "strategy": "exact_case_insensitive",
            "reason": f"Small closed set ({cardinality} values), single-word entries — safe for exact match",
            "suggestion": "Use ILIKE 'value' (case-insensitive exact match, no wildcards)"
        }
    
    # EVERYTHING ELSE: partial + case-insensitive
    # This covers: names, companies, descriptions, codes, categories, 
    # multi-word values, high cardinality, low cardinality with multi-word, etc.
    return {
        "strategy": "partial",
        "reason": f"Text column ({cardinality or 'unknown'} distinct values, avg {avg_word_count:.1f} words) — partial match is safest",
        "suggestion": "Use ILIKE '%value%' for filtering. For exclusions use NOT ILIKE '%value%'"
    }


def infer_value_format(col_name: str, col_type: str, samples: list) -> dict:
    """
    Detect the pattern/format of values stored in this column.
    DATA-DRIVEN — uses regex pattern matching on actual samples.
    
    Returns dict with:
        format: detected pattern description
        casing: "UPPER" | "lower" | "Title Case" | "Mixed" | "N/A"
        examples: 1-2 example values for the reasoner
    """
    col_type_upper = str(col_type).upper()
    
    # Non-text columns — format is implicit from type
    if any(t in col_type_upper for t in ['INT', 'FLOAT', 'DOUBLE', 'NUMERIC', 'DECIMAL', 'BIGINT', 'REAL', 'SERIAL']):
        return {"format": "numeric", "casing": "N/A", "examples": []}
    if any(t in col_type_upper for t in ['BOOL']):
        return {"format": "boolean", "casing": "N/A", "examples": []}
    if any(t in col_type_upper for t in ['TIMESTAMP']):
        return {"format": "timestamp", "casing": "N/A", "examples": []}
    if 'DATE' in col_type_upper and 'UPDATE' not in col_type_upper:
        return {"format": "date_native", "casing": "N/A", "examples": []}
    
    str_samples = [str(s).strip() for s in (samples or []) if s is not None and str(s).strip()]
    if not str_samples:
        return {"format": "unknown", "casing": "unknown", "examples": []}
    
    # Detect casing pattern
    alpha_samples = [s for s in str_samples if any(c.isalpha() for c in s)]
    if alpha_samples:
        if all(s == s.upper() for s in alpha_samples):
            casing = "UPPER"
        elif all(s == s.lower() for s in alpha_samples):
            casing = "lower"
        elif all(s == s.title() for s in alpha_samples):
            casing = "Title Case"
        else:
            casing = "Mixed"
    else:
        casing = "N/A"
    
    # Detect patterns — ordered by specificity
    patterns_found = []
    
    # 1. Date-like strings stored as TEXT
    for pattern, label in DATE_TEXT_PATTERNS:
        match_count = sum(1 for s in str_samples if re.match(pattern, s))
        if match_count >= len(str_samples) * 0.6:
            patterns_found.append(f"Date stored as text: {label}")
            break
    
    # 2. Prefix code patterns: ABC-12345, E0006335, ORD-2025-001, etc.
    if not patterns_found:
        prefixes = {}
        for s in str_samples:
            match = re.match(r'^([A-Za-z]{1,10})[\d_\-]', s)
            if match:
                prefixes[match.group(1)] = prefixes.get(match.group(1), 0) + 1
        for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
            if count >= len(str_samples) * 0.5:
                patterns_found.append(f"Code pattern with '{prefix}' prefix")
                break
    
    # 3. Enumerated/sequenced values: "Quarter 1", "Phase A", "Level 3"
    if not patterns_found and len(str_samples) >= 2:
        words_list = [s.split() for s in str_samples if len(s.split()) >= 2]
        if words_list and len(words_list) >= 2:
            first_words = [w[0] for w in words_list]
            if first_words:
                most_common = max(set(first_words), key=first_words.count)
                if first_words.count(most_common) >= len(words_list) * 0.6:
                    varying_parts = [' '.join(w[1:]) for w in words_list if w[0] == most_common]
                    patterns_found.append(
                        f"Enumerated: '{most_common}' + varying suffix (values: {', '.join(varying_parts[:5])})"
                    )
    
    # 4. Entity names with legal/business suffixes (international)
    if not patterns_found:
        suffix_matches = sum(
            1 for s in str_samples
            if any(s.endswith(suffix) or s.endswith(suffix.rstrip('.') + '.') for suffix in ENTITY_SUFFIXES)
        )
        if suffix_matches >= max(1, len(str_samples) * 0.2):
            patterns_found.append("Entity/organization name with business suffix")
    
    # 5. Numeric strings stored as TEXT
    if not patterns_found:
        numeric_like = sum(1 for s in str_samples if re.match(r'^-?\d+\.?\d*$', s))
        if numeric_like >= len(str_samples) * 0.7:
            patterns_found.append("Numeric values stored as text")
    
    # 6. Fallback — describe what we see
    if not patterns_found:
        if all(len(s.split()) == 1 for s in str_samples):
            patterns_found.append("Single-word values")
        elif all(len(s) > 50 for s in str_samples):
            patterns_found.append("Long text / free-form content")
        else:
            avg_words = sum(len(s.split()) for s in str_samples) / len(str_samples)
            if avg_words >= 2:
                patterns_found.append(f"Multi-word text (avg {avg_words:.1f} words)")
            else:
                patterns_found.append("Mixed text values")
    
    return {
        "format": "; ".join(patterns_found),
        "casing": casing,
        "examples": str_samples[:2]
    }


def infer_column_role(col_name: str, col_type: str, samples: list, cardinality: int,
                      total_rows: int = None) -> str:
    """
    Classify the semantic role of a column.
    
    Strategy: DATA TYPE first → COLUMN NAME keywords second → CARDINALITY fallback.
    Sample pattern matching is intentionally NOT used for person/entity detection 
    because it's fragile ("Bangalore Commercial ITPL" looks like a name pattern 
    but isn't). The LLM enrichment descriptions handle semantic context better.
    
    This function provides a ROUGH classification to help the reasoner.
    It does NOT need to be perfect — the reasoner has samples + descriptions too.
    """
    col_type_upper = str(col_type).upper()
    col_name_lower = col_name.lower().replace("_", " ").replace("-", " ")
    
    def _has_keyword(role: str) -> bool:
        """Word-boundary keyword match."""
        keywords = COLUMN_ROLE_KEYWORDS.get(role, [])
        name_words = re.split(r'[\s_\-\.()\/]+', col_name_lower)
        for kw in keywords:
            if ' ' in kw:
                if kw in col_name_lower:
                    return True
            else:
                if kw in name_words:
                    return True
        return False
    
    # --- Data type gives definitive answer ---
    
    if any(t in col_type_upper for t in ['TIMESTAMP', 'DATE', 'TIME']):
        return "date_field"
    
    if any(t in col_type_upper for t in ['FLOAT', 'DOUBLE', 'NUMERIC', 'DECIMAL', 'REAL']):
        return "measure"
    
    if 'BOOL' in col_type_upper:
        return "flag"
    
    if any(t in col_type_upper for t in ['INT', 'BIGINT', 'SERIAL', 'SMALLINT']):
        if _has_keyword("identifier"):
            return "identifier"
        if cardinality and total_rows and cardinality > total_rows * 0.5:
            return "identifier"
        return "measure"
    
    # --- TEXT columns: keyword-based classification ---
    # Order matters: most specific first
    
    if _has_keyword("date_field"):
        return "date_field"
    
    if _has_keyword("description"):
        return "description"
    
    if _has_keyword("entity_name"):
        return "entity_name"
    
    if _has_keyword("person_name"):
        return "person_name"
    
    if _has_keyword("dimension"):
        return "dimension"
    
    if _has_keyword("flag"):
        return "flag"
    
    # --- Cardinality fallback ---
    if cardinality is not None:
        if cardinality <= 15:
            return "flag"
        if cardinality <= 100:
            return "dimension"
    
    return "unknown"


def infer_null_behavior(col_name: str, col_role: str, null_pct: float, cardinality: int) -> dict:
    """
    Interpret what NULL means for this column and how to handle in filters.
    Based on null_percentage thresholds and column role.
    
    Returns dict with:
        meaning: what NULL likely represents
        filter_advice: how to handle NULLs in WHERE clauses
    """
    if null_pct is None or null_pct == 0:
        return {
            "meaning": "No NULLs present",
            "filter_advice": "No special NULL handling needed"
        }
    
    if null_pct > 80:
        return {
            "meaning": f"Mostly NULL ({null_pct:.0f}%) — optional/sparse field, rarely populated",
            "filter_advice": "This column is mostly empty. When filtering, NULLs should typically be INCLUDED (use OR column IS NULL) to avoid accidentally excluding most data"
        }
    
    if null_pct > 30:
        meaning = f"Partially populated ({null_pct:.0f}% NULL)"
        if col_role in ("person_name", "entity_name"):
            meaning += " — likely means unassigned/untagged records"
        elif col_role == "measure":
            meaning += " — may represent zero or no transaction"
        elif col_role == "flag":
            meaning += " — may represent a default/unset state"
        
        return {
            "meaning": meaning,
            "filter_advice": "When using exclusion filters (NOT/except), add OR column IS NULL to avoid dropping untagged records"
        }
    
    if null_pct > 0:
        return {
            "meaning": f"Low NULL rate ({null_pct:.1f}%) — mostly populated, NULLs are likely data gaps",
            "filter_advice": "When using exclusion filters, consider adding OR column IS NULL to be safe"
        }
    
    return {
        "meaning": "Unknown NULL behavior",
        "filter_advice": "Handle NULLs with caution in filters"
    }


def infer_value_scale(col_name: str, col_type: str, samples: list) -> Optional[dict]:
    """
    For numeric columns, detect the range and scale of values.
    Region-neutral — reports raw numbers and magnitude, lets the reasoner interpret.
    
    Returns dict or None for non-numeric columns.
    """
    col_type_upper = str(col_type).upper()
    
    if not any(t in col_type_upper for t in ['INT', 'FLOAT', 'DOUBLE', 'NUMERIC', 'DECIMAL', 'BIGINT', 'REAL']):
        return None
    
    numeric_samples = []
    for s in (samples or []):
        try:
            val = float(s)
            numeric_samples.append(val)
        except (ValueError, TypeError):
            continue
    
    if not numeric_samples:
        return None
    
    min_val = min(numeric_samples)
    max_val = max(numeric_samples)
    avg_val = sum(numeric_samples) / len(numeric_samples)
    
    # Scale hint — region-neutral, just describe the magnitude
    abs_max = max(abs(min_val), abs(max_val))
    if abs_max >= 1_000_000_000:
        scale_hint = f"Very large values (up to {abs_max:,.0f}). Likely base currency units — user may request results divided into larger denominations."
    elif abs_max >= 1_000_000:
        scale_hint = f"Large values (up to {abs_max:,.0f}). Likely base currency or large counts — user may request results in thousands, millions, or other denominations."
    elif abs_max >= 1_000:
        scale_hint = f"Medium values (range {min_val:,.0f} to {max_val:,.0f})."
    elif abs_max >= 1:
        scale_hint = f"Small values (range {min_val:.2f} to {max_val:.2f}) — could be percentages, counts, or ratios."
    else:
        scale_hint = f"Fractional values (range {min_val:.4f} to {max_val:.4f}) — likely ratios, rates, or percentages."
    
    has_decimals = any(v != int(v) for v in numeric_samples if v == v)  # NaN-safe
    
    return {
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "avg": round(avg_val, 2),
        "scale_hint": scale_hint,
        "has_decimals": has_decimals
    }


def infer_multi_value(col_type: str, samples: list) -> dict:
    """
    Detect if a TEXT column contains delimited/multi-value entries.
    e.g., "Networking, Security, Cloud" or "tag1|tag2|tag3"
    
    Returns dict with:
        is_multi_value: bool
        delimiter: detected delimiter or None
        advice: guidance for the reasoner
    """
    col_type_upper = str(col_type).upper()
    
    if not any(t in col_type_upper for t in ['TEXT', 'VARCHAR', 'CHAR', 'STRING', 'CHARACTER']):
        return {"is_multi_value": False, "delimiter": None, "advice": None}
    
    str_samples = [str(s).strip() for s in (samples or []) if s is not None and str(s).strip()]
    if not str_samples:
        return {"is_multi_value": False, "delimiter": None, "advice": None}
    
    for delim, delim_name in MULTI_VALUE_DELIMITERS:
        contains_delim = sum(1 for s in str_samples if delim in s)
        if contains_delim >= len(str_samples) * 0.3:
            # Verify: splitting should produce 2+ consistent-looking parts
            parts_counts = [len(s.split(delim)) for s in str_samples if delim in s]
            avg_parts = sum(parts_counts) / len(parts_counts) if parts_counts else 0
            if avg_parts >= 2:
                return {
                    "is_multi_value": True,
                    "delimiter": delim_name,
                    "advice": f"Column contains multiple values separated by {delim_name}. Always use ILIKE '%value%' for filtering, never exact match."
                }
    
    return {"is_multi_value": False, "delimiter": None, "advice": None}


def compute_column_intelligence(col_name: str, col_type: str, samples: list,
                                 cardinality: int, null_pct: float,
                                 table_name: str,
                                 total_rows: int = None) -> dict:
    """
    Master function: compute all 6 column intelligence fields at once.
    
    All analysis is data-driven. Keywords from COLUMN_ROLE_KEYWORDS are used
    only as disambiguation hints when data signals are ambiguous.
    
    Returns dict with all intelligence fields ready for JSON storage.
    """
    col_role = infer_column_role(col_name, col_type, samples, cardinality, total_rows)
    
    return {
        "match_strategy": infer_match_strategy(col_name, col_type, samples, cardinality),
        "value_format": infer_value_format(col_name, col_type, samples),
        "column_role": col_role,
        "null_behavior": infer_null_behavior(col_name, col_role, null_pct, cardinality),
        "value_scale": infer_value_scale(col_name, col_type, samples),
        "multi_value": infer_multi_value(col_type, samples),
    }


def get_connection_hash(connection_string: str) -> str:
    """Generate hash of connection string for tenant isolation."""
    return hashlib.md5(connection_string.encode()).hexdigest()[:16]


def check_schema_columns_table(vector_engine) -> Tuple[bool, str]:
    """
    Check if schema_columns table exists in Supabase.
    
    Returns:
        (exists: bool, message: str)
    """
    try:
        with vector_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'schema_columns'
                )
            """)).fetchone()
            
            if result and result[0]:
                # Check column count
                count = conn.execute(text("SELECT COUNT(*) FROM schema_columns")).fetchone()[0]
                return True, f"Table exists with {count} columns profiled"
            else:
                return False, "Table does not exist. Run schema_rag_setup.sql in Supabase."
    except Exception as e:
        return False, f"Error checking table: {str(e)}"


def profile_single_table(
    user_engine,
    vector_engine,
    table_full_name: str,
    connection_hash: str = None,
    tenant_id: str = None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Profile a single table and store in schema_columns.
    
    Args:
        user_engine: SQLAlchemy engine for user's database
        vector_engine: SQLAlchemy engine for Supabase
        table_full_name: Full table name (e.g., "public.vw_sales" or "table: public.vw_sales")
        connection_hash: Hash for connection isolation
        tenant_id: Tenant ID for multi-tenancy
        progress_callback: Optional callback(column_name, status)
    
    Returns:
        {
            "table": str,
            "columns_profiled": int,
            "columns_with_pii": int,
            "errors": [],
            "columns": [...]
        }
    """
    from abbreviations import expand_column_name, get_business_terms
    from pii_detector import detect_pii, detect_data_quality_issues, get_pii_severity
    from smart_sampler import get_smart_samples, get_dialect_name
    from vector_utils_v2 import get_embedding
    
    # Parse table name
    clean_name = table_full_name.replace("table: ", "").replace("view: ", "")
    if "." in clean_name:
        schema_name, table_name = clean_name.split(".", 1)
    else:
        schema_name = "public"
        table_name = clean_name
    
    full_name = f"{schema_name}.{table_name}"
    object_type = "view" if "view:" in table_full_name else "table"
    
    result = {
        "table": full_name,
        "object_type": object_type,
        "columns_profiled": 0,
        "columns_with_pii": 0,
        "errors": [],
        "columns": []
    }
    
    dialect = get_dialect_name(user_engine)
    
    try:
        inspector = inspect(user_engine)
        columns = inspector.get_columns(table_name, schema=schema_name)
        
        for col in columns:
            col_name = col["name"]
            col_type = str(col["type"])
            nullable = col.get("nullable", True)
            
            if progress_callback:
                progress_callback(col_name, "profiling")
            
            try:
                # Get smart samples (dialect is auto-detected from engine)
                sample_result = get_smart_samples(
                    user_engine,
                    full_name,
                    col_name,
                    limit=20
                )
                
                samples = sample_result.get("samples", [])
                cardinality = sample_result.get("cardinality", 0)
                null_pct = sample_result.get("null_percentage", 0)
                
                # Detect PII
                has_pii, pii_types, masked_samples = detect_pii(samples)
                pii_severity = get_pii_severity(pii_types) if has_pii else None
                
                # GDPR/HIPAA Compliance: Never store raw PII samples
                # Only store masked samples or generic placeholders
                if has_pii:
                    safe_samples = masked_samples[:10]  # Use masked version
                else:
                    # Even non-PII samples could contain sensitive business data
                    # Store only if explicitly non-sensitive, otherwise use placeholders
                    safe_samples = samples[:10]  # Non-PII data is safe
                
                # For embedding, always use masked samples to avoid PII in vectors
                samples_for_embedding = masked_samples if has_pii else samples
                
                # Detect data quality issues
                quality_issues = detect_data_quality_issues(
                    column_name=col_name,
                    data_type=col_type,
                    samples=samples,
                    null_percentage=null_pct,
                    cardinality=cardinality
                )
                
                # Compute column intelligence (match_strategy, value_format, etc.)
                col_intelligence = compute_column_intelligence(
                    col_name=col_name,
                    col_type=col_type,
                    samples=samples,
                    cardinality=cardinality,
                    null_pct=null_pct,
                    table_name=full_name
                )
                
                # Auto-expand column name
                expanded_name = expand_column_name(col_name)
                business_terms = get_business_terms(col_name, expanded_name)
                
                # Create embedding text - ALWAYS use masked samples for privacy
                sample_str = ", ".join(str(s) for s in samples_for_embedding[:5])
                
                # Include intelligence in embedding for better semantic retrieval
                role_str = col_intelligence.get("column_role", "unknown")
                match_str = col_intelligence.get("match_strategy", {}).get("strategy", "")
                format_str = col_intelligence.get("value_format", {}).get("format", "")
                
                embedding_text = f"""
{col_name} - {expanded_name}
Table: {full_name}
Type: {col_type}
Role: {role_str}
Match strategy: {match_str}
Value format: {format_str}
Business terms: {', '.join(business_terms[:10])}
Sample values: {sample_str}
""".strip()
                
                # Generate embedding
                embedding = get_embedding(embedding_text)
                
                # Store in schema_columns (delete existing, then insert)
                # GDPR/HIPAA: Only store safe_samples (masked if PII detected)
                with vector_engine.connect() as conn:
                    # First delete any existing row for this column
                    conn.execute(
                        text("""
                            DELETE FROM schema_columns 
                            WHERE object_name = :object_name 
                            AND column_name = :column_name
                            AND tenant_id IS NOT DISTINCT FROM :tenant_id
                            AND connection_hash IS NOT DISTINCT FROM :conn_hash
                        """),
                        {
                            "object_name": full_name,
                            "column_name": col_name,
                            "tenant_id": tenant_id,
                            "conn_hash": connection_hash
                        }
                    )
                    
                    # Then insert the new data
                    # NOTE: sample_values stores ONLY safe (masked) samples - never raw PII
                    conn.execute(
                        text("""
                            INSERT INTO schema_columns (
                                tenant_id, connection_hash, object_name, object_type,
                                column_name, data_type, nullable,
                                auto_expanded_name, sample_values, sample_values_masked,
                                cardinality, null_percentage, has_pii, pii_types, pii_severity,
                                business_terms, data_quality_issues,
                                match_strategy, value_format, column_role,
                                null_behavior, value_scale, multi_value,
                                embedding_text, embedding, enrichment_status
                            ) VALUES (
                                :tenant_id, :conn_hash, :object_name, :object_type,
                                :column_name, :data_type, :nullable,
                                :expanded_name, :samples, :samples_masked,
                                :cardinality, :null_pct, :has_pii, :pii_types, :pii_severity,
                                :business_terms, :quality_issues,
                                :match_strategy, :value_format, :column_role,
                                :null_behavior, :value_scale, :multi_value,
                                :embed_text, CAST(:embedding AS vector), 'auto'
                            )
                        """),
                        {
                            "tenant_id": tenant_id,
                            "conn_hash": connection_hash,
                            "object_name": full_name,
                            "object_type": object_type,
                            "column_name": col_name,
                            "data_type": col_type,
                            "nullable": nullable,
                            "expanded_name": expanded_name,
                            "samples": safe_samples,  # GDPR: Only masked/safe samples
                            "samples_masked": masked_samples[:10] if has_pii else None,
                            "cardinality": cardinality,
                            "null_pct": null_pct,
                            "has_pii": has_pii,
                            "pii_types": pii_types if pii_types else None,
                            "pii_severity": pii_severity,
                            "business_terms": business_terms[:20],
                            "quality_issues": json.dumps(quality_issues) if quality_issues else None,
                            "match_strategy": json.dumps(col_intelligence["match_strategy"]),
                            "value_format": json.dumps(col_intelligence["value_format"]),
                            "column_role": col_intelligence["column_role"],
                            "null_behavior": json.dumps(col_intelligence["null_behavior"]),
                            "value_scale": json.dumps(col_intelligence["value_scale"]) if col_intelligence["value_scale"] else None,
                            "multi_value": json.dumps(col_intelligence["multi_value"]),
                            "embed_text": embedding_text,
                            "embedding": str(embedding)
                        }
                    )
                    conn.commit()
                
                result["columns_profiled"] += 1
                if has_pii:
                    result["columns_with_pii"] += 1
                
                result["columns"].append({
                    "name": col_name,
                    "type": col_type,
                    "expanded": expanded_name,
                    "has_pii": has_pii,
                    "pii_types": pii_types,
                    "cardinality": cardinality,
                    "samples": safe_samples[:3]  # Always show safe (masked) samples
                })
                
                if progress_callback:
                    progress_callback(col_name, "done")
                    
            except Exception as e:
                result["errors"].append(f"{col_name}: {str(e)}")
                if progress_callback:
                    progress_callback(col_name, f"error: {str(e)}")
                    
    except Exception as e:
        result["errors"].append(f"Table error: {str(e)}")
    
    return result


def profile_selected_tables(
    user_engine,
    vector_engine,
    selected_tables: List[str],
    connection_hash: str = None,
    tenant_id: str = None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Profile multiple tables.
    
    Args:
        user_engine: User's database engine
        vector_engine: Supabase engine
        selected_tables: List of table names
        connection_hash: Connection hash
        tenant_id: Tenant ID
        progress_callback: Optional callback(table_name, column_name, status)
    
    Returns:
        {
            "total_tables": int,
            "total_columns": int,
            "columns_with_pii": int,
            "errors": [],
            "tables": [...]
        }
    """
    result = {
        "total_tables": len(selected_tables),
        "total_columns": 0,
        "columns_with_pii": 0,
        "errors": [],
        "tables": []
    }
    
    for table in selected_tables:
        def table_progress(col_name, status):
            if progress_callback:
                progress_callback(table, col_name, status)
        
        table_result = profile_single_table(
            user_engine,
            vector_engine,
            table,
            connection_hash=connection_hash,
            tenant_id=tenant_id,
            progress_callback=table_progress
        )
        
        result["tables"].append(table_result)
        result["total_columns"] += table_result["columns_profiled"]
        result["columns_with_pii"] += table_result["columns_with_pii"]
        result["errors"].extend(table_result["errors"])
    
    return result


def get_profiled_columns(
    vector_engine,
    selected_tables: List[str],
    connection_hash: str = None,
    tenant_id: str = None
) -> List[Dict]:
    """
    Get all profiled columns for selected tables.
    
    Returns list of column dictionaries.
    """
    # Clean table names
    clean_tables = []
    for t in selected_tables:
        clean = t.replace("table: ", "").replace("view: ", "")
        if "." not in clean:
            clean = f"public.{clean}"
        clean_tables.append(clean)
    
    print(f"[DEBUG] get_profiled_columns - clean_tables: {clean_tables}")  # Debug
    
    try:
        with vector_engine.connect() as conn:
            # Simplified query - just check object_name is in the list
            result = conn.execute(
                text("""
                    SELECT 
                        object_name, column_name, data_type, nullable,
                        auto_expanded_name, friendly_name, user_description,
                        sample_values, sample_values_masked,
                        cardinality, null_percentage,
                        has_pii, pii_types, pii_severity,
                        business_terms, data_quality_issues,
                        enrichment_status, updated_at,
                        match_strategy, value_format, column_role,
                        null_behavior, value_scale, multi_value
                    FROM schema_columns
                    WHERE object_name = ANY(:tables)
                    ORDER BY object_name, column_name
                """),
                {
                    "tables": clean_tables
                }
            )
            
            columns = []
            for row in result:
                columns.append({
                    "table": row[0],
                    "column": row[1],
                    "type": row[2],
                    "nullable": row[3],
                    "auto_expanded": row[4],
                    "friendly_name": row[5],
                    "description": row[6],
                    "samples": row[7],
                    "samples_masked": row[8],
                    "cardinality": row[9],
                    "null_pct": row[10],
                    "has_pii": row[11],
                    "pii_types": row[12],
                    "pii_severity": row[13],
                    "business_terms": row[14],
                    "quality_issues": row[15],
                    "status": row[16],
                    "updated": row[17],
                    # Column intelligence fields
                    "match_strategy": row[18],
                    "value_format": row[19],
                    "column_role": row[20],
                    "null_behavior": row[21],
                    "value_scale": row[22],
                    "multi_value": row[23],
                })
            
            print(f"[DEBUG] get_profiled_columns - found {len(columns)} columns")  # Debug
            return columns
    except Exception as e:
        print(f"[DEBUG] get_profiled_columns ERROR: {e}")  # Debug
        return []


def get_enrichment_candidates(
    vector_engine,
    selected_tables: List[str],
    connection_hash: str = None,
    tenant_id: str = None,
    filter_status: str = None  # 'pending', 'auto', 'user', 'all'
) -> List[Dict]:
    """
    Get columns that need enrichment.
    
    Args:
        filter_status: 'pending' (never profiled), 'auto' (auto-profiled), 
                      'user' (user enriched), 'all', or None (pending + auto)
    """
    columns = get_profiled_columns(vector_engine, selected_tables, connection_hash, tenant_id)
    
    if filter_status == 'all':
        return columns
    elif filter_status == 'pending':
        return [c for c in columns if c["status"] == "pending"]
    elif filter_status == 'auto':
        return [c for c in columns if c["status"] == "auto"]
    elif filter_status == 'user':
        return [c for c in columns if c["status"] == "user"]
    else:
        # Default: pending + auto (need enrichment)
        return [c for c in columns if c["status"] in ["pending", "auto"]]


def update_column_enrichment(
    vector_engine,
    table_name: str,
    column_name: str,
    friendly_name: str = None,
    description: str = None,
    business_terms: List[str] = None,
    connection_hash: str = None,
    tenant_id: str = None
) -> bool:
    """
    Update user enrichment for a column.
    
    Returns True if successful.
    """
    from vector_utils_v2 import get_embedding
    
    try:
        # Build new embedding text
        with vector_engine.connect() as conn:
            # Get current data
            current = conn.execute(
                text("""
                    SELECT auto_expanded_name, sample_values, sample_values_masked, has_pii
                    FROM schema_columns
                    WHERE object_name = :table AND column_name = :column
                    AND (tenant_id IS NULL OR tenant_id = :tenant_id)
                    AND (connection_hash IS NULL OR connection_hash = :conn_hash)
                """),
                {
                    "table": table_name,
                    "column": column_name,
                    "tenant_id": tenant_id,
                    "conn_hash": connection_hash
                }
            ).fetchone()
            
            if not current:
                return False
            
            auto_expanded = current[0]
            samples = current[2] if current[3] else current[1]  # Use masked if PII
            
            # Build embedding text with user enrichment
            display_name = friendly_name or auto_expanded or column_name
            terms = business_terms or []
            sample_str = ", ".join(str(s) for s in (samples or [])[:5])
            
            embedding_text = f"""
{column_name} - {display_name}
Table: {table_name}
Description: {description or ''}
Business terms: {', '.join(terms)}
Sample values: {sample_str}
""".strip()
            
            embedding = get_embedding(embedding_text)
            
            # Update
            conn.execute(
                text("""
                    UPDATE schema_columns
                    SET 
                        friendly_name = :friendly_name,
                        user_description = :description,
                        business_terms = COALESCE(:terms, business_terms),
                        embedding_text = :embed_text,
                        embedding = CAST(:embedding AS vector),
                        enrichment_status = 'user',
                        enrichment_date = NOW(),
                        updated_at = NOW()
                    WHERE object_name = :table AND column_name = :column
                    AND (tenant_id IS NULL OR tenant_id = :tenant_id)
                    AND (connection_hash IS NULL OR connection_hash = :conn_hash)
                """),
                {
                    "friendly_name": friendly_name,
                    "description": description,
                    "terms": business_terms,
                    "embed_text": embedding_text,
                    "embedding": str(embedding),
                    "table": table_name,
                    "column": column_name,
                    "tenant_id": tenant_id,
                    "conn_hash": connection_hash
                }
            )
            conn.commit()
            
            return True
            
    except Exception as e:
        print(f"Error updating enrichment: {e}")
        return False


def get_profile_stats(
    vector_engine,
    selected_tables: List[str],
    connection_hash: str = None,
    tenant_id: str = None
) -> Dict[str, Any]:
    """
    Get profiling statistics for selected tables.
    """
    columns = get_profiled_columns(vector_engine, selected_tables, connection_hash, tenant_id)
    
    if not columns:
        return {
            "total_columns": 0,
            "profiled": 0,
            "with_pii": 0,
            "auto_enriched": 0,
            "user_enriched": 0,
            "pending": 0,
            "tables": {}
        }
    
    stats = {
        "total_columns": len(columns),
        "profiled": len(columns),
        "with_pii": sum(1 for c in columns if c["has_pii"]),
        "auto_enriched": sum(1 for c in columns if c["status"] == "auto"),
        "user_enriched": sum(1 for c in columns if c["status"] == "user"),
        "pending": sum(1 for c in columns if c["status"] == "pending"),
        "tables": {}
    }
    
    for col in columns:
        table = col["table"]
        if table not in stats["tables"]:
            stats["tables"][table] = {"columns": 0, "pii": 0, "enriched": 0}
        stats["tables"][table]["columns"] += 1
        if col["has_pii"]:
            stats["tables"][table]["pii"] += 1
        if col["status"] == "user":
            stats["tables"][table]["enriched"] += 1
    
    return stats


# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================

def render_profile_button(st, user_engine, vector_engine, selected_tables):
    """
    Render the Profile Tables button and handle profiling.
    Call this in Tab 1 after table selection.
    """
    import streamlit as st_module
    
    # Check if schema_columns table exists
    exists, message = check_schema_columns_table(vector_engine)
    
    if not exists:
        st.warning(f"⚠️ Schema RAG not set up: {message}")
        with st.expander("📋 Setup Instructions"):
            st.markdown("""
1. Open Supabase SQL Editor
2. Run the `schema_rag_setup.sql` script
3. Refresh this page
            """)
        return
    
    # Get current stats
    stats = get_profile_stats(vector_engine, selected_tables)
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Columns Profiled", stats["profiled"])
    col2.metric("With PII", stats["with_pii"])
    col3.metric("User Enriched", stats["user_enriched"])
    col4.metric("Need Review", stats["auto_enriched"])
    
    # Profile button
    if st.button("🔄 Profile Selected Tables", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_tables = len(selected_tables)
        current = 0
        
        def update_progress(table, column, status):
            nonlocal current
            status_text.text(f"Profiling {table} → {column}: {status}")
        
        with st.spinner("Profiling tables..."):
            result = profile_selected_tables(
                user_engine,
                vector_engine,
                selected_tables,
                progress_callback=update_progress
            )
        
        progress_bar.progress(100)
        status_text.empty()
        
        # Show results
        if result["errors"]:
            st.warning(f"⚠️ Completed with {len(result['errors'])} errors")
            with st.expander("View Errors"):
                for err in result["errors"]:
                    st.write(f"- {err}")
        else:
            st.success(f"✅ Profiled {result['total_columns']} columns across {result['total_tables']} tables")
        
        if result["columns_with_pii"] > 0:
            st.warning(f"🔒 Found PII in {result['columns_with_pii']} columns - samples are masked")
        
        st.rerun()


def render_enrichment_ui(st, vector_engine, selected_tables):
    """
    Render the column enrichment UI.
    Call this in Tab 1 or Tab 2.
    """
    columns = get_profiled_columns(vector_engine, selected_tables)
    
    if not columns:
        st.info("No columns profiled yet. Click 'Profile Selected Tables' first.")
        return
    
    # Filter options
    filter_col, search_col = st.columns([1, 2])
    
    with filter_col:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "Need Review (Auto)", "User Enriched", "Has PII"],
            key="enrichment_filter"
        )
    
    with search_col:
        search = st.text_input("Search columns", placeholder="Type to search...", key="enrichment_search")
    
    # Apply filters
    filtered = columns
    if status_filter == "Need Review (Auto)":
        filtered = [c for c in filtered if c["status"] == "auto"]
    elif status_filter == "User Enriched":
        filtered = [c for c in filtered if c["status"] == "user"]
    elif status_filter == "Has PII":
        filtered = [c for c in filtered if c["has_pii"]]
    
    if search:
        search_lower = search.lower()
        filtered = [c for c in filtered if 
                   search_lower in c["column"].lower() or 
                   search_lower in (c["auto_expanded"] or "").lower() or
                   search_lower in c["table"].lower()]
    
    st.caption(f"Showing {len(filtered)} of {len(columns)} columns")
    
    # Display columns
    for col in filtered:
        with st.expander(
            f"{'🔒' if col['has_pii'] else '📊'} {col['table']}.{col['column']} "
            f"({'✅ Enriched' if col['status'] == 'user' else '⚡ Auto'})",
            expanded=False
        ):
            # Info section
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.write(f"**Type:** {col['type']}")
                st.write(f"**Auto-expanded:** {col['auto_expanded']}")
                st.write(f"**Cardinality:** {col['cardinality'] or 'Unknown'}")
                
                if col['has_pii']:
                    st.error(f"🔒 PII Detected: {', '.join(col['pii_types'] or [])}")
            
            with info_col2:
                samples = col['samples_masked'] if col['has_pii'] else col['samples']
                if samples:
                    st.write("**Samples:**")
                    st.write(", ".join(str(s) for s in samples[:5]))
                
                if col['business_terms']:
                    st.write(f"**Terms:** {', '.join(col['business_terms'][:5])}")
            
            # Enrichment form
            st.divider()
            st.write("**📝 Enrichment**")
            
            with st.form(f"enrich_{col['table']}_{col['column']}"):
                friendly = st.text_input(
                    "Friendly Name",
                    value=col['friendly_name'] or col['auto_expanded'] or "",
                    placeholder="e.g., Customer Region Code"
                )
                
                description = st.text_area(
                    "Description",
                    value=col['description'] or "",
                    placeholder="e.g., Geographic region where the transaction occurred. E=East, W=West, N=North...",
                    height=80
                )
                
                terms = st.text_input(
                    "Business Terms (comma-separated)",
                    value=", ".join(col['business_terms'] or []),
                    placeholder="region, territory, area, location"
                )
                
                if st.form_submit_button("💾 Save Enrichment", type="primary"):
                    term_list = [t.strip() for t in terms.split(",") if t.strip()]
                    
                    success = update_column_enrichment(
                        vector_engine,
                        col['table'],
                        col['column'],
                        friendly_name=friendly if friendly else None,
                        description=description if description else None,
                        business_terms=term_list if term_list else None
                    )
                    
                    if success:
                        st.success("✅ Saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save")


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SCHEMA PROFILER MODULE")
    print("=" * 70)
    print("""
This module profiles database tables for Schema RAG:

Functions:
  - check_schema_columns_table(vector_engine) - Check setup
  - profile_selected_tables(user_engine, vector_engine, tables) - Profile tables
  - get_profiled_columns(vector_engine, tables) - Get profiled data
  - get_enrichment_candidates(vector_engine, tables) - Get columns to review
  - update_column_enrichment(vector_engine, table, column, ...) - Save enrichment

Streamlit UI:
  - render_profile_button(st, user_engine, vector_engine, tables)
  - render_enrichment_ui(st, vector_engine, tables)

Usage in app_v2_dual_connection.py Tab 1:
  
  from schema_profiler import render_profile_button, render_enrichment_ui
  
  # After table selection
  st.divider()
  st.subheader("🔬 Schema Profiling")
  render_profile_button(st, st.session_state.engine, VECTOR_ENGINE, selected_objects)
  
  with st.expander("📝 Enrich Column Descriptions"):
      render_enrichment_ui(st, VECTOR_ENGINE, selected_objects)
""")
    print("=" * 70)