"""
Smart Column Sampler
=====================
Intelligent sampling that handles edge cases:
- Large tables (uses TABLESAMPLE)
- NULL-heavy columns
- Distinct vs repeated values
- Dialect-specific SQL

Usage:
    from smart_sampler import get_smart_samples, get_column_stats
    
    result = get_smart_samples(engine, "public.orders", "region", limit=10)
"""

from sqlalchemy import text, inspect
from typing import List, Dict, Any, Optional, Tuple
import re

from pii_detector import detect_pii, detect_data_quality_issues


# =============================================================================
# DIALECT UTILITIES
# =============================================================================

def get_dialect_name(engine) -> str:
    """Get normalized dialect name from SQLAlchemy engine."""
    dialect = engine.dialect.name.lower()
    
    if dialect in ['postgres', 'postgresql']:
        return 'postgresql'
    elif dialect in ['mysql', 'mariadb']:
        return 'mysql'
    elif dialect in ['mssql', 'sqlserver']:
        return 'mssql'
    elif dialect in ['oracle']:
        return 'oracle'
    elif dialect in ['sqlite']:
        return 'sqlite'
    else:
        return dialect


def parse_table_name(table_name: str) -> Tuple[str, str]:
    """
    Parse schema.table into components.
    
    Returns:
        (schema, table) tuple
    """
    if '.' in table_name:
        parts = table_name.split('.')
        return parts[0], parts[1]
    return 'public', table_name


def quote_identifier(identifier: str, dialect: str) -> str:
    """Quote an identifier for the specific dialect."""
    quote_chars = {
        'postgresql': '"',
        'mysql': '`',
        'mssql': '[',
        'oracle': '"',
        'sqlite': '"'
    }
    
    char = quote_chars.get(dialect, '"')
    end_char = ']' if dialect == 'mssql' else char
    
    # Handle schema.table
    if '.' in identifier:
        parts = identifier.split('.')
        return f"{char}{parts[0]}{end_char}.{char}{parts[1]}{end_char}"
    
    return f"{char}{identifier}{end_char}"


def quote_table(schema: str, table: str, dialect: str) -> str:
    """Quote a table name with schema for the specific dialect."""
    quote_chars = {
        'postgresql': '"',
        'mysql': '`',
        'mssql': '[',
        'oracle': '"',
        'sqlite': '"'
    }
    
    char = quote_chars.get(dialect, '"')
    end_char = ']' if dialect == 'mssql' else char
    
    if dialect == 'sqlite':
        return f"{char}{table}{end_char}"
    
    return f"{char}{schema}{end_char}.{char}{table}{end_char}"


# =============================================================================
# ROW COUNT ESTIMATION
# =============================================================================

def get_row_count_fast(engine, table_name: str) -> int:
    """
    Get approximate row count quickly without scanning full table.
    Uses database statistics where available.
    """
    dialect = get_dialect_name(engine)
    schema, table = parse_table_name(table_name)
    
    try:
        with engine.connect() as conn:
            if dialect == 'postgresql':
                # Use pg_stat for estimate
                result = conn.execute(text("""
                    SELECT reltuples::bigint AS estimate
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = :table
                    AND n.nspname = :schema
                """), {"table": table, "schema": schema})
                row = result.fetchone()
                if row and row[0] and row[0] > 0:
                    return int(row[0])
                    
            elif dialect == 'mysql':
                result = conn.execute(text("""
                    SELECT TABLE_ROWS 
                    FROM information_schema.TABLES 
                    WHERE TABLE_SCHEMA = :schema
                    AND TABLE_NAME = :table
                """), {"table": table, "schema": schema})
                row = result.fetchone()
                if row and row[0]:
                    return int(row[0])
                    
            elif dialect == 'mssql':
                result = conn.execute(text(f"""
                    SELECT SUM(rows) AS estimate
                    FROM sys.partitions
                    WHERE object_id = OBJECT_ID(:full_name)
                    AND index_id IN (0, 1)
                """), {"full_name": f"{schema}.{table}"})
                row = result.fetchone()
                if row and row[0]:
                    return int(row[0])
            
            elif dialect == 'oracle':
                result = conn.execute(text("""
                    SELECT NUM_ROWS
                    FROM ALL_TABLES
                    WHERE OWNER = :schema
                    AND TABLE_NAME = :table
                """), {"table": table.upper(), "schema": schema.upper()})
                row = result.fetchone()
                if row and row[0]:
                    return int(row[0])
            
            # Fallback: actual count (may be slow for large tables)
            qt = quote_table(schema, table, dialect)
            result = conn.execute(text(f"SELECT COUNT(*) FROM {qt}"))
            row = result.fetchone()
            return int(row[0]) if row else 0
            
    except Exception as e:
        print(f"Warning: Could not get row count for {table_name}: {e}")
        return 0


# =============================================================================
# COLUMN STATISTICS
# =============================================================================

def get_column_stats(
    engine,
    table_name: str,
    column_name: str
) -> Dict[str, Any]:
    """
    Get statistics for a column.
    
    Returns:
        {
            "distinct_count": int,
            "null_count": int,
            "null_percentage": float,
            "total_rows": int,
            "min_value": Any,
            "max_value": Any
        }
    """
    dialect = get_dialect_name(engine)
    schema, table = parse_table_name(table_name)
    
    qt = quote_table(schema, table, dialect)
    qc = quote_identifier(column_name, dialect)
    
    stats = {
        "distinct_count": 0,
        "null_count": 0,
        "null_percentage": 0.0,
        "total_rows": 0,
        "min_value": None,
        "max_value": None
    }
    
    try:
        with engine.connect() as conn:
            # Get counts
            sql = f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT {qc}) as distinct_count,
                    SUM(CASE WHEN {qc} IS NULL THEN 1 ELSE 0 END) as null_count
                FROM {qt}
            """
            result = conn.execute(text(sql))
            row = result.fetchone()
            
            if row:
                stats["total_rows"] = int(row[0]) if row[0] else 0
                stats["distinct_count"] = int(row[1]) if row[1] else 0
                stats["null_count"] = int(row[2]) if row[2] else 0
                
                if stats["total_rows"] > 0:
                    stats["null_percentage"] = (stats["null_count"] / stats["total_rows"]) * 100
            
            # Get min/max for appropriate types
            try:
                sql = f"SELECT MIN({qc}), MAX({qc}) FROM {qt}"
                result = conn.execute(text(sql))
                row = result.fetchone()
                if row:
                    stats["min_value"] = row[0]
                    stats["max_value"] = row[1]
            except:
                pass  # Min/max may fail for some column types
                
    except Exception as e:
        print(f"Warning: Could not get stats for {column_name}: {e}")
    
    return stats


# =============================================================================
# SMART SAMPLING
# =============================================================================

def get_smart_samples(
    engine,
    table_name: str,
    column_name: str,
    limit: int = 10,
    include_stats: bool = True,
    check_pii: bool = True
) -> Dict[str, Any]:
    """
    Get intelligent samples from a column, handling edge cases.
    
    Features:
    - Uses DISTINCT to avoid repeated values
    - Handles large tables efficiently
    - Detects and masks PII
    - Reports data quality issues
    
    Args:
        engine: SQLAlchemy engine
        table_name: Full table name (schema.table)
        column_name: Column name
        limit: Maximum samples to return
        include_stats: Whether to include column statistics
        check_pii: Whether to check for PII
    
    Returns:
        {
            "samples": [...],
            "samples_masked": [...],  # If PII detected
            "cardinality": int,
            "null_percentage": float,
            "total_rows": int,
            "sampling_method": str,
            "has_pii": bool,
            "pii_types": [...],
            "data_quality_issues": [...],
            "warnings": [...]
        }
    """
    dialect = get_dialect_name(engine)
    schema, table = parse_table_name(table_name)
    
    result = {
        "samples": [],
        "samples_masked": None,
        "cardinality": 0,
        "null_percentage": 0.0,
        "total_rows": 0,
        "sampling_method": "unknown",
        "has_pii": False,
        "pii_types": [],
        "data_quality_issues": [],
        "warnings": []
    }
    
    try:
        # Step 1: Get row count estimate
        row_count = get_row_count_fast(engine, table_name)
        result["total_rows"] = row_count
        
        if row_count == 0:
            result["warnings"].append("empty_table")
            result["sampling_method"] = "none"
            return result
        
        # Step 2: Get column statistics
        if include_stats:
            stats = get_column_stats(engine, table_name, column_name)
            result["cardinality"] = stats["distinct_count"]
            result["null_percentage"] = stats["null_percentage"]
            result["total_rows"] = stats["total_rows"] or row_count
        
        # Check if all nulls
        if result["cardinality"] == 0:
            result["warnings"].append("all_null_or_no_distinct")
            result["sampling_method"] = "none"
            return result
        
        # Step 3: Choose sampling strategy and get samples
        qt = quote_table(schema, table, dialect)
        qc = quote_identifier(column_name, dialect)
        
        if result["cardinality"] <= 50:
            # Low cardinality - get ALL distinct values
            samples = _get_all_distinct(engine, qt, qc, dialect, limit=50)
            result["sampling_method"] = "all_distinct"
            
        elif row_count < 100000:
            # Medium table - simple distinct sample
            samples = _get_distinct_sample(engine, qt, qc, dialect, limit)
            result["sampling_method"] = "distinct_sample"
            
        else:
            # Large table - use efficient sampling
            samples = _get_random_sample_large(engine, qt, qc, dialect, limit, row_count)
            result["sampling_method"] = "random_sample"
        
        result["samples"] = samples
        
        # Step 4: Check for PII
        if check_pii and samples:
            has_pii, pii_types, masked = detect_pii(samples)
            result["has_pii"] = has_pii
            result["pii_types"] = pii_types
            if has_pii:
                result["samples_masked"] = masked
                result["warnings"].append(f"pii_detected:{','.join(pii_types)}")
        
        # Step 5: Check data quality
        col_type = _get_column_type(engine, table_name, column_name)
        issues = detect_data_quality_issues(
            column_name,
            col_type,
            samples,
            null_percentage=result["null_percentage"],
            cardinality=result["cardinality"],
            total_rows=result["total_rows"]
        )
        result["data_quality_issues"] = issues
        
        # Add issue types to warnings
        for issue in issues:
            if issue["severity"] in ["error", "warning"]:
                result["warnings"].append(issue["type"])
        
    except Exception as e:
        result["warnings"].append(f"sampling_error:{str(e)}")
        result["sampling_method"] = "failed"
    
    return result


def _get_all_distinct(engine, qt: str, qc: str, dialect: str, limit: int = 50) -> List:
    """Get all distinct values (for low cardinality columns)."""
    with engine.connect() as conn:
        sql = f"""
            SELECT DISTINCT {qc}
            FROM {qt}
            WHERE {qc} IS NOT NULL
            ORDER BY {qc}
        """
        
        # Add limit based on dialect
        if dialect in ['postgresql', 'mysql', 'sqlite']:
            sql += f" LIMIT {limit}"
        elif dialect == 'mssql':
            sql = sql.replace("SELECT DISTINCT", f"SELECT DISTINCT TOP {limit}")
        elif dialect == 'oracle':
            sql = f"SELECT * FROM ({sql}) WHERE ROWNUM <= {limit}"
        
        result = conn.execute(text(sql))
        return [row[0] for row in result]


def _get_distinct_sample(engine, qt: str, qc: str, dialect: str, limit: int) -> List:
    """Get distinct sample for medium-sized tables."""
    with engine.connect() as conn:
        if dialect in ['postgresql', 'mysql', 'sqlite']:
            sql = f"""
                SELECT DISTINCT {qc}
                FROM {qt}
                WHERE {qc} IS NOT NULL
                LIMIT {limit}
            """
        elif dialect == 'mssql':
            sql = f"""
                SELECT DISTINCT TOP {limit} {qc}
                FROM {qt}
                WHERE {qc} IS NOT NULL
            """
        elif dialect == 'oracle':
            sql = f"""
                SELECT DISTINCT {qc}
                FROM {qt}
                WHERE {qc} IS NOT NULL
                AND ROWNUM <= {limit}
            """
        else:
            sql = f"""
                SELECT DISTINCT {qc}
                FROM {qt}
                WHERE {qc} IS NOT NULL
                LIMIT {limit}
            """
        
        result = conn.execute(text(sql))
        return [row[0] for row in result]


def _get_random_sample_large(
    engine, 
    qt: str, 
    qc: str, 
    dialect: str, 
    limit: int,
    row_count: int
) -> List:
    """Get random sample from large tables using efficient methods."""
    with engine.connect() as conn:
        if dialect == 'postgresql':
            # Use TABLESAMPLE for PostgreSQL
            # Calculate percentage to get approximately 1000 rows
            sample_pct = min(100, max(0.01, (1000 / row_count) * 100))
            sql = f"""
                SELECT DISTINCT {qc}
                FROM {qt} TABLESAMPLE BERNOULLI({sample_pct})
                WHERE {qc} IS NOT NULL
                LIMIT {limit}
            """
            
        elif dialect == 'mssql':
            # Use TABLESAMPLE for SQL Server
            sql = f"""
                SELECT DISTINCT TOP {limit} {qc}
                FROM {qt} TABLESAMPLE (1000 ROWS)
                WHERE {qc} IS NOT NULL
            """
            
        elif dialect == 'mysql':
            # MySQL doesn't have TABLESAMPLE, use RAND()
            sql = f"""
                SELECT DISTINCT {qc}
                FROM {qt}
                WHERE {qc} IS NOT NULL
                ORDER BY RAND()
                LIMIT {limit}
            """
            
        elif dialect == 'oracle':
            # Use SAMPLE for Oracle
            sql = f"""
                SELECT DISTINCT {qc}
                FROM (
                    SELECT {qc}
                    FROM {qt} SAMPLE(1)
                    WHERE {qc} IS NOT NULL
                )
                WHERE ROWNUM <= {limit}
            """
            
        else:
            # Generic fallback
            sql = f"""
                SELECT DISTINCT {qc}
                FROM {qt}
                WHERE {qc} IS NOT NULL
                LIMIT {limit}
            """
        
        try:
            result = conn.execute(text(sql))
            return [row[0] for row in result]
        except Exception as e:
            # Fallback if TABLESAMPLE fails
            print(f"Warning: Efficient sampling failed, using fallback: {e}")
            return _get_distinct_sample(engine, qt, qc, dialect, limit)


def _get_column_type(engine, table_name: str, column_name: str) -> Optional[str]:
    """Get the SQL data type of a column."""
    try:
        schema, table = parse_table_name(table_name)
        inspector = inspect(engine)
        columns = inspector.get_columns(table, schema=schema)
        
        for col in columns:
            if col['name'].lower() == column_name.lower():
                return str(col['type'])
        
        return None
    except:
        return None


# =============================================================================
# BATCH SAMPLING
# =============================================================================

def sample_all_columns(
    engine,
    table_name: str,
    limit_per_column: int = 10,
    check_pii: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Sample all columns in a table.
    
    Args:
        engine: SQLAlchemy engine
        table_name: Table name
        limit_per_column: Max samples per column
        check_pii: Whether to check for PII
    
    Returns:
        Dictionary mapping column names to their sample results
    """
    schema, table = parse_table_name(table_name)
    inspector = inspect(engine)
    
    try:
        columns = inspector.get_columns(table, schema=schema)
    except Exception as e:
        return {"_error": {"message": f"Could not get columns: {e}"}}
    
    results = {}
    
    for col in columns:
        col_name = col['name']
        col_type = str(col['type'])
        
        # Skip binary/blob columns
        if any(t in col_type.upper() for t in ['BLOB', 'BINARY', 'BYTEA', 'IMAGE', 'VARBINARY']):
            results[col_name] = {
                "samples": [],
                "sampling_method": "skipped",
                "warnings": ["binary_column"],
                "data_type": col_type
            }
            continue
        
        result = get_smart_samples(
            engine,
            table_name,
            col_name,
            limit=limit_per_column,
            check_pii=check_pii
        )
        result["data_type"] = col_type
        result["nullable"] = col.get('nullable', True)
        
        results[col_name] = result
    
    return results


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SMART SAMPLER MODULE")
    print("=" * 70)
    print("\nThis module provides intelligent column sampling with:")
    print("  • Efficient sampling for large tables (TABLESAMPLE)")
    print("  • DISTINCT values to avoid repetition")
    print("  • Automatic PII detection and masking")
    print("  • Data quality issue detection")
    print("  • Dialect-aware SQL generation")
    print("\nUsage:")
    print("  from smart_sampler import get_smart_samples, sample_all_columns")
    print("  ")
    print("  # Single column")
    print("  result = get_smart_samples(engine, 'public.orders', 'region')")
    print("  ")
    print("  # All columns in table")
    print("  results = sample_all_columns(engine, 'public.orders')")
    print("=" * 70)
