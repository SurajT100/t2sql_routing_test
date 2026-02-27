"""
Column Enrichment with Opus LLM
================================
One-time activity: Use Opus to generate rich, intelligent descriptions
for each column that help SQL Coder understand business context.

Usage:
    from column_enrichment import enrich_columns_with_opus, get_column_descriptions
    
    # One-time (in Tab 1 after profiling):
    enrich_columns_with_opus(vector_engine, table_name, llm_provider="claude_opus")
    
    # At query time:
    descriptions = get_column_descriptions(vector_engine, ["Margin", "Date", "U_Ordertype"])
"""

from typing import Dict, List, Any, Optional
from sqlalchemy import text
import json


def generate_column_description_prompt(
    table_name: str,
    column_name: str,
    data_type: str,
    sample_values: List[str],
    all_columns: List[str]
) -> str:
    """
    Generate prompt for Opus to describe a column.
    """
    samples_str = ", ".join(str(s) for s in sample_values[:10]) if sample_values else "No samples"
    all_cols_str = ", ".join(all_columns[:20])
    
    return f"""You are a database expert. Analyze this column and provide a concise business description.

TABLE: {table_name}
COLUMN: {column_name}
DATA TYPE: {data_type}
SAMPLE VALUES: {samples_str}
OTHER COLUMNS IN TABLE: {all_cols_str}

Provide a 2-3 line description that includes:
1. What this column represents in business terms
2. How it should be used in SQL queries (aggregation? filter? join key?)
3. Any important notes (e.g., "exclude Rebate values", "use for revenue calculations")

Be specific and practical. This description will help an SQL generator write correct queries.

OUTPUT FORMAT (JSON):
{{
  "business_meaning": "What this column represents",
  "sql_usage": "How to use in queries (SUM, COUNT, WHERE, JOIN, GROUP BY)",
  "notes": "Important considerations or common filters"
}}"""


def enrich_single_column(
    column_info: Dict,
    table_columns: List[str],
    llm_provider: str = "claude_opus"
) -> Dict[str, str]:
    """
    Use Opus to generate rich description for a single column.
    
    Args:
        column_info: Dict with column_name, data_type, sample_values
        table_columns: List of all column names in the table
        llm_provider: LLM provider to use
    
    Returns:
        Dict with business_meaning, sql_usage, notes
    """
    from llm_v2 import call_llm
    
    prompt = generate_column_description_prompt(
        table_name=column_info.get("table_name", "unknown"),
        column_name=column_info["column_name"],
        data_type=column_info.get("data_type", "unknown"),
        sample_values=column_info.get("sample_values", []),
        all_columns=table_columns
    )
    
    try:
        response, tokens = call_llm(prompt, llm_provider, prefill="{")
        
        # Ensure valid JSON
        response = response.strip()
        if not response.startswith("{"):
            response = "{" + response
        if not response.endswith("}"):
            response = response + "}"
        
        result = json.loads(response)
        
        # Combine into a single rich description
        description = f"{result.get('business_meaning', '')} {result.get('sql_usage', '')} {result.get('notes', '')}"
        
        return {
            "opus_description": description.strip(),
            "business_meaning": result.get("business_meaning", ""),
            "sql_usage": result.get("sql_usage", ""),
            "notes": result.get("notes", ""),
            "tokens_used": tokens
        }
        
    except Exception as e:
        return {
            "opus_description": f"Column {column_info['column_name']} of type {column_info.get('data_type', 'unknown')}",
            "business_meaning": "",
            "sql_usage": "",
            "notes": "",
            "error": str(e)
        }


def enrich_columns_with_opus(
    vector_engine,
    table_name: str,
    llm_provider: str = "claude_opus",
    progress_callback=None
) -> Dict[str, Any]:
    """
    Enrich all columns in a table with Opus-generated descriptions.
    This is a ONE-TIME activity run after schema profiling.
    
    Args:
        vector_engine: SQLAlchemy engine for Supabase
        table_name: Full table name (e.g., "public.SAP")
        llm_provider: LLM to use for enrichment
        progress_callback: Optional callback(column_name, status)
    
    Returns:
        {
            "enriched": int,
            "errors": int,
            "total_tokens": int,
            "columns": [...]
        }
    """
    result = {
        "enriched": 0,
        "errors": 0,
        "total_tokens": 0,
        "columns": []
    }
    
    try:
        with vector_engine.connect() as conn:
            # Get all columns for this table
            columns = conn.execute(
                text("""
                    SELECT column_name, data_type, sample_values
                    FROM schema_columns
                    WHERE object_name = :table_name
                    ORDER BY column_name
                """),
                {"table_name": table_name}
            ).fetchall()
            
            if not columns:
                result["error"] = f"No columns found for {table_name}"
                return result
            
            # Get list of all column names for context
            all_column_names = [c[0] for c in columns]
            
            for col in columns:
                col_name = col[0]
                
                if progress_callback:
                    progress_callback(col_name, "enriching")
                
                # Generate description with Opus
                col_info = {
                    "table_name": table_name,
                    "column_name": col_name,
                    "data_type": col[1],
                    "sample_values": col[2] if col[2] else []
                }
                
                enrichment = enrich_single_column(
                    col_info,
                    all_column_names,
                    llm_provider
                )
                
                if "error" not in enrichment:
                    # Update the column in database
                    conn.execute(
                        text("""
                            UPDATE schema_columns
                            SET 
                                opus_description = :opus_desc,
                                opus_business_meaning = :business,
                                opus_sql_usage = :sql_usage,
                                opus_notes = :notes,
                                enriched_at = NOW()
                            WHERE object_name = :table_name
                              AND column_name = :col_name
                        """),
                        {
                            "opus_desc": enrichment["opus_description"],
                            "business": enrichment.get("business_meaning", ""),
                            "sql_usage": enrichment.get("sql_usage", ""),
                            "notes": enrichment.get("notes", ""),
                            "table_name": table_name,
                            "col_name": col_name
                        }
                    )
                    conn.commit()
                    
                    result["enriched"] += 1
                    result["total_tokens"] += enrichment.get("tokens_used", {}).get("input", 0)
                    result["total_tokens"] += enrichment.get("tokens_used", {}).get("output", 0)
                else:
                    result["errors"] += 1
                
                result["columns"].append({
                    "column": col_name,
                    "description": enrichment.get("opus_description", ""),
                    "error": enrichment.get("error")
                })
                
    except Exception as e:
        result["error"] = str(e)
    
    return result


def get_column_descriptions(
    vector_engine,
    table_name: str,
    column_names: List[str]
) -> Dict[str, str]:
    """
    Retrieve Opus-generated descriptions for specific columns.
    Called at QUERY TIME after Reasoning LLM identifies needed columns.
    
    Args:
        vector_engine: SQLAlchemy engine for Supabase
        table_name: Table name (e.g., "public.SAP")
        column_names: List of column names to get descriptions for
    
    Returns:
        Dict mapping column_name -> opus_description
    """
    descriptions = {}
    
    try:
        with vector_engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT column_name, opus_description, opus_business_meaning, 
                           opus_sql_usage, opus_notes
                    FROM schema_columns
                    WHERE object_name = :table_name
                      AND column_name = ANY(:columns)
                """),
                {"table_name": table_name, "columns": column_names}
            )
            
            for row in result:
                col_name = row[0]
                # Combine all Opus fields into rich description
                desc_parts = [p for p in [row[1], row[2], row[3], row[4]] if p]
                descriptions[col_name] = " ".join(desc_parts) if desc_parts else f"Column: {col_name}"
                
    except Exception as e:
        # Fallback: return column names as descriptions
        for col in column_names:
            descriptions[col] = f"Column: {col}"
    
    return descriptions


def get_all_column_descriptions_for_table(
    vector_engine,
    table_name: str
) -> List[Dict[str, str]]:
    """
    Get all Opus descriptions for a table (for display in UI).
    
    Args:
        vector_engine: SQLAlchemy engine
        table_name: Table name
    
    Returns:
        List of {column_name, data_type, opus_description, enriched_at}
    """
    results = []
    
    try:
        with vector_engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT column_name, data_type, opus_description, 
                           opus_business_meaning, opus_sql_usage, opus_notes,
                           enriched_at
                    FROM schema_columns
                    WHERE object_name = :table_name
                    ORDER BY column_name
                """),
                {"table_name": table_name}
            ).fetchall()
            
            for row in rows:
                results.append({
                    "column_name": row[0],
                    "data_type": row[1],
                    "opus_description": row[2] or "",
                    "business_meaning": row[3] or "",
                    "sql_usage": row[4] or "",
                    "notes": row[5] or "",
                    "enriched_at": row[6].isoformat() if row[6] else None
                })
                
    except Exception as e:
        pass
    
    return results


def format_column_descriptions_for_sql_coder(
    descriptions: Dict[str, str]
) -> str:
    """
    Format column descriptions for inclusion in SQL Coder prompt.
    
    Args:
        descriptions: Dict from get_column_descriptions()
    
    Returns:
        Formatted string for prompt
    """
    if not descriptions:
        return ""
    
    lines = ["COLUMN CONTEXT (from database expert):"]
    for col_name, desc in descriptions.items():
        lines.append(f"• {col_name}: {desc}")
    
    return "\n".join(lines)


# SQL to add new columns to schema_columns table
SCHEMA_UPDATE_SQL = """
-- Run this in Supabase to add Opus enrichment columns
ALTER TABLE schema_columns 
ADD COLUMN IF NOT EXISTS opus_description TEXT,
ADD COLUMN IF NOT EXISTS opus_business_meaning TEXT,
ADD COLUMN IF NOT EXISTS opus_sql_usage TEXT,
ADD COLUMN IF NOT EXISTS opus_notes TEXT,
ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMP;

-- Create index for faster retrieval
CREATE INDEX IF NOT EXISTS idx_schema_columns_opus 
ON schema_columns(object_name, column_name) 
WHERE opus_description IS NOT NULL;
"""


if __name__ == "__main__":
    print("Column Enrichment Module")
    print("=" * 60)
    print("\nTo add Opus enrichment columns to your database, run:")
    print(SCHEMA_UPDATE_SQL)
