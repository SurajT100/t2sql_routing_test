"""
Schema RAG - Intelligent Schema Retrieval
==========================================
Retrieves only relevant columns for a question instead of full schema.
Dramatically reduces tokens while maintaining accuracy.

Usage:
    from schema_rag import get_relevant_schema, format_schema_for_llm
    
    result = get_relevant_schema(engine, question, ["public.vw_sales"])
    schema_text = format_schema_for_llm(result, "postgresql")
"""

from sqlalchemy import text, inspect
from typing import List, Dict, Any, Optional, Tuple
import json

from abbreviations import expand_column_name, get_business_terms


# =============================================================================
# SCHEMA RETRIEVAL
# =============================================================================

def get_relevant_schema(
    vector_engine,
    question: str,
    selected_tables: List[str],
    top_k: int = 15,
    similarity_threshold: float = 0.60,
    include_all_from_top_tables: bool = True
) -> Dict[str, Any]:
    """
    Retrieve only relevant columns for the question using vector similarity.
    
    Args:
        vector_engine: SQLAlchemy engine for vector database (Supabase)
        question: User's natural language question
        selected_tables: List of tables/views user has selected
        top_k: Maximum columns to retrieve
        similarity_threshold: Minimum similarity score
        include_all_from_top_tables: If True, include all columns from tables with matches
    
    Returns:
        {
            "columns": [
                {"table": str, "column": str, "type": str, "description": str, "similarity": float},
                ...
            ],
            "tables_used": [str],
            "token_estimate": int,
            "retrieval_stats": {...}
        }
    """
    from vector_utils_v2 import get_embedding
    
    # Get question embedding
    question_embedding = get_embedding(question)
    
    result = {
        "columns": [],
        "tables_used": [],
        "token_estimate": 0,
        "retrieval_stats": {
            "total_retrieved": 0,
            "threshold": similarity_threshold,
            "method": "vector_search"
        }
    }
    
    try:
        with vector_engine.connect() as conn:
            # Check if schema_columns table exists
            table_check = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'schema_columns'
                )
            """)).fetchone()
            
            if not table_check or not table_check[0]:
                # Fall back to basic schema if schema_columns doesn't exist
                result["retrieval_stats"]["method"] = "fallback_basic"
                return result
            
            # Search for relevant columns
            search_result = conn.execute(
                text("""
                    SELECT 
                        object_name,
                        column_name,
                        data_type,
                        friendly_name,
                        user_description,
                        business_terms,
                        sample_values,
                        1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                    FROM schema_columns
                    WHERE 
                        object_name = ANY(:tables)
                        AND embedding IS NOT NULL
                        AND 1 - (embedding <=> CAST(:embedding AS vector)) > :threshold
                    ORDER BY similarity DESC
                    LIMIT :top_k
                """),
                {
                    "embedding": str(question_embedding),
                    "tables": selected_tables,
                    "threshold": similarity_threshold,
                    "top_k": top_k
                }
            )
            
            tables_with_matches = set()
            
            for row in search_result:
                result["columns"].append({
                    "table": row[0],
                    "column": row[1],
                    "type": row[2],
                    "friendly_name": row[3],
                    "description": row[4],
                    "business_terms": row[5],
                    "samples": row[6][:3] if row[6] else [],
                    "similarity": round(row[7], 3)
                })
                tables_with_matches.add(row[0])
            
            result["tables_used"] = list(tables_with_matches)
            result["retrieval_stats"]["total_retrieved"] = len(result["columns"])
            
            # If requested, get all columns from tables that had matches
            if include_all_from_top_tables and tables_with_matches:
                # Get columns from matched tables that weren't already retrieved
                existing_cols = {(c["table"], c["column"]) for c in result["columns"]}
                
                additional = conn.execute(
                    text("""
                        SELECT 
                            object_name,
                            column_name,
                            data_type,
                            friendly_name,
                            user_description,
                            sample_values
                        FROM schema_columns
                        WHERE object_name = ANY(:tables)
                        AND (object_name, column_name) NOT IN (
                            SELECT unnest(:existing_tables), unnest(:existing_cols)
                        )
                        LIMIT 20
                    """),
                    {
                        "tables": list(tables_with_matches),
                        "existing_tables": [c["table"] for c in result["columns"]],
                        "existing_cols": [c["column"] for c in result["columns"]]
                    }
                )
                
                for row in additional:
                    if (row[0], row[1]) not in existing_cols:
                        result["columns"].append({
                            "table": row[0],
                            "column": row[1],
                            "type": row[2],
                            "friendly_name": row[3],
                            "description": row[4],
                            "samples": row[5][:3] if row[5] else [],
                            "similarity": 0.0,  # Not from similarity search
                            "source": "table_context"
                        })
            
    except Exception as e:
        result["retrieval_stats"]["error"] = str(e)
        result["retrieval_stats"]["method"] = "failed"
    
    # Estimate tokens
    result["token_estimate"] = _estimate_schema_tokens(result["columns"])
    
    return result


def get_relevant_schema_simple(
    engine,
    question: str,
    selected_tables: List[str],
    top_k: int = 15
) -> Dict[str, Any]:
    """
    Simple schema retrieval without vector search.
    Uses keyword matching on column names and descriptions.
    
    Use this if schema_columns table is not set up yet.
    """
    from abbreviations import expand_column_name
    
    result = {
        "columns": [],
        "tables_used": [],
        "token_estimate": 0,
        "retrieval_stats": {
            "total_retrieved": 0,
            "method": "keyword_match"
        }
    }
    
    # Extract keywords from question
    question_lower = question.lower()
    keywords = set(question_lower.replace('?', '').replace(',', '').split())
    keywords -= {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 
                 'to', 'for', 'of', 'with', 'by', 'from', 'what', 'how', 'when', 
                 'where', 'who', 'give', 'show', 'me', 'all', 'get', 'list'}
    
    inspector = inspect(engine)
    
    for table_full in selected_tables:
        # Parse table name
        if '.' in table_full:
            parts = table_full.replace('table: ', '').replace('view: ', '').split('.')
            schema = parts[0]
            table = parts[1]
        else:
            schema = 'public'
            table = table_full.replace('table: ', '').replace('view: ', '')
        
        try:
            columns = inspector.get_columns(table, schema=schema)
            
            for col in columns:
                col_name = col['name']
                col_type = str(col['type'])
                
                # Expand column name
                expanded = expand_column_name(col_name)
                expanded_words = set(expanded.lower().split())
                
                # Check for keyword matches
                matches = keywords & expanded_words
                match_score = len(matches) / len(keywords) if keywords else 0
                
                # Also check if any keyword is in the column name directly
                if any(kw in col_name.lower() for kw in keywords):
                    match_score += 0.3
                
                if match_score > 0 or len(result["columns"]) < 5:
                    result["columns"].append({
                        "table": f"{schema}.{table}",
                        "column": col_name,
                        "type": col_type,
                        "friendly_name": expanded,
                        "description": None,
                        "samples": [],
                        "similarity": match_score
                    })
                    
                    if f"{schema}.{table}" not in result["tables_used"]:
                        result["tables_used"].append(f"{schema}.{table}")
        
        except Exception as e:
            result["retrieval_stats"]["error"] = str(e)
    
    # Sort by match score and limit
    result["columns"] = sorted(result["columns"], key=lambda x: x["similarity"], reverse=True)[:top_k]
    result["retrieval_stats"]["total_retrieved"] = len(result["columns"])
    result["token_estimate"] = _estimate_schema_tokens(result["columns"])
    
    return result


def _estimate_schema_tokens(columns: List[Dict]) -> int:
    """Estimate token count for schema columns."""
    # Rough estimate: ~20 tokens per column (name, type, description, samples)
    return len(columns) * 20


# =============================================================================
# SCHEMA FORMATTING
# =============================================================================

def format_schema_for_llm(
    schema_result: Dict[str, Any],
    dialect: str,
    compact: bool = True
) -> str:
    """
    Format retrieved schema for LLM consumption.
    
    Args:
        schema_result: Result from get_relevant_schema
        dialect: Database dialect (postgresql, mysql, etc.)
        compact: If True, use minimal formatting
    
    Returns:
        Formatted schema string
    """
    if not schema_result.get("columns"):
        return "No schema information available."
    
    lines = []
    
    # Add dialect info
    dialect_display = dialect.upper() if dialect else "SQL"
    lines.append(f"DATABASE: {dialect_display}")
    
    # Group by table
    by_table = {}
    for col in schema_result["columns"]:
        table = col["table"]
        if table not in by_table:
            by_table[table] = []
        by_table[table].append(col)
    
    for table, cols in by_table.items():
        lines.append(f"\n{table}:")
        
        for col in cols:
            if compact:
                # Compact format: column (TYPE) - description [samples]
                parts = [f"  • {col['column']} ({col['type']})"]
                
                # Add description or friendly name
                desc = col.get("friendly_name") or col.get("description")
                if desc and desc != col['column']:
                    parts.append(f"- {desc}")
                
                # Add samples if available
                samples = col.get("samples", [])
                if samples:
                    sample_str = ", ".join(str(s) for s in samples[:3])
                    parts.append(f"[e.g. {sample_str}]")
                
                lines.append(" ".join(parts))
            else:
                # Verbose format
                lines.append(f"  {col['column']}:")
                lines.append(f"    Type: {col['type']}")
                if col.get("friendly_name"):
                    lines.append(f"    Meaning: {col['friendly_name']}")
                if col.get("description"):
                    lines.append(f"    Description: {col['description']}")
                if col.get("samples"):
                    lines.append(f"    Examples: {col['samples'][:3]}")
                if col.get("business_terms"):
                    lines.append(f"    Terms: {col['business_terms'][:5]}")
    
    return "\n".join(lines)


def format_schema_as_json(schema_result: Dict[str, Any]) -> str:
    """
    Format schema as compact JSON for LLM.
    Even more token-efficient for some models.
    """
    if not schema_result.get("columns"):
        return "{}"
    
    # Group by table
    by_table = {}
    for col in schema_result["columns"]:
        table = col["table"]
        if table not in by_table:
            by_table[table] = []
        by_table[table].append({
            "col": col["column"],
            "type": col["type"],
            "desc": col.get("friendly_name") or col.get("description", ""),
        })
    
    return json.dumps(by_table, separators=(',', ':'))


# =============================================================================
# FULL SCHEMA FALLBACK
# =============================================================================

def get_full_schema(
    engine,
    selected_tables: List[str],
    dialect: str
) -> str:
    """
    Get full schema for selected tables (fallback when schema RAG not available).
    
    Args:
        engine: SQLAlchemy engine
        selected_tables: List of table names
        dialect: Database dialect
    
    Returns:
        Formatted schema string
    """
    inspector = inspect(engine)
    lines = [f"DATABASE: {dialect.upper()}\n"]
    
    for table_full in selected_tables:
        # Parse table name
        if '.' in table_full:
            parts = table_full.replace('table: ', '').replace('view: ', '').split('.')
            schema = parts[0]
            table = parts[1]
        else:
            schema = 'public'
            table = table_full.replace('table: ', '').replace('view: ', '')
        
        try:
            columns = inspector.get_columns(table, schema=schema)
            lines.append(f"\n{schema}.{table}:")
            
            for col in columns:
                col_name = col['name']
                col_type = str(col['type'])
                nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                lines.append(f"  • {col_name} ({col_type}) {nullable}")
        
        except Exception as e:
            lines.append(f"\n{table_full}: Error - {str(e)}")
    
    return "\n".join(lines)


# =============================================================================
# SCHEMA PROFILING (Setup)
# =============================================================================

def profile_table_for_rag(
    engine,
    vector_engine,
    table_name: str,
    tenant_id: str = None,
    connection_hash: str = None
) -> Dict[str, Any]:
    """
    Profile a table and store column embeddings for Schema RAG.
    
    Args:
        engine: SQLAlchemy engine for the user's database
        vector_engine: SQLAlchemy engine for vector database (Supabase)
        table_name: Full table name (schema.table)
        tenant_id: Tenant identifier for multi-tenancy
        connection_hash: Hash of connection string for isolation
    
    Returns:
        Profile results with column count and any errors
    """
    from smart_sampler import sample_all_columns
    from vector_utils_v2 import get_embedding
    from abbreviations import expand_column_name, get_business_terms
    
    result = {
        "table": table_name,
        "columns_profiled": 0,
        "columns_with_pii": 0,
        "errors": []
    }
    
    # Sample all columns
    samples = sample_all_columns(engine, table_name, limit_per_column=10)
    
    if "_error" in samples:
        result["errors"].append(samples["_error"]["message"])
        return result
    
    # Process each column
    for col_name, col_data in samples.items():
        try:
            # Create embedding text
            expanded_name = expand_column_name(col_name)
            business_terms = get_business_terms(col_name, expanded_name)
            
            # Use masked samples if PII detected
            sample_values = col_data.get("samples_masked") or col_data.get("samples", [])
            sample_str = ", ".join(str(s) for s in sample_values[:5]) if sample_values else ""
            
            embedding_text = f"""
{col_name} - {expanded_name}
Table: {table_name}
Type: {col_data.get('data_type', 'unknown')}
Business terms: {', '.join(business_terms[:10])}
Sample values: {sample_str}
""".strip()
            
            # Generate embedding
            embedding = get_embedding(embedding_text)
            
            # Store in vector database
            with vector_engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO schema_columns (
                            tenant_id, connection_hash, object_name, column_name,
                            data_type, nullable, auto_expanded_name, sample_values,
                            sample_values_masked, cardinality, has_pii, 
                            business_terms, embedding_text, embedding,
                            enrichment_status
                        ) VALUES (
                            :tenant_id, :conn_hash, :table, :column,
                            :dtype, :nullable, :expanded, :samples,
                            :samples_masked, :cardinality, :has_pii,
                            :terms, :embed_text, CAST(:embedding AS vector),
                            'auto'
                        )
                        ON CONFLICT (tenant_id, connection_hash, object_name, column_name)
                        DO UPDATE SET
                            data_type = EXCLUDED.data_type,
                            sample_values = EXCLUDED.sample_values,
                            sample_values_masked = EXCLUDED.sample_values_masked,
                            cardinality = EXCLUDED.cardinality,
                            has_pii = EXCLUDED.has_pii,
                            embedding_text = EXCLUDED.embedding_text,
                            embedding = EXCLUDED.embedding,
                            updated_at = NOW()
                    """),
                    {
                        "tenant_id": tenant_id,
                        "conn_hash": connection_hash,
                        "table": table_name,
                        "column": col_name,
                        "dtype": col_data.get("data_type"),
                        "nullable": col_data.get("nullable", True),
                        "expanded": expanded_name,
                        "samples": col_data.get("samples", []),
                        "samples_masked": col_data.get("samples_masked"),
                        "cardinality": col_data.get("cardinality", 0),
                        "has_pii": col_data.get("has_pii", False),
                        "terms": business_terms[:20],
                        "embed_text": embedding_text,
                        "embedding": str(embedding)
                    }
                )
                conn.commit()
            
            result["columns_profiled"] += 1
            if col_data.get("has_pii"):
                result["columns_with_pii"] += 1
                
        except Exception as e:
            result["errors"].append(f"{col_name}: {str(e)}")
    
    return result


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SCHEMA RAG MODULE")
    print("=" * 70)
    print("\nThis module provides intelligent schema retrieval:")
    print("  • Vector-based column search")
    print("  • Keyword fallback when vectors unavailable")
    print("  • Token-efficient formatting")
    print("  • Schema profiling for RAG setup")
    print("\nUsage:")
    print("  from schema_rag import get_relevant_schema, format_schema_for_llm")
    print("  ")
    print("  # With vector search (requires schema_columns table)")
    print("  result = get_relevant_schema(vector_engine, question, tables)")
    print("  schema_text = format_schema_for_llm(result, 'postgresql')")
    print("  ")
    print("  # Simple keyword-based (no setup required)")
    print("  result = get_relevant_schema_simple(engine, question, tables)")
    print("=" * 70)
