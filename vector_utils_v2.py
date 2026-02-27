"""
Enhanced Vector Utils V2 - Multi-Tier RAG Retrieval
====================================================
Implements intelligent context retrieval with:
1. Tier 1: Critical rules (always included)
2. Tier 2: Keyword matching (fast lookup)
3. Tier 3: Vector similarity search (semantic)
4. Tier 4: Similar query examples

Uses REAL sentence-transformers embeddings (not dummy)
"""

from sqlalchemy import text
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, date
import json

# ============================================================================
# GLOBAL EMBEDDING MODEL
# ============================================================================

# Initialize once to avoid reloading
print("🔄 Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded (384 dimensions)")


# ============================================================================
# CORE EMBEDDING FUNCTION
# ============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Generate REAL embedding using Sentence Transformers
    
    Args:
        text: Input text to embed
        
    Returns:
        384-dimensional vector as list
    """
    embedding = embedding_model.encode(text, convert_to_tensor=False)
    return embedding.tolist()


# ============================================================================
# TIER 1: CRITICAL RULES (Always Included)
# ============================================================================

def get_critical_rules(engine) -> List[Dict[str, Any]]:
    """
    Retrieve rules that must ALWAYS be included
    These are typically:
    - Priority 1 (Critical)
    - is_mandatory = TRUE
    - Rules with auto_apply = TRUE in rule_data
    
    Returns:
        List of critical rules with metadata
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT 
                    id,
                    rule_name,
                    rule_type,
                    rule_description,
                    rule_data,
                    trigger_keywords,
                    applies_to_tables,
                    priority,
                    1.0 as similarity
                FROM business_rules_v2
                WHERE 
                    is_active = TRUE
                    AND (
                        is_mandatory = TRUE
                        OR priority = 1
                        OR rule_data->>'auto_apply' = 'true'
                    )
                ORDER BY priority ASC, id ASC
            """)
        )
        
        rules = []
        for row in result:
            rules.append({
                "id": row[0],
                "rule_name": row[1],
                "rule_type": row[2],
                "description": row[3],
                "rule_data": row[4],
                "keywords": row[5],
                "tables": row[6],
                "priority": row[7],
                "similarity": row[8],
                "tier": "critical",
                "reason": "Always applied (mandatory)"
            })
        
        return rules


# ============================================================================
# TIER 2: KEYWORD MATCHING (Fast Lookup)
# ============================================================================

def get_keyword_matched_rules(engine, question: str, 
                               exclude_ids: List[int] = None,
                               enhanced_keywords: List[str] = None) -> List[Dict[str, Any]]:
    """
    Fast keyword matching using PostgreSQL text search
    
    IMPROVED: Searches ALL text fields (name, description, rule_data, keywords)
    to catch rules even when keywords array doesn't have exact matches.
    
    Example: "monthly revenue" -> matches rules with "revenue" in name, description, OR keywords
    
    Args:
        question: User's question
        exclude_ids: Rule IDs to exclude (already retrieved in Tier 1)
        enhanced_keywords: Additional keywords from semantic analysis
        
    Returns:
        List of matched rules
    """
    
    # Extract potential keywords from question (simple tokenization)
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 
                  'at', 'to', 'for', 'of', 'with', 'by', 'from', 'what', 
                  'how', 'when', 'where', 'who', 'give', 'show', 'me', 'all',
                  'get', 'list', 'find', 'total', 'sum', 'count', 'average'}
    
    words = question.lower().replace('?', '').replace(',', '').replace("'", '').split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Add enhanced keywords from semantic analysis
    if enhanced_keywords:
        keywords.extend(enhanced_keywords)
        keywords = list(set(keywords))  # Deduplicate
    
    if not keywords:
        return []
    
    exclude_clause = ""
    if exclude_ids:
        exclude_clause = "AND id != ALL(:exclude_ids)"
    
    # Build ILIKE patterns for each keyword
    keyword_patterns = [f"%{k}%" for k in keywords]
    
    try:
        with engine.connect() as conn:
            # Search ALL text fields: rule_name, rule_description, rule_data, trigger_keywords
            result = conn.execute(
                text(f"""
                    SELECT 
                        id,
                        rule_name,
                        rule_type,
                        rule_description,
                        rule_data,
                        trigger_keywords,
                        applies_to_tables,
                        priority,
                        is_mandatory,
                        0.9 as similarity
                    FROM business_rules_v2
                    WHERE 
                        is_active = TRUE
                        AND (
                            -- Match in trigger_keywords array
                            EXISTS (
                                SELECT 1 FROM unnest(trigger_keywords) kw 
                                WHERE kw ILIKE ANY(:patterns)
                            )
                            -- OR match in rule_name
                            OR rule_name ILIKE ANY(:patterns)
                            -- OR match in rule_description
                            OR rule_description ILIKE ANY(:patterns)
                            -- OR match in rule_data (JSON as text)
                            OR rule_data::text ILIKE ANY(:patterns)
                        )
                        {exclude_clause}
                    ORDER BY 
                        is_mandatory DESC,
                        priority ASC
                    LIMIT 15
                """),
                {"patterns": keyword_patterns, "exclude_ids": exclude_ids} if exclude_ids 
                else {"patterns": keyword_patterns}
            )
            
            rules = []
            for row in result:
                rules.append({
                    "id": row[0],
                    "rule_name": row[1],
                    "rule_type": row[2],
                    "description": row[3],
                    "rule_data": row[4],
                    "keywords": row[5],
                    "tables": row[6],
                    "priority": row[7],
                    "is_mandatory": row[8],
                    "similarity": row[9],
                    "tier": "keyword",
                    "reason": f"Text match in fields"
                })
            
            return rules
    except Exception as e:
        print(f"[DEBUG] Keyword search error: {e}")
        return []


# ============================================================================
# TIER 3: VECTOR SIMILARITY SEARCH
# ============================================================================

def search_similar_rules(engine, question: str, top_k: int = 10, 
                         threshold: float = 0.45,
                         exclude_ids: List[int] = None) -> List[Dict[str, Any]]:
    """
    Semantic similarity search using vector embeddings
    
    IMPROVED: Lower threshold (0.45) to catch more relevant rules.
    Better to retrieve more rules and let LLM filter than miss important ones.
    
    Example: "How much did we earn last month?" 
             -> semantically similar to "monthly revenue"
    
    Args:
        question: User's question
        top_k: Number of results to return
        threshold: Minimum similarity score (0-1) - lowered to 0.45
        exclude_ids: Rule IDs to exclude (already retrieved)
        
    Returns:
        List of similar rules with similarity scores
    """
    
    # Generate embedding for question
    query_embedding = get_embedding(question)
    
    exclude_clause = ""
    params = {
        "embedding": str(query_embedding),
        "threshold": threshold,
        "top_k": top_k
    }
    
    if exclude_ids:
        exclude_clause = "AND id != ALL(:exclude_ids)"
        params["exclude_ids"] = exclude_ids
    
    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
                SELECT 
                    id,
                    rule_name,
                    rule_type,
                    rule_description,
                    rule_data,
                    trigger_keywords,
                    applies_to_tables,
                    priority,
                    1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
                FROM business_rules_v2
                WHERE 
                    is_active = TRUE
                    AND 1 - (embedding <=> CAST(:embedding AS vector)) > :threshold
                    {exclude_clause}
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :top_k
            """),
            params
        )
        
        rules = []
        for row in result:
            rules.append({
                "id": row[0],
                "rule_name": row[1],
                "rule_type": row[2],
                "description": row[3],
                "rule_data": row[4],
                "keywords": row[5],
                "tables": row[6],
                "priority": row[7],
                "similarity": float(row[8]),
                "tier": "vector",
                "reason": f"Semantic match ({row[8]:.2f})"
            })
        
        return rules


# ============================================================================
# TIER 4: SIMILAR QUERY EXAMPLES
# ============================================================================

def search_similar_examples(engine, question: str, top_k: int = 3,
                            threshold: float = 0.70) -> List[Dict[str, Any]]:
    """
    Find similar verified query examples for few-shot learning
    
    Args:
        question: User's question
        top_k: Number of examples to return
        threshold: Minimum similarity score
        
    Returns:
        List of similar examples with SQL and explanation
    """
    
    query_embedding = get_embedding(question)
    
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT 
                    question,
                    sql_query,
                    explanation,
                    query_type,
                    concepts,
                    1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
                FROM query_examples_v2
                WHERE 
                    is_verified = TRUE
                    AND is_active = TRUE
                    AND 1 - (embedding <=> CAST(:embedding AS vector)) > :threshold
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :top_k
            """),
            {
                "embedding": str(query_embedding),
                "threshold": threshold,
                "top_k": top_k
            }
        )
        
        examples = []
        for row in result:
            examples.append({
                "question": row[0],
                "sql": row[1],
                "explanation": row[2],
                "query_type": row[3],
                "concepts": row[4],
                "similarity": float(row[5])
            })
        
        return examples


# ============================================================================
# MAIN CONTEXT RETRIEVAL FUNCTION
# ============================================================================

def get_relevant_context(engine, question: str, 
                         schema: dict = None,
                         enable_vector: bool = True,
                         enable_vector_search: bool = True,
                         similarity_threshold: float = 0.45) -> Dict[str, Any]:
    """
    Main function to retrieve all relevant context for a question
    
    Retrieval Strategy:
    1. Analyze query semantics to understand needs
    2. Always get critical rules (Tier 1)
    3. Enhanced keyword matching with semantic expansion (Tier 2) - searches ALL fields
    4. Vector similarity if enabled (Tier 3) - lowered threshold for better recall
    5. Similar examples (Tier 4)
    
    Args:
        engine: Database engine
        question: User's question
        schema: Database schema (for intelligent analysis)
        enable_vector: Whether to use vector search (legacy param)
        enable_vector_search: Whether to use vector search
        similarity_threshold: Minimum similarity score (default 0.45)
        
    Returns:
        Dictionary with all retrieved context and statistics
    """
    
    print(f"[DEBUG RAG] get_relevant_context called with question: {question[:50]}...")
    print(f"[DEBUG RAG] enable_vector={enable_vector}, enable_vector_search={enable_vector_search}")
    
    # Support both param names for backwards compatibility
    use_vector = enable_vector and enable_vector_search
    
    all_rules = []
    excluded_ids = []
    
    # SEMANTIC ANALYSIS (if schema provided)
    enhanced_keywords = []
    if schema:
        try:
            from query_analyzer import get_enhanced_retrieval_keywords
            enhanced_keywords = get_enhanced_retrieval_keywords(question, schema)
        except ImportError:
            # query_analyzer not available, skip semantic enhancement
            pass
        except Exception:
            # Any other error, skip semantic enhancement
            pass
    
    # TIER 1: Critical rules (always included)
    try:
        critical_rules = get_critical_rules(engine)
        print(f"[DEBUG RAG] Tier 1 Critical rules: {len(critical_rules)}")
    except Exception as e:
        print(f"[DEBUG RAG] Tier 1 ERROR: {e}")
        critical_rules = []
    all_rules.extend(critical_rules)
    excluded_ids = [r["id"] for r in critical_rules]
    
    # TIER 2: Keyword matching (fast) - with enhanced keywords if available
    try:
        keyword_rules = get_keyword_matched_rules(engine, question, excluded_ids, enhanced_keywords if enhanced_keywords else None)
        print(f"[DEBUG RAG] Tier 2 Keyword rules: {len(keyword_rules)}")
    except Exception as e:
        print(f"[DEBUG RAG] Tier 2 ERROR: {e}")
        keyword_rules = []
    all_rules.extend(keyword_rules)
    excluded_ids.extend([r["id"] for r in keyword_rules])
    
    # TIER 3: Vector similarity (if enabled) - IMPROVED: lower threshold, more results
    vector_rules = []
    if use_vector:
        try:
            vector_rules = search_similar_rules(engine, question, 
                                               top_k=10, threshold=similarity_threshold,
                                               exclude_ids=excluded_ids)
            print(f"[DEBUG RAG] Tier 3 Vector rules: {len(vector_rules)}")
        except Exception as e:
            print(f"[DEBUG RAG] Tier 3 ERROR: {e}")
        all_rules.extend(vector_rules)
    
    print(f"[DEBUG RAG] Total rules found: {len(all_rules)}")
    
    # TIER 4: Similar examples
    examples = []
    if use_vector:
        examples = search_similar_examples(engine, question, 
                                          top_k=3, threshold=0.70)
    
    # Organize rules by type
    rules_by_type = {
        "critical": [r for r in all_rules if r["tier"] == "critical"],
        "metric": [r for r in all_rules if r["rule_type"] == "metric"],
        "join": [r for r in all_rules if r["rule_type"] == "join"],
        "filter": [r for r in all_rules if r["rule_type"] == "filter"],
        "transform": [r for r in all_rules if r["rule_type"] == "transform"],
        "default": [r for r in all_rules if r["rule_type"] == "default"],
        "mapping": [r for r in all_rules if r["rule_type"] == "mapping"],
        "other": [r for r in all_rules if r["rule_type"] not in 
                 ["metric", "join", "filter", "transform", "default", "mapping"]]
    }
    
    # Calculate statistics
    stats = {
        "total_rules": len(all_rules),
        "critical_rules": len(critical_rules),
        "keyword_rules": len(keyword_rules),
        "vector_rules": len(vector_rules),
        "total_examples": len(examples),
        "avg_rule_similarity": (
            sum(r.get("similarity", 0) for r in vector_rules) / len(vector_rules)
            if vector_rules else 0.0
        ),
        "avg_example_similarity": (
            sum(e["similarity"] for e in examples) / len(examples)
            if examples else 0.0
        ),
        "retrieval_breakdown": {
            "tier_1_critical": len(critical_rules),
            "tier_2_keyword": len(keyword_rules),
            "tier_3_vector": len(vector_rules),
            "tier_4_examples": len(examples)
        }
    }
    
    return {
        "rules": all_rules,
        "all_rules": all_rules,  # Alias for backward compatibility
        "rules_by_type": rules_by_type,
        "examples": examples,
        "stats": stats,
        "question": question,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# FORMAT CONTEXT FOR LLM (Plain Text)
# ============================================================================

def format_context_for_llm(context: Dict[str, Any]) -> str:
    """
    Format retrieved context into readable text for LLM
    
    Args:
        context: Context dictionary from get_relevant_context()
        
    Returns:
        Formatted string for LLM prompt
    """
    
    # Safety check for empty context
    if not context or "rules_by_type" not in context:
        return "No business rules retrieved."
    
    output = []
    rules_by_type = context.get("rules_by_type", {})
    
    # Critical Rules (always show first)
    if rules_by_type.get("critical"):
        output.append("⚠️  CRITICAL RULES (ALWAYS APPLY):")
        for rule in rules_by_type["critical"]:
            output.append(f"\n• {rule['rule_name']}")
            output.append(f"  {rule.get('description', '')}")
            if rule.get("rule_data"):
                data = rule["rule_data"]
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        pass
                # Extract key info from JSONB
                if isinstance(data, dict) and "condition" in data:
                    output.append(f"  Condition: {data['condition']}")
    
    # Metric Definitions
    if rules_by_type.get("metric"):
        output.append("\n\n📊 METRIC DEFINITIONS:")
        for rule in rules_by_type["metric"]:
            output.append(f"\n• {rule['rule_name']}")
            if rule.get("rule_data"):
                data = rule["rule_data"]
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        pass
                if isinstance(data, dict):
                    if "formula" in data:
                        output.append(f"  Formula: {data['formula']}")
                    if "user_terms" in data:
                        output.append(f"  User terms: {', '.join(data['user_terms'])}")
    
    # Join Relationships
    if rules_by_type.get("join"):
        output.append("\n\n🔗 JOIN RELATIONSHIPS:")
        for rule in rules_by_type["join"]:
            output.append(f"\n• {rule['rule_name']}")
            if rule.get("rule_data"):
                data = rule["rule_data"]
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        pass
                if isinstance(data, dict) and "join_condition" in data:
                    output.append(f"  {data['join_condition']}")
                    if "guidance" in data:
                        output.append(f"  Guidance: {data['guidance']}")
    
    # Filter Rules
    if rules_by_type.get("filter"):
        output.append("\n\n🎯 FILTER RULES:")
        for rule in rules_by_type["filter"]:
            output.append(f"\n• {rule['rule_name']}")
            if rule.get("rule_data"):
                data = rule["rule_data"]
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        pass
                if isinstance(data, dict) and "sql_pattern" in data:
                    output.append(f"  {data['sql_pattern']}")
    
    # Mappings & Transforms
    other_rules = (rules_by_type.get("mapping", []) + 
                   rules_by_type.get("transform", []) + 
                   rules_by_type.get("other", []))
    if other_rules:
        output.append("\n\n🔄 BUSINESS RULES:")
        for rule in other_rules:
            output.append(f"\n• {rule['rule_name']}")
            output.append(f"  {rule['description']}")
    
    # Similar Examples
    if context["examples"]:
        output.append("\n\n💡 SIMILAR QUERY EXAMPLES:")
        for i, ex in enumerate(context["examples"], 1):
            output.append(f"\nExample {i} (similarity: {ex['similarity']:.2f}):")
            output.append(f"Q: {ex['question']}")
            output.append(f"SQL: {ex['sql']}")
            if ex.get("explanation"):
                output.append(f"Explanation: {ex['explanation']}")
    
    return "\n".join(output)


# ============================================================================
# FORMAT CONTEXT FOR LLM (Structured JSON)
# ============================================================================

def format_context_as_json(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format context as structured JSON for LLM
    Better for precise parsing by reasoning LLM
    
    Returns:
        Structured dictionary that can be JSON-serialized
    """
    
    def parse_rule_data(rule):
        """Helper to parse JSONB rule_data"""
        data = rule.get("rule_data", {})
        if isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {}
        return data
    
    structured = {
        "critical_rules": [
            {
                "name": r["rule_name"],
                "description": r["description"],
                "data": parse_rule_data(r),
                "mandatory": True
            }
            for r in context["rules_by_type"]["critical"]
        ],
        "metrics": {
            r["rule_name"]: parse_rule_data(r)
            for r in context["rules_by_type"]["metric"]
        },
        "joins": [
            parse_rule_data(r)
            for r in context["rules_by_type"]["join"]
        ],
        "filters": {
            r["rule_name"]: parse_rule_data(r)
            for r in context["rules_by_type"]["filter"]
        },
        "transforms": [
            parse_rule_data(r)
            for r in context["rules_by_type"]["transform"]
        ],
        "examples": context["examples"],
        "metadata": {
            "question": context["question"],
            "stats": context["stats"]
        }
    }
    
    return structured


# ============================================================================
# QUERY LOGGING
# ============================================================================

def log_query_to_history(engine, question: str, generated_sql: str,
                         reasoning_llm: str, coding_llm: str,
                         reasoning_tokens: Dict[str, int], 
                         coding_tokens: Dict[str, int],
                         context: Dict[str, Any],
                         reasoning_output: str,
                         execution_success: bool = None,
                         execution_error: str = None,
                         rows_returned: int = None):
    """
    Log query execution to history for analysis
    """
    
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO query_history_v2
                (question, generated_sql, reasoning_llm, coding_llm,
                 reasoning_tokens_in, reasoning_tokens_out,
                 coding_tokens_in, coding_tokens_out,
                 rag_enabled, rules_retrieved, examples_retrieved,
                 avg_rule_similarity, avg_example_similarity,
                 retrieved_context, reasoning_output,
                 execution_success, execution_error, rows_returned)
                VALUES
                (:question, :sql, :reasoning_llm, :coding_llm,
                 :r_in, :r_out, :c_in, :c_out,
                 :rag_enabled, :rules_count, :examples_count,
                 :avg_rule_sim, :avg_example_sim,
                 :context::jsonb, :reasoning_output,
                 :exec_success, :exec_error, :rows_returned)
            """),
            {
                "question": question,
                "sql": generated_sql,
                "reasoning_llm": reasoning_llm,
                "coding_llm": coding_llm,
                "r_in": reasoning_tokens["input"],
                "r_out": reasoning_tokens["output"],
                "c_in": coding_tokens["input"],
                "c_out": coding_tokens["output"],
                "rag_enabled": context["stats"]["total_rules"] > 0,
                "rules_count": context["stats"]["total_rules"],
                "examples_count": context["stats"]["total_examples"],
                "avg_rule_sim": context["stats"]["avg_rule_similarity"],
                "avg_example_sim": context["stats"]["avg_example_similarity"],
                "context": json.dumps(context),
                "reasoning_output": reasoning_output,
                "exec_success": execution_success,
                "exec_error": execution_error,
                "rows_returned": rows_returned
            }
        )
        conn.commit()


# ============================================================================
# UTILITY: Calculate Current FY Dates
# ============================================================================

def get_current_fy_dates() -> Tuple[str, str]:
    """
    Calculate current financial year dates (April to March)
    
    Returns:
        (fy_start, fy_end) as ISO date strings
    """
    today = date.today()
    
    if today.month >= 4:  # April onwards
        fy_start = date(today.year, 4, 1)
        fy_end = date(today.year + 1, 3, 31)
    else:  # January-March
        fy_start = date(today.year - 1, 4, 1)
        fy_end = date(today.year, 3, 31)
    
    return fy_start.isoformat(), fy_end.isoformat()


if __name__ == "__main__":
    # Quick test
    print("\n" + "="*70)
    print("VECTOR UTILS V2 - Ready for use!")
    print("="*70)
    print("\nKey Functions:")
    print("• get_relevant_context(engine, question, enable_vector=True)")
    print("• format_context_for_llm(context)")
    print("• format_context_as_json(context)")
    print("• log_query_to_history(...)")
    print("\nMulti-Tier Retrieval:")
    print("  Tier 1: Critical rules (always)")
    print("  Tier 2: Keyword matching (fast)")
    print("  Tier 3: Vector similarity (semantic)")
    print("  Tier 4: Similar examples (few-shot)")
    print("="*70)