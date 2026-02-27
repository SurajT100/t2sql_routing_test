"""
Independent SQL Review System with Opus
========================================
Opus acts as independent validator with no prior bias.
Reviews only: Question + SQL + Results
"""

def create_opus_review_prompt(
    user_question: str,
    generated_sql: str,
    sql_results: any,
    schema: dict,
    sql_error: str = None
) -> str:
    """
    Create unbiased review prompt for Opus.
    CRITICAL: Does NOT include Sonnet's analysis to avoid confirmation bias.
    
    Args:
        user_question: Original user question
        generated_sql: SQL that was generated
        sql_results: Query results (or None if error)
        schema: Database schema for validation
        sql_error: Error message if SQL failed
    
    Returns:
        Review prompt for Opus
    """
    
    if sql_error:
        # SQL FAILED - Quick review
        prompt = f"""You are a senior database expert reviewing failed SQL.

USER QUESTION: {user_question}

GENERATED SQL:
```sql
{generated_sql}
```

ERROR MESSAGE:
{sql_error}

DATABASE SCHEMA:
{schema}

TASK: Determine if the SQL is fundamentally correct but has a minor syntax issue, or if it's logically wrong.

OUTPUT JSON:
{{
  "verdict": "INCORRECT",
  
  "confidence": 0.95,
  
  "error_type": "syntax|logic|schema|other",
  
  "issues": [
    "Specific problem found"
  ],
  
  "reasoning": "Brief explanation (2-3 sentences)",
  
  "guidance_for_regeneration": "What needs to change"
}}
"""
    
    else:
        # SQL SUCCEEDED - Deep review
        if hasattr(sql_results, 'head'):
            results_preview = sql_results.head(20).to_string()
            row_count = len(sql_results)
            column_names = list(sql_results.columns)
        else:
            results_preview = str(sql_results)[:2000]
            row_count = "unknown"
            column_names = []
        
        prompt = f"""You are a senior database expert with NO prior context. Your job is to independently verify if SQL correctly answers a question.

USER QUESTION: {user_question}

GENERATED SQL:
```sql
{generated_sql}
```

RESULTS PREVIEW (first 20 of {row_count} rows):
Columns: {column_names}

{results_preview}

DATABASE SCHEMA (for validation):
{schema}

TASK: Independently assess if this SQL correctly and completely answers the user's question.

REVIEW CHECKLIST:
1. Does SQL logic match what the question asks?
2. Are correct tables and columns used?
3. Are filters/aggregations appropriate?
4. Does result structure answer the question?
5. Are there any missing conditions or wrong calculations?

VERDICT OPTIONS:
- CORRECT: SQL perfectly answers the question
- INCORRECT: SQL has clear errors or doesn't answer question
- UNCERTAIN: Edge case or ambiguous, may need human review

OUTPUT JSON:
{{
  "verdict": "CORRECT|INCORRECT|UNCERTAIN",
  
  "confidence": 0.0-1.0,
  
  "issues": [
    "If INCORRECT or UNCERTAIN: specific problems"
  ],
  
  "reasoning": "Why you gave this verdict (2-3 sentences max)",
  
  "guidance_for_regeneration": "If INCORRECT: what needs to change"
}}

CRITICAL: Be strict. If something seems off, mark as INCORRECT or UNCERTAIN.
Better to reject a query than approve wrong data.
"""
    
    return prompt


def call_opus_reviewer(
    user_question: str,
    generated_sql: str,
    sql_results: any,
    schema: dict,
    sql_error: str = None
) -> dict:
    """
    Call Opus for independent SQL review.
    
    Returns:
        {
            "verdict": "CORRECT|INCORRECT|UNCERTAIN",
            "confidence": float,
            "issues": list,
            "reasoning": str,
            "guidance": str,
            "tokens": dict
        }
    """
    from llm_v2 import call_llm
    import json
    
    prompt = create_opus_review_prompt(
        user_question,
        generated_sql,
        sql_results,
        schema,
        sql_error
    )
    
    # Use Opus with prefill for clean JSON
    response, tokens = call_llm(
        prompt,
        "claude_opus",  # Will need to add this provider
        prefill="{"
    )
    
    try:
        review_data = json.loads(response)
        review_data["tokens"] = tokens
        return review_data
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "issues": ["Failed to parse Opus response"],
            "reasoning": "Review response was not valid JSON",
            "guidance": response,
            "tokens": tokens
        }


def refinement_loop(
    user_question: str,
    schema: dict,
    business_context: str,
    max_retries: int = 3
) -> dict:
    """
    Complete refinement loop: Sonnet → Qwen → Opus → Retry if needed
    
    Returns:
        {
            "sql": str,
            "results": DataFrame,
            "verdict": str,
            "attempts": int,
            "reviews": list,
            "final_review": dict
        }
    """
    from llm_v2 import call_llm
    from db import run_sql
    import json
    
    attempts = []
    
    for attempt in range(1, max_retries + 1):
        # PHASE 1: Sonnet plans
        if attempt == 1:
            # First attempt - fresh analysis
            reasoning_prompt = create_reasoning_prompt(
                user_question, schema, business_context
            )
        else:
            # Retry - incorporate Opus feedback
            reasoning_prompt = create_refinement_prompt(
                user_question,
                schema,
                business_context,
                previous_attempt=attempts[-1]
            )
        
        sonnet_response, sonnet_tokens = call_llm(
            reasoning_prompt,
            "claude_sonnet",
            prefill="{"
        )
        
        # PHASE 2: Qwen executes
        sql_prompt = create_sql_prompt(sonnet_response, schema)
        qwen_response, qwen_tokens = call_llm(
            sql_prompt,
            "vertex_qwen"
        )
        
        sql = clean_sql(qwen_response)
        
        # PHASE 3: Execute SQL
        try:
            results = run_sql(engine, sql)
            sql_error = None
        except Exception as e:
            results = None
            sql_error = str(e)
        
        # PHASE 4: Opus reviews
        opus_review = call_opus_reviewer(
            user_question,
            sql,
            results,
            schema,
            sql_error
        )
        
        attempts.append({
            "attempt": attempt,
            "sonnet_analysis": sonnet_response,
            "sql": sql,
            "results": results,
            "error": sql_error,
            "opus_review": opus_review
        })
        
        # Check verdict
        if opus_review["verdict"] == "CORRECT":
            return {
                "sql": sql,
                "results": results,
                "verdict": "CORRECT",
                "attempts": attempt,
                "reviews": attempts,
                "final_review": opus_review
            }
        
        elif opus_review["verdict"] == "UNCERTAIN":
            # Flag but return results
            return {
                "sql": sql,
                "results": results,
                "verdict": "UNCERTAIN",
                "attempts": attempt,
                "reviews": attempts,
                "final_review": opus_review,
                "warning": "Results may be inaccurate - manual review recommended"
            }
        
        # INCORRECT - continue loop if retries left
        if attempt == max_retries:
            return {
                "sql": sql,
                "results": results,
                "verdict": "FAILED_AFTER_RETRIES",
                "attempts": attempt,
                "reviews": attempts,
                "final_review": opus_review,
                "error": "Could not generate correct SQL after 3 attempts"
            }
    
    # Should never reach here
    return None


if __name__ == "__main__":
    print("Independent Opus Review System")
    print("=" * 60)
    print("\nFeatures:")
    print("  ✓ Unbiased independent review")
    print("  ✓ No confirmation bias")
    print("  ✓ Structured feedback")
    print("  ✓ Automatic retry loop")
    print("  ✓ Human escalation after 3 failures")
