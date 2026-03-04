"""
Query Classifier
=================
Classifies user questions as EASY, MEDIUM, HARD, or ANALYSIS using Llama via Groq.
This enables tiered processing to optimize tokens and accuracy.

Usage:
    from query_classifier import classify_query, QueryComplexity
    
    result = classify_query("Show total sales by region")
    # {"complexity": "medium", "reason": "Has aggregation", "tokens": {...}}
"""

from enum import Enum
from typing import Dict, Any
import json
import re


class QueryComplexity(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ANALYSIS = "analysis"


# =============================================================================
# CLASSIFICATION PROMPT
# =============================================================================

CLASSIFIER_PROMPT = """You are a SQL query complexity classifier. Analyze the question and classify it.

CLASSIFICATION RULES:

**EASY** - Direct data retrieval:
• Single table query (no JOINs mentioned or implied)
• No aggregations (SUM, COUNT, AVG, MAX, MIN, GROUP BY)
• No business terms needing interpretation (like "sales", "revenue", "margin")
• Simple filters only (equals, greater/less than, IN, LIKE)
• No date calculations (quarters, fiscal years, YTD, YoY)
• No sorting requirements mentioned
Examples:
- "Show all customers"
- "List products where price > 100"
- "Get orders from January"
- "Show me the customers table"
- "What are the column names in orders?"

**MEDIUM** - Standard analytics:
• Has ONE of these complexities:
  - Aggregation (SUM, COUNT, AVG, total, average, count of)
  - Simple business terms (sales, revenue, margin, profit)
  - Date filtering (last month, Q3, this year, recent)
  - Simple JOIN (between 2 tables)
  - GROUP BY with one dimension
  - Basic sorting (top, bottom, highest, lowest)
Examples:
- "Total sales by region"
- "Count of customers per city"
- "Orders from Q3 2024"
- "Top 10 products by quantity"
- "Average order value"

**HARD** - Complex analytics:
• Has TWO OR MORE of these:
  - Multiple aggregations
  - Multiple business terms
  - Complex JOINs (3+ tables)
  - Complex date logic (comparisons, YoY, growth rates)
  - Business rule exclusions (exclude cancelled orders, filter out returns)
  - Comparative analysis (vs, compare, growth, change, difference)
  - Ranking with conditions
  - Subqueries implied
  - Percentage calculations
Examples:
- "Compare Q3 vs Q4 revenue by region excluding cancelled orders"
- "Top 10 customers by margin with YoY growth"
- "Monthly sales trend with running total"
- "Revenue by product category as percentage of total"
- "Customers who ordered more this year than last year"

**ANALYSIS** - Multi-step reasoning required (cannot be answered in a single SQL query):
• Requires two or more SEPARATE queries whose results must be combined
• What-if / scenario simulation (e.g. "what would revenue be if we raised prices by 10%?")
• Impact analysis that needs a baseline query then an adjustment query
• Trend analysis where each period must be fetched independently before comparing
• Comparative ranking where rank depends on a metric computed in a prior step
• Any question that explicitly asks for derived metrics across multiple independent datasets
• Multi-metric dashboards where each metric needs its own query
Use ANALYSIS only when a single SQL query (even with CTEs) is genuinely insufficient — the answer requires running multiple queries and combining their outputs programmatically.
Examples:
- "What would happen to our margin if we removed the top 5 loss-making products?"
- "Which customers improved their order frequency compared to the same period last year, and what is the revenue impact?"
- "Show me the top 3 regions by growth rate and then drill into each region's product mix"
- "Simulate the effect of a 15% price increase on our top-selling SKUs"

USER QUESTION: {question}

Respond with ONLY this JSON (no other text):
{{"complexity": "easy|medium|hard|analysis", "reason": "brief explanation"}}"""


# =============================================================================
# KEYWORD-BASED PRE-CLASSIFICATION
# =============================================================================

# Keywords that suggest complexity levels
EASY_INDICATORS = [
    "show all", "list all", "get all", "display all",
    "show me the", "what is the", "what are the",
    "column names", "table structure", "schema",
]

MEDIUM_INDICATORS = [
    # Aggregations
    "total", "sum", "count", "average", "avg", "mean",
    "maximum", "minimum", "max", "min",
    # Grouping
    "by region", "by city", "by product", "by customer", "by month", "by year",
    "per region", "per city", "per product", "per customer", "per month",
    "group by", "grouped",
    # Simple date
    "last month", "this month", "this year", "last year",
    "q1", "q2", "q3", "q4", "quarter",
    "recent", "latest",
    # Simple sorting
    "top 10", "top 5", "top 20", "bottom 10",
    "highest", "lowest", "best", "worst",
]

HARD_INDICATORS = [
    # Comparisons
    "compare", "versus", "vs", "comparison",
    "growth", "change", "difference", "delta",
    "yoy", "year over year", "mom", "month over month",
    "increase", "decrease", "trend",
    # Complex calculations
    "percentage", "percent of", "% of", "ratio",
    "running total", "cumulative", "rolling",
    "rank", "ranking", "percentile",
    # Exclusions/conditions
    "excluding", "exclude", "without", "except",
    "only where", "filtered by",
    # Multiple dimensions
    "by region and", "by product and", "by customer and",
    # Subqueries
    "who have", "that have", "customers who", "products that",
    "more than average", "above average", "below average",
]

ANALYSIS_INDICATORS = [
    # What-if / simulation
    "what if", "what would happen", "simulate", "simulation",
    "if we", "if prices", "if we raised", "if we removed",
    "scenario", "hypothetical",
    # Impact analysis
    "impact of", "effect of", "impact on", "effect on",
    "revenue impact", "margin impact",
    # Multi-step reasoning
    "and then", "followed by", "based on those", "using those results",
    "drill into", "drill down",
    # Improvement / change detection across independent datasets
    "improved their", "who improved", "who changed",
    "compared to same period", "compared to last year",
]


def _quick_classify(question: str) -> tuple:
    """
    Quick keyword-based classification before calling LLM.
    Returns (suggested_complexity, confidence)
    """
    q_lower = question.lower()

    analysis_score = sum(1 for ind in ANALYSIS_INDICATORS if ind in q_lower)
    hard_score = sum(1 for ind in HARD_INDICATORS if ind in q_lower)
    medium_score = sum(1 for ind in MEDIUM_INDICATORS if ind in q_lower)
    easy_score = sum(1 for ind in EASY_INDICATORS if ind in q_lower)

    # Strong analysis signal — multiple indicators or a single very strong one
    if analysis_score >= 2:
        return QueryComplexity.ANALYSIS, 0.85
    if analysis_score >= 1 and hard_score >= 1:
        return QueryComplexity.ANALYSIS, 0.7

    # If multiple hard indicators, definitely hard
    if hard_score >= 2:
        return QueryComplexity.HARD, 0.9

    # If one hard indicator and some medium, likely hard
    if hard_score >= 1 and medium_score >= 1:
        return QueryComplexity.HARD, 0.7

    # If medium indicators present
    if medium_score >= 1:
        return QueryComplexity.MEDIUM, 0.8

    # If easy indicators or very short/simple question
    if easy_score >= 1 or len(question.split()) <= 5:
        return QueryComplexity.EASY, 0.6

    # Default to medium with low confidence (will use LLM)
    return QueryComplexity.MEDIUM, 0.3


# =============================================================================
# MAIN CLASSIFICATION FUNCTION
# =============================================================================

def classify_query(
    question: str,
    use_llm: bool = True,
    llm_provider: str = "groq"
) -> Dict[str, Any]:
    """
    Classify query complexity.
    
    Args:
        question: User's natural language question
        use_llm: Whether to use LLM for classification (if False, uses keywords only)
        llm_provider: LLM provider to use for classification
    
    Returns:
        {
            "complexity": "easy" | "medium" | "hard",
            "reason": str,
            "confidence": float,
            "method": "keyword" | "llm",
            "tokens": {"input": int, "output": int},
            "prompt": str,  # For debugging
            "response": str  # For debugging
        }
    """
    # Quick keyword-based classification
    quick_result, confidence = _quick_classify(question)
    
    # If high confidence or LLM disabled, return keyword result
    if not use_llm or confidence >= 0.85:
        return {
            "complexity": quick_result.value,
            "reason": f"Keyword-based classification (confidence: {confidence:.0%})",
            "confidence": confidence,
            "method": "keyword",
            "tokens": {"input": 0, "output": 0},
            "prompt": "",
            "response": ""
        }
    
    # Use LLM for uncertain cases
    try:
        from llm_v2 import call_llm
        
        prompt = CLASSIFIER_PROMPT.format(question=question)
        
        response, tokens = call_llm(
            prompt,
            llm_provider,
            stop_sequences=["}"]
        )
        
        # Ensure JSON is complete
        raw_response = response
        response = response.strip()
        if not response.endswith("}"):
            response += "}"
        
        # Parse response
        result = json.loads(response)
        
        complexity = result.get("complexity", "medium").lower()
        if complexity not in ["easy", "medium", "hard", "analysis"]:
            complexity = "medium"
        
        return {
            "complexity": complexity,
            "reason": result.get("reason", "LLM classification"),
            "confidence": 0.9,  # LLM results are generally reliable
            "method": "llm",
            "tokens": tokens,
            "prompt": prompt,
            "response": raw_response
        }
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, fall back to keyword result but keep trace
        return {
            "complexity": quick_result.value,
            "reason": f"LLM response parsing failed, using keyword fallback",
            "confidence": confidence,
            "method": "keyword_fallback",
            "tokens": tokens if 'tokens' in locals() else {"input": 0, "output": 0},
            "error": str(e),
            "prompt": prompt if 'prompt' in locals() else "",
            "response": raw_response if 'raw_response' in locals() else (response if 'response' in locals() else "")
        }
        
    except Exception as e:
        # If LLM call fails, fall back to keyword result
        return {
            "complexity": quick_result.value,
            "reason": f"LLM call failed, using keyword fallback",
            "confidence": confidence,
            "method": "keyword_fallback",
            "tokens": {"input": 0, "output": 0},
            "error": str(e),
            "prompt": "",
            "response": ""
        }


def classify_query_batch(
    questions: list,
    use_llm: bool = True
) -> list:
    """
    Classify multiple queries.
    
    Args:
        questions: List of questions
        use_llm: Whether to use LLM
    
    Returns:
        List of classification results
    """
    return [classify_query(q, use_llm=use_llm) for q in questions]


# =============================================================================
# COMPLEXITY-BASED CONFIGURATION
# =============================================================================

def get_flow_config(complexity: str) -> Dict[str, Any]:
    """
    Get processing configuration based on complexity.
    
    Args:
        complexity: "easy", "medium", or "hard"
    
    Returns:
        Configuration dict for query processing
    """
    configs = {
        "easy": {
            "use_reasoning_llm": False,
            "use_rule_rag": True,
            "use_schema_rag": True,
            "schema_rag_top_k": 10,
            "rule_rag_top_k": 3,
            "enable_opus": False,
            "max_retries": 1,
            "expected_tokens": 1500,
            "description": "Direct to SQL LLM with minimal context"
        },
        "medium": {
            "use_reasoning_llm": True,
            "use_rule_rag": True,
            "use_schema_rag": True,
            "schema_rag_top_k": 15,
            "rule_rag_top_k": 5,
            "enable_opus": False,  # Optional
            "max_retries": 2,
            "expected_tokens": 4000,
            "description": "Standard flow with reasoning"
        },
        "hard": {
            "use_reasoning_llm": True,
            "use_rule_rag": True,
            "use_schema_rag": True,
            "schema_rag_top_k": 20,
            "rule_rag_top_k": 8,
            "enable_opus": True,  # Recommended
            "max_retries": 3,
            "expected_tokens": 8000,
            "description": "Full flow with Opus validation"
        },
        "analysis": {
            "use_reasoning_llm": True,
            "use_rule_rag": True,
            "use_schema_rag": True,
            "schema_rag_top_k": 20,
            "rule_rag_top_k": 8,
            "enable_opus": True,
            "max_retries": 3,
            "expected_tokens": 20000,
            "description": "Multi-step Analyzer Agent with sub-query decomposition"
        }
    }

    return configs.get(complexity, configs["medium"])


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUERY CLASSIFIER TEST")
    print("=" * 70)
    
    test_questions = [
        # Easy
        ("Show all customers", "easy"),
        ("List products where price > 100", "easy"),
        ("What are the columns in orders table?", "easy"),
        
        # Medium
        ("Total sales by region", "medium"),
        ("Count of customers per city", "medium"),
        ("Top 10 products by quantity sold", "medium"),
        ("Orders from Q3 2024", "medium"),
        
        # Hard
        ("Compare Q3 vs Q4 revenue by region excluding cancelled orders", "hard"),
        ("Top 10 customers by margin with YoY growth", "hard"),
        ("Revenue by product as percentage of total", "hard"),
        ("Customers who ordered more this year than last year", "hard"),
    ]
    
    print("\nKeyword-based classification (no LLM):")
    print("-" * 60)
    
    correct = 0
    for question, expected in test_questions:
        result = classify_query(question, use_llm=False)
        status = "✓" if result["complexity"] == expected else "✗"
        if result["complexity"] == expected:
            correct += 1
        print(f"{status} [{result['complexity']:6}] {question[:50]}")
    
    print(f"\nAccuracy: {correct}/{len(test_questions)} ({correct/len(test_questions)*100:.0f}%)")
    
    print("\n" + "=" * 70)
    print("FLOW CONFIGURATIONS")
    print("=" * 70)
    
    for complexity in ["easy", "medium", "hard"]:
        config = get_flow_config(complexity)
        print(f"\n{complexity.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Reasoning LLM: {config['use_reasoning_llm']}")
        print(f"  Opus Review: {config['enable_opus']}")
        print(f"  Expected tokens: ~{config['expected_tokens']}")
    
    print("\n" + "=" * 70)