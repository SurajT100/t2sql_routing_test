"""
Query Classifier
=================
Classifies user questions as EASY, MEDIUM, HARD, or ANALYSIS using a single
LLM call (Claude Haiku — cheapest/fastest model).

When LLM classification is disabled via toggle, a simple keyword fallback
is used instead.  The keyword fallback is NEVER used as a pre-check or
bypass — it runs only when the LLM is unavailable or disabled.

Usage:
    from query_classifier import classify_query

    result = classify_query("Show total sales by region")
    # {"complexity": "medium", "reason": "Has aggregation", "tokens": {...}}
"""

from typing import Dict, Any
import json
import re


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
# KEYWORD FALLBACK (used only when LLM is disabled or unavailable)
# =============================================================================

def _keyword_fallback(question: str) -> tuple:
    """
    Minimal keyword classifier used ONLY as a last resort.
    Returns (complexity_str, confidence).
    """
    q = question.lower()

    analysis_words = ("what if", "simulate", "scenario", "impact of", "effect of",
                      "drill into", "what would happen", "if we raised", "if we removed")
    hard_words = ("compare", "versus", " vs ", "yoy", "year over year", "growth rate",
                  "percentage", "% of", "ranking", "running total", "excluding")
    medium_words = ("total", "sum", "count", "average", "avg", "group by", "grouped",
                    "by region", "by month", "by year", "top 10", "top 5", "highest", "lowest")

    if any(w in q for w in analysis_words):
        return "analysis", 0.7
    if any(w in q for w in hard_words):
        return "hard", 0.7
    if any(w in q for w in medium_words):
        return "medium", 0.8
    return "medium", 0.4  # safe default


# =============================================================================
# MAIN CLASSIFICATION FUNCTION
# =============================================================================

def classify_query(
    question: str,
    use_llm: bool = True,
    llm_provider: str = "claude_haiku",
) -> Dict[str, Any]:
    """
    Classify query complexity using a single LLM call (Claude Haiku).

    Args:
        question:     User's natural language question.
        use_llm:      When True (default), use the LLM. When False, use the
                      keyword fallback (e.g. when the toggle is off).
        llm_provider: LLM provider.  Defaults to claude_haiku (cheapest/fastest).

    Returns:
        {
            "complexity": "easy" | "medium" | "hard" | "analysis",
            "reason":     str,
            "confidence": float,
            "method":     "llm" | "keyword",
            "tokens":     {"input": int, "output": int},
            "prompt":     str,   # for debugging
            "response":   str,   # for debugging
        }
    """
    if not use_llm:
        complexity, confidence = _keyword_fallback(question)
        return {
            "complexity": complexity,
            "reason": f"Keyword-based classification (LLM disabled, confidence: {confidence:.0%})",
            "confidence": confidence,
            "method": "keyword",
            "tokens": {"input": 0, "output": 0},
            "prompt": "",
            "response": "",
        }

    prompt = CLASSIFIER_PROMPT.format(question=question)
    tokens: Dict[str, int] = {"input": 0, "output": 0}
    raw_response = ""

    try:
        from llm_v2 import call_llm

        raw_response, tokens = call_llm(prompt, llm_provider, stop_sequences=["}"])

        print(f"[CLASSIFIER] Raw LLM response ({llm_provider}): {raw_response!r}")

        # Strip markdown code fences if the model wrapped the JSON
        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text.strip()).strip()

        # stop_sequences=["}"] strips the closing brace — add it back
        if not text.endswith("}"):
            text += "}"

        # If there's extra text before the JSON object, extract just the object
        if not text.startswith("{"):
            m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if m:
                text = m.group(0)

        result = json.loads(text)
        complexity = result.get("complexity", "medium").lower()
        if complexity not in ("easy", "medium", "hard", "analysis"):
            complexity = "medium"

        return {
            "complexity": complexity,
            "reason": result.get("reason", "LLM classification"),
            "confidence": 0.9,
            "method": "llm",
            "tokens": tokens,
            "prompt": prompt,
            "response": raw_response,
        }

    except Exception as e:
        # LLM failed (network error, parse error, etc.) — use keyword fallback
        complexity, confidence = _keyword_fallback(question)
        return {
            "complexity": complexity,
            "reason": f"LLM failed ({type(e).__name__}) — keyword fallback used",
            "confidence": confidence,
            "method": "keyword_fallback",
            "tokens": tokens,
            "error": str(e),
            "prompt": prompt,
            "response": raw_response,
        }


# =============================================================================
# COMPLEXITY-BASED CONFIGURATION
# =============================================================================

def get_flow_config(complexity: str) -> Dict[str, Any]:
    """Get processing configuration based on complexity."""
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
            "description": "Direct to SQL LLM with minimal context",
        },
        "medium": {
            "use_reasoning_llm": True,
            "use_rule_rag": True,
            "use_schema_rag": True,
            "schema_rag_top_k": 15,
            "rule_rag_top_k": 5,
            "enable_opus": False,
            "max_retries": 2,
            "expected_tokens": 4000,
            "description": "Standard flow with reasoning",
        },
        "hard": {
            "use_reasoning_llm": True,
            "use_rule_rag": True,
            "use_schema_rag": True,
            "schema_rag_top_k": 20,
            "rule_rag_top_k": 8,
            "enable_opus": True,
            "max_retries": 3,
            "expected_tokens": 8000,
            "description": "Full flow with Opus validation",
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
            "description": "Multi-step Analyzer Agent with sub-query decomposition",
        },
    }
    return configs.get(complexity, configs["medium"])


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUERY CLASSIFIER TEST — keyword fallback")
    print("=" * 70)

    test_questions = [
        ("Show all customers", "easy"),
        ("List products where price > 100", "easy"),
        ("Total sales by region", "medium"),
        ("Top 10 products by quantity sold", "medium"),
        ("Compare Q3 vs Q4 revenue by region excluding cancelled orders", "hard"),
        ("Customers who ordered more this year than last year", "hard"),
        ("Simulate the effect of a 15% price increase on our top-selling SKUs", "analysis"),
    ]

    correct = 0
    for question, expected in test_questions:
        result = classify_query(question, use_llm=False)
        status = "✓" if result["complexity"] == expected else "✗"
        if result["complexity"] == expected:
            correct += 1
        print(f"{status} [{result['complexity']:8}] {question[:60]}")

    print(f"\nAccuracy: {correct}/{len(test_questions)} ({correct/len(test_questions)*100:.0f}%)")
