"""
Prompt Templates for Reasoning → SQL LLM Chain
===============================================
These templates ensure consistent, structured communication
between reasoning and SQL generation LLMs.
"""

import json
from typing import Dict, Any
from datetime import date


def get_reasoning_prompt(question: str, schema: Dict, context: Dict, 
                        use_json: bool = True) -> str:
    """
    Generate prompt for reasoning LLM
    
    Args:
        question: User's natural language question
        schema: Database schema dictionary
        context: Retrieved RAG context
        use_json: Whether to request JSON output (recommended)
    
    Returns:
        Formatted prompt string
    """
    
    if use_json:
        return _get_json_reasoning_prompt(question, schema, context)
    else:
        return _get_text_reasoning_prompt(question, schema, context)


def _get_json_reasoning_prompt(question: str, schema: Dict, context: Dict) -> str:
    """
    JSON-based reasoning prompt - SEMANTIC FOCUS
    Includes confidence assessment and clarifying questions.
    """
    
    from vector_utils_v2 import format_context_for_llm
    formatted_context = format_context_for_llm(context)
    
    today = date.today()
    
    prompt = f"""You are a semantic query interpreter for a database system.

QUESTION: {question}

SCHEMA:
{json.dumps(schema, indent=2)}

BUSINESS RULES:
{formatted_context}

TODAY: {today.isoformat()}

TASK: Understand the question and translate business terms using the rules provided.

INSTRUCTIONS:
1. Check business rules for ALL term definitions before interpreting
2. Business rules override any general knowledge
3. Assess your confidence based on available information

CONFIDENCE LEVELS:
- "high": All terms defined in business rules, question is clear
- "medium": Some assumptions made but reasonable, proceed with warnings
- "low": Critical terms undefined or question is ambiguous, need clarification

OUTPUT JSON (no markdown):
{{
  "confidence": "high|medium|low",
  
  "clarifying_questions": [
    "Only if confidence is low - questions to ask user"
  ],
  
  "warnings": [
    "Only if confidence is medium - assumptions made"
  ],
  
  "analysis": {{
    "intent": "What user wants in one sentence",
    
    "data_need": {{
      "find": "What entities to retrieve",
      "measure": "What to calculate/aggregate",
      "condition": "Filtering condition in plain English"
    }},
    
    "term_mappings": {{
      "business_term_from_question": "exact_value_from_business_rules"
    }},
    
    "tables": ["schema.table"],
    
    "data_relationship": "How tables relate to answer this question",
    
    "mandatory_rules": [
      "Each mandatory filter/rule from business rules that applies"
    ],
    
    "output": {{
      "show": ["columns to display"],
      "transform": "unit conversion if specified in rules",
      "sort": "ordering if specified"
    }}
  }}
}}

WHEN TO USE EACH CONFIDENCE LEVEL:
- high: Every term in question has a definition in business rules
- medium: Made reasonable assumptions (e.g., defaulted to current year), can proceed
- low: Critical term undefined (e.g., "Q3" not in rules), ambiguous metric, unclear table relationships

If confidence is low, provide clarifying_questions and minimal analysis.
If confidence is medium or high, provide full analysis."""

    return prompt


def _get_text_reasoning_prompt(question: str, schema: Dict, context: Dict) -> str:
    """
    Text-based reasoning prompt (FALLBACK)
    """
    
    from vector_utils_v2 import format_context_for_llm
    formatted_context = format_context_for_llm(context)
    
    prompt = f"""You are a business-to-technical query translator.

SCHEMA:
{json.dumps(schema, indent=2)}

BUSINESS RULES:
{formatted_context}

QUESTION: {question}

TASK: Translate business terms using the rules provided. Do NOT write SQL.

OUTPUT:

CONFIDENCE: [high/medium/low]

CLARIFYING QUESTIONS (if low confidence):
[Questions to ask user]

WARNINGS (if medium confidence):
[Assumptions made]

INTENT:
[What user wants]

TERM TRANSLATIONS:
[Business term → value from business rules]

TABLES NEEDED:
[Tables required]

DATA RELATIONSHIP:
[How tables connect]

MANDATORY RULES TO APPLY:
[Filters/rules from business rules]

OUTPUT FORMAT:
[Columns, transforms, sorting]"""

    return prompt


def get_sql_generation_prompt(reasoning_output: str, schema: Dict = None, use_json_input: bool = True) -> str:
    """
    Generate prompt for SQL coding LLM
    
    Args:
        reasoning_output: Output from reasoning LLM
        schema: Full database schema (optional but recommended)
        use_json_input: Whether reasoning output is JSON
    
    Returns:
        SQL generation prompt
    """
    
    schema_section = ""
    if schema:
        schema_section = f"""
DATABASE SCHEMA (ALL Available Tables & Columns):
{json.dumps(schema, indent=2)}

IMPORTANT: Use ONLY the exact table and column names from the schema above.
Case-sensitive. If a column is not in schema, DO NOT use it.
"""
    
    if use_json_input:
        prompt = f"""You are a PostgreSQL expert code generator.

QUERY ANALYSIS FROM BUSINESS ANALYST:
{reasoning_output}
{schema_section}
YOUR TASK:
Generate ONLY the SQL query. The analysis above tells you what to select, filter, join, and calculate.

CRITICAL POSTGRESQL REQUIREMENTS:
1. **IDENTIFIER QUOTING** - ALWAYS double-quote ALL identifiers:
   ✓ CORRECT: SELECT "XYZ", "PQR" FROM "schema"."table"
   ✗ WRONG: SELECT XYZ, PQR FROM schema.table

2. **String VALUES** - Use single quotes:
   ✓ CORRECT: WHERE "abc" = 'john'
   ✗ WRONG: WHERE "abc" = "john"

3. **Schema-qualified tables**:
   ✓ CORRECT: FROM "schema"."table"
   ✗ WRONG: FROM schema.table

4. **APPLY ALL mandatory_rules** from the analysis
5. **USE term_mappings** for filter values (e.g., if analysis says "Bangalore", use "Bangalore")
6. **EXACT columns** - Use column names exactly as shown in schema or analysis

EXAMPLES OF CORRECT SYNTAX:
```sql
SELECT "schema"."table"."Region", SUM("schema"."table"."Margin") / 100000 AS "sales_lakhs"
FROM "schema"."table"
WHERE "schema"."table"."Month" IN ('2024-10', '2024-11', '2024-12')
  AND "schema"."table"."U_Ordertype" <> 'Rebate'
GROUP BY "schema"."table"."Region"
ORDER BY "sales_lakhs" DESC;
```

OUTPUT: SQL query only. No explanation, no markdown, no comments."""
    
    else:
        prompt = f"""You are a PostgreSQL expert.

QUERY ANALYSIS:
{reasoning_output}
{schema_section}
REQUIREMENTS:
- PostgreSQL dialect
- Double-quote all identifiers
- Apply all mandatory rules mentioned
- Use translated terms for filter values

OUTPUT: SQL query only. No explanation, no markdown."""
    
    return prompt


# ============================================================================
# CONFIDENCE HANDLING
# ============================================================================

def process_reasoning_output(reasoning_json: Dict) -> Dict[str, Any]:
    """
    Process reasoning output and determine next action based on confidence.
    
    Returns:
        {
            "action": "proceed" | "warn_and_proceed" | "ask_user",
            "analysis": dict or None,
            "warnings": list or None,
            "questions": list or None
        }
    """
    confidence = reasoning_json.get("confidence", "medium")
    
    if confidence == "high":
        return {
            "action": "proceed",
            "analysis": reasoning_json.get("analysis"),
            "warnings": None,
            "questions": None
        }
    
    elif confidence == "medium":
        return {
            "action": "warn_and_proceed",
            "analysis": reasoning_json.get("analysis"),
            "warnings": reasoning_json.get("warnings", []),
            "questions": None
        }
    
    else:  # low
        return {
            "action": "ask_user",
            "analysis": reasoning_json.get("analysis"),  # May be partial
            "warnings": None,
            "questions": reasoning_json.get("clarifying_questions", [])
        }


# ============================================================================
# VALIDATION & TESTING
# ============================================================================

def validate_reasoning_json(json_str: str) -> Dict[str, Any]:
    """
    Validate and parse reasoning LLM JSON output
    
    Returns:
        Parsed JSON dict or raises ValueError
    """
    try:
        # Clean potential markdown wrapper
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        data = json.loads(json_str)
        
        # Validate required fields
        required = ["confidence"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # If not low confidence, analysis is required
        if data.get("confidence") != "low":
            if "analysis" not in data:
                raise ValueError("Analysis required for medium/high confidence")
        
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from reasoning LLM: {e}")


def clean_sql_output(sql: str) -> str:
    """
    Clean SQL output from coding LLM
    Removes markdown, comments, extra whitespace
    """
    # Remove markdown code blocks
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    # Remove SQL comments
    lines = sql.split("\n")
    cleaned_lines = []
    for line in lines:
        # Remove -- comments
        if "--" in line:
            line = line.split("--")[0]
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    sql = " ".join(cleaned_lines)
    
    # Remove extra whitespace
    import re
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    return sql


if __name__ == "__main__":
    print("Prompt Templates V2 - Ready!")
    print("\nKey Functions:")
    print("  • get_reasoning_prompt(question, schema, context)")
    print("  • get_sql_generation_prompt(reasoning_output)")
    print("  • validate_reasoning_json(json_str)")
    print("  • process_reasoning_output(reasoning_json)")
    print("  • clean_sql_output(sql)")
    print("\nConfidence Levels:")
    print("  • high   → proceed to SQL generation")
    print("  • medium → warn user, proceed to SQL generation")
    print("  • low    → ask clarifying questions, no SQL")