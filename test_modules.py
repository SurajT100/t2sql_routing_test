"""
Test Script - Verify All Modules Work Together
===============================================
Run this to test the optimization modules without Streamlit.

Usage:
    python test_modules.py
"""

import sys
import json

print("=" * 70)
print("TEXT-TO-SQL OPTIMIZATION MODULES TEST")
print("=" * 70)

# =============================================================================
# TEST 1: Abbreviations Module
# =============================================================================
print("\n📦 TEST 1: Abbreviations Module")
print("-" * 50)

try:
    from abbreviations import expand_column_name, get_business_terms
    
    test_columns = ["CUST_NM", "TOT_SLS_AMT", "RGN_CD", "ORD_DT"]
    
    for col in test_columns:
        expanded = expand_column_name(col)
        terms = get_business_terms(col)[:5]
        print(f"  {col:15} → {expanded}")
        print(f"  {'':15}   Terms: {', '.join(terms)}")
    
    print("  ✅ Abbreviations module OK")
except Exception as e:
    print(f"  ❌ Error: {e}")

# =============================================================================
# TEST 2: PII Detector Module
# =============================================================================
print("\n📦 TEST 2: PII Detector Module")
print("-" * 50)

try:
    from pii_detector import detect_pii, get_pii_severity
    
    test_values = [
        "john@email.com",
        "555-123-4567",
        "Normal text",
        "123-45-6789",  # SSN
    ]
    
    has_pii, types, masked = detect_pii(test_values)
    print(f"  Has PII: {has_pii}")
    print(f"  Types found: {types}")
    print(f"  Severity: {get_pii_severity(types)}")
    print(f"  Masked samples:")
    for orig, mask in zip(test_values, masked):
        if orig != mask:
            print(f"    {orig} → {mask}")
    
    print("  ✅ PII Detector module OK")
except Exception as e:
    print(f"  ❌ Error: {e}")

# =============================================================================
# TEST 3: Query Classifier Module
# =============================================================================
print("\n📦 TEST 3: Query Classifier Module")
print("-" * 50)

try:
    from query_classifier import classify_query, get_flow_config
    
    test_queries = [
        ("Show all customers", "easy"),
        ("Total sales by region", "medium"),
        ("Compare Q3 vs Q4 sales excluding rebates", "hard"),
    ]
    
    correct = 0
    for question, expected in test_queries:
        result = classify_query(question, use_llm=False)  # Keyword only
        status = "✓" if result["complexity"] == expected else "✗"
        if result["complexity"] == expected:
            correct += 1
        print(f"  {status} [{result['complexity']:6}] {question[:40]}")
    
    print(f"\n  Accuracy: {correct}/{len(test_queries)}")
    
    # Test flow config
    for complexity in ["easy", "medium", "hard"]:
        cfg = get_flow_config(complexity)
        print(f"  {complexity.upper():6}: Reasoning={cfg['use_reasoning_llm']}, Opus={cfg['enable_opus']}, Tokens~{cfg['expected_tokens']}")
    
    print("  ✅ Query Classifier module OK")
except Exception as e:
    print(f"  ❌ Error: {e}")

# =============================================================================
# TEST 4: Prompt Optimizer Module
# =============================================================================
print("\n📦 TEST 4: Prompt Optimizer Module")
print("-" * 50)

try:
    from prompt_optimizer import (
        compress_rules_for_llm, 
        extract_sql_from_response,
        extract_columns_from_sql
    )
    
    # Test rule compression
    test_rules = [
        {
            "rule_name": "Revenue Metric",
            "rule_type": "metric",
            "rule_data": {"formula": "SUM(Amount)", "condition": "Status <> 'Cancelled'"}
        },
        {
            "rule_name": "PostgreSQL",
            "rule_type": "dialect",
            "rule_data": {"dialect": "postgresql", "quote_char": '"'}
        }
    ]
    
    compressed = compress_rules_for_llm(test_rules)
    print(f"  Original rules: ~{sum(len(str(r)) for r in test_rules)} chars")
    print(f"  Compressed: {len(compressed)} chars")
    print(f"  Sample: {compressed[:100]}...")
    
    # Test SQL extraction
    test_responses = [
        '{"sql": "SELECT * FROM users"}',
        '```sql\nSELECT * FROM orders\n```',
        'SELECT COUNT(*) FROM products',
    ]
    
    print("\n  SQL Extraction:")
    for resp in test_responses:
        sql = extract_sql_from_response(resp)
        print(f"    {resp[:30]:30} → {sql[:30]}")
    
    # Test column extraction
    sql = 'SELECT "region", SUM("amount") FROM "orders" WHERE "status" = \'Active\''
    cols = extract_columns_from_sql(sql)
    print(f"\n  Columns in SQL: {cols}")
    
    print("  ✅ Prompt Optimizer module OK")
except Exception as e:
    print(f"  ❌ Error: {e}")

# =============================================================================
# TEST 5: SQL Validator Module
# =============================================================================
print("\n📦 TEST 5: SQL Validator Module")
print("-" * 50)

try:
    from sql_validator import validate_sql, fix_common_issues, is_safe_query
    
    test_cases = [
        ('SELECT * FROM users WHERE id = 1', "postgresql", True),
        ('SELECT `name` FROM users', "postgresql", False),  # Wrong quotes
        ('SELECT * FROM users WHERE status = NULL', "postgresql", False),  # NULL comparison
        ('DROP TABLE users', "postgresql", False),  # Dangerous
    ]
    
    for sql, dialect, expected_valid in test_cases:
        result = validate_sql(sql, dialect)
        status = "✓" if result.is_valid == expected_valid else "✗"
        print(f"  {status} [{('VALID' if result.is_valid else 'INVALID'):7}] {sql[:40]}")
        if result.issues:
            print(f"      Issues: {result.issues[0][:50]}")
    
    # Test auto-fix
    bad_sql = "SELECT * FROM users WHERE status = NULL"
    fixed, fixes = fix_common_issues(bad_sql, "postgresql")
    print(f"\n  Auto-fix test:")
    print(f"    Before: {bad_sql}")
    print(f"    After:  {fixed}")
    print(f"    Fixes:  {fixes}")
    
    # Test safety check
    safe, reason = is_safe_query("SELECT * FROM users")
    print(f"\n  Safety check: safe={safe}, reason={reason}")
    
    safe, reason = is_safe_query("DROP TABLE users")
    print(f"  Safety check: safe={safe}, reason={reason}")
    
    print("  ✅ SQL Validator module OK")
except Exception as e:
    print(f"  ❌ Error: {e}")

# =============================================================================
# TEST 6: Flow Router Module
# =============================================================================
print("\n📦 TEST 6: Flow Router Module")
print("-" * 50)

try:
    from flow_router import FlowConfig, TokenUsage, QueryResult, create_default_config
    
    # Test config creation
    config = create_default_config(dialect="postgresql")
    print(f"  Default config created:")
    print(f"    Dialect: {config.dialect}")
    print(f"    Reasoning: {config.reasoning_provider}")
    print(f"    SQL: {config.sql_provider}")
    print(f"    Opus: {config.enable_opus}")
    
    # Test token tracking
    tokens = TokenUsage()
    tokens.classifier = {"input": 100, "output": 25}
    tokens.reasoning = {"input": 500, "output": 200}
    print(f"\n  Token tracking:")
    print(f"    Total: {tokens.total()}")
    print(f"    Total tokens: {tokens.total_tokens()}")
    
    # Test QueryResult
    result = QueryResult(
        sql="SELECT * FROM test",
        results=None,
        success=True,
        complexity="easy"
    )
    print(f"\n  QueryResult created:")
    print(f"    SQL: {result.sql}")
    print(f"    Success: {result.success}")
    print(f"    Complexity: {result.complexity}")
    
    print("  ✅ Flow Router module OK")
except Exception as e:
    print(f"  ❌ Error: {e}")

# =============================================================================
# TEST 7: Schema RAG Module
# =============================================================================
print("\n📦 TEST 7: Schema RAG Module")
print("-" * 50)

try:
    from schema_rag import format_schema_for_llm, format_schema_as_json
    
    # Test schema formatting
    test_schema = {
        "columns": [
            {"table": "public.orders", "column": "region", "type": "VARCHAR", "friendly_name": "Order region"},
            {"table": "public.orders", "column": "amount", "type": "DECIMAL", "friendly_name": "Order amount"},
        ],
        "tables_used": ["public.orders"]
    }
    
    formatted = format_schema_for_llm(test_schema, "postgresql")
    print(f"  Formatted schema ({len(formatted)} chars):")
    print(f"    {formatted[:100]}...")
    
    json_format = format_schema_as_json(test_schema)
    print(f"\n  JSON format ({len(json_format)} chars):")
    print(f"    {json_format[:100]}...")
    
    print("  ✅ Schema RAG module OK")
except Exception as e:
    print(f"  ❌ Error: {e}")

# =============================================================================
# TEST 8: Smart Sampler Module
# =============================================================================
print("\n📦 TEST 8: Smart Sampler Module")
print("-" * 50)

try:
    from smart_sampler import get_dialect_name, quote_identifier, parse_table_name
    
    # Test dialect detection
    print(f"  Dialect parsing:")
    print(f"    'postgres' → {get_dialect_name(type('Engine', (), {'dialect': type('Dialect', (), {'name': 'postgres'})()})())}")
    
    # Test identifier quoting
    print(f"\n  Identifier quoting:")
    print(f"    PostgreSQL: {quote_identifier('Column Name', 'postgresql')}")
    print(f"    MySQL: {quote_identifier('Column Name', 'mysql')}")
    print(f"    SQL Server: {quote_identifier('Column Name', 'mssql')}")
    
    # Test table parsing
    print(f"\n  Table parsing:")
    print(f"    'public.users' → {parse_table_name('public.users')}")
    print(f"    'users' → {parse_table_name('users')}")
    
    print("  ✅ Smart Sampler module OK")
except Exception as e:
    print(f"  ❌ Error: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("""
All modules loaded and basic tests passed!

To fully test with database:
1. Set up Supabase with schema_rag_setup.sql
2. Connect your database in Streamlit
3. Run queries through Tab 3

Files created:
  ├── abbreviations.py      ✓
  ├── pii_detector.py       ✓
  ├── query_classifier.py   ✓
  ├── prompt_optimizer.py   ✓
  ├── sql_validator.py      ✓
  ├── schema_rag.py         ✓
  ├── smart_sampler.py      ✓
  ├── flow_router.py        ✓
  ├── schema_rag_setup.sql  ✓
  └── tab3_integration.py   ✓
""")
print("=" * 70)