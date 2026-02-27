"""
Database Dialect Auto-Detection and Rule Generation
===================================================
Automatically detects database type and creates appropriate syntax rules.
Works with: PostgreSQL, MySQL, SQL Server, Oracle
"""

from datetime import datetime

DIALECT_TEMPLATES = {
    'postgresql': {
        'name': 'PostgreSQL Dialect',
        'description': '''Connected database is PostgreSQL.

SQL SYNTAX REQUIREMENTS:
1. Identifiers: Use double quotes
   - Tables: "schema"."table_name"
   - Columns: "column_name"
   
2. String values: Use single quotes
   - Correct: WHERE "Region" = 'Bangalore'
   - Wrong: WHERE "Region" = "Bangalore"

3. Schema qualification: Always use schema.table
   - Correct: FROM "public"."SAP"
   - Wrong: FROM SAP

4. Case sensitivity: PostgreSQL is case-sensitive with quoted identifiers
   - "Region" ≠ "region" ≠ "REGION"

CORRECT EXAMPLE:
SELECT "column1", SUM("column2")
FROM "schema"."table"
WHERE "Date1" IN ('2024-10', '2024-11', '2024-12')
  AND "Type" <> 'Rebate'
GROUP BY "Area"
ORDER BY "column3" DESC;

WRONG EXAMPLE (DO NOT USE):
SELECT Region, SUM(Margin) / 100000 AS sales_lakhs
FROM public.SAP
WHERE Month IN ('2024-10', '2024-11', '2024-12')
GROUP BY Region;

COMMON ERRORS TO AVOID:
❌ Missing quotes on identifiers
❌ Using double quotes for string values
❌ Omitting schema name
❌ Wrong quote character (backticks or brackets)
''',
        'keywords': ['dialect', 'postgres', 'postgresql', 'syntax', 'sql', 'query', 'format', 'quotes', 'identifiers'],
        'quote_char': '"',
        'string_quote': "'",
        'schema_separator': '.',
        'display_name': 'PostgreSQL'
    },
    
    'mysql': {
        'name': 'MySQL Dialect',
        'description': '''Connected database is MySQL.

SQL SYNTAX REQUIREMENTS:
1. Identifiers: Use backticks
   - Tables: `database`.`table_name`
   - Columns: `column_name`
   
2. String values: Use single quotes
   - Correct: WHERE `Region` = 'Bangalore'
   - Wrong: WHERE `Region` = "Bangalore"

3. Database qualification: database.table_name
   - Correct: FROM `FBS_DB`.`SAP`
   - Can omit database if default: FROM `SAP`

4. Case sensitivity: Depends on OS (case-insensitive on Windows, sensitive on Linux)

CORRECT EXAMPLE:
SELECT `Region`, SUM(`Margin`) / 100000 AS `sales_lakhs`
FROM `FBS_DB`.`SAP`
WHERE `Month` IN ('2024-10', '2024-11', '2024-12')
  AND `U_Ordertype` <> 'Rebate'
GROUP BY `Region`
ORDER BY `sales_lakhs` DESC;

WRONG EXAMPLE (DO NOT USE):
SELECT "Region", SUM("Margin") / 100000 AS "sales_lakhs"
FROM FBS_DB.SAP
WHERE Month IN ('2024-10', '2024-11', '2024-12')
GROUP BY Region;

COMMON ERRORS TO AVOID:
❌ Using double quotes instead of backticks
❌ Using double quotes for strings
❌ Missing backticks on reserved words
''',
        'keywords': ['dialect', 'mysql', 'syntax', 'sql', 'query', 'format', 'backticks', 'identifiers'],
        'quote_char': '`',
        'string_quote': "'",
        'schema_separator': '.',
        'display_name': 'MySQL'
    },
    
    'mssql': {
        'name': 'SQL Server Dialect',
        'description': '''Connected database is Microsoft SQL Server.

SQL SYNTAX REQUIREMENTS:
1. Identifiers: Use square brackets
   - Tables: [schema].[table_name]
   - Columns: [column_name]
   - Alternative: Double quotes (if QUOTED_IDENTIFIER is ON)
   
2. String values: Use single quotes
   - Correct: WHERE [Region] = 'Bangalore'
   - Wrong: WHERE [Region] = "Bangalore"

3. Schema qualification: schema.table_name
   - Correct: FROM [dbo].[SAP]
   - Default schema is usually 'dbo'

4. Case sensitivity: Depends on collation (usually case-insensitive)

CORRECT EXAMPLE:
SELECT [Region], SUM([Margin]) / 100000 AS [sales_lakhs]
FROM [dbo].[SAP]
WHERE [Month] IN ('2024-10', '2024-11', '2024-12')
  AND [U_Ordertype] <> 'Rebate'
GROUP BY [Region]
ORDER BY [sales_lakhs] DESC;

ALTERNATIVE (with double quotes):
SELECT "Region", SUM("Margin") / 100000 AS "sales_lakhs"
FROM "dbo"."SAP"
WHERE "Month" IN ('2024-10', '2024-11', '2024-12')
GROUP BY "Region";

WRONG EXAMPLE (DO NOT USE):
SELECT Region, SUM(Margin) / 100000 AS sales_lakhs
FROM dbo.SAP
WHERE Month IN ('2024-10', '2024-11', '2024-12')
GROUP BY Region;

COMMON ERRORS TO AVOID:
❌ Using backticks instead of brackets
❌ Using double quotes for strings
❌ Omitting schema for user tables
''',
        'keywords': ['dialect', 'mssql', 'sqlserver', 'sql server', 'tsql', 'syntax', 'sql', 'query', 'format', 'brackets'],
        'quote_char': '[',
        'quote_char_end': ']',
        'string_quote': "'",
        'schema_separator': '.',
        'display_name': 'Microsoft SQL Server'
    },
    
    'oracle': {
        'name': 'Oracle Dialect',
        'description': '''Connected database is Oracle.

SQL SYNTAX REQUIREMENTS:
1. Identifiers: Two options
   - Case-sensitive: Use double quotes: "Region", "Margin"
   - Case-insensitive: Use uppercase without quotes: REGION, MARGIN
   - Oracle converts unquoted identifiers to uppercase
   
2. String values: Use single quotes
   - Correct: WHERE "Region" = 'Bangalore'

3. Schema qualification: owner.table_name
   - Correct: FROM SALES.SAP_DATA
   - Or: FROM "sales"."sap_data" (case-sensitive)

4. Case sensitivity: 
   - Unquoted identifiers → converted to UPPERCASE
   - Quoted identifiers → case-sensitive

CORRECT EXAMPLE (mixed case with quotes):
SELECT "Region", SUM("Margin") / 100000 AS "sales_lakhs"
FROM "sales"."SAP"
WHERE "Month" IN ('2024-10', '2024-11', '2024-12')
  AND "U_Ordertype" <> 'Rebate'
GROUP BY "Region"
ORDER BY "sales_lakhs" DESC;

CORRECT EXAMPLE (uppercase without quotes):
SELECT REGION, SUM(MARGIN) / 100000 AS SALES_LAKHS
FROM SALES.SAP
WHERE MONTH IN ('2024-10', '2024-11', '2024-12')
  AND U_ORDERTYPE <> 'Rebate'
GROUP BY REGION
ORDER BY SALES_LAKHS DESC;

COMMON ERRORS TO AVOID:
❌ Mixing quoted and unquoted (e.g., "Region" vs REGION)
❌ Using lowercase without quotes (becomes uppercase)
❌ Using backticks or brackets
''',
        'keywords': ['dialect', 'oracle', 'syntax', 'sql', 'query', 'format', 'quotes', 'identifiers', 'uppercase'],
        'quote_char': '"',
        'string_quote': "'",
        'schema_separator': '.',
        'display_name': 'Oracle'
    },
    
    'sqlite': {
        'name': 'SQLite Dialect',
        'description': '''Connected database is SQLite.

SQL SYNTAX REQUIREMENTS:
1. Identifiers: Use double quotes or backticks
   - Preferred: "table_name", "column_name"
   - Alternative: `table_name`, `column_name`
   
2. String values: Use single quotes
   - Correct: WHERE "Region" = 'Bangalore'

3. No schema qualification: SQLite doesn't have schemas
   - Correct: FROM SAP
   - No schema prefix needed

4. Case sensitivity: Case-insensitive by default

CORRECT EXAMPLE:
SELECT "Region", SUM("Margin") / 100000 AS "sales_lakhs"
FROM "SAP"
WHERE "Month" IN ('2024-10', '2024-11', '2024-12')
  AND "U_Ordertype" <> 'Rebate'
GROUP BY "Region"
ORDER BY "sales_lakhs" DESC;

COMMON ERRORS TO AVOID:
❌ Adding schema prefix (e.g., public.SAP)
❌ Using brackets
❌ Using double quotes for strings
''',
        'keywords': ['dialect', 'sqlite', 'syntax', 'sql', 'query', 'format', 'quotes'],
        'quote_char': '"',
        'string_quote': "'",
        'schema_separator': None,
        'display_name': 'SQLite'
    }
}


def get_dialect_name(engine):
    """
    Get normalized dialect name from SQLAlchemy engine.
    
    Args:
        engine: SQLAlchemy engine
    
    Returns:
        str: Normalized dialect name (postgresql, mysql, mssql, oracle, sqlite)
    """
    dialect = engine.dialect.name.lower()
    
    # Normalize dialect names
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
        # Unknown dialect - default to generic SQL
        return 'generic'


def get_dialect_template(dialect_name):
    """
    Get template for a specific dialect.
    
    Args:
        dialect_name: Name of dialect (postgresql, mysql, etc.)
    
    Returns:
        dict: Dialect template or None if not found
    """
    return DIALECT_TEMPLATES.get(dialect_name)


def create_dialect_rule_data(engine):
    """
    Create rule data for detected database dialect.
    
    Args:
        engine: SQLAlchemy engine
    
    Returns:
        dict: Rule data ready to be saved, or None if dialect not supported
    """
    dialect_name = get_dialect_name(engine)
    template = get_dialect_template(dialect_name)
    
    if not template:
        return None
    
    rule_data = {
        "rule_type": "dialect",
        "dialect": dialect_name,
        "quote_char": template['quote_char'],
        "string_quote": template['string_quote'],
        "schema_separator": template.get('schema_separator'),
        "auto_generated": True,
        "generated_at": datetime.now().isoformat()
    }
    
    if 'quote_char_end' in template:
        rule_data['quote_char_end'] = template['quote_char_end']
    
    return {
        "name": template['name'],
        "description": template['description'],
        "keywords": template['keywords'],
        "rule_data": rule_data,
        "display_name": template['display_name']
    }


if __name__ == "__main__":
    print("Database Dialect Templates")
    print("=" * 60)
    print(f"\nSupported dialects: {len(DIALECT_TEMPLATES)}")
    for dialect, template in DIALECT_TEMPLATES.items():
        print(f"\n{template['display_name']}:")
        print(f"  Quote char: {template['quote_char']}")
        print(f"  Keywords: {', '.join(template['keywords'][:5])}...")
