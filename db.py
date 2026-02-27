"""
Database utility functions
"""
from sqlalchemy import create_engine, inspect, text
import pandas as pd


def connect_db(connection_string: str):
    """Connect to database"""
    engine = create_engine(connection_string)
    return engine


def get_schema(engine) -> dict:
    """Get full database schema"""
    inspector = inspect(engine)
    schema_dict = {}
    
    for schema_name in inspector.get_schema_names():
        if schema_name in ['information_schema', 'pg_catalog']:
            continue
            
        for table_name in inspector.get_table_names(schema=schema_name):
            full_name = f"{schema_name}.{table_name}"
            cols = inspector.get_columns(table_name, schema=schema_name)
            schema_dict[full_name] = [c["name"] for c in cols]
    
    return schema_dict


def get_tables_and_views(engine):
    """Get list of tables and views"""
    inspector = inspect(engine)
    tables = []
    views = []
    
    for schema_name in inspector.get_schema_names():
        if schema_name in ['information_schema', 'pg_catalog']:
            continue
            
        for table in inspector.get_table_names(schema=schema_name):
            tables.append(f"{schema_name}.{table}")
            
        for view in inspector.get_view_names(schema=schema_name):
            views.append(f"{schema_name}.{view}")
    
    return tables, views


def run_sql(engine, sql: str) -> pd.DataFrame:
    """Execute SQL and return DataFrame"""
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df
