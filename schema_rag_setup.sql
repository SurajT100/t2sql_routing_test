-- ============================================================================
-- SCHEMA RAG DATABASE SETUP
-- ============================================================================
-- Run this in your Supabase SQL Editor to enable Schema RAG
-- Requires: pgvector extension (already enabled in Supabase)
-- ============================================================================

-- Enable pgvector if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- TABLE: schema_columns
-- Stores column metadata and embeddings for Schema RAG
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_columns (
    -- Primary identifiers
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    tenant_id UUID,                              -- For multi-tenant isolation
    connection_hash TEXT,                        -- Hash of connection string
    
    -- Object identification
    object_name TEXT NOT NULL,                   -- Full table/view name (schema.table)
    object_type TEXT DEFAULT 'table',            -- 'table' or 'view'
    column_name TEXT NOT NULL,
    
    -- Column metadata
    data_type TEXT,
    nullable BOOLEAN DEFAULT TRUE,
    is_primary_key BOOLEAN DEFAULT FALSE,
    is_foreign_key BOOLEAN DEFAULT FALSE,
    
    -- Auto-enrichment (from abbreviations.py)
    auto_expanded_name TEXT,                     -- "region code" from "rgn_cd"
    sample_values TEXT[],                        -- Raw sample values
    sample_values_masked TEXT[],                 -- PII-masked samples
    cardinality INTEGER,                         -- Distinct value count
    null_percentage DECIMAL(5,2),                -- % of NULL values
    
    -- PII detection
    has_pii BOOLEAN DEFAULT FALSE,
    pii_types TEXT[],                            -- ['email', 'phone', 'ssn']
    pii_severity TEXT,                           -- 'critical', 'high', 'medium', 'low'
    
    -- User enrichment
    friendly_name TEXT,                          -- User-provided name
    user_description TEXT,                       -- User-provided description
    business_terms TEXT[],                       -- ['region', 'territory', 'area']
    value_mappings JSONB,                        -- {"BLR": "Bangalore", "MUM": "Mumbai"}
    
    -- Enrichment tracking
    enrichment_status TEXT DEFAULT 'pending',    -- 'pending', 'auto', 'user', 'verified'
    enrichment_date TIMESTAMP WITH TIME ZONE,
    
    -- Data quality flags
    data_quality_issues JSONB,                   -- From pii_detector.py
    
    -- Embedding for RAG
    embedding_text TEXT,                         -- Text used for embedding
    embedding vector(384),                       -- Sentence-transformer embedding
    
    -- Usage tracking (for learning)
    times_retrieved INTEGER DEFAULT 0,           -- How often retrieved by RAG
    times_query_succeeded INTEGER DEFAULT 0,     -- How often led to successful query
    last_retrieved_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint per tenant/connection/table/column
    UNIQUE(tenant_id, connection_hash, object_name, column_name)
);

-- ============================================================================
-- TABLE: schema_objects
-- Stores table/view level metadata
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_objects (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    tenant_id UUID,
    connection_hash TEXT,
    
    -- Object identification
    object_name TEXT NOT NULL,                   -- Full name (schema.table)
    object_type TEXT NOT NULL,                   -- 'table' or 'view'
    schema_name TEXT,
    table_name TEXT,
    
    -- Metadata
    row_count BIGINT,
    column_count INTEGER,
    estimated_size_bytes BIGINT,
    
    -- User enrichment
    friendly_name TEXT,
    description TEXT,
    business_terms TEXT[],
    is_recommended BOOLEAN DEFAULT TRUE,         -- FALSE if should use view instead
    
    -- Embedding
    embedding_text TEXT,
    embedding vector(384),
    
    -- Profiling status
    profile_status TEXT DEFAULT 'pending',       -- 'pending', 'profiling', 'complete', 'error'
    profile_date TIMESTAMP WITH TIME ZONE,
    profile_error TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tenant_id, connection_hash, object_name)
);

-- ============================================================================
-- TABLE: rule_column_dependencies
-- Deterministic map of which columns each rule mandates be included.
-- Populated by rule_dependency_extractor.py whenever rules are saved.
-- Queried by context_agent.py to inject mandatory columns even when the
-- LLM omits them from Pass 1 column identification.
-- ============================================================================

CREATE TABLE IF NOT EXISTS rule_column_dependencies (
    id              SERIAL PRIMARY KEY,
    table_name      TEXT NOT NULL,       -- Table the column belongs to
    column_name     TEXT NOT NULL,       -- Column that must be included
    rule_name       TEXT NOT NULL,       -- Business rule that requires this column
    reason          TEXT,                -- Why this column is required
    dependency_type TEXT DEFAULT 'filter',  -- 'filter', 'metric', 'date', 'join'
    auto_apply      BOOLEAN DEFAULT TRUE,   -- Whether to auto-inject into context
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(table_name, column_name, rule_name)
);

CREATE INDEX IF NOT EXISTS idx_rule_col_deps_table
ON rule_column_dependencies(table_name);

CREATE INDEX IF NOT EXISTS idx_rule_col_deps_auto_apply
ON rule_column_dependencies(auto_apply) WHERE auto_apply = TRUE;

-- ============================================================================
-- INDEXES for performance
-- ============================================================================

-- Vector similarity search index (IVFFlat for approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_schema_columns_embedding 
ON schema_columns 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_schema_objects_embedding 
ON schema_objects 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Tenant/connection isolation
CREATE INDEX IF NOT EXISTS idx_schema_columns_tenant 
ON schema_columns(tenant_id, connection_hash);

CREATE INDEX IF NOT EXISTS idx_schema_objects_tenant 
ON schema_objects(tenant_id, connection_hash);

-- Object name lookups
CREATE INDEX IF NOT EXISTS idx_schema_columns_object 
ON schema_columns(object_name);

CREATE INDEX IF NOT EXISTS idx_schema_columns_enrichment 
ON schema_columns(enrichment_status);

-- ============================================================================
-- FUNCTION: Update timestamp trigger
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables
DROP TRIGGER IF EXISTS update_schema_columns_updated_at ON schema_columns;
CREATE TRIGGER update_schema_columns_updated_at
    BEFORE UPDATE ON schema_columns
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_schema_objects_updated_at ON schema_objects;
CREATE TRIGGER update_schema_objects_updated_at
    BEFORE UPDATE ON schema_objects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- FUNCTION: Search columns by similarity
-- ============================================================================

CREATE OR REPLACE FUNCTION search_schema_columns(
    p_query_embedding vector(384),
    p_tables TEXT[],
    p_tenant_id UUID DEFAULT NULL,
    p_connection_hash TEXT DEFAULT NULL,
    p_threshold FLOAT DEFAULT 0.6,
    p_limit INT DEFAULT 15
)
RETURNS TABLE (
    object_name TEXT,
    column_name TEXT,
    data_type TEXT,
    friendly_name TEXT,
    user_description TEXT,
    business_terms TEXT[],
    sample_values TEXT[],
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sc.object_name,
        sc.column_name,
        sc.data_type,
        sc.friendly_name,
        sc.user_description,
        sc.business_terms,
        COALESCE(sc.sample_values_masked, sc.sample_values) as sample_values,
        1 - (sc.embedding <=> p_query_embedding) as similarity
    FROM schema_columns sc
    WHERE 
        sc.object_name = ANY(p_tables)
        AND sc.embedding IS NOT NULL
        AND (p_tenant_id IS NULL OR sc.tenant_id = p_tenant_id)
        AND (p_connection_hash IS NULL OR sc.connection_hash = p_connection_hash)
        AND 1 - (sc.embedding <=> p_query_embedding) > p_threshold
    ORDER BY similarity DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- FUNCTION: Get columns needing enrichment
-- ============================================================================

CREATE OR REPLACE FUNCTION get_columns_needing_enrichment(
    p_tenant_id UUID DEFAULT NULL,
    p_connection_hash TEXT DEFAULT NULL,
    p_limit INT DEFAULT 50
)
RETURNS TABLE (
    object_name TEXT,
    column_name TEXT,
    data_type TEXT,
    auto_expanded_name TEXT,
    sample_values TEXT[],
    has_pii BOOLEAN,
    enrichment_status TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sc.object_name,
        sc.column_name,
        sc.data_type,
        sc.auto_expanded_name,
        sc.sample_values,
        sc.has_pii,
        sc.enrichment_status
    FROM schema_columns sc
    WHERE 
        sc.enrichment_status IN ('pending', 'auto')
        AND (p_tenant_id IS NULL OR sc.tenant_id = p_tenant_id)
        AND (p_connection_hash IS NULL OR sc.connection_hash = p_connection_hash)
    ORDER BY 
        CASE WHEN sc.has_pii THEN 0 ELSE 1 END,  -- PII columns first
        sc.object_name,
        sc.column_name
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE DATA (for testing)
-- ============================================================================

-- Uncomment to insert sample data for testing:
/*
INSERT INTO schema_columns (
    object_name, column_name, data_type, 
    auto_expanded_name, business_terms,
    embedding_text, enrichment_status
) VALUES 
    ('public.vw_sales', 'rgn_cd', 'VARCHAR', 
     'region code', ARRAY['region', 'territory', 'area'],
     'rgn_cd region code - geographic sales territory', 'auto'),
    ('public.vw_sales', 'tot_sls', 'DECIMAL', 
     'total sales', ARRAY['sales', 'revenue', 'amount'],
     'tot_sls total sales - sum of sales amount', 'auto'),
    ('public.vw_sales', 'mgn_amt', 'DECIMAL', 
     'margin amount', ARRAY['margin', 'profit', 'gross'],
     'mgn_amt margin amount - profit margin', 'auto');
*/

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check table was created
SELECT 'schema_columns' as table_name, COUNT(*) as row_count FROM schema_columns
UNION ALL
SELECT 'schema_objects' as table_name, COUNT(*) as row_count FROM schema_objects;

-- Check indexes
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename IN ('schema_columns', 'schema_objects');

-- ============================================================================
-- NOTES
-- ============================================================================
/*
USAGE:

1. Profile a table (from Python):
   - Call sample_all_columns() to get samples
   - Call expand_column_name() for auto-expansion
   - Call detect_pii() for PII detection
   - Generate embedding with sentence-transformer
   - Insert into schema_columns

2. Search for relevant columns (from Python):
   - Generate embedding for user question
   - Call search_schema_columns() function
   - Use results in LLM prompt

3. User enrichment (from Streamlit UI):
   - Show columns with enrichment_status = 'pending' or 'auto'
   - Let user add friendly_name, description, business_terms
   - Update enrichment_status to 'user' or 'verified'

EMBEDDING MODEL:
- Using sentence-transformers with 384 dimensions
- Model: 'all-MiniLM-L6-v2' (same as your existing RAG)
- Ensure consistency with vector_utils_v2.py

MULTI-TENANT:
- Use tenant_id for user isolation
- Use connection_hash for database isolation within tenant
- Both are optional for single-tenant setups
*/
