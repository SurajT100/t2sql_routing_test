"""
Text-to-SQL RAG Testing Platform V2 - DUAL CONNECTION
======================================================
Architecture:
- LOCAL DB: User's data (connected via UI) - for querying
- SUPABASE: Vector DB for RAG rules (from .env) - automatic/hidden

Users connect their database via UI
Rules automatically saved to Supabase (backend)
"""

import streamlit as st
from urllib.parse import quote_plus
from sqlalchemy import inspect, text
import json
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

from schema_profiler import (
    check_schema_columns_table,
    profile_selected_tables,
    get_profiled_columns,
    get_profile_stats,
    update_column_enrichment
)
from flow_router import process_query, FlowConfig
from query_classifier import classify_query, get_flow_config

# Load environment variables
load_dotenv()

from db import connect_db, get_schema, run_sql, get_tables_and_views
from llm_v2 import call_llm, calculate_cost
from vector_utils_v2 import (
    get_relevant_context, 
    format_context_for_llm,
    log_query_to_history,
    get_embedding
)
from prompt_templates import get_reasoning_prompt, get_sql_generation_prompt, clean_sql_output
from smart_keywords import extract_smart_keywords, enhance_description_with_keywords
from dialect_templates import create_dialect_rule_data, get_dialect_name, DIALECT_TEMPLATES

# ======================
# VECTOR DB CONNECTION (Supabase - Backend Only)
# ======================
VECTOR_ENGINE = None
VECTOR_CONNECTED = False
VECTOR_ERROR = None

try:
    supabase_url = os.getenv("SUPABASE_CONNECTION_STRING")
    if supabase_url:
        VECTOR_ENGINE = connect_db(supabase_url)
        VECTOR_CONNECTED = True
    else:
        VECTOR_ERROR = "SUPABASE_CONNECTION_STRING not found in .env file"
except Exception as e:
    VECTOR_ERROR = f"Failed to connect to Supabase: {str(e)}"


# ======================
# PAGE CONFIG
# ======================
st.set_page_config("Text-to-SQL RAG Platform", layout="wide", page_icon="🧠")

# ======================
# INITIALIZE SESSION STATE
# ======================
if "token_usage" not in st.session_state:
    st.session_state.token_usage = {
        "opus": {"input": 0, "output": 0},
        "nvidia": {"input": 0, "output": 0},
        "o1": {"input": 0, "output": 0},
        "claude": {"input": 0, "output": 0},
        "groq": {"input": 0, "output": 0},
        "grok": {"input": 0, "output": 0},
        "qwen": {"input": 0, "output": 0}
    }

if "query_log" not in st.session_state:
    st.session_state.query_log = []

if "query_counter" not in st.session_state:
    st.session_state.query_counter = 0

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Database Setup"

if "selected_objects" not in st.session_state:
    st.session_state.selected_objects = []

# ======================
# HEADER
# ======================
st.title("🧠 Text-to-SQL RAG Testing Platform V4")
st.caption("Connect your database → Add business rules → Test queries → Compare LLMs")

# Show connection status
col_status1, col_status2 = st.columns(2)
with col_status1:
    if "engine" in st.session_state:
        st.success("✅ Your Database: Connected")
    else:
        st.info("⬅️ Connect your database in sidebar")

with col_status2:
    if VECTOR_CONNECTED:
        st.success("✅ RAG System: Ready")
    else:
        st.error(f"❌ RAG System: {VECTOR_ERROR}")
        st.caption("⚠️ Add SUPABASE_CONNECTION_STRING to .env file")

# ======================
# SIDEBAR - DATABASE CONNECTION
# ======================
with st.sidebar:
    st.header("🗄️ Database Connection")
    
    connection_method = st.radio(
        "Connection Method",
        ["Connection String", "Manual Entry"],
        key="conn_method"
    )
    
    if connection_method == "Connection String":
        connection_string = st.text_area(
            "PostgreSQL Connection String",
            placeholder="postgresql://user:password@host:port/database",
            height=100,
            key="conn_str"
        )
        db_url = connection_string.strip() if connection_string else ""
    else:
        db_user = st.text_input("Username", value="postgres")
        db_password = st.text_input("Password", type="password")
        db_host = st.text_input("Host", value="localhost")
        db_port = st.text_input("Port", value="5432")
        db_name = st.text_input("Database", value="postgres")
        
        db_url = (
            f"postgresql://{quote_plus(db_user)}:"
            f"{quote_plus(db_password)}@"
            f"{db_host}:{db_port}/{db_name}"
        )
    
    if st.button("🔌 Connect", type="primary"):
        try:
            engine = connect_db(db_url)
            st.session_state.engine = engine
            st.session_state.schema = get_schema(engine)
            
            # AUTO-DETECT DIALECT AND CREATE RULE
            dialect_name = get_dialect_name(engine)
            dialect_rule_info = create_dialect_rule_data(engine)
            
            if dialect_rule_info:
                # Check if dialect rule already exists
                with VECTOR_ENGINE.connect() as conn:
                    existing_dialect = conn.execute(
                        text("""
                            SELECT id, rule_name, rule_data
                            FROM business_rules_v2
                            WHERE is_active = TRUE
                              AND (rule_name LIKE '%Dialect%' 
                                   OR rule_data::text LIKE '%"rule_type": "dialect"%')
                            LIMIT 1
                        """)
                    ).fetchone()
                    
                    if existing_dialect:
                        # Check if it's the same dialect
                        existing_data = json.loads(existing_dialect[2]) if isinstance(existing_dialect[2], str) else existing_dialect[2]
                        existing_dialect_name = existing_data.get('dialect')
                        
                        if existing_dialect_name != dialect_name:
                            # Different database type - delete old rule
                            conn.execute(
                                text("DELETE FROM business_rules_v2 WHERE id = :id"),
                                {"id": existing_dialect[0]}
                            )
                            conn.commit()
                            st.warning(f"⚠️ Database type changed from {existing_dialect_name} to {dialect_name}. Updated dialect rule.")
                            create_new_dialect_rule = True
                        else:
                            # Same dialect - no action needed
                            create_new_dialect_rule = False
                            st.info(f"ℹ️ Using existing {dialect_rule_info['display_name']} syntax rule")
                    else:
                        # No existing dialect rule
                        create_new_dialect_rule = True
                    
                    # Create new dialect rule if needed
                    if create_new_dialect_rule:
                        # Generate embedding
                        embedding_text = f"{dialect_rule_info['name']} {dialect_rule_info['description']} {' '.join(dialect_rule_info['keywords'])}"
                        embedding = get_embedding(embedding_text)
                        
                        conn.execute(
                            text("""
                                INSERT INTO business_rules_v2
                                (rule_name, rule_description, rule_type, rule_data,
                                 trigger_keywords, priority, is_mandatory, embedding)
                                VALUES
                                (:name, :desc, 'default', CAST(:data AS jsonb),
                                 CAST(:keywords AS text[]), 1, TRUE, CAST(:embedding AS vector))
                            """),
                            {
                                "name": dialect_rule_info['name'],
                                "desc": dialect_rule_info['description'],
                                "data": json.dumps(dialect_rule_info['rule_data']),
                                "keywords": dialect_rule_info['keywords'],
                                "embedding": str(embedding)
                            }
                        )
                        conn.commit()
                        
                        st.success(f"🔧 Auto-created {dialect_rule_info['display_name']} syntax rule")
            
            # Show success with dialect info
            display_name = dialect_rule_info['display_name'] if dialect_rule_info else dialect_name.upper()
            st.success(f"✅ Connected to {display_name}!")
            
            # Show dialect info box
            st.info(f"""
            **Database Type Detected:** {display_name}
            
            📋 A critical rule for {display_name} SQL syntax has been automatically created.
            This ensures all generated queries use the correct syntax for your database.
            
            View it in **Tab 2: Business Rules** under 🤖 Auto-Generated Rules.
            """)
            
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    # Show connection status
    if "engine" in st.session_state:
        st.success("🟢 Database Connected")
    
    st.divider()
    
    # LLM Configuration
    st.header("🤖 LLM Configuration")
    
    reasoning_llm = st.selectbox(
        "Reasoning LLM",
        [
            "NVIDIA Qwen 3 Next 80B (Thinking)",
            "OpenAI o1-mini (Recommended)",
            "OpenAI o1 (Best Reasoning)",
            "Claude Sonnet 4",
            "Claude Haiku 4.5",
            "Groq Llama 3.3 70B",
            "xAI Grok Beta",
            "Qwen 2.5 Coder 32B"
        ],
        key="reasoning_llm"
    )
    
    reasoning_map = {
        "NVIDIA Qwen 3 Next 80B (Thinking)": "nvidia_qwen3",
        "OpenAI o1-mini (Recommended)": "o1_mini",
        "OpenAI o1 (Best Reasoning)": "o1",
        "Claude Sonnet 4": "claude_sonnet",
        "Claude Haiku 4.5": "claude_haiku",
        "Groq Llama 3.3 70B": "groq",
        "xAI Grok Beta": "grok",
        "Qwen 2.5 Coder 32B": "vertex_qwen"
    }
    
    coding_llm = st.selectbox(
        "SQL Coding LLM",
        [
            "Groq Llama 3.3 70B",
            "Qwen 2.5 Coder 32B",
            "Claude Haiku 4.5",
            "Claude Sonnet 4"
        ],
        key="coding_llm"
    )
    
    coding_map = {
        "Groq Llama 3.3 70B": "groq",
        "Qwen 2.5 Coder 32B": "vertex_qwen",
        "Claude Haiku 4.5": "claude_haiku",
        "Claude Sonnet 4": "claude_sonnet"
    }
    
    st.session_state.reasoning_provider = reasoning_map[reasoning_llm]
    st.session_state.coding_provider = coding_map[coding_llm]
    
    st.divider()
    
    # Independent Review Mode
    st.header("🎯 Independent Review Mode")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        enable_opus_review = st.checkbox(
            "Enable Independent Review (Highly Recommended)",
            value=True,
            help="Independent LLM reviews SQL with no confirmation bias. "
                 "Ensures high accuracy. Auto-retries if errors found."
        )
    
    with col2:
        if enable_opus_review:
            max_retries = st.number_input(
                "Max Retries",
                min_value=1,
                max_value=5,
                value=3,
                help="How many times to retry if reviewer finds errors"
            )
        else:
            max_retries = 0
    
    if enable_opus_review:
        reviewer_llm = st.selectbox(
            "Select Reviewer LLM",
            ["Claude Opus", "Qwen Coder (Vertex)"],
            index=0,
            help="Claude Opus: Best accuracy, higher cost | Qwen Coder: Good accuracy, lower cost"
        )
        st.session_state.reviewer_provider = {
            "Claude Opus": "claude_opus",
            "Qwen Coder (Vertex)": "vertex_qwen_thinking"
        }[reviewer_llm]
    else:
        st.session_state.reviewer_provider = "claude_opus"
    
    st.session_state.enable_opus_review = enable_opus_review
    st.session_state.max_opus_retries = max_retries
    
    if enable_opus_review:
        reviewer_name = "Opus" if st.session_state.reviewer_provider == "claude_opus" else "Qwen Coder"
        st.success(f"✅ **{reviewer_name} Independent Review Active**")
        st.info(f"📊 **How it works:**\n"
                "1. **Sonnet** analyzes question + business rules → Creates plan\n"
                "2. **Qwen** implements plan → Generates SQL query\n"
                "3. Query executes → Returns results\n"
                f"4. **{reviewer_name}** independently reviews → Validates correctness\n"
                "5. If INCORRECT → Feedback → Regenerate (max retries)\n"
                "6. If CORRECT → Results shown to user ✅")
    else:
        st.warning("⚠️ **Review Disabled** - SQL not validated. Use only for testing!")
    
    st.divider()
    
    # Token Usage
    st.header("📊 Token Usage")
    with st.expander("View Statistics"):
        for provider, tokens in st.session_state.token_usage.items():
            total = tokens['input'] + tokens['output']
            if total > 0:
                st.write(f"**{provider.upper()}**")
                st.write(f"In: {tokens['input']:,} | Out: {tokens['output']:,}")
                st.write(f"Total: {total:,}")
                st.divider()

# ======================
# MAIN TABS
# ======================
tab1, tab2, tab3 = st.tabs([
    "📦 1. Database Setup",
    "📝 2. Business Rules",
    "🔍 3. Test Queries"
])

# ======================
# TAB 1: DATABASE SETUP
# ======================
with tab1:
    st.header("Database Setup")
    
    if "engine" not in st.session_state:
        st.info("👈 Please connect to a database using the sidebar")
    else:
        st.success("✅ Database connected successfully")
        
        # Table Selection
        st.subheader("Select Tables to Expose")
        
        tables, views = get_tables_and_views(st.session_state.engine)
        
        all_objects = (
            [f"table: {t}" for t in tables] +
            [f"view: {v}" for v in views]
        )
        
        selected_objects = st.multiselect(
            "Choose tables/views for LLM to query",
            options=all_objects,
            key="selected_objects"
        )
        
        if selected_objects:
            st.success(f"✅ {len(selected_objects)} objects selected")
            
            # Show schema preview
            with st.expander("📘 View Schema Details"):
                inspector = inspect(st.session_state.engine)
                
                for obj in selected_objects:
                    obj_type, full_name = obj.split(": ")
                    
                    if "." in full_name:
                        schema, table = full_name.split(".")
                    else:
                        schema = "public"
                        table = full_name
                    
                    cols = inspector.get_columns(table, schema=schema)
                    
                    st.write(f"**{full_name}** ({obj_type})")
                    col_list = [f"`{c['name']}` ({c['type']})" for c in cols]
                    st.write(", ".join(col_list))
                    st.divider()

        # ═══════════════════════════════════════════════════════════════════
        # SCHEMA PROFILING SECTION
        # ═══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("🔬 Schema Profiling for RAG")
        st.caption("Profile tables to enable intelligent column retrieval")
        
        # Check if schema_columns table exists
        from schema_profiler import (
            check_schema_columns_table, 
            profile_selected_tables,
            get_profiled_columns,
            get_profile_stats,
            update_column_enrichment
        )
        
        table_exists, setup_message = check_schema_columns_table(VECTOR_ENGINE)
        
        if not table_exists:
            st.warning(f"⚠️ Column Profiling not configured: {setup_message}")
            with st.expander("📋 Setup Instructions", expanded=True):
                st.markdown("""
**To enable Column Profiling & Opus Enrichment:**

1. Open your **Supabase SQL Editor**
2. Copy and run the SQL below
3. Refresh this page

This creates the `schema_columns` table for storing column metadata, descriptions, and embeddings.
                """)
                
                st.code("""
-- Run this in Supabase SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_columns (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    tenant_id UUID,
    connection_hash TEXT,
    object_name TEXT NOT NULL,
    object_type TEXT DEFAULT 'table',
    column_name TEXT NOT NULL,
    data_type TEXT,
    nullable BOOLEAN DEFAULT TRUE,
    auto_expanded_name TEXT,
    friendly_name TEXT,
    user_description TEXT,
    opus_description TEXT,
    opus_business_meaning TEXT,
    opus_sql_usage TEXT,
    opus_notes TEXT,
    enriched_at TIMESTAMP,
    sample_values TEXT[],
    sample_values_masked TEXT[],
    cardinality INTEGER,
    null_percentage DECIMAL(5,2),
    has_pii BOOLEAN DEFAULT FALSE,
    pii_types TEXT[],
    pii_severity TEXT,
    business_terms TEXT[],
    data_quality_issues JSONB,
    embedding_text TEXT,
    embedding vector(384),
    enrichment_status TEXT DEFAULT 'pending',
    enrichment_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(tenant_id, connection_hash, object_name, column_name)
);

CREATE INDEX IF NOT EXISTS idx_schema_columns_embedding 
ON schema_columns USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                """, language="sql")
        
        else:
            # Column profiling is set up - show profiling UI
            stats = get_profile_stats(VECTOR_ENGINE, selected_objects)
            print(f"[DEBUG] Tab 1 - selected_objects: {selected_objects}")  # Debug
            print(f"[DEBUG] Tab 1 - stats: {stats}")  # Debug
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            col_stat1.metric("📊 Profiled", stats["profiled"])
            col_stat2.metric("🔒 With PII", stats["with_pii"])
            col_stat3.metric("✅ Enriched", stats["user_enriched"])
            col_stat4.metric("⚡ Auto Only", stats["auto_enriched"])
            
            col_btn1, col_btn2 = st.columns([1, 2])
            
            with col_btn1:
                profile_clicked = st.button(
                    "🔄 Profile Tables", 
                    type="primary", 
                    use_container_width=True,
                    help="Scan tables, detect PII, expand abbreviations"
                )
            
            with col_btn2:
                if stats["profiled"] > 0:
                    st.caption(f"Last profiled: {stats['profiled']} columns across {len(stats['tables'])} tables")
                else:
                    st.caption("No columns profiled yet. Click 'Profile Tables' to start.")
            
            if profile_clicked:
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(table, column, status):
                        clean_table = table.replace("table: ", "").replace("view: ", "")
                        status_text.text(f"📊 {clean_table} → {column}")
                    
                    with st.spinner("Profiling tables..."):
                        result = profile_selected_tables(
                            st.session_state.engine,
                            VECTOR_ENGINE,
                            selected_objects,
                            progress_callback=update_progress
                        )
                    
                    progress_bar.progress(100)
                    status_text.empty()
                
                if result["errors"]:
                    st.warning(f"⚠️ Completed with {len(result['errors'])} errors")
                    with st.expander("View Errors"):
                        for err in result["errors"]:
                            st.error(err)
                else:
                    st.success(f"✅ Profiled {result['total_columns']} columns across {result['total_tables']} tables")
                
                if result["columns_with_pii"] > 0:
                    st.warning(f"🔒 Detected PII in {result['columns_with_pii']} columns - samples are automatically masked")
                
                with st.expander("📋 Profiling Results", expanded=True):
                    for table_result in result["tables"]:
                        st.write(f"**{table_result['table']}** ({table_result['object_type']})")
                        
                        if table_result["columns"]:
                            col_data = []
                            for col in table_result["columns"]:
                                pii_badge = "🔒 " + ", ".join(col["pii_types"]) if col["has_pii"] else ""
                                col_data.append({
                                    "Column": col["name"],
                                    "Type": col["type"],
                                    "Expanded Name": col["expanded"],
                                    "PII": pii_badge,
                                    "Samples": ", ".join(str(s) for s in col["samples"][:3])
                                })
                            
                            import pandas as pd
                            st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)
                        
                        st.divider()
                
                # Don't rerun - keep results visible
                # Stats will update on next page load
            
            # Column Enrichment UI
            if stats["profiled"] > 0:
                with st.expander("📝 Enrich Column Descriptions", expanded=False):
                    st.caption("Add friendly names and descriptions to improve RAG accuracy")
                    
                    columns = get_profiled_columns(VECTOR_ENGINE, selected_objects)
                    
                    filter_col, search_col = st.columns([1, 2])
                    
                    with filter_col:
                        status_filter = st.selectbox(
                            "Filter",
                            ["All Columns", "⚡ Need Review", "✅ User Enriched", "🔒 Has PII"],
                            key="schema_enrich_filter"
                        )
                    
                    with search_col:
                        search_term = st.text_input(
                            "Search",
                            placeholder="Search columns...",
                            key="schema_enrich_search"
                        )
                    
                    filtered_columns = columns
                    
                    if status_filter == "⚡ Need Review":
                        filtered_columns = [c for c in filtered_columns if c["status"] in ["auto", "pending"]]
                    elif status_filter == "✅ User Enriched":
                        filtered_columns = [c for c in filtered_columns if c["status"] == "user"]
                    elif status_filter == "🔒 Has PII":
                        filtered_columns = [c for c in filtered_columns if c["has_pii"]]
                    
                    if search_term:
                        search_lower = search_term.lower()
                        filtered_columns = [c for c in filtered_columns if 
                                          search_lower in c["column"].lower() or
                                          search_lower in (c["auto_expanded"] or "").lower() or
                                          search_lower in c["table"].lower()]
                    
                    st.caption(f"Showing {len(filtered_columns)} of {len(columns)} columns")
                    
                    for i, col in enumerate(filtered_columns[:20]):
                        status_icon = "✅" if col["status"] == "user" else "⚡"
                        pii_icon = "🔒" if col["has_pii"] else ""
                        
                        with st.expander(
                            f"{status_icon} {pii_icon} **{col['column']}** ({col['table']}) - {col['auto_expanded'] or col['type']}",
                            expanded=False
                        ):
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                st.write(f"**Type:** `{col['type']}`")
                                st.write(f"**Auto-expanded:** {col['auto_expanded'] or 'N/A'}")
                                if col['cardinality']:
                                    st.write(f"**Distinct values:** {col['cardinality']}")
                            
                            with info_col2:
                                samples = col['samples_masked'] if col['has_pii'] else col['samples']
                                if samples:
                                    st.write("**Sample values:**")
                                    st.code(", ".join(str(s) for s in samples[:5]))
                                
                                if col['has_pii']:
                                    st.error(f"🔒 PII: {', '.join(col['pii_types'] or [])}")
                            
                            st.divider()
                            
                            form_key = f"enrich_form_{col['table']}_{col['column']}_{i}"
                            
                            with st.form(form_key):
                                new_friendly = st.text_input(
                                    "Friendly Name",
                                    value=col['friendly_name'] or col['auto_expanded'] or "",
                                    key=f"friendly_{form_key}",
                                    placeholder="e.g., Customer Region Code"
                                )
                                
                                new_description = st.text_area(
                                    "Description",
                                    value=col['description'] or "",
                                    key=f"desc_{form_key}",
                                    placeholder="What this column means, valid values, business logic...",
                                    height=80
                                )
                                
                                current_terms = ", ".join(col['business_terms'] or [])
                                new_terms = st.text_input(
                                    "Business Terms (comma-separated)",
                                    value=current_terms,
                                    key=f"terms_{form_key}",
                                    placeholder="region, territory, area, location"
                                )
                                
                                submitted = st.form_submit_button("💾 Save", type="primary")
                                
                                if submitted:
                                    term_list = [t.strip() for t in new_terms.split(",") if t.strip()] if new_terms else None
                                    
                                    success = update_column_enrichment(
                                        VECTOR_ENGINE,
                                        col['table'],
                                        col['column'],
                                        friendly_name=new_friendly if new_friendly else None,
                                        description=new_description if new_description else None,
                                        business_terms=term_list
                                    )
                                    
                                    if success:
                                        st.success("✅ Saved!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to save")
                    
                    if len(filtered_columns) > 20:
                        st.info(f"Showing first 20 columns. Use search to find specific columns.")
                
                # ═══════════════════════════════════════════════════════════════════
                # OPUS AI ENRICHMENT (One-time activity)
                # ═══════════════════════════════════════════════════════════════════
                with st.expander("🤖 AI Column Enrichment (Opus)", expanded=False):
                    st.caption("Use Claude Opus to automatically generate intelligent column descriptions. This is a ONE-TIME activity that improves SQL generation accuracy.")
                    
                    st.info("""
                    **How it works:**
                    1. Opus analyzes each column (name, type, samples)
                    2. Generates business-aware descriptions
                    3. Stores descriptions for SQL Coder to use at query time
                    
                    **Example output for 'Margin' column:**
                    > "Sales revenue/profit value in currency. Use SUM(Margin) for total sales calculations. Primary metric for revenue queries."
                    """)
                    
                    # Table selection for enrichment
                    tables_for_enrichment = []
                    for obj in selected_objects:
                        _, full_name = obj.split(": ")
                        tables_for_enrichment.append(full_name)
                    
                    selected_table_enrich = st.selectbox(
                        "Select table to enrich",
                        tables_for_enrichment,
                        key="opus_enrich_table"
                    )
                    
                    # Show current enrichment status
                    try:
                        from column_enrichment import get_all_column_descriptions_for_table
                        
                        existing_enrichments = get_all_column_descriptions_for_table(VECTOR_ENGINE, selected_table_enrich)
                        enriched_count = sum(1 for e in existing_enrichments if e.get("opus_description"))
                        total_count = len(existing_enrichments)
                        
                        if enriched_count > 0:
                            st.success(f"✅ {enriched_count}/{total_count} columns already enriched with Opus")
                            
                            # Show existing descriptions
                            with st.expander("View Opus Descriptions", expanded=False):
                                for col_info in existing_enrichments:
                                    if col_info.get("opus_description"):
                                        st.write(f"**{col_info['column_name']}** ({col_info['data_type']})")
                                        st.caption(col_info['opus_description'])
                                        st.divider()
                        else:
                            st.warning(f"⚡ {total_count} columns ready for enrichment")
                    except Exception as e:
                        st.warning(f"Could not check enrichment status: {str(e)[:100]}")
                        enriched_count = 0
                    
                    col_enrich1, col_enrich2 = st.columns([1, 2])
                    
                    with col_enrich1:
                        enrich_clicked = st.button(
                            "🚀 Enrich with Opus",
                            type="primary",
                            use_container_width=True,
                            help="Uses Claude Opus to generate intelligent descriptions"
                        )
                    
                    with col_enrich2:
                        st.caption("⏱️ Takes ~2-3 seconds per column")
                        st.caption("📊 ~500-800 tokens per column")
                    
                    if enrich_clicked:
                        try:
                            from column_enrichment import enrich_columns_with_opus
                            
                            progress_container = st.container()
                            
                            with progress_container:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Get column count for progress
                                columns_to_enrich = get_profiled_columns(VECTOR_ENGINE, [f"table: {selected_table_enrich}"])
                                total_cols = len(columns_to_enrich)
                                current_col = [0]  # Use list for mutable reference
                                
                                def enrich_progress(col_name, status):
                                    current_col[0] += 1
                                    progress_bar.progress(min(current_col[0] / max(total_cols, 1), 1.0))
                                    status_text.text(f"🤖 Opus analyzing: {col_name}...")
                                
                                with st.spinner(f"Enriching {total_cols} columns with Opus..."):
                                    result = enrich_columns_with_opus(
                                        VECTOR_ENGINE,
                                        selected_table_enrich,
                                        llm_provider="claude_opus",
                                        progress_callback=enrich_progress
                                    )
                                
                                progress_bar.progress(100)
                                status_text.empty()
                            
                            if result.get("error"):
                                st.error(f"❌ Error: {result['error']}")
                            else:
                                st.success(f"""
                                ✅ **Opus Enrichment Complete!**
                                - Columns enriched: {result['enriched']}
                                - Errors: {result['errors']}
                                - Total tokens used: {result['total_tokens']:,}
                                """)
                                
                                # Show generated descriptions
                                with st.expander("📋 Generated Descriptions", expanded=True):
                                    for col_result in result.get("columns", []):
                                        if col_result.get("error"):
                                            st.error(f"❌ {col_result['column']}: {col_result['error']}")
                                        else:
                                            st.write(f"**{col_result['column']}**")
                                            st.caption(col_result.get('description', 'No description'))
                                            st.divider()
                                
                        except ImportError:
                            st.error("❌ column_enrichment.py not found. Please download and add it to your project.")
                        except Exception as e:
                            st.error(f"❌ Enrichment failed: {str(e)}")

# ======================
# TAB 2: BUSINESS RULES MANAGEMENT
# ======================
with tab2:
    st.header("Business Rules Management")
    
    if "engine" not in st.session_state:
        st.warning("👈 Please connect to database first (Tab 1)")
    elif "selected_objects" not in st.session_state or not st.session_state.selected_objects:
        st.warning("👈 Please select tables first (Tab 1)")
    else:
        # Build selected schema for dropdowns
        inspector = inspect(st.session_state.engine)
        selected_schema = {}
        
        for obj in st.session_state.selected_objects:
            _, full_name = obj.split(": ")
            
            if "." in full_name:
                schema, table = full_name.split(".")
            else:
                schema = "public"
                table = full_name
            
            cols = inspector.get_columns(table, schema=schema)
            selected_schema[full_name] = [c["name"] for c in cols]
        
        # Rule Type Selection
        st.subheader("Add New Rule")
        
        rule_type = st.selectbox(
            "What type of rule do you want to add?",
            [
                "📊 Metric Definition",
                "🔗 Table Join",
                "🎯 Filter Rule",
                "🔄 Value Mapping",
                "⚠️ Critical Default Rule",
                "📖 Table/Column Description",
                "💡 Query Example"
            ],
            key="rule_type_selector"
        )
        
        st.divider()
        
        # ======================
        # METRIC DEFINITION FORM
        # ======================
        if rule_type == "📊 Metric Definition":
            st.subheader("Add Metric Definition")
            
            with st.form("metric_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    metric_name = st.text_input(
                        "Metric Name*",
                        placeholder="e.g., Total Sales",
                        help="Unique name for this metric"
                    )
                    
                    user_terms = st.text_input(
                        "User Terms (comma-separated)*",
                        placeholder="e.g., sales, revenue, margin, achievement",
                        help="How users might refer to this metric"
                    )
                    
                    table_name = st.selectbox(
                        "Table*",
                        options=list(selected_schema.keys())
                    )
                
                with col2:
                    if table_name:
                        column_name = st.selectbox(
                            "Column*",
                            options=selected_schema[table_name]
                        )
                    else:
                        column_name = st.text_input("Column*")
                    
                    aggregation = st.selectbox(
                        "Aggregation*",
                        options=["SUM", "COUNT", "AVG", "MIN", "MAX", "COUNT DISTINCT"]
                    )
                    
                    transform = st.text_input(
                        "Transform (optional)",
                        placeholder="e.g., / 100000",
                        help="Mathematical transformation to apply"
                    )
                
                display_unit = st.text_input(
                    "Display Unit (optional)",
                    placeholder="e.g., lakhs, millions, USD"
                )
                
                st.write("**Mandatory Filters**")
                apply_fy_filter = st.checkbox(
                    "Apply Current FY Filter automatically",
                    value=False
                )
                
                description = st.text_area(
                    "Description",
                    placeholder="What does this metric represent?"
                )
                
                col_test, col_save = st.columns(2)
                
                with col_test:
                    test_rule = st.form_submit_button("🧪 Test Rule", use_container_width=True)
                
                with col_save:
                    save_rule = st.form_submit_button("💾 Save Rule", type="primary", use_container_width=True)
                
                if test_rule or save_rule:
                    # Validate
                    if not metric_name or not user_terms or not table_name or not column_name:
                        st.error("❌ Please fill all required fields (*)")
                    else:
                        # Build rule data
                        keywords = [k.strip() for k in user_terms.split(",")]
                        
                        rule_data = {
                            "user_terms": keywords,
                            "table": table_name,
                            "column": column_name,
                            "aggregation": aggregation,
                            "formula": f"{aggregation}({column_name})" + (f" {transform}" if transform else ""),
                            "alias": metric_name.replace(" ", "_"),
                            "display": {"unit": display_unit} if display_unit else {}
                        }
                        
                        if apply_fy_filter:
                            rule_data["mandatory_filters"] = [
                                {
                                    "column": "Date",
                                    "condition": "current_fy",
                                    "auto_apply": True
                                }
                            ]
                        
                        if test_rule:
                            st.success("✅ Rule structure is valid!")
                            st.json(rule_data)
                            
                            # Test embedding
                            embedding_text = f"{metric_name} {description} {user_terms}"
                            embedding = get_embedding(embedding_text)
                            st.info(f"✅ Embedding generated ({len(embedding)} dimensions)")
                        
                        if save_rule:
                            try:
                                # Generate embedding
                                embedding_text = f"{metric_name} {description} {user_terms}"
                                embedding = get_embedding(embedding_text)
                                
                                # Save to database
                                with VECTOR_ENGINE.connect() as conn:
                                    conn.execute(
                                        text("""
                                            INSERT INTO business_rules_v2
                                            (rule_name, rule_description, rule_type, rule_data,
                                             trigger_keywords, applies_to_tables, applies_to_columns,
                                             priority, is_mandatory, embedding)
                                            VALUES
                                            (:name, :desc, 'metric', CAST(:data AS jsonb),
                                             CAST(:keywords AS text[]), CAST(:tables AS text[]), 
                                             CAST(:columns AS text[]), 2, FALSE, CAST(:embedding AS vector))
                                            ON CONFLICT (rule_name) DO UPDATE SET
                                                rule_data = EXCLUDED.rule_data,
                                                trigger_keywords = EXCLUDED.trigger_keywords,
                                                updated_at = NOW()
                                        """),
                                        {
                                            "name": metric_name,
                                            "desc": description,
                                            "data": json.dumps(rule_data),
                                            "keywords": keywords,
                                            "tables": [table_name],
                                            "columns": [column_name],
                                            "embedding": str(embedding)
                                        }
                                    )
                                    conn.commit()
                                
                                st.success(f"✅ Metric '{metric_name}' saved successfully!")
                                st.balloons()
                                
                            except Exception as e:
                                st.error(f"❌ Failed to save: {str(e)}")
        
        # ======================
        # TABLE JOIN FORM
        # ======================
        elif rule_type == "🔗 Table Join":
            st.subheader("Add Table Join Relationship")
            
            with st.form("join_form"):
                join_name = st.text_input(
                    "Relationship Name*",
                    placeholder="e.g., Orders and Customers connect via Customer_ID"
                )
                
                st.info("💡 Just define the relationship - LLM will choose appropriate join type (INNER/LEFT/RIGHT) based on the user's query")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Table 1**")
                    table1 = st.selectbox(
                        "Table*",
                        options=list(selected_schema.keys()),
                        key="table1"
                    )
                    
                    if table1:
                        column1 = st.selectbox(
                            "Column*",
                            options=selected_schema[table1],
                            key="column1"
                        )
                    else:
                        column1 = None
                
                with col2:
                    st.write("**Table 2**")
                    table2 = st.selectbox(
                        "Table*",
                        options=list(selected_schema.keys()),
                        key="table2"
                    )
                    
                    if table2:
                        column2 = st.selectbox(
                            "Column*",
                            options=selected_schema[table2],
                            key="column2"
                        )
                    else:
                        column2 = None
                
                description = st.text_area(
                    "What does this relationship represent?*",
                    placeholder="e.g., Links customer orders with customer details using customer_id",
                    help="Explain what connects these tables - this helps LLM choose the right join type"
                )
                
                keywords = st.text_input(
                    "Keywords (comma-separated)",
                    placeholder="e.g., employee, kam, sales rep, join crm sap",
                    help="Words that should trigger using this relationship"
                )
                
                col_test, col_save = st.columns(2)
                
                with col_test:
                    test_join = st.form_submit_button("🧪 Test Rule", use_container_width=True)
                
                with col_save:
                    save_join = st.form_submit_button("💾 Save Rule", type="primary", use_container_width=True)
                
                if test_join or save_join:
                    if not join_name or not table1 or not column1 or not table2 or not column2 or not description:
                        st.error("❌ Please fill all required fields")
                    else:
                        rule_data = {
                            "relationship_name": join_name,
                            "description": description,
                            "table1": table1,
                            "column1": column1,
                            "table2": table2,
                            "column2": column2,
                            "join_condition": f'"{table1}"."{column1}" = "{table2}"."{column2}"',
                            "join_type": "LLM_DECIDES",  # LLM chooses based on query
                            "guidance": description
                        }
                        
                        if test_join:
                            st.success("✅ Relationship structure is valid!")
                            st.code(f'-- Relationship:\n{rule_data["join_condition"]}\n\n-- LLM will use this in INNER/LEFT/RIGHT/FULL JOIN as needed', language="sql")
                            st.info(f"📝 Guidance for LLM: {description}")
                        
                        if save_join:
                            try:
                                keyword_list = [k.strip() for k in keywords.split(",")] if keywords else []
                                # Add table names to keywords automatically
                                keyword_list.extend([table1.lower(), table2.lower(), "join", "relationship"])
                                
                                embedding_text = f"{join_name} {description} {' '.join(keyword_list)}"
                                embedding = get_embedding(embedding_text)
                                
                                with VECTOR_ENGINE.connect() as conn:
                                    conn.execute(
                                        text("""
                                            INSERT INTO business_rules_v2
                                            (rule_name, rule_description, rule_type, rule_data,
                                             trigger_keywords, applies_to_tables, applies_to_columns,
                                             priority, embedding)
                                            VALUES
                                            (:name, :desc, 'join', CAST(:data AS jsonb),
                                             CAST(:keywords AS text[]), CAST(:tables AS text[]),
                                             CAST(:columns AS text[]), 2, CAST(:embedding AS vector))
                                        """),
                                        {
                                            "name": join_name,
                                            "desc": description,
                                            "data": json.dumps(rule_data),
                                            "keywords": keyword_list,
                                            "tables": [table1, table2],
                                            "columns": [column1, column2],
                                            "embedding": str(embedding)
                                        }
                                    )
                                    conn.commit()
                                
                                st.success(f"✅ Relationship '{join_name}' saved! LLM will choose appropriate join type.")
                                st.balloons()
                                
                            except Exception as e:
                                st.error(f"❌ Failed: {str(e)}")
        
        # ======================
        # FILTER RULE FORM
        # ======================
        elif rule_type == "🎯 Filter Rule":
            st.subheader("Add Filter Rule")
            
            with st.form("filter_form"):
                filter_name = st.text_input(
                    "Filter Name*",
                    placeholder="e.g., Won Opportunities"
                )
                
                user_terms = st.text_input(
                    "User Terms*",
                    placeholder="e.g., won, sold, closed, successful"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    table_name = st.selectbox(
                        "Table*",
                        options=list(selected_schema.keys())
                    )
                    
                    if table_name:
                        column_name = st.selectbox(
                            "Column*",
                            options=selected_schema[table_name]
                        )
                    else:
                        column_name = None
                
                with col2:
                    operator = st.selectbox(
                        "Operator*",
                        options=["IN", "NOT IN", "=", "!=", ">", "<", ">=", "<=", "LIKE", "BETWEEN"]
                    )
                    
                    values = st.text_input(
                        "Values (comma-separated)*",
                        placeholder="e.g., Sold, OASSubmitted"
                    )
                
                description = st.text_area(
                    "Description",
                    placeholder="When to apply this filter?"
                )
                
                col_test, col_save = st.columns(2)
                
                with col_test:
                    test_filter = st.form_submit_button("🧪 Test", use_container_width=True)
                
                with col_save:
                    save_filter = st.form_submit_button("💾 Save", type="primary", use_container_width=True)
                
                if test_filter or save_filter:
                    if not filter_name or not user_terms or not table_name or not column_name or not values:
                        st.error("❌ Fill all required fields")
                    else:
                        value_list = [v.strip() for v in values.split(",")]
                        keywords = [k.strip() for k in user_terms.split(",")]
                        
                        # Build SQL pattern
                        if operator in ["IN", "NOT IN"]:
                            values_str = ", ".join([f"'{v}'" for v in value_list])
                            sql_pattern = f'"{table_name}"."{column_name}" {operator} ({values_str})'
                        elif operator == "BETWEEN":
                            sql_pattern = f'"{table_name}"."{column_name}" BETWEEN \'{value_list[0]}\' AND \'{value_list[1]}\''
                        else:
                            sql_pattern = f'"{table_name}"."{column_name}" {operator} \'{value_list[0]}\''
                        
                        rule_data = {
                            "filter_name": filter_name,
                            "user_terms": keywords,
                            "table": table_name,
                            "column": column_name,
                            "operator": operator,
                            "values": value_list,
                            "sql_pattern": sql_pattern
                        }
                        
                        if test_filter:
                            st.success("✅ Filter valid!")
                            st.code(sql_pattern, language="sql")
                        
                        if save_filter:
                            try:
                                embedding_text = f"{filter_name} {description} {user_terms}"
                                embedding = get_embedding(embedding_text)
                                
                                with VECTOR_ENGINE.connect() as conn:
                                    conn.execute(
                                        text("""
                                            INSERT INTO business_rules_v2
                                            (rule_name, rule_description, rule_type, rule_data,
                                             trigger_keywords, applies_to_tables, applies_to_columns,
                                             priority, embedding)
                                            VALUES
                                            (:name, :desc, 'filter', CAST(:data AS jsonb),
                                             CAST(:keywords AS text[]), CAST(:tables AS text[]),
                                             CAST(:columns AS text[]), 2, CAST(:embedding AS vector))
                                        """),
                                        {
                                            "name": filter_name,
                                            "desc": description,
                                            "data": json.dumps(rule_data),
                                            "keywords": keywords,
                                            "tables": [table_name],
                                            "columns": [column_name],
                                            "embedding": str(embedding)
                                        }
                                    )
                                    conn.commit()
                                
                                st.success(f"✅ Filter '{filter_name}' saved!")
                                st.balloons()
                                
                            except Exception as e:
                                st.error(f"❌ Failed: {str(e)}")
        
        # ======================
        # CRITICAL DEFAULT RULE
        # ======================
        elif rule_type == "⚠️ Critical Default Rule":
            st.subheader("Add Critical Default Rule")
            st.caption("These rules are ALWAYS applied automatically")
            
            with st.form("default_form"):
                rule_name = st.text_input(
                    "Rule Name*",
                    placeholder="e.g., Current FY Filter"
                )
                
                description = st.text_area(
                    "Description*",
                    placeholder="What does this rule do and when is it applied?"
                )
                
                applies_to_queries = st.text_input(
                    "Apply to queries containing (keywords)*",
                    placeholder="e.g., sales, revenue, pipeline"
                )
                
                sql_pattern = st.text_area(
                    "SQL Pattern*",
                    placeholder="e.g., Date >= '2024-04-01' AND Date <= '2025-03-31'",
                    help="Use placeholders like {fy_start}, {fy_end} for dynamic values"
                )
                
                auto_apply = st.checkbox(
                    "⚠️ Always include this rule (mandatory)",
                    value=True
                )
                
                col_test, col_save = st.columns(2)
                
                with col_test:
                    test_default = st.form_submit_button("🧪 Test", use_container_width=True)
                
                with col_save:
                    save_default = st.form_submit_button("💾 Save", type="primary", use_container_width=True)
                
                if test_default or save_default:
                    if not rule_name or not description or not sql_pattern:
                        st.error("❌ Fill all required fields")
                    else:
                        keywords = [k.strip() for k in applies_to_queries.split(",")] if applies_to_queries else []
                        
                        rule_data = {
                            "default_name": rule_name,
                            "description": description,
                            "auto_apply": auto_apply,
                            "applies_to_queries": keywords,
                            "sql_pattern": sql_pattern
                        }
                        
                        if test_default:
                            st.success("✅ Rule structure valid!")
                            st.json(rule_data)
                        
                        if save_default:
                            try:
                                # USE UNIVERSAL SMART KEYWORD EXTRACTION
                                manual_keywords = [k.strip() for k in applies_to_queries.split(",")] if applies_to_queries else []
                                
                                auto_keywords = extract_smart_keywords(
                                    rule_name=rule_name,
                                    description=description,
                                    rule_type="default",
                                    rule_data=rule_data
                                )
                                
                                # Combine manual + auto keywords
                                all_keywords = list(set(manual_keywords) | auto_keywords)
                                
                                # Auto-enhance description
                                enhanced_description = enhance_description_with_keywords(
                                    description,
                                    auto_keywords
                                )
                                
                                embedding_text = f"{rule_name} {enhanced_description} {' '.join(all_keywords)}"
                                embedding = get_embedding(embedding_text)
                                
                                with VECTOR_ENGINE.connect() as conn:
                                    conn.execute(
                                        text("""
                                            INSERT INTO business_rules_v2
                                            (rule_name, rule_description, rule_type, rule_data,
                                             trigger_keywords, priority, is_mandatory, embedding)
                                            VALUES
                                            (:name, :desc, 'default', CAST(:data AS jsonb),
                                             CAST(:keywords AS text[]), 1, :mandatory, CAST(:embedding AS vector))
                                        """),
                                        {
                                            "name": rule_name,
                                            "desc": enhanced_description,
                                            "data": json.dumps(rule_data),
                                            "keywords": all_keywords,
                                            "mandatory": auto_apply,
                                            "embedding": str(embedding)
                                        }
                                    )
                                    conn.commit()
                                
                                st.success(f"✅ Critical rule '{rule_name}' saved!")
                                st.info(f"🔍 Auto-generated {len(auto_keywords)} additional keywords: {', '.join(list(auto_keywords)[:10])}...")
                                st.balloons()
                                
                            except Exception as e:
                                st.error(f"❌ Failed: {str(e)}")
        
        # ======================
        # VALUE MAPPING FORM
        # ======================
        elif rule_type == "🔄 Value Mapping":
            st.subheader("Add Value Mapping")
            
            with st.form("mapping_form"):
                mapping_name = st.text_input(
                    "Mapping Name*",
                    placeholder="e.g., Quarter to Months"
                )
                
                st.write("**Define Mappings**")
                
                num_mappings = st.number_input("Number of mappings", min_value=1, max_value=20, value=4)
                
                mappings = {}
                for i in range(num_mappings):
                    col1, col2 = st.columns(2)
                    with col1:
                        user_term = st.text_input(f"User says", key=f"user_term_{i}", placeholder="e.g., Q1")
                    with col2:
                        db_value = st.text_input(f"Database has", key=f"db_value_{i}", placeholder="e.g., April,May,June")
                    
                    if user_term and db_value:
                        mappings[user_term] = db_value.split(",")
                
                description = st.text_area("Description")
                
                col_test, col_save = st.columns(2)
                
                with col_test:
                    test_mapping = st.form_submit_button("🧪 Test", use_container_width=True)
                
                with col_save:
                    save_mapping = st.form_submit_button("💾 Save", type="primary", use_container_width=True)
                
                if test_mapping or save_mapping:
                    if not mapping_name or not mappings:
                        st.error("❌ Add at least one mapping")
                    else:
                        # AUTO-GENERATE comprehensive keywords from mapping
                        auto_keywords = set()
                        all_user_terms = []
                        all_db_values = []
                        
                        for user_term, db_values in mappings.items():
                            # Split "User says" by comma and clean
                            user_variations = [t.strip().lower() for t in user_term.split(',')]
                            all_user_terms.extend(user_variations)
                            auto_keywords.update(user_variations)
                            
                            # Add database values
                            for db_val in db_values:
                                cleaned = db_val.strip().lower()
                                all_db_values.append(cleaned)
                                auto_keywords.add(cleaned)
                        
                        # Add mapping name as keyword too
                        auto_keywords.add(mapping_name.lower())
                        
                        # AUTO-ENHANCE description
                        auto_context = f"""

[AUTO-GENERATED SEARCH CONTEXT]
This mapping handles user terms: {', '.join(sorted(set(all_user_terms)))}
Maps to database values: {', '.join(sorted(set(all_db_values)))}
When user mentions any of these terms, this mapping applies.
Searchable keywords: {' '.join(sorted(auto_keywords))}"""
                        
                        enhanced_description = (description or "User-defined mapping") + auto_context
                        
                        rule_data = {
                            "mapping_type": "value_mapping",
                            "mappings": mappings,
                            "auto_enhanced": True
                        }
                        
                        if test_mapping:
                            st.success("✅ Mapping valid!")
                            st.write("**Your Mappings:**")
                            st.json(mappings)
                            st.write("**Auto-generated keywords for RAG:**")
                            st.code(', '.join(sorted(auto_keywords)))
                            st.info(f"💡 RAG will be able to find this rule when user mentions: {', '.join(list(auto_keywords)[:10])}...")
                        
                        if save_mapping:
                            try:
                                # USE UNIVERSAL SMART KEYWORD EXTRACTION
                                auto_keywords = extract_smart_keywords(
                                    rule_name=mapping_name,
                                    description=description or "",
                                    rule_type="mapping",
                                    rule_data=rule_data
                                )
                                
                                # Auto-enhance description
                                enhanced_description = enhance_description_with_keywords(
                                    description or "",
                                    auto_keywords
                                )
                                
                                keywords_list = list(auto_keywords)
                                embedding_text = f"{mapping_name} {enhanced_description} {' '.join(keywords_list)}"
                                embedding = get_embedding(embedding_text)
                                
                                with VECTOR_ENGINE.connect() as conn:
                                    conn.execute(
                                        text("""
                                            INSERT INTO business_rules_v2
                                            (rule_name, rule_description, rule_type, rule_data,
                                             trigger_keywords, priority, embedding)
                                            VALUES
                                            (:name, :desc, 'mapping', CAST(:data AS jsonb),
                                             CAST(:keywords AS text[]), 2, CAST(:embedding AS vector))
                                        """),
                                        {
                                            "name": mapping_name,
                                            "desc": enhanced_description,
                                            "data": json.dumps(rule_data),
                                            "keywords": keywords_list,
                                            "embedding": str(embedding)
                                        }
                                    )
                                    conn.commit()
                                
                                st.success(f"✅ Mapping '{mapping_name}' saved!")
                                st.info(f"🔍 RAG will find this rule when queries mention: {', '.join(keywords_list[:8])}...")
                                st.balloons()
                                
                            except Exception as e:
                                st.error(f"❌ Failed: {str(e)}")
        
        # ======================
        # TABLE/COLUMN DESCRIPTION
        # ======================
        elif rule_type == "📖 Table/Column Description":
            st.subheader("Add Table/Column Description")
            st.caption("Help RAG understand what your data means")
            
            # Radio button OUTSIDE form (so it can control what's shown)
            desc_type = st.radio(
                "What are you describing?",
                ["Table", "Column"],
                key="desc_type_radio"
            )
            
            if desc_type == "Table":
                # TABLE DESCRIPTION FORM
                with st.form("table_description_form"):
                    table_name = st.selectbox(
                        "Table*",
                        options=list(selected_schema.keys()),
                        key="table_desc_table"
                    )
                    
                    table_description = st.text_area(
                        "What this table contains*",
                        placeholder="e.g., Orders and transactions from ERP system",
                        help="Describe what kind of data this table stores"
                    )
                    
                    user_terms = st.text_input(
                        "Common user terms for this table (comma-separated)",
                        placeholder="e.g., sales, orders, revenue, transactions",
                        help="Words users might use to refer to this table"
                    )
                    
                    save_desc = st.form_submit_button("💾 Save Table Description", type="primary")
                    
                    if save_desc:
                        if not table_description:
                            st.error("❌ Add description")
                        else:
                            try:
                                # Manual keywords from user
                                manual_keywords = [k.strip() for k in user_terms.split(",")] if user_terms else []
                                
                                # USE SMART KEYWORD EXTRACTION
                                rule_data = {
                                    "description_type": "table",
                                    "table": table_name,
                                    "description": table_description,
                                    "user_terms": manual_keywords
                                }
                                
                                auto_keywords = extract_smart_keywords(
                                    rule_name=f"Description: {table_name}",
                                    description=table_description,
                                    rule_type="term_alias",
                                    rule_data=rule_data
                                )
                                
                                # Combine manual + auto keywords
                                all_keywords = list(set(manual_keywords) | auto_keywords)
                                
                                # Auto-enhance description
                                enhanced_description = enhance_description_with_keywords(
                                    table_description,
                                    auto_keywords
                                )
                                
                                embedding_text = f"{table_name} {enhanced_description} {' '.join(all_keywords)}"
                                embedding = get_embedding(embedding_text)
                                
                                with VECTOR_ENGINE.connect() as conn:
                                    conn.execute(
                                        text("""
                                            INSERT INTO business_rules_v2
                                            (rule_name, rule_description, rule_type, rule_data,
                                             trigger_keywords, applies_to_tables, priority, embedding)
                                            VALUES
                                            (:name, :desc, 'term_alias', CAST(:data AS jsonb),
                                             CAST(:keywords AS text[]), CAST(:tables AS text[]), 3, 
                                             CAST(:embedding AS vector))
                                        """),
                                        {
                                            "name": f"Description: {table_name}",
                                            "desc": enhanced_description,
                                            "data": json.dumps(rule_data),
                                            "keywords": all_keywords,
                                            "tables": [table_name],
                                            "embedding": str(embedding)
                                        }
                                    )
                                    conn.commit()
                                
                                st.success(f"✅ Description for '{table_name}' saved!")
                                st.info(f"🔍 Auto-generated {len(auto_keywords)} additional keywords: {', '.join(list(auto_keywords)[:10])}...")
                                
                            except Exception as e:
                                st.error(f"❌ Failed: {str(e)}")
            
            else:  # COLUMN DESCRIPTION FORM
                with st.form("column_description_form"):
                    table_name = st.selectbox(
                        "Table*",
                        options=list(selected_schema.keys()),
                        key="col_desc_table"
                    )
                    
                    # Column dropdown
                    column_name = st.selectbox(
                        "Column*",
                        options=selected_schema.get(table_name, []) if table_name else [],
                        key="col_desc_column"
                    )
                    
                    column_description = st.text_area(
                        "What this column means*",
                        placeholder="e.g., Employee profit margin in rupees (not converted to lakhs yet)",
                        help="Explain what this column represents and any important details"
                    )
                    
                    user_terms = st.text_input(
                        "User might call it (comma-separated)",
                        placeholder="e.g., margin, profit, sales, revenue",
                        help="Words users might use to refer to this column"
                    )
                    
                    save_col_desc = st.form_submit_button("💾 Save Column Description", type="primary")
                    
                    if save_col_desc:
                        if not table_name or not column_name:
                            st.error("❌ Select table and column")
                        elif not column_description:
                            st.error("❌ Add description")
                        else:
                            try:
                                # Manual keywords from user
                                manual_keywords = [k.strip() for k in user_terms.split(",")] if user_terms else []
                                
                                # USE SMART KEYWORD EXTRACTION
                                rule_data = {
                                    "description_type": "column",
                                    "table": table_name,
                                    "column": column_name,
                                    "description": column_description,
                                    "user_terms": manual_keywords
                                }
                                
                                auto_keywords = extract_smart_keywords(
                                    rule_name=f"Column: {table_name}.{column_name}",
                                    description=column_description,
                                    rule_type="term_alias",
                                    rule_data=rule_data
                                )
                                
                                # Combine manual + auto keywords
                                all_keywords = list(set(manual_keywords) | auto_keywords)
                                
                                # Auto-enhance description
                                enhanced_description = enhance_description_with_keywords(
                                    column_description,
                                    auto_keywords
                                )
                                
                                embedding_text = f"{table_name} {column_name} {enhanced_description} {' '.join(all_keywords)}"
                                embedding = get_embedding(embedding_text)
                                
                                with VECTOR_ENGINE.connect() as conn:
                                    conn.execute(
                                        text("""
                                            INSERT INTO business_rules_v2
                                            (rule_name, rule_description, rule_type, rule_data,
                                             trigger_keywords, applies_to_tables, applies_to_columns,
                                             priority, embedding)
                                            VALUES
                                            (:name, :desc, 'term_alias', CAST(:data AS jsonb),
                                             CAST(:keywords AS text[]), CAST(:tables AS text[]),
                                             CAST(:columns AS text[]), 3, CAST(:embedding AS vector))
                                        """),
                                        {
                                            "name": f"Column: {table_name}.{column_name}",
                                            "desc": enhanced_description,
                                            "data": json.dumps(rule_data),
                                            "keywords": all_keywords,
                                            "tables": [table_name],
                                            "columns": [column_name],
                                            "embedding": str(embedding)
                                        }
                                    )
                                    conn.commit()
                                
                                st.success(f"✅ Description for '{table_name}.{column_name}' saved!")
                                st.info(f"🔍 Auto-generated {len(auto_keywords)} additional keywords: {', '.join(list(auto_keywords)[:10])}...")
                                
                            except Exception as e:
                                st.error(f"❌ Failed: {str(e)}")
        
        # ======================
        # QUERY EXAMPLE FORM
        # ======================
        elif rule_type == "💡 Query Example":
            st.subheader("Add Query Example (Training Data)")
            
            with st.form("example_form"):
                question = st.text_input(
                    "Example Question*",
                    placeholder="e.g., What is the total revenue by region?"
                )
                
                sql_query = st.text_area(
                    "Correct SQL*",
                    placeholder="SELECT SUM(Amount) FROM Orders WHERE Region='East'...",
                    height=150
                )
                
                explanation = st.text_area(
                    "Explanation",
                    placeholder="Why this SQL is correct? What techniques used?"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    query_type = st.selectbox(
                        "Query Complexity",
                        ["easy", "medium", "hard"]
                    )
                
                with col2:
                    tables_used = st.multiselect(
                        "Tables Used",
                        options=list(selected_schema.keys())
                    )
                
                col_test, col_save = st.columns(2)
                
                with col_test:
                    test_example = st.form_submit_button("🧪 Test SQL", use_container_width=True)
                
                with col_save:
                    save_example = st.form_submit_button("💾 Save Example", type="primary", use_container_width=True)
                
                if test_example:
                    if not sql_query:
                        st.error("❌ Enter SQL to test")
                    else:
                        try:
                            df = run_sql(st.session_state.engine, sql_query)
                            st.success(f"✅ SQL executes successfully! ({len(df)} rows)")
                            st.dataframe(df.head())
                        except Exception as e:
                            st.error(f"❌ SQL error: {str(e)}")
                
                if save_example:
                    if not question or not sql_query:
                        st.error("❌ Fill required fields")
                    else:
                        try:
                            embedding_text = f"{question} {explanation}"
                            embedding = get_embedding(embedding_text)
                            
                            with VECTOR_ENGINE.connect() as conn:
                                conn.execute(
                                    text("""
                                        INSERT INTO query_examples_v2
                                        (question, sql_query, explanation, query_type, 
                                         tables_used, is_verified, embedding)
                                        VALUES
                                        (:question, :sql, :explanation, :qtype,
                                         CAST(:tables AS text[]), TRUE, CAST(:embedding AS vector))
                                    """),
                                    {
                                        "question": question,
                                        "sql": sql_query,
                                        "explanation": explanation,
                                        "qtype": query_type,
                                        "tables": tables_used,
                                        "embedding": str(embedding)
                                    }
                                )
                                conn.commit()
                            
                            st.success("✅ Example saved!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Failed: {str(e)}")
        
        st.divider()
        
        # ======================
        # VIEW SAVED RULES
        # ======================
        st.subheader("📚 Saved Rules")
        
        try:
            with VECTOR_ENGINE.connect() as conn:
                rules_df = pd.read_sql(
                    text("""
                        SELECT rule_name, rule_type, priority, 
                               is_mandatory, created_at, id, rule_data
                        FROM business_rules_v2
                        WHERE is_active = TRUE
                        ORDER BY priority ASC, created_at DESC
                    """),
                    conn
                )
            
            if len(rules_df) > 0:
                # Separate auto-generated rules from user rules
                auto_rules = []
                user_rules = []
                
                for idx, row in rules_df.iterrows():
                    try:
                        rule_data = json.loads(row['rule_data']) if isinstance(row['rule_data'], str) else row['rule_data']
                        if rule_data and rule_data.get('auto_generated'):
                            auto_rules.append(row)
                        else:
                            user_rules.append(row)
                    except:
                        user_rules.append(row)
                
                # Show auto-generated rules
                if auto_rules:
                    st.write("### 🤖 Auto-Generated Rules")
                    st.caption("These rules are automatically created by the system. Regenerate by reconnecting to database.")
                    
                    for rule in auto_rules:
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.write(f"🔴 **{rule['rule_name']}** (Critical)")
                                if rule['rule_type'] == 'dialect':
                                    st.caption("Ensures correct SQL syntax for your database type")
                            with col2:
                                if st.button("ℹ️ Info", key=f"info_{rule['id']}"):
                                    st.info("This is an auto-generated rule. To update it, reconnect to your database.")
                            st.divider()
                
                # Show user rules
                if user_rules:
                    st.write("### 📝 Your Custom Rules")
                    user_rules_df = pd.DataFrame(user_rules)
                    st.dataframe(
                        user_rules_df[['rule_name', 'rule_type', 'priority', 'is_mandatory']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.write("### 📝 Your Custom Rules")
                    st.info("No custom rules yet. Add your first rule above!")
                
                # Delete rule option (only for user rules)
                if user_rules:
                    with st.expander("🗑️ Delete Rules"):
                        user_rule_names = [r['rule_name'] for r in user_rules]
                        rule_to_delete = st.selectbox(
                            "Select rule to delete",
                            options=user_rule_names,
                            key="delete_rule_select"
                        )
                        
                        if st.button("🗑️ Delete Rule", type="secondary"):
                            try:
                                with VECTOR_ENGINE.connect() as conn:
                                    conn.execute(
                                        text("DELETE FROM business_rules_v2 WHERE rule_name = :name"),
                                        {"name": rule_to_delete}
                                    )
                                    conn.commit()
                                
                                st.success(f"✅ Deleted '{rule_to_delete}'")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Failed: {str(e)}")
                
                # Edit rule option
                if user_rules:
                    with st.expander("✏️ View/Edit Rule Details"):
                        user_rule_names = [r['rule_name'] for r in user_rules]
                        rule_to_edit = st.selectbox(
                            "Select rule to view",
                            options=user_rule_names,
                            key="edit_rule_select"
                    )
                    
                    if st.button("👁️ View Details"):
                        try:
                            with VECTOR_ENGINE.connect() as conn:
                                result = conn.execute(
                                    text("""
                                        SELECT rule_name, rule_type, rule_description, 
                                               rule_data, trigger_keywords, priority, is_mandatory
                                        FROM business_rules_v2
                                        WHERE rule_name = :name
                                    """),
                                    {"name": rule_to_edit}
                                )
                                rule = result.fetchone()
                            
                            if rule:
                                st.write(f"**Name:** {rule[0]}")
                                st.write(f"**Type:** {rule[1]}")
                                st.write(f"**Description:** {rule[2]}")
                                st.write(f"**Priority:** {rule[5]} {'(🔴 Critical)' if rule[6] else ''}")
                                st.write(f"**Keywords:** {', '.join(rule[4]) if rule[4] else 'None'}")
                                
                                st.write("**Rule Data (JSON):**")
                                try:
                                    rule_data = json.loads(rule[3]) if isinstance(rule[3], str) else rule[3]
                                    st.json(rule_data)
                                except:
                                    st.code(rule[3])
                                
                                st.info("💡 To edit: Delete this rule and create a new one with updated values")
                                
                        except Exception as e:
                            st.error(f"❌ Failed: {str(e)}")
            else:
                st.info("No rules saved yet. Add your first rule above!")
                
        except Exception as e:
            st.warning(f"Could not load rules: {str(e)}")

# ======================
# TAB 3: TEST QUERIES
# ======================
with tab3:
    st.markdown("""
    <style>
    .chart-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
    .chart-question-label {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .chart-question-text {
        font-size: 14px;
        color: #475569;
        margin-bottom: 16px;
        font-style: italic;
    }
    .row-count-badge {
        display: inline-block;
        background: #f0fdf4;
        color: #16a34a;
        border: 1px solid #bbf7d0;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 13px;
        font-weight: 500;
        padding: 6px 16px;
        border-radius: 6px 6px 0 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("🔍 Test Queries")
    
    if "engine" not in st.session_state:
        st.warning("👈 Connect to database first")
    elif "selected_objects" not in st.session_state or not st.session_state.selected_objects:
        st.warning("👈 Select tables/views first")
    else:
        # Import optimization modules
        from flow_router import process_query, FlowConfig, QueryResult
        from query_classifier import classify_query, get_flow_config
        
        # Build schema for reference
        inspector = inspect(st.session_state.engine)
        selected_schema = {}
        
        for obj in st.session_state.selected_objects:
            _, full_name = obj.split(": ")
            if "." in full_name:
                schema, table = full_name.split(".")
            else:
                schema = "public"
                table = full_name
            cols = inspector.get_columns(table, schema=schema)
            selected_schema[full_name] = [c["name"] for c in cols]
        
        # ═══════════════════════════════════════════════════════════════════
        # OPTIMIZATION SETTINGS
        # ═══════════════════════════════════════════════════════════════════
        with st.expander("⚙️ Optimization Settings", expanded=False):
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                st.write("**Query Classification**")
                enable_classification = st.checkbox(
                    "Enable Llama Classification", 
                    value=True,
                    help="Classify queries as Easy/Medium/Hard to optimize tokens"
                )
            
            with col_opt2:
                st.write("**RAG Settings**")
                enable_rule_rag = st.checkbox(
                    "Enable Rule RAG", 
                    value=True,
                    help="Retrieve business rules for the query"
                )
                enable_opus_descriptions = st.checkbox(
                    "Use Opus Descriptions", 
                    value=True,
                    help="Include AI-generated column descriptions (requires enrichment in Tab 1)"
                )
                compress_rules = st.checkbox(
                    "Compress Rules", 
                    value=True,
                    help="Use compact JSON format (~70% token savings)"
                )
                enable_resolver = st.checkbox(
                    "🔍 Entity Resolver",
                    value=True,
                    help="Resolve fuzzy entity names (Dell → Dell Pvt Ltd) against live DB at query time. "
                         "Adds ~50-200ms per entity but dramatically improves filter accuracy."
                )
            
            with col_opt3:
                st.write("**Validation & Cache**")
                enable_sql_validation = st.checkbox(
                    "Pre-validate SQL", 
                    value=True,
                    help="Catch syntax errors before execution (free)"
                )
                enable_cache = st.checkbox(
                    "Enable Query Cache",
                    value=True,
                    help="Cache SQL results - same question = instant response, zero tokens"
                )
                enable_semantic_cache = st.checkbox(
                    "Semantic Cache",
                    value=True,
                    help="Match similar questions (e.g., 'show sales' ≈ 'display sales')",
                    disabled=not enable_cache
                )
                enable_charts = st.checkbox(
                    "📊 Display Charts",
                    value=True,
                    help="AI generates a Plotly chart tailored to your data and question"
                )

            
            with st.expander("🎯 Independent Review", expanded=False):
                col_rev1, col_rev2 = st.columns(2)
                
                with col_rev1:
                    opus_mode = st.radio(
                        "Review Mode",
                        ["Disabled", "Auto (Hard only)", "Always"],
                        index=1,
                        horizontal=False
                    )
                    opus_config = {"Disabled": False, "Auto (Hard only)": "auto", "Always": True}[opus_mode]
                
                with col_rev2:
                    reviewer_provider = st.selectbox(
                        "Reviewer LLM",
                        ["Claude Opus", "Qwen Coder (Vertex)"],
                        index=0,
                        help="Choose which LLM reviews the SQL",
                        disabled=(opus_mode == "Disabled")
                    )
                    reviewer_provider_map = {
                        "Claude Opus": "claude_opus",
                        "Qwen Coder (Vertex)": "vertex_qwen_thinking"
                    }
                    selected_reviewer = reviewer_provider_map[reviewer_provider]
        
        # ═══════════════════════════════════════════════════════════════════
        # QUERY INPUT
        # ═══════════════════════════════════════════════════════════════════
        st.divider()
        
        question = st.text_input(
            "Your Question",
            placeholder="e.g., Show total sales by region excluding rebates",
            key="user_question_optimized"
        )
        
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            generate_clicked = st.button("🚀 Generate SQL", type="primary", use_container_width=True)
        with col_info:
            if enable_classification:
                st.caption("💡 Query will be classified → optimized flow → SQL generated")
            else:
                st.caption("💡 Using standard flow for all queries")
        
        # ═══════════════════════════════════════════════════════════════════
        # MAIN PROCESSING
        # ═══════════════════════════════════════════════════════════════════
        if generate_clicked:
            if not question:
                st.error("❌ Please enter a question")
            else:
                st.session_state.query_counter += 1
                st.session_state['last_chart_tokens'] = {"input": 0, "output": 0}  # Reset per query

                # Get dialect
                dialect = st.session_state.get('detected_dialect', 'postgresql')
                
                # Create config
                config = FlowConfig(
                    enable_classification=enable_classification,
                    classification_provider="groq",
                    enable_rule_rag=enable_rule_rag,
                    enable_opus_descriptions=enable_opus_descriptions,
                    enable_cache=enable_cache,
                    enable_semantic_cache=enable_semantic_cache if enable_cache else False,
                    enable_resolver=enable_resolver,
                    compress_rules=compress_rules,
                    validate_sql=enable_sql_validation,
                    auto_fix_sql=True,
                    enable_opus=opus_config,
                    opus_provider=selected_reviewer,
                    reasoning_provider=st.session_state.get('reasoning_provider', 'claude_sonnet'),
                    sql_provider=st.session_state.get('coding_provider', 'groq'),
                    dialect=dialect,
                    dialect_info={
                        "dialect": dialect,
                        "quote_char": '"' if dialect in ['postgresql', 'oracle'] else '`',
                        "string_quote": "'"
                    }
                )
                
                # ───────────────────────────────────────────────────────────
                # STEP 1: CLASSIFICATION
                # ───────────────────────────────────────────────────────────
                if enable_classification:
                    with st.spinner("🏷️ Classifying query..."):
                        classification = classify_query(question, use_llm=False)
                    
                    complexity = classification["complexity"]
                    complexity_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}[complexity]
                    flow_cfg = get_flow_config(complexity)
                    
                    col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
                    col_c1.metric("Complexity", f"{complexity_emoji} {complexity.upper()}")
                    col_c2.caption(f"**Reason:** {classification['reason']}")
                    col_c3.caption(f"**Est. tokens:** ~{flow_cfg['expected_tokens']}")
                else:
                    complexity = "medium"
                
                # ───────────────────────────────────────────────────────────
                # STEP 2: PROCESS QUERY
                # ───────────────────────────────────────────────────────────
                with st.spinner(f"⚙️ Processing ({complexity.upper()} flow)..."):
                    result = process_query(
                        question=question,
                        engine=st.session_state.engine,
                        vector_engine=VECTOR_ENGINE,
                        selected_tables=st.session_state.selected_objects,
                        config=config
                    )
                
                # ───────────────────────────────────────────────────────────
                # DISPLAY RESULTS
                # ───────────────────────────────────────────────────────────
                st.divider()
                st.subheader("📊 Processing Summary")
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Complexity", result.complexity.upper())
                col_s2.metric("Total Tokens", f"{result.tokens.total_tokens():,}")
                col_s3.metric("Time", f"{result.total_time_ms}ms")
                
                # Show cache hit or success status
                if result.cache_hit:
                    col_s4.metric("Status", f"⚡ Cache Hit ({result.cache_hit_type})")
                else:
                    col_s4.metric("Status", "✅ Success" if result.success else "❌ Failed")
                
                st.caption(f"**Flow:** {result.flow_path}")
                
                # ───────────────────────────────────────────────────────────
                # TOKEN BREAKDOWN
                # ───────────────────────────────────────────────────────────
                with st.expander("💰 Token Usage Breakdown", expanded=True):
                    tokens = result.tokens
                    token_data = []

                    stage_rows = [
                        ("🏷️ Classifier", tokens.classifier),
                        ("🧠 Reasoning", tokens.reasoning),
                        ("🧠 Reasoning Pass 1", tokens.reasoning_pass1),
                        ("🧠 Reasoning Pass 2", tokens.reasoning_pass2),
                        ("🧠 Opus Complex", tokens.opus_complex),
                        ("⚙️ SQL Gen", tokens.sql_gen),
                        ("🎯 Opus", tokens.opus),
                        ("🔧 Refinement", tokens.refinement),
                        ("🩹 Error Fix (Reasoning)", tokens.error_fix_reasoning),
                        ("🩹 Error Fix (Opus)", tokens.error_fix_opus),
                        ("📊 Chart Builder", tokens.chart),
                    ]

                    for stage_label, stage_tokens in stage_rows:
                        if stage_tokens["input"] + stage_tokens["output"] > 0:
                            token_data.append({
                                "Stage": stage_label,
                                "Input": stage_tokens["input"],
                                "Output": stage_tokens["output"]
                            })
                    
                    # Show resolver stats (no LLM tokens — DB query time)
                    if hasattr(result, 'resolver_result') and result.resolver_result:
                        resolver_r = result.resolver_result
                        token_data.append({
                            "Stage": f"🔍 Resolver ({resolver_r.queries_run} queries, {resolver_r.total_time_ms}ms)",
                            "Input": 0,
                            "Output": 0
                        })

                    if token_data:
                        for row in token_data:
                            row["Total"] = row["Input"] + row["Output"]
                        
                        df_tokens = pd.DataFrame(token_data)
                        grand_total = tokens.total()
                        total_row = {
                            "Stage": "**TOTAL**",
                            "Input": grand_total["input"],
                            "Output": grand_total["output"],
                            "Total": grand_total["input"] + grand_total["output"]
                        }
                        df_tokens = pd.concat([df_tokens, pd.DataFrame([total_row])], ignore_index=True)
                        st.dataframe(df_tokens, use_container_width=True, hide_index=True)

                        if hasattr(result, 'resolver_result') and result.resolver_result:
                            st.caption("ℹ️ Resolver runs DB lookups only, so it shows activity/time but 0 LLM tokens.")
                        
                        # Savings calculation
                        baseline = 15000
                        actual = total_row["Total"]
                        if actual < baseline:
                            savings_pct = ((baseline - actual) / baseline) * 100
                            st.success(f"💰 **Saved ~{baseline - actual:,} tokens ({savings_pct:.0f}%)** vs baseline")
                
                # ───────────────────────────────────────────────────────────
                # CONTEXT RETRIEVED
                # ───────────────────────────────────────────────────────────
                with st.expander("📚 Context Retrieved", expanded=False):
                    col_ctx1, col_ctx2 = st.columns(2)
                    with col_ctx1:
                        st.metric("Rules Retrieved", result.rules_retrieved)
                        if result.rules_compressed and result.rules_compressed != "[]":
                            # Show compressed preview
                            st.caption("**Compressed Rules (sent to LLM):**")
                            st.code(result.rules_compressed[:500] + "..." if len(result.rules_compressed) > 500 else result.rules_compressed, language="json")
                            
                            # Try to parse and show rule names
                            try:
                                import json
                                rules_list = json.loads(result.rules_compressed)
                                if rules_list:
                                    st.caption("**Rules Applied:**")
                                    for r in rules_list[:10]:
                                        rule_name = r.get("name", r.get("rule_name", "Unknown"))
                                        rule_type = r.get("type", r.get("rule_type", ""))
                                        st.write(f"• **{rule_name}** ({rule_type})")
                            except:
                                pass
                        else:
                            st.info("No business rules retrieved")
                            
                    with col_ctx2:
                        st.metric("Schema Mode", "Full Schema" if result.columns_retrieved == -1 else f"{result.columns_retrieved} columns")
                        if result.schema_text:
                            st.code(result.schema_text[:800] + "..." if len(result.schema_text) > 800 else result.schema_text)
                
                # ───────────────────────────────────────────────────────────
                # VALIDATION
                # ───────────────────────────────────────────────────────────
                if result.validation_result:
                    with st.expander("✅ SQL Validation", expanded=not result.validation_result.get("is_valid", True)):
                        if result.validation_result.get("is_valid"):
                            st.success("SQL passed validation")
                        else:
                            st.error("Validation issues found")
                            for issue in result.validation_result.get("issues", []):
                                st.write(f"- ❌ {issue}")
                        
                        if result.sql_fixed:
                            st.info(f"**Auto-fixes applied:** {', '.join(result.fixes_applied)}")
                
                # ───────────────────────────────────────────────────────────
                # GENERATED SQL
                # ───────────────────────────────────────────────────────────
                st.divider()
                st.subheader("💻 Generated SQL")
                st.code(result.sql, language="sql")
                
                
                # ───────────────────────────────────────────────────────────
                # OPUS REVIEW
                # ───────────────────────────────────────────────────────────
                if result.opus_review:
                    st.divider()
                    st.subheader("🎯 Opus Review")
                    
                    verdict = result.final_verdict
                    confidence = result.opus_review.get("confidence", 0)
                    
                    if verdict == "CORRECT":
                        st.success(f"✅ **CORRECT** (Confidence: {confidence:.0%})")
                    elif verdict == "UNCERTAIN":
                        st.warning(f"⚠️ **UNCERTAIN** (Confidence: {confidence:.0%})")
                    else:
                        st.error(f"❌ **{verdict}**")
                    
                    if result.opus_review.get("reasoning"):
                        st.write(f"**Reasoning:** {result.opus_review['reasoning']}")
                    
                    if result.final_verdict in ("INCORRECT", "FAILED_AFTER_RETRIES"):
                        if getattr(result, 'opus_blocked_soft', False):
                            st.warning("⚠️ Results shown with caution — Opus flagged a logic issue but data is available")

                    # Show the SQL Opus reviewed — guaranteed to match result.sql
                    # because flow_router now runs Opus AFTER all validation and
                    # auto-fix stages complete. No more stale-draft problem.
                    st.write("**📋 SQL reviewed by Opus:**")
                    st.code(result.sql, language="sql")
                
                # ───────────────────────────────────────────────────────────
                # QUERY RESULTS
                # ───────────────────────────────────────────────────────────
                st.divider()
                st.subheader("📈 Query Results")

                if result.success:
                    if result.results is not None and hasattr(result.results, 'shape'):

                        st.markdown(
                            f'<div class="row-count-badge">'
                            f'✅ {len(result.results)} rows &nbsp;·&nbsp; '
                            f'{len(result.results.columns)} columns'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        if getattr(result, 'opus_blocked_soft', False):
                            st.warning(
                                "⚠️ **Review recommended** — Opus flagged a potential logic issue. "
                                "Results are shown but may not fully match your question. "
                                "Check the Opus Review section above."
                            )
                        if enable_charts and not result.results.empty:
                            chart_tab, data_tab = st.tabs(["📊 Chart", "📋 Data"])

                            with chart_tab:
                                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                                st.markdown(
                                    f'<div class="chart-question-label">Query</div>'
                                    f'<div class="chart-question-text">{question}</div>',
                                    unsafe_allow_html=True
                                )

                                try:
                                    from chart_builder import build_and_render_chart

                                    chart_rendered, chart_tokens = build_and_render_chart(
                                        df=result.results,
                                        question=question,
                                        llm_provider=st.session_state.get('reasoning_provider', 'claude_sonnet')
                                    )

                                    if chart_tokens:
                                        result.tokens.chart = {
                                            "input": chart_tokens.get("input", 0),
                                            "output": chart_tokens.get("output", 0)
                                        }

                                    if not chart_rendered:
                                        st.info("📋 No suitable chart for this result — showing table")
                                        st.dataframe(result.results, use_container_width=True)

                                    # Show chart token usage directly below chart
                                    if chart_tokens and chart_tokens.get("input", 0) + chart_tokens.get("output", 0) > 0:
                                        st.caption(
                                            f"📊 Chart Builder: "
                                            f"{chart_tokens['input']} input · "
                                            f"{chart_tokens['output']} output · "
                                            f"**{chart_tokens['input'] + chart_tokens['output']} total tokens**"
                                        )

                                except ImportError:
                                    st.warning("⚠️ chart_builder.py not found in project directory.")
                                    st.dataframe(result.results, use_container_width=True)
                                except Exception as e:
                                    print(f"[CHART] Unexpected error: {e}")
                                    st.dataframe(result.results, use_container_width=True)

                                st.markdown('</div>', unsafe_allow_html=True)

                                try:
                                    import io
                                    csv_buf = io.StringIO()
                                    result.results.to_csv(csv_buf, index=False)
                                    st.download_button(
                                        label="⬇️ Download CSV",
                                        data=csv_buf.getvalue(),
                                        file_name=f"result_{result.complexity}.csv",
                                        mime="text/csv"
                                    )
                                except Exception:
                                    pass

                            with data_tab:
                                st.dataframe(result.results, use_container_width=True)
                                try:
                                    import io
                                    csv_buf = io.StringIO()
                                    result.results.to_csv(csv_buf, index=False)
                                    st.download_button(
                                        label="⬇️ Download CSV",
                                        data=csv_buf.getvalue(),
                                        file_name=f"result_{result.complexity}.csv",
                                        mime="text/csv",
                                        key="csv_data_tab"
                                    )
                                except Exception:
                                    pass

                        else:
                            st.dataframe(result.results, use_container_width=True)
                            try:
                                import io
                                csv_buf = io.StringIO()
                                result.results.to_csv(csv_buf, index=False)
                                st.download_button(
                                    label="⬇️ Download CSV",
                                    data=csv_buf.getvalue(),
                                    file_name="result.csv",
                                    mime="text/csv"
                                )
                            except Exception:
                                pass

                    elif result.results is not None:
                        st.write(result.results)
                    else:
                        st.info("Query executed but returned no results")

                else:
                    error_msg = result.error or ""
                    if error_msg.startswith("OPUS_BLOCKED:"):
                        # Opus flagged the result as logically incorrect
                        # Show the reasoning as a friendly explanation
                        opus_explanation = error_msg.replace("OPUS_BLOCKED:", "").strip()
                        st.warning("⚠️ **This query cannot be answered accurately with the available data**")
                        st.markdown(
                            f"""
                            <div style="
                                background: #fffbeb;
                                border: 1px solid #fde68a;
                                border-left: 4px solid #f59e0b;
                                border-radius: 8px;
                                padding: 16px 20px;
                                margin-top: 8px;
                            ">
                                <div style="font-size:13px; color:#92400e; font-weight:600; margin-bottom:6px;">
                                    🎯 Why this happened
                                </div>
                                <div style="font-size:14px; color:#78350f; line-height:1.6;">
                                    {opus_explanation}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.info("💡 Try adding more specific time period details, or contact your data team if the required data relationship doesn't exist.")
                    else:
                        st.error(f"❌ Query failed: {error_msg}")
                
                # ───────────────────────────────────────────────────────────
                # TIMING & DEBUG
                # ───────────────────────────────────────────────────────────
                with st.expander("⏱️ Timing & Debug", expanded=False):
                    st.write("**Stage Times:**")
                    for stage, time_ms in result.stage_times.items():
                        st.write(f"- {stage}: {time_ms}ms")
                    
                    st.write("**Stages Completed:**", result.stages_completed)
                
                # ───────────────────────────────────────────────────────────
                # LLM INPUT/OUTPUT VIEWER
                # ───────────────────────────────────────────────────────────
                with st.expander("🔍 LLM Input/Output (Debug)", expanded=False):
                    st.caption("💡 Full prompts and responses - no truncation")
                    
                    llm_tabs = st.tabs(["📊 Classifier", "🧠 Reasoning", "🔍 Resolver", "⚙️ SQL Coder", "🎯 Opus Review", "🔄 Refinement", "⚡ SQL Error Fix (Reasoning)", "🔧 SQL Error Fix (Opus)"])

                    
                    with llm_tabs[0]:  # Classifier
                        if result.llm_trace.classifier_input:
                            st.write("**📥 INPUT PROMPT:**")
                            st.text_area("Classifier Input", result.llm_trace.classifier_input, height=300, key="clf_in")
                            st.write("**📤 OUTPUT:**")
                            st.text_area("Classifier Output", result.llm_trace.classifier_output, height=150, key="clf_out")
                        else:
                            st.info("Classifier used keyword-based classification (no LLM call)")
                    
                    with llm_tabs[1]:  # Reasoning
                        has_pass1 = bool(result.llm_trace.reasoning_pass1_input)
                        has_pass2 = bool(result.llm_trace.reasoning_pass2_input)
                        has_legacy = bool(result.llm_trace.reasoning_input)
                        
                        if has_pass1 or has_pass2:
                            # Two-pass reasoning flow
                            if has_pass1:
                                st.write("**📥 PASS 1 — Column Identification:**")
                                st.text_area("Pass 1 Input", result.llm_trace.reasoning_pass1_input, height=300, key="pass1_in")
                                st.write("**📤 Pass 1 Output:**")
                                st.text_area("Pass 1 Output", result.llm_trace.reasoning_pass1_output, height=200, key="pass1_out")
                                st.divider()
                            if has_pass2:
                                st.write("**📥 PASS 2 — Full Plan with Metadata:**")
                                st.text_area("Pass 2 Input", result.llm_trace.reasoning_pass2_input, height=300, key="pass2_in")
                                st.write("**📤 Pass 2 Output:**")
                                st.text_area("Pass 2 Output", result.llm_trace.reasoning_pass2_output, height=200, key="pass2_out")
                        elif has_legacy:
                            # Legacy single-pass reasoning
                            st.write("**📥 INPUT PROMPT:**")
                            st.text_area("Reasoning Input", result.llm_trace.reasoning_input, height=400, key="reason_in")
                            st.write("**📤 OUTPUT:**")
                            st.text_area("Reasoning Output", result.llm_trace.reasoning_output, height=300, key="reason_out")
                        else:
                            st.info("Reasoning LLM not used for this query (SIMPLE flow skips reasoning)")
                    
                    with llm_tabs[2]:  # Resolver
                        if hasattr(result, 'resolver_result') and result.resolver_result:
                            r = result.resolver_result
                            st.write(f"**Entities resolved:** {len(r.resolutions)} | "
                                     f"**DB queries:** {r.queries_run} | "
                                     f"**Time:** {r.total_time_ms}ms | "
                                     f"**All resolved:** {'✅ Yes' if r.all_resolved else '⚠️ No'}")
                            st.divider()
                            for res in r.resolutions:
                                confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(res.confidence, "⚪")
                                st.write(f"**{res.table}.{res.column}** — User typed: \"{res.user_value}\"")
                                st.write(f"  {confidence_icon} Strategy: **{res.strategy.upper()}** → `{res.filter_condition}`")
                                if res.exact_match:
                                    st.write(f"  Exact match: \"{res.exact_match}\"")
                                elif res.partial_matches:
                                    st.write(f"  Matches ({res.match_count}): {res.partial_matches[:5]}")
                                if res.warning:
                                    st.warning(f"⚠️ {res.warning}")
                                st.caption(f"Query: `{res.query_used[:200]}`")
                                st.divider()
                        elif result.llm_trace.resolver_summary:
                            st.text_area("Resolver Summary", result.llm_trace.resolver_summary, height=300, key="resolver_summary")
                        else:
                            st.info("Entity Resolver not used (no string filters or disabled)")
                    
                    with llm_tabs[3]:  # SQL Coder
                        if result.llm_trace.sql_gen_input:
                            st.write("**📥 INPUT PROMPT:**")
                            st.text_area("SQL Coder Input", result.llm_trace.sql_gen_input, height=400, key="sql_in")
                            st.write("**📤 OUTPUT:**")
                            st.text_area("SQL Coder Output", result.llm_trace.sql_gen_output, height=200, key="sql_out")
                        else:
                            st.info("SQL Coder not used for this query (EASY flow uses Reasoning only)")
                    
                    with llm_tabs[4]:  # Opus
                        if result.llm_trace.opus_input:
                            st.write("**📥 INPUT PROMPT:**")
                            st.text_area("Opus Input", result.llm_trace.opus_input, height=400, key="opus_in")
                            st.write("**📤 OUTPUT:**")
                            st.text_area("Opus Output", result.llm_trace.opus_output, height=200, key="opus_out")
                        else:
                            st.info("Opus review not used for this query")
                    
                    with llm_tabs[5]:  # Refinement
                        if result.llm_trace.refinement_input:
                            st.write("**📥 INPUT PROMPT:**")
                            st.text_area("Refinement Input", result.llm_trace.refinement_input, height=400, key="refine_in")
                            st.write("**📤 OUTPUT:**")
                            st.text_area("Refinement Output", result.llm_trace.refinement_output, height=200, key="refine_out")
                        else:
                            st.info("Refinement not triggered (query was correct or Opus not enabled)")
                    with llm_tabs[6]:  # Error Fix - Reasoning
                        if result.llm_trace.error_fix_reasoning_input:
                            st.write("**📥 INPUT PROMPT (Attempt 1 — Reasoning LLM):**")
                            st.text_area("Error Fix Reasoning Input", result.llm_trace.error_fix_reasoning_input, height=400, key="err_reason_in")
                            st.write("**📤 OUTPUT:**")
                            st.text_area("Error Fix Reasoning Output", result.llm_trace.error_fix_reasoning_output, height=200, key="err_reason_out")
                            if result.error_recovery_method == "reasoning_llm":
                                st.success("✅ Reasoning LLM resolved the error")
                            else:
                                st.warning("⚠️ This fix failed — escalated to Opus")
                        else:
                            st.info("No error recovery needed for this query")

                    with llm_tabs[7]:  # Error Fix - Opus
                        if result.llm_trace.error_fix_opus_input:
                            st.write("**📥 INPUT PROMPT (Attempt 2 — Opus):**")
                            st.text_area("Error Fix Opus Input", result.llm_trace.error_fix_opus_input, height=400, key="err_opus_in")
                            st.write("**📤 OUTPUT:**")
                            st.text_area("Error Fix Opus Output", result.llm_trace.error_fix_opus_output, height=200, key="err_opus_out")
                            if result.error_recovery_method == "opus":
                                st.success("✅ Opus fixed the error after Reasoning LLM could not")
                            else:
                                st.error("❌ Opus also could not fix the error")
                        else:
                            st.info("Opus error fix not needed for this query")                
                # ───────────────────────────────────────────────────────────
                # LOG QUERY - COMPREHENSIVE TRACKING
                # ───────────────────────────────────────────────────────────
                query_log_entry = {
                    # Basic Info
                    "Query Number": st.session_state.query_counter,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Question": question,
                    "Complexity": result.complexity,
                    
                    # Classifier (Llama)
                    "Classifier Input Tokens": result.tokens.classifier["input"],
                    "Classifier Output Tokens": result.tokens.classifier["output"],
                    
                    # Reasoning LLM (Sonnet)
                    "Reasoning Input Tokens": result.tokens.reasoning["input"],
                    "Reasoning Output Tokens": result.tokens.reasoning["output"],
                    "Reasoning Pass1 Input Tokens": result.tokens.reasoning_pass1["input"],
                    "Reasoning Pass1 Output Tokens": result.tokens.reasoning_pass1["output"],
                    "Reasoning Pass2 Input Tokens": result.tokens.reasoning_pass2["input"],
                    "Reasoning Pass2 Output Tokens": result.tokens.reasoning_pass2["output"],
                    "Opus Complex Input Tokens": result.tokens.opus_complex["input"],
                    "Opus Complex Output Tokens": result.tokens.opus_complex["output"],
                    
                    # SQL Coder (Llama/Groq)
                    "SQL Coder Input Tokens": result.tokens.sql_gen["input"],
                    "SQL Coder Output Tokens": result.tokens.sql_gen["output"],
                    
                    # Opus Reviewer
                    "Opus Input Tokens": result.tokens.opus["input"],
                    "Opus Output Tokens": result.tokens.opus["output"],
                    "Opus Verdict": result.final_verdict or "Not Used",
                    
                    # Refinement (if Opus found error)
                    "Refinement Input Tokens": result.tokens.refinement["input"],
                    "Refinement Output Tokens": result.tokens.refinement["output"],
                    "Error Fix Reasoning Input Tokens": result.tokens.error_fix_reasoning["input"],
                    "Error Fix Reasoning Output Tokens": result.tokens.error_fix_reasoning["output"],
                    "Error Fix Opus Input Tokens": result.tokens.error_fix_opus["input"],
                    "Error Fix Opus Output Tokens": result.tokens.error_fix_opus["output"],
                    "Chart Builder Input Tokens": result.tokens.chart["input"],
                    "Chart Builder Output Tokens": result.tokens.chart["output"],
                    
                    # Total Tokens
                    "Total Input Tokens": result.tokens.total()["input"],
                    "Total Output Tokens": result.tokens.total()["output"],
                    "Total Tokens": result.tokens.total_tokens(),
                    
                    # RAG Details
                    "Rule RAG Used": "Yes" if enable_rule_rag else "No",
                    "Opus Descriptions": "Yes" if enable_opus_descriptions else "No",
                    "Rules Retrieved": result.rules_retrieved,
                    
                    # Resolver Details
                    "Resolver Enabled": "Yes" if enable_resolver else "No",
                    "Entities Resolved": getattr(result, 'entities_resolved', 0),
                    "Resolver Time (ms)": getattr(result, 'resolver_time_ms', 0),
                    "Resolver Queries": result.resolver_result.queries_run if hasattr(result, 'resolver_result') and result.resolver_result else 0,
                    "All Entities Resolved": "Yes" if (hasattr(result, 'resolver_result') and result.resolver_result and result.resolver_result.all_resolved) else "N/A",
                    
                    # SQL & Results
                    "Generated SQL": result.sql,
                    "SQL Valid": "Yes" if result.validation_result and result.validation_result.get("is_valid", True) else "No",
                    "Auto-Fixed": "Yes" if result.sql_fixed else "No",
                    "Execution Success": "Yes" if result.success else "No",
                    "Rows Returned": len(result.results) if result.success and result.results is not None and hasattr(result.results, '__len__') else 0,
                    "Error": result.error or "",
                    
                    # Timing
                    "Total Time (ms)": result.total_time_ms,
                    "Flow Path": result.flow_path,
                    
                    # User Status (to be filled manually)
                    "Accuracy Status": ""  # User marks: Correct / Incorrect / Partial
                }
                st.session_state.query_log.append(query_log_entry)
                
                # Update global token tracking
                if result.tokens.reasoning["input"] > 0:
                    provider = config.reasoning_provider.split("_")[0]
                    if provider not in st.session_state.token_usage:
                        st.session_state.token_usage[provider] = {"input": 0, "output": 0}
                    st.session_state.token_usage[provider]["input"] += result.tokens.reasoning["input"]
                    st.session_state.token_usage[provider]["output"] += result.tokens.reasoning["output"]
                
                if result.tokens.sql_gen["input"] > 0:
                    provider = config.sql_provider.split("_")[0]
                    if provider not in st.session_state.token_usage:
                        st.session_state.token_usage[provider] = {"input": 0, "output": 0}
                    st.session_state.token_usage[provider]["input"] += result.tokens.sql_gen["input"]
                    st.session_state.token_usage[provider]["output"] += result.tokens.sql_gen["output"]
                
                if result.tokens.opus["input"] > 0:
                    if "opus" not in st.session_state.token_usage:
                        st.session_state.token_usage["opus"] = {"input": 0, "output": 0}
                    st.session_state.token_usage["opus"]["input"] += result.tokens.opus["input"]
                    st.session_state.token_usage["opus"]["output"] += result.tokens.opus["output"]

# ======================
# SIDEBAR: EXCEL EXPORT
# ======================

with st.sidebar:
    st.divider()
    st.header("📥 Export Query Log")
    
    if len(st.session_state.query_log) > 0:
        st.success(f"✅ {len(st.session_state.query_log)} queries logged")
        
        # Show preview
        with st.expander("Preview Log"):
            preview_df = pd.DataFrame(st.session_state.query_log)
            preview_cols = ["Query Number", "Question", "Complexity", "Total Tokens", "Opus Verdict", "Execution Success"]
            available_cols = [c for c in preview_cols if c in preview_df.columns]
            st.dataframe(preview_df[available_cols].head(5))
        
        # Export to Excel
        from io import BytesIO
        output = BytesIO()
        
        # Create DataFrame - use all available columns
        df_export = pd.DataFrame(st.session_state.query_log)
        
        # Define desired column order (only include columns that exist)
        desired_columns = [
            # Basic Info
            "Query Number",
            "Timestamp",
            "Question",
            "Complexity",
            
            # Classifier (Llama)
            "Classifier Input Tokens",
            "Classifier Output Tokens",
            
            # Reasoning LLM
            "Reasoning Input Tokens",
            "Reasoning Output Tokens",
            
            # SQL Coder
            "SQL Coder Input Tokens",
            "SQL Coder Output Tokens",
            
            # Opus Reviewer
            "Opus Input Tokens",
            "Opus Output Tokens",
            "Opus Verdict",
            
            # Refinement
            "Refinement Input Tokens",
            "Refinement Output Tokens",
            
            # Totals
            "Total Input Tokens",
            "Total Output Tokens",
            "Total Tokens",
            
            # RAG Details
            "Rule RAG Used",
            "Opus Descriptions",
            "Rules Retrieved",
            
            # Resolver Details
            "Resolver Enabled",
            "Entities Resolved",
            "Resolver Time (ms)",
            "Resolver Queries",
            "All Entities Resolved",
            
            # SQL & Results
            "Generated SQL",
            "SQL Valid",
            "Auto-Fixed",
            "Execution Success",
            "Rows Returned",
            "Error",
            
            # Timing
            "Total Time (ms)",
            "Flow Path",
            
            # User Status
            "Accuracy Status"
        ]
        
        # Only include columns that exist in the dataframe
        export_columns = [c for c in desired_columns if c in df_export.columns]
        df_export = df_export[export_columns]
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Query Log')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Query Log']
            for idx, col in enumerate(df_export.columns):
                max_length = max(
                    df_export[col].astype(str).map(len).max(),
                    len(col)
                )
                # Limit SQL column to reasonable width
                if col == "Generated SQL":
                    max_length = min(max_length, 80)
                if col == "Question":
                    max_length = min(max_length, 60)
                # Convert index to Excel column letter (handles columns beyond Z)
                col_letter = chr(65 + idx) if idx < 26 else chr(64 + idx // 26) + chr(65 + idx % 26)
                worksheet.column_dimensions[col_letter].width = min(max_length + 2, 80)
            
            # Add Summary Sheet
            summary_data = {
                "Metric": [
                    "Total Queries",
                    "Easy Queries",
                    "Medium Queries", 
                    "Hard Queries",
                    "",
                    "Total Classifier Tokens",
                    "Total Reasoning Tokens",
                    "Total SQL Coder Tokens",
                    "Total Opus Tokens",
                    "Total Refinement Tokens",
                    "Grand Total Tokens",
                    "",
                    "Queries with Rule RAG",
                    "Queries with Opus Descriptions",
                    "Queries with Opus Review",
                    "",
                    "Successful Executions",
                    "Failed Executions",
                    "Auto-Fixed SQLs"
                ],
                "Value": [
                    len(df_export),
                    len(df_export[df_export["Complexity"] == "easy"]) if "Complexity" in df_export.columns else 0,
                    len(df_export[df_export["Complexity"] == "medium"]) if "Complexity" in df_export.columns else 0,
                    len(df_export[df_export["Complexity"] == "hard"]) if "Complexity" in df_export.columns else 0,
                    "",
                    df_export["Classifier Input Tokens"].sum() + df_export["Classifier Output Tokens"].sum() if "Classifier Input Tokens" in df_export.columns else 0,
                    df_export["Reasoning Input Tokens"].sum() + df_export["Reasoning Output Tokens"].sum() if "Reasoning Input Tokens" in df_export.columns else 0,
                    df_export["SQL Coder Input Tokens"].sum() + df_export["SQL Coder Output Tokens"].sum() if "SQL Coder Input Tokens" in df_export.columns else 0,
                    df_export["Opus Input Tokens"].sum() + df_export["Opus Output Tokens"].sum() if "Opus Input Tokens" in df_export.columns else 0,
                    df_export["Refinement Input Tokens"].sum() + df_export["Refinement Output Tokens"].sum() if "Refinement Input Tokens" in df_export.columns else 0,
                    df_export["Total Tokens"].sum() if "Total Tokens" in df_export.columns else 0,
                    "",
                    len(df_export[df_export["Rule RAG Used"] == "Yes"]) if "Rule RAG Used" in df_export.columns else 0,
                    len(df_export[df_export["Opus Descriptions"] == "Yes"]) if "Opus Descriptions" in df_export.columns else 0,
                    len(df_export[df_export["Opus Verdict"] != "Not Used"]) if "Opus Verdict" in df_export.columns else 0,
                    "",
                    len(df_export[df_export["Execution Success"] == "Yes"]) if "Execution Success" in df_export.columns else 0,
                    len(df_export[df_export["Execution Success"] == "No"]) if "Execution Success" in df_export.columns else 0,
                    len(df_export[df_export["Auto-Fixed"] == "Yes"]) if "Auto-Fixed" in df_export.columns else 0
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, index=False, sheet_name='Summary')
            
            # Auto-adjust summary sheet
            summary_ws = writer.sheets['Summary']
            summary_ws.column_dimensions['A'].width = 30
            summary_ws.column_dimensions['B'].width = 15
        
        excel_data = output.getvalue()
        
        st.download_button(
            label="📥 Download Excel",
            data=excel_data,
            file_name=f"query_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
        
        st.info("💡 Fill **Accuracy Status** column manually: Correct / Incorrect / Partial")
        
        # Clear log button
        if st.button("🗑️ Clear Log"):
            st.session_state.query_log = []
            st.session_state.query_counter = 0
            st.success("Log cleared!")
            st.rerun()
    else:
        st.info("No queries logged yet. Generate some queries in Tab 3!")

# ======================
# FOOTER
# ======================
st.divider()
st.caption("💡 Built with Streamlit • Powered by RAG • Multi-LLM Testing Platform")
