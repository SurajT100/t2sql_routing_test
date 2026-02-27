"""
TAB 1 INTEGRATION - Schema Profiling
=====================================
Add this code to Tab 1 in app_v2_dual_connection.py

STEP 1: Add import at top of file:
    from schema_profiler import (
        check_schema_columns_table,
        profile_selected_tables,
        get_profiled_columns,
        get_profile_stats,
        update_column_enrichment
    )

STEP 2: Find this line in Tab 1 (around line 400):
    st.divider()
    
    # And the closing of tab1

STEP 3: Add the code below BEFORE the tab1 closes
"""

# =============================================================================
# CODE TO ADD IN TAB 1 (after table selection, before tab1 ends)
# =============================================================================

TAB1_SCHEMA_PROFILING_CODE = '''
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
            st.warning(f"⚠️ Schema RAG not configured: {setup_message}")
            with st.expander("📋 Setup Instructions", expanded=True):
                st.markdown("""
**To enable Schema RAG:**

1. Open your **Supabase SQL Editor**
2. Copy and run the contents of `schema_rag_setup.sql`
3. Refresh this page

This creates the `schema_columns` table for storing column metadata and embeddings.
                """)
                
                # Show the SQL
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
            # Schema RAG is set up - show profiling UI
            
            # Get current stats
            stats = get_profile_stats(VECTOR_ENGINE, selected_objects)
            
            # Stats display
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            col_stat1.metric("📊 Profiled", stats["profiled"])
            col_stat2.metric("🔒 With PII", stats["with_pii"])
            col_stat3.metric("✅ Enriched", stats["user_enriched"])
            col_stat4.metric("⚡ Auto Only", stats["auto_enriched"])
            
            # Profile button
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
            
            # Handle profiling
            if profile_clicked:
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_tables = len(selected_objects)
                    tables_done = [0]  # Use list to allow modification in closure
                    
                    def update_progress(table, column, status):
                        # Clean table name for display
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
                
                # Show results
                if result["errors"]:
                    st.warning(f"⚠️ Completed with {len(result['errors'])} errors")
                    with st.expander("View Errors"):
                        for err in result["errors"]:
                            st.error(err)
                else:
                    st.success(f"✅ Profiled {result['total_columns']} columns across {result['total_tables']} tables")
                
                if result["columns_with_pii"] > 0:
                    st.warning(f"🔒 Detected PII in {result['columns_with_pii']} columns - samples are automatically masked")
                
                # Show detailed results
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
                
                st.rerun()
            
            # ═══════════════════════════════════════════════════════════════
            # COLUMN ENRICHMENT UI
            # ═══════════════════════════════════════════════════════════════
            if stats["profiled"] > 0:
                with st.expander("📝 Enrich Column Descriptions", expanded=False):
                    st.caption("Add friendly names and descriptions to improve RAG accuracy")
                    
                    # Get profiled columns
                    columns = get_profiled_columns(VECTOR_ENGINE, selected_objects)
                    
                    # Filters
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
                    
                    # Apply filters
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
                    
                    # Display columns for enrichment
                    for i, col in enumerate(filtered_columns[:20]):  # Limit to 20 for performance
                        status_icon = "✅" if col["status"] == "user" else "⚡"
                        pii_icon = "🔒" if col["has_pii"] else ""
                        
                        with st.expander(
                            f"{status_icon} {pii_icon} **{col['column']}** ({col['table']}) - {col['auto_expanded'] or col['type']}",
                            expanded=False
                        ):
                            # Column info
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
                            
                            # Enrichment form
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
'''

# =============================================================================
# PRINT INSTRUCTIONS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TAB 1 SCHEMA PROFILING INTEGRATION")
    print("=" * 70)
    print("""
INSTRUCTIONS:

1. Add import at TOP of app_v2_dual_connection.py:

   from schema_profiler import (
       check_schema_columns_table,
       profile_selected_tables,
       get_profiled_columns,
       get_profile_stats,
       update_column_enrichment
   )

2. Find Tab 1 section (around line 355-401)

3. Add the code AFTER the schema preview expander closes
   (after line ~400, before tab1 ends)

4. The code is in the TAB1_SCHEMA_PROFILING_CODE variable above

5. Make sure schema_profiler.py is in the same directory

6. Run schema_rag_setup.sql in Supabase first!
""")
    print("=" * 70)
    print("\nCode to add:")
    print("-" * 70)
    print(TAB1_SCHEMA_PROFILING_CODE)
