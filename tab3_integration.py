"""
TAB 3 REPLACEMENT - Optimized Query Processing
==============================================
This file contains the updated Tab 3 code to replace in app_v2_dual_connection.py

Integration steps:
1. Add imports at top of app_v2_dual_connection.py
2. Replace the `with tab3:` section with this code
3. Run and test!

New Features:
- Query classification (Easy/Medium/Hard)
- Schema RAG (relevant columns only)
- Compressed rules format
- SQL pre-validation
- Token tracking by stage
"""

# =============================================================================
# IMPORTS TO ADD AT TOP OF app_v2_dual_connection.py
# =============================================================================
"""
# Add these imports near the top of your file:

from flow_router import process_query, FlowConfig, QueryResult, create_default_config
from query_classifier import classify_query, get_flow_config
from sql_validator import validate_sql
from prompt_optimizer import compress_rules_for_llm
"""

# =============================================================================
# TAB 3 REPLACEMENT CODE
# Copy everything below and replace your existing `with tab3:` section
# =============================================================================

TAB3_CODE = '''
with tab3:
    st.header("🔍 Test Queries")
    
    if "engine" not in st.session_state:
        st.warning("👈 Connect to database first")
    elif "selected_objects" not in st.session_state or not st.session_state.selected_objects:
        st.warning("👈 Select tables/views first")
    else:
        # ═══════════════════════════════════════════════════════════════════
        # CONFIGURATION PANEL
        # ═══════════════════════════════════════════════════════════════════
        with st.expander("⚙️ Optimization Settings", expanded=False):
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                st.write("**Query Classification**")
                enable_classification = st.checkbox(
                    "Enable Llama Classification", 
                    value=True,
                    help="Classify queries as Easy/Medium/Hard to optimize token usage"
                )
                if enable_classification:
                    classification_provider = st.selectbox(
                        "Classification Model",
                        ["groq", "claude_haiku"],
                        index=0,
                        help="Fast model for classification"
                    )
                else:
                    classification_provider = "groq"
            
            with col_opt2:
                st.write("**RAG Settings**")
                enable_rule_rag = st.checkbox("Enable Rule RAG", value=True)
                enable_schema_rag = st.checkbox(
                    "Enable Schema RAG", 
                    value=False,
                    help="Retrieve only relevant columns (requires schema_columns table)"
                )
                compress_rules = st.checkbox(
                    "Compress Rules", 
                    value=True,
                    help="Use compact JSON format for rules (saves ~70% tokens)"
                )
            
            with col_opt3:
                st.write("**Validation & Review**")
                enable_sql_validation = st.checkbox(
                    "Pre-validate SQL", 
                    value=True,
                    help="Check SQL syntax before execution (free)"
                )
                auto_fix_sql = st.checkbox(
                    "Auto-fix common issues", 
                    value=True,
                    help="Fix NULL comparisons, typos, etc."
                )
                
                opus_mode = st.radio(
                    "Opus Review",
                    ["Disabled", "Auto (Hard only)", "Always"],
                    index=1,
                    horizontal=True
                )
                
                # Map to config values
                opus_config = {
                    "Disabled": False,
                    "Auto (Hard only)": "auto",
                    "Always": True
                }[opus_mode]
        
        # ═══════════════════════════════════════════════════════════════════
        # QUERY INPUT
        # ═══════════════════════════════════════════════════════════════════
        st.divider()
        
        question = st.text_input(
            "Your Question",
            placeholder="e.g., Show total sales by region excluding rebates",
            key="user_question_v2"
        )
        
        # Generate button
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            generate_clicked = st.button("🚀 Generate SQL", type="primary", use_container_width=True)
        
        with col_info:
            if enable_classification:
                st.caption("💡 Query will be classified → optimized flow selected → SQL generated")
            else:
                st.caption("💡 Using standard Medium flow for all queries")
        
        # ═══════════════════════════════════════════════════════════════════
        # MAIN PROCESSING
        # ═══════════════════════════════════════════════════════════════════
        if generate_clicked:
            if not question:
                st.error("❌ Please enter a question")
            else:
                st.session_state.query_counter += 1
                
                # Get dialect from session state or default
                dialect = st.session_state.get('detected_dialect', 'postgresql')
                
                # Create configuration
                config = FlowConfig(
                    enable_classification=enable_classification,
                    classification_provider=classification_provider,
                    enable_rule_rag=enable_rule_rag,
                    enable_schema_rag=enable_schema_rag,
                    compress_rules=compress_rules,
                    validate_sql=enable_sql_validation,
                    auto_fix_sql=auto_fix_sql,
                    enable_opus=opus_config,
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
                    with st.spinner("🏷️ Classifying query complexity..."):
                        classification = classify_query(
                            question,
                            use_llm=True,
                            llm_provider=classification_provider
                        )
                    
                    complexity = classification["complexity"]
                    complexity_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}[complexity]
                    
                    col_class1, col_class2, col_class3 = st.columns([1, 2, 1])
                    with col_class1:
                        st.metric("Complexity", f"{complexity_emoji} {complexity.upper()}")
                    with col_class2:
                        st.caption(f"**Reason:** {classification['reason']}")
                    with col_class3:
                        flow_cfg = get_flow_config(complexity)
                        st.caption(f"**Est. tokens:** ~{flow_cfg['expected_tokens']}")
                else:
                    complexity = "medium"
                    classification = {"complexity": "medium", "reason": "Classification disabled", "tokens": {"input": 0, "output": 0}}
                
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
                
                # Flow Summary
                st.divider()
                st.subheader("📊 Query Processing Summary")
                
                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                
                with col_sum1:
                    st.metric("Complexity", result.complexity.upper())
                
                with col_sum2:
                    total_tokens = result.tokens.total_tokens()
                    st.metric("Total Tokens", f"{total_tokens:,}")
                
                with col_sum3:
                    st.metric("Time", f"{result.total_time_ms}ms")
                
                with col_sum4:
                    if result.success:
                        st.metric("Status", "✅ Success")
                    else:
                        st.metric("Status", "❌ Failed")
                
                # Flow path
                st.caption(f"**Flow:** {result.flow_path}")
                
                # ───────────────────────────────────────────────────────────
                # TOKEN BREAKDOWN
                # ───────────────────────────────────────────────────────────
                with st.expander("💰 Token Usage Breakdown", expanded=True):
                    tokens = result.tokens
                    
                    # Create DataFrame for token breakdown
                    token_data = []
                    
                    if tokens.classifier["input"] > 0 or tokens.classifier["output"] > 0:
                        token_data.append({
                            "Stage": "🏷️ Classifier",
                            "Input": tokens.classifier["input"],
                            "Output": tokens.classifier["output"],
                            "Total": tokens.classifier["input"] + tokens.classifier["output"]
                        })
                    
                    if tokens.reasoning["input"] > 0 or tokens.reasoning["output"] > 0:
                        token_data.append({
                            "Stage": "🧠 Reasoning",
                            "Input": tokens.reasoning["input"],
                            "Output": tokens.reasoning["output"],
                            "Total": tokens.reasoning["input"] + tokens.reasoning["output"]
                        })
                    
                    if tokens.sql_gen["input"] > 0 or tokens.sql_gen["output"] > 0:
                        token_data.append({
                            "Stage": "⚙️ SQL Gen",
                            "Input": tokens.sql_gen["input"],
                            "Output": tokens.sql_gen["output"],
                            "Total": tokens.sql_gen["input"] + tokens.sql_gen["output"]
                        })
                    
                    if tokens.opus["input"] > 0 or tokens.opus["output"] > 0:
                        token_data.append({
                            "Stage": "🎯 Opus Review",
                            "Input": tokens.opus["input"],
                            "Output": tokens.opus["output"],
                            "Total": tokens.opus["input"] + tokens.opus["output"]
                        })
                    
                    if tokens.refinement["input"] > 0 or tokens.refinement["output"] > 0:
                        token_data.append({
                            "Stage": "🔄 Refinement",
                            "Input": tokens.refinement["input"],
                            "Output": tokens.refinement["output"],
                            "Total": tokens.refinement["input"] + tokens.refinement["output"]
                        })
                    
                    if token_data:
                        df_tokens = pd.DataFrame(token_data)
                        
                        # Add totals row
                        totals = {
                            "Stage": "**TOTAL**",
                            "Input": sum(t["Input"] for t in token_data),
                            "Output": sum(t["Output"] for t in token_data),
                            "Total": sum(t["Total"] for t in token_data)
                        }
                        df_tokens = pd.concat([df_tokens, pd.DataFrame([totals])], ignore_index=True)
                        
                        st.dataframe(df_tokens, use_container_width=True, hide_index=True)
                        
                        # Comparison with baseline
                        baseline_tokens = 15000  # Typical without optimization
                        savings = baseline_tokens - totals["Total"]
                        savings_pct = (savings / baseline_tokens) * 100
                        
                        if savings > 0:
                            st.success(f"💰 **Saved ~{savings:,} tokens ({savings_pct:.0f}%)** compared to baseline")
                    else:
                        st.info("No token usage recorded")
                
                # ───────────────────────────────────────────────────────────
                # CONTEXT RETRIEVED
                # ───────────────────────────────────────────────────────────
                with st.expander("📚 Context Retrieved", expanded=False):
                    col_ctx1, col_ctx2 = st.columns(2)
                    
                    with col_ctx1:
                        st.metric("Rules Retrieved", result.rules_retrieved)
                        if result.rules_compressed and result.rules_compressed != "[]":
                            st.write("**Compressed Rules:**")
                            st.code(result.rules_compressed[:500] + "..." if len(result.rules_compressed) > 500 else result.rules_compressed, language="json")
                    
                    with col_ctx2:
                        if result.columns_retrieved == -1:
                            st.metric("Schema Mode", "Full Schema")
                        else:
                            st.metric("Columns Retrieved", result.columns_retrieved)
                        
                        if result.schema_text:
                            st.write("**Schema Context:**")
                            st.code(result.schema_text[:1000] + "..." if len(result.schema_text) > 1000 else result.schema_text)
                
                # ───────────────────────────────────────────────────────────
                # VALIDATION RESULTS
                # ───────────────────────────────────────────────────────────
                if result.validation_result:
                    with st.expander("✅ SQL Validation", expanded=not result.validation_result["is_valid"]):
                        if result.validation_result["is_valid"]:
                            st.success("SQL passed validation")
                        else:
                            st.error("SQL validation issues found")
                            
                            if result.validation_result["issues"]:
                                st.write("**Issues:**")
                                for issue in result.validation_result["issues"]:
                                    st.write(f"- ❌ {issue}")
                            
                            if result.validation_result["warnings"]:
                                st.write("**Warnings:**")
                                for warning in result.validation_result["warnings"]:
                                    st.write(f"- ⚠️ {warning}")
                        
                        if result.sql_fixed:
                            st.info("**Auto-fixes applied:**")
                            for fix in result.fixes_applied:
                                st.write(f"- 🔧 {fix}")
                
                # ───────────────────────────────────────────────────────────
                # GENERATED SQL
                # ───────────────────────────────────────────────────────────
                st.divider()
                st.subheader("💻 Generated SQL")
                st.code(result.sql, language="sql")
                
                # Copy button
                st.button("📋 Copy SQL", on_click=lambda: st.write("Copied!"))
                
                # ───────────────────────────────────────────────────────────
                # OPUS REVIEW (if ran)
                # ───────────────────────────────────────────────────────────
                if result.opus_review:
                    st.divider()
                    st.subheader("🎯 Opus Review")
                    
                    verdict = result.final_verdict
                    
                    if verdict == "CORRECT":
                        st.success(f"✅ **Verdict: CORRECT** (Confidence: {result.opus_review.get('confidence', 0):.0%})")
                    elif verdict == "UNCERTAIN":
                        st.warning(f"⚠️ **Verdict: UNCERTAIN** (Confidence: {result.opus_review.get('confidence', 0):.0%})")
                    else:
                        st.error(f"❌ **Verdict: {verdict}**")
                    
                    if result.opus_review.get("reasoning"):
                        st.write(f"**Reasoning:** {result.opus_review['reasoning']}")
                    
                    if result.opus_review.get("issues"):
                        st.write("**Issues:**")
                        for issue in result.opus_review["issues"]:
                            st.write(f"- {issue}")
                    
                    st.caption(f"Opus attempts: {result.opus_attempts}")
                
                # ───────────────────────────────────────────────────────────
                # QUERY RESULTS
                # ───────────────────────────────────────────────────────────
                st.divider()
                st.subheader("📈 Query Results")
                
                if result.success:
                    if result.results is not None:
                        if hasattr(result.results, 'shape'):
                            st.success(f"✅ Returned {len(result.results)} rows, {len(result.results.columns)} columns")
                            st.dataframe(result.results, use_container_width=True)
                        else:
                            st.write(result.results)
                    else:
                        st.info("Query executed but returned no results")
                else:
                    st.error(f"❌ Query failed: {result.error}")
                    
                    # Show error analysis
                    if "syntax" in result.error.lower():
                        st.info("💡 **Tip:** This looks like a syntax error. Check the SQL validation section above.")
                    elif "column" in result.error.lower() or "does not exist" in result.error.lower():
                        st.info("💡 **Tip:** A column or table might not exist. Verify the schema.")
                
                # ───────────────────────────────────────────────────────────
                # TIMING BREAKDOWN
                # ───────────────────────────────────────────────────────────
                with st.expander("⏱️ Timing Breakdown", expanded=False):
                    timing_data = [
                        {"Stage": stage.replace("_", " ").title(), "Time (ms)": time}
                        for stage, time in result.stage_times.items()
                    ]
                    if timing_data:
                        df_timing = pd.DataFrame(timing_data)
                        st.dataframe(df_timing, use_container_width=True, hide_index=True)
                        
                        # Bar chart
                        st.bar_chart(df_timing.set_index("Stage")["Time (ms)"])
                
                # ───────────────────────────────────────────────────────────
                # DEBUG INFO
                # ───────────────────────────────────────────────────────────
                with st.expander("🐛 Debug Info", expanded=False):
                    st.write("**Stages Completed:**", result.stages_completed)
                    st.write("**Configuration:**")
                    st.json({
                        "enable_classification": config.enable_classification,
                        "enable_rule_rag": config.enable_rule_rag,
                        "enable_schema_rag": config.enable_schema_rag,
                        "compress_rules": config.compress_rules,
                        "validate_sql": config.validate_sql,
                        "enable_opus": str(config.enable_opus),
                        "dialect": config.dialect
                    })
                
                # ───────────────────────────────────────────────────────────
                # UPDATE SESSION STATE TOKEN TRACKING
                # ───────────────────────────────────────────────────────────
                # Map tokens to existing tracking structure
                if result.tokens.reasoning["input"] > 0:
                    provider_key = config.reasoning_provider.split("_")[0]
                    if provider_key not in st.session_state.token_usage:
                        st.session_state.token_usage[provider_key] = {"input": 0, "output": 0}
                    st.session_state.token_usage[provider_key]["input"] += result.tokens.reasoning["input"]
                    st.session_state.token_usage[provider_key]["output"] += result.tokens.reasoning["output"]
                
                if result.tokens.sql_gen["input"] > 0:
                    provider_key = config.sql_provider.split("_")[0]
                    if provider_key not in st.session_state.token_usage:
                        st.session_state.token_usage[provider_key] = {"input": 0, "output": 0}
                    st.session_state.token_usage[provider_key]["input"] += result.tokens.sql_gen["input"]
                    st.session_state.token_usage[provider_key]["output"] += result.tokens.sql_gen["output"]
                
                if result.tokens.opus["input"] > 0:
                    if "opus" not in st.session_state.token_usage:
                        st.session_state.token_usage["opus"] = {"input": 0, "output": 0}
                    st.session_state.token_usage["opus"]["input"] += result.tokens.opus["input"]
                    st.session_state.token_usage["opus"]["output"] += result.tokens.opus["output"]
'''

# =============================================================================
# SAVE THE TAB3 CODE FOR REFERENCE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TAB 3 REPLACEMENT CODE")
    print("=" * 70)
    print("\nInstructions:")
    print("1. Add imports at top of app_v2_dual_connection.py:")
    print("   from flow_router import process_query, FlowConfig, QueryResult")
    print("   from query_classifier import classify_query, get_flow_config")
    print("")
    print("2. Find the line: `with tab3:`")
    print("3. Replace everything in that block with the TAB3_CODE above")
    print("")
    print("4. Test with a simple query!")
    print("=" * 70)
