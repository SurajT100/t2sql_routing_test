"""
APP.PY PATCH — SQL Validation Display Section
=============================================
Replace the existing SQL Validation expander block in Tab 3 (Test Queries)
with this updated version.

FIND this block in app.py (around the VALIDATION comment):
─────────────────────────────────────────────────────────
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

REPLACE WITH the block below:
─────────────────────────────────────────────────────────
"""

# ───────────────────────────────────────────────────────────
# VALIDATION  (UPDATED — shows original vs fixed SQL diff)
# ───────────────────────────────────────────────────────────

# Determine expander title and whether to auto-open it
_validation_has_issues = (
    result.validation_result and not result.validation_result.get("is_valid", True)
)
_recovery_happened = result.error_recovery_attempted and result.error_recovery_method != "none"
_expander_open = _validation_has_issues or _recovery_happened

_expander_title = "✅ SQL Validation"
if _recovery_happened:
    method_label = {
        "reasoning_llm": "⚡ Auto-fixed by Reasoning LLM",
        "opus":          "🔧 Auto-fixed by Opus",
    }.get(result.error_recovery_method, "🔧 Auto-fixed")
    _expander_title = f"{method_label}"

if result.validation_result:
    with st.expander(_expander_title, expanded=_expander_open):

        # ── Static validation result ──────────────────────────────
        if result.validation_result.get("is_valid"):
            st.success("SQL passed pre-execution validation")
        else:
            st.error("Pre-execution validation issues found")
            for issue in result.validation_result.get("issues", []):
                st.write(f"- ❌ {issue}")

        # ── Error recovery section ────────────────────────────────
        if result.error_recovery_attempted:
            st.divider()

            if result.error_recovery_method == "none":
                # Both attempts failed — show friendly error
                st.error("⚠️ Auto-repair failed after two attempts")
                if result.original_error:
                    with st.expander("🔍 Technical error detail"):
                        st.code(result.original_error, language="text")

            else:
                # Recovery succeeded — show what happened
                method_display = {
                    "reasoning_llm": ("⚡ Reasoning LLM", "blue"),
                    "opus":          ("🔧 Opus",          "green"),
                }.get(result.error_recovery_method, ("🔧 Auto-fix", "blue"))

                st.info(
                    f"**{method_display[0]} automatically repaired the SQL** "
                    f"after the first execution attempt failed."
                )

                # Show the original error that triggered recovery
                if result.original_error:
                    with st.expander("❌ Original error that was fixed"):
                        st.code(result.original_error, language="text")

                # Show before / after SQL diff
                if result.original_sql and result.original_sql != result.sql:
                    col_before, col_after = st.columns(2)

                    with col_before:
                        st.write("**🔴 Original SQL (failed)**")
                        st.code(result.original_sql, language="sql")

                    with col_after:
                        st.write("**🟢 Fixed SQL (executed)**")
                        st.code(result.sql, language="sql")

                    # Highlight what changed at a line level
                    original_lines = set(result.original_sql.strip().splitlines())
                    fixed_lines    = set(result.sql.strip().splitlines())
                    added_lines    = fixed_lines - original_lines
                    removed_lines  = original_lines - fixed_lines

                    if added_lines or removed_lines:
                        with st.expander("🔍 What changed"):
                            if removed_lines:
                                st.write("**Removed:**")
                                for line in removed_lines:
                                    if line.strip():
                                        st.code(f"- {line}", language="sql")
                            if added_lines:
                                st.write("**Added:**")
                                for line in added_lines:
                                    if line.strip():
                                        st.code(f"+ {line}", language="sql")

        # ── Legacy auto-fix display (pre-execution fixes by sql_validator) ──
        elif result.sql_fixed and result.fixes_applied:
            non_recovery_fixes = [
                f for f in result.fixes_applied
                if f not in ("reasoning_llm_error_fix", "opus_error_fix")
            ]
            if non_recovery_fixes:
                st.info(f"**Pre-execution auto-fixes:** {', '.join(non_recovery_fixes)}")

# ─────────────────────────────────────────────────────────
# Also update the LLM Input/Output viewer tabs to include
# the two new error recovery trace tabs.
#
# FIND this in the LLM I/O expander section:
#     llm_tabs = st.tabs(["📊 Classifier", "🧠 Reasoning", "⚙️ SQL Coder", "🎯 Opus", "🔄 Refinement"])
#
# REPLACE WITH:
# ─────────────────────────────────────────────────────────

llm_tabs = st.tabs([
    "📊 Classifier",
    "🧠 Reasoning",
    "⚙️ SQL Coder",
    "🎯 Opus Review",
    "🔄 Refinement",
    "⚡ Error Fix (Reasoning)",
    "🔧 Error Fix (Opus)"
])

with llm_tabs[0]:  # Classifier
    if result.llm_trace.classifier_input:
        st.write("**📥 INPUT PROMPT:**")
        st.text_area("Classifier Input", result.llm_trace.classifier_input, height=300, key="clf_in")
        st.write("**📤 OUTPUT:**")
        st.text_area("Classifier Output", result.llm_trace.classifier_output, height=150, key="clf_out")
    else:
        st.info("Classifier used keyword-based classification (no LLM call)")

with llm_tabs[1]:  # Reasoning
    if result.llm_trace.reasoning_input:
        st.write("**📥 INPUT PROMPT:**")
        st.text_area("Reasoning Input", result.llm_trace.reasoning_input, height=400, key="reason_in")
        st.write("**📤 OUTPUT:**")
        st.text_area("Reasoning Output", result.llm_trace.reasoning_output, height=300, key="reason_out")
    else:
        st.info("Reasoning LLM not used for this query")

with llm_tabs[2]:  # SQL Coder
    if result.llm_trace.sql_gen_input:
        st.write("**📥 INPUT PROMPT:**")
        st.text_area("SQL Coder Input", result.llm_trace.sql_gen_input, height=400, key="sql_in")
        st.write("**📤 OUTPUT:**")
        st.text_area("SQL Coder Output", result.llm_trace.sql_gen_output, height=200, key="sql_out")
    else:
        st.info("SQL Coder not used for this query (EASY flow uses Reasoning only)")

with llm_tabs[3]:  # Opus Review
    if result.llm_trace.opus_input:
        st.write("**📥 INPUT PROMPT:**")
        st.text_area("Opus Input", result.llm_trace.opus_input, height=400, key="opus_in")
        st.write("**📤 OUTPUT:**")
        st.text_area("Opus Output", result.llm_trace.opus_output, height=200, key="opus_out")
    else:
        st.info("Opus review not used for this query")

with llm_tabs[4]:  # Refinement
    if result.llm_trace.refinement_input:
        st.write("**📥 INPUT PROMPT:**")
        st.text_area("Refinement Input", result.llm_trace.refinement_input, height=400, key="refine_in")
        st.write("**📤 OUTPUT:**")
        st.text_area("Refinement Output", result.llm_trace.refinement_output, height=200, key="refine_out")
    else:
        st.info("Refinement not triggered (query was correct or Opus review not enabled)")

with llm_tabs[5]:  # Error Fix - Reasoning LLM
    if result.llm_trace.error_fix_reasoning_input:
        st.write("**📥 INPUT PROMPT (Attempt 1 — Reasoning LLM):**")
        st.text_area("Error Fix Reasoning Input", result.llm_trace.error_fix_reasoning_input, height=400, key="err_reason_in")
        st.write("**📤 OUTPUT:**")
        st.text_area("Error Fix Reasoning Output", result.llm_trace.error_fix_reasoning_output, height=200, key="err_reason_out")

        if result.error_recovery_method == "reasoning_llm":
            st.success("✅ This fix succeeded — Reasoning LLM resolved the error")
        else:
            st.warning("⚠️ This fix failed — escalated to Opus (see next tab)")
    else:
        st.info("No error recovery needed for this query")

with llm_tabs[6]:  # Error Fix - Opus
    if result.llm_trace.error_fix_opus_input:
        st.write("**📥 INPUT PROMPT (Attempt 2 — Opus):**")
        st.text_area("Error Fix Opus Input", result.llm_trace.error_fix_opus_input, height=400, key="err_opus_in")
        st.write("**📤 OUTPUT:**")
        st.text_area("Error Fix Opus Output", result.llm_trace.error_fix_opus_output, height=200, key="err_opus_out")

        if result.error_recovery_method == "opus":
            st.success("✅ Opus fixed the error after Reasoning LLM could not")
        else:
            st.error("❌ Opus also failed to fix the error")
    else:
        st.info("Opus error fix not needed (either no error, or Reasoning LLM fixed it)")

# ─────────────────────────────────────────────────────────
# Also update the token breakdown table in app.py to show
# the two new error recovery token fields.
#
# FIND this section in the Token Usage Breakdown expander:
#     if tokens.opus["input"] + tokens.opus["output"] > 0:
#         token_data.append({"Stage": "🎯 Opus", ...})
#     if tokens.refinement["input"] + tokens.refinement["output"] > 0:
#         token_data.append({"Stage": "🔄 Refinement", ...})
#
# ADD these two lines after the refinement block:
# ─────────────────────────────────────────────────────────

if tokens.error_fix_reasoning["input"] + tokens.error_fix_reasoning["output"] > 0:
    token_data.append({
        "Stage": "⚡ Error Fix (Reasoning)",
        "Input": tokens.error_fix_reasoning["input"],
        "Output": tokens.error_fix_reasoning["output"]
    })

if tokens.error_fix_opus["input"] + tokens.error_fix_opus["output"] > 0:
    token_data.append({
        "Stage": "🔧 Error Fix (Opus)",
        "Input": tokens.error_fix_opus["input"],
        "Output": tokens.error_fix_opus["output"]
    })
