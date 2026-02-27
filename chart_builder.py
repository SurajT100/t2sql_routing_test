"""
Chart Builder
=============
Uses Claude Sonnet to generate Plotly chart code on the fly.
LLM sees actual data, question, and style guidelines — handles
sorting, formatting, edge cases automatically.

Usage:
    from chart_builder import build_and_render_chart
    build_and_render_chart(df, question, llm_provider="claude_sonnet")
"""

import pandas as pd
import streamlit as st
import traceback
import json
from typing import Optional


# =============================================================================
# STYLE GUIDELINES — injected into every Chart Builder prompt
# =============================================================================

CHART_STYLE_GUIDELINES = """
PLOTLY STYLE REQUIREMENTS (follow exactly):

TEMPLATE & FONTS:
- template="plotly_white"
- font=dict(family="DM Sans, Segoe UI, sans-serif", size=13, color="#1e293b")
- title_font=dict(size=16, family="DM Sans, Segoe UI, sans-serif", color="#0f172a", bold=True)

COLORS (use this palette in order):
COLORS = ["#4F86C6", "#F4845F", "#57B894", "#A78BFA", "#F59E0B",
          "#EC4899", "#06B6D4", "#84CC16"]

LAYOUT:
- plot_bgcolor="white"
- paper_bgcolor="white"  
- margin=dict(l=50, r=30, t=70, b=60)
- legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
- showlegend=True (only if multiple series)

AXES:
- showgrid=True on y-axis only, gridcolor="#f1f5f9", gridwidth=1
- showgrid=False on x-axis
- linecolor="#e2e8f0", linewidth=1
- Remove top and right axis borders: mirror=False
- Rotate x-axis labels if more than 6 categories: tickangle=-35

NUMBERS & FORMATTING:
- Format large numbers in axis ticks using tickformat
- If values > 10,000,000 → use ticksuffix=" Cr", divide by 10000000
- If values > 100,000 → use ticksuffix=" L", divide by 100000
- Add comma separators for all numbers
- Show data labels on bars only if <= 12 bars

SORTING RULES (CRITICAL):
- Time series data: ALWAYS sort chronologically by date/period column
- Month names: sort Jan→Dec not alphabetically
- Quarter names: sort Q1→Q4
- Bar charts: sort by value descending UNLESS it's a ranking/ordered question
- Never plot in random DataFrame order

DATE/TIMESTAMP FORMATTING:
- Convert full timestamps like "2025-06-01 00:00:00+00:00" to "Jun 2025"
- Use pd.to_datetime() then .dt.strftime("%b %Y") for month columns
- Use .dt.strftime("%b '%y") for compact month labels

BARS:
- marker_line_width=0 (no bar borders)
- opacity=0.9
- Bar gap: bargap=0.3

LINES:
- line_width=2.5
- markers=True, marker_size=7, marker_line_width=0

PIE/DONUT:
- hole=0.38 (donut style)
- Cap at 8 slices, group rest as "Other"
- textinfo="percent+label", textposition="inside"
"""


# =============================================================================
# CHART BUILDER — LLM generates Plotly code
# =============================================================================

def generate_chart_code(
    df: pd.DataFrame,
    question: str,
    llm_provider: str = "claude_sonnet"
) -> tuple[Optional[str], dict]:
    """
    Ask Claude Sonnet to write Plotly code for this specific DataFrame.

    Args:
        df: Query result DataFrame
        question: Original user question
        llm_provider: LLM provider key

    Returns:
        Tuple of (code string or None, token usage dict)
    """
    try:
        from llm_v2 import call_llm

        # Send first 20 rows so LLM sees real values
        sample_rows = min(20, len(df))
        data_sample = df.head(sample_rows).to_string(index=False)
        dtypes_info = df.dtypes.to_string()

        prompt = f"""You are an expert data visualization engineer.
Write Python code using Plotly to create the best chart for this data.

USER QUESTION: {question}

DATAFRAME INFO:
Columns and dtypes:
{dtypes_info}

Total rows: {len(df)}

Sample data (first {sample_rows} rows):
{data_sample}

{CHART_STYLE_GUIDELINES}

TASK:
1. Analyze the question and data to choose the most appropriate chart type
2. Sort the data correctly (time series chronologically, bars by value etc.)
3. Format dates/timestamps to readable labels
4. Apply all style guidelines above
5. Write complete, runnable Python code

AVAILABLE VARIABLES (already in scope, do NOT redefine):
- df: the pandas DataFrame with query results
- COLORS: list of hex color strings (from style guidelines)

IMPORTS AVAILABLE:
- import plotly.express as px
- import plotly.graph_objects as go
- import pandas as pd

REQUIRED OUTPUT FORMAT:
- Variable named `fig` must be the final Plotly figure
- No st.plotly_chart() call — just create fig
- No import statements needed
- No print statements
- Handle edge cases (empty data, single row, etc.)

EXAMPLE STRUCTURE:
# Sort data
df_chart = df.copy()
df_chart['Month'] = pd.to_datetime(df_chart['Month']).dt.strftime('%b %Y')
df_chart = df_chart.sort_values(...)

# Create figure
fig = go.Figure()
fig.add_trace(...)
fig.update_layout(
    title="...",
    template="plotly_white",
    ...
)

Return ONLY the Python code. No explanation, no markdown fences."""

        response, tokens = call_llm(prompt, llm_provider)
        print(f"[CHART BUILDER] Generated chart code ({tokens.get('output', 0)} output tokens)")

        # Strip markdown fences if LLM added them
        code = response.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            lines = lines[:-1] if lines[-1].strip() == "```" else lines
            code = "\n".join(lines)

        return code.strip(), tokens

    except Exception as e:
        print(f"[CHART BUILDER] Code generation error: {e}")
        return None, {"input": 0, "output": 0}


# =============================================================================
# SAFE EXECUTOR — runs LLM-generated code in a sandboxed namespace
# =============================================================================

def execute_chart_code(
    code: str,
    df: pd.DataFrame
) -> Optional[object]:
    """
    Safely execute LLM-generated Plotly code.

    Runs in an isolated namespace with only the libraries
    the LLM is allowed to use. Returns the `fig` variable.

    Args:
        code: Python code string from LLM
        df: DataFrame to pass into the code namespace

    Returns:
        Plotly figure object, or None if execution failed
    """
    import plotly.express as px
    import plotly.graph_objects as go

    # Isolated namespace — LLM code runs here
    namespace = {
        "df": df.copy(),
        "pd": pd,
        "px": px,
        "go": go,
        "COLORS": [
            "#4F86C6", "#F4845F", "#57B894", "#A78BFA",
            "#F59E0B", "#EC4899", "#06B6D4", "#84CC16"
        ]
    }

    try:
        exec(code, namespace)

        fig = namespace.get("fig")
        if fig is None:
            print("[CHART BUILDER] Code executed but no 'fig' variable found")
            return None

        return fig

    except Exception as e:
        print(f"[CHART BUILDER] Code execution error: {e}")
        print(f"[CHART BUILDER] Traceback:\n{traceback.format_exc()}")
        print(f"[CHART BUILDER] Code that failed:\n{code}")
        return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def build_and_render_chart(
    df: pd.DataFrame,
    question: str,
    llm_provider: str = "claude_sonnet"
) -> tuple[bool, dict]:
    """
    Full pipeline: generate chart code → execute → render in Streamlit.

    Args:
        df: Query result DataFrame
        question: Original user question
        llm_provider: LLM to use for chart code generation

    Returns:
        Tuple of (rendered successfully, token usage dict)
    """
    empty_tokens = {"input": 0, "output": 0}

    if df is None or df.empty:
        st.dataframe(df, use_container_width=True)
        return False, empty_tokens

    # Skip chart for single-column or very wide results
    if len(df.columns) < 2 and len(df) > 1:
        st.dataframe(df, use_container_width=True)
        return False, empty_tokens

    with st.spinner("📊 Building chart..."):
        code, tokens = generate_chart_code(df, question, llm_provider)

    if not code:
        print("[CHART BUILDER] No code generated, falling back to table")
        return False, tokens

    fig = execute_chart_code(code, df)

    if fig is None:
        print("[CHART BUILDER] Execution failed, falling back to table")
        return False, tokens

    # Render chart
    st.plotly_chart(fig, use_container_width=True)

    # Show generated code in expander for debugging
    with st.expander("🔍 View generated chart code", expanded=False):
        st.code(code, language="python")

    return True, tokens
