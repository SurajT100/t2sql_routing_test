"""
Chart Utils
============
Automatic chart detection and rendering for SQL query results.

Flow:
    DataFrame → detect_chart_type() → render_chart()
    If ambiguous → get_chart_type_from_llm() → render_chart()

Usage:
    from chart_utils import detect_chart_config, render_chart

    config = detect_chart_config(df, question, llm_provider="groq")
    render_chart(df, config)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List, Tuple
import streamlit as st
import json


# =============================================================================
# CHART CONFIG DATACLASS
# =============================================================================

def make_chart_config(
    chart_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    y_cols: Optional[List[str]] = None,
    color: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    metric_value: Optional[Any] = None,
    metric_label: str = "",
    confidence: str = "rule_based"  # "rule_based" | "llm"
) -> Dict[str, Any]:
    return {
        "chart_type": chart_type,   # bar | line | grouped_bar | pie | metric | multi_line | table
        "x": x,
        "y": y,
        "y_cols": y_cols or [],
        "color": color,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "metric_value": metric_value,
        "metric_label": metric_label,
        "confidence": confidence
    }


# =============================================================================
# RULE-BASED CHART DETECTION
# =============================================================================

def detect_chart_config(
    df: pd.DataFrame,
    question: str = "",
    llm_provider: str = "groq"
) -> Dict[str, Any]:
    """
    Detect the best chart type for the given DataFrame and question.

    Priority:
        1. Rule-based detection (free, instant)
        2. LLM fallback for ambiguous cases (Groq, nearly free)

    Args:
        df: Query result DataFrame
        question: Original user question (helps LLM)
        llm_provider: LLM to use if rule-based is ambiguous

    Returns:
        Chart config dict
    """
    if df is None or df.empty:
        return make_chart_config("table")

    # Classify columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    text_cols = df.select_dtypes(include='object').columns.tolist()
    all_cols = df.columns.tolist()

    date_cols = [
        c for c in all_cols
        if any(kw in c.lower() for kw in [
            'date', 'month', 'year', 'quarter', 'week', 'period',
            'fy', 'q1', 'q2', 'q3', 'q4', 'jan', 'feb', 'mar',
            'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ])
    ]

    rows = len(df)
    n_numeric = len(numeric_cols)
    n_text = len(text_cols)

    print(f"[CHART] rows={rows}, numeric={numeric_cols}, text={text_cols}, date={date_cols}")

    # ── Single value → Metric card ──────────────────────────────────────────
    if rows == 1 and n_numeric == 1 and n_text == 0:
        val = df[numeric_cols[0]].iloc[0]
        return make_chart_config(
            "metric",
            metric_value=val,
            metric_label=numeric_cols[0],
            title=question[:60] if question else numeric_cols[0]
        )

    # ── Single row, multiple metrics → Metric cards ─────────────────────────
    if rows == 1 and n_numeric > 1:
        return make_chart_config(
            "multi_metric",
            y_cols=numeric_cols,
            title=question[:60] if question else "Summary"
        )

    # ── Date/time column + numeric → Line chart ─────────────────────────────
    if date_cols and n_numeric >= 1:
        x_col = date_cols[0]
        if n_numeric == 1:
            return make_chart_config(
                "line",
                x=x_col,
                y=numeric_cols[0],
                title=question[:60] if question else f"{numeric_cols[0]} over time",
                x_label=x_col,
                y_label=numeric_cols[0]
            )
        else:
            return make_chart_config(
                "multi_line",
                x=x_col,
                y_cols=numeric_cols,
                title=question[:60] if question else "Trends over time",
                x_label=x_col
            )

    # ── 1 text + 1 numeric → Bar chart ──────────────────────────────────────
    if n_text == 1 and n_numeric == 1:
        # Use horizontal bar if many categories or long labels
        use_horizontal = (
            rows > 8 or
            df[text_cols[0]].astype(str).str.len().max() > 15
        )
        return make_chart_config(
            "bar_h" if use_horizontal else "bar",
            x=text_cols[0],
            y=numeric_cols[0],
            title=question[:60] if question else f"{numeric_cols[0]} by {text_cols[0]}",
            x_label=text_cols[0],
            y_label=numeric_cols[0]
        )

    # ── 1 text + 2 numeric → Grouped bar ────────────────────────────────────
    if n_text == 1 and n_numeric == 2:
        return make_chart_config(
            "grouped_bar",
            x=text_cols[0],
            y_cols=numeric_cols,
            title=question[:60] if question else f"Comparison by {text_cols[0]}",
            x_label=text_cols[0]
        )

    # ── 2 text cols + 1 numeric → Grouped bar with color ────────────────────
    if n_text == 2 and n_numeric == 1:
        return make_chart_config(
            "bar",
            x=text_cols[0],
            y=numeric_cols[0],
            color=text_cols[1],
            title=question[:60] if question else f"{numeric_cols[0]} by {text_cols[0]}",
            x_label=text_cols[0],
            y_label=numeric_cols[0]
        )

    # ── 1 text + many numeric → Ask LLM ─────────────────────────────────────
    if n_text >= 1 and n_numeric > 2:
        print(f"[CHART] Ambiguous — asking LLM for chart type")
        llm_config = get_chart_type_from_llm(question, all_cols, df.head(3).to_dict(), llm_provider)
        if llm_config:
            return llm_config

    # ── Fallback → Table ─────────────────────────────────────────────────────
    print(f"[CHART] No chart rule matched, defaulting to table")
    return make_chart_config("table")


# =============================================================================
# LLM CHART TYPE SELECTOR (Groq fallback)
# =============================================================================

def get_chart_type_from_llm(
    question: str,
    columns: List[str],
    sample_data: Dict,
    llm_provider: str = "groq"
) -> Optional[Dict[str, Any]]:
    """
    Ask Groq to decide chart type for ambiguous cases.
    Fast, cheap (Groq), only called when rule-based fails.
    """
    try:
        from llm_v2 import call_llm

        prompt = f"""You are a data visualization expert. 
Choose the best chart for this data.

QUESTION: {question}
COLUMNS: {columns}
SAMPLE DATA: {json.dumps(sample_data, default=str)[:500]}

AVAILABLE CHART TYPES:
- bar: 1 category + 1 metric
- bar_h: horizontal bar (many categories or long labels)
- grouped_bar: 1 category + 2 metrics side by side
- line: time series (x must be date/period column)
- multi_line: time series with multiple metrics
- pie: parts of a whole (max 8 slices)
- table: complex/unsuitable for charts

Return ONLY valid JSON:
{{"chart_type": "bar", "x": "column_name", "y": "column_name", "y_cols": [], "color": null, "title": "Chart title"}}

Rules:
- x must be an exact column name from COLUMNS
- y must be an exact column name from COLUMNS  
- y_cols is list of column names for multi-series charts
- color is optional grouping column name or null"""

        response, _ = call_llm(prompt, llm_provider)

        # Parse response
        try:
            # Strip markdown if present
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            data = json.loads(clean)
            chart_type = data.get("chart_type", "table")

            return make_chart_config(
                chart_type=chart_type,
                x=data.get("x"),
                y=data.get("y"),
                y_cols=data.get("y_cols", []),
                color=data.get("color"),
                title=data.get("title", question[:60]),
                confidence="llm"
            )
        except Exception as e:
            print(f"[CHART] LLM response parse error: {e}")
            return None

    except Exception as e:
        print(f"[CHART] LLM chart detection error: {e}")
        return None


# =============================================================================
# CHART RENDERING
# =============================================================================

# Consistent color palette for all charts
CHART_COLORS = [
    "#4F86C6",  # Blue
    "#F4845F",  # Orange
    "#57B894",  # Green
    "#A78BFA",  # Purple
    "#F59E0B",  # Amber
    "#EC4899",  # Pink
    "#06B6D4",  # Cyan
    "#84CC16",  # Lime
]

CHART_TEMPLATE = "plotly_white"

CHART_LAYOUT = dict(
    font=dict(family="Segoe UI, sans-serif", size=13),
    title_font=dict(size=15, color="#1e293b"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)


def _format_number(val) -> str:
    """Format large numbers for display."""
    try:
        v = float(val)
        if abs(v) >= 1_00_00_000:   # 1 Cr+
            return f"₹{v/1_00_00_000:.2f} Cr"
        elif abs(v) >= 1_00_000:    # 1 Lakh+
            return f"₹{v/1_00_000:.2f} L"
        elif abs(v) >= 1_000:
            return f"{v:,.0f}"
        else:
            return f"{v:,.2f}"
    except:
        return str(val)


def render_chart(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Render the appropriate chart in Streamlit using Plotly.

    Args:
        df: Result DataFrame
        config: Chart config from detect_chart_config()
    """
    chart_type = config.get("chart_type", "table")
    title = config.get("title", "")
    x_col = config.get("x")
    y_col = config.get("y")
    y_cols = config.get("y_cols", [])
    color_col = config.get("color")

    # Validate columns exist in df
    def col_exists(c):
        return c and c in df.columns

    try:
        # ── Metric card ──────────────────────────────────────────────────────
        if chart_type == "metric":
            val = config.get("metric_value")
            label = config.get("metric_label", "")
            st.metric(label=label, value=_format_number(val))
            return

        # ── Multi metric cards ───────────────────────────────────────────────
        if chart_type == "multi_metric":
            valid_cols = [c for c in y_cols if col_exists(c)]
            cols = st.columns(min(len(valid_cols), 4))
            for i, col_name in enumerate(valid_cols):
                with cols[i % len(cols)]:
                    val = df[col_name].iloc[0]
                    st.metric(label=col_name, value=_format_number(val))
            return

        # ── Bar chart ────────────────────────────────────────────────────────
        if chart_type == "bar":
            if not col_exists(x_col) or not col_exists(y_col):
                _fallback_table(df)
                return

            # Sort by value descending for cleaner look
            df_sorted = df.sort_values(y_col, ascending=False)

            fig = px.bar(
                df_sorted,
                x=x_col,
                y=y_col,
                color=color_col if col_exists(color_col) else None,
                title=title,
                color_discrete_sequence=CHART_COLORS,
                template=CHART_TEMPLATE,
                text=y_col
            )
            fig.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside',
                marker_line_width=0
            )
            fig.update_layout(**CHART_LAYOUT)
            fig.update_xaxes(title_text=config.get("x_label", x_col))
            fig.update_yaxes(title_text=config.get("y_label", y_col))
            st.plotly_chart(fig, use_container_width=True)
            return

        # ── Horizontal bar ───────────────────────────────────────────────────
        if chart_type == "bar_h":
            if not col_exists(x_col) or not col_exists(y_col):
                _fallback_table(df)
                return

            df_sorted = df.sort_values(y_col, ascending=True)

            fig = px.bar(
                df_sorted,
                x=y_col,
                y=x_col,
                orientation='h',
                title=title,
                color_discrete_sequence=CHART_COLORS,
                template=CHART_TEMPLATE,
                text=y_col
            )
            fig.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside',
                marker_line_width=0
            )
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            return

        # ── Grouped bar ──────────────────────────────────────────────────────
        if chart_type == "grouped_bar":
            if not col_exists(x_col):
                _fallback_table(df)
                return

            valid_y = [c for c in y_cols if col_exists(c)]
            if not valid_y:
                _fallback_table(df)
                return

            fig = go.Figure()
            for i, y_c in enumerate(valid_y):
                fig.add_trace(go.Bar(
                    name=y_c,
                    x=df[x_col],
                    y=df[y_c],
                    marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                    text=df[y_c],
                    texttemplate='%{text:,.0f}',
                    textposition='outside'
                ))

            fig.update_layout(
                barmode='group',
                title=title,
                template=CHART_TEMPLATE,
                **CHART_LAYOUT
            )
            fig.update_xaxes(title_text=config.get("x_label", x_col))
            st.plotly_chart(fig, use_container_width=True)
            return

        # ── Line chart ───────────────────────────────────────────────────────
        if chart_type == "line":
            if not col_exists(x_col) or not col_exists(y_col):
                _fallback_table(df)
                return

            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=CHART_COLORS,
                template=CHART_TEMPLATE,
                markers=True
            )
            fig.update_traces(line_width=2.5, marker_size=7)
            fig.update_layout(**CHART_LAYOUT)
            fig.update_xaxes(title_text=config.get("x_label", x_col))
            fig.update_yaxes(title_text=config.get("y_label", y_col))
            st.plotly_chart(fig, use_container_width=True)
            return

        # ── Multi-line chart ─────────────────────────────────────────────────
        if chart_type == "multi_line":
            if not col_exists(x_col):
                _fallback_table(df)
                return

            valid_y = [c for c in y_cols if col_exists(c)]
            if not valid_y:
                _fallback_table(df)
                return

            fig = go.Figure()
            for i, y_c in enumerate(valid_y):
                fig.add_trace(go.Scatter(
                    name=y_c,
                    x=df[x_col],
                    y=df[y_c],
                    mode='lines+markers',
                    line=dict(
                        color=CHART_COLORS[i % len(CHART_COLORS)],
                        width=2.5
                    ),
                    marker=dict(size=7)
                ))

            fig.update_layout(
                title=title,
                template=CHART_TEMPLATE,
                **CHART_LAYOUT
            )
            fig.update_xaxes(title_text=config.get("x_label", x_col))
            st.plotly_chart(fig, use_container_width=True)
            return

        # ── Pie chart ────────────────────────────────────────────────────────
        if chart_type == "pie":
            if not col_exists(x_col) or not col_exists(y_col):
                _fallback_table(df)
                return

            # Cap at 8 slices — group rest as "Other"
            if len(df) > 8:
                top = df.nlargest(7, y_col)
                other_val = df[y_col].sum() - top[y_col].sum()
                other_row = pd.DataFrame({x_col: ["Other"], y_col: [other_val]})
                df_pie = pd.concat([top, other_row], ignore_index=True)
            else:
                df_pie = df

            fig = px.pie(
                df_pie,
                names=x_col,
                values=y_col,
                title=title,
                color_discrete_sequence=CHART_COLORS,
                template=CHART_TEMPLATE,
                hole=0.35  # Donut style — cleaner
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            return

        # ── Default: Table ───────────────────────────────────────────────────
        _fallback_table(df)

    except Exception as e:
        print(f"[CHART] Render error: {e}")
        st.warning(f"⚠️ Chart rendering failed: {str(e)[:100]}")
        _fallback_table(df)


def _fallback_table(df: pd.DataFrame) -> None:
    """Show DataFrame as styled table when chart is not possible."""
    st.dataframe(df, use_container_width=True)
