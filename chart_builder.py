"""
Chart Builder
=============
Adaptive chart planner + renderer for Streamlit query results.

- Uses LLM to choose chart intent from question + dataframe profile
- Uses Altair for modern visual styling (primary)
- Falls back to Plotly/table when needed
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd
import streamlit as st


MAX_PROFILE_ROWS = 50
DEFAULT_TOP_N = 20


VIS_PLANNING_GUIDELINES = """
You are a senior data-visualization planner.
Return ONLY strict JSON (no markdown) with this schema:
{
  "render": "chart" | "table" | "metric",
  "chart_type": "bar" | "line" | "area" | "scatter" | "pie" | "donut" | null,
  "x": "column_name_or_null",
  "y": "column_name_or_null",
  "color": "column_name_or_null",
  "aggregation": "none" | "sum" | "avg" | "count",
  "sort": "x_asc" | "x_desc" | "y_asc" | "y_desc" | "time_asc",
  "limit": integer,
  "title": "short_title",
  "reason": "one sentence"
}

Rules:
1) Choose chart based on BOTH question intent and data shape.
2) If only one value/row is meaningful, use render='metric'.
3) If data cannot be visualized reliably, use render='table'.
4) Prefer time-series charts when question asks trend/growth over time.
5) For long category results, set reasonable limit (10-25).
6) Pick existing columns exactly as named.
7) If no safe chart mapping, do not guess.
"""


def _normalize_col(col: str) -> str:
    return str(col).strip()


def _profile_df(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = []
    categorical_cols = []

    for col in df.columns:
        if col in numeric_cols:
            continue
        converted = pd.to_datetime(df[col], errors="coerce")
        if converted.notna().sum() >= max(3, int(len(df) * 0.6)):
            datetime_cols.append(col)
        else:
            categorical_cols.append(col)

    profile = {
        "rows": len(df),
        "cols": len(df.columns),
        "columns": [str(c) for c in df.columns],
        "numeric_columns": numeric_cols,
        "datetime_columns": datetime_cols,
        "categorical_columns": categorical_cols,
        "sample": df.head(min(MAX_PROFILE_ROWS, len(df))).to_dict(orient="records"),
    }
    return profile


def _safe_json_loads(text: str) -> Optional[dict[str, Any]]:
    content = text.strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1]).strip()
    try:
        return json.loads(content)
    except Exception:
        return None


def _plan_chart(df: pd.DataFrame, question: str, llm_provider: str) -> tuple[Optional[dict[str, Any]], dict[str, int]]:
    from llm_v2 import call_llm

    profile = _profile_df(df)
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"DATAFRAME_PROFILE_JSON:\n{json.dumps(profile, default=str)[:20000]}\n\n"
        f"{VIS_PLANNING_GUIDELINES}\n"
    )

    response, tokens = call_llm(prompt, llm_provider)
    plan = _safe_json_loads(response)
    if not plan:
        return None, tokens
    return plan, tokens


def _apply_plan(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    x = plan.get("x")
    y = plan.get("y")
    agg = (plan.get("aggregation") or "none").lower()

    if x and x in out.columns and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(out[x], errors="coerce")):
        out[x] = pd.to_datetime(out[x], errors="coerce")

    if x and y and x in out.columns and y in out.columns and agg in {"sum", "avg", "count"}:
        if agg == "sum":
            out = out.groupby(x, as_index=False)[y].sum()
        elif agg == "avg":
            out = out.groupby(x, as_index=False)[y].mean()
        else:
            out = out.groupby(x, as_index=False)[y].count()

    sort_mode = plan.get("sort")
    if sort_mode == "time_asc" and x in out.columns:
        out = out.sort_values(x, ascending=True)
    elif sort_mode == "x_asc" and x in out.columns:
        out = out.sort_values(x, ascending=True)
    elif sort_mode == "x_desc" and x in out.columns:
        out = out.sort_values(x, ascending=False)
    elif sort_mode == "y_asc" and y in out.columns:
        out = out.sort_values(y, ascending=True)
    elif sort_mode == "y_desc" and y in out.columns:
        out = out.sort_values(y, ascending=False)

    limit = plan.get("limit", DEFAULT_TOP_N)
    if isinstance(limit, int) and limit > 0 and len(out) > limit:
        out = out.head(limit)

    return out


def _render_altair_chart(df: pd.DataFrame, plan: dict[str, Any]) -> bool:
    import altair as alt

    chart_type = plan.get("chart_type")
    x = plan.get("x")
    y = plan.get("y")
    color = plan.get("color")
    title = plan.get("title") or "Query Chart"

    if not x or x not in df.columns:
        return False

    base = alt.Chart(df).properties(title=title).encode(x=alt.X(x, sort=None))

    if chart_type in {"bar", "line", "area", "scatter"} and (not y or y not in df.columns):
        return False

    if chart_type == "bar":
        chart = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(y=alt.Y(y), color=color if color in df.columns else alt.value("#4F86C6"))
    elif chart_type == "line":
        chart = base.mark_line(point=True, strokeWidth=3).encode(y=alt.Y(y), color=color if color in df.columns else alt.value("#4F86C6"))
    elif chart_type == "area":
        chart = base.mark_area(opacity=0.65).encode(y=alt.Y(y), color=color if color in df.columns else alt.value("#4F86C6"))
    elif chart_type == "scatter":
        chart = base.mark_circle(size=100, opacity=0.8).encode(y=alt.Y(y), color=color if color in df.columns else alt.value("#4F86C6"), tooltip=list(df.columns))
    else:
        return False

    st.altair_chart(chart.interactive(), use_container_width=True)
    return True


def _render_plotly_pie(df: pd.DataFrame, plan: dict[str, Any]) -> bool:
    import plotly.express as px

    x = plan.get("x")
    y = plan.get("y")
    if not x or not y or x not in df.columns or y not in df.columns:
        return False

    chart_type = plan.get("chart_type")
    if chart_type == "donut":
        fig = px.pie(df, names=x, values=y, hole=0.45, title=plan.get("title") or "Query Chart")
    else:
        fig = px.pie(df, names=x, values=y, title=plan.get("title") or "Query Chart")

    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    return True


def build_and_render_chart(
    df: pd.DataFrame,
    question: str,
    llm_provider: str = "claude_sonnet"
) -> tuple[bool, dict]:
    """
    Generate chart plan with LLM and render with modern chart libraries.
    """
    empty_tokens = {"input": 0, "output": 0}

    if df is None or df.empty:
        return False, empty_tokens

    # Handle very small result sets smartly
    if len(df) == 1:
        st.info("📌 Single-row result detected. Showing key values instead of a chart.")
        cols = st.columns(min(len(df.columns), 4))
        row = df.iloc[0]
        for i, col in enumerate(df.columns[:4]):
            cols[i].metric(str(col), str(row[col]))
        return True, empty_tokens

    with st.spinner("📊 Planning best visualization..."):
        plan, tokens = _plan_chart(df, question, llm_provider)

    if not plan:
        return False, tokens

    render = (plan.get("render") or "chart").lower()
    if render == "table":
        return False, tokens

    if render == "metric":
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            metric_col = numeric_cols[0]
            st.metric(metric_col, f"{df[metric_col].iloc[0]:,.2f}" if len(df) else "0")
            return True, tokens
        return False, tokens

    prepared = _apply_plan(df, plan)
    if prepared.empty:
        return False, tokens

    chart_type = (plan.get("chart_type") or "").lower()

    rendered = False
    if chart_type in {"pie", "donut"}:
        rendered = _render_plotly_pie(prepared, plan)
    else:
        rendered = _render_altair_chart(prepared, plan)

    if rendered:
        reason = plan.get("reason")
        if reason:
            st.caption(f"🧠 Chart logic: {reason}")

    return rendered, tokens
