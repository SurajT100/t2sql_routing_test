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


def _is_likely_datetime(series: pd.Series) -> bool:
    """Heuristic: treat as datetime only when mostly parseable and not numeric-like."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    converted = pd.to_datetime(series, errors="coerce")
    threshold = max(3, int(len(series) * 0.6))
    return converted.notna().sum() >= threshold


def _coerce_numeric_if_possible(series: pd.Series) -> pd.Series:
    """Convert object-like numeric columns (e.g. '123.45') safely to numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    converted = pd.to_numeric(series, errors="coerce")
    threshold = max(3, int(len(series) * 0.6))
    if converted.notna().sum() >= threshold:
        return converted
    return series


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

    # Keep planner I/O visible for debugging in UI
    st.session_state['chart_planner_debug'] = {
        'question': question,
        'prompt': prompt,
        'raw_response': response,
        'profile': profile
    }

    plan = _safe_json_loads(response)
    if not plan:
        return None, tokens

    st.session_state['chart_planner_debug']['plan'] = plan
    return plan, tokens


def _apply_plan(df: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    x = plan.get("x")
    y = plan.get("y")
    agg = (plan.get("aggregation") or "none").lower()

    # Ensure chart axes use real usable values
    if x and x in out.columns and _is_likely_datetime(out[x]):
        out[x] = pd.to_datetime(out[x], errors="coerce")
    if y and y in out.columns:
        out[y] = _coerce_numeric_if_possible(out[y])

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

    # Prevent blank charts from null axis/value rows
    drop_cols = [c for c in [x, y] if c and c in out.columns]
    if drop_cols:
        out = out.dropna(subset=drop_cols)

    # Drop textual null-like categories that often come from SQL output formatting
    if x and x in out.columns:
        out[x] = out[x].astype(str).str.strip()
        out = out[~out[x].str.lower().isin({'null', 'none', 'nan', ''})]

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
        if color in df.columns:
            chart = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(y=alt.Y(y), color=alt.Color(color))
        else:
            chart = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#4F86C6").encode(y=alt.Y(y))
    elif chart_type == "line":
        if color in df.columns:
            chart = base.mark_line(point=True, strokeWidth=3).encode(y=alt.Y(y), color=alt.Color(color))
        else:
            chart = base.mark_line(point=True, strokeWidth=3, color="#4F86C6").encode(y=alt.Y(y))
    elif chart_type == "area":
        if color in df.columns:
            chart = base.mark_area(opacity=0.65).encode(y=alt.Y(y), color=alt.Color(color))
        else:
            chart = base.mark_area(opacity=0.65, color="#4F86C6").encode(y=alt.Y(y))
    elif chart_type == "scatter":
        if color in df.columns:
            chart = base.mark_circle(size=100, opacity=0.8).encode(y=alt.Y(y), color=alt.Color(color), tooltip=list(df.columns))
        else:
            chart = base.mark_circle(size=100, opacity=0.8, color="#4F86C6").encode(y=alt.Y(y), tooltip=list(df.columns))
    else:
        return False

    st.altair_chart(chart.interactive(), use_container_width=True)
    return True




def _render_plotly_xy(df: pd.DataFrame, plan: dict[str, Any]) -> bool:
    import plotly.express as px

    chart_type = (plan.get("chart_type") or "").lower()
    x = plan.get("x")
    y = plan.get("y")
    color = plan.get("color") if plan.get("color") in df.columns else None
    title = plan.get("title") or "Query Chart"

    if not x or not y or x not in df.columns or y not in df.columns:
        return False

    if chart_type == "bar":
        fig = px.bar(df, x=x, y=y, color=color, title=title, text_auto='.2s')
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y, color=color, title=title, markers=True)
    elif chart_type == "area":
        fig = px.area(df, x=x, y=y, color=color, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
    else:
        return False

    fig.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
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






def _find_column_by_keywords(df: pd.DataFrame, keywords: list[str], numeric_required: bool = False) -> Optional[str]:
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in keywords):
            if numeric_required and not pd.api.types.is_numeric_dtype(df[col]):
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.notna().sum() < max(3, int(len(df) * 0.6)):
                    continue
            return col
    return None


def _build_wobby_like_recommendations(df: pd.DataFrame, question: str) -> list[dict[str, Any]]:
    """Deterministic smart recommendations inspired by BI products."""
    recommendations: list[dict[str, Any]] = []
    q = question.lower()

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols or not cat_cols:
        return recommendations

    region_col = _find_column_by_keywords(df, ["region", "state", "zone", "territory"])
    pipeline_col = _find_column_by_keywords(df, ["pipeline", "potential", "open"], numeric_required=True)
    conversion_col = _find_column_by_keywords(df, ["conversion", "win_rate", "win rate", "close_rate", "close rate"], numeric_required=True)

    # Pattern: strong pipeline but poor conversion -> scatter + rank bar
    if pipeline_col and conversion_col and ("pipeline" in q and ("conversion" in q or "win rate" in q or "poor" in q or "strong" in q)):
        dim = region_col or cat_cols[0]
        recommendations.append({
            "label": "Recommended: Risk Quadrant",
            "render": "chart",
            "chart_type": "scatter",
            "x": pipeline_col,
            "y": conversion_col,
            "color": dim,
            "aggregation": "none",
            "sort": "x_desc",
            "limit": DEFAULT_TOP_N,
            "title": f"{pipeline_col} vs {conversion_col} by {dim}",
            "reason": "Shows high pipeline but low conversion outliers clearly."
        })
        recommendations.append({
            "label": "Alternative: Conversion Ranking",
            "render": "chart",
            "chart_type": "bar",
            "x": dim,
            "y": conversion_col,
            "color": None,
            "aggregation": "none",
            "sort": "y_asc",
            "limit": DEFAULT_TOP_N,
            "title": f"{conversion_col} by {dim} (low to high)",
            "reason": "Ranks weakest conversion regions for action."
        })
        return recommendations

    # Generic defaults
    dim = region_col or cat_cols[0]
    metric = numeric_cols[0]
    recommendations.append({
        "label": "Recommended: Category Comparison",
        "render": "chart",
        "chart_type": "bar",
        "x": dim,
        "y": metric,
        "color": None,
        "aggregation": "none",
        "sort": "y_desc",
        "limit": DEFAULT_TOP_N,
        "title": f"{metric} by {dim}",
        "reason": "Best default for comparing categories."
    })

    if len(numeric_cols) > 1:
        recommendations.append({
            "label": "Alternative: Metric Relationship",
            "render": "chart",
            "chart_type": "scatter",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "color": dim,
            "aggregation": "none",
            "sort": "x_desc",
            "limit": DEFAULT_TOP_N,
            "title": f"{numeric_cols[1]} vs {numeric_cols[0]}",
            "reason": "Useful for relationship and outlier detection."
        })

    return recommendations

def _render_manual_chart(df: pd.DataFrame) -> bool:
    """Render user-selected chart using chosen x/y axes."""
    import plotly.express as px

    if df is None or df.empty:
        return False

    cols = list(df.columns)
    if len(cols) < 2:
        st.info("Manual chart mode requires at least 2 columns.")
        return False

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    default_y = numeric_cols[0] if numeric_cols else cols[1]
    default_x = cols[0] if cols[0] != default_y else cols[1]

    m1, m2, m3 = st.columns([1, 1, 1])
    with m1:
        chart_type = st.selectbox(
            "Chart Type",
            ["bar", "line", "area", "scatter", "pie", "donut"],
            index=0,
            key="manual_chart_type"
        )
    with m2:
        x = st.selectbox("X Axis", cols, index=cols.index(default_x), key="manual_chart_x")
    with m3:
        y = st.selectbox("Y Axis", cols, index=cols.index(default_y), key="manual_chart_y")

    if x == y:
        st.warning("Please choose different columns for X and Y axes.")
        return False

    # Clean selected columns for plotting
    df_plot = df.copy()
    if y in df_plot.columns:
        df_plot[y] = pd.to_numeric(df_plot[y], errors="coerce")
    df_plot = df_plot.dropna(subset=[x, y])

    if df_plot.empty:
        st.info("No plottable rows for selected axes.")
        return False

    title = f"{y} by {x}"
    if chart_type == "bar":
        fig = px.bar(df_plot, x=x, y=y, title=title, text_auto='.2s')
    elif chart_type == "line":
        fig = px.line(df_plot, x=x, y=y, title=title, markers=True)
    elif chart_type == "area":
        fig = px.area(df_plot, x=x, y=y, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(df_plot, x=x, y=y, title=title)
    elif chart_type == "pie":
        fig = px.pie(df_plot, names=x, values=y, title=title)
    elif chart_type == "donut":
        fig = px.pie(df_plot, names=x, values=y, hole=0.45, title=title)
    else:
        return False

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

    view_mode = st.radio(
        "Chart Mode",
        ["Smart (Auto)", "Manual"],
        horizontal=True,
        key="chart_view_mode"
    )

    if view_mode == "Manual":
        rendered = _render_manual_chart(df)
        return rendered, empty_tokens

    # Wobby-like deterministic recommender first for consistency
    deterministic_plans = _build_wobby_like_recommendations(df, question)
    plan = None
    tokens = empty_tokens

    if deterministic_plans:
        labels = [p.get("label", f"Option {i+1}") for i, p in enumerate(deterministic_plans)]
        picked = st.selectbox("Smart Chart Recommendation", labels, index=0, key="smart_reco_pick")
        plan = deterministic_plans[labels.index(picked)]
    else:
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
        st.info("📋 Chart plan produced no plottable rows from the final table. Showing data instead.")
        return False, tokens

    chart_type = (plan.get("chart_type") or "").lower()

    rendered = False
    if chart_type in {"pie", "donut"}:
        rendered = _render_plotly_pie(prepared, plan)
    else:
        rendered = _render_plotly_xy(prepared, plan)
        if not rendered:
            rendered = _render_altair_chart(prepared, plan)

    if rendered:
        reason = plan.get("reason")
        if reason:
            st.caption(f"🧠 Chart logic: {reason}")

    with st.expander("🔍 View chart planner input/output", expanded=False):
        if deterministic_plans:
            st.caption("Deterministic recommendation plan")
            st.json(plan)
        else:
            dbg = st.session_state.get('chart_planner_debug', {})
            if dbg.get('prompt'):
                st.caption("Planner Prompt")
                st.code(dbg['prompt'][:8000])
            if dbg.get('raw_response'):
                st.caption("Planner Raw Response")
                st.code(str(dbg['raw_response'])[:4000])
            if dbg.get('plan'):
                st.caption("Parsed Plan")
                st.json(dbg['plan'])
        st.caption("Prepared table used for chart")
        st.dataframe(prepared.head(30), use_container_width=True)

    return rendered, tokens
