import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd

# pycausalimpact uses DataFrame.applymap, removed in pandas 2.2 (renamed to .map)
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map
import numpy as np
import plotly.graph_objects as go
from causalimpact import CausalImpact
from datetime import timedelta

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SEO Causal Impact Analysis",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INTERVENTIONS = {
    "Site Migration": {
        "date": "2025-04-15",
        "description": "Full site migration from HTTP to HTTPS with URL restructuring",
        "expected_effect": "Initial traffic drop followed by gradual recovery",
        "icon": "🔄",
    },
    "Core Web Vitals Fix": {
        "date": "2025-06-01",
        "description": "Comprehensive Core Web Vitals optimization — LCP, FID, CLS improvements",
        "expected_effect": "Gradual improvement in search visibility (+15 %)",
        "icon": "⚡",
    },
    "Content Overhaul": {
        "date": "2025-08-01",
        "description": "Major content refresh: 150+ pages updated with E-E-A-T signals",
        "expected_effect": "Step-change increase in organic traffic (+20 %)",
        "icon": "📝",
    },
    "Technical SEO Audit": {
        "date": "2025-10-01",
        "description": "Technical audit fixing crawlability, indexation & schema markup",
        "expected_effect": "Moderate sustained improvement (+10 %)",
        "icon": "🔧",
    },
}

METRIC_MAP = {
    "Search Console Clicks": "search_clicks",
    "Organic Sessions (GA4)": "organic_sessions",
    "Organic Conversions (GA4)": "organic_conversions",
}

# ---------------------------------------------------------------------------
# Sample data generation
# ---------------------------------------------------------------------------
@st.cache_data
def generate_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=365, freq="D")
    t = np.arange(365)

    # --- Covariates (unaffected by SEO interventions) ----------------------
    weekly = np.sin(2 * np.pi * t / 7)
    branded_searches = 200 + 0.05 * t + 15 * weekly + np.random.normal(0, 8, 365)
    industry_trend = 500 + 0.10 * t + np.random.normal(0, 10, 365)

    # --- Base metrics (pre-intervention) -----------------------------------
    base_clicks = (
        0.8 * branded_searches + 0.3 * industry_trend + 50
        + 20 * weekly + np.random.normal(0, 12, 365)
    )
    base_sessions = (
        1.2 * branded_searches + 0.5 * industry_trend + 100
        + 30 * weekly + np.random.normal(0, 20, 365)
    )
    base_conversions = 0.04 * base_sessions + 3 + np.random.normal(0, 1.5, 365)

    # --- Apply intervention effects ----------------------------------------
    clicks = base_clicks.copy()
    sessions = base_sessions.copy()
    conversions = base_conversions.copy()

    def day_idx(date_str):
        return (pd.Timestamp(date_str) - pd.Timestamp("2025-01-01")).days

    # 1. Site Migration (2025-04-15): dip then recovery
    idx_mig = day_idx("2025-04-15")
    for i in range(idx_mig, min(idx_mig + 45, 365)):
        dip = -0.20 * np.exp(-0.05 * (i - idx_mig))  # exponential recovery
        clicks[i] *= (1 + dip)
        sessions[i] *= (1 + dip)
        conversions[i] *= (1 + dip)

    # 2. Core Web Vitals Fix (2025-06-01): gradual +15 %
    idx_cwv = day_idx("2025-06-01")
    for i in range(idx_cwv, 365):
        ramp = min((i - idx_cwv) / 30, 1.0)  # ramps over 30 days
        lift = 0.15 * ramp
        clicks[i] *= (1 + lift)
        sessions[i] *= (1 + lift)
        conversions[i] *= (1 + lift)

    # 3. Content Overhaul (2025-08-01): step +20 %
    idx_co = day_idx("2025-08-01")
    for i in range(idx_co, 365):
        clicks[i] *= 1.20
        sessions[i] *= 1.20
        conversions[i] *= 1.20

    # 4. Technical SEO Audit (2025-10-01): sustained +10 %
    idx_ta = day_idx("2025-10-01")
    for i in range(idx_ta, 365):
        clicks[i] *= 1.10
        sessions[i] *= 1.10
        conversions[i] *= 1.10

    # Ensure no negatives
    clicks = np.maximum(clicks, 0)
    sessions = np.maximum(sessions, 0)
    conversions = np.maximum(conversions, 0)

    return pd.DataFrame(
        {
            "search_clicks": np.round(clicks, 1),
            "organic_sessions": np.round(sessions, 1),
            "organic_conversions": np.round(conversions, 1),
            "branded_searches": np.round(branded_searches, 1),
            "industry_trend": np.round(industry_trend, 1),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_sorted_intervention_dates():
    return sorted(INTERVENTIONS.items(), key=lambda x: x[1]["date"])


def get_periods(intervention_name: str, data_index):
    sorted_interventions = get_sorted_intervention_dates()
    names = [n for n, _ in sorted_interventions]
    pos = names.index(intervention_name)
    intervention_date = sorted_interventions[pos][1]["date"]

    pre_start = data_index[0].strftime("%Y-%m-%d")
    pre_end = (pd.Timestamp(intervention_date) - timedelta(days=1)).strftime("%Y-%m-%d")
    post_start = intervention_date
    if pos < len(sorted_interventions) - 1:
        next_date = sorted_interventions[pos + 1][1]["date"]
        post_end = (pd.Timestamp(next_date) - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        post_end = data_index[-1].strftime("%Y-%m-%d")

    return [pre_start, pre_end], [post_start, post_end], intervention_date


def run_analysis(data, metric_col, pre_period, post_period, alpha):
    ci_data = data[[metric_col, "branded_searches", "industry_trend"]].copy()
    ci = CausalImpact(
        ci_data,
        pre_period,
        post_period,
        alpha=alpha,
        prior_level_sd=None,
    )
    return ci


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------
CHART_COLORS = dict(
    actual="#e63946",
    predicted="#4361ee",
    ci_fill="rgba(67, 97, 238, 0.15)",
    effect="#e63946",
    effect_fill="rgba(230, 57, 70, 0.15)",
    cum="#2a9d8f",
    cum_fill="rgba(42, 157, 143, 0.15)",
    vline="#adb5bd",
)


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=40),
        height=320,
        legend=dict(orientation="h", y=-0.18),
    )


def _add_vline(fig, date_str):
    fig.add_shape(
        type="line",
        x0=date_str, x1=date_str,
        y0=0, y1=1,
        yref="paper",
        line=dict(dash="dash", color=CHART_COLORS["vline"], width=1.5),
    )
    fig.add_annotation(
        x=date_str, y=1, yref="paper",
        text="Intervention", showarrow=False,
        font=dict(color=CHART_COLORS["vline"], size=11),
        yshift=10,
    )


def chart_original(ci, intervention_date, metric_label):
    llb = ci.trained_model.filter_results.loglikelihood_burn
    inf = ci.inferences.iloc[llb:]
    obs = pd.concat([ci.pre_data.iloc[llb:, 0], ci.post_data.iloc[:, 0]])

    fig = go.Figure()
    # CI band
    fig.add_trace(go.Scatter(
        x=inf.index, y=inf["preds_upper"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=inf.index, y=inf["preds_lower"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor=CHART_COLORS["ci_fill"], showlegend=False, hoverinfo="skip",
    ))
    # Predicted
    fig.add_trace(go.Scatter(
        x=inf.index, y=inf["preds"], mode="lines",
        line=dict(color=CHART_COLORS["predicted"], dash="dash", width=1.5),
        name="Predicted (counterfactual)",
    ))
    # Actual
    fig.add_trace(go.Scatter(
        x=obs.index, y=obs.values, mode="lines",
        line=dict(color=CHART_COLORS["actual"], width=2.5),
        name=f"Observed {metric_label}",
    ))
    _add_vline(fig, intervention_date)
    fig.update_layout(**_base_layout("Original vs Counterfactual Prediction"))
    return fig


def chart_pointwise(ci, intervention_date):
    llb = ci.trained_model.filter_results.loglikelihood_burn
    inf = ci.inferences.iloc[llb:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inf.index, y=inf["point_effects_upper"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=inf.index, y=inf["point_effects_lower"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor=CHART_COLORS["effect_fill"], showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=inf.index, y=inf["point_effects"], mode="lines",
        line=dict(color=CHART_COLORS["effect"], width=1.5),
        name="Pointwise effect",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#999", line_width=1)
    _add_vline(fig, intervention_date)
    fig.update_layout(**_base_layout("Pointwise Causal Effect"))
    return fig


def chart_cumulative(ci, intervention_date):
    llb = ci.trained_model.filter_results.loglikelihood_burn
    inf = ci.inferences.iloc[llb:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inf.index, y=inf["post_cum_effects_upper"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=inf.index, y=inf["post_cum_effects_lower"], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor=CHART_COLORS["cum_fill"], showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=inf.index, y=inf["post_cum_effects"], mode="lines",
        line=dict(color=CHART_COLORS["cum"], width=1.5),
        name="Cumulative effect",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#999", line_width=1)
    _add_vline(fig, intervention_date)
    fig.update_layout(**_base_layout("Cumulative Causal Effect"))
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # ---- Header -----------------------------------------------------------
    st.markdown(
        """
        <h1 style='margin-bottom:0'>📊 SEO Causal Impact Analysis</h1>
        <p style='color:#6c757d; font-size:1.1rem; margin-top:0'>
        Measure the true causal impact of SEO activities using Bayesian Structural Time Series
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "🔗 **Google Analytics 4 & Search Console integrations are coming soon!**  \n"
        "Currently using realistic sample data for demonstration purposes. "
        "Connect your own GA4 property and Search Console site to analyse real performance data.",
        icon="🚀",
    )

    data = generate_sample_data()

    # ---- Sidebar ----------------------------------------------------------
    with st.sidebar:
        st.header("Analysis Settings")

        selected_metric_label = st.selectbox(
            "Select Metric",
            list(METRIC_MAP.keys()),
            help="Choose the search performance metric to analyse.",
        )
        metric_col = METRIC_MAP[selected_metric_label]

        selected_activity = st.selectbox(
            "Select SEO Activity",
            list(INTERVENTIONS.keys()),
            help="Choose the SEO activity whose impact you want to measure.",
        )

        confidence = st.slider(
            "Confidence Level",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Width of the credible interval for the Bayesian model.",
        )

        st.divider()

        # Intervention info card
        info = INTERVENTIONS[selected_activity]
        pre_period, post_period, intervention_date = get_periods(
            selected_activity, data.index
        )
        st.markdown(f"### {info['icon']} {selected_activity}")
        st.markdown(f"**Date:** {info['date']}")
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Expected effect:** {info['expected_effect']}")
        st.caption(f"Pre-period: {pre_period[0]} to {pre_period[1]}")
        st.caption(f"Post-period: {post_period[0]} to {post_period[1]}")

        st.divider()
        run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

        st.divider()
        st.markdown(
            "<p style='color:#adb5bd; font-size:0.8rem'>"
            "Powered by <b>pycausalimpact</b> — a Python port of Google's "
            "CausalImpact R package using Bayesian structural time-series models."
            "</p>",
            unsafe_allow_html=True,
        )

    # ---- Run analysis -----------------------------------------------------
    if run_btn:
        alpha = round(1 - confidence, 2)
        with st.spinner("Running Bayesian structural time-series model — this may take a few seconds..."):
            try:
                ci = run_analysis(data, metric_col, pre_period, post_period, alpha)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return

        st.session_state["ci"] = ci
        st.session_state["intervention_date"] = intervention_date
        st.session_state["metric_label"] = selected_metric_label
        st.session_state["confidence"] = confidence

    # ---- Display results --------------------------------------------------
    if "ci" not in st.session_state:
        # Landing state — show a quick data preview
        st.markdown("### Sample Data Preview")
        col_left, col_right = st.columns(2)
        with col_left:
            st.line_chart(data[metric_col], height=260, use_container_width=True)
        with col_right:
            st.dataframe(data.head(30), use_container_width=True, height=260)
        st.caption("Select a metric and SEO activity in the sidebar, then click **Run Analysis**.")
        return

    ci = st.session_state["ci"]
    intervention_date = st.session_state["intervention_date"]
    metric_label = st.session_state["metric_label"]
    conf = st.session_state["confidence"]

    # --- Summary metrics ---------------------------------------------------
    sd = ci.summary_data
    avg_effect = sd.loc["abs_effect", "average"]
    cum_effect = sd.loc["abs_effect", "cumulative"]
    rel_effect = sd.loc["rel_effect", "average"] * 100
    p_value = ci.p_value
    prob_effect = (1 - p_value) * 100

    st.markdown("### Key Findings")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Avg. Daily Causal Effect",
        f"{avg_effect:+,.1f}",
        delta=f"{rel_effect:+.1f} %",
    )
    c2.metric(
        "Cumulative Effect",
        f"{cum_effect:+,.1f}",
    )
    c3.metric(
        "Probability of Effect",
        f"{prob_effect:.1f} %",
    )
    c4.metric(
        "Statistical Significance",
        "Yes" if p_value < (1 - conf) else "No",
        delta=f"p = {p_value:.4f}",
        delta_color="inverse",
    )

    # --- Charts ------------------------------------------------------------
    st.markdown("### Visualisations")
    st.plotly_chart(chart_original(ci, intervention_date, metric_label), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_pointwise(ci, intervention_date), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_cumulative(ci, intervention_date), use_container_width=True)

    # --- Summary & report --------------------------------------------------
    st.markdown("### Posterior Inference")
    tab_summary, tab_report = st.tabs(["Summary Table", "Detailed Report"])

    with tab_summary:
        summary_text = ci.summary()
        st.code(summary_text, language=None)

    with tab_report:
        report_text = ci.summary(output="report")
        # Split into paragraphs and render each clearly
        paragraphs = [p.strip() for p in report_text.strip().split("\n\n") if p.strip()]
        for para in paragraphs:
            # Preserve single newlines within a paragraph as line breaks
            formatted = para.replace("\n", "  \n")
            st.markdown(formatted)
            st.markdown("---")

    # --- Raw data ----------------------------------------------------------
    with st.expander("Raw Inference Data"):
        st.dataframe(ci.inferences, use_container_width=True)


if __name__ == "__main__":
    main()
