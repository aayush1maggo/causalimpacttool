# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate virtual environment (Windows)
venv\Scripts\activate

# Run the app
streamlit run app.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

This is a single-file Streamlit app (`app.py`) for measuring the causal impact of SEO interventions using Bayesian Structural Time Series (BSTS) via the `pycausalimpact` library (Python port of Google's CausalImpact R package).

### Data flow

1. `generate_sample_data()` — cached; produces a 365-day `pd.DataFrame` (indexed by date) with three target metrics (`search_clicks`, `organic_sessions`, `organic_conversions`) and two covariates (`branded_searches`, `industry_trend`). Each of four hardcoded interventions in `INTERVENTIONS` is baked into the simulated data sequentially, so effects compound.

2. `get_periods()` — derives the pre/post split for a chosen intervention by treating the next intervention's date as the end of the post period. Pre-period always starts at the first data point.

3. `run_analysis()` — calls `CausalImpact` with the target metric plus both covariates as a multi-column DataFrame. The covariates serve as the control series for counterfactual estimation.

4. Three Plotly chart builders (`chart_original`, `chart_pointwise`, `chart_cumulative`) read from `ci.inferences` and `ci.trained_model.filter_results.loglikelihood_burn` to trim the burn-in period before plotting.

5. Results are stored in `st.session_state` so the UI can re-render without re-running the model.

### Key design constraints

- The four interventions in `INTERVENTIONS` are ordered chronologically and their effects are **stacked** in the simulated data — each later intervention multiplies on top of all prior ones. Any change to intervention dates or order must be reflected consistently in both `INTERVENTIONS` and `generate_sample_data()`.
- `CausalImpact` expects a `pd.DataFrame` where the **first column is the response** and remaining columns are covariates — column order in `run_analysis()` matters.
- The `loglikelihood_burn` offset is applied manually before rendering charts; skipping it causes the inference indices to misalign with the observed data.
- GA4 and Search Console integrations are planned but not yet implemented; the app currently runs entirely on synthetic data.
