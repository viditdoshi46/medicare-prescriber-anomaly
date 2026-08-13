"""
Medicare Part D Prescriber Anomaly Detection - interactive dashboard.

Run:  streamlit run app/streamlit_app.py

Tabs:
  1. Overview        - KPIs, spend concentration, flag reasons, by-specialty
  2. Flagged list    - filterable, ranked table of prescribers to review
  3. Investigate     - drill into one prescriber's peer z-scores (the "why")
"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Rx Anomaly Detection | Vidit Doshi",
                   layout="wide", page_icon="💊")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1250px;}
  .hero {background: linear-gradient(120deg, #3d2b8c 0%, #7b3fb0 100%);
         color:#fff; padding:26px 30px; border-radius:14px; margin-bottom:8px;}
  .hero h1 {color:#fff; margin:0; font-size:1.85rem; font-weight:700;}
  .hero p {color:#e7defb; margin:6px 0 0 0; font-size:1.02rem;}
  .hero .by {color:#c8b6ec; font-size:0.9rem; margin-top:10px;}
  div[data-testid="stMetric"] {background:#f7f5fc; border:1px solid #e6e0f2;
      border-radius:12px; padding:14px 16px;}
  div[data-testid="stMetricValue"] {color:#3d2b8c !important; font-weight:700;}
  div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {
      color:#334155 !important; font-weight:600;}
  .stTabs [data-baseweb="tab"] {font-weight:600;}
</style>
""", unsafe_allow_html=True)

HOVER = dict(bgcolor="white", bordercolor="#ded3f2",
             font=dict(color="#3d2b8c", size=13))

# Plain-English labels for the peer-metric z-scores (Investigate tab)
METRIC_LABELS = {
    "cost_per_claim": "Cost per claim",
    "cost_per_bene": "Cost per beneficiary",
    "claims_per_bene": "Claims per beneficiary",
    "avg_day_supply": "Avg days supply",
    "brand_cost_share": "Brand-name cost share",
    "opioid_claim_rate": "Opioid claim rate",
    "log_total_cost": "Total drug cost (log)",
}

ACCENT = "#d98324"   # amber accent, distinct from Project 1's red


@st.cache_resource(show_spinner="First load: building the dataset & scores (~30s)…")
def ensure_pipeline():
    """On a fresh deploy the processed files may be absent; build them once."""
    if (PROC / "scored.csv").exists() or (PROC / "scored.parquet").exists():
        return True
    src = ROOT / "src"
    steps = ["make_synthetic.py", "clean.py", "features.py", "anomaly.py"]
    # prefer real data if available
    if (ROOT / "data" / "raw" / "prescribers.csv").exists():
        steps = ["clean.py", "features.py", "anomaly.py"]
    for s in steps:
        subprocess.run([sys.executable, str(src / s)], check=True)
    return True


@st.cache_data
def load():
    p = PROC / "scored.parquet"
    if not p.exists():
        p = PROC / "scored.csv"
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    metrics = json.loads((REPORTS / "metrics.json").read_text()) \
        if (REPORTS / "metrics.json").exists() else {}
    return df, metrics


try:
    ensure_pipeline()
    df, metrics = load()
except Exception as exc:
    st.error(f"Could not load data. Run `python run_all.py` locally.\n\n{exc}")
    st.stop()

st.markdown("""
<div class="hero">
  <h1>💊 Medicare Part D Prescriber Anomaly Detection</h1>
  <p>Flagging the prescribers whose billing patterns look most unusual versus
     their own specialty peers — so a program-integrity team can review the
     highest-risk, highest-dollar providers first.</p>
  <div class="by">Built by Vidit Doshi · SQL · Python · scikit-learn · Streamlit · AWS/Snowflake-ready</div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️  About this project — the business problem & approach", expanded=True):
    st.markdown("""
**The problem.** Medicare Part D pays out tens of billions in drug spend across
~1M prescribers. Auditors can only review a handful, so the question is:
**which prescribers should they look at first?**

**The approach.** Aggregate each prescriber's billing into rate metrics
(cost per claim, cost per beneficiary, opioid rate, brand share…), then score
how far each sits from **peers in the same specialty** using robust z-scores
**and** an IsolationForest. A cardiologist and a family-practice NP shouldn't be
compared directly — peer-relative scoring is the whole point.

**The payoff.** The flagged top 2% concentrate a disproportionate share of
spend, and every flag comes with a plain-English reason ("high cost per
beneficiary") an auditor can act on.
""")

flagged = df[df["is_flagged"] == 1].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prescribers analyzed", f"{len(df):,}")
c2.metric("Flagged for review", f"{len(flagged):,}",
          help=f"Top {metrics.get('flagged_pct', 0.02)*100:.0f}% by anomaly score")
c3.metric("Share of $ in flagged",
          f"{metrics.get('flagged_cost_share', 0)*100:.1f}%",
          help="Portion of total drug cost attributable to flagged prescribers")
if "precision_at_k" in metrics:
    c4.metric("Precision@k (validation)",
              f"{metrics['precision_at_k']*100:.0f}%",
              help="On synthetic data with known anomalies, share of the top-k "
                   "that are true injected anomalies.")
else:
    c4.metric("Total drug cost", f"${df['Tot_Drug_Cst'].sum()/1e6:,.0f}M")

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🚩 Flagged prescribers", "🔎 Investigate"])

# ---------------- Overview ----------------
with tab1:
    left, right = st.columns(2)
    with left:
        d = flagged["top_driver"].value_counts().sort_values()
        fig = px.bar(x=d.values, y=d.index, orientation="h",
                     labels={"x": "Flagged prescribers", "y": ""},
                     title="Why prescribers were flagged", color_discrete_sequence=[ACCENT])
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x} flagged<extra></extra>")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                          hoverlabel=HOVER)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        s = (df.groupby("Prscrbr_Type")["is_flagged"].mean() * 100).sort_values()
        fig = px.bar(x=s.values, y=s.index, orientation="h",
                     labels={"x": "% flagged", "y": ""},
                     title="Flag rate by specialty",
                     color=s.values, color_continuous_scale="Purples")
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:.1f}% flagged<extra></extra>")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                          coloraxis_showscale=False, hoverlabel=HOVER)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Flagged spend by specialty** (where an audit team gets the most $ back)")
    by = (flagged.groupby("Prscrbr_Type")
          .agg(flagged=("is_flagged", "sum"),
               flagged_cost=("Tot_Drug_Cst", "sum")).reset_index()
          .sort_values("flagged_cost", ascending=False))
    by["flagged_cost"] = by["flagged_cost"].apply(lambda v: f"${v:,.0f}")
    st.dataframe(by.rename(columns={"Prscrbr_Type": "Specialty",
                                    "flagged": "Flagged",
                                    "flagged_cost": "Flagged drug cost"}),
                 hide_index=True, use_container_width=True)

# ---------------- Flagged list ----------------
with tab2:
    a, b, c = st.columns(3)
    specs = ["All"] + sorted(df["Prscrbr_Type"].unique())
    states = ["All"] + sorted(df["Prscrbr_State_Abrvtn"].unique())
    drivers = ["All"] + sorted(flagged["top_driver"].dropna().unique())
    sspec = a.selectbox("Specialty", specs)
    sstate = b.selectbox("State", states)
    sdriver = c.selectbox("Flag reason", drivers)

    view = flagged.copy()
    if sspec != "All":
        view = view[view["Prscrbr_Type"] == sspec]
    if sstate != "All":
        view = view[view["Prscrbr_State_Abrvtn"] == sstate]
    if sdriver != "All":
        view = view[view["top_driver"] == sdriver]

    view = view.sort_values("anomaly_score", ascending=False)
    show = view[["rank", "Prscrbr_NPI", "Prscrbr_Last_Org_Name",
                 "Prscrbr_State_Abrvtn", "Prscrbr_Type", "Tot_Clms",
                 "Tot_Drug_Cst", "cost_per_claim", "top_driver",
                 "anomaly_score"]].copy()
    show["Tot_Drug_Cst"] = show["Tot_Drug_Cst"].apply(lambda v: f"${v:,.0f}")
    show["cost_per_claim"] = show["cost_per_claim"].apply(lambda v: f"${v:,.0f}")
    show["anomaly_score"] = show["anomaly_score"].round(3)
    st.caption(f"{len(show):,} flagged prescribers match the filters")
    st.dataframe(show.rename(columns={
        "Prscrbr_NPI": "NPI", "Prscrbr_Last_Org_Name": "Name",
        "Prscrbr_State_Abrvtn": "State", "Prscrbr_Type": "Specialty",
        "Tot_Clms": "Claims", "Tot_Drug_Cst": "Drug cost ($)",
        "cost_per_claim": "Cost/claim ($)", "top_driver": "Flag reason",
        "anomaly_score": "Score"}), hide_index=True, use_container_width=True)

# ---------------- Investigate ----------------
with tab3:
    st.write("Pick a flagged prescriber to see how far each metric sits from "
             "their specialty peers (robust z-score; >3 is unusual).")
    opts = (flagged.sort_values("anomaly_score", ascending=False)
            .head(200))
    label = opts.apply(
        lambda r: f"#{int(r['rank'])} · {r['Prscrbr_Type']} · {r['Prscrbr_State_Abrvtn']} "
                  f"· NPI {int(r['Prscrbr_NPI'])}", axis=1)
    pick = st.selectbox("Prescriber", label.tolist())
    row = opts.iloc[label.tolist().index(pick)]

    zcols = [c for c in df.columns if c.startswith("z_")]
    zvals = {c.replace("z_", ""): float(row[c]) for c in zcols}
    zdf = (pd.DataFrame({"metric": [METRIC_LABELS.get(k, k) for k in zvals],
                         "peer_z": list(zvals.values())})
           .sort_values("peer_z"))
    fig = px.bar(zdf, x="peer_z", y="metric", orientation="h",
                 labels={"peer_z": "Peer z-score (SDs from specialty median)", "metric": ""},
                 color="peer_z", color_continuous_scale="PuOr_r",
                 range_color=[-6, 6])
    fig.add_vline(x=3, line_dash="dash", line_color=ACCENT)
    fig.add_vline(x=-3, line_dash="dash", line_color=ACCENT)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>z = %{x:.2f}<extra></extra>")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10),
                      coloraxis_showscale=False, hoverlabel=HOVER)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"**Primary reason:** {row['top_driver']} "
            f"(peer z = {row['top_driver_z']}). "
            f"Total drug cost ${row['Tot_Drug_Cst']:,.0f} across "
            f"{int(row['Tot_Clms']):,} claims.")

st.caption("Built by Vidit Doshi · Peer-relative anomaly detection · "
           "SQL · Python · scikit-learn · Streamlit · AWS/Snowflake-ready")
