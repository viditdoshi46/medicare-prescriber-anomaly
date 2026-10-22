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

GRID = ("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='40' height='40'%3E"
        "%3Cpath d='M40 0H0V40' fill='none' stroke='%23ffffff' stroke-opacity='.045'/%3E%3C/svg%3E")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  html, body, [class*="css"], .stMarkdown {font-family:'Space Grotesk', sans-serif;}
  .stApp {background: radial-gradient(1000px 500px at 85% -10%, rgba(139,92,246,.22), transparent 60%),
                      radial-gradient(800px 400px at 0% 0%, rgba(245,165,36,.10), transparent 55%), #0e0c1e;}
  .block-container {padding-top: 2.2rem; max-width: 1260px;}
  h1,h2,h3 {font-family:'Space Grotesk', sans-serif !important; letter-spacing:-.02em;}
  @keyframes rise {from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:none;}}
  @keyframes scan {0% {top:-4px; opacity:0;} 10% {opacity:.9;} 90% {opacity:.9;} 100% {top:100%; opacity:0;}}

  .hero {position:relative; overflow:hidden; padding:34px 38px 30px; border-radius:22px; color:#fff;
         background: linear-gradient(135deg, rgba(139,92,246,.35), rgba(245,165,36,.12) 70%), url("__GRID__"), #17133a;
         border:1px solid rgba(167,139,250,.35); box-shadow:0 24px 50px -28px rgba(139,92,246,.8); animation: rise .6s both;}
  .hero::after {content:""; position:absolute; left:0; right:0; height:3px; top:0; background:linear-gradient(90deg, transparent, #f5a524, transparent);
         animation: scan 5s linear infinite;}
  .hero .tag {display:inline-block; font-family:'JetBrains Mono', monospace; font-size:.74rem; letter-spacing:.12em; color:#f5a524;
         border:1px solid rgba(245,165,36,.5); padding:4px 10px; border-radius:6px; background:rgba(245,165,36,.08);}
  .hero h1 {color:#fff !important; margin:14px 0 10px; font-size:2.45rem; line-height:1.08; font-weight:700;}
  .hero p {color:#d9d1ff; margin:0; font-size:1.05rem; max-width:64ch; line-height:1.6;}
  .hero .chips {margin-top:18px; display:flex; flex-wrap:wrap; gap:8px;}
  .hero .chips span {font-family:'JetBrains Mono', monospace; font-size:.74rem; padding:5px 11px; border-radius:6px;
         background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.16); color:#efe9ff;}

  .kpi {background:linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); border:1px solid rgba(167,139,250,.28);
        border-radius:16px; padding:18px 20px; animation: rise .6s both; transition: transform .25s, border-color .25s; position:relative; overflow:hidden;}
  .kpi:hover {transform: translateY(-4px); border-color:#f5a524;}
  .kpi::after {content:""; position:absolute; right:-30px; top:-30px; width:90px; height:90px; border-radius:50%; background:var(--c,#8b5cf6); opacity:.18; filter:blur(8px);}
  .kpi .l {font-size:.74rem; letter-spacing:.14em; text-transform:uppercase; color:#a99ee0; font-weight:600;}
  .kpi .v {font-family:'JetBrains Mono', monospace; font-weight:600; font-size:2.05rem; color:#fff; margin:6px 0 2px;}
  .kpi .n {font-size:.8rem; color:#8c82c4;}

  .story {background:rgba(255,255,255,.04); border:1px solid rgba(167,139,250,.22); border-radius:16px; padding:18px 20px; height:100%;}
  .story .k {font-family:'JetBrains Mono', monospace; font-size:.7rem; letter-spacing:.14em; color:#f5a524; text-transform:uppercase;}
  .story h4 {margin:6px 0 8px; font-size:1.15rem; color:#fff;}
  .story p {margin:0; font-size:.9rem; line-height:1.6; color:#c4bbf0;}

  .sec {margin:24px 0 6px;} .sec h3 {margin:0; font-size:1.45rem; color:#fff;} .sec p {margin:2px 0 0; color:#9d93d6; font-size:.92rem;}

  .dossier {background:linear-gradient(160deg, rgba(139,92,246,.22), rgba(255,255,255,.03)); border:1px solid rgba(167,139,250,.4);
            border-radius:18px; padding:20px 22px; animation: rise .5s both;}
  .dossier .rank {font-family:'JetBrains Mono', monospace; color:#f5a524; font-size:.8rem; letter-spacing:.12em;}
  .dossier h4 {margin:6px 0 12px; font-size:1.4rem; color:#fff;}
  .dossier .row {display:flex; justify-content:space-between; gap:10px; padding:7px 0; border-top:1px solid rgba(255,255,255,.08); font-size:.9rem;}
  .dossier .row span:first-child {color:#a99ee0;} .dossier .row span:last-child {font-family:'JetBrains Mono', monospace; color:#fff; text-align:right;}
  .dossier .why {margin-top:12px; padding:11px 13px; border-radius:10px; background:rgba(245,165,36,.12); border:1px solid rgba(245,165,36,.4); color:#ffe2b0; font-size:.92rem;}

  .stTabs [data-baseweb="tab-list"] {gap:8px; border-bottom:none; margin-top:8px;}
  .stTabs [data-baseweb="tab"] {background:rgba(255,255,255,.05); border:1px solid rgba(167,139,250,.25); border-radius:10px; padding:8px 20px; font-weight:600; height:auto;}
  .stTabs [aria-selected="true"] {background:#f5a524; color:#17133a !important; border-color:#f5a524;}
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}
  .stPlotlyChart {background:rgba(255,255,255,.03); border:1px solid rgba(167,139,250,.2); border-radius:16px; padding:8px 8px 0;}
  .footer {text-align:center; color:#7e74b8; font-size:.82rem; margin:34px 0 6px; font-family:'JetBrains Mono', monospace;}
</style>
""".replace("__GRID__", GRID), unsafe_allow_html=True)


def kpi(col, label, value, note, color="#8b5cf6", delay=0):
    col.markdown(f'<div class="kpi" style="--c:{color}; animation-delay:{delay}ms">'
                 f'<div class="l">{label}</div><div class="v">{value}</div><div class="n">{note}</div></div>',
                 unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec"><h3>{title}</h3>' + (f'<p>{sub}</p>' if sub else "") + '</div>',
                unsafe_allow_html=True)


import plotly.graph_objects as go
import plotly.io as pio
pio.templates["audit"] = go.layout.Template(layout=dict(
    font=dict(family="Space Grotesk, sans-serif", size=13, color="#e6e0ff"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#f5a524", "#8b5cf6", "#22d3ee", "#f472b6"],
    xaxis=dict(gridcolor="rgba(167,139,250,.14)", zerolinecolor="rgba(167,139,250,.3)", linecolor="rgba(167,139,250,.25)"),
    yaxis=dict(gridcolor="rgba(167,139,250,.14)", zerolinecolor="rgba(167,139,250,.3)", linecolor="rgba(167,139,250,.25)"),
    title=dict(font=dict(size=16, color="#ffffff")),
))
pio.templates.default = "audit"

HOVER = dict(bgcolor="#1a1738", bordercolor="#8b5cf6", font=dict(color="#ffffff", size=13, family="Space Grotesk"))

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

ACCENT = "#f5a524"   # amber accent


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
  <span class="tag">PROGRAM INTEGRITY · CASE QUEUE</span>
  <h1>Medicare Part D Prescriber Anomaly Detection</h1>
  <p>Flag the prescribers whose billing looks most unusual compared with their own specialty peers,
     so an integrity team can review the highest-risk, highest-dollar providers first.</p>
  <div class="chips"><span>SQL</span><span>Python</span><span>scikit-learn</span><span>IsolationForest</span><span>Streamlit</span><span>Snowflake-ready</span></div>
</div>
""", unsafe_allow_html=True)

st.write("")
s1, s2, s3 = st.columns(3)
s1.markdown("""<div class="story"><div class="k">The problem</div><h4>A sliver can be audited</h4>
<p>Part D pays out tens of billions across about a million prescribers. Auditors can review a handful,
so the question is which prescribers to look at first.</p></div>""", unsafe_allow_html=True)
s2.markdown("""<div class="story"><div class="k">The approach</div><h4>Compare to true peers</h4>
<p>Each prescriber's billing becomes rate metrics, scored against their own specialty with robust
median/MAD z-scores plus an IsolationForest. A cardiologist is never compared with a family NP.</p></div>""", unsafe_allow_html=True)
s3.markdown("""<div class="story"><div class="k">The payoff</div><h4>Flags with reasons</h4>
<p>The flagged top 2% hold a disproportionate share of spend, and every flag carries a plain-English
reason, like high cost per beneficiary, that an auditor can act on.</p></div>""", unsafe_allow_html=True)

flagged = df[df["is_flagged"] == 1].copy()

st.write("")
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Prescribers analyzed", f"{len(df):,}", "one row per prescriber", "#8b5cf6", 0)
kpi(c2, "Flagged for review", f"{len(flagged):,}",
    f"top {metrics.get('flagged_pct', 0.02)*100:.0f}% by anomaly score", "#f5a524", 80)
kpi(c3, "Share of $ in flagged", f"{metrics.get('flagged_cost_share', 0)*100:.1f}%",
    "of total drug cost", "#22d3ee", 160)
if "precision_at_k" in metrics:
    kpi(c4, "Precision@k", f"{metrics['precision_at_k']*100:.0f}%",
        "on planted anomalies", "#f472b6", 240)
else:
    kpi(c4, "Total drug cost", f"${df['Tot_Drug_Cst'].sum()/1e6:,.0f}M", "across all prescribers", "#f472b6", 240)

st.write("")
tab1, tab2, tab3 = st.tabs(["📊  Overview", "🚩  Flagged prescribers", "🔎  Investigate"])

# ---------------- Overview ----------------
with tab1:
    section("Where the anomalies are", "Flag reasons on the left, flag rate by specialty on the right.")
    left, right = st.columns(2)
    with left:
        d = flagged["top_driver"].value_counts().sort_values()
        fig = px.bar(x=d.values, y=d.index, orientation="h",
                     labels={"x": "Flagged prescribers", "y": ""},
                     title="Why prescribers were flagged", color_discrete_sequence=[ACCENT])
        fig.update_traces(marker_cornerradius=6, hovertemplate="<b>%{y}</b><br>%{x} flagged<extra></extra>")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), hoverlabel=HOVER)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        sp = (df.groupby("Prscrbr_Type")["is_flagged"].mean() * 100).sort_values()
        fig = px.bar(x=sp.values, y=sp.index, orientation="h",
                     labels={"x": "% flagged", "y": ""},
                     title="Flag rate by specialty",
                     color=sp.values, color_continuous_scale=[[0, "#3b2f7a"], [1, "#a78bfa"]])
        fig.update_traces(marker_cornerradius=6, hovertemplate="<b>%{y}</b><br>%{x:.1f}% flagged<extra></extra>")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10),
                          coloraxis_showscale=False, hoverlabel=HOVER)
        st.plotly_chart(fig, use_container_width=True)

    section("Where the money is", "Every prescriber by cost per claim and total drug cost. Flagged prescribers glow amber.")
    pool = df.dropna(subset=["cost_per_claim", "Tot_Drug_Cst"])
    pool = pool[(pool["cost_per_claim"] > 0) & (pool["Tot_Drug_Cst"] > 0)]
    rest = pool[pool["is_flagged"] != 1]
    rest = rest.sample(min(len(rest), 3500), random_state=1)
    fl = pool[pool["is_flagged"] == 1]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=rest["cost_per_claim"], y=rest["Tot_Drug_Cst"], mode="markers", name="Other prescribers",
                               marker=dict(size=5, color="rgba(167,139,250,.35)"),
                               hovertemplate="Cost/claim $%{x:,.0f}<br>Total $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scattergl(x=fl["cost_per_claim"], y=fl["Tot_Drug_Cst"], mode="markers", name="Flagged",
                               marker=dict(size=8, color="#f5a524", line=dict(color="#fff", width=.6)),
                               hovertemplate="<b>Flagged</b><br>Cost/claim $%{x:,.0f}<br>Total $%{y:,.0f}<extra></extra>"))
    fig.update_xaxes(type="log", title="Cost per claim ($, log scale)")
    fig.update_yaxes(type="log", title="Total drug cost ($, log scale)")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), hoverlabel=HOVER,
                      legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(fig, use_container_width=True)

    section("Flagged spend by specialty", "Where an audit team gets the most dollars back.")
    by = (flagged.groupby("Prscrbr_Type")
          .agg(flagged=("is_flagged", "sum"),
               flagged_cost=("Tot_Drug_Cst", "sum")).reset_index()
          .sort_values("flagged_cost", ascending=False))
    st.dataframe(by.rename(columns={"Prscrbr_Type": "Specialty", "flagged": "Flagged",
                                    "flagged_cost": "Flagged drug cost"}),
                 hide_index=True, use_container_width=True,
                 column_config={"Flagged drug cost": st.column_config.ProgressColumn(
                     "Flagged drug cost", format="$%d", min_value=0,
                     max_value=float(by["flagged_cost"].max() * 1.05))})

# ---------------- Flagged list ----------------
with tab2:
    section("The review queue", "Filter by specialty, state, or reason. Highest anomaly score first.")
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
    show["anomaly_score"] = show["anomaly_score"].round(3)
    st.caption(f"{len(show):,} flagged prescribers match the filters")
    st.dataframe(show.rename(columns={
        "Prscrbr_NPI": "NPI", "Prscrbr_Last_Org_Name": "Name",
        "Prscrbr_State_Abrvtn": "State", "Prscrbr_Type": "Specialty",
        "Tot_Clms": "Claims", "Tot_Drug_Cst": "Drug cost ($)",
        "cost_per_claim": "Cost/claim ($)", "top_driver": "Flag reason",
        "anomaly_score": "Score"}), hide_index=True, use_container_width=True,
        column_config={"Drug cost ($)": st.column_config.NumberColumn(format="$%d"),
                       "Cost/claim ($)": st.column_config.NumberColumn(format="$%d"),
                       "Claims": st.column_config.NumberColumn(format="%d"),
                       "Score": st.column_config.ProgressColumn(
                           "Score", format="%.3f", min_value=0,
                           max_value=float(max(show["anomaly_score"].max(), 0.01)))})

# ---------------- Investigate ----------------
with tab3:
    section("Open a case", "Pick a flagged prescriber to see how far each metric sits from their specialty peers (robust z, above 3 is unusual).")
    opts = (flagged.sort_values("anomaly_score", ascending=False)
            .head(200))
    label = opts.apply(
        lambda r: f"#{int(r['rank'])} · {r['Prscrbr_Type']} · {r['Prscrbr_State_Abrvtn']} "
                  f"· NPI {int(r['Prscrbr_NPI'])}", axis=1)
    pick = st.selectbox("Prescriber", label.tolist(), label_visibility="collapsed")
    row = opts.iloc[label.tolist().index(pick)]

    zcols = [c for c in df.columns if c.startswith("z_")]
    zvals = {c.replace("z_", ""): float(row[c]) for c in zcols}
    zdf = (pd.DataFrame({"metric": [METRIC_LABELS.get(k, k) for k in zvals],
                         "peer_z": list(zvals.values())})
           .sort_values("peer_z"))
    zdf["state"] = zdf["peer_z"].apply(lambda z: "Unusual (|z| > 3)" if abs(z) > 3 else "Within normal range")
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        fig = px.bar(zdf, x="peer_z", y="metric", orientation="h", color="state",
                     color_discrete_map={"Unusual (|z| > 3)": "#f5a524", "Within normal range": "#5b4bb3"},
                     labels={"peer_z": "Peer z-score (SDs from specialty median)", "metric": ""})
        fig.add_vline(x=3, line_dash="dash", line_color="#f5a524")
        fig.add_vline(x=-3, line_dash="dash", line_color="#f5a524")
        fig.update_traces(marker_cornerradius=6, hovertemplate="<b>%{y}</b><br>z = %{x:.2f}<extra></extra>")
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10), hoverlabel=HOVER,
                          legend=dict(orientation="h", y=1.07, x=0, title_text=""))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown(
            f'''<div class="dossier"><div class="rank">CASE #{int(row['rank'])}</div>
<h4>{row['Prscrbr_Type']}</h4>
<div class="row"><span>State</span><span>{row['Prscrbr_State_Abrvtn']}</span></div>
<div class="row"><span>NPI</span><span>{int(row['Prscrbr_NPI'])}</span></div>
<div class="row"><span>Claims</span><span>{int(row['Tot_Clms']):,}</span></div>
<div class="row"><span>Total drug cost</span><span>${row['Tot_Drug_Cst']:,.0f}</span></div>
<div class="row"><span>Cost per claim</span><span>${row['cost_per_claim']:,.0f}</span></div>
<div class="why"><b>Primary reason:</b> {row['top_driver']} (peer z = {row['top_driver_z']})</div></div>''',
            unsafe_allow_html=True)

st.markdown('<div class="footer">Built by Vidit Doshi · Peer-relative anomaly detection · '
            'SQL · Python · scikit-learn · Streamlit</div>', unsafe_allow_html=True)
