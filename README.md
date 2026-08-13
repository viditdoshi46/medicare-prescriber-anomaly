# Medicare Part D Prescriber Anomaly Detection

**Business question:** Medicare Part D pays out tens of billions in drug spend across roughly a million prescribers, and program-integrity teams can only review a tiny fraction. **Which prescribers should they investigate first — and why?**

**Headline finding:** Scoring each prescriber against **peers in their own specialty**, the flagged top **2%** concentrate **6.5% of total drug spend** — a strong return on a limited audit budget. On synthetic data with known planted anomalies, the method recovers **~70%** of them in the flagged set at **~63% precision** in the top-k, and **every flag ships with a plain-English reason** (e.g., "high cost per beneficiary") an auditor can act on.

---

## Why peer-relative scoring

An oncologist's cost-per-claim dwarfs a family-practice NP's — comparing them directly would flag every specialist and miss the real outliers. So the core idea is **compare each prescriber only to peers in the same specialty**. A prescriber is suspicious when they're extreme *relative to people who do the same job*.

## Method (two complementary signals)

1. **Robust peer z-scores.** For each rate metric, z-score a prescriber against their specialty using **median / MAD** (robust statistics resist the very outliers we're hunting — a mean/SD z-score gets dragged around by them). This is fully interpretable and tells you *which* metric is off.
2. **IsolationForest** over the peer-standardized metrics — an unsupervised model that catches **multivariate** outliers a single-metric rule would miss.

Prescribers are ranked by the IsolationForest score; the top peer z-score is attached as the human-readable "why."

**Metrics engineered per prescriber:** cost per claim, cost per beneficiary, claims per beneficiary, average days-supply, brand-name cost share, opioid claim rate, and (log) total cost.

---

## Results (`reports/metrics.json`)

| Metric | Value |
|---|---|
| Prescribers analyzed | 14,997 |
| Flagged for review (top 2%) | 300 |
| **Share of total drug $ in flagged** | **6.5%** |
| Precision@k (vs. planted anomalies) | 63% |
| Recall of planted anomalies in flagged set | 70% |

Figures in `reports/figures/`: anomaly-score distribution, flag-reason mix, flag rate by specialty.

## Recommendation (how a program-integrity team uses this)

1. **Work the ranked list top-down** — the flagged 2% give outsized dollar coverage per review.
2. **Lead with the driver** — "high cost per beneficiary," "high opioid claim rate," etc. focuses each review.
3. **Prioritize by flagged dollars per specialty** (Overview tab) to put audit hours where the recoverable spend is.

---

## Architecture (end-to-end, cloud-ready)

```
CMS Part D "by Provider" ─► data/raw ─► clean.py ─► features.py ─► anomaly.py ─► reports/ + scored table
        │                       │           │            │              │
 (download_data / S3 landing)  clean    per-provider   peer z-scores + IsolationForest
                                          metrics                         │
             SQL layer (DuckDB now / Snowflake-ready)        app/streamlit_app.py (dashboard)
```

- **Python** — cleaning, feature engineering, anomaly detection (scikit-learn).
- **SQL** — the business questions in `sql/duckdb_analysis.sql` (runs locally) and `sql/snowflake_setup.sql`, which loads via an **S3 external stage** and even reproduces the peer-outlier logic in pure SQL window functions for warehouse-native review.
- **Streamlit + Plotly** — dashboard: KPIs, flag-reason mix, a filterable ranked review list, and a per-prescriber drill-down showing peer z-scores.

**Why DuckDB for the demo, Snowflake for production:** the real CMS file is millions of rows (~2 GB) — the natural home is a warehouse. DuckDB runs the identical SQL locally so anyone can clone and execute instantly; `snowflake_setup.sql` shows the same workload lifting into Snowflake on an S3 landing zone.

---

## Dataset

[**CMS — Medicare Part D Prescribers, by Provider**](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers): one row per prescriber per year with total claims, beneficiaries, drug cost, brand/generic split, and opioid/antibiotic measures.

> **Reproducibility note:** the demo runs on a **schema-matched synthetic file** (`src/make_synthetic.py`) that mirrors the real CMS column names and per-specialty distributions and plants a known set of anomalies — which is what lets the pipeline *validate itself* (precision/recall above). The real file is millions of rows and can't be versioned on GitHub, so `src/download_data.py` pulls a sample from the CMS API when you run `python run_all.py --real` — no downstream code changes, the column names match.

---

## Run it

```bash
pip install -r requirements.txt

python run_all.py            # full pipeline on the offline demo data
python run_all.py --real     # download a real CMS sample first

python src/run_sql.py                    # the SQL business questions
streamlit run app/streamlit_app.py       # interactive dashboard
```

## Repo structure

```
medicare-prescriber-anomaly/
├── README.md
├── requirements.txt
├── run_all.py                  # one-command pipeline
├── src/
│   ├── config.py               # paths + constants
│   ├── download_data.py        # real CMS sample via API
│   ├── make_synthetic.py       # offline schema-matched demo data
│   ├── clean.py                # numeric coercion, low-volume drop
│   ├── features.py             # per-prescriber rate metrics
│   ├── anomaly.py              # peer z-scores + IsolationForest + validation
│   └── run_sql.py              # runs the DuckDB analysis
├── sql/
│   ├── duckdb_analysis.sql     # business questions, runs locally
│   └── snowflake_setup.sql     # S3 → Snowflake load + SQL peer-outlier view
├── app/
│   └── streamlit_app.py        # interactive dashboard
└── reports/
    ├── metrics.json
    └── figures/*.png
```

## Skills demonstrated

SQL (DuckDB + Snowflake window functions) · Python (pandas, scikit-learn) · unsupervised anomaly detection (robust peer z-scores + IsolationForest) · peer-benchmarking methodology · Streamlit/Plotly dashboarding · reproducible, cloud-ready pipeline design.

---
*Built by Vidit Doshi.*
