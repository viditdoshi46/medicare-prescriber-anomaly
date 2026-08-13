"""
Step 3 - Peer-relative anomaly detection.

Two complementary signals:
  1. Robust peer z-scores: for each metric, z-score a prescriber against peers
     in the SAME specialty using median / MAD (robust to the very outliers we're
     hunting). This is fully interpretable — it tells you *which* metric is off.
  2. IsolationForest over the peer-standardized metrics: an unsupervised model
     that flags multivariate outliers a single-metric rule would miss.

We rank prescribers by the IsolationForest score, attach the top driving metric
from the peer z-scores (the "why"), and — because the synthetic data ships a
ground-truth label — report precision/recall so the method is validated.

Run:  python src/anomaly.py
Outputs: data/processed/scored.parquet/.csv, reports/metrics.json,
         reports/figures/*.png
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from config import DATA_PROCESSED, REPORTS, FIGURES, ANOMALY_FEATURES

CONTAMINATION = 0.02   # flag ~top 2% as anomalous
EPS = 1e-9
DRIVER_LABELS = {
    "cost_per_claim": "High cost per claim",
    "cost_per_bene": "High cost per beneficiary",
    "claims_per_bene": "High claims per beneficiary",
    "avg_day_supply": "Unusual days supply",
    "brand_cost_share": "High brand-name cost share",
    "opioid_claim_rate": "High opioid claim rate",
    "log_total_cost": "Unusually high total cost",
}


def peer_robust_z(df: pd.DataFrame) -> pd.DataFrame:
    """Robust z-score of each feature within its specialty peer group."""
    z = pd.DataFrame(index=df.index)
    for f in ANOMALY_FEATURES:
        grp = df.groupby("Prscrbr_Type")[f]
        med = grp.transform("median")
        mad = grp.transform(lambda s: (s - s.median()).abs().median())
        z[f] = 0.6745 * (df[f] - med) / (mad + EPS)
    return z


def main() -> None:
    src = DATA_PROCESSED / "features.parquet"
    if not src.exists():
        src = DATA_PROCESSED / "features.csv"
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)

    z = peer_robust_z(df)
    Xz = z[ANOMALY_FEATURES].clip(-15, 15).fillna(0).values

    iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION,
                          random_state=42)
    iso.fit(Xz)
    # higher = more anomalous
    df["anomaly_score"] = -iso.decision_function(Xz)
    df["is_flagged"] = (iso.predict(Xz) == -1).astype(int)

    # the "why": metric with the largest positive peer z-score
    pos_z = z[ANOMALY_FEATURES].clip(lower=0)
    df["top_driver"] = pos_z.idxmax(axis=1).map(DRIVER_LABELS)
    df["top_driver_z"] = pos_z.max(axis=1).round(2)
    for f in ANOMALY_FEATURES:
        df[f"z_{f}"] = z[f].round(2)

    df = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # ---- validation against ground truth (synthetic only) --------------
    metrics = {
        "n_prescribers": int(len(df)),
        "n_flagged": int(df["is_flagged"].sum()),
        "flagged_pct": round(float(df["is_flagged"].mean()), 4),
        "total_drug_cost": float(df["Tot_Drug_Cst"].sum()),
        "flagged_drug_cost": float(df.loc[df["is_flagged"] == 1, "Tot_Drug_Cst"].sum()),
    }
    metrics["flagged_cost_share"] = round(
        metrics["flagged_drug_cost"] / metrics["total_drug_cost"], 4)

    if "_is_injected_anomaly" in df.columns:
        y = df["_is_injected_anomaly"].values
        k = int(y.sum())
        topk = df.head(k)
        metrics["ground_truth_anomalies"] = int(k)
        metrics["precision_at_k"] = round(float(topk["_is_injected_anomaly"].mean()), 4)
        flagged = df[df["is_flagged"] == 1]
        metrics["recall_in_flagged"] = round(
            float(flagged["_is_injected_anomaly"].sum() / max(k, 1)), 4)

    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    keep = (["rank", "Prscrbr_NPI", "Prscrbr_Last_Org_Name", "Prscrbr_First_Name",
             "Prscrbr_State_Abrvtn", "Prscrbr_Type", "Tot_Clms", "Tot_Benes",
             "Tot_Drug_Cst", "anomaly_score", "is_flagged", "top_driver",
             "top_driver_z"] + ANOMALY_FEATURES
            + [f"z_{f}" for f in ANOMALY_FEATURES]
            + (["_is_injected_anomaly"] if "_is_injected_anomaly" in df.columns else []))
    scored = df[[c for c in keep if c in df.columns]]
    try:
        scored.to_parquet(DATA_PROCESSED / "scored.parquet", index=False)
    except Exception:
        pass
    scored.to_csv(DATA_PROCESSED / "scored.csv", index=False)

    _figures(df)
    print("[anomaly] saved scored table, metrics, figures")


def _figures(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(df["anomaly_score"], bins=60, color="#2c7fb8")
    thr = df.loc[df["is_flagged"] == 1, "anomaly_score"].min()
    plt.axvline(thr, color="#c0392b", ls="--", label="flag threshold")
    plt.xlabel("Anomaly score"); plt.ylabel("Prescribers")
    plt.title("Anomaly score distribution"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "score_distribution.png", dpi=120)
    plt.close()

    d = (df[df["is_flagged"] == 1]["top_driver"].value_counts())
    plt.figure(figsize=(6, 4))
    plt.barh(d.index[::-1], d.values[::-1], color="#c0392b")
    plt.xlabel("Flagged prescribers"); plt.title("Why prescribers were flagged")
    plt.tight_layout(); plt.savefig(FIGURES / "flag_reasons.png", dpi=120)
    plt.close()

    by_spec = (df.groupby("Prscrbr_Type")["is_flagged"].mean()
               .sort_values() * 100)
    plt.figure(figsize=(6, 5))
    plt.barh(by_spec.index, by_spec.values, color="#2c7fb8")
    plt.xlabel("% of prescribers flagged"); plt.title("Flag rate by specialty")
    plt.tight_layout(); plt.savefig(FIGURES / "flag_rate_by_specialty.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
