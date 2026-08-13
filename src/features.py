"""
Step 2 - Engineer per-prescriber metrics used for peer-relative anomaly scoring.

The core idea: a prescriber isn't suspicious in absolute terms, but relative to
peers in the SAME specialty. So we compute normalized rate metrics that are
comparable across providers, then (in anomaly.py) z-score them within specialty.

Metrics:
  cost_per_claim    = Tot_Drug_Cst / Tot_Clms
  cost_per_bene     = Tot_Drug_Cst / Tot_Benes
  claims_per_bene   = Tot_Clms / Tot_Benes
  avg_day_supply    = Tot_Day_Suply / Tot_Clms
  brand_cost_share  = Brnd_Tot_Drug_Cst / Tot_Drug_Cst
  opioid_claim_rate = Opioid_Tot_Clms / Tot_Clms
  log_total_cost    = log1p(Tot_Drug_Cst)

Run:  python src/features.py  ->  data/processed/features.parquet/.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, ANOMALY_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cost_per_claim"] = df["Tot_Drug_Cst"] / df["Tot_Clms"]
    df["cost_per_bene"] = df["Tot_Drug_Cst"] / df["Tot_Benes"]
    df["claims_per_bene"] = df["Tot_Clms"] / df["Tot_Benes"]
    df["avg_day_supply"] = df["Tot_Day_Suply"] / df["Tot_Clms"]
    df["brand_cost_share"] = (df.get("Brnd_Tot_Drug_Cst", 0)
                              / df["Tot_Drug_Cst"]).clip(0, 1)
    if "Opioid_Tot_Clms" in df.columns:
        df["opioid_claim_rate"] = (df["Opioid_Tot_Clms"] / df["Tot_Clms"]).clip(0, 1)
    elif "Opioid_Prscrbr_Rate" in df.columns:
        df["opioid_claim_rate"] = (df["Opioid_Prscrbr_Rate"] / 100).clip(0, 1)
    else:
        df["opioid_claim_rate"] = 0.0
    df["log_total_cost"] = np.log1p(df["Tot_Drug_Cst"])

    df[ANOMALY_FEATURES] = df[ANOMALY_FEATURES].replace(
        [np.inf, -np.inf], np.nan)
    df = df.dropna(subset=ANOMALY_FEATURES).reset_index(drop=True)
    return df


def main() -> None:
    src = DATA_PROCESSED / "clean.parquet"
    if not src.exists():
        src = DATA_PROCESSED / "clean.csv"
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    out = add_features(df)
    try:
        out.to_parquet(DATA_PROCESSED / "features.parquet", index=False)
    except Exception:
        pass
    out.to_csv(DATA_PROCESSED / "features.csv", index=False)
    print(f"[features] {out.shape} | features: {ANOMALY_FEATURES}")


if __name__ == "__main__":
    main()
