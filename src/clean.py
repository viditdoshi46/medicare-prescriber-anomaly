"""
Step 1 - Clean the raw prescriber summary.

Decisions (documented for the README):
  * Coerce money/count columns to numeric (real CMS files ship them as strings
    with '$' or commas in some vintages).
  * Drop prescribers with < MIN_CLAIMS total claims — CMS suppresses <11, and
    tiny volumes produce unstable per-claim ratios that masquerade as outliers.
  * Drop rows missing the fields needed to compute peer metrics.

Run:  python src/clean.py   ->  data/processed/clean.parquet/.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import DATA_RAW, DATA_PROCESSED, MIN_CLAIMS

NUMERIC = ["Tot_Clms", "Tot_30day_Fills", "Tot_Drug_Cst", "Tot_Benes",
           "Tot_Day_Suply", "Brnd_Tot_Clms", "Brnd_Tot_Drug_Cst",
           "Gnrc_Tot_Clms", "Gnrc_Tot_Drug_Cst", "Opioid_Tot_Clms",
           "Opioid_Tot_Drug_Cst", "Opioid_Prscrbr_Rate", "Antbtc_Tot_Clms",
           "Bene_Avg_Age", "Bene_Avg_Risk_Scre"]


def load_raw() -> pd.DataFrame:
    if not DATA_RAW.exists():
        raise SystemExit(
            f"Raw data not found at {DATA_RAW}.\n"
            "Run `python src/download_data.py` (real) or "
            "`python src/make_synthetic.py` (offline demo) first.")
    return pd.read_csv(DATA_RAW, dtype=str)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    for c in NUMERIC:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                     .str.replace(r"[$,]", "", regex=True)
                     .replace({"": np.nan, "nan": np.nan}))
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep the ground-truth label if the synthetic generator added it
    if "_is_injected_anomaly" in df.columns:
        df["_is_injected_anomaly"] = pd.to_numeric(
            df["_is_injected_anomaly"], errors="coerce").fillna(0).astype(int)

    df["Prscrbr_Type"] = df.get("Prscrbr_Type").fillna("Unknown")

    before = len(df)
    df = df[df["Tot_Clms"].notna() & (df["Tot_Clms"] >= MIN_CLAIMS)]
    df = df[df["Tot_Benes"].notna() & (df["Tot_Benes"] > 0)]
    df = df[df["Tot_Drug_Cst"].notna() & (df["Tot_Drug_Cst"] > 0)]
    dropped = before - len(df)

    df = df.reset_index(drop=True)
    print(f"[clean] raw={n0:,} | dropped low-volume/invalid={dropped:,} | "
          f"final={len(df):,}")
    return df


def main() -> None:
    df = clean(load_raw())
    try:
        df.to_parquet(DATA_PROCESSED / "clean.parquet", index=False)
    except Exception as exc:
        print(f"[clean] parquet skipped ({exc})")
    df.to_csv(DATA_PROCESSED / "clean.csv", index=False)
    print(f"[clean] wrote {DATA_PROCESSED/'clean.csv'}")


if __name__ == "__main__":
    main()
