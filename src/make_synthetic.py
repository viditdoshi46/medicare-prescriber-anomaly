"""
Generate a synthetic stand-in for the CMS 'Medicare Part D Prescribers -
by Provider' summary file.

It reproduces the real column names and realistic per-specialty distributions,
then injects a small fraction of genuinely anomalous prescribers (extreme
cost-per-claim, extreme opioid rates, extreme cost-per-beneficiary) so the
anomaly-detection pipeline has real signal to find. This makes the repo run
fully offline. For the real data, run src/download_data.py (the column names
match, so no downstream code changes are needed).

Usage:
    python src/make_synthetic.py                # 15,000 prescribers
    python src/make_synthetic.py --rows 40000
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "prescribers.csv"

RNG = np.random.default_rng(7)

# specialty -> (baseline cost/claim mean, sd, opioid propensity, brand share)
SPECIALTIES = {
    "Family Practice":      (48, 18, 0.05, 0.20),
    "Internal Medicine":    (55, 22, 0.06, 0.22),
    "Nurse Practitioner":   (42, 16, 0.07, 0.18),
    "Cardiology":           (80, 30, 0.02, 0.30),
    "Psychiatry":           (70, 28, 0.03, 0.28),
    "Oncology":             (220, 90, 0.04, 0.55),
    "Rheumatology":         (260, 110, 0.03, 0.60),
    "Endocrinology":        (140, 60, 0.02, 0.45),
    "Pain Management":      (95, 40, 0.35, 0.25),
    "Physician Assistant":  (44, 17, 0.08, 0.18),
    "Dermatology":          (75, 30, 0.01, 0.35),
    "Neurology":            (110, 45, 0.05, 0.40),
}
STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
          "NJ", "VA", "WA", "AZ", "TN", "MA", "IN", "MO", "MD", "WI"]
LAST_NAMES = ["Smith", "Johnson", "Patel", "Nguyen", "Garcia", "Lee", "Brown",
              "Davis", "Martinez", "Wilson", "Kim", "Chen", "Lopez", "Khan",
              "Miller", "Shah", "Anderson", "Thomas", "Rao", "Cohen"]
FIRST_NAMES = ["James", "Mary", "Robert", "Priya", "David", "Linda", "John",
               "Wei", "Maria", "Ahmed", "Susan", "Michael", "Sofia", "Raj",
               "Emily", "Daniel", "Anna", "Carlos", "Grace", "Omar"]


def generate(n: int) -> pd.DataFrame:
    specs = list(SPECIALTIES)
    spec_p = np.array([0.16, 0.15, 0.16, 0.06, 0.06, 0.03, 0.02, 0.03,
                       0.05, 0.14, 0.03, 0.05])
    spec_p = spec_p / spec_p.sum()
    spec = RNG.choice(specs, size=n, p=spec_p)

    df = pd.DataFrame()
    df["Prscrbr_NPI"] = 1_000_000_000 + np.arange(n)
    df["Prscrbr_Last_Org_Name"] = RNG.choice(LAST_NAMES, n)
    df["Prscrbr_First_Name"] = RNG.choice(FIRST_NAMES, n)
    df["Prscrbr_City"] = "City_" + RNG.integers(1, 400, n).astype(str)
    df["Prscrbr_State_Abrvtn"] = RNG.choice(STATES, n)
    df["Prscrbr_Type"] = spec

    # total claims: lognormal, varies by specialty scale
    base_claims = RNG.lognormal(mean=6.4, sigma=0.7, size=n)
    df["Tot_Clms"] = np.clip(base_claims, 11, None).round().astype(int)
    df["Tot_Benes"] = np.clip(
        (df["Tot_Clms"] * RNG.uniform(0.18, 0.4, n)).round(), 11, None).astype(int)

    cpc_mean = np.array([SPECIALTIES[s][0] for s in spec])
    cpc_sd = np.array([SPECIALTIES[s][1] for s in spec])
    cost_per_claim = np.clip(RNG.normal(cpc_mean, cpc_sd), 5, None)

    df["Tot_Drug_Cst"] = (df["Tot_Clms"] * cost_per_claim).round(2)
    df["Tot_Day_Suply"] = (df["Tot_Clms"] * RNG.uniform(25, 45, n)).round().astype(int)
    df["Tot_30day_Fills"] = (df["Tot_Day_Suply"] / 30).round(1)

    brand_share = np.clip(
        RNG.normal([SPECIALTIES[s][3] for s in spec], 0.08), 0.01, 0.95)
    df["Brnd_Tot_Clms"] = (df["Tot_Clms"] * brand_share).round().astype(int)
    df["Gnrc_Tot_Clms"] = (df["Tot_Clms"] - df["Brnd_Tot_Clms"]).clip(lower=0)
    # brand drugs cost ~4x generic per claim
    gen_cpc = cost_per_claim * 0.5
    brand_cpc = cost_per_claim * 2.2
    df["Brnd_Tot_Drug_Cst"] = (df["Brnd_Tot_Clms"] * brand_cpc).round(2)
    df["Gnrc_Tot_Drug_Cst"] = (df["Gnrc_Tot_Clms"] * gen_cpc).round(2)

    opioid_rate = np.clip(
        RNG.normal([SPECIALTIES[s][2] for s in spec], 0.03), 0, 0.9)
    df["Opioid_Tot_Clms"] = (df["Tot_Clms"] * opioid_rate).round().astype(int)
    df["Opioid_Tot_Drug_Cst"] = (df["Opioid_Tot_Clms"]
                                 * RNG.uniform(20, 60, n)).round(2)
    df["Opioid_Prscrbr_Rate"] = (100 * df["Opioid_Tot_Clms"]
                                 / df["Tot_Clms"]).round(2)
    df["Antbtc_Tot_Clms"] = (df["Tot_Clms"]
                             * RNG.uniform(0.02, 0.12, n)).round().astype(int)
    df["Bene_Avg_Age"] = np.clip(RNG.normal(71, 6, n), 30, 95).round(1)
    df["Bene_Avg_Risk_Scre"] = np.clip(RNG.normal(1.2, 0.35, n), 0.3, 5).round(3)

    # ---- inject anomalies (~1.5%) --------------------------------------
    n_anom = int(n * 0.015)
    idx = RNG.choice(n, size=n_anom, replace=False)
    kinds = RNG.choice(["cost", "opioid", "bene"], size=n_anom, p=[0.45, 0.3, 0.25])
    for i, k in zip(idx, kinds):
        if k == "cost":                      # extreme cost per claim
            mult = RNG.uniform(4, 9)
            df.loc[i, "Tot_Drug_Cst"] = round(df.loc[i, "Tot_Drug_Cst"] * mult, 2)
            df.loc[i, "Brnd_Tot_Drug_Cst"] = round(
                df.loc[i, "Brnd_Tot_Drug_Cst"] * mult, 2)
        elif k == "opioid":                  # extreme opioid prescribing
            oc = int(df.loc[i, "Tot_Clms"] * RNG.uniform(0.75, 0.98))
            df.loc[i, "Opioid_Tot_Clms"] = oc
            df.loc[i, "Opioid_Prscrbr_Rate"] = round(
                100 * oc / df.loc[i, "Tot_Clms"], 2)
        else:                                # extreme cost per beneficiary
            df.loc[i, "Tot_Benes"] = max(11, int(df.loc[i, "Tot_Benes"] * 0.15))
    df["_is_injected_anomaly"] = 0
    df.loc[idx, "_is_injected_anomaly"] = 1   # ground-truth label for evaluation

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=15000)
    args = ap.parse_args()
    df = generate(args.rows)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} prescribers x {df.shape[1]} cols -> {OUT}")
    print(f"Injected anomalies: {int(df['_is_injected_anomaly'].sum()):,} "
          f"({df['_is_injected_anomaly'].mean():.1%})")


if __name__ == "__main__":
    main()
