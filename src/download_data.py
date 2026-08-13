"""
Download a sample of the real CMS 'Medicare Part D Prescribers - by Provider'
summary dataset via the public data.cms.gov API. Run on a machine with open
internet; it writes data/raw/prescribers.csv with the same column names the
pipeline expects.

The full file is millions of rows (~2 GB). By default we pull a capped sample
(--rows) so it is practical to run and to version. Increase --rows for more.

Find the current distribution id:
  https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers
  -> "Medicare Part D Prescribers - by Provider" -> API -> copy the UUID.

Usage:
    python src/download_data.py                 # ~50k-row sample, latest year
    python src/download_data.py --rows 200000
    python src/download_data.py --distribution <uuid>
"""
from __future__ import annotations
import argparse
import json
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "prescribers.csv"

# A recent "by Provider" distribution id. If CMS rotates it, pass --distribution.
DEFAULT_DISTRIBUTION = "3d7d6f6f-6b0e-4a0e-8f2a-000000000000"  # placeholder
API = "https://data.cms.gov/data-api/v1/dataset/{dist}/data"

# Columns we keep (CMS names) — these match make_synthetic.py output.
KEEP = [
    "Prscrbr_NPI", "Prscrbr_Last_Org_Name", "Prscrbr_First_Name",
    "Prscrbr_City", "Prscrbr_State_Abrvtn", "Prscrbr_Type",
    "Tot_Clms", "Tot_30day_Fills", "Tot_Drug_Cst", "Tot_Benes",
    "Tot_Day_Suply", "Brnd_Tot_Clms", "Brnd_Tot_Drug_Cst",
    "Gnrc_Tot_Clms", "Gnrc_Tot_Drug_Cst", "Opioid_Tot_Clms",
    "Opioid_Tot_Drug_Cst", "Opioid_Prscrbr_Rate", "Antbtc_Tot_Clms",
    "Bene_Avg_Age", "Bene_Avg_Risk_Scre",
]


def fetch(dist: str, rows: int) -> "list[dict]":
    out, offset, page = [], 0, 5000
    while len(out) < rows:
        url = f"{API.format(dist=dist)}?size={page}&offset={offset}"
        with urllib.request.urlopen(url, timeout=120) as r:
            batch = json.loads(r.read().decode())
        if not batch:
            break
        out.extend(batch)
        offset += page
        print(f"  fetched {len(out):,} rows...")
    return out[:rows]


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--distribution", default=DEFAULT_DISTRIBUTION)
    args = ap.parse_args()

    if args.distribution == DEFAULT_DISTRIBUTION:
        print("NOTE: using a placeholder distribution id. If this fails, grab "
              "the current UUID from the CMS dataset page (see module docstring) "
              "and pass --distribution <uuid>.")
    print(f"Downloading up to {args.rows:,} rows from CMS...")
    rows = fetch(args.distribution, args.rows)
    if not rows:
        raise SystemExit("No rows returned. Check the --distribution id.")
    df = pd.DataFrame(rows)
    keep = [c for c in KEEP if c in df.columns]
    df = df[keep]
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {OUT}")


if __name__ == "__main__":
    main()
