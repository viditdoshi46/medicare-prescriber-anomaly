"""Central paths and constants for the Medicare prescriber anomaly project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "prescribers.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_PROCESSED, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# Minimum claims for a prescriber to be scored (CMS suppresses <11; we also
# want stable peer statistics, so we drop very-low-volume prescribers).
MIN_CLAIMS = 50

# Features used for peer-relative anomaly scoring (per-provider, per-specialty)
ANOMALY_FEATURES = [
    "cost_per_claim",
    "cost_per_bene",
    "claims_per_bene",
    "avg_day_supply",
    "brand_cost_share",
    "opioid_claim_rate",
    "log_total_cost",
]
