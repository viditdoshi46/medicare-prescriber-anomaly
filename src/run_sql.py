"""
Run the DuckDB analysis queries against data/processed/scored.csv and print the
results. Proves the SQL layer works locally with zero cloud setup.

Run:  python src/run_sql.py
"""
from pathlib import Path
import duckdb
from config import DATA_PROCESSED

SCORED = DATA_PROCESSED / "scored.csv"

QUERIES = {
    "Spend concentration in the flagged 2%": """
        SELECT COUNT(*) AS prescribers, SUM(is_flagged) AS flagged,
               ROUND(100.0*SUM(is_flagged)/COUNT(*),2) AS flagged_pct,
               ROUND(100.0*SUM(CASE WHEN is_flagged=1 THEN Tot_Drug_Cst END)
                          /SUM(Tot_Drug_Cst),1) AS flagged_cost_share_pct
        FROM p""",
    "Top 10 prescribers to review": """
        SELECT rank, Prscrbr_State_Abrvtn AS st, Prscrbr_Type,
               Tot_Clms, ROUND(Tot_Drug_Cst,0) AS drug_cost,
               top_driver, ROUND(anomaly_score,3) AS score
        FROM p WHERE is_flagged=1 ORDER BY anomaly_score DESC LIMIT 10""",
    "Flag rate by specialty": """
        SELECT Prscrbr_Type, COUNT(*) AS n, SUM(is_flagged) AS flagged,
               ROUND(100.0*SUM(is_flagged)/COUNT(*),1) AS flag_rate_pct
        FROM p GROUP BY Prscrbr_Type ORDER BY flag_rate_pct DESC""",
    "Why prescribers get flagged": """
        SELECT top_driver, COUNT(*) AS flagged
        FROM p WHERE is_flagged=1 GROUP BY top_driver ORDER BY flagged DESC""",
}


def main() -> None:
    if not SCORED.exists():
        raise SystemExit("Run src/anomaly.py first to create scored.csv")
    con = duckdb.connect()
    con.execute(f"CREATE TABLE p AS SELECT * FROM read_csv_auto('{SCORED}', header=true)")
    for title, q in QUERIES.items():
        print(f"\n=== {title} ===")
        print(con.execute(q).df().to_string(index=False))


if __name__ == "__main__":
    main()
