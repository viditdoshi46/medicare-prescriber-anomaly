-- ============================================================
-- Medicare Part D prescriber analysis - SQL (DuckDB, runs locally)
-- Run:  duckdb < sql/duckdb_analysis.sql
-- or via src/run_sql.py which loads data/processed/scored.csv
-- These are the business questions the dashboard answers.
-- ============================================================

CREATE OR REPLACE TABLE presc AS
SELECT * FROM read_csv_auto('data/processed/scored.csv', header=true);

-- 1) Headline: how much spend is concentrated in the flagged 2%?
SELECT
    COUNT(*)                                    AS prescribers,
    SUM(is_flagged)                             AS flagged,
    ROUND(100.0 * SUM(is_flagged)/COUNT(*), 2)  AS flagged_pct,
    ROUND(SUM(Tot_Drug_Cst), 0)                 AS total_cost,
    ROUND(SUM(CASE WHEN is_flagged=1 THEN Tot_Drug_Cst END), 0) AS flagged_cost,
    ROUND(100.0 * SUM(CASE WHEN is_flagged=1 THEN Tot_Drug_Cst END)
                / SUM(Tot_Drug_Cst), 1)         AS flagged_cost_share_pct
FROM presc;

-- 2) Top 20 highest-priority prescribers to review
SELECT rank, Prscrbr_NPI, Prscrbr_Last_Org_Name, Prscrbr_State_Abrvtn,
       Prscrbr_Type, Tot_Clms, ROUND(Tot_Drug_Cst,0) AS drug_cost,
       ROUND(cost_per_claim,0) AS cost_per_claim,
       top_driver, ROUND(anomaly_score,3) AS score
FROM presc
WHERE is_flagged = 1
ORDER BY anomaly_score DESC
LIMIT 20;

-- 3) Flag rate and dollars by specialty (where to focus an audit team)
SELECT Prscrbr_Type,
       COUNT(*)                                     AS prescribers,
       SUM(is_flagged)                              AS flagged,
       ROUND(100.0*SUM(is_flagged)/COUNT(*), 1)     AS flag_rate_pct,
       ROUND(SUM(CASE WHEN is_flagged=1 THEN Tot_Drug_Cst END), 0) AS flagged_cost
FROM presc
GROUP BY Prscrbr_Type
ORDER BY flagged_cost DESC NULLS LAST;

-- 4) Why prescribers get flagged (dominant driver mix)
SELECT top_driver, COUNT(*) AS flagged,
       ROUND(SUM(Tot_Drug_Cst),0) AS drug_cost
FROM presc WHERE is_flagged = 1
GROUP BY top_driver ORDER BY flagged DESC;

-- 5) Geography: states with the most flagged spend
SELECT Prscrbr_State_Abrvtn AS state,
       SUM(is_flagged) AS flagged,
       ROUND(SUM(CASE WHEN is_flagged=1 THEN Tot_Drug_Cst END),0) AS flagged_cost
FROM presc GROUP BY state ORDER BY flagged_cost DESC NULLS LAST LIMIT 10;
