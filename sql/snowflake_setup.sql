-- ============================================================
-- Snowflake setup + load + peer-outlier analysis at scale.
-- Pipeline:  CMS CSV -> AWS S3 (external stage) -> Snowflake table.
-- The full CMS file is millions of rows, so this is where the workload
-- belongs in production. Replace <...> placeholders with your values.
-- ============================================================

CREATE WAREHOUSE IF NOT EXISTS ANALYTICS_WH
    WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;
CREATE DATABASE IF NOT EXISTS MEDICARE;
CREATE SCHEMA  IF NOT EXISTS MEDICARE.PART_D;
USE WAREHOUSE ANALYTICS_WH;
USE SCHEMA MEDICARE.PART_D;

-- 1) Landing table (CMS 'by Provider' summary, subset of columns)
CREATE OR REPLACE TABLE PRESCRIBERS (
    Prscrbr_NPI            NUMBER,
    Prscrbr_Last_Org_Name  STRING,
    Prscrbr_First_Name     STRING,
    Prscrbr_City           STRING,
    Prscrbr_State_Abrvtn   STRING,
    Prscrbr_Type           STRING,
    Tot_Clms               NUMBER,
    Tot_30day_Fills        FLOAT,
    Tot_Drug_Cst           FLOAT,
    Tot_Benes              NUMBER,
    Tot_Day_Suply          NUMBER,
    Brnd_Tot_Clms          NUMBER,
    Brnd_Tot_Drug_Cst      FLOAT,
    Gnrc_Tot_Clms          NUMBER,
    Gnrc_Tot_Drug_Cst      FLOAT,
    Opioid_Tot_Clms        NUMBER,
    Opioid_Tot_Drug_Cst    FLOAT,
    Opioid_Prscrbr_Rate    FLOAT,
    Antbtc_Tot_Clms        NUMBER,
    Bene_Avg_Age           FLOAT,
    Bene_Avg_Risk_Scre     FLOAT
);

-- 2) Load from an S3 external stage (AWS layer)
CREATE STORAGE INTEGRATION IF NOT EXISTS S3_INT
    TYPE = EXTERNAL_STAGE STORAGE_PROVIDER = 'S3' ENABLED = TRUE
    STORAGE_AWS_ROLE_ARN = '<your-iam-role-arn>'
    STORAGE_ALLOWED_LOCATIONS = ('s3://<your-bucket>/partd/');

CREATE OR REPLACE FILE FORMAT CSV_FF
    TYPE = CSV FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1 NULL_IF = ('');

CREATE OR REPLACE STAGE PARTD_STAGE
    STORAGE_INTEGRATION = S3_INT
    URL = 's3://<your-bucket>/partd/'
    FILE_FORMAT = CSV_FF;

COPY INTO PRESCRIBERS
FROM @PARTD_STAGE/prescribers.csv
FILE_FORMAT = (FORMAT_NAME = CSV_FF)
MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
ON_ERROR = 'CONTINUE';

-- 3) Peer-relative outliers in pure SQL (window functions over specialty).
--    This complements the Python IsolationForest: analysts can run it directly
--    in the warehouse with no ML stack. Flags prescribers whose cost-per-claim
--    is > 3 robust-SDs above their specialty peers.
CREATE OR REPLACE VIEW V_PEER_OUTLIERS AS
WITH base AS (
    SELECT *,
           Tot_Drug_Cst / NULLIF(Tot_Clms,0)  AS cost_per_claim,
           Tot_Drug_Cst / NULLIF(Tot_Benes,0) AS cost_per_bene,
           Opioid_Tot_Clms / NULLIF(Tot_Clms,0) AS opioid_rate
    FROM PRESCRIBERS
    WHERE Tot_Clms >= 50
), stats AS (
    SELECT *,
        MEDIAN(cost_per_claim) OVER (PARTITION BY Prscrbr_Type) AS med_cpc,
        MEDIAN(ABS(cost_per_claim - MEDIAN(cost_per_claim)
               OVER (PARTITION BY Prscrbr_Type)))
               OVER (PARTITION BY Prscrbr_Type) AS mad_cpc
    FROM base
)
SELECT Prscrbr_NPI, Prscrbr_Last_Org_Name, Prscrbr_State_Abrvtn, Prscrbr_Type,
       Tot_Clms, Tot_Drug_Cst, cost_per_claim, opioid_rate,
       0.6745 * (cost_per_claim - med_cpc) / NULLIF(mad_cpc,0) AS cpc_peer_z
FROM stats
WHERE 0.6745 * (cost_per_claim - med_cpc) / NULLIF(mad_cpc,0) > 3
ORDER BY cpc_peer_z DESC;

-- 4) Spend concentration by specialty
SELECT Prscrbr_Type,
       COUNT(*) AS prescribers,
       ROUND(SUM(Tot_Drug_Cst),0) AS total_cost
FROM PRESCRIBERS GROUP BY Prscrbr_Type ORDER BY total_cost DESC;
