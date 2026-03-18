PRAGMA memory_limit='6GB';
PRAGMA threads=8;
PRAGMA temp_directory='./duckdb_spill_dir';
PRAGMA enable_progress_bar=true;

-- 1. Base Transactions
CREATE TEMP TABLE ResolvedTx AS
WITH RawTx AS (
    SELECT
        t.from_account,
        t.to_account,
        t.is_laundering,
        CAST(t.timestamp AS TIMESTAMP) AS ts,
        t.amount_sent_c * fx.rate AS amount_sent_usd
    FROM read_parquet('data/HI_Small/1_filtered_transactions.parquet') t
    LEFT JOIN read_parquet('data/fx_rates.parquet') fx
      ON t.payment_currency = fx.currency
     AND CAST(t.timestamp AS DATE) = fx.date
),
AccMap AS (
    SELECT "Account Number" AS acc_num, "Entity ID" AS entity_id
    FROM read_parquet('data/HI_Small/2_filtered_accounts.parquet')
)
SELECT
    t.ts,
    CAST(t.ts AS DATE) AS tx_date,
    src.entity_id AS source_entity,
    tgt.entity_id AS target_entity,
    t.amount_sent_usd AS adj_sent,
    CAST(t.is_laundering AS INTEGER) AS is_laundering
FROM RawTx t
JOIN AccMap src ON t.from_account = src.acc_num
JOIN AccMap tgt ON t.to_account = tgt.acc_num;

-- 2. Calendar of target dates
CREATE TEMP TABLE Calendar AS
SELECT unnest(generate_series(
    (SELECT MIN(tx_date) + INTERVAL 1 DAY FROM ResolvedTx),
    (SELECT MAX(tx_date) FROM ResolvedTx),
    INTERVAL 1 DAY
))::DATE AS window_date;

-- 3. LOOKBACK EDGES (Strictly T-7 to T-1) -> For NetworkX Topology
COPY (
    SELECT
        strftime(c.window_date, '%Y-%m-%d') AS window_date,
        r.source_entity AS source,
        r.target_entity AS target, 
        SUM(r.adj_sent) AS volume,
        COUNT(*) AS count,
        COALESCE(STDDEV_SAMP(r.adj_sent), 0.0) AS amount_std,
        COALESCE(STDDEV_SAMP(EXTRACT(EPOCH FROM r.ts)), 0.0) AS time_variance,
        SUM(r.adj_sent) * LOG2(1 + COUNT(*)) * (1 + 1.0 / (1.0 + (COALESCE(STDDEV_SAMP(r.adj_sent), 0.0) / ((SUM(r.adj_sent) / COUNT(*)) + 1e-9)))) AS weight
    FROM Calendar c
    JOIN ResolvedTx r 
      ON r.tx_date >= (c.window_date - INTERVAL 7 DAY)
     AND r.tx_date < c.window_date -- STRICTLY BEFORE TARGET DAY
    GROUP BY c.window_date, r.source_entity, r.target_entity
) TO 'data/HI_Small/lookback_edges.parquet' (FORMAT PARQUET);

-- 4. TARGET NODES (Strictly T) -> For Labels and Prediction Targets
COPY (
    WITH NodeSent AS (
        SELECT tx_date AS window_date, source_entity AS entity_id, SUM(adj_sent) AS vol_sent, COUNT(*) AS tx_count_sent, COALESCE(STDDEV_SAMP(EXTRACT(EPOCH FROM ts)), 0.0) AS time_variance
        FROM ResolvedTx GROUP BY tx_date, source_entity
    ),
    NodeRecv AS (
        SELECT tx_date AS window_date, target_entity AS entity_id, SUM(adj_sent) AS vol_recv, COUNT(*) AS tx_count_recv
        FROM ResolvedTx GROUP BY tx_date, target_entity
    ),
    EntityFraud AS (
        SELECT tx_date AS window_date, entity_id, MAX(is_fraud) AS is_fraud
        FROM (
            SELECT tx_date, source_entity AS entity_id, is_laundering AS is_fraud FROM ResolvedTx
            UNION ALL
            SELECT tx_date, target_entity AS entity_id, is_laundering AS is_fraud FROM ResolvedTx
        ) WHERE entity_id IS NOT NULL GROUP BY tx_date, entity_id
    )
    SELECT
        strftime(COALESCE(s.window_date, r.window_date), '%Y-%m-%d') AS window_date,
        COALESCE(s.entity_id, r.entity_id) AS entity_id,
        COALESCE(s.vol_sent, 0.0) AS vol_sent,
        COALESCE(r.vol_recv, 0.0) AS vol_recv,
        COALESCE(s.tx_count_sent, 0) + COALESCE(r.tx_count_recv, 0) AS tx_count,
        COALESCE(s.time_variance, 0.0) AS time_variance,
        COALESCE(ef.is_fraud, 0) AS is_fraud
    FROM NodeSent s
    FULL OUTER JOIN NodeRecv r ON s.window_date = r.window_date AND s.entity_id = r.entity_id
    LEFT JOIN EntityFraud ef ON COALESCE(s.window_date, r.window_date) = ef.window_date AND COALESCE(s.entity_id, r.entity_id) = ef.entity_id
) TO 'data/HI_Small/target_nodes.parquet' (FORMAT PARQUET);
