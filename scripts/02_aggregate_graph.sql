PRAGMA memory_limit='6GB';
PRAGMA threads=8;
PRAGMA temp_directory='./duckdb_spill_dir';
PRAGMA enable_progress_bar=true;
PRAGMA enable_profiling='query_tree';

-- 1. Materialize the windowed transactions
 -- Weirdnodes - any transactions originally made in a currency other than the euro were converted to euros using the exchange rate applicable on the exact date of the transaction
 -- Anomaly Detection in Graphs of Bank Transactions for Anti Money Laundering Applications: the money amounts are converted to a single currency.
CREATE TEMP TABLE WindowedTx AS
WITH RawTx AS (
    SELECT
        t.from_account,
        t.to_account,
        t.is_laundering,
        CAST(t.timestamp AS TIMESTAMP) AS ts,
        t.amount_sent_c * fx.rate AS amount_sent_usd
    FROM read_parquet('data/LI_Large/1_filtered_transactions.parquet') t
    LEFT JOIN read_parquet('data/fx_rates.parquet') fx
      ON t.payment_currency = fx.currency
     AND CAST(t.timestamp AS DATE) = fx.date
),
AccMap AS (
    SELECT 
        "Account Number" AS acc_num, 
        "Entity ID" AS entity_id
    FROM read_parquet('data/LI_Large/2_filtered_accounts.parquet')
),
ResolvedTx AS (
    SELECT
        t.ts,
        src.entity_id AS source_entity,
        tgt.entity_id AS target_entity,
        t.amount_sent_usd AS adj_sent,
        CAST(t.is_laundering AS INTEGER) AS is_laundering
    FROM RawTx t
    JOIN AccMap src ON t.from_account = src.acc_num
    JOIN AccMap tgt ON t.to_account = tgt.acc_num
),
Calendar AS (
    SELECT unnest(generate_series(
        (SELECT CAST(MIN(ts) AS DATE) + INTERVAL 1 DAY FROM ResolvedTx),
        (SELECT CAST(MAX(ts) AS DATE) FROM ResolvedTx),
        INTERVAL 1 DAY
    ))::DATE AS window_date
)
SELECT 
    c.window_date, 
    r.source_entity,
    r.target_entity,
    r.adj_sent,
    r.is_laundering,
    r.ts
FROM Calendar c
JOIN ResolvedTx r 
  ON r.ts > (c.window_date - INTERVAL 7 DAY)
 AND r.ts <= c.window_date;

-- 2. Aggregate and Export Edges
COPY (
    SELECT
        strftime(window_date, '%Y-%m-%d') AS window_date,
        source_entity AS source, --Anomaly Detection in Graphs of Bank Transactions for Anti Money Laundering Applications
        target_entity AS target, 
        SUM(adj_sent) AS volume,
        COUNT(*) AS count,
        COALESCE(STDDEV_SAMP(adj_sent), 0.0) AS amount_std, -- OddBall
        COALESCE(STDDEV_SAMP(EXTRACT(EPOCH FROM ts)), 0.0) AS time_variance,
        SUM(adj_sent) * LOG2(1 + COUNT(*)) * (1 + 1.0 / (1.0 + (COALESCE(STDDEV_SAMP(adj_sent), 0.0) / ((SUM(adj_sent) / COUNT(*)) + 1e-9)))) AS weight --OddBall
    FROM WindowedTx
    GROUP BY window_date, source_entity, target_entity
) TO 'data/LI_Large/aggregated_edges.parquet' (FORMAT PARQUET);

-- 3. Aggregate and Export Nodes
COPY (
    WITH NodeSent AS ( --Anomaly Detection in Graphs of Bank Transactions for Anti Money Laundering Applications
        SELECT 
            window_date, 
            source_entity AS entity_id, 
            SUM(adj_sent) AS vol_sent, 
            COUNT(*) AS tx_count_sent,
            COALESCE(STDDEV_SAMP(EXTRACT(EPOCH FROM ts)), 0.0) AS time_variance
        FROM WindowedTx 
        GROUP BY window_date, source_entity
    ),
    NodeRecv AS (
        SELECT 
            window_date, 
            target_entity AS entity_id, 
            SUM(adj_sent) AS vol_recv, 
            COUNT(*) AS tx_count_recv
        FROM WindowedTx 
        GROUP BY window_date, target_entity
    ),
    EntityFraud AS (
        SELECT
            entity_id,
            MAX(is_fraud) AS is_fraud
        FROM (
            SELECT
                source_entity AS entity_id,
                CASE WHEN is_laundering = 1 THEN 1 ELSE 0 END AS is_fraud
            FROM WindowedTx
            UNION ALL
            SELECT
                target_entity AS entity_id,
                CASE WHEN is_laundering = 1 THEN 1 ELSE 0 END AS is_fraud
            FROM WindowedTx
        ) f
        WHERE entity_id IS NOT NULL
        GROUP BY entity_id
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
    FULL OUTER JOIN NodeRecv r
        ON s.window_date = r.window_date AND s.entity_id = r.entity_id
    LEFT JOIN EntityFraud ef
        ON COALESCE(s.entity_id, r.entity_id) = ef.entity_id
) TO 'data/LI_Large/aggregated_nodes.parquet' (FORMAT PARQUET);
