PRAGMA memory_limit='4GB';
PRAGMA threads=8;
PRAGMA temp_directory='./duckdb_spill_dir';
PRAGMA enable_progress_bar=true;
PRAGMA enable_profiling='query_tree';

-- 1. Materialize the windowed transactions
CREATE TEMP TABLE WindowedTx AS
WITH RawTx AS (
    SELECT
        from_account,
        to_account,
        amount_sent_c,
        is_laundering,
        -- DuckDB natively parses Parquet logical types. No manual epoch math required.
        CAST(timestamp AS TIMESTAMP) AS ts
    FROM read_parquet([
        'data/HI_Large/1_filtered_normal_transactions.parquet',
        'data/HI_Large/2_filtered_laundering_transactions.parquet'
    ])
),
AccMap AS (
    SELECT 
        "Account Number" AS acc_num, 
        "Entity ID" AS entity_id
    FROM read_parquet('data/HI_Large/3_filtered_accounts.parquet')
),
AccCounts AS (
    SELECT acc_num, COUNT(*) AS entity_count
    FROM AccMap
    GROUP BY acc_num
),
ResolvedTx AS (
    SELECT
        t.ts,
        src.entity_id AS source_entity,
        tgt.entity_id AS target_entity,
        t.amount_sent_c / (src_c.entity_count * tgt_c.entity_count) AS adj_sent
    FROM RawTx t
    JOIN AccMap src ON t.from_account = src.acc_num
    JOIN AccMap tgt ON t.to_account = tgt.acc_num
    JOIN AccCounts src_c ON t.from_account = src_c.acc_num
    JOIN AccCounts tgt_c ON t.to_account = tgt_c.acc_num
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
    r.adj_sent
FROM Calendar c
JOIN ResolvedTx r 
  ON r.ts > (c.window_date - INTERVAL 7 DAY) 
 AND r.ts <= c.window_date;

-- 2. Aggregate and Export Edges
COPY (
    SELECT
        strftime(window_date, '%Y-%m-%d') AS window_date,
        source_entity AS source,
        target_entity AS target,
        SUM(adj_sent) AS volume,
        COUNT(*) AS count,
        COALESCE(STDDEV_SAMP(adj_sent), 0.0) AS amount_std,
        SUM(adj_sent) * LOG2(1 + COUNT(*)) * (1 + 1.0 / (1.0 + (COALESCE(STDDEV_SAMP(adj_sent), 0.0) / ((SUM(adj_sent) / COUNT(*)) + 1e-9)))) AS weight
    FROM WindowedTx
    GROUP BY window_date, source_entity, target_entity
) TO 'data/HI_Large/aggregated_edges.parquet' (FORMAT PARQUET);

-- 3. Aggregate and Export Nodes
COPY (
    WITH NodeSent AS (
        SELECT 
            window_date, 
            source_entity AS entity_id, 
            SUM(adj_sent) AS vol_sent, 
            COUNT(*) AS tx_count_sent
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
    )
    SELECT
        strftime(COALESCE(s.window_date, r.window_date), '%Y-%m-%d') AS window_date,
        COALESCE(s.entity_id, r.entity_id) AS entity_id,
        COALESCE(s.vol_sent, 0.0) AS vol_sent,
        COALESCE(r.vol_recv, 0.0) AS vol_recv,
        COALESCE(s.tx_count_sent, 0) + COALESCE(r.tx_count_recv, 0) AS tx_count
    FROM NodeSent s
    FULL OUTER JOIN NodeRecv r
        ON s.window_date = r.window_date AND s.entity_id = r.entity_id
) TO 'data/HI_Large/aggregated_nodes.parquet' (FORMAT PARQUET);
