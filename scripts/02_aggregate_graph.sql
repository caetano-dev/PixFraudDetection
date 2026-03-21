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
        t.amount_sent_c * fx.rate AS amount_sent_usd, -- anomaly detection of graphs, weirdnodes
        t.payment_currency,
        t.payment_format
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
    CAST(t.is_laundering AS INTEGER) AS is_laundering,
    t.payment_currency,
    t.payment_format
FROM RawTx t
JOIN AccMap src ON t.from_account = src.acc_num
JOIN AccMap tgt ON t.to_account = tgt.acc_num;

-- 2. Sliding Window Calendar (Discrete Non-Cumulative Windows)
CREATE TEMP TABLE WindowCalendar AS
WITH DataBounds AS (
    SELECT 
        MIN(tx_date) AS first_date,
        MAX(tx_date) AS last_date
    FROM ResolvedTx
),
WindowParams AS (
    SELECT 3 AS window_size_days, 1 AS window_stride_days
)
SELECT 
    ROW_NUMBER() OVER (ORDER BY window_start) - 1 AS window_id,
    window_start,
    window_start + (window_size_days * INTERVAL 1 DAY) AS window_end
FROM (
    SELECT 
        unnest(generate_series(
            (SELECT first_date FROM DataBounds)::TIMESTAMP,
            (SELECT last_date FROM DataBounds)::TIMESTAMP,
            (SELECT window_stride_days FROM WindowParams) * INTERVAL 1 DAY
        ))::DATE AS window_start,
        (SELECT window_size_days FROM WindowParams) AS window_size_days
) sub
WHERE window_start + (window_size_days * INTERVAL 1 DAY) <= (SELECT last_date + INTERVAL 1 DAY FROM DataBounds);

-- 3. SLIDING WINDOW EDGES (Strict Non-Cumulative)
-- Each edge is aggregated ONLY from transactions within [window_start, window_end)
COPY (
    SELECT
        w.window_id,
        strftime(w.window_start, '%Y-%m-%d') AS window_start,
        strftime(w.window_end, '%Y-%m-%d') AS window_end,
        r.source_entity AS source,
        r.target_entity AS target,
        SUM(r.adj_sent) AS volume,
        COUNT(*) AS count,
        COALESCE(STDDEV_SAMP(EXTRACT(EPOCH FROM r.ts)), 0.0) AS time_variance,
    FROM WindowCalendar w
    JOIN ResolvedTx r 
      ON r.tx_date >= w.window_start 
     AND r.tx_date < w.window_end
    GROUP BY w.window_id, w.window_start, w.window_end, r.source_entity, r.target_entity
) TO 'data/HI_Small/lookback_edges.parquet' (FORMAT PARQUET);

-- 4. TARGET NODES (Aggregated per Window) -> Labels and Behavioral Features
COPY (
    WITH NodeSent AS (
        SELECT 
            w.window_id,
            w.window_start,
            w.window_end,
            r.source_entity AS entity_id, 
            SUM(r.adj_sent) AS vol_sent, 
            COUNT(*) AS tx_count_sent, 
            COALESCE(STDDEV_SAMP(EXTRACT(EPOCH FROM r.ts)), 0.0) AS time_variance,
            COUNT(DISTINCT r.payment_currency) AS distinct_currencies_sent,
            SUM(CASE WHEN r.payment_format = 'Wire' THEN 1 ELSE 0 END) AS wire_count_sent,
            SUM(CASE WHEN r.payment_format = 'Cash' THEN 1 ELSE 0 END) AS cash_count_sent,
            SUM(CASE WHEN r.payment_format = 'Bitcoin' THEN 1 ELSE 0 END) AS bitcoin_count_sent,
            SUM(CASE WHEN r.payment_format = 'Cheque' THEN 1 ELSE 0 END) AS cheque_count_sent,
            SUM(CASE WHEN r.payment_format = 'Credit Card' THEN 1 ELSE 0 END) AS credit_card_count_sent,
            SUM(CASE WHEN r.payment_format = 'ACH' THEN 1 ELSE 0 END) AS ach_count_sent,
            SUM(CASE WHEN r.payment_format = 'Reinvestment' THEN 1 ELSE 0 END) AS reinvestment_count_sent
        FROM WindowCalendar w
        JOIN ResolvedTx r ON r.tx_date >= w.window_start AND r.tx_date < w.window_end
        GROUP BY w.window_id, w.window_start, w.window_end, r.source_entity
    ),
    NodeRecv AS (
        SELECT 
            w.window_id,
            w.window_start,
            w.window_end,
            r.target_entity AS entity_id, 
            SUM(r.adj_sent) AS vol_recv, 
            COUNT(*) AS tx_count_recv,
            COUNT(DISTINCT r.payment_currency) AS distinct_currencies_recv,
            SUM(CASE WHEN r.payment_format = 'Wire' THEN 1 ELSE 0 END) AS wire_count_recv,
            SUM(CASE WHEN r.payment_format = 'Cash' THEN 1 ELSE 0 END) AS cash_count_recv,
            SUM(CASE WHEN r.payment_format = 'Bitcoin' THEN 1 ELSE 0 END) AS bitcoin_count_recv,
            SUM(CASE WHEN r.payment_format = 'Cheque' THEN 1 ELSE 0 END) AS cheque_count_recv,
            SUM(CASE WHEN r.payment_format = 'Credit Card' THEN 1 ELSE 0 END) AS credit_card_count_recv,
            SUM(CASE WHEN r.payment_format = 'ACH' THEN 1 ELSE 0 END) AS ach_count_recv,
            SUM(CASE WHEN r.payment_format = 'Reinvestment' THEN 1 ELSE 0 END) AS reinvestment_count_recv
        FROM WindowCalendar w
        JOIN ResolvedTx r ON r.tx_date >= w.window_start AND r.tx_date < w.window_end
        GROUP BY w.window_id, w.window_start, w.window_end, r.target_entity
    ),
    EntityFraud AS (
        SELECT window_id, entity_id, MAX(is_fraud) AS is_fraud 
        FROM (
            SELECT w.window_id, r.source_entity AS entity_id, r.is_laundering AS is_fraud 
            FROM WindowCalendar w
            JOIN ResolvedTx r ON r.tx_date >= w.window_start AND r.tx_date < w.window_end
            UNION ALL
            SELECT w.window_id, r.target_entity AS entity_id, r.is_laundering AS is_fraud 
            FROM WindowCalendar w
            JOIN ResolvedTx r ON r.tx_date >= w.window_start AND r.tx_date < w.window_end
        ) AS sub_union
        WHERE entity_id IS NOT NULL 
        GROUP BY window_id, entity_id
    )
    SELECT
        COALESCE(s.window_id, r.window_id) AS window_id,
        strftime(COALESCE(s.window_start, r.window_start), '%Y-%m-%d') AS window_start,
        strftime(COALESCE(s.window_end, r.window_end), '%Y-%m-%d') AS window_end,
        COALESCE(s.entity_id, r.entity_id) AS entity_id,
        COALESCE(s.vol_sent, 0.0) AS vol_sent,
        COALESCE(r.vol_recv, 0.0) AS vol_recv,
        COALESCE(s.tx_count_sent, 0) + COALESCE(r.tx_count_recv, 0) AS tx_count,
        COALESCE(s.time_variance, 0.0) AS time_variance,
        COALESCE(s.distinct_currencies_sent, 0) AS distinct_currencies_sent,
        COALESCE(r.distinct_currencies_recv, 0) AS distinct_currencies_recv,
        COALESCE(s.wire_count_sent, 0) AS wire_count_sent,
        COALESCE(r.wire_count_recv, 0) AS wire_count_recv,
        COALESCE(s.cash_count_sent, 0) AS cash_count_sent,
        COALESCE(r.cash_count_recv, 0) AS cash_count_recv,
        COALESCE(s.bitcoin_count_sent, 0) AS bitcoin_count_sent,
        COALESCE(r.bitcoin_count_recv, 0) AS bitcoin_count_recv,
        COALESCE(s.cheque_count_sent, 0) AS cheque_count_sent,
        COALESCE(r.cheque_count_recv, 0) AS cheque_count_recv,
        COALESCE(s.credit_card_count_sent, 0) AS credit_card_count_sent,
        COALESCE(r.credit_card_count_recv, 0) AS credit_card_count_recv,
        COALESCE(s.ach_count_sent, 0) AS ach_count_sent,
        COALESCE(r.ach_count_recv, 0) AS ach_count_recv,
        COALESCE(s.reinvestment_count_sent, 0) AS reinvestment_count_sent,
        COALESCE(r.reinvestment_count_recv, 0) AS reinvestment_count_recv,

        COALESCE(ef.is_fraud, 0) AS is_fraud
    FROM NodeSent s
    FULL OUTER JOIN NodeRecv r ON s.window_id = r.window_id AND s.entity_id = r.entity_id
    LEFT JOIN EntityFraud ef ON COALESCE(s.window_id, r.window_id) = ef.window_id AND COALESCE(s.entity_id, r.entity_id) = ef.entity_id
) TO 'data/HI_Small/target_nodes.parquet' (FORMAT PARQUET);
