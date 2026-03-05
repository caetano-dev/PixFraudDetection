import duckdb
import os
import shutil
import pandas as pd
from google.colab import drive

# 1. Mount Drive & Define Paths
drive.mount('/content/drive')
drive_dir = '/content/drive/MyDrive/AML/HI_Large'
local_out_dir = '/content/local_output'
spill_dir = '/content/duckdb_spill_dir'

os.makedirs(local_out_dir, exist_ok=True)
os.makedirs(spill_dir, exist_ok=True)

# Connect to a persistent disk-backed database to prevent RAM exhaustion
con = duckdb.connect('/content/pipeline.db')

# 2. Strict RAM boundaries and Table Initialization
con.execute(f"""
    PRAGMA memory_limit='8GB';
    PRAGMA threads=2;
    PRAGMA temp_directory='{spill_dir}';
    PRAGMA enable_progress_bar=false;

    DROP TABLE IF EXISTS FinalEdges;
    DROP TABLE IF EXISTS FinalNodes;

    CREATE TABLE FinalEdges (
        window_date VARCHAR, source VARCHAR, target VARCHAR,
        volume DOUBLE, count BIGINT, amount_std DOUBLE, weight DOUBLE
    );
    CREATE TABLE FinalNodes (
        window_date VARCHAR, entity_id VARCHAR,
        vol_sent DOUBLE, vol_recv DOUBLE, tx_count BIGINT
    );
""")

# 3. Materialize ResolvedTx
print("Materializing ResolvedTx...")
con.execute(f"""
    CREATE TEMP TABLE ResolvedTx AS
    WITH RawTx AS (
        SELECT
            from_account, to_account, amount_sent_c,
            CAST(timestamp AS TIMESTAMP) AS ts
        FROM read_parquet([
            '{drive_dir}/1_filtered_normal_transactions.parquet',
            '{drive_dir}/2_filtered_laundering_transactions.parquet'
        ])
    ),
    AccMap AS (
        SELECT "Account Number" AS acc_num, "Entity ID" AS entity_id
        FROM read_parquet('{drive_dir}/3_filtered_accounts.parquet')
    ),
    AccCounts AS (
        SELECT acc_num, COUNT(*) AS entity_count
        FROM AccMap GROUP BY acc_num
    )
    SELECT
        t.ts, src.entity_id AS source_entity, tgt.entity_id AS target_entity,
        t.amount_sent_c / (src_c.entity_count * tgt_c.entity_count) AS adj_sent
    FROM RawTx t
    JOIN AccMap src ON t.from_account = src.acc_num
    JOIN AccMap tgt ON t.to_account = tgt.acc_num
    JOIN AccCounts src_c ON t.from_account = src_c.acc_num
    JOIN AccCounts tgt_c ON t.to_account = tgt_c.acc_num;
""")

# 4. Fetch date range matching the original MIN(ts) + 1 DAY logic
min_date_str, max_date_str = con.execute("SELECT MIN(CAST(ts AS DATE)), MAX(CAST(ts AS DATE)) FROM ResolvedTx").fetchone()
date_range = pd.date_range(start=pd.to_datetime(min_date_str) + pd.Timedelta(days=1), end=max_date_str)

print(f"Processing rolling windows from {date_range[0].strftime('%Y-%m-%d')} to {max_date_str}...")

# 5. Iterative Processing
for current_date in date_range:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"Aggregating {date_str}...")

    con.execute(f"""
        INSERT INTO FinalEdges
        SELECT
            '{date_str}' AS window_date,
            source_entity AS source, target_entity AS target, SUM(adj_sent) AS volume, COUNT(*) AS count,
            COALESCE(STDDEV_SAMP(adj_sent), 0.0) AS amount_std,
            SUM(adj_sent) * LOG2(1 + COUNT(*)) * (1 + 1.0 / (1.0 + (COALESCE(STDDEV_SAMP(adj_sent), 0.0) / ((SUM(adj_sent) / COUNT(*)) + 1e-9)))) AS weight
        FROM ResolvedTx
        WHERE ts > (CAST('{date_str}' AS DATE) - INTERVAL 7 DAY)
          AND ts <= CAST('{date_str}' AS DATE)
        GROUP BY source_entity, target_entity;

        INSERT INTO FinalNodes
        WITH WindowData AS (
            SELECT source_entity, target_entity, adj_sent
            FROM ResolvedTx
            WHERE ts > (CAST('{date_str}' AS DATE) - INTERVAL 7 DAY)
              AND ts <= CAST('{date_str}' AS DATE)
        ),
        NodeSent AS (
            SELECT source_entity AS entity_id, SUM(adj_sent) AS vol_sent, COUNT(*) AS tx_count_sent
            FROM WindowData GROUP BY source_entity
        ),
        NodeRecv AS (
            SELECT target_entity AS entity_id, SUM(adj_sent) AS vol_recv, COUNT(*) AS tx_count_recv
            FROM WindowData GROUP BY target_entity
        )
        SELECT
            '{date_str}' AS window_date,
            COALESCE(s.entity_id, r.entity_id) AS entity_id,
            COALESCE(s.vol_sent, 0.0) AS vol_sent, COALESCE(r.vol_recv, 0.0) AS vol_recv,
            COALESCE(s.tx_count_sent, 0) + COALESCE(r.tx_count_recv, 0) AS tx_count
        FROM NodeSent s FULL OUTER JOIN NodeRecv r ON s.entity_id = r.entity_id;
    """)

# 6. Export as monolithic Parquet files
print("Exporting monolithic Parquet files...")
con.execute(f"COPY FinalEdges TO '{local_out_dir}/aggregated_edges.parquet' (FORMAT PARQUET);")
con.execute(f"COPY FinalNodes TO '{local_out_dir}/aggregated_nodes.parquet' (FORMAT PARQUET);")

# 7. Sync to Drive
print("Syncing to Google Drive...")
shutil.copy(f"{local_out_dir}/aggregated_edges.parquet", f"{drive_dir}/aggregated_edges.parquet")
shutil.copy(f"{local_out_dir}/aggregated_nodes.parquet", f"{drive_dir}/aggregated_nodes.parquet")

print("Pipeline executed successfully.")
