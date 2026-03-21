import duckdb
from pathlib import Path
from src.config import DATA_PATH, WINDOW_SIZE, WINDOW_STRIDE

def main():
    sql_path = Path("./scripts/02_aggregate_graph.sql")
    query = sql_path.read_text()
    
    query = query.replace("'data/HI_Small/", f"'{DATA_PATH}/")
    query = query.replace("SELECT 1 AS window_size_days, 1 AS window_stride_days", 
                          f"SELECT {WINDOW_SIZE} AS window_size_days, {WINDOW_STRIDE} AS window_stride_days")
    
    print(f"Executing aggregation with {WINDOW_SIZE}-day windows and {WINDOW_STRIDE}-day stride on {DATA_PATH}...")
    duckdb.execute(query)
    print("Graph aggregation complete.")

if __name__ == "__main__":
    main()
