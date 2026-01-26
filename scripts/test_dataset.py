import pandas as pd

# script to check which transaction is the last, just to check if the date cutoff is correct

def main():
    # Load data (Fix spelling if file on disk is correct)
    df = pd.read_parquet('../data/2_filtered_laundering_transactions.parquet')
    
    # Ensure datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Efficient calculation using pandas aggregation
    # Group by attempt_id, find min and max timestamp, then subtract
    grouped = df.groupby('attempt_id')['timestamp'].agg(['min', 'max'])
    grouped['duration_seconds'] = (grouped['max'] - grouped['min']).dt.total_seconds()
    
    # Sort to find the longest ones easily
    longest_durations = grouped['duration_seconds'].sort_values(ascending=False)
    
    print('Longest 5 laundering durations (seconds):')
    print(longest_durations.head(5))

    # If you specifically need the absolute longest single value:
    print(f"\nMax duration: {longest_durations.iloc[0]} seconds")

if __name__ == "__main__":
    main()