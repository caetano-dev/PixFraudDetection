import pandas as pd
df = pd.read_parquet('../data/HI_Large/2_filtered_laundering_transactions.parquet')

df = df[df['timestamp'] <= '2022-11-06'] # HI large
#df = df[df['timestamp'] <= '2022-09-11']  # LI
df.to_parquet('../data/HI_Large/cleaned_laundering_transactions.parquet', index=False)

# check max date
print(df['timestamp'].min())
print(df['timestamp'].max())