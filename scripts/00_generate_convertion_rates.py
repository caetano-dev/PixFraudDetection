import yfinance as yf
import pandas as pd

# The start date is Aug 1, 2022. End date is exclusive in yfinance, so we use Nov 6.
START_DATE = '2022-08-01'
YF_END_DATE = '2022-11-06'
END_DATE_ACTUAL = '2022-11-05'

currencies = {
    'Euro': 'EURUSD=X',
    'UK Pound': 'GBPUSD=X',
    'Swiss Franc': 'CHFUSD=X',
    'Australian Dollar': 'AUDUSD=X',
    'Canadian Dollar': 'CADUSD=X',
    'Brazil Real': 'BRLUSD=X',
    'Shekel': 'ILSUSD=X',
    'Saudi Riyal': 'SARUSD=X',
    'Yuan': 'CNYUSD=X',
    'Mexican Peso': 'MXNUSD=X',
    'Ruble': 'RUBUSD=X',
    'Rupee': 'INRUSD=X',
    'Yen': 'JPYUSD=X',
    'Bitcoin': 'BTC-USD'
}

calendar = pd.date_range(start=START_DATE, end=END_DATE_ACTUAL)

dfs = []
for name, ticker in currencies.items():
    data = yf.download(ticker, start=START_DATE, end=YF_END_DATE, interval='1d')
    
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        df = data[['Close']].reset_index()
        df.columns = ['date', 'rate']
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        
        temp_df = pd.DataFrame({'date': calendar})
        temp_df = temp_df.merge(df, on='date', how='left')
        temp_df['currency'] = name
        
        temp_df['rate'] = temp_df['rate'].ffill().bfill()
        dfs.append(temp_df)

rates_df = pd.concat(dfs, ignore_index=True)
rates_df['date'] = rates_df['date'].dt.date

usd_df = pd.DataFrame({'date': calendar.date, 'rate': 1.0, 'currency': 'US Dollar'})
rates_df = pd.concat([rates_df, usd_df], ignore_index=True)

rates_df.to_parquet('data/fx_rates.parquet', engine='pyarrow')
