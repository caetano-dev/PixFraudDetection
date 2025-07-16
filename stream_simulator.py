import redis
import pandas as pd
import numpy as np
import json
import time

r = redis.Redis(host='localhost', port=6379, db=0)
df_transactions = pd.read_csv('./data/pix_transactions.csv')
channel_name = 'pix_transactions'

print(f"Starting to publish transactions to channel '{channel_name}'...")

for index, row in df_transactions.iterrows():
    message = row.to_dict()
    message_json = json.dumps(message)
    r.publish(channel_name, message_json)
    print(f"Published: {message_json}")
#    time.sleep(np.random.uniform(0.02, 0.06))

print("Finished publishing all transactions.")