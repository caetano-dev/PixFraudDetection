import redis
import pandas as pd
import numpy as np
import json
import time
from src.config.config import REDIS_HOST, REDIS_PORT, REDIS_DB

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
df_transactions = pd.read_csv('src/data/pix_transactions.csv')
channel_name = 'pix_transactions'

print(f"Starting to publish transactions to channel '{channel_name}'...")

for index, row in df_transactions.iterrows():
    message = row.to_dict()
    message_json = json.dumps(message)
    r.publish(channel_name, message_json)
    print(f"Published: {message_json}")
    time.sleep(np.random.uniform(0.01, 0.02))

print("Finished publishing all transactions.")