import pandas as pd
import numpy as np
import random
import uuid
import os
from multiprocessing import Pool, cpu_count
from faker import Faker
from datetime import datetime, timedelta
import time

def get_realistic_transaction_time(current_date):
    """Generates a realistic timestamp based on hourly probabilities, reflecting typical usage patterns."""
    hour_probabilities = [
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 00:00–05:59
        0.04, 0.06, 0.08, 0.09, 0.09, 0.08,  # 06:00–11:59 (Morning peak)
        0.07, 0.06, 0.06, 0.05, 0.05, 0.04,  # 12:00–17:59 (Afternoon)
        0.03, 0.02, 0.01, 0.01, 0.01, 0.01   # 18:00–23:59 (Evening drop)
    ]
    total_prob = sum(hour_probabilities)
    hour_probabilities = [p / total_prob for p in hour_probabilities]
    
    hour = np.random.choice(24, p=hour_probabilities)
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return current_date.replace(hour=hour, minute=minute, second=second)

print("Starting dataset generation...")
fake = Faker('pt_BR')

# Worker for parallel account generation
def _account_worker(_):
    fake_local = Faker('pt_BR')
    account_type = random.choice(['individual', 'business'])
    account_id = generate_cpf() if account_type == 'individual' else generate_cnpj()
    num_devices = random.randint(1, 3)
    devices = [str(uuid.uuid4()) for _ in range(num_devices)]
    ips = [generate_ip_address() for _ in range(num_devices)]
    creation_date = fake_local.date_time_between(start_date='-3y', end_date='now')
    temp_info = {
        'account_age_days': (datetime.now() - creation_date).days,
        'is_verified': random.choice([True, False]),
        'phone_verified': random.choice([True, False]),
        'ips': ips
    }
    return {
        'account_id': account_id,
        'holder_name': fake_local.name() if account_type == 'individual' else fake_local.company(),
        'account_type': account_type,
        'creation_date': creation_date.isoformat(),
        'account_age_days': temp_info['account_age_days'],
        'state': fake_local.state_abbr(),
        'city': fake_local.city(),
        'is_verified': temp_info['is_verified'],
        'phone_verified': temp_info['phone_verified'],
        'risk_score': calculate_initial_risk(temp_info),
        'devices': '|'.join(devices),
        'ips': '|'.join(ips)
    }

def generate_cpf():
    """Generates a random, mathematically valid CPF number."""
    def calculate_digit(cpf_partial):
        s = sum(int(d) * (len(cpf_partial) + 1 - i) for i, d in enumerate(cpf_partial))
        result = 11 - (s % 11)
        return str(result) if result < 10 else '0'

    cpf_base = [random.randint(0, 9) for _ in range(9)]
    cpf_base_str = "".join(map(str, cpf_base))
    d1 = calculate_digit(cpf_base_str)
    d2 = calculate_digit(cpf_base_str + d1)
    return f"{cpf_base_str[:3]}.{cpf_base_str[3:6]}.{cpf_base_str[6:9]}-{d1}{d2}"

def generate_cnpj():
    """Generates a random, mathematically valid CNPJ number."""
    n = [random.randint(0, 9) for _ in range(12)]
    n.append((sum(n[i] * (6 - (i % 8 + 2)) for i in range(12)) * 10 % 11) % 10)
    n.append((sum(n[i] * (7 - (i % 8 + 2)) for i in range(13)) * 10 % 11) % 10)
    cnpj = "".join(map(str, n))
    return f"{cnpj[:2]}.{cnpj[2:5]}.{cnpj[5:8]}/{cnpj[8:12]}-{cnpj[12:]}"

def generate_ip_address():
    """Generates a random IPv4 address."""
    return fake.ipv4()

def calculate_initial_risk(account_info):
    """Calculates an initial risk score based on account features."""
    risk = 0
    if account_info['account_age_days'] < 30:
        risk += 0.3
    if not account_info['is_verified']:
        risk += 0.2
    if not account_info['phone_verified']:
        risk += 0.2
    if len(account_info['ips']) > 2:
        risk += 0.1
    return min(round(random.uniform(0.1, 0.3) + risk, 2), 1.0)

def get_account_details(df, account_id):
    """Helper to get devices and IPs for a given account."""
    # optimized lookup by index
    account_info = df.loc[account_id]
    devices = account_info['devices'].split('|')
    ips = account_info['ips'].split('|')
    return random.choice(devices), random.choice(ips)

def create_normal_transaction(accounts_df, date, is_weekday=None):
    """Simulates a simple, everyday transaction between any two accounts."""
    if is_weekday is None:
        is_weekday = date.weekday() < 5
    sender_id, receiver_id = np.random.choice(accounts_df['account_id'], 2, replace=False)
    device_id, ip_address = get_account_details(accounts_df, sender_id)
    is_weekend = date.weekday() >= 5
    if is_weekend:
        amount = round(np.random.lognormal(mean=4.5, sigma=1.2), 2)
    else:
        amount = round(np.random.lognormal(mean=5.5, sigma=1.5), 2)
    
    amount = max(amount, 1.0)

    sender_info = accounts_df.loc[sender_id]
    receiver_info = accounts_df.loc[receiver_id]

    return {
        'transaction_id': str(uuid.uuid4()),
        'sender_id': sender_id,
        'receiver_id': receiver_id,
        'amount': amount,
        'timestamp': date.isoformat(),
        'fraud_flag': 'NORMAL',
        'device_id': device_id,
        'ip_address': ip_address,
        'transaction_type': random.choice(['transfer', 'payment', 'purchase']),
        'channel': random.choice(['mobile_app', 'web', 'atm']),
        'merchant_category': random.choice(['grocery', 'restaurant', 'retail', 'gas_station', 'other']),
        'hour_of_day': date.hour,
        'day_of_week': date.weekday(),
        'is_weekend': date.weekday() >= 5,
        'same_state': sender_info['state'] == receiver_info['state'],
        'sender_verified': sender_info['is_verified'],
        'receiver_verified': receiver_info['is_verified'],
        'sender_state': sender_info['state'],
        'receiver_state': receiver_info['state'],
        'sender_risk_score': sender_info['risk_score'],
        'receiver_risk_score': receiver_info['risk_score']
    }

def create_smurfing_ring(accounts_df, date):
    """Simulates a smurfing ring with realistic fraud patterns, including rapid distribution and cash-out."""
    ring_transactions = []

    high_risk_accounts = accounts_df[accounts_df['risk_score'] > 0.5]
    if len(high_risk_accounts) < 5:
        return []

    # A victim with a low-to-medium risk score is more realistic
    victim_pool = accounts_df[accounts_df['risk_score'] < 0.4]
    if victim_pool.empty:
        return []
    victim_id = np.random.choice(victim_pool['account_id'])
    victim_info = accounts_df[accounts_df['account_id'] == victim_id].iloc[0]

    potential_mules = high_risk_accounts[high_risk_accounts['account_id'] != victim_id]
    if len(potential_mules) < 2:
        return []

    num_mules = min(np.random.randint(2, 5), len(potential_mules))
    mules = np.random.choice(potential_mules['account_id'], num_mules, replace=False)
    central_mule = mules[0]
    central_mule_info = accounts_df[accounts_df['account_id'] == central_mule].iloc[0]
    secondary_mules = mules[1:] if len(mules) > 1 else []

    fraud_device, fraud_ip = str(uuid.uuid4()), generate_ip_address()

    # Fraudulent amounts are often just below common thresholds
    stolen_amount = round(np.random.uniform(8000.0, 9990.0), 2)
    transaction_time = date + timedelta(seconds=np.random.randint(10, 60)) # Faster initial transfer
    victim_device, victim_ip = get_account_details(accounts_df, victim_id)

    ring_transactions.append({
        'transaction_id': str(uuid.uuid4()),
        'sender_id': victim_id,
        'receiver_id': central_mule,
        'amount': stolen_amount,
        'timestamp': transaction_time.isoformat(),
        'fraud_flag': 'SMURFING_VICTIM_TRANSFER',
        'device_id': victim_device,
        'ip_address': victim_ip,
        'transaction_type': 'transfer',
        'channel': 'mobile_app',
        'merchant_category': 'other',
        'hour_of_day': transaction_time.hour,
        'day_of_week': transaction_time.weekday(),
        'is_weekend': transaction_time.weekday() >= 5,
        'same_state': victim_info['state'] == central_mule_info['state'],
        'sender_verified': victim_info['is_verified'],
        'receiver_verified': central_mule_info['is_verified'],
        'sender_state': victim_info['state'],
        'receiver_state': central_mule_info['state'],
        'sender_risk_score': victim_info['risk_score'],
        'receiver_risk_score': central_mule_info['risk_score']
    })

    if len(secondary_mules) == 0:
        return ring_transactions

    # Rapid distribution to secondary mules
    amount_per_mule = round(stolen_amount / len(secondary_mules), 2)
    for mule in secondary_mules:
        transaction_time += timedelta(seconds=np.random.randint(20, 90)) # Very fast distribution
        mule_info = accounts_df[accounts_df['account_id'] == mule].iloc[0]

        ring_transactions.append({
            'transaction_id': str(uuid.uuid4()),
            'sender_id': central_mule,
            'receiver_id': mule,
            'amount': round(amount_per_mule * np.random.uniform(0.95, 1.05), 2),
            'timestamp': transaction_time.isoformat(),
            'fraud_flag': 'SMURFING_DISTRIBUTION',
            'device_id': fraud_device,
            'ip_address': fraud_ip,
            'transaction_type': 'transfer',
            'channel': 'mobile_app',
            'merchant_category': 'other',
            'hour_of_day': transaction_time.hour,
            'day_of_week': transaction_time.weekday(),
            'is_weekend': transaction_time.weekday() >= 5,
            'same_state': central_mule_info['state'] == mule_info['state'],
            'sender_verified': central_mule_info['is_verified'],
            'receiver_verified': mule_info['is_verified'],
            'sender_state': central_mule_info['state'],
            'receiver_state': mule_info['state'],
            'sender_risk_score': central_mule_info['risk_score'],
            'receiver_risk_score': mule_info['risk_score']
        })

        # Add a final "cash-out" transaction for each secondary mule
        cash_out_time = transaction_time + timedelta(minutes=np.random.randint(1, 10))
        cash_out_receiver_pool = high_risk_accounts[~high_risk_accounts['account_id'].isin(mules)]
        if not cash_out_receiver_pool.empty:
            cash_out_receiver_id = np.random.choice(cash_out_receiver_pool['account_id'])
            cash_out_receiver_info = accounts_df[accounts_df['account_id'] == cash_out_receiver_id].iloc[0]

            ring_transactions.append({
                'transaction_id': str(uuid.uuid4()),
                'sender_id': mule,
                'receiver_id': cash_out_receiver_id,
                'amount': round(amount_per_mule * np.random.uniform(0.9, 0.98), 2), # Slightly less to leave dust
                'timestamp': cash_out_time.isoformat(),
                'fraud_flag': 'SMURFING_CASH_OUT',
                'device_id': fraud_device,
                'ip_address': fraud_ip,
                'transaction_type': 'payment',
                'channel': 'web',
                'merchant_category': 'digital_wallet',
                'hour_of_day': cash_out_time.hour,
                'day_of_week': cash_out_time.weekday(),
                'is_weekend': cash_out_time.weekday() >= 5,
                'same_state': mule_info['state'] == cash_out_receiver_info['state'],
                'sender_verified': mule_info['is_verified'],
                'receiver_verified': cash_out_receiver_info['is_verified'],
                'sender_state': mule_info['state'],
                'receiver_state': cash_out_receiver_info['state'],
                'sender_risk_score': mule_info['risk_score'],
                'receiver_risk_score': cash_out_receiver_info['risk_score']
            })

    return ring_transactions

def create_circular_payment_ring(accounts_df, date):
    """Simulates circular payments to build artificial transaction history."""
    ring_transactions = []

    high_risk_accounts = accounts_df[accounts_df['risk_score'] > 0.6]
    if len(high_risk_accounts) < 3:
        return []

    num_in_cycle = np.random.randint(3, min(6, len(high_risk_accounts) + 1))
    cycle_accounts = np.random.choice(high_risk_accounts['account_id'], num_in_cycle, replace=False)

    fraud_device, fraud_ip = str(uuid.uuid4()), generate_ip_address()

    # Use amounts that look less suspicious (not perfect round numbers)
    base_amount = round(np.random.uniform(1000.0, 5000.0) / 100) * 100 * np.random.uniform(0.95, 1.05)
    transaction_time = date

    # Add "warm-up" transactions for one of the accounts in the cycle
    warmup_account_id = cycle_accounts[0]
    for _ in range(np.random.randint(1, 4)):
        warmup_receiver = np.random.choice(accounts_df[accounts_df['risk_score'] < 0.3]['account_id'])
        if warmup_receiver == warmup_account_id: continue

        warmup_time = transaction_time - timedelta(days=np.random.randint(1, 5), hours=np.random.randint(1,12))
        warmup_amount = round(np.random.uniform(20.0, 150.0), 2)
        sender_info = accounts_df[accounts_df['account_id'] == warmup_account_id].iloc[0]
        receiver_info = accounts_df[accounts_df['account_id'] == warmup_receiver].iloc[0]
        device_id, ip_address = get_account_details(accounts_df, warmup_account_id)

        ring_transactions.append({
            'transaction_id': str(uuid.uuid4()),
            'sender_id': warmup_account_id,
            'receiver_id': warmup_receiver,
            'amount': warmup_amount,
            'timestamp': warmup_time.isoformat(),
            'fraud_flag': 'NORMAL', # Disguised as normal activity
            'device_id': device_id,
            'ip_address': ip_address,
            'transaction_type': 'payment',
            'channel': 'mobile_app',
            'merchant_category': 'retail',
            'hour_of_day': warmup_time.hour,
            'day_of_week': warmup_time.weekday(),
            'is_weekend': warmup_time.weekday() >= 5,
            'same_state': sender_info['state'] == receiver_info['state'],
            'sender_verified': sender_info['is_verified'],
            'receiver_verified': receiver_info['is_verified'],
            'sender_state': sender_info['state'],
            'receiver_state': receiver_info['state'],
            'sender_risk_score': sender_info['risk_score'],
            'receiver_risk_score': receiver_info['risk_score']
        })


    for i in range(num_in_cycle):
        sender_id = cycle_accounts[i]
        receiver_id = cycle_accounts[(i + 1) % num_in_cycle]
        transaction_time += timedelta(minutes=np.random.randint(1, 15)) # Faster cycles

        sender_info = accounts_df.loc[sender_id]
        receiver_info = accounts_df.loc[receiver_id]

        ring_transactions.append({
            'transaction_id': str(uuid.uuid4()),
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'amount': round(base_amount * np.random.uniform(0.98, 1.02), 2),
            'timestamp': transaction_time.isoformat(),
            'fraud_flag': 'CIRCULAR_PAYMENT',
            'device_id': fraud_device,
            'ip_address': fraud_ip,
            'transaction_type': 'transfer',
            'channel': 'mobile_app',
            'merchant_category': 'other',
            'hour_of_day': transaction_time.hour,
            'day_of_week': transaction_time.weekday(),
            'is_weekend': transaction_time.weekday() >= 5,
            'same_state': sender_info['state'] == receiver_info['state'],
            'sender_verified': sender_info['is_verified'],
            'receiver_verified': receiver_info['is_verified'],
            'sender_state': sender_info['state'],
            'receiver_state': receiver_info['state'],
            'sender_risk_score': sender_info['risk_score'],
            'receiver_risk_score': receiver_info['risk_score']
        })

    return ring_transactions

def create_salary_payments(accounts_df, date):
    """Simulates legitimate salary payments from businesses to individuals."""
    salary_transactions = []

    businesses = accounts_df[accounts_df['account_type'] == 'business']
    individuals = accounts_df[accounts_df['account_type'] == 'individual']

    if len(businesses) == 0 or len(individuals) < 5:
        return []

    employer_id = np.random.choice(businesses['account_id'])
    employer_device, employer_ip = get_account_details(accounts_df, employer_id)

    num_employees = min(np.random.randint(5, 31), len(individuals))
    employees = np.random.choice(individuals['account_id'], num_employees, replace=False)

    for employee_id in employees:
        salary_amount = round(np.random.uniform(2500.0, 8000.0), 2)
        transaction_time = date + timedelta(minutes=np.random.randint(0, 60))

        sender_info = accounts_df[accounts_df['account_id'] == employer_id].iloc[0]
        receiver_info = accounts_df[accounts_df['account_id'] == employee_id].iloc[0]

        salary_transactions.append({
            'transaction_id': str(uuid.uuid4()),
            'sender_id': employer_id,
            'receiver_id': employee_id,
            'amount': salary_amount,
            'timestamp': transaction_time.isoformat(),
            'fraud_flag': 'NORMAL_SALARY',
            'device_id': employer_device,
            'ip_address': employer_ip,
            'transaction_type': 'transfer',
            'channel': 'web',
            'merchant_category': 'business_services',
            'hour_of_day': transaction_time.hour,
            'day_of_week': transaction_time.weekday(),
            'is_weekend': transaction_time.weekday() >= 5,
            'same_state': sender_info['state'] == receiver_info['state'],
            'sender_verified': sender_info['is_verified'],
            'receiver_verified': receiver_info['is_verified'],
            'sender_state': sender_info['state'],
            'receiver_state': receiver_info['state'],
            'sender_risk_score': sender_info['risk_score'],
            'receiver_risk_score': receiver_info['risk_score']
        })

    return salary_transactions

def create_microtransaction(accounts_df, date):
    """Simulates small value transactions (coffee, snacks, transport)."""
    sender_id, receiver_id = np.random.choice(accounts_df['account_id'], 2, replace=False)
    device_id, ip_address = get_account_details(accounts_df, sender_id)
    amount = round(np.random.uniform(1.0, 50.0), 2)

    sender_info = accounts_df[accounts_df['account_id'] == sender_id].iloc[0]
    receiver_info = accounts_df[accounts_df['account_id'] == receiver_id].iloc[0]

    return {
        'transaction_id': str(uuid.uuid4()),
        'sender_id': sender_id,
        'receiver_id': receiver_id,
        'amount': amount,
        'timestamp': date.isoformat(),
        'fraud_flag': 'NORMAL_MICRO',
        'device_id': device_id,
        'ip_address': ip_address,
        'transaction_type': 'payment',
        'channel': random.choice(['mobile_app', 'pos']),
        'merchant_category': random.choice(['grocery', 'restaurant', 'transport']),
        'hour_of_day': date.hour,
        'day_of_week': date.weekday(),
        'is_weekend': date.weekday() >= 5,
        'same_state': sender_info['state'] == receiver_info['state'],
        'sender_verified': sender_info['is_verified'],
        'receiver_verified': receiver_info['is_verified'],
        'sender_state': sender_info['state'],
        'receiver_state': receiver_info['state'],
        'sender_risk_score': sender_info['risk_score'],
        'receiver_risk_score': receiver_info['risk_score']
    }

def create_business_transaction(accounts_df, date):
    """Simulates B2B transactions with higher amounts."""
    businesses = accounts_df[accounts_df['account_type'] == 'business']
    if len(businesses) < 2:
        return create_normal_transaction(accounts_df, date)

    sender_id, receiver_id = np.random.choice(businesses['account_id'], 2, replace=False)
    device_id, ip_address = get_account_details(accounts_df, sender_id)
    amount = round(np.random.uniform(1000.0, 25000.0), 2)

    sender_info = accounts_df[accounts_df['account_id'] == sender_id].iloc[0]
    receiver_info = accounts_df[accounts_df['account_id'] == receiver_id].iloc[0]

    return {
        'transaction_id': str(uuid.uuid4()),
        'sender_id': sender_id,
        'receiver_id': receiver_id,
        'amount': amount,
        'timestamp': date.isoformat(),
        'fraud_flag': 'NORMAL_B2B',
        'device_id': device_id,
        'ip_address': ip_address,
        'transaction_type': 'transfer',
        'channel': 'web',
        'merchant_category': 'business_services',
        'hour_of_day': date.hour,
        'day_of_week': date.weekday(),
        'is_weekend': date.weekday() >= 5,
        'same_state': sender_info['state'] == receiver_info['state'],
        'sender_verified': sender_info['is_verified'],
        'receiver_verified': receiver_info['is_verified'],
        'sender_state': sender_info['state'],
        'receiver_state': receiver_info['state'],
        'sender_risk_score': sender_info['risk_score'],
        'receiver_risk_score': receiver_info['risk_score']
    }

def generate_accounts(num_accounts):
    """Generate and save synthetic account data."""
    data_folder = './data'
    os.makedirs(data_folder, exist_ok=True)

    # Parallel account generation with progress feedback
    accounts = []
    total_accounts = num_accounts
    start_time = time.time()
    with Pool(cpu_count()) as pool:
        for idx, record in enumerate(pool.imap_unordered(_account_worker, range(total_accounts)), 1):
            accounts.append(record)
            if idx % max(1, total_accounts // 10) == 0 or idx == total_accounts:
                elapsed = time.time() - start_time
                eta = elapsed / idx * (total_accounts - idx)
                print(f"Accounts generated: {idx}/{total_accounts} "
                      f"({idx/total_accounts*100:.1f}%), ETA {eta:.1f}s")

    df = pd.DataFrame(accounts)
    df.to_csv('./data/pix_accounts.csv', index=False)
    return df

def _day_worker(args):
    day_idx, tx_count, start_dt, smurf_set, circ_set, df = args
    daily = []
    curr_date = start_dt + timedelta(days=day_idx)
    # Fraud injection
    if day_idx in smurf_set:
        daily.extend(create_smurfing_ring(df, get_realistic_transaction_time(curr_date)))
    if day_idx in circ_set:
        daily.extend(create_circular_payment_ring(df, get_realistic_transaction_time(curr_date)))
    # Normal transactions
    for _ in range(tx_count):
        ttime = get_realistic_transaction_time(curr_date)
        rv = np.random.rand()
        if curr_date.day in [5, 20] and rv < 0.1:
            daily.extend(create_salary_payments(df, ttime))
        elif rv < 0.05:
            daily.append(create_business_transaction(df, ttime))
        elif rv < 0.45:
            daily.append(create_microtransaction(df, ttime))
        else:
            daily.append(create_normal_transaction(df, ttime))
    return daily

def generate_transactions(accounts_df, total_tx, start_date, end_date, smurf_events, circular_events):
    """
    Generate transactions with a more realistic daily distribution and controlled fraud injection.
    Parallelized using multiprocessing for per-day workloads.
    """
    # Prepare accounts for fast lookup
    if accounts_df.index.name != 'account_id':
        accounts_df = accounts_df.set_index('account_id', drop=False)

    # Pre-calculate daily transaction counts
    num_days = (end_date - start_date).days + 1
    daily_weights = np.array([5 if i % 7 < 5 else 1 for i in range(num_days)], dtype=float)
    daily_weights /= daily_weights.sum()
    transactions_per_day = np.random.multinomial(total_tx, daily_weights)

    # Schedule fraud events
    fraud_days = sorted(random.sample(range(num_days), smurf_events + circular_events))
    smurfing_days = set(fraud_days[:smurf_events])
    circular_days = set(fraud_days[smurf_events:])

    # Build parameters for each day
    params = [
        (day_idx, tx_count, start_date, smurfing_days, circular_days, accounts_df)
        for day_idx, tx_count in enumerate(transactions_per_day)
    ]

    # Parallel execution across CPU cores with progress feedback
    transactions = []
    total_days = len(params)
    start_time = time.time()
    with Pool(cpu_count()) as pool:
        for day_idx, day_list in enumerate(pool.imap_unordered(_day_worker, params), 1):
            transactions.extend(day_list)
            if day_idx % max(1, total_days // 10) == 0 or day_idx == total_days:
                elapsed = time.time() - start_time
                eta = elapsed / day_idx * (total_days - day_idx)
                print(f"Processed day {day_idx}/{total_days} ({day_idx/total_days*100:.1f}%), ETA {eta:.1f}s")

    df_tx = pd.DataFrame(transactions).dropna().sample(frac=1).reset_index(drop=True)
    df_tx.to_csv('./data/pix_transactions.csv', index=False)
    return df_tx

def main():
    """Main function to orchestrate the data generation."""
    NUM_ACCOUNTS = 10000
    TOTAL_TRANSACTIONS = 1050000
    REALISTIC_FRAUD_RATE = 0.00007 # 0.007%
    
    target_fraud_tx = TOTAL_TRANSACTIONS * REALISTIC_FRAUD_RATE
    smurf_events = int((target_fraud_tx * 0.6) / 11) # Assume 60% of fraud is smurfing
    circular_events = int((target_fraud_tx * 0.4) / 4) # And 40% is circular

    print("Generating accounts...")
    accounts_df = generate_accounts(NUM_ACCOUNTS)
    print(f"{len(accounts_df)} accounts generated.")

    print("Generating transactions...")
    transactions_df = generate_transactions(
        accounts_df,
        TOTAL_TRANSACTIONS,
        datetime(2024, 1, 1),
        datetime(2024, 12, 31),
        smurf_events,
        circular_events
    )
    
    print(f"Successfully generated {len(transactions_df)} transactions.")
    
    # --- Final Summary ---
    fraud_counts = transactions_df['fraud_flag'].value_counts()
    normal_flags = ['NORMAL', 'NORMAL_MICRO', 'NORMAL_SALARY', 'NORMAL_B2B']
    total_fraud = sum(count for flag, count in fraud_counts.items() if flag not in normal_flags)
    fraud_rate = (total_fraud / len(transactions_df)) * 100

    print("\n--- Dataset Summary ---")
    print(f"Total transactions: {len(transactions_df):,}")
    print(f"Total accounts: {len(accounts_df):,}")
    print(f"Target fraud rate: ~{REALISTIC_FRAUD_RATE*100:.4f}%")
    print(f"Actual fraud rate: {fraud_rate:.4f}%")
    print("\nTransaction Type Distribution:")
    for flag, count in fraud_counts.items():
        print(f"  - {flag}: {count:,} ({count/len(transactions_df)*100:.2f}%)")

if __name__ == "__main__":
    main()