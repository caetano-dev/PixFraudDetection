import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import uuid

print("Starting dataset generation...")
fake = Faker('pt_BR')

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
    # Newer accounts are riskier
    if account_info['account_age_days'] < 30:
        risk += 0.3
    # Unverified accounts are riskier
    if not account_info['is_verified']:
        risk += 0.2
    if not account_info['phone_verified']:
        risk += 0.2
    # Multiple IPs can be a risk factor
    if len(account_info['ips']) > 2:
        risk += 0.1
    return min(round(random.uniform(0.1, 0.3) + risk, 2), 1.0)

# Enhanced account generation with more realistic features
NUM_ACCOUNTS = 4000
print(f"Generating {NUM_ACCOUNTS} accounts...")
accounts_data = []

for _ in range(NUM_ACCOUNTS):
    account_type = random.choice(['individual', 'business'])
    
    account_id = generate_cpf() if account_type == 'individual' else generate_cnpj()
    
    # Generate multiple devices and IPs per account (realistic for modern users)
    num_devices = random.randint(1, 3)
    devices = [str(uuid.uuid4()) for _ in range(num_devices)]
    ips = [generate_ip_address() for _ in range(num_devices)]

    # Account creation date affects fraud risk
    creation_date = fake.date_time_between(start_date='-3y', end_date='now')
    
    # Temporary dict to hold data for risk calculation
    temp_account_info = {
        'account_age_days': (datetime.now() - creation_date).days,
        'is_verified': random.choice([True, False]),
        'phone_verified': random.choice([True, False]),
        'ips': ips
    }

    # Enhanced account features for better fraud detection
    accounts_data.append({
        'account_id': account_id,
        'holder_name': fake.name() if account_type == 'individual' else fake.company(),
        'account_type': account_type,
        'creation_date': creation_date.isoformat(),
        'account_age_days': temp_account_info['account_age_days'],
        'state': fake.state_abbr(),
        'city': fake.city(),
        'is_verified': temp_account_info['is_verified'],
        'phone_verified': temp_account_info['phone_verified'],
        'risk_score': calculate_initial_risk(temp_account_info),
        'devices': '|'.join(devices),  # Store as pipe-separated string for CSV
        'ips': '|'.join(ips)  # Store as pipe-separated string for CSV
    })

accounts_df = pd.DataFrame(accounts_data)
accounts_df.to_csv('pix_accounts.csv', index=False, mode='a')
print("Accounts generated and saved to pix_accounts.csv.")

# Transaction generation with time-based patterns and realistic fraud rates
transactions = []
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

def get_account_details(df, account_id):
    """Helper to get devices and IPs for a given account."""
    account_info = df[df['account_id'] == account_id].iloc[0]
    devices = account_info['devices'].split('|')
    ips = account_info['ips'].split('|')
    return random.choice(devices), random.choice(ips)

def create_normal_transaction(accounts_df, date):
    """Simulates a simple, everyday transaction between any two accounts."""
    sender_id, receiver_id = np.random.choice(accounts_df['account_id'], 2, replace=False)
    device_id, ip_address = get_account_details(accounts_df, sender_id)
    amount = round(np.random.uniform(5.0, 500.0), 2)
    
    sender_info = accounts_df[accounts_df['account_id'] == sender_id].iloc[0]
    receiver_info = accounts_df[accounts_df['account_id'] == receiver_id].iloc[0]
    
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
    """Simulates a smurfing ring with realistic fraud patterns."""
    ring_transactions = []
    
    high_risk_accounts = accounts_df[accounts_df['risk_score'] > 0.5]
    if len(high_risk_accounts) < 5:
        return []

    victim_id = np.random.choice(accounts_df['account_id'])
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

    stolen_amount = round(np.random.uniform(10000.0, 50000.0), 2)
    transaction_time = date + timedelta(minutes=np.random.randint(1, 5))
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

    if not secondary_mules.any():
        return ring_transactions

    amount_per_mule = round(stolen_amount / len(secondary_mules), 2)
    for mule in secondary_mules:
        transaction_time += timedelta(seconds=np.random.randint(30, 120))
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

    base_amount = round(np.random.uniform(1000.0, 5000.0), 2)
    transaction_time = date
    
    for i in range(num_in_cycle):
        sender_id = cycle_accounts[i]
        receiver_id = cycle_accounts[(i + 1) % num_in_cycle]
        transaction_time += timedelta(minutes=np.random.randint(5, 30))
        
        sender_info = accounts_df[accounts_df['account_id'] == sender_id].iloc[0]
        receiver_info = accounts_df[accounts_df['account_id'] == receiver_id].iloc[0]

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

# Generate transactions with realistic patterns and fraud rates
TOTAL_TRANSACTIONS = 40000
current_date = start_date

print(f"Generating {TOTAL_TRANSACTIONS} transactions... This is going to take a few seconds.")
while len(transactions) < TOTAL_TRANSACTIONS:
    current_date += timedelta(minutes=np.random.randint(1, 30))
    if current_date > end_date:
        current_date = start_date

    hour = current_date.hour
    is_weekend = current_date.weekday() >= 5
    is_business_hour = 9 <= hour <= 17
    is_late_night = 0 <= hour <= 4

    transaction_types = [
        'salary', 
        'b2b', 
        'micro', 
        'smurfing', 
        'circular', 
        'normal'
    ]
    
    weights = [
        0.0,    
        0.10,   
        0.25,   
        0.0005, 
        0.0005, 
        0.65    
    ]

    if current_date.day in [5, 20]:
        weights[0] = 0.8  
        weights[1] = 0.01 
        weights[2] = 0.05
        weights[5] = 0.14
    
    if not is_business_hour:
        weights[1] = 0.0

    if is_late_night or is_weekend:
        weights[3] *= 3  
        weights[4] *= 3  

    total_weight = sum(weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        normalized_weights = [0] * len(weights)
        normalized_weights[-1] = 1.0 

    chosen_type = np.random.choice(transaction_types, p=normalized_weights)

    if chosen_type == 'salary':
        transactions.extend(create_salary_payments(accounts_df, current_date))
    elif chosen_type == 'b2b':
        transactions.append(create_business_transaction(accounts_df, current_date))
    elif chosen_type == 'micro':
        transactions.append(create_microtransaction(accounts_df, current_date))
    elif chosen_type == 'smurfing':
        transactions.extend(create_smurfing_ring(accounts_df, current_date))
    elif chosen_type == 'circular':
        transactions.extend(create_circular_payment_ring(accounts_df, current_date))
    else: 
        transactions.append(create_normal_transaction(accounts_df, current_date))
        
    if len(transactions) % 10000 == 0 and len(transactions) > 0:
        print(f"  ... {len(transactions)} transactions generated.")

print("Generation complete. Preparing final CSV...")
transactions_df = pd.DataFrame(transactions)
transactions_df = transactions_df.sample(frac=1).reset_index(drop=True)  
transactions_df.to_csv('pix_transactions.csv', index=False, mode='a')

print(f"Successfully created pix_transactions.csv with {len(transactions_df)} rows.")

fraud_counts = transactions_df['fraud_flag'].value_counts()
normal_flags = ['NORMAL', 'NORMAL_MICRO', 'NORMAL_SALARY', 'NORMAL_B2B']
total_fraud = sum(count for flag, count in fraud_counts.items() if flag not in normal_flags)
fraud_rate = (total_fraud / len(transactions_df)) * 100

print(f"\nDataset Summary:")
print(f"Total transactions: {len(transactions_df):,}")
print(f"Total accounts: {len(accounts_df):,}")
print(f"Fraud rate: {fraud_rate:.2f}%")
print(f"Transaction types:")
for flag, count in fraud_counts.items():
    print(f"  {flag}: {count:,} ({count/len(transactions_df)*100:.1f}%)")