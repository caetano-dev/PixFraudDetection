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
        re = 11 - (s % 11)
        return str(re) if re < 10 else '0'
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

NUM_ACCOUNTS = 5000
print(f"Generating {NUM_ACCOUNTS} accounts...")
accounts_data = []
for _ in range(NUM_ACCOUNTS):
    account_type = random.choice(['individual', 'business'])
    
    account_id = generate_cpf() if account_type == 'individual' else generate_cnpj()
    
    num_devices = random.randint(1, 3)
    devices = [str(uuid.uuid4()) for _ in range(num_devices)]
    ips = [generate_ip_address() for _ in range(num_devices)]

    accounts_data.append({
        'account_id': account_id,
        'holder_name': fake.name() if account_type == 'individual' else fake.company(),
        'account_type': account_type,
        'creation_date': fake.date_time_between(start_date='-3y', end_date='now').isoformat(),
        'devices': devices, # Store as a list
        'ips': ips # Store as a list
    })

accounts_df = pd.DataFrame(accounts_data)
accounts_df.to_csv('pix_accounts.csv', index=False)
print("Accounts generated and saved to pix_accounts.csv.")


transactions = []
start_date = datetime(2025, 6, 1)
end_date = datetime(2025, 7, 5)

def get_account_details(df, account_id):
    """Helper to get devices and IPs for a given account."""
    account_info = df[df['account_id'] == account_id].iloc[0]
    return random.choice(account_info['devices']), random.choice(account_info['ips'])

def create_normal_transaction(accounts_df, date):
    """Simulates a simple, everyday transaction between any two accounts."""
    sender_id, receiver_id = np.random.choice(accounts_df['account_id'], 2, replace=False)
    device_id, ip_address = get_account_details(accounts_df, sender_id)
    amount = round(np.random.uniform(5.0, 500.0), 2)
    return {
        'transaction_id': str(uuid.uuid4()),
        'sender_id': sender_id,
        'receiver_id': receiver_id,
        'amount': amount,
        'timestamp': date.isoformat(),
        'fraud_flag': 'NORMAL',
        'device_id': device_id,
        'ip_address': ip_address
    }

def create_smurfing_ring(accounts_df, date):
    """Simulates a smurfing ring, now using any account type as a mule."""
    ring_transactions = []
    
    victim_id = np.random.choice(accounts_df['account_id'])
    mules = np.random.choice(accounts_df[accounts_df['account_id'] != victim_id]['account_id'], 11, replace=False)
    central_mule = mules[0]
    secondary_mules = mules[1:]

    # All mules in a fraud ring often share the same device/IP
    fraud_device, fraud_ip = str(uuid.uuid4()), generate_ip_address()

    # Victim transfer to central mule
    stolen_amount = round(np.random.uniform(10000.0, 40000.0), 2)
    transaction_time = date + timedelta(minutes=np.random.randint(1, 5))
    victim_device, victim_ip = get_account_details(accounts_df, victim_id)
    ring_transactions.append({
        'transaction_id': str(uuid.uuid4()),'sender_id': victim_id,'receiver_id': central_mule,
        'amount': stolen_amount,'timestamp': transaction_time.isoformat(),
        'fraud_flag': 'SMURFING_VICTIM_TRANSFER','device_id': victim_device,'ip_address': victim_ip
    })

    # Central mule distributes funds
    amount_per_mule = round(stolen_amount / len(secondary_mules), 2)
    for mule in secondary_mules:
        transaction_time += timedelta(seconds=np.random.randint(20, 90))
        ring_transactions.append({
            'transaction_id': str(uuid.uuid4()), 'sender_id': central_mule, 'receiver_id': mule,
            'amount': amount_per_mule * np.random.uniform(0.9, 1.1), 'timestamp': transaction_time.isoformat(),
            'fraud_flag': 'SMURFING_DISTRIBUTION', 'device_id': fraud_device, 'ip_address': fraud_ip
        })
    return ring_transactions

def create_salary_payments(accounts_df, date):
    """Simulates a business paying salaries to its employees."""
    salary_transactions = []
    # Find business and individual accounts
    businesses = accounts_df[accounts_df['account_type'] == 'business']
    individuals = accounts_df[accounts_df['account_type'] == 'individual']
    
    if len(businesses) == 0 or len(individuals) < 10:
        return [] # Not enough accounts to simulate this

    # Pick one company to be the employer
    employer_id = np.random.choice(businesses['account_id'])
    employer_device, employer_ip = get_account_details(accounts_df, employer_id)

    # Pay 10 to 50 employees
    num_employees = np.random.randint(10, 51)
    employees = np.random.choice(individuals['account_id'], num_employees, replace=False)

    for employee_id in employees:
        salary_amount = round(np.random.uniform(2500.0, 8000.0), 2)
        transaction_time = date + timedelta(minutes=np.random.randint(0, 120))
        salary_transactions.append({
            'transaction_id': str(uuid.uuid4()), 'sender_id': employer_id, 'receiver_id': employee_id,
            'amount': salary_amount, 'timestamp': transaction_time.isoformat(),
            'fraud_flag': 'NORMAL_SALARY', 'device_id': employer_device, 'ip_address': employer_ip
        })
    return salary_transactions

def create_circular_payment_ring(accounts_df, date):
    """
    Simulates a circular payment to build 'good' history.
    A -> B -> C -> D -> A
    """
    ring_transactions = []
    
    # Select 3 to 5 accounts for the cycle
    num_in_cycle = np.random.randint(3, 6)
    cycle_accounts = np.random.choice(accounts_df['holder_id'], num_in_cycle, replace=False)

    amount = round(np.random.uniform(1000.0, 5000.0), 2)
    transaction_time = date
    
    for i in range(num_in_cycle):
        sender = cycle_accounts[i]
        # The last one sends back to the first one to close the loop
        receiver = cycle_accounts[(i + 1) % num_in_cycle]
        transaction_time += timedelta(minutes=np.random.randint(5, 30))
        
        ring_transactions.append({
            'transaction_id': str(uuid.uuid4()),
            'sender_id': sender,
            'receiver_id': receiver,
            'amount': amount * np.random.uniform(0.95, 1.05), # Slight variations
            'timestamp': transaction_time.isoformat(),
            'fraud_flag': 'CIRCULAR_PAYMENT'
        })
        
    return ring_transactions


TOTAL_TRANSACTIONS = 100000
current_date = start_date

print(f"Generating {TOTAL_TRANSACTIONS} transactions...")
while len(transactions) < TOTAL_TRANSACTIONS:
    current_date += timedelta(minutes=np.random.randint(1, 30))
    if current_date > end_date:
        current_date = start_date

    rand_val = np.random.rand()
    
    # Introduce a specific day for salary payments for realism
    if current_date.day == 5 and np.random.rand() < 0.8: # High chance of salaries on day 5
         transactions.extend(create_salary_payments(accounts_df, current_date))
    elif rand_val < 0.01: # 1% chance of a smurfing event
        transactions.extend(create_smurfing_ring(accounts_df, current_date))
    elif rand_val < 0.02: # 2% chance of circular payment 
        transactions.extend(create_circular_payment_ring(accounts_df, current_date))
    else: # 98% chance of a normal transaction
        transactions.append(create_normal_transaction(accounts_df, current_date))
        
    if len(transactions) % 10000 == 0 and len(transactions) > 0:
        print(f"  ... {len(transactions)} transactions generated.")

print("Generation complete. Preparing final CSV...")
transactions_df = pd.DataFrame(transactions)
transactions_df = transactions_df.sample(frac=1).reset_index(drop=True)
transactions_df.to_csv('pix_transactions.csv', index=False)

print(f"Successfully created pix_transactions.csv with {len(transactions_df)} rows.")