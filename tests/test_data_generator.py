import unittest
import pandas as pd
from data_generator import (
    generate_cpf,
    generate_cnpj,
    calculate_initial_risk,
    create_normal_transaction,
    create_smurfing_ring,
    create_circular_payment_ring,
)

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        """Set up a dummy accounts DataFrame for testing."""
        self.accounts_df = pd.DataFrame({
            'account_id': [generate_cpf() for _ in range(10)],
            'risk_score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'devices': ['dev1|dev2'] * 10,
            'ips': ['ip1|ip2'] * 10,
            'account_type': ['individual'] * 10,
            'is_verified': [True] * 10,
            'state': ['SP'] * 10,
        })
        self.accounts_df = self.accounts_df.set_index('account_id', drop=False)

    def test_generate_cpf(self):
        """Test if the generated CPF has a valid format."""
        cpf = generate_cpf()
        self.assertRegex(cpf, r'\d{3}\.\d{3}\.\d{3}-\d{2}')

    def test_generate_cnpj(self):
        """Test if the generated CNPJ has a valid format."""
        cnpj = generate_cnpj()
        self.assertRegex(cnpj, r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}')

    def test_calculate_initial_risk(self):
        """Test the initial risk calculation logic."""
        account_info = {
            'account_age_days': 15,
            'is_verified': False,
            'phone_verified': False,
            'ips': ['ip1', 'ip2', 'ip3']
        }
        risk = calculate_initial_risk(account_info)
        self.assertGreater(risk, 0.5)

    def test_create_normal_transaction(self):
        """Test the creation of a normal transaction."""
        transaction = create_normal_transaction(self.accounts_df, pd.Timestamp.now())
        self.assertEqual(transaction['fraud_flag'], 'NORMAL')
        self.assertIn(transaction['sender_id'], self.accounts_df['account_id'].values)
        self.assertIn(transaction['receiver_id'], self.accounts_df['account_id'].values)

    def test_create_smurfing_ring(self):
        """Test the creation of a smurfing ring."""
        transactions = create_smurfing_ring(self.accounts_df, pd.Timestamp.now())
        self.assertGreater(len(transactions), 1)
        self.assertTrue(all(t['fraud_flag'].startswith('SMURFING') for t in transactions))
        self.assertTrue(all(t['sender_id'] != t['receiver_id'] for t in transactions))
        self.assertTrue(all(t['sender_id'] in self.accounts_df['account_id'].values for t in transactions))
        self.assertTrue(all(t['receiver_id'] in self.accounts_df['account_id'].values for t in transactions))
        self.assertTrue(all(t['amount'] > 0 for t in transactions))

    def test_create_circular_payment_ring(self):
        """Test the creation of a circular payment ring."""
        transactions = create_circular_payment_ring(self.accounts_df, pd.Timestamp.now())
        self.assertGreater(len(transactions), 1)
        self.assertTrue(all(t['fraud_flag'] == 'CIRCULAR_PAYMENT' or t['fraud_flag'] == 'NORMAL' for t in transactions))
        self.assertTrue(all(t['sender_id'] != t['receiver_id'] for t in transactions))
        self.assertTrue(all(t['sender_id'] in self.accounts_df['account_id'].values for t in transactions))
        self.assertTrue(all(t['receiver_id'] in self.accounts_df['account_id'].values for t in transactions))
        self.assertTrue(all(t['amount'] > 0 for t in transactions))

if __name__ == '__main__':
    unittest.main()
