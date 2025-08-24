import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from feature_engineering import (
    get_graph_features,
    get_transactional_features,
    get_money_flow_features,
    get_velocity_features,
    create_evaluation_dataset,
)

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        """Set up a mock Neo4j connection."""
        self.mock_conn = MagicMock()

    def test_get_graph_features(self):
        """Test the graph features extraction."""
        mock_df = pd.DataFrame([{'accountId': 'a1', 'totalDegree': 5}])
        self.mock_conn.query.return_value = mock_df
        
        df = get_graph_features(self.mock_conn)
        
        self.mock_conn.query.assert_called_once()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_get_transactional_features(self):
        """Test the transactional features extraction."""
        mock_df = pd.DataFrame([{'accountId': 'a1', 'totalTransactions': 10}])
        self.mock_conn.query.return_value = mock_df
        
        df = get_transactional_features(self.mock_conn)
        
        self.mock_conn.query.assert_called_once()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_get_money_flow_features(self):
        """Test the money flow features extraction."""
        mock_df = pd.DataFrame([{'accountId': 'a1', 'totalFlowAmount': 5000}])
        self.mock_conn.query.return_value = mock_df
        
        df = get_money_flow_features(self.mock_conn)
        
        self.mock_conn.query.assert_called_once()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_get_velocity_features(self):
        """Test the velocity features extraction."""
        mock_df = pd.DataFrame([{'accountId': 'a1', 'avgTimeBetweenTransactions': 3600}])
        self.mock_conn.query.return_value = mock_df
        
        df = get_velocity_features(self.mock_conn)
        
        self.mock_conn.query.assert_called_once()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    @patch('feature_engineering.pd.DataFrame.to_csv')
    def test_create_evaluation_dataset(self, mock_to_csv):
        """Test the creation of the evaluation dataset."""
        features_df = pd.DataFrame({'accountId': ['a1', 'a2'], 'feature1': [1, 2]})
        fraud_labels_df = pd.DataFrame({'accountId': ['a1'], 'isFraudulent': [1]})
        
        self.mock_conn.query.return_value = fraud_labels_df
        
        evaluation_df = create_evaluation_dataset(self.mock_conn, features_df)
        
        self.mock_conn.query.assert_called_once()
        mock_to_csv.assert_called_once()
        self.assertIn('isFraudulent', evaluation_df.columns)
        self.assertEqual(evaluation_df.loc[evaluation_df['accountId'] == 'a1', 'isFraudulent'].iloc[0], 1)
        self.assertEqual(evaluation_df.loc[evaluation_df['accountId'] == 'a2', 'isFraudulent'].iloc[0], 0)

if __name__ == '__main__':
    unittest.main()
