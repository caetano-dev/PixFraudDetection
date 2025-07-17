import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from anomaly_detection import (
    apply_isolation_forest,
    apply_local_outlier_factor,
    main as anomaly_main,
)

class TestAnomalyDetection(unittest.TestCase):

    def setUp(self):
        """Set up a dummy features DataFrame for testing."""
        # Create a dataset with a clear outlier
        self.features_list = ['totalTransactions', 'totalAmount']
        self.features_df = pd.DataFrame({
            'accountId': [f'acc_{i}' for i in range(10)],
            'totalTransactions': np.random.randint(5, 15, 10),
            'totalAmount': np.random.uniform(100, 1000, 10),
        })
        # Add an obvious outlier
        outlier = pd.DataFrame({
            'accountId': ['acc_10'],
            'totalTransactions': [1000],
            'totalAmount': [500000],
        })
        self.features_df = pd.concat([self.features_df, outlier], ignore_index=True)

    @patch('anomaly_detection.joblib.dump')
    def test_apply_isolation_forest(self, mock_joblib_dump):
        """Test the Isolation Forest model application."""
        result_df = apply_isolation_forest(self.features_df.copy(), self.features_list)
        
        self.assertIn('anomaly_score', result_df.columns)
        self.assertIn('is_outlier', result_df.columns)
        self.assertEqual(mock_joblib_dump.call_count, 2) # Model and scaler
        self.assertGreater(result_df['is_outlier'].sum(), 0)
        # The most anomalous should be our outlier
        self.assertEqual(result_df.sort_values('anomaly_score').iloc[0]['accountId'], 'acc_10')

    @patch('anomaly_detection.joblib.dump')
    def test_apply_local_outlier_factor(self, mock_joblib_dump):
        """Test the Local Outlier Factor model application."""
        # Increase contamination for a small dataset to ensure the outlier is caught
        result_df = apply_local_outlier_factor(self.features_df.copy(), self.features_list, contamination=0.2)
        
        self.assertIn('anomaly_score', result_df.columns)
        self.assertIn('is_outlier', result_df.columns)
        mock_joblib_dump.assert_called_once() # Only scaler is saved
        self.assertGreater(result_df['is_outlier'].sum(), 0)

    @patch('anomaly_detection.argparse.ArgumentParser.parse_args')
    @patch('anomaly_detection.pd.read_csv')
    @patch('anomaly_detection.apply_isolation_forest')
    @patch('anomaly_detection.pd.DataFrame.to_csv')
    def test_main_with_isolation_forest(self, mock_to_csv, mock_apply_if, mock_read_csv, mock_parse_args):
        """Test the main function with the isolation_forest algorithm."""
        mock_parse_args.return_value = MagicMock(algorithm='isolation_forest', no_save=True)
        mock_read_csv.return_value = self.features_df
        
        # The mock needs to return a dataframe with the columns the main function expects
        result_df = self.features_df.copy()
        result_df['anomaly_score'] = 0.1
        result_df['is_outlier'] = 0
        result_df.loc[10, 'is_outlier'] = 1
        mock_apply_if.return_value = result_df
        
        anomaly_main()
        
        mock_read_csv.assert_called_once()
        mock_apply_if.assert_called_once()
        mock_to_csv.assert_not_called()

    @patch('anomaly_detection.argparse.ArgumentParser.parse_args')
    @patch('anomaly_detection.pd.read_csv')
    @patch('anomaly_detection.apply_local_outlier_factor')
    @patch('anomaly_detection.pd.DataFrame.to_csv')
    def test_main_with_lof(self, mock_to_csv, mock_apply_lof, mock_read_csv, mock_parse_args):
        """Test the main function with the lof algorithm."""
        mock_parse_args.return_value = MagicMock(algorithm='lof', n_neighbors=5, contamination=0.1, no_save=True)
        mock_read_csv.return_value = self.features_df

        # The mock needs to return a dataframe with the columns the main function expects
        result_df = self.features_df.copy()
        result_df['anomaly_score'] = 1.5
        result_df['is_outlier'] = 0
        result_df.loc[10, 'is_outlier'] = 1
        mock_apply_lof.return_value = result_df
        
        anomaly_main()
        
        mock_read_csv.assert_called_once()
        mock_apply_lof.assert_called_once()
        mock_to_csv.assert_not_called()

if __name__ == '__main__':
    unittest.main()
