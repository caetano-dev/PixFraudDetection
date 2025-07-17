import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from local_outlier_factor import (
    get_community_features,
    apply_lof_within_communities,
    analyze_lof_results,
    evaluate_fraud_detection_performance,
    update_neo4j_with_lof_results,
    main as lof_main,
    Neo4jConnection,
)

class TestLocalOutlierFactor(unittest.TestCase):

    def setUp(self):
        """Set up common test data and mocks."""
        self.mock_conn = MagicMock(spec=Neo4jConnection)
        
        # Sample DataFrame with multiple communities and an outlier
        self.features_df = pd.DataFrame({
            'accountId': [f'acc_{i}' for i in range(12)],
            'community': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
            'riskScore': np.random.rand(12),
            'isVerified': np.random.randint(0, 2, 12),
            'sentTransactions': np.random.randint(1, 10, 12),
            'receivedTransactions': np.random.randint(1, 10, 12),
            'totalTransactionAmount': np.random.uniform(100, 1000, 12),
            'avgSentAmount': np.random.uniform(10, 100, 12),
            'avgReceivedAmount': np.random.uniform(10, 100, 12),
            'maxSentAmount': np.random.uniform(100, 500, 12),
            'totalConnections': np.random.randint(1, 5, 12),
            'internalConnectionRatio': np.random.rand(12),
            'internalAmountRatio': np.random.rand(12),
            'fraudulentTransactions': [0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0]
        })
        # Add a clear outlier to community 1
        self.features_df.loc[9, 'totalTransactionAmount'] = 1_000_000
        self.features_df.loc[9, 'riskScore'] = 0.99

    def test_get_community_features(self):
        """Test fetching community features from Neo4j."""
        self.mock_conn.query.return_value = pd.DataFrame({'accountId': ['acc_1']})
        result_df = get_community_features(self.mock_conn)
        self.mock_conn.query.assert_called_once()
        self.assertIn("WHERE a.communityId IS NOT NULL", self.mock_conn.query.call_args[0][0])
        self.assertFalse(result_df.empty)

    def test_apply_lof_within_communities(self):
        """Test applying LOF within each community."""
        # Use a smaller n_neighbors for the test data
        result_df = apply_lof_within_communities(self.features_df, n_neighbors=3, contamination=0.2)
        
        self.assertIn('lof_score', result_df.columns)
        self.assertIn('is_lof_outlier', result_df.columns)
        
        # Check community 1 for the outlier
        community1_results = result_df[result_df['community'] == 1]
        self.assertEqual(community1_results['is_lof_outlier'].sum(), 1)

        # Check community 2 (too small) was skipped and has default values
        community2_results = result_df[result_df['community'] == 2]
        self.assertEqual(community2_results['is_lof_outlier'].sum(), 0)
        self.assertTrue((community2_results['lof_score'] == 0).all())

    @patch('local_outlier_factor.evaluate_fraud_detection_performance')
    def test_analyze_lof_results(self, mock_evaluate):
        """Test the analysis of LOF results."""
        lof_results = self.features_df.copy()
        lof_results['lof_score'] = np.random.rand(len(lof_results))
        lof_results['is_lof_outlier'] = 0
        lof_results.loc[9, 'is_lof_outlier'] = 1

        with patch('builtins.print'): # Suppress print output
            top_outliers, community_summary = analyze_lof_results(lof_results)

        mock_evaluate.assert_called_once_with(lof_results)
        self.assertEqual(len(top_outliers), 12) # Should return all accounts sorted
        self.assertEqual(len(community_summary), 3) # 3 unique communities
        self.assertEqual(community_summary.loc[community_summary['community'] == 1, 'outlier_count'].iloc[0], 1)

    def test_evaluate_fraud_detection_performance(self):
        """Test the fraud detection performance metrics calculation."""
        eval_df = pd.DataFrame({
            'fraudulentTransactions': [1, 0, 1, 0, 1, 0],
            'is_lof_outlier':         [1, 1, 0, 0, 1, 0],
            'community':              [0, 0, 1, 1, 2, 2], # Add community column
            'accountId':              [f'acc_{i}' for i in range(6)] # Add accountId column
        })
        # TP=2, FP=1, FN=1, TN=2
        with patch('builtins.print'): # Suppress print output
            metrics = evaluate_fraud_detection_performance(eval_df)
        
        self.assertAlmostEqual(metrics['precision'], 2/3)
        self.assertAlmostEqual(metrics['recall'], 2/3)
        self.assertEqual(metrics['true_positives'], 2)
        self.assertEqual(metrics['false_positives'], 1)

    def test_update_neo4j_with_lof_results(self):
        """Test updating Neo4j with LOF results."""
        lof_results = self.features_df[['accountId']].copy()
        lof_results['lof_score'] = 0.5
        lof_results['is_lof_outlier'] = 0
        
        update_neo4j_with_lof_results(self.mock_conn, lof_results)
        
        self.mock_conn.query.assert_called_once()
        query_arg = self.mock_conn.query.call_args[0][0]
        self.assertIn("UNWIND $batch AS row", query_arg)
        self.assertIn("SET a.lof_score = row.lof_score", query_arg)

    @patch('local_outlier_factor.update_neo4j_with_lof_results')
    @patch('local_outlier_factor.analyze_lof_results')
    @patch('local_outlier_factor.apply_lof_within_communities')
    @patch('local_outlier_factor.get_community_features')
    @patch('local_outlier_factor.Neo4jConnection')
    def test_main_flow(self, mock_conn_class, mock_get_features, mock_apply_lof, mock_analyze, mock_update):
        """Test the main execution flow."""
        # Mock setup
        mock_conn_class.return_value = self.mock_conn
        self.mock_conn.query.return_value = pd.DataFrame({'count': [10]}) # Community check passes
        mock_get_features.return_value = self.features_df
        mock_apply_lof.return_value = self.features_df
        mock_analyze.return_value = (pd.DataFrame(), pd.DataFrame())

        with patch('pandas.DataFrame.to_csv'): # Mock CSV saving
            lof_main()

        # Assertions
        self.mock_conn.query.assert_called_once_with("MATCH (a:Account) WHERE a.communityId IS NOT NULL RETURN count(a) as count")
        mock_get_features.assert_called_once()
        mock_apply_lof.assert_called_once()
        mock_analyze.assert_called_once()
        mock_update.assert_called_once()
        self.mock_conn.close.assert_called_once()

    @patch('local_outlier_factor.Neo4jConnection')
    def test_main_no_communities(self, mock_conn_class):
        """Test main flow when no communities are found."""
        mock_conn_class.return_value = self.mock_conn
        self.mock_conn.query.return_value = pd.DataFrame({'count': [0]}) # Community check fails

        with patch('builtins.print') as mock_print:
            lof_main()
            mock_print.assert_any_call("Error: No communities found. Please run community_detection.py first.")

if __name__ == '__main__':
    unittest.main()
