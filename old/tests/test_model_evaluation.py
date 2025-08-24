import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from model_evaluation import (
    load_analysis_results,
    evaluate_community_fraud_detection,
    evaluate_lof_fraud_detection,
    evaluate_global_anomaly_detection,
    create_fraud_detection_summary,
    save_evaluation_metrics,
    main as evaluation_main,
)

class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        """Set up sample dataframes for testing."""
        self.community_df = pd.DataFrame({
            'communityId': [0, 1, 2],
            'communitySize': [10, 5, 8],
            'totalFraudTransactions': [5, 0, 2],
            'fraudRate': [0.5, 0.0, 0.25],
            'totalAmount': [10000, 5000, 8000]
        })
        self.lof_df = pd.DataFrame({
            'accountId': [f'acc_{i}' for i in range(5)],
            'fraudulentTransactions': [1, 0, 1, 0, 0],
            'is_lof_outlier': [1, 1, 0, 0, 1],
            'lof_score': [-2.5, -1.1, -0.9, -1.0, -2.0]
        })
        self.anomaly_df = pd.DataFrame({
            'accountId': [f'acc_{i}' for i in range(5)],
            'anomaly_score': [-0.2, -0.1, 0.1, 0.0, -0.3],
            'is_outlier': [1, 1, 0, 0, 1]
        })
        self.eval_df = pd.DataFrame({
            'accountId': [f'acc_{i}' for i in range(5)],
            'ground_truth_fraud': [1, 0, 1, 0, 1]
        })

    @patch('model_evaluation.os.path.exists')
    @patch('model_evaluation.pd.read_csv')
    def test_load_analysis_results(self, mock_read_csv, mock_exists):
        """Test loading of analysis result files."""
        mock_exists.return_value = True
        mock_read_csv.side_effect = [self.community_df, self.lof_df, self.anomaly_df]
        
        results = load_analysis_results()
        
        self.assertEqual(mock_read_csv.call_count, 3)
        self.assertIn('community_analysis', results)
        self.assertIn('lof_analysis', results)
        self.assertIn('anomaly_scores', results)
        self.assertTrue(results['community_analysis'].equals(self.community_df))

    def test_evaluate_community_fraud_detection(self):
        """Test the evaluation of community detection results."""
        with patch('builtins.print'): # Suppress print
            results = evaluate_community_fraud_detection(self.community_df)
        
        self.assertEqual(results['total_communities'], 3)
        self.assertEqual(results['fraud_communities'], 2)
        self.assertAlmostEqual(results['fraud_community_rate'], 2/3)

    def test_evaluate_lof_fraud_detection(self):
        """Test the evaluation of LOF results."""
        with patch('builtins.print'): # Suppress print
            results = evaluate_lof_fraud_detection(self.lof_df)
        
        self.assertIn('classification_report', results)
        self.assertIn('confusion_matrix', results)
        # TP=1, FP=2, FN=1, TN=1
        # The key for the fraud class is '1' when using output_dict=True
        self.assertEqual(results['classification_report']['1']['precision'], 1/3)
        self.assertEqual(results['classification_report']['1']['recall'], 1/2)

    @patch('model_evaluation.pd.read_csv')
    def test_evaluate_global_anomaly_detection(self, mock_read_csv):
        """Test the evaluation of global anomaly detection scores."""
        mock_read_csv.return_value = self.eval_df
        
        with patch('builtins.print'): # Suppress print
            results = evaluate_global_anomaly_detection(self.anomaly_df)
        
        mock_read_csv.assert_called_once_with("./data/evaluation_dataset.csv")
        self.assertIn('classification_report', results)
        self.assertIn('roc_auc', results)
        # is_outlier: [1, 1, 0, 0, 1], ground_truth: [1, 0, 1, 0, 1]
        # TP=2, FP=1, FN=1, TN=1
        # The key for the fraud class is '1'
        self.assertEqual(results['classification_report']['1']['recall'], 2/3)
        self.assertGreater(results['roc_auc'], 0)

    @patch('model_evaluation.evaluate_global_anomaly_detection')
    @patch('model_evaluation.evaluate_lof_fraud_detection')
    @patch('model_evaluation.evaluate_community_fraud_detection')
    def test_create_fraud_detection_summary(self, mock_eval_comm, mock_eval_lof, mock_eval_global):
        """Test the creation of the summary report."""
        mock_eval_comm.return_value = {'fraud_communities': 2, 'total_communities': 3}
        mock_eval_lof.return_value = {'classification_report': {'weighted avg': {'f1-score': 0.5}}}
        mock_eval_global.return_value = {'roc_auc': 0.8}
        
        analysis_results = {
            'community_analysis': self.community_df,
            'lof_analysis': self.lof_df,
            'anomaly_scores': self.anomaly_df
        }
        
        with patch('builtins.print'): # Suppress print
            summary = create_fraud_detection_summary(analysis_results)
            
        mock_eval_comm.assert_called_once()
        mock_eval_lof.assert_called_once()
        mock_eval_global.assert_called_once()
        self.assertIn('community', summary)
        self.assertIn('lof', summary)
        self.assertIn('global_anomaly', summary)

    @patch('model_evaluation.pd.DataFrame.to_csv')
    def test_save_evaluation_metrics(self, mock_to_csv):
        """Test saving the evaluation metrics to a CSV."""
        summary = {
            'community': {'fraud_community_rate': 0.66},
            'lof': {'classification_report': {'weighted avg': {'f1-score': 0.5}}},
            'global_anomaly': {'roc_auc': 0.8}
        }
        
        with patch('builtins.print'): # Suppress print
            save_evaluation_metrics(summary)
            
        mock_to_csv.assert_called_once()
        # Check the first argument (path) of the call
        self.assertEqual(mock_to_csv.call_args[0][0], "./data/fraud_detection_metrics.csv")

    @patch('model_evaluation.save_evaluation_metrics')
    @patch('model_evaluation.create_fraud_detection_summary')
    @patch('model_evaluation.load_analysis_results')
    def test_main_flow(self, mock_load, mock_create_summary, mock_save):
        """Test the main execution flow of the script."""
        mock_load.return_value = {'community_analysis': self.community_df}
        mock_create_summary.return_value = {'community': {'fraud_community_rate': 0.66}}
        
        evaluation_main()
        
        mock_load.assert_called_once()
        mock_create_summary.assert_called_once()
        mock_save.assert_called_once()

if __name__ == '__main__':
    unittest.main()
