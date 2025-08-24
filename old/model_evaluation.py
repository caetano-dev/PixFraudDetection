import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
import os

def load_analysis_results():
    """Load all analysis results for evaluation."""
    results = {}
    
    # Load community analysis
    if os.path.exists("./data/community_analysis.csv"):
        results['community_analysis'] = pd.read_csv("./data/community_analysis.csv")
    
    # Load LOF analysis
    if os.path.exists("./data/lof_analysis.csv"):
        results['lof_analysis'] = pd.read_csv("./data/lof_analysis.csv")
    
    # Load anomaly detection results
    if os.path.exists("./data/anomaly_scores.csv"):
        results['anomaly_scores'] = pd.read_csv("./data/anomaly_scores.csv")

    return results

def evaluate_community_fraud_detection(community_df):
    """Evaluate how well community detection identifies fraud rings."""
    print("="*60)
    print("COMMUNITY DETECTION FRAUD EVALUATION")
    print("="*60)
    
    if community_df is None or community_df.empty:
        print("No community analysis data available.")
        return None
    
    # Calculate community-level fraud metrics
    total_communities = len(community_df)
    fraud_communities = len(community_df[community_df['totalFraudTransactions'] > 0])
    
    print(f"Total communities detected: {total_communities}")
    print(f"Communities with fraud: {fraud_communities}")
    print(f"Fraud community rate: {fraud_communities/total_communities:.2%}")
    
    # Top fraud communities
    top_fraud_communities = community_df.nlargest(10, 'fraudRate')
    print(f"\nTop 10 Communities by Fraud Rate:")
    print(top_fraud_communities[['communityId', 'communitySize', 'fraudRate', 'totalFraudTransactions', 'totalAmount']].to_string(index=False))
    
    return {
        'total_communities': total_communities,
        'fraud_communities': fraud_communities,
        'fraud_community_rate': fraud_communities/total_communities
    }

def evaluate_lof_fraud_detection(lof_df):
    """Evaluate LOF fraud detection performance."""
    print("\n" + "="*60)
    print("LOCAL OUTLIER FACTOR FRAUD EVALUATION")
    print("="*60)
    
    if lof_df is None or lof_df.empty:
        print("No LOF analysis data available.")
        return None
    
    if 'fraudulentTransactions' not in lof_df.columns:
        print("Fraud transaction data not available in LOF results.")
        return None
    
    # Create binary fraud labels
    has_fraud = (lof_df['fraudulentTransactions'] > 0).astype(int)
    is_outlier = lof_df['is_lof_outlier']
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Classification Report:")
    print(classification_report(has_fraud, is_outlier, target_names=['Normal', 'Fraud']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(has_fraud, is_outlier)
    print(f"                 Predicted")
    print(f"              Normal  Outlier")
    print(f"Actual Normal   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"Actual Fraud    {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    # LOF score distribution analysis
    fraud_scores = lof_df[lof_df['fraudulentTransactions'] > 0]['lof_score']
    normal_scores = lof_df[lof_df['fraudulentTransactions'] == 0]['lof_score']
    
    print(f"\nLOF Score Distribution:")
    print(f"Fraud accounts - Mean: {fraud_scores.mean():.3f}, Std: {fraud_scores.std():.3f}")
    print(f"Normal accounts - Mean: {normal_scores.mean():.3f}, Std: {normal_scores.std():.3f}")
    
    # Top LOF outliers with fraud information
    top_outliers = lof_df.nsmallest(20, 'lof_score')
    fraud_in_top = (top_outliers['fraudulentTransactions'] > 0).sum()
    
    print(f"\nTop 20 LOF Outliers Analysis:")
    print(f"Fraud accounts in top 20: {fraud_in_top}/20 ({fraud_in_top/20:.1%})")
    
    return {
        'classification_report': classification_report(has_fraud, is_outlier, output_dict=True),
        'confusion_matrix': cm,
        'fraud_score_stats': {
            'fraud_mean': fraud_scores.mean(),
            'fraud_std': fraud_scores.std(),
            'normal_mean': normal_scores.mean(),
            'normal_std': normal_scores.std()
        },
        'top_outliers_fraud_rate': fraud_in_top/20
    }

def evaluate_global_anomaly_detection(anomaly_df):
    roc_auc = None
    """Evaluate global anomaly detection performance."""
    print("\n" + "="*60)
    print("GLOBAL ANOMALY DETECTION EVALUATION")
    print("="*60)
    
    if anomaly_df is None or anomaly_df.empty:
        print("No anomaly detection data available.")
        return None
    
    # We need to load the evaluation dataset to get ground truth
    try:
        eval_df = pd.read_csv("./data/evaluation_dataset.csv")
        print("Using evaluation dataset with ground truth labels.")
        
        # Merge with anomaly scores
        merged_df = eval_df.merge(anomaly_df[['accountId', 'anomaly_score', 'is_outlier']], 
                                 on='accountId', how='inner')
        
        if 'ground_truth_fraud' in merged_df.columns:
            has_fraud = merged_df['ground_truth_fraud']
            is_outlier = merged_df['is_outlier']
            
            print("Classification Report:")
            print(classification_report(has_fraud, is_outlier, target_names=['Normal', 'Fraud']))
            
            # ROC AUC if we have continuous scores
            if 'anomaly_score' in merged_df.columns:
                # For anomaly scores, lower scores indicate more anomalous behavior
                # So we need to invert the scores for ROC calculation
                roc_auc = roc_auc_score(has_fraud, -merged_df['anomaly_score'])
                print(f"\nROC AUC Score: {roc_auc:.3f}")
            
            return {
                'classification_report': classification_report(has_fraud, is_outlier, output_dict=True),
                'roc_auc': roc_auc if 'anomaly_score' in merged_df.columns else None
            }
    except FileNotFoundError:
        print("Evaluation dataset not found. Cannot compute ground truth metrics.")
        return None

def create_fraud_detection_summary(results):
    """Create a comprehensive fraud detection summary."""
    print("\n" + "="*80)
    print("COMPREHENSIVE FRAUD DETECTION PERFORMANCE SUMMARY")
    print("="*80)
    
    summary = {}
    
    # Community detection summary
    if 'community_analysis' in results:
        community_eval = evaluate_community_fraud_detection(results['community_analysis'])
        if community_eval:
            summary['community'] = community_eval
    
    # LOF summary
    if 'lof_analysis' in results:
        lof_eval = evaluate_lof_fraud_detection(results['lof_analysis'])
        if lof_eval:
            summary['lof'] = lof_eval
    
    # Global anomaly detection summary
    if 'anomaly_scores' in results:
        anomaly_eval = evaluate_global_anomaly_detection(results['anomaly_scores'])
        if anomaly_eval:
            summary['global_anomaly'] = anomaly_eval
    
    # Overall summary
    print(f"\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    
    if 'community' in summary:
        print(f"• Community Detection: {summary['community']['fraud_communities']} fraud communities out of {summary['community']['total_communities']} total")
    
    if 'lof' in summary:
        lof_f1 = summary['lof']['classification_report']['weighted avg']['f1-score']
        print(f"• Local Outlier Factor: F1-Score = {lof_f1:.3f}")
    
    if 'global_anomaly' in summary:
        if summary['global_anomaly']['roc_auc']:
            print(f"• Global Anomaly Detection: ROC AUC = {summary['global_anomaly']['roc_auc']:.3f}")
    
    print(f"\nNote: All models use only production-ready features (no fraud flags in training)")
    print(f"Fraud flags are used only for post-hoc evaluation and dashboard insights")
    
    return summary

def save_evaluation_metrics(summary):
    """Save evaluation metrics to CSV for dashboard consumption."""
    if not summary:
        return
    
    # Flatten the summary for CSV export
    metrics_data = []
    
    if 'community' in summary:
        metrics_data.append({
            'model': 'Community Detection',
            'metric': 'Fraud Community Rate',
            'value': summary['community']['fraud_community_rate'],
            'description': 'Percentage of communities containing fraudulent activity'
        })
    
    if 'lof' in summary:
        lof_metrics = summary['lof']['classification_report']['weighted avg']
        for metric_name, value in lof_metrics.items():
            if isinstance(value, (int, float)):
                metrics_data.append({
                    'model': 'Local Outlier Factor',
                    'metric': metric_name.replace('-', '_'),
                    'value': value,
                    'description': f'LOF {metric_name} score'
                })
    
    if 'global_anomaly' in summary and summary['global_anomaly']['roc_auc']:
        metrics_data.append({
            'model': 'Global Anomaly Detection',
            'metric': 'roc_auc',
            'value': summary['global_anomaly']['roc_auc'],
            'description': 'Area under ROC curve'
        })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv("./data/fraud_detection_metrics.csv", index=False)
        print(f"\nEvaluation metrics saved to fraud_detection_metrics.csv")

def main():
    """Main evaluation function."""
    print("PIX Fraud Detection System - Model Evaluation")
    print("="*60)
    
    # Load all analysis results
    results = load_analysis_results()
    
    if not results:
        print("No analysis results found. Please run the following scripts first:")
        print("1. python community_detection.py")
        print("2. python local_outlier_factor.py") 
        print("3. python feature_engineering.py")
        print("4. python anomaly_detection.py")
        return
    
    # Perform comprehensive evaluation
    summary = create_fraud_detection_summary(results)
    
    # Save metrics for dashboard
    save_evaluation_metrics(summary)
    
    print(f"\nEvaluation complete! Check fraud_detection_metrics.csv for dashboard integration.")

if __name__ == "__main__":
    main()
