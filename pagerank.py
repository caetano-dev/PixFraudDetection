import numpy as np
from numpy.random.mtrand import weibull
import pandas as pd
import networkx as nx
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve)
from typing import Optional
from buildGraph import main as build_graph_main

# tol = 1e-6 is the default in NetworkX's pagerank implementation
def compute_pagerank(G: nx.DiGraph, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6, personalization: Optional[dict] = None, weight: str = 'weight') -> dict:
    pagerank_scores = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol, personalization=personalization, weight=weight)
    return pagerank_scores

def rank_nodes_by_pagerank(pagerank_scores: dict, descending: bool = True) -> list:
    ranked = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=descending )
    return ranked

def get_node_labels(G: nx.DiGraph) -> dict:
    return {node: G.nodes[node].get('label', 0) for node in G.nodes()}

def evaluate_pagerank_effectiveness(ranked_nodes: list, labels: dict, k_values: list = [10, 50, 100, 500, 1000]) -> dict:
    # k_values: Values of k for precision@k and recall@k metrics
    metrics = {}
    # Total counts
    total_nodes = len(ranked_nodes)
    total_fraud = sum(labels.values())
    total_normal = total_nodes - total_fraud
    
    metrics['total_nodes'] = total_nodes
    metrics['total_fraud'] = total_fraud
    metrics['total_normal'] = total_normal
    metrics['fraud_rate'] = total_fraud / total_nodes if total_nodes > 0 else 0
    
    nodes_ordered = [node for node, _ in ranked_nodes]
    scores_ordered = np.array([score for _, score in ranked_nodes])
    labels_ordered = np.array([labels.get(node, 0) for node in nodes_ordered])
    
    metrics['precision_at_k'] = {}
    metrics['recall_at_k'] = {}
    metrics['fraud_found_at_k'] = {}
    
    for k in k_values:
        if k <= total_nodes:
            top_k_labels = labels_ordered[:k]
            fraud_in_top_k = np.sum(top_k_labels)
            
            precision_at_k = fraud_in_top_k / k
            recall_at_k = fraud_in_top_k / total_fraud if total_fraud > 0 else 0
            
            metrics['precision_at_k'][k] = precision_at_k
            metrics['recall_at_k'][k] = recall_at_k
            metrics['fraud_found_at_k'][k] = int(fraud_in_top_k)
    
    if total_fraud > 0 and total_normal > 0:
        try:
            metrics['roc_auc'] = roc_auc_score(labels_ordered, scores_ordered)
        except ValueError:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    
    # Area under precision-recall curve
    if total_fraud > 0:
        try:
            metrics['average_precision'] = average_precision_score(labels_ordered, scores_ordered)
        except ValueError:
            metrics['average_precision'] = None
    else:
        metrics['average_precision'] = None
    
    metrics['lift_at_k'] = {}
    baseline_rate = total_fraud / total_nodes if total_nodes > 0 else 0
    
    for k in k_values:
        if k <= total_nodes and baseline_rate > 0:
            precision_k = metrics['precision_at_k'].get(k, 0)
            lift = precision_k / baseline_rate
            metrics['lift_at_k'][k] = lift
    
    return metrics

def print_evaluation_report(metrics: dict, ranked_nodes: list, labels: dict):
    print("Pagerank")
    print("=" * 10)
    
    print("\n[Dataset Statistics]")
    print(f"  Total entities: {metrics['total_nodes']:,}")
    print(f"  Fraudulent entities: {metrics['total_fraud']:,}")
    print(f"  Normal entities: {metrics['total_normal']:,}")
    print(f"  Baseline fraud rate: {metrics['fraud_rate']:.4%}")
    
    print("\n[Precision@K] (Fraud rate in top-K ranked entities)")
    print("-" * 50)
    for k, precision in sorted(metrics['precision_at_k'].items()):
        fraud_found = metrics['fraud_found_at_k'].get(k, 0)
        lift = metrics['lift_at_k'].get(k, 0)
        print(f"  Top {k:>5}: {precision:.4%} precision | {fraud_found:>5} fraud found | {lift:.2f}x lift")
    
    print("\n[Recall@K] (Fraction of total fraud found in top-K)")
    print("-" * 50)
    for k, recall in sorted(metrics['recall_at_k'].items()):
        print(f"  Top {k:>5}: {recall:.4%} recall")
    
    print("\n[Global Metrics]")
    print("-" * 50)
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC Score: {metrics['roc_auc']:.4f}")
    else:
        print("  ROC-AUC Score: N/A (insufficient class diversity)")
    
    if metrics['average_precision'] is not None:
        print(f"  Average Precision (PR-AUC): {metrics['average_precision']:.4f}")
    else:
        print("  Average Precision: N/A")
    
    # Show top 20 ranked entities
    print("\n[Top 20 Ranked Entities by PageRank]")
    print("-" * 70)
    print(f"{'Rank':<6} {'Entity':<30} {'PageRank Score':<18} {'Is Fraud?':<10}")
    print("-" * 70)
    
    for i, (node, score) in enumerate(ranked_nodes[:20], 1):
        is_fraud = "YES" if labels.get(node, 0) == 1 else "NO"
        print(f"{i:<6} {str(node):<30} {score:<18.10f} {is_fraud:<10}")
    
    # Effectiveness assessment
    print("\n[Effectiveness Assessment]")
    print("-" * 50)
    
    baseline = metrics['fraud_rate']
    
    # Check if PageRank is better than random
    best_precision_k = max(metrics['precision_at_k'].values()) if metrics['precision_at_k'] else 0
    
    if best_precision_k > baseline * 1.5:
        print("    PageRank is way better than random selection")
        print(f"    Best precision@K ({best_precision_k:.4%}) is {best_precision_k/baseline:.2f}x better than baseline ({baseline:.4%})")
    elif best_precision_k > baseline:
        print("    PageRank shows improvement over random selection")
        print(f"    Best precision@K ({best_precision_k:.4%}) is {best_precision_k/baseline:.2f}x better than baseline ({baseline:.4%})")
    else:
        print("    PageRank is worse than random selection")
        print("    Consider using alternative methods or feature engineering")
    
    print("TODO: Ver quais valores fazem sentido comparar com o roc_auc...")
    if metrics['roc_auc'] is not None:
        if metrics['roc_auc'] > 0.7:
            print(f"    ROC-AUC of {metrics['roc_auc']:.4f} indicates GOOD discriminative ability")
        elif metrics['roc_auc'] > 0.5:
            print(f"    ROC-AUC of {metrics['roc_auc']:.4f} indicates WEAK discriminative ability")
        else:
            print(f"    ROC-AUC of {metrics['roc_auc']:.4f} indicates NO discriminative ability")


def run_pagerank_analysis(G: nx.DiGraph, alpha: float = 0.85, use_weights: bool = True) -> tuple[list, dict]:
    print("RUNNING PAGERANK ANALYSIS")
    print("=" * 70)
    
    print("\n[Step 1] Computing PageRank scores...")
    weight_param = 'weight' if use_weights else None
    print(weight_param)
    pagerank_scores = compute_pagerank(G, alpha=alpha, weight=weight_param)
    print(f"  Computed scores for {len(pagerank_scores):,} nodes")
    
    # Rank nodes
    print("\n[Step 2] Ranking nodes by PageRank score...")
    ranked_nodes = rank_nodes_by_pagerank(pagerank_scores, descending=True)
    
    # Get labels
    print("\n[Step 3] Extracting ground truth labels...")
    labels = get_node_labels(G)
    
    # Evaluate
    print("\n[Step 4] Evaluating effectiveness...")
    k_values = [10, 50, 100, 500, 1000, 5000]
    # Filter k_values to not exceed total nodes
    k_values = [k for k in k_values if k <= len(ranked_nodes)]
    
    metrics = evaluate_pagerank_effectiveness(ranked_nodes, labels, k_values)
    
    # Print report
    print_evaluation_report(metrics, ranked_nodes, labels)
    
    return ranked_nodes, metrics


def main():
    print("Building transaction graph...")
    G = build_graph_main()
    # Run PageRank analysis
    ranked_nodes, metrics = run_pagerank_analysis(G, alpha=0.85, use_weights=True)
    # Export results to CSV
    """
    print("\n[Exporting Results]")
    results_df = pd.DataFrame([
        {
            'rank': i + 1,
            'entity': node,
            'pagerank_score': score,
            'is_fraud': get_node_labels(G).get(node, 0)
        }
        for i, (node, score) in enumerate(ranked_nodes)
    ])
    
    results_df.to_csv("pagerank_rankings.csv", index=False)
    print("  Rankings exported to: pagerank_rankings.csv")
    """
    return ranked_nodes, metrics

if __name__ == "__main__":
    ranked_nodes, metrics = main()