import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

CUTOFF_DATE = pd.Timestamp('2022-11-05')


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three parquet files containing transactions and account mappings."""
    data_path = Path(data_dir)
    
    normal_transactions = pd.read_parquet(data_path / "1_filtered_normal_transactions.parquet")
    laundering_transactions = pd.read_parquet(data_path / "2_filtered_laundering_transactions.parquet")
    accounts = pd.read_parquet(data_path / "3_filtered_accounts.parquet")
    
    print(f"Loaded {len(normal_transactions):,} normal transactions")
    print(f"Loaded {len(laundering_transactions):,} laundering transactions")
    print(f"Loaded {len(accounts):,} account mappings")
    
    return normal_transactions, laundering_transactions, accounts


def preprocess_transactions(
    normal_transactions: pd.DataFrame,
    laundering_transactions: pd.DataFrame,
    accounts: pd.DataFrame
) -> pd.DataFrame:
    """
    Concatenate transactions and map accounts to entities.
    
    Returns a DataFrame with source_entity and target_entity columns.
    """
    # Concatenate normal and laundering transactions
    all_transactions = pd.concat([normal_transactions, laundering_transactions], ignore_index=True)
    print(f"Total transactions after concatenation: {len(all_transactions):,}")
    
    # Convert timestamp to datetime
    all_transactions['timestamp'] = pd.to_datetime(all_transactions['timestamp'])
    
    # Filter out transactions after the cutoff date (November 5th)
    before_filter = len(all_transactions)
    all_transactions = all_transactions[all_transactions['timestamp'] <= CUTOFF_DATE]
    print(f"Transactions after filtering by cutoff date ({CUTOFF_DATE.date()}): {len(all_transactions):,} (removed {before_filter - len(all_transactions):,})")
    
    # Create account to entity mapping dictionary
    account_to_entity = accounts.set_index('Account Number')['Entity ID'].to_dict()
    
    # Map from_account -> source_entity
    all_transactions['source_entity'] = all_transactions['from_account'].map(account_to_entity)
    
    # Map to_account -> target_entity
    all_transactions['target_entity'] = all_transactions['to_account'].map(account_to_entity)
    
    # Report any unmapped accounts
    unmapped_source = all_transactions['source_entity'].isna().sum()
    unmapped_target = all_transactions['target_entity'].isna().sum()
    
    if unmapped_source > 0 or unmapped_target > 0:
        print(f"Warning: {unmapped_source:,} transactions have unmapped source accounts")
        print(f"Warning: {unmapped_target:,} transactions have unmapped target accounts")
        # Drop transactions with unmapped entities
        all_transactions = all_transactions.dropna(subset=['source_entity', 'target_entity'])
        print(f"Transactions after dropping unmapped: {len(all_transactions):,}")
    
    return all_transactions


def compute_edge_attributes(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactions between entity pairs to compute edge attributes.
    
    Returns a DataFrame with:
    - source_entity, target_entity (edge endpoints)
    - weight: total transaction volume (sum of amount_sent_c)
    - count: number of transactions (frequency)
    """
    edge_agg = transactions.groupby(['source_entity', 'target_entity']).agg(
        weight=('amount_sent_c', 'sum'),
        count=('amount_sent_c', 'count')
    ).reset_index()
    
    print(f"Aggregated into {len(edge_agg):,} unique edges")
    return edge_agg


def compute_entity_labels(transactions: pd.DataFrame) -> dict:
    """
    Compute entity labels: 1 if involved in any laundering transaction, else 0.
    """
    laundering_txns = transactions[transactions['is_laundering'] == 1]
    
    laundering_sources = set(laundering_txns['source_entity'].unique())
    laundering_targets = set(laundering_txns['target_entity'].unique())
    laundering_entities = laundering_sources | laundering_targets
    
    all_entities = set(transactions['source_entity'].unique()) | set(transactions['target_entity'].unique())
    
    labels = {entity: 1 if entity in laundering_entities else 0 for entity in all_entities}
    
    num_laundering = sum(labels.values())
    print(f"Entities involved in laundering: {num_laundering:,} / {len(labels):,}")
    
    return labels


def compute_temporal_statistics(transactions: pd.DataFrame) -> dict:
    """
    Compute temporal statistics for each entity based on their transaction history.
    
    For each entity, calculates statistics on amounts sent and received:
    - mean, max, min, std for sent amounts
    - mean, max, min, std for received amounts
    """
    entity_stats = {}
    
    # Statistics for amounts sent (entity as source)
    sent_stats = transactions.groupby('source_entity')['amount_sent_c'].agg(
        ['mean', 'max', 'min', 'std']
    ).rename(columns={
        'mean': 'sent_amount_mean',
        'max': 'sent_amount_max',
        'min': 'sent_amount_min',
        'std': 'sent_amount_std'
    })
    
    # Statistics for amounts received (entity as target)
    received_stats = transactions.groupby('target_entity')['amount_sent_c'].agg(
        ['mean', 'max', 'min', 'std']
    ).rename(columns={
        'mean': 'received_amount_mean',
        'max': 'received_amount_max',
        'min': 'received_amount_min',
        'std': 'received_amount_std'
    })
    
    # Get all unique entities
    all_entities = set(transactions['source_entity'].unique()) | set(transactions['target_entity'].unique())
    
    # Build stats dictionary for each entity
    for entity in all_entities:
        stats = {}
        
        # Sent statistics
        if entity in sent_stats.index:
            for col in sent_stats.columns:
                stats[col] = sent_stats.loc[entity, col]
        else:
            stats['sent_amount_mean'] = 0.0
            stats['sent_amount_max'] = 0.0
            stats['sent_amount_min'] = 0.0
            stats['sent_amount_std'] = 0.0
        
        # Received statistics
        if entity in received_stats.index:
            for col in received_stats.columns:
                stats[col] = received_stats.loc[entity, col]
        else:
            stats['received_amount_mean'] = 0.0
            stats['received_amount_max'] = 0.0
            stats['received_amount_min'] = 0.0
            stats['received_amount_std'] = 0.0
        
        for key in stats:
          # accounts that have only one transaction will have NaN std
            if pd.isna(stats[key]):
                stats[key] = 0.0
        
        entity_stats[entity] = stats
    
    return entity_stats


def build_graph(
    edge_attributes: pd.DataFrame,
    entity_labels: dict,
    temporal_stats: dict
) -> nx.DiGraph:
    """
    Construct the directed weighted graph with all node and edge attributes.
    """
    G = nx.DiGraph()
    
    # Add edges with attributes
    for _, row in edge_attributes.iterrows():
        G.add_edge(
            row['source_entity'],
            row['target_entity'],
            weight=row['weight'],
            count=row['count']
        )
    
    # Compute structural features for each node
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # Compute in_strength and out_strength (weighted degrees)
    in_strength = {}
    out_strength = {}
    
    for node in G.nodes():
        # Sum of incoming edge weights
        in_strength[node] = sum(
            G[pred][node]['weight'] for pred in G.predecessors(node)
        )
        # Sum of outgoing edge weights
        out_strength[node] = sum(
            G[node][succ]['weight'] for succ in G.successors(node)
        )
    
    # Set node attributes
    for node in G.nodes():
        # Inherent attributes
        G.nodes[node]['label'] = entity_labels.get(node, 0)
        
        # Structural/Flow attributes
        G.nodes[node]['in_degree'] = in_degrees[node]
        G.nodes[node]['out_degree'] = out_degrees[node]
        G.nodes[node]['in_strength'] = in_strength[node]
        G.nodes[node]['out_strength'] = out_strength[node]
        
        # Average amounts (handle division by zero)
        G.nodes[node]['avg_in_amount'] = (
            in_strength[node] / in_degrees[node] if in_degrees[node] > 0 else 0.0
        )
        G.nodes[node]['avg_out_amount'] = (
            out_strength[node] / out_degrees[node] if out_degrees[node] > 0 else 0.0
        )
        
        # Temporal statistics
        if node in temporal_stats:
            for stat_name, stat_value in temporal_stats[node].items():
                G.nodes[node][stat_name] = stat_value
        else:
            # Default temporal stats
            G.nodes[node]['sent_amount_mean'] = 0.0
            G.nodes[node]['sent_amount_max'] = 0.0
            G.nodes[node]['sent_amount_min'] = 0.0
            G.nodes[node]['sent_amount_std'] = 0.0
            G.nodes[node]['received_amount_mean'] = 0.0
            G.nodes[node]['received_amount_max'] = 0.0
            G.nodes[node]['received_amount_min'] = 0.0
            G.nodes[node]['received_amount_std'] = 0.0
    
    return G


def print_graph_summary(G: nx.DiGraph):
    """Print a summary of the graph and sample node/edge attributes."""
    print("\n" + "=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)
    print(f"Number of nodes (Entities): {G.number_of_nodes():,}")
    print(f"Number of edges (Transaction flows): {G.number_of_edges():,}")
    print(f"Graph density: {nx.density(G):.6f}")
    
    # Count labeled nodes
    labeled_nodes = sum(1 for n in G.nodes() if G.nodes[n].get('label', 0) == 1)
    print(f"Nodes with laundering label: {labeled_nodes:,} ({100*labeled_nodes/G.number_of_nodes():.2f}%)")
    
    # Sample nodes
    print("\n" + "-" * 60)
    print("SAMPLE NODE ATTRIBUTES (First 2 nodes)")
    print("-" * 60)
    nodes_list = list(G.nodes())[:2]
    for node in nodes_list:
        print(f"\nNode: {node}")
        for attr, value in G.nodes[node].items():
            if isinstance(value, float):
                print(f"  {attr}: {value:,.2f}")
            else:
                print(f"  {attr}: {value}")
    
    # Sample edges
    print("\n" + "-" * 60)
    print("SAMPLE EDGE ATTRIBUTES (First 2 edges)")
    print("-" * 60)
    edges_list = list(G.edges())[:2]
    for source, target in edges_list:
        print(f"\nEdge: {source} -> {target}")
        for attr, value in G.edges[source, target].items():
            if isinstance(value, float):
                print(f"  {attr}: {value:,.2f}")
            else:
                print(f"  {attr}: {value}")
    
    # Temporal range
    print("\n" + "-" * 60)
    print("DATA COVERAGE")
    print("-" * 60)


def main():
    """Main function to orchestrate the graph building pipeline."""
    print("Building Transaction Graph for AML Detection")
    print("=" * 60)
    print("\n[Step 1] Loading data...")
    normal_txns, laundering_txns, accounts = load_data()
    print("\n[Step 2] Preprocessing transactions...")
    transactions = preprocess_transactions(normal_txns, laundering_txns, accounts)
    print(f"Temporal range: {transactions['timestamp'].min()} to {transactions['timestamp'].max()}")
    print("\n[Step 3] Computing edge attributes...")
    edge_attrs = compute_edge_attributes(transactions)
    print("\n[Step 4] Computing entity labels...")
    entity_labels = compute_entity_labels(transactions)
    print("\n[Step 5] Computing temporal statistics...")
    temporal_stats = compute_temporal_statistics(transactions)
    print("\n[Step 6] Building the graph...")
    G = build_graph(edge_attrs, entity_labels, temporal_stats)
    print_graph_summary(G)
    return G

if __name__ == "__main__":
    G = main()