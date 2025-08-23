import pandas as pd
import networkx as nx
import os
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Tuple, Optional

class TemporalFinancialGraph:
    """
    A class to build and manage temporal financial transaction graphs
    with support for windowed analysis and laundering detection.
    """
    
    def __init__(self, data_dir: str = 'data', processed_dir: str = 'processed'):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, processed_dir)
        self.G_all = nx.MultiDiGraph()  # All-time multigraph
        self.transactions_df = None
        self.accounts_df = None
        
    def load_data(self):
        """Load and combine transaction and account data"""
        try:
            transactions_df = pd.read_csv(os.path.join(self.processed_dir, '1_filtered_normal_transactions.csv'))
            laundering_df = pd.read_csv(os.path.join(self.processed_dir, '2_filtered_laundering_transactions.csv'))
            self.accounts_df = pd.read_csv(os.path.join(self.processed_dir, '3_filtered_accounts.csv'))

            # Combine transactions
            self.transactions_df = pd.concat([transactions_df, laundering_df], ignore_index=True)
            self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])
            
            # Create composite transaction ID for robust matching
            self.transactions_df['transaction_id'] = self.transactions_df.apply(
                lambda row: f"{row['timestamp'].strftime('%Y/%m/%d %H:%M')}_{row['from_account']}_{row['to_account']}_{row['amount_sent']}_{row['currency_sent']}_{row['payment_type']}", 
                axis=1
            )
            
            print(f"Loaded {len(self.transactions_df)} transactions and {len(self.accounts_df)} accounts")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find required CSV files: {e}")
    
    def build_all_time_graph(self):
        """Build the complete all-time graph"""
        # Add nodes with attributes
        for _, row in self.accounts_df.iterrows():
            self.G_all.add_node(
                row['account_id_hex'],
                bank_name=row['bank_name'],
                entity_name=row['entity_name'],
                is_laundering_involved=False  # Will be updated based on edges
            )
        
        # Add edges with attributes
        laundering_nodes = set()
        
        for _, row in self.transactions_df.iterrows():
            src, dst = row['from_account'], row['to_account']
            
            if self.G_all.has_node(src) and self.G_all.has_node(dst):
                # Convert amount to Decimal for precision
                amount = Decimal(str(row['amount_sent']))
                
                is_laundering = bool(row.get('is_laundering', False))
                if is_laundering:
                    laundering_nodes.update([src, dst])
                
                self.G_all.add_edge(
                    src, dst,
                    amount=amount,
                    timestamp=row['timestamp'],
                    payment_type=row['payment_type'],
                    is_laundering=is_laundering,
                    attempt_id=row.get('attempt_id', 0),
                    attempt_type=row.get('attempt_type', 'NORMAL'),
                    transaction_id=row['transaction_id']
                )
        
        # Update node laundering involvement
        for node in laundering_nodes:
            self.G_all.nodes[node]['is_laundering_involved'] = True
        
        print(f"Built all-time graph: {self.G_all.number_of_nodes()} nodes, {self.G_all.number_of_edges()} edges")
        print(f"Laundering-involved nodes: {len(laundering_nodes)}")
    
    def get_windowed_graph(self, center_time: datetime, window_days: int) -> nx.MultiDiGraph:
        """
        Create a windowed subgraph around a specific time.
        
        Args:
            center_time: Center timestamp for the window
            window_days: Window size in days (3 or 7 typically)
        
        Returns:
            NetworkX MultiDiGraph for the specified window
        """
        start_time = center_time - timedelta(days=window_days/2)
        end_time = center_time + timedelta(days=window_days/2)
        
        # Filter edges by timestamp
        windowed_edges = [
            (u, v, k, d) for u, v, k, d in self.G_all.edges(keys=True, data=True)
            if start_time <= d['timestamp'] <= end_time
        ]
        
        # Create windowed subgraph
        G_window = nx.MultiDiGraph()
        
        # Add all nodes (maintain structure)
        G_window.add_nodes_from(self.G_all.nodes(data=True))
        
        # Add filtered edges
        for u, v, k, d in windowed_edges:
            G_window.add_edge(u, v, key=k, **d)
        
        return G_window
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics for the graph"""
        laundering_edges = sum(1 for _, _, d in self.G_all.edges(data=True) if d.get('is_laundering'))
        laundering_nodes = sum(1 for _, d in self.G_all.nodes(data=True) if d.get('is_laundering_involved'))
        
        return {
            'total_nodes': self.G_all.number_of_nodes(),
            'total_edges': self.G_all.number_of_edges(),
            'laundering_edges': laundering_edges,
            'laundering_nodes': laundering_nodes,
            'laundering_edge_rate': laundering_edges / self.G_all.number_of_edges() if self.G_all.number_of_edges() > 0 else 0,
            'laundering_node_rate': laundering_nodes / self.G_all.number_of_nodes() if self.G_all.number_of_nodes() > 0 else 0
        }

# Usage example
if __name__ == "__main__":
    # Build the graph
    graph_builder = TemporalFinancialGraph()
    graph_builder.load_data()
    graph_builder.build_all_time_graph()
    
    # Get summary
    stats = graph_builder.get_summary_stats()
    print("\n=== Graph Summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example windowed analysis
    if not graph_builder.transactions_df.empty:
        sample_time = graph_builder.transactions_df['timestamp'].iloc[len(graph_builder.transactions_df)//2]
        G_3day = graph_builder.get_windowed_graph(sample_time, 3)
        G_7day = graph_builder.get_windowed_graph(sample_time, 7)
        
        print(f"\n3-day window around {sample_time}: {G_3day.number_of_edges()} edges")
        print(f"7-day window around {sample_time}: {G_7day.number_of_edges()} edges")

    print("\n--- Manual Sanity Checks ---")
    
    G = graph_builder.G_all
    
    # Check the degrees of the most connected nodes
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    print("Top 5 nodes by degree:")
    for node, degree in degrees[:5]:
        print(f"Node {node}: Degree {degree}")
        
    # Check a node involved in both sending and receiving
    sample_node = degrees[0][0]
    in_edges = G.in_edges(sample_node, data=True)
    out_edges = G.out_edges(sample_node, data=True)
    print(f"\nSample node {sample_node} in-edges:")
    for u, v, d in in_edges:
        print(f"From {u} to {v}, Amount: {d['amount']}, Laundering: {d['is_laundering']}")
        
    print(f"\nSample node {sample_node} out-edges:")
    for u, v, d in out_edges:
        print(f"From {u} to {v}, Amount: {d['amount']}, Laundering: {d['is_laundering']}")