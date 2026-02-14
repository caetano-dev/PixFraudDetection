import pandas as pd
import networkx as nx

def analyze_graph_health(transactions_file: str):
    print(f"Loading {transactions_file}...")
    df = pd.read_parquet(transactions_file)
    
    # 1. Basic Stats
    total_tx = len(df)
    laundering_tx = df['is_laundering'].sum()
    print(f"Total Transactions: {total_tx:,}")
    print(f"Laundering Transactions: {laundering_tx:,}")
    
    # 2. Build the Directed Graph
    print("\nBuilding Directed Graph...")
    G = nx.from_pandas_edgelist(
        df, 
        source='from_account', 
        target='to_account', 
        edge_attr='is_laundering',
        create_using=nx.DiGraph
    )
    
    # 3. Component Fragmentation
    wcc = list(nx.weakly_connected_components(G))
    largest_wcc = max(wcc, key=len)
    percent_in_lcc = (len(largest_wcc) / len(G)) * 100
    
    print("\n--- GRAPH TOPOLOGY HEALTH ---")
    print(f"Total Nodes: {len(G):,}")
    print(f"Number of Isolated Components: {len(wcc):,}")
    print(f"Nodes in Largest Component: {len(largest_wcc):,} ({percent_in_lcc:.2f}%)")
    
    if percent_in_lcc < 80:
        print("WARNING: Graph is highly fragmented. Centrality metrics will suffer.")
        
    # 4. Broken Laundering Chains (Dead Ends)
    print("\n--- LAUNDERING CHAIN HEALTH ---")
    bad_nodes = set(df[df['is_laundering'] == 1]['from_account']).union(
                set(df[df['is_laundering'] == 1]['to_account']))
    
    dead_ends = 0
    source_nodes = 0
    pass_through = 0
    
    for node in bad_nodes:
        if node not in G: continue
        
        in_laundering = sum(1 for _, _, d in G.in_edges(node, data=True) if d.get('is_laundering') == 1)
        out_laundering = sum(1 for _, _, d in G.out_edges(node, data=True) if d.get('is_laundering') == 1)
        
        if in_laundering > 0 and out_laundering == 0:
            dead_ends += 1
        elif in_laundering == 0 and out_laundering > 0:
            source_nodes += 1
        elif in_laundering > 0 and out_laundering > 0:
            pass_through += 1
            
    print(f"Total Laundering Nodes: {len(bad_nodes):,}")
    print(f"Pass-through (Healthy Chain): {pass_through:,}")
    print(f"Dead Ends (Money goes in, nothing goes out): {dead_ends:,}")
    print(f"Sources (Money goes out, nothing comes in): {source_nodes:,}")
    
    if dead_ends > pass_through:
        print("CRITICAL: Your filtering broke the laundering chains. Most fraud nodes are dead ends.")

if __name__ == "__main__":
    # Point this to the output of your clean_dataset.py script
    analyze_graph_health("data/HI_Small/1_filtered_normal_transactions.parquet")
    analyze_graph_health("data/HI_Small/2_filtered_laundering_transactions.parquet")
#    analyze_graph_health("data/HI_Small/1_filtered_normal_transactions.parquet")