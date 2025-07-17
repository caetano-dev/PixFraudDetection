import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import networkx as nx
from community_detection import (
    fetch_graph_data,
    build_networkx_graph,
    detect_communities_louvain,
    write_communities_to_neo4j,
    analyze_communities,
    main as community_main,
    Neo4jConnection,
)

class TestCommunityDetection(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.mock_conn = MagicMock(spec=Neo4jConnection)
        self.relationships = [
            {'source': 'acc1', 'target': 'acc2', 'weight': 100.0, 'transaction_count': 2},
            {'source': 'acc2', 'target': 'acc3', 'weight': 200.0, 'transaction_count': 5},
            {'source': 'acc1', 'target': 'acc3', 'weight': 50.0, 'transaction_count': 1},
            # Add a duplicate edge to test aggregation
            {'source': 'acc1', 'target': 'acc2', 'weight': 25.0, 'transaction_count': 1},
        ]
        self.partition = {'acc1': 0, 'acc2': 0, 'acc3': 0, 'acc4': 1}

    def test_fetch_graph_data(self):
        """Test fetching graph data from Neo4j."""
        self.mock_conn.query.return_value = self.relationships
        result = fetch_graph_data(self.mock_conn)
        
        self.mock_conn.query.assert_called_once()
        self.assertIn("MATCH (a1:Account)-[mf:MONEY_FLOW]->(a2:Account)", self.mock_conn.query.call_args[0][0])
        self.assertEqual(len(result), 4)

    def test_build_networkx_graph(self):
        """Test building a NetworkX graph."""
        G = build_networkx_graph(self.relationships)
        
        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 3)
        self.assertEqual(G.number_of_edges(), 3)
        # Check if weights and counts were aggregated correctly
        self.assertEqual(G['acc1']['acc2']['weight'], 125.0)
        self.assertEqual(G['acc1']['acc2']['transaction_count'], 3)
        self.assertEqual(G['acc2']['acc3']['weight'], 200.0)

    @patch('community_detection.community_louvain')
    def test_detect_communities_louvain(self, mock_community_louvain):
        """Test Louvain community detection."""
        mock_community_louvain.best_partition.return_value = self.partition
        mock_community_louvain.modularity.return_value = 0.5
        
        G = build_networkx_graph(self.relationships)
        partition, modularity = detect_communities_louvain(G)
        
        mock_community_louvain.best_partition.assert_called_once_with(G, weight='weight', random_state=42)
        mock_community_louvain.modularity.assert_called_once()
        self.assertEqual(modularity, 0.5)
        self.assertEqual(len(partition), 4)

    def test_write_communities_to_neo4j(self):
        """Test writing community data back to Neo4j."""
        # Mock the session and run method
        mock_session = MagicMock()
        # Configure the mock connection's _driver attribute
        self.mock_conn._driver = MagicMock()
        self.mock_conn._driver.session.return_value.__enter__.return_value = mock_session

        write_communities_to_neo4j(self.mock_conn, self.partition)

        self.assertGreater(mock_session.run.call_count, 0)
        # Check the query in the first call
        first_call_args = mock_session.run.call_args_list[0]
        self.assertIn("UNWIND $community_data AS cd", first_call_args[0][0])
        # Check that data is passed correctly
        self.assertIn("community_data", first_call_args[1])
        self.assertEqual(len(first_call_args[1]['community_data']), 4)

    @patch('community_detection.pd.DataFrame.to_csv')
    def test_analyze_communities(self, mock_to_csv):
        """Test community analysis function."""
        analysis_results = [
            {'communityId': 0, 'communitySize': 3, 'totalAmount': 1000, 'totalFraudTransactions': 2, 'fraudRate': 0.1},
            {'communityId': 1, 'communitySize': 1, 'totalAmount': 500, 'totalFraudTransactions': 1, 'fraudRate': 0.5},
        ]
        self.mock_conn.query.return_value = analysis_results
        
        df = analyze_communities(self.mock_conn)
        
        self.mock_conn.query.assert_called_once()
        self.assertIn("WHERE a.communityId IS NOT NULL", self.mock_conn.query.call_args[0][0])
        mock_to_csv.assert_called_once_with("./data/community_analysis.csv", index=False)
        self.assertEqual(len(df), 2)

    @patch('community_detection.analyze_communities')
    @patch('community_detection.write_communities_to_neo4j')
    @patch('community_detection.detect_communities_louvain')
    @patch('community_detection.build_networkx_graph')
    @patch('community_detection.fetch_graph_data')
    @patch('community_detection.Neo4jConnection')
    def test_main_flow(self, mock_neo4j_conn, mock_fetch, mock_build, mock_detect, mock_write, mock_analyze):
        """Test the main execution flow."""
        # Setup mocks
        mock_neo4j_conn.return_value = self.mock_conn
        mock_fetch.return_value = self.relationships
        mock_graph = nx.Graph()
        mock_graph.add_edge('a', 'b')
        mock_build.return_value = mock_graph
        mock_detect.return_value = (self.partition, 0.5)
        mock_analyze.return_value = pd.DataFrame({'communityId': [0]})

        # Run main
        community_main()

        # Assertions
        mock_neo4j_conn.assert_called_once()
        mock_fetch.assert_called_once_with(self.mock_conn)
        mock_build.assert_called_once_with(self.relationships)
        mock_detect.assert_called_once_with(mock_graph)
        mock_write.assert_called_once_with(self.mock_conn, self.partition)
        mock_analyze.assert_called_once_with(self.mock_conn)
        self.mock_conn.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()
