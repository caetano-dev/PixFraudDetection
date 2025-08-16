# type: ignore[attr-defined]
import unittest
from unittest.mock import patch, MagicMock
import json
import redis

from ingestion_engine import (
    init_redis,
    init_neo4j,
    ingest_transaction,
    main as ingestion_main
)

class TestIngestionEngine(unittest.TestCase):

    @patch('ingestion_engine.redis.Redis')
    def test_init_redis_success(self, mock_redis_class):
        """Test successful Redis connection."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        
        client = init_redis()
        
        mock_redis_instance.ping.assert_called_once()
        self.assertEqual(client, mock_redis_instance)

    @patch('ingestion_engine.redis.Redis')
    def test_init_redis_failure(self, mock_redis_class):
        """Test Redis connection failure."""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.side_effect = redis.exceptions.ConnectionError
        mock_redis_class.return_value = mock_redis_instance
        
        with self.assertLogs('ingestion_engine', level='ERROR'):
            with self.assertRaises(redis.exceptions.ConnectionError):
                init_redis()

    @patch('ingestion_engine.GraphDatabase.driver')
    def test_init_neo4j_success(self, mock_driver_class):
        """Test successful Neo4j connection."""
        mock_driver_instance = MagicMock()
        mock_session = MagicMock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session
        mock_driver_class.return_value = mock_driver_instance
        
        driver = init_neo4j()
        
        mock_driver_class.assert_called_once()
        mock_session.run.assert_called_once_with("RETURN 1")
        self.assertEqual(driver, mock_driver_instance)

    @patch('ingestion_engine.GraphDatabase.driver')
    def test_init_neo4j_failure(self, mock_driver_class):
        """Test Neo4j connection failure."""
        mock_driver_class.side_effect = Exception("Connection failed")
        
        with self.assertLogs('ingestion_engine', level='ERROR'):
            with self.assertRaises(Exception):
                init_neo4j()

    def test_ingest_transaction(self):
        """Test the transaction ingestion logic."""
        mock_driver = MagicMock()
        tx_data = {
            'transaction_id': 'tx123', 'sender_id': 'cpf1', 'receiver_id': 'cpf2',
            'amount': 100.0, 'timestamp': '2023-01-01T12:00:00', 'fraud_flag': 'NORMAL',
            'device_id': 'dev1', 'ip_address': '1.1.1.1', 'transaction_type': 'transfer',
            'channel': 'mobile_app', 'merchant_category': 'retail', 'hour_of_day': 12,
            'day_of_week': 6, 'is_weekend': True, 'same_state': False,
            'sender_verified': True, 'receiver_verified': False,
            'sender_state': 'SP', 'receiver_state': 'RJ',
            'sender_risk_score': 0.1, 'receiver_risk_score': 0.8
        }
        
        ingest_transaction(mock_driver, tx_data)
        
        mock_driver.execute_query.assert_called_once()
        called_args, called_kwargs = mock_driver.execute_query.call_args
        self.assertEqual(called_kwargs, tx_data)

    @patch('ingestion_engine.init_redis')
    @patch('ingestion_engine.init_neo4j')
    @patch('ingestion_engine.ingest_transaction')
    @patch('ingestion_engine.config', {'app': {'clear_on_start': False}})
    def test_main_loop(self, mock_ingest, mock_init_neo4j, mock_init_redis):
        """Test the main loop of the ingestion engine."""
        mock_redis_client = MagicMock()
        mock_pubsub = MagicMock()

        mock_messages = [
            {'type': 'message', 'data': json.dumps({'transaction_id': 'tx1'})},
            {'type': 'message', 'data': json.dumps({'transaction_id': 'tx2'})},
        ]

        mock_pubsub.listen.return_value = iter(mock_messages)
        mock_redis_client.pubsub.return_value = mock_pubsub
        mock_init_redis.return_value = mock_redis_client

        mock_pubsub.listen.side_effect = [iter(mock_messages), KeyboardInterrupt]

        try:
            ingestion_main()
        except KeyboardInterrupt:
            self.fail("KeyboardInterrupt should be caught within ingestion_main and not propagate")

        mock_init_redis.assert_called_once()
        mock_init_neo4j.assert_called_once()
        mock_redis_client.pubsub.assert_called_once_with()
        mock_pubsub.subscribe.assert_called_once_with('pix_transactions')

        self.assertEqual(mock_ingest.call_count, 2)
        mock_ingest.assert_any_call(mock_init_neo4j.return_value, {'transaction_id': 'tx1'})
        mock_ingest.assert_any_call(mock_init_neo4j.return_value, {'transaction_id': 'tx2'})

if __name__ == '__main__':
    unittest.main()
