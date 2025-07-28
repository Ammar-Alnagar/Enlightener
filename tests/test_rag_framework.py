import unittest
from unittest.mock import patch, MagicMock
from core.rag_framework import RAGFramework

class TestRAGFramework(unittest.TestCase):

    @patch('core.rag_framework.yaml.safe_load')
    @patch('core.rag_framework.load_dotenv')
    @patch('core.rag_framework.HuggingFaceEmbeddings')
    @patch('core.rag_framework.ChatGroq')
    @patch('core.rag_framework.QdrantClient')
    @patch('core.rag_framework.Qdrant')
    def test_initialization(self, mock_qdrant, mock_qdrant_client, mock_chat_groq, mock_hf_embeddings, mock_load_dotenv, mock_safe_load):
        # Mock the config file
        mock_safe_load.return_value = {
            'qdrant_api_key': 'test_qdrant_key',
            'groq_api_key': 'test_groq_key',
            'qdrant_url': 'test_qdrant_url',
            'embedding_model': 'test_embedding_model',
            'llm_model': 'test_llm_model',
            'temperature': 0.1,
            'max_tokens': None,
            'timeout': None,
            'max_retries': 2,
            'qdrant_collection_name': 'test_collection',
            'search_type': 'similarity',
            'search_kwargs': {'k': 5},
            'use_contextual_compression': False
        }

        # Instantiate the RAGFramework
        rag_framework = RAGFramework()

        # Assert that the dependencies were called with the correct parameters
        mock_load_dotenv.assert_called_once()
        mock_hf_embeddings.assert_called_with(model_name='test_embedding_model')
        mock_chat_groq.assert_called_with(
            model='test_llm_model',
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key='test_groq_key'
        )
        mock_qdrant_client.assert_called_with(
            url='test_qdrant_url',
            api_key='test_qdrant_key',
            prefer_grpc=True
        )
        mock_qdrant.assert_called_with(
            client=mock_qdrant_client.return_value,
            collection_name='test_collection',
            embeddings=mock_hf_embeddings.return_value
        )

        # Assert that the rag_chain is created
        self.assertIsNotNone(rag_framework.rag_chain)

if __name__ == '__main__':
    unittest.main()
