import unittest
from unittest.mock import patch, MagicMock
from core.indexing import Indexer

class TestIndexer(unittest.TestCase):

    @patch('core.indexing.yaml.safe_load')
    @patch('core.indexing.load_dotenv')
    @patch('core.indexing.DirectoryLoader')
    @patch('core.indexing.RecursiveCharacterTextSplitter')
    @patch('core.indexing.HuggingFaceEmbeddings')
    @patch('core.indexing.QdrantClient')
    @patch('core.indexing.Qdrant')
    def test_run(self, mock_qdrant, mock_qdrant_client, mock_hf_embeddings, mock_text_splitter, mock_dir_loader, mock_load_dotenv, mock_safe_load):
        # Mock the config file
        mock_safe_load.return_value = {
            'qdrant_api_key': 'test_qdrant_key',
            'qdrant_url': 'test_qdrant_url',
            'data_dir': 'test_data_dir',
            'chunk_size': 100,
            'chunk_overlap': 10,
            'embedding_model': 'test_embedding_model',
            'qdrant_collection_name': 'test_collection',
            'vector_size': 128,
            'distance_metric': 'Cosine'
        }

        # Mock the return values of the dependencies
        mock_dir_loader.return_value.load.return_value = [MagicMock()]
        mock_text_splitter.return_value.split_documents.return_value = [MagicMock()]
        mock_qdrant_client.return_value.get_collections.return_value.collections = []

        # Instantiate the Indexer
        indexer = Indexer()
        indexer.run()

        # Assert that the dependencies were called with the correct parameters
        mock_load_dotenv.assert_called_once()
        mock_dir_loader.assert_called_with(
            'test_data_dir',
            glob="**/*.pdf",
            loader_cls=unittest.mock.ANY
        )
        mock_text_splitter.assert_called_with(
            chunk_size=100,
            chunk_overlap=10,
            add_start_index=True
        )
        mock_hf_embeddings.assert_called_with(model_name='test_embedding_model')
        mock_qdrant_client.assert_called_with(
            url='test_qdrant_url',
            api_key='test_qdrant_key',
            prefer_grpc=True
        )
        mock_qdrant.assert_called()
        mock_qdrant.return_value.add_documents.assert_called()


if __name__ == '__main__':
    unittest.main()
