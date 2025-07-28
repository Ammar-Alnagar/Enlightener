import os
import yaml
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self, config_path='config/config.yaml'):
        load_dotenv()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", self.config['qdrant_api_key'])
        self.qdrant_url = os.getenv("QDRANT_URL", self.config['qdrant_url'])
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            prefer_grpc=True
        )

    def load_and_chunk_documents(self):
        logger.info("Loading PDF documents...")
        loader = DirectoryLoader(
            self.config['data_dir'],
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            add_start_index=True,
        )

        logger.info("Splitting documents into chunks...")
        texts = []
        for doc in tqdm(documents, desc="Splitting Docs"):
            texts.extend(text_splitter.split_documents([doc]))
        logger.info(f"Created {len(texts)} text chunks.")
        return texts

    def create_vectorstore(self, texts):
        logger.info("Creating vector embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name=self.config['embedding_model'])
        logger.info("Creating vector store...")

        collection_name = self.config['qdrant_collection_name']
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.config['vector_size'],
                    distance=models.Distance[self.config['distance_metric'].upper()]
                )
            )
            logger.info(f"Created new collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists, continuing...")

        vectorstore = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=embeddings
        )

        logger.info("Adding documents to the vector store...")
        vectorstore.add_documents(texts)
        logger.info("Documents added to the vector store successfully.")
        return vectorstore

    def run(self):
        texts = self.load_and_chunk_documents()
        self.create_vectorstore(texts)

if __name__ == "__main__":
    indexer = Indexer()
    indexer.run()
    logger.info("Done!")
