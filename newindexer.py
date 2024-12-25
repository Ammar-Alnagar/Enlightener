import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant # Changed import
from qdrant_client import QdrantClient, models
from tqdm import tqdm  # for progress bars
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def load_and_chunk_documents(data_dir="Data", chunk_size=500, chunk_overlap=300):
    """Loads PDF documents, splits them into chunks, and returns the chunks."""
    print("Loading PDF documents...")
    # use glob **/*.pdf to get all pdf in data folder and subfolders 
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    print("Splitting documents into chunks...")
    # Show progress with tqdm
    texts = []
    for doc in tqdm(documents, desc="Splitting Docs"):
       texts.extend(text_splitter.split_documents([doc]))
    print(f"Created {len(texts)} text chunks.")
    return texts

def create_vectorstore(texts, model_name="BAAI/bge-large-en-v1.5", persist_dir="./qdrant_mawared", collection_name="mawared"):
    """Creates a Qdrant vector store from text chunks."""
    print("Creating vector embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name) # Changed to HuggingFaceEmbeddings
    print("Creating vector store...")


    # Connect to Qdrant Client
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True
    )

    # Get embedding dimension by embedding a test string
    vector_size = 1024

    # Create collection if it doesn't exist
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

    # Create and return the vector store
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )

    # Add documents to the vector store
    vectorstore.add_documents(texts)
    
    return vectorstore

def main():
    texts = load_and_chunk_documents()
    vectorstore = create_vectorstore(texts)
    

if __name__ == "__main__":
    main()
    print("Done!")