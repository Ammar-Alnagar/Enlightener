from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from qdrant_client import QdrantClient, models
from langchain_qdrant import Qdrant
from typing import Dict, List, Tuple
import time
from datetime import datetime, timedelta
import numpy as np
from collections import OrderedDict

class QueryCache:
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache = OrderedDict()
        
    def _cleanup_expired(self):
        current_time = datetime.now()
        expired_keys = [
            k for k, v in self.cache.items() 
            if current_time - v['timestamp'] > self.ttl
        ]
        for k in expired_keys:
            self.cache.pop(k)

    def get(self, query: str) -> Dict:
        self._cleanup_expired()
        if query in self.cache:
            entry = self.cache.pop(query)  # Remove and re-add to maintain LRU order
            if datetime.now() - entry['timestamp'] <= self.ttl:
                self.cache[query] = entry
                return entry['data']
        return None

    def put(self, query: str, data: Dict):
        self._cleanup_expired()
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest item
        self.cache[query] = {
            'timestamp': datetime.now(),
            'data': data
        }

class CacheAugmentedRetriever:
    def __init__(self, vector_store, embeddings, cache_size: int = 1000, cache_ttl: int = 24):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.cache = QueryCache(max_size=cache_size, ttl_hours=cache_ttl)
        
    def compute_query_embedding(self, query: str) -> np.ndarray:
        return self.embeddings.embed_query(query)

    def find_similar_cached_queries(self, query: str, threshold: float = 0.85) -> List[Tuple[str, float]]:
        query_embedding = self.compute_query_embedding(query)
        similar_queries = []
        
        for cached_query in self.cache.cache.keys():
            cached_embedding = self.compute_query_embedding(cached_query)
            similarity = np.dot(query_embedding, cached_embedding)
            if similarity > threshold:
                similar_queries.append((cached_query, similarity))
                
        return sorted(similar_queries, key=lambda x: x[1], reverse=True)

    def get_relevant_context(self, query: str, k: int = 5) -> Dict:
        # Check cache first
        cached_result = self.cache.get(query)
        if cached_result:
            return cached_result

        # Find similar cached queries
        similar_queries = self.find_similar_cached_queries(query)
        if similar_queries:
            # Combine contexts from similar cached queries
            combined_context = []
            for similar_query, similarity in similar_queries[:3]:  # Use top 3 similar queries
                cached_data = self.cache.get(similar_query)
                if cached_data:
                    combined_context.extend(cached_data['documents'])

        # Get new results from vector store
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Prepare data for caching
        context_data = {
            'documents': [doc.page_content for doc, score in results],
            'scores': [score for _, score in results],
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the results
        self.cache.put(query, context_data)
        
        return context_data

    def __call__(self, query: str) -> str:
        context_data = self.get_relevant_context(query)
        if not context_data['documents']:
            return "No relevant context found. Please rephrase your query."
        
        # Combine contexts, weighted by relevance scores
        combined_context = "\n\n".join(context_data['documents'])
        return combined_context

# HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Qdrant Client Setup
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=True
)

collection_name = "mawared"

# Try to create collection, handle if it already exists
try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=768,  # GTE-large embedding size
            distance=models.Distance.COSINE
        ),
    )
    print(f"Created new collection: {collection_name}")
except Exception as e:
    if "already exists" in str(e):
        print(f"Collection {collection_name} already exists, continuing...")
    else:
        raise e

# Create Qdrant vector store
db = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings,
)

# Create cache-augmented retriever
cag_retriever = CacheAugmentedRetriever(db, embeddings)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Enhanced prompt template for CAG
template = """
You are an expert assistant implementing Cache Augmented Generation (CAG). Your responses should leverage both the cached context history and newly retrieved information to provide comprehensive, accurate answers. Consider both historical context patterns and fresh relevant information.

Key aspects to consider:
1. Historical Context: Analyze patterns from cached similar queries
2. Fresh Information: Incorporate newly retrieved relevant context
3. Consistency: Ensure responses align with previously cached information
4. Relevance: Focus on the most pertinent information for the current query

Context:
{context}

Question:
{question}

Please provide a detailed response that synthesizes both historical and current context:
"""

prompt = ChatPromptTemplate.from_template(template)

# Create the CAG chain using LCEL
cag_chain = (
    {"context": cag_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Function to ask questions
def ask_question(question):
    print("Answer:\t", end=" ", flush=True)
    for chunk in cag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("\n\nAsk a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        answer = ask_question(user_question)