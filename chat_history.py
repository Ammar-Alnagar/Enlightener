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
from langchain_qdrant import QdrantVectorStore
from typing import List, Dict
from dataclasses import dataclass
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")

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

# Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Create a dataclass to store chat history
@dataclass
class ChatHistory:
    messages: List[Dict[str, str]] = None

    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        self.messages.append({"role": "assistant", "content": message})

    def get_chat_history(self) -> str:
        if not self.messages:
            return ""
        
        formatted_history = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in self.messages[:-1]  # Exclude the current question
        ])
        return formatted_history

# Updated prompt template with chat history
template = """
You are an expert assistant specializing in the LONG COT RAG. Your task is to answer the user's question based on the provided context and chat history. Use the chat history to maintain context and provide more relevant answers while ensuring responses are grounded in the retrieved context.

Previous Conversation:
{chat_history}

Chain-of-Thought (CoT):
Break down complex queries into logical, step-by-step reasoning. Use tags like [Step 1], [Step 2], etc., to organize your process clearly:

[Step 1] Review chat history to understand the conversation context
[Step 2] Identify key entities, actions, and objectives in the user's question
[Step 3] Analyze the provided context for relevant information about these entities and actions within the Mawared HR System
[Step 4] If sufficient information exists, synthesize a concise answer based on the context and chat history
[Step 5] If information is missing, explicitly identify gaps using [Missing Information] tags and determine what additional details are required

Context:
{context}

Current Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Initialize chat history
chat_history = ChatHistory()

# Modified RAG chain to include chat history
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def combine_inputs(input_dict):
    return {
        "context": format_docs(input_dict["context"]),
        "question": input_dict["question"],
        "chat_history": input_dict["chat_history"]
    }

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
        "chat_history": lambda x: chat_history.get_chat_history()
    }
    | combine_inputs
    | prompt
    | llm
    | StrOutputParser()
)

# Modified function to ask questions
def ask_question(question: str) -> None:
    print("Answer:\t", end=" ", flush=True)
    
    # Add the user's question to chat history
    chat_history.add_user_message(question)
    
    # Get and stream the response
    response = ""
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
        response += chunk
    
    # Add the assistant's response to chat history
    chat_history.add_assistant_message(response)
    print("\n")

# Example usage
if __name__ == "__main__":
    print("Chat session started. Type 'quit' to exit or 'clear history' to start a new conversation.")
    while True:
        user_question = input("\n\nAsk a question: ").strip()
        
        if user_question.lower() == 'quit':
            break
        elif user_question.lower() == 'clear history':
            chat_history = ChatHistory()
            print("Chat history cleared.")
            continue
            
        ask_question(user_question)