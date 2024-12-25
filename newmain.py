from langchain_community.vectorstores import Qdrant
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from qdrant_client import QdrantClient, models

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


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Create prompt template
template = """
You are an expert assistant specializing in the Mawared HR System. Your task is to answer the user's question strictly based on the provided context. If the context lacks sufficient information, ask focused clarifying questions to gather additional details.

To improve your responses, follow these steps:

Chain-of-Thought (COT): Break down complex queries into logical steps. Use tags like [Step 1], [Step 2], etc., to label each part of the reasoning process. This helps structure your thinking and ensure clarity. For example:

[Step 1] Identify the key details in the context relevant to the question.
[Step 2] Break down any assumptions or information gaps.
[Step 3] Combine all pieces to form the final, well-reasoned response.
Reasoning: Demonstrate a clear logical connection between the context and your answer at each step. If information is missing or unclear, indicate the gap using tags like [Missing Information] and ask relevant follow-up questions to fill that gap.

Clarity and Precision: Provide direct, concise answers focused only on the context. Avoid including speculative or unrelated information.

Follow-up Questions: If the context is insufficient, focus on asking specific, relevant questions. Label them as [Clarifying Question] to indicate they are needed to complete the response. For example:

[Clarifying Question] Could you specify which employee section you're referring to?
Context:
{context}

Question:
{question}

Answer
"""

prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Function to ask questions
def ask_question(question):
    print("Answer:\t", end=" ", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("\n \n \n Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        answer = ask_question(user_question)
        # print("\nFull answer received.\n")