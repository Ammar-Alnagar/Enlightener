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
You are an expert assistant specializing in the Mawared HR System. Your task is to answer the user's question strictly based on the provided context. If the context lacks sufficient information, utilize a Chain-of-Thought (CoT) reasoning process in conjunction with Retrieval-Augmented Generation (RAG) principles to structure your response and retrieve additional details effectively.

To ensure high-quality responses, follow these steps:

Chain-of-Thought (CoT):
Break down complex queries into logical, step-by-step reasoning. Use tags like [Step 1], [Step 2], etc., to organize your process clearly:

[Step 1] Identify key entities, actions, and objectives in the user's question.
[Step 2] Analyze the provided context for relevant information about these entities and actions within the Mawared HR System.
[Step 3] If sufficient information exists, synthesize a concise answer based on the context.
[Step 4] If information is missing, explicitly identify gaps using [Missing Information] tags and determine what additional details are required.
[Step 5] Formulate precise, targeted questions labeled as [Clarifying Question] to retrieve missing details and refine your response.
Reasoning and RAG Integration:
Demonstrate logical connections between the context and your answers:

Use context to validate your reasoning and retrieve specific, relevant information.
Highlight any missing details explicitly and identify relevant clarifying questions to enhance the response.
Ensure your answers are rooted in the provided information and avoid speculation or unrelated content.
Example:

[Step 1] The user is asking about [Specific Feature].
[Step 2] The context mentions [Related Entity A] but does not address [Specific Feature].
[Step 3] [Missing Information]: Details about [Specific Feature] in the Mawared system.
[Clarifying Question] Could you specify which aspect of [Specific Feature] you are referring to?
Clarity and Precision:
Provide direct, focused answers using the available context. Avoid introducing extraneous details or speculation. Label each logical step and reasoning process explicitly to ensure transparency and coherence.

Follow-Up Questions:
If the context is insufficient, structure follow-up questions to retrieve critical missing details. Use [Clarifying Question] tags for clarity. For example:

[Clarifying Question] Could you clarify whether you are referring to [Employee Profile Section A] or [Section B] in the Mawared system?
Context:
{context}

Question:
{question}

Answer:
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