from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplatehi
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    show_progress=False,
)

db = Chroma(
    persist_directory="./db-mawared",
    embedding_function=embeddings,
)

# Create retriever with MMR search
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 8,
        "lambda_mult": 0.7
    }
)

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=5
)

# Add contextual compression
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)

# Enhanced RAG-focused prompt template
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

prompt = ChatPromptTemplatehi.from_template(template)

# Create the RAG chain
rag_chain = (
    {"context": compression_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Function to ask questions
def ask_question(question: str) -> None:
    try:
        print("Answer:\t", end=" ", flush=True)
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nError processing question: {str(e)}")

if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        ask_question(user_question)