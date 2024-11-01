from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
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
    model="llama-3.1-70b-versatile",
    temperature=0,
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
أنت مساعد متخصص في الإجابة على الأسئلة باستخدام المعلومات المقدمة حصراً في السياق أدناه. يجب أن تعتمد إجاباتك فقط على المعلومات الموجودة في النص المسترجع.

القواعد الأساسية:
1. استخدم فقط المعلومات الموجودة في السياق المقدم
2. إذا لم تجد المعلومات في السياق، قل بوضوح "لا أستطيع الإجابة على هذا السؤال بناءً على السياق المتوفر"
3. لا تستخدم معرفتك العامة أو معلومات خارجية
4. لا تستنتج أو تخمن معلومات غير موجودة في السياق
5. كن دقيقاً في اقتباس المعلومات من السياق

السياق المقدم:
{context}

السؤال:
{question}

الإجابة:
"""

prompt = ChatPromptTemplate.from_template(template)

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