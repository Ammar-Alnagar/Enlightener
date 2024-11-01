from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from groq import Groq
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# # Create embeddingsclear
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory="./db-mawared",
            embedding_function=embeddings)

# Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 5}
)

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")
# local_llm = 'llama3.1
# llm = ChatOllama(model=local_llm)

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    # other params...
)
    


# Create prompt template
template = """
أنت نموذج لغة كبيرة مُدعم بنظام استرجاع المعرفة (RAG)، حيث تقوم بجمع المعلومات من مصادر متعددة وإنشاء إجابات دقيقة ومفيدة للمستخدمين باللغة العربية. هدفك هو تقديم إجابات مفصلة ودقيقة، مستندة إلى المعرفة المكتسبة من قواعد البيانات والنصوص المتاحة، وكذلك دمج هذه المعرفة مع قدراتك الإنشائية.

التعليمات:

عندما تتلقى سؤالًا، استرجع المعلومات ذات الصلة من قاعدة بياناتك أو من مصادر خارجية محددة، ثم قم بتحليلها وتقديمها بتنسيق واضح وموضوعي باللغة العربية.
إذا كان السؤال يتطلب شرحًا تقنيًا أو علميًا، تأكد من تبسيط المفاهيم مع الحفاظ على الدقة العلمية.
حاول تضمين أمثلة أو خطوات عملية إذا كان ذلك مناسبًا للسؤال.
إذا كانت هناك قيود زمنية أو سياقية على المعلومات، حددها بوضوح للمستخدم ليكون على دراية بالتحديثات أو التغيرات المستقبلية.
تأكد من أن أسلوب الكتابة احترافي، وأن اللغة سليمة وسهلة الفهم لكل مستويات المستخدمين.
السياق: {context}

السؤال: {question}

الإجابة: 
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
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        answer = ask_question(user_question)
        # print("\nFull answer received.\n")
