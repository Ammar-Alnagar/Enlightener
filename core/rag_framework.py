import os
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from qdrant_client import QdrantClient
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class RAGFramework:
    def __init__(self, config_path='config/config.yaml'):
        load_dotenv()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", self.config['qdrant_api_key'])
        self.groq_api_key = os.getenv("GROQ_API_KEY", self.config['groq_api_key'])
        self.qdrant_url = os.getenv("QDRANT_URL", self.config['qdrant_url'])

        self.embeddings = self._get_embedding_model()
        self.llm = self._get_llm()
        self.vector_store = self._get_vector_store()
        self.retriever = self._get_retriever()
        self.rag_chain = self._create_rag_chain()

    def _get_embedding_model(self):
        return HuggingFaceEmbeddings(model_name=self.config['embedding_model'])

    def _get_llm(self):
        return ChatGroq(
            model=self.config['llm_model'],
            temperature=self.config['temperature'],
            max_tokens=self.config['max_tokens'],
            timeout=self.config['timeout'],
            max_retries=self.config['max_retries'],
            api_key=self.groq_api_key
        )

    def _get_vector_store(self):
        client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            prefer_grpc=True
        )
        return Qdrant(
            client=client,
            collection_name=self.config['qdrant_collection_name'],
            embeddings=self.embeddings,
        )

    def _get_retriever(self):
        retriever = self.vector_store.as_retriever(
            search_type=self.config['search_type'],
            search_kwargs=self.config['search_kwargs']
        )
        if self.config['use_contextual_compression']:
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_retriever=retriever,
                base_compressor=compressor
            )
            return compression_retriever
        return retriever

    def _create_rag_chain(self):
        template = """
        You are an expert assistant. Your task is to answer the user's question strictly based on the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask_question(self, question: str):
        return self.rag_chain.stream(question)

if __name__ == '__main__':
    rag = RAGFramework()
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        for chunk in rag.ask_question(user_question):
            print(chunk, end="", flush=True)
        print("\n")
