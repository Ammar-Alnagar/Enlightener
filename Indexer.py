# from langchain_experimental.text_splitter import SemanticChunker # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load documents from a PDF file
loader = DirectoryLoader("./2024/gemma2_local_rag", glob="**/*.pdf")

print("pdf loaded loader")

documents = loader.load()

print(len(documents))

# # Create embeddingsclear
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

# # Create Semantic Text Splitter
# text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="interquartile")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=300,
    add_start_index=True,
)

# # Split documents into chunks
texts = text_splitter.split_documents(documents)

# # Create vector store
vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding= embeddings,
    persist_directory="./db-mawared")

print("vectorstore created")