from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

# Check if CUDA is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load documents from a PDF file
loader = DirectoryLoader("Data", glob="**/*.pdf")
print("PDF files loaded")
documents = loader.load()
print(f"Number of documents loaded: {len(documents)}")

# Create embeddings using multilingual-e5-large model
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# Create text splitter with Arabic-friendly settings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True,
    # Separators prioritized for Arabic text
    separators=["\n\n", "\n", ".", "!", "?", "،", ";", ",", " ", ""]
)

# Split documents into chunks
texts = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding=embeddings,
    persist_directory="./db-mawared"
)
print("Vector store created successfully")