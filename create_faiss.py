# Required imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv
import os

# Step 1: Load raw data (Text files about schemes)
DATA_PATH = "data/"

def load_text_files(data):
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_text_files(DATA_PATH)
#print("Length of documents: ", len(documents))

# Step 2: Split Text into Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
#print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Load Embedding Model (Gemini Flash)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_embedding_model():
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=GOOGLE_API_KEY)
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Create Vector Embeddings using FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, embedding_model)

# Step 5: Save FAISS index locally
faiss_db.save_local(DB_FAISS_PATH)

print(f"FAISS index saved to {DB_FAISS_PATH}")
