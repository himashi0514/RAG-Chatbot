import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the hidden API key from the .env file
load_dotenv()

# Define where we want to save the database on your laptop
CHROMA_PATH = "chroma_db"

def main():
    print("Loading PDFs from the 'data' folder...")
    # Read all PDFs in the directory
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    
    if not docs:
        print("No documents found! Make sure you put PDFs in the 'data' folder")
        return

    print(f"Loaded {len(docs)} pages")

    # Split the text into overlapping chunks
    # chunk_size: How many characters per block
    # chunk_overlap: How many characters to share between blocks (so sentences don't break)
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"Successfully split into {len(chunks)} text chunks.")

    print("Downloading Open-Source Embedding Model (This might take a minute)...")
    # This is a highly efficient model built by Hugging Face specifically for RAG
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("4. Building the Vector Database and saving to disk...")
    # This turns the text into math and saves it in the 'chroma_db' folder
    db = Chroma.from_documents(
        chunks, 
        embedding_model, 
        persist_directory=CHROMA_PATH
    )
    
    print("Ingestion Complete! The Vector Database is ready.")

if __name__ == "__main__":
    main()