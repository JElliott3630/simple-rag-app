# Create and populate a Chroma database
import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function


CHROMA_PATH = './chroma'
DATA_PATH = './data'

def create_db(dir, embedding_function):
    documents = load_documents(dir)
    chunks = split_documents(documents)
    save_to_chroma(chunks, embedding_function)

# Load all documents from the specified directory
def load_documents(dir):
    document_loader = PyPDFDirectoryLoader(dir)
    return document_loader.load()

# Split text recursively from documents into chunks
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    sample_chunk = chunks[10] if len(chunks) > 10 else chunks[0]
    print(f'Sample page content: \n{sample_chunk.page_content}')
    print(f'Sample chunk metadata: \n{sample_chunk.metadata}')

    return chunks

# Save chunks with generated embeddings to Chroma DB
def save_to_chroma(chunks, embedding_function):
    print('Starting Chroma save...')

    # Delete existing Chroma DB if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)

    print("creating db")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH  # Ensures local persistence
    )
    print("db created")

    print(f'Saved {len(chunks)} chunks to Chroma DB at {CHROMA_PATH}')

if __name__ == "__main__":
    create_db(DATA_PATH)
