import streamlit as st
import os
import shutil
import create_db
import rag_query
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


CHROMA_PATH = './chroma'


@st.cache_resource
def get_embedding_function():
    print("Initializing embedding function...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
    return embeddings

@st.cache_resource
def get_chroma_connection():
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db



st.title("This is my basic RAG app")

# Initialize session state
if "responses" not in st.session_state:
    st.session_state["responses"] = []  # Store all responses
if "database_created" not in st.session_state:
    st.session_state["database_created"] = False  # Track if database is created

# Function to create the database
def create_database(uploaded_files):
    destination_dir = "./data"
    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    os.mkdir(destination_dir)

    # Save uploaded files to the temporary directory
    for pdf_file in uploaded_files:
        destination_path = os.path.join(destination_dir, pdf_file.name)
        with open(destination_path, "wb") as f:
            f.write(pdf_file.getbuffer())

    # Create the database
    create_db.create_db(destination_dir, get_embedding_function())
    st.session_state["database_created"] = True
    st.write("Database created successfully.")

# Function to execute the query
def execute_query(query_text):
    if not st.session_state["database_created"]:
        st.write("Please upload files and create the database first!")
        return

    # Execute the RAG query
    response = rag_query.query_rag(query_text, get_chroma_connection())
    st.write(response)
    # st.session_state["responses"].append({"query": query_text, "response": response})

# File uploader
uploaded_files = st.file_uploader("Upload a file below", type="pdf", accept_multiple_files=True)

if uploaded_files and not st.session_state["database_created"]:
    if st.button("Create Database"):
        create_database(uploaded_files)

# Display previous responses and dynamically add buttons for new queries
# if st.session_state["responses"]:
#     st.write("### Responses:")
#     for idx, entry in enumerate(st.session_state["responses"]):
#         st.write(f"**Q{idx + 1}:** {entry['query']}")
#         st.write(f"**A{idx + 1}:** {entry['response']}")
#         st.markdown("---")

# Input for new query
query_text = st.text_input("Ask a new question!", placeholder="Enter your question here...")

# Dynamically create a button for the query
execute_query_button = st.button("Execute Query")
if execute_query_button:
    if query_text.strip():
        
        execute_query(query_text)


        # Dynamically create a new button below the query result
        # st.button(f"Show responses for: {query_text}")