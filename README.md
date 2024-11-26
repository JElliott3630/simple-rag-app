# simple-rag-app
This is my very simple RAG app, rendered locally using streamlit and ollama (locally hosted LLM)
It functions but runs slow, I believe because my computer isn't built to handle a local LLM.

P1 - Local Ollama Setup:
1. Install Ollama
Download and install the Ollama service from Ollama's official website - https://ollama.com/
Follow the installation instructions for your operating system.


2. Pull Required Models with:

ollama pull mxbai-embed-large

ollama pull llama2

4. Verify the models are available:
ollama list
(Ensure mxbai-embed-large (for embeddings) and llama2 (for query generation) appear in the list)

5. Start the Ollama Service Locally on localhost:11434:
ollama serve



P2 - Installation and running
1. Install required libraries with - pip install -r requirements.txt

2. Run streamlit app with - streamlit run app.py
