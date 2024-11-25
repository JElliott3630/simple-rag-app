from langchain_ollama import OllamaEmbeddings

# Simply create embedding function from Ollama - if not sufficient, leverage openai free tokens
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
    return embeddings

