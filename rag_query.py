import argparse
import os
import shutil
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from langchain_ollama import OllamaLLM

CHROMA_PATH = './chroma'

PROMPT_TEMPLATE = '''
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
'''


# parse args to get query then run RAG
# def rag_query():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The Query Text")
#     args = parser.parse_args()
#     rag_response = query_rag(args.query_text)
#     return rag_response

# Query RAG function - gathers top k similar embeddings from db then generates a response using ollama LLM text generationdef query_rag(query_text):
def query_rag(query_text, db):
    # print('creating access to db')
    # db = Chroma(
    #     persist_directory=CHROMA_PATH, embedding_function=embedding_function
    # )
    print("db accessed")

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.5:
        print('No results found')
        return 

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model='llama2')
    response_text = model.invoke(prompt)

    sources = [doc.metadata for doc, _score in results]
    formatted_response = f"Response: {response_text} \nSources: {sources}"
    print(formatted_response)
    return formatted_response

if __name__ == "__main__":
    # rag_query()
    query_rag('How much money does each player start with in monopoly')