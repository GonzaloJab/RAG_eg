from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
import ollama

# --- Custom OllamaEmbeddings (unchanged) ---
class OllamaEmbeddings:
    def __init__(self, model_name='jina/jina-embeddings-v2-base-es'):
        self.model_name = model_name
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            try:
                response = ollama.embeddings(model=self.model_name, prompt=text)
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"Error generating embedding for text: {e}")
                embeddings.append([0.0] * 768)
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding for query: {e}")
            return [0.0] * 768

INDEX_NAME = "book_index_improved"

def runn_llm(query: str, chat_history: list[dict[str, str]]):
    # Vector store / retriever
    embedding_function = OllamaEmbeddings()
    docsearch = Chroma(
        collection_name=INDEX_NAME,
        embedding_function=embedding_function,
        persist_directory="chroma_book_db_improved",
    )
    base_retriever = docsearch.as_retriever()

    # LLM
    llm = ChatOllama(model="llama3.2:3b")

    # Prompts from hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # History-aware retriever (NO chat_history here)
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=base_retriever,
        prompt=rephrase_prompt,
    )

    # QA doc-combiner chain
    stuff_documents_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt,
    )

    # Full RAG chain
    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain,
    )

    # Pass chat_history at run time IN THE INPUT DICT
    response = qa.invoke({
        "input": query,
        "chat_history": chat_history,  # list of {"role": "...", "content": "..."} or messages
    })

    # For create_stuff_documents_chain the answer key is "answer"
    return response["answer"]

if __name__ == "__main__":
    # Provide an explicit (possibly empty) history
    history = []
    print(runn_llm("¿Qué es la Alianza?", history))