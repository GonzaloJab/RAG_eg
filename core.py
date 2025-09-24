from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma

from langchain_ollama import ChatOllama
import ollama

# Custom OllamaEmbeddings class
class OllamaEmbeddings:
    def __init__(self, model_name='jina/jina-embeddings-v2-base-es'):
        self.model_name = model_name
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            try:
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                embedding = response['embedding']
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for text: {e}")
                embeddings.append([0.0] * 768)
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        try:
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embedding for query: {e}")
            return [0.0] * 768

INDEX_NAME = "book_index_improved"

def runn_llm(query: str):
    embedding_function = OllamaEmbeddings()
    docsearch = Chroma(
        collection_name=INDEX_NAME,
        embedding_function=embedding_function,
        persist_directory="chroma_book_db_improved"
    )

    llm = ChatOllama(model="llama3.2:3b")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=stuff_documents_chain
    )
    response = qa.invoke({"input": query})

    return response['answer']

if __name__ == "__main__":
    print(runn_llm("¿Qué es la Alianza?"))