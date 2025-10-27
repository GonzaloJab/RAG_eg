"""
RAG Query System using Ollama Llama3.2 and Jina embeddings
This script allows you to ask questions in Spanish about your vectorized PDF.
"""

import ollama
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configuration
PERSIST_DIR = "chroma_book_db_ollama"
COLLECTION = "book_index_ollama"

# Ollama embeddings class (same as in vectorization)
class OllamaEmbeddings:
    def __init__(self, model_name='jina/jina-embeddings-v2-base-es'):
        self.model_name = model_name
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents using Ollama's Jina model."""
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
        """Generate embedding for a single query text."""
        try:
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embedding for query: {e}")
            return [0.0] * 768

def load_vector_database():
    """Load the existing vector database."""
    try:
        embedding_function = OllamaEmbeddings()
        vectordb = Chroma(
            collection_name=COLLECTION,
            embedding_function=embedding_function,
            persist_directory=PERSIST_DIR,
        )
        return vectordb
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None

def retrieve_relevant_chunks(vectordb, query: str, k: int = 3):
    """Retrieve the most relevant chunks for a query."""
    try:
        # Get similar documents
        docs = vectordb.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []

def create_context(docs):
    """Create context from retrieved documents."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"Documento {i} (Página {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}")
    return "\n\n".join(context_parts)

def ask_llama(question: str, context: str, model_name: str = "llama3.2:3b"):
    """Ask Llama3.2 a question with context."""
    prompt = f"""Eres un asistente que responde preguntas basándote en el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA: {question}

INSTRUCCIONES:
- Responde en español
- Basa tu respuesta únicamente en el contexto proporcionado
- Si no puedes encontrar la respuesta en el contexto, di "No puedo encontrar información específica sobre esto en el documento"
- Sé preciso y conciso
- Si hay información relevante en múltiples partes del contexto, combínala de manera coherente

RESPUESTA:"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error al consultar el modelo: {e}"

def main():
    """Main function to run the RAG system."""
    print("🔍 Cargando base de datos vectorial...")
    vectordb = load_vector_database()
    
    if vectordb is None:
        print("❌ No se pudo cargar la base de datos. Asegúrate de haber ejecutado el script de vectorización primero.")
        return
    
    print("✅ Base de datos cargada correctamente")
    print("\n" + "="*60)
    print("🤖 Sistema RAG con Llama3.2 y Jina Embeddings")
    print("💬 Haz preguntas en español sobre tu documento")
    print("📝 Escribe 'salir' para terminar")
    print("="*60)
    
    while True:
        try:
            # Get user question
            question = input("\n❓ Tu pregunta: ").strip()
            
            if question.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break
            
            if not question:
                print("⚠️ Por favor, escribe una pregunta.")
                continue
            
            print("\n🔍 Buscando información relevante...")
            
            # Retrieve relevant chunks
            docs = retrieve_relevant_chunks(vectordb, question, k=3)
            
            if not docs:
                print("❌ No se encontraron documentos relevantes.")
                continue
            
            print(f"✅ Encontrados {len(docs)} documentos relevantes")
            
            # Create context
            context = create_context(docs)
            
            print("🤖 Generando respuesta con Llama3.2...")
            
            # Ask Llama
            answer = ask_llama(question, context)
            
            print(f"\n📄 CONTEXTO UTILIZADO:")
            print("-" * 40)
            for i, doc in enumerate(docs, 1):
                print(f"📄 Documento {i} (Página {doc.metadata.get('page', 'N/A')}):")
                print(f"   {doc.page_content[:200]}...")
                print()
            
            print(f"\n🤖 RESPUESTA:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
