from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import ollama

PDF_PATH = r"E:\langchain-course\0_RAG_eg\libro_searchable_2.pdf"
PERSIST_DIR = "chroma_book_db_ollama"
COLLECTION = "book_index_ollama"

# 1) Extract text per page (readable PDF)
reader = PdfReader(PDF_PATH)
pages = []
for i, page in enumerate(reader.pages, start=1):
    txt = page.extract_text() or ""  # pypdf per-page text extraction
    if txt.strip():
        pages.append(Document(page_content=txt, metadata={"source": PDF_PATH, "page": i}))

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(pages)

# 3) Ollama embeddings using Jina model
class OllamaEmbeddings:
    def __init__(self, model_name='jina/jina-embeddings-v2-base-es'):
        self.model_name = model_name
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of documents using Ollama's Jina model.
        """
        embeddings = []
        for text in texts:
            try:
                # Use Ollama to generate embeddings with the Jina model
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                # Extract the embedding vector
                embedding = response['embedding']
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for text: {e}")
                # Fallback: create a zero vector of appropriate size (768 for Jina)
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a single query text.
        """
        try:
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embedding for query: {e}")
            return [0.0] * 768

# Create the embedding function
embedding_function = OllamaEmbeddings()

# 4) Create / persist local Chroma index
vectordb = Chroma(
    collection_name=COLLECTION,
    embedding_function=embedding_function,        # Use the OllamaEmbeddings class
    persist_directory=PERSIST_DIR,        # local on-disk database
)
vectordb.add_texts(
    [d.page_content for d in chunks],
    metadatas=[d.metadata for d in chunks],
)

print(f"Indexed {len(chunks)} chunks into {PERSIST_DIR!r} using Jina embeddings")
