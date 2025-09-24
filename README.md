# RAG System with Ollama and ChromaDB

A Retrieval-Augmented Generation (RAG) system that processes PDF documents using OCR, creates vector embeddings with Ollama's Jina model, and enables question-answering with Llama3.2.

## 🚀 Features

- **PDF OCR Processing**: Convert scanned PDFs to searchable text using `ocrmypdf`
- **Vector Embeddings**: Generate embeddings using Ollama's Jina model (Spanish-optimized)
- **Vector Database**: Store and retrieve document chunks using ChromaDB
- **Question Answering**: Query documents using Llama3.2 with context-aware responses
- **Spanish Language Support**: Optimized for Spanish text processing

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Tesseract OCR (for PDF processing)

### Ollama Models Required
Before running the system, ensure you have these models installed in Ollama:

```bash
# Install Jina embeddings model (Spanish)
ollama pull jina/jina-embeddings-v2-base-es

# Install Llama3.2 for question answering
ollama pull llama3.2:3b
```

### Tesseract OCR Installation
- **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd 0_RAG_eg
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   ```bash
   # Create .env file if needed
   touch .env
   ```

## 📁 Project Structure

```
0_RAG_eg/
├── 0_ocr_pdf.py              # OCR processing for PDFs
├── 1_vectorizePDF_ollama.py  # Create vector embeddings
├── 2_query_rag_ollama.py     # RAG query system
├── core.py                   # Core RAG functionality
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── docs/                     # PDF documents (ignored by git)
├── chroma_book_db_ollama/    # Vector database (ignored by git)
└── chroma_book_db_improved/  # Alternative vector database (ignored by git)
```

## 🚀 Usage

### Step 1: OCR Processing
Convert scanned PDFs to searchable text:

```bash
python 0_ocr_pdf.py
```

This will:
- Process all PDFs in the `docs/` folder
- Create searchable versions with `_searchable.pdf` suffix
- Extract text to `.txt` files

### Step 2: Vectorize Documents
Create vector embeddings and store in ChromaDB:

```bash
python 1_vectorizePDF_ollama.py
```

This will:
- Extract text from the searchable PDF
- Split text into chunks
- Generate embeddings using Jina model
- Store in ChromaDB for retrieval

### Step 3: Query the RAG System
Ask questions about your documents:

```bash
python 2_query_rag_ollama.py
```

Interactive features:
- Ask questions in Spanish
- Get context-aware answers
- View source documents used for answers
- Type 'salir' to exit

### Alternative: Core RAG Function
Use the core RAG functionality programmatically:

```python
from core import runn_llm

# Ask a question
answer = runn_llm("¿Qué es la Alianza?")
print(answer)
```

## 🔧 Configuration

### PDF Path Configuration
Update the PDF path in `1_vectorizePDF_ollama.py`:
```python
PDF_PATH = r"path/to/your/document.pdf"
```

### Database Configuration
Modify database settings in the scripts:
```python
PERSIST_DIR = "chroma_book_db_ollama"  # Database directory
COLLECTION = "book_index_ollama"       # Collection name
```

### Model Configuration
Change the models used:
```python
# For embeddings
model_name = 'jina/jina-embeddings-v2-base-es'

# For question answering
model_name = "llama3.2:3b"
```

## 📊 How It Works

1. **Document Processing**: PDFs are processed with OCR to extract text
2. **Text Chunking**: Documents are split into manageable chunks (1000 chars, 150 overlap)
3. **Embedding Generation**: Each chunk is converted to vector embeddings using Jina
4. **Vector Storage**: Embeddings are stored in ChromaDB for fast retrieval
5. **Query Processing**: User questions are embedded and matched against stored vectors
6. **Context Retrieval**: Most relevant chunks are retrieved as context
7. **Answer Generation**: Llama3.2 generates answers based on retrieved context

## 🐛 Troubleshooting

### Common Issues

**Ollama not running:**
```bash
# Start Ollama service
ollama serve
```

**Model not found:**
```bash
# List installed models
ollama list

# Install missing models
ollama pull jina/jina-embeddings-v2-base-es
ollama pull llama3.2:3b
```

**Tesseract not found:**
- Ensure Tesseract is installed and in your PATH
- On Windows, add Tesseract to your system PATH

**ChromaDB errors:**
- Delete the database folder and re-run vectorization
- Check file permissions for the database directory

## 📝 Example Queries

Try these example questions in Spanish:
- "¿Cuál es el tema principal del documento?"
- "¿Qué información se menciona sobre [tema específico]?"
- "Resume los puntos más importantes"
- "¿Qué conclusiones se pueden extraer?"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM hosting
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [LangChain](https://langchain.com/) for RAG framework
- [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) for PDF processing
