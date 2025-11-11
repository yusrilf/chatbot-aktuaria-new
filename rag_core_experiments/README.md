# RAG Core Experiments

This folder contains the **core RAG (Retrieval-Augmented Generation) components** extracted from the fullstack Actuarial Chatbot system. The code has been simplified and optimized for experimentation in Jupyter notebooks.

## 📁 Project Structure

```
rag_core_experiments/
├── config.py                    # Configuration settings
├── document_loader.py           # Document loading & parsing
├── chunker.py                   # Semantic chunking logic
├── embedding_service.py         # Embedding generation & vector storage
├── rag_pipeline.py             # Complete RAG pipeline
├── RAG_Experiments.ipynb       # Main experiment notebook
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎯 Core Components

### 1. **Document Loading** (`document_loader.py`)
- Load markdown documents with YAML front-matter
- Extract metadata from documents
- Support for multiple file formats

### 2. **Semantic Chunking** (`chunker.py`)
- Split documents into semantic chunks
- Configurable chunk size and overlap
- Token-based chunking with metadata preservation

### 3. **Embedding & Vector Storage** (`embedding_service.py`)
- Generate embeddings using OpenAI
- Store embeddings in ChromaDB
- Similarity search functionality

### 4. **RAG Pipeline** (`rag_pipeline.py`)
- Complete question-answering pipeline
- Retrieval + Generation workflow
- Context assembly and answer generation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd rag_core_experiments
pip install -r requirements.txt
```

### 2. Setup Environment Variables

Create a `.env` file in this directory:

```bash
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Override default settings
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

### 3. Run the Experiment Notebook

```bash
jupyter notebook RAG_Experiments.ipynb
```

## 📊 RAG Pipeline Flow

```
1. Document Loading
   └─> Load markdown files with metadata

2. Chunking
   └─> Split into semantic chunks (1000 tokens, 150 overlap)

3. Embedding
   └─> Generate embeddings (OpenAI text-embedding-3-large)

4. Vector Indexing
   └─> Store in ChromaDB vector database

5. Retrieval
   └─> Similarity search for relevant chunks

6. Generation
   └─> Generate answer using LLM with retrieved context
```

## 💡 Usage Examples

### Basic RAG Query

```python
from config import config
from document_loader import DocumentLoader
from chunker import SemanticChunker
from embedding_service import EmbeddingService
from rag_pipeline import RAGPipeline

# 1. Load documents
loader = DocumentLoader()
documents = loader.load_documents(['path/to/document.md'])

# 2. Chunk documents
chunker = SemanticChunker()
chunks = []
for doc in documents:
    doc_chunks = chunker.chunk_document(
        content=doc.content,
        metadata=doc.metadata,
        doc_name=doc.filename
    )
    chunks.extend(doc_chunks)

# 3. Initialize embedding service & add chunks
embedding_service = EmbeddingService()
embedding_service.add_chunks(chunks)

# 4. Create RAG pipeline
rag = RAGPipeline(embedding_service=embedding_service)

# 5. Ask questions
response = rag.query("What is actuarial science?", k=5)
print(response['answer'])
```

### Custom Configuration

```python
# Custom chunking
chunker = SemanticChunker(
    max_chunk_size=1500,
    overlap_size=200
)

# Custom embedding service
embedding_service = EmbeddingService(
    embedding_model='text-embedding-3-small',
    collection_name='my_experiments'
)

# Custom RAG pipeline
rag = RAGPipeline(
    embedding_service=embedding_service,
    llm_model='gpt-4',
    temperature=0.0
)
```

## 🧪 Experiments to Try

1. **Chunking Strategies**
   - Compare different chunk sizes (500, 1000, 1500 tokens)
   - Test different overlap sizes
   - Analyze chunk size vs retrieval quality

2. **Embedding Models**
   - Compare OpenAI embedding models
   - Test retrieval accuracy

3. **Retrieval Methods**
   - Pure semantic search
   - Hybrid search (semantic + BM25)
   - Different `k` values for top-k retrieval

4. **LLM Models**
   - Compare different OpenAI models (GPT-3.5, GPT-4, etc.)
   - Test with different temperatures

## 🔑 Key Differences from Full System

This core version simplifies the fullstack system by:

- ✅ Removing Flask API endpoints
- ✅ Removing complex orchestration layers
- ✅ Removing session management
- ✅ Removing advanced features (reranking, verification, etc.)
- ✅ Focusing on core RAG functionality

**What's preserved:**
- ✅ Document loading & parsing
- ✅ Semantic chunking with overlap
- ✅ Embedding generation
- ✅ Vector storage (ChromaDB)
- ✅ Similarity search
- ✅ RAG question answering

## 📝 Configuration Options

See `config.py` for all available settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Maximum chunk size in tokens |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks in tokens |
| `EMBEDDING_MODEL` | text-embedding-3-large | OpenAI embedding model |
| `OPENAI_MODEL` | gpt-4o-mini | LLM model for generation |
| `TOP_K_RESULTS` | 5 | Number of chunks to retrieve |
| `MAX_CONTEXT_LENGTH` | 8000 | Maximum context length for LLM |

## 🐛 Troubleshooting

### ChromaDB Error
```python
# Clear the collection if you get errors
embedding_service.clear_collection()
```

### Token Limit Exceeded
```python
# Reduce max_context_length in config.py
MAX_CONTEXT_LENGTH = 4000
```

### No Results Found
```python
# Check if documents were added
stats = embedding_service.get_collection_stats()
print(stats)
```

## 📚 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

## 🤝 Contributing

Feel free to experiment and extend this codebase:
- Add new chunking strategies
- Implement BM25 hybrid search
- Add evaluation metrics (precision, recall, NDCG)
- Integrate with other vector databases

## 📄 License

Extracted from the Actuarial Chatbot project for educational and experimental purposes.
