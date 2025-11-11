"""Core RAG Configuration - Simplified for Experiments

This module contains essential configuration for running RAG experiments.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGConfig:
    """Simplified RAG configuration for experiments."""

    # OpenAI Settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')

    # Chunking Settings
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))  # tokens
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '150'))  # tokens
    CHARS_PER_TOKEN = 4  # estimation ratio

    # Vector Store Settings
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './experiments_data/vectorstore')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'rag_experiments')

    # Retrieval Settings
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '5'))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))

    # Hybrid Search Settings
    SEMANTIC_WEIGHT = float(os.getenv('SEMANTIC_WEIGHT', '0.7'))
    BM25_WEIGHT = float(os.getenv('BM25_WEIGHT', '0.3'))
    BM25_K1 = float(os.getenv('BM25_K1', '1.2'))
    BM25_B = float(os.getenv('BM25_B', '0.75'))

    # RAG Settings
    MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', '8000'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))

config = RAGConfig()
