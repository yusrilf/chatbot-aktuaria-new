#!/usr/bin/env python3
"""Configuration Management for Actuarial Chatbot.

This module provides centralized configuration management for the actuarial
chatbot application, including API keys, database settings, and optimization
parameters.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for actuarial chatbot application.
    
    This class centralizes all configuration settings including:
    - API keys and model configurations
    - Database and vector store settings
    - Document processing parameters
    - Hybrid search optimization settings
    - Logging configuration
    
    All settings can be overridden via environment variables.
    """
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    COHERE_RERANK_MODEL = os.getenv('COHERE_RERANK_MODEL', 'rerank-multilingual-v3.0')
    OPENAI_MODEL = 'gpt-4.1'
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
    
    # ChromaDB Settings
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './data/vectorstore')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'actuarial_documents')
    
    # Vector Backend Toggle
    VECTOR_BACKEND = os.getenv('VECTOR_BACKEND', 'pinecone')
    
    # Pinecone Settings
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'aktuaria-docs')
    PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE', 'default')
    PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
    PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')
    PINECONE_DIMENSION = int(os.getenv('PINECONE_DIMENSION', '3072'))
    PINECONE_METRIC = os.getenv('PINECONE_METRIC', 'cosine')
    
    # Flask Settings
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    PORT = int(os.getenv('FLASK_PORT', '5001'))  # Default port 5001 (avoids macOS AirPlay conflict)
    
    # Document Processing
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    
    # Chat Settings
    MAX_CONTEXT_LENGTH = 500  # Reduced for more concise responses
    SIMILARITY_THRESHOLD = 0.70
    TOP_K_RESULTS = 15
    MEMORY_WINDOW_SIZE = int(os.getenv('MEMORY_WINDOW_SIZE', '6'))
    
    # Vector Search Configuration - Pure vector search with reranker
    VECTOR_SEARCH_ENABLED = True  # Pure vector search enabled
    
    # Score filtering thresholds
    MIN_RELEVANCE_SCORE = float(os.getenv('MIN_RELEVANCE_SCORE', '0.10'))            # Minimum score for relevance (lowered for better retrieval)
    RETRIEVAL_MULTIPLIER = int(os.getenv('RETRIEVAL_MULTIPLIER', '3'))               # Multiply k for initial retrieval
    
    # Score normalization settings
    ENABLE_SCORE_NORMALIZATION = os.getenv('ENABLE_SCORE_NORMALIZATION', 'True').lower() == 'true'
    NORMALIZED_SCORE_MIN = float(os.getenv('NORMALIZED_SCORE_MIN', '0.1'))           # Minimum normalized score
    NORMALIZED_SCORE_MAX = float(os.getenv('NORMALIZED_SCORE_MAX', '1.0'))           # Maximum normalized score
    
    # Relevance categorization thresholds (for normalized scores)
    HIGH_RELEVANCE_THRESHOLD = float(os.getenv('HIGH_RELEVANCE_THRESHOLD', '0.7'))   # High relevance threshold
    MEDIUM_RELEVANCE_THRESHOLD = float(os.getenv('MEDIUM_RELEVANCE_THRESHOLD', '0.5')) # Medium relevance threshold
    
    # Reranking Settings
    ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'True').lower() == 'true'
    
    # Hybrid Retrieval Settings
    HYBRID_SEMANTIC_WEIGHT = float(os.getenv('HYBRID_SEMANTIC_WEIGHT', '0.7'))       # Weight for semantic retrieval
    HYBRID_BM25_WEIGHT = float(os.getenv('HYBRID_BM25_WEIGHT', '0.3'))               # Weight for BM25 retrieval
    BM25_K1 = float(os.getenv('BM25_K1', '1.2'))                                     # BM25 k1 parameter
    BM25_B = float(os.getenv('BM25_B', '0.75'))                                      # BM25 b parameter
    HYBRID_NORMALIZE_SCORES = os.getenv('HYBRID_NORMALIZE_SCORES', 'True').lower() == 'true'  # Normalize scores
    HYBRID_MIN_SCORE_THRESHOLD = float(os.getenv('HYBRID_MIN_SCORE_THRESHOLD', '0.0'))  # Minimum score threshold
    
    # Default Values
    DEFAULT_K = 5
    DEFAULT_THRESHOLD = 0.7
    INITIAL_RETRIEVAL_K = 20
    RETRIEVAL_MULTIPLIER = 2

    # Specific K values for different use cases
    CALCULATION_SEARCH_K = 8  # For calculation handler hybrid search
    CALCULATION_FALLBACK_K = 3  # For calculation handler fallback search
    PLANNER_SEARCH_K = 50  # For structured planner search
    PLANNER_KEY_DOCS_K = 2  # For structured planner key documents
    DOCUMENT_LISTING_K = 1000  # For document listing operations

    # Threshold configurations
    VECTOR_SEARCH_THRESHOLD = 0.7  # For vector search similarity threshold
    CALCULATION_THRESHOLD = 0.6  # For calculation-specific threshold

    # Multiplier configurations
    HYBRID_SEARCH_MULTIPLIER = 2  # For hybrid search K multiplication
    
    # Dynamic Retrieval Settings
    ENABLE_DYNAMIC_RETRIEVAL = os.getenv('ENABLE_DYNAMIC_RETRIEVAL', 'True').lower() == 'true'  # Enable dynamic retrieval based on threshold
    DYNAMIC_RETRIEVAL_THRESHOLD = float(os.getenv('DYNAMIC_RETRIEVAL_THRESHOLD', '0.1'))        # Threshold for dynamic retrieval (10% - lowered for better retrieval)
    DYNAMIC_RETRIEVAL_MAX_DOCS = int(os.getenv('DYNAMIC_RETRIEVAL_MAX_DOCS', '50'))             # Maximum documents to return in dynamic mode
    DYNAMIC_RETRIEVAL_MIN_DOCS = int(os.getenv('DYNAMIC_RETRIEVAL_MIN_DOCS', '2'))              # Minimum documents to return in dynamic mode (lowered from 3)
    
    # Performance Metrics Settings
    ENABLE_PERFORMANCE_METRICS = os.getenv('ENABLE_PERFORMANCE_METRICS', 'False').lower() == 'true'  # Disable by default for production
    PERFORMANCE_METRICS_IN_RESPONSE = os.getenv('PERFORMANCE_METRICS_IN_RESPONSE', 'False').lower() == 'true'  # Include in API responses
    PERFORMANCE_LOGGING_ENABLED = os.getenv('PERFORMANCE_LOGGING_ENABLED', 'True').lower() == 'true'  # Log performance metrics
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

config = Config()

