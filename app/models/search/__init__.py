#!/usr/bin/env python3
"""Search package for hybrid search functionality.

This package provides comprehensive search capabilities combining
semantic search with BM25 keyword search using adaptive weighting.

Modules:
    text_preprocessor: Text preprocessing utilities for Indonesian text
    search_result: Data model for search results
    bm25_index: BM25 indexing for keyword-based search
    result_combiner: Utilities for combining search results
    hybrid_search_manager: Main hybrid search manager

Example:
    >>> from app.models.search import HybridSearchManager, SearchResult
    >>> manager = HybridSearchManager()
    >>> manager.build_index(documents)
    >>> results = manager.hybrid_search(query, semantic_results)
"""

from .text_preprocessor import TextPreprocessor
from .search_result import SearchResult
from .bm25_index import BM25Index
from .result_combiner import ResultCombiner, CombinedResult
from .hybrid_search_manager import HybridSearchManager

__all__ = [
    'TextPreprocessor',
    'SearchResult',
    'BM25Index',
    'ResultCombiner',
    'CombinedResult',
    'HybridSearchManager'
]

__version__ = '1.0.0'
__author__ = 'Actuarial Chatbot Team'
__description__ = 'Hybrid search functionality with semantic and keyword search'