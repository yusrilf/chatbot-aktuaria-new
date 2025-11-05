"""Hybrid search functionality combining semantic and BM25 search.

This module provides backward compatibility imports for the refactored
hybrid search components. The actual implementation has been moved to
the app.models.search package for better organization.
"""

import logging
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document

# Import from refactored modules
from .search import (
    TextPreprocessor,
    SearchResult,
    BM25Index,
    HybridSearchManager,
    ResultCombiner,
    CombinedResult
)
from .query_classifier import QueryClassifier

logger = logging.getLogger(__name__)

# Backward compatibility aliases
class HybridSearch(HybridSearchManager):
    """Backward compatibility alias for HybridSearchManager.
    
    Deprecated: Use HybridSearchManager from app.models.search instead.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning."""
        logger.warning(
            "HybridSearch is deprecated. Use HybridSearchManager from app.models.search instead."
        )
        super().__init__(*args, **kwargs)


# Legacy function for backward compatibility
def create_hybrid_search_manager(semantic_weight: float = 0.85,
                                bm25_weight: float = 0.15,
                                **kwargs) -> HybridSearchManager:
    """Create a hybrid search manager with default settings.
    
    Args:
        semantic_weight: Weight for semantic search results
        bm25_weight: Weight for BM25 search results
        **kwargs: Additional arguments for HybridSearchManager
        
    Returns:
        Configured HybridSearchManager instance
    """
    return HybridSearchManager(
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
        **kwargs
    )


# Export main classes for backward compatibility
__all__ = [
    'TextPreprocessor',
    'SearchResult', 
    'BM25Index',
    'HybridSearchManager',
    'HybridSearch',  # Deprecated
    'ResultCombiner',
    'CombinedResult',
    'create_hybrid_search_manager'
]