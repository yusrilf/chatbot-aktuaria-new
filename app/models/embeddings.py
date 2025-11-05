"""Embeddings module - refactored into smaller components.

This module provides a backward-compatible interface to the refactored
embeddings components while maintaining the original API.
"""

# Import all components from the refactored modules
from .embeddings.vector_store_manager import VectorStoreManager
from .embeddings.document_manager import DocumentManager
from .embeddings.search_manager import SearchManager
from .embeddings.psak219_manager import PSAK219Manager

# Maintain backward compatibility by exposing the main class
__all__ = [
    'VectorStoreManager',
    'DocumentManager',
    'SearchManager', 
    'PSAK219Manager'
]

# The VectorStoreManager now delegates to the component managers
# All original functionality is preserved through the modular structure