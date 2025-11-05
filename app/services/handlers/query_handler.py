"""Query Handler - Main Interface for Query Processing.

This module provides backward compatibility and acts as the main interface
for the refactored query processing system.
"""

# Import all components from the refactored query processing modules
from .query_processing.query_handler import QueryHandler
from .query_processing.external_query_processor import ExternalQueryProcessor
from .query_processing.datastory_query_processor import DatastoryQueryProcessor
from .query_processing.custom_query_processor import CustomQueryProcessor
from .query_processing.psak219_query_processor import PSAK219QueryProcessor
from .query_processing.retrieval_manager import RetrievalManager

# Export all classes for backward compatibility
__all__ = [
    'QueryHandler',
    'ExternalQueryProcessor',
    'DatastoryQueryProcessor', 
    'CustomQueryProcessor',
    'PSAK219QueryProcessor',
    'RetrievalManager'
]

# For direct imports (backward compatibility)
from .query_processing.query_handler import QueryHandler as QueryHandler