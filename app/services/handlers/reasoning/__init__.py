"""Tree of Thought Reasoning Package.

This package contains all components for Tree of Thought (ToT) reasoning implementation:
- Query Expansion Service
- Multi-Path Generator
- Evaluation Layer
- Validation Layer
- Tree of Thought Service
"""

__version__ = "1.0.0"
__author__ = "Aktuaria Chatbot Team"

# Import main components for easy access
try:
    from .tree_of_thought_service import TreeOfThoughtService
    from .query_expansion_service import QueryExpansionService
    from .multipath_generator import MultipathReasoningGenerator as MultiPathGenerator
    from .evaluation_layer import EvaluationLayer
    from .validation_layer import ValidationLayer
    
    __all__ = [
        "TreeOfThoughtService",
        "QueryExpansionService", 
        "MultiPathGenerator",
        "EvaluationLayer",
        "ValidationLayer"
    ]
except ImportError as e:
    # Handle import errors gracefully during development
    print(f"Warning: Some ToT components not available: {e}")
    __all__ = []