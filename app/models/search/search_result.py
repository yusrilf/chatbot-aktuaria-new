"""Search result data model for hybrid search functionality.

This module defines the SearchResult dataclass used to represent
search results with relevance scores and metadata.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class SearchResult:
    """Represents a search result with content and relevance metrics.
    
    Attributes:
        content: The text content of the search result
        score: Relevance score (0.0 to 1.0)
        source: Source identifier or document name
        metadata: Additional metadata about the result
        rank: Position in search results (optional)
    """
    
    content: str
    score: float
    source: str
    metadata: Optional[Dict[str, Any]] = None
    rank: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate search result data after initialization.
        
        Raises:
            ValueError: If score is not between 0.0 and 1.0
        """
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")
        
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary format.
        
        Returns:
            Dictionary representation of the search result
        """
        return {
            'content': self.content,
            'score': self.score,
            'source': self.source,
            'metadata': self.metadata,
            'rank': self.rank
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create SearchResult from dictionary data.
        
        Args:
            data: Dictionary containing search result data
            
        Returns:
            SearchResult instance
        """
        return cls(
            content=data['content'],
            score=data['score'],
            source=data['source'],
            metadata=data.get('metadata'),
            rank=data.get('rank')
        )
    
    def __str__(self) -> str:
        """String representation of search result.
        
        Returns:
            Formatted string with score and content preview
        """
        content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"SearchResult(score={self.score:.3f}, source={self.source}, content='{content_preview}')"