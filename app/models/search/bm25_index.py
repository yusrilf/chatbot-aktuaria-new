"""BM25 indexing functionality for keyword-based search.

This module provides BM25 indexing capabilities for efficient
keyword-based document retrieval with configurable parameters.
"""

import logging
from typing import List, Tuple, Optional

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None
    logging.getLogger(__name__).warning(
        "Optional dependency 'rank_bm25' not available. BM25Index will be disabled."
    )

from langchain_core.documents import Document

from .text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 Index untuk keyword-based search.
    
    This class provides BM25 indexing functionality with configurable
    parameters for term frequency saturation and document length normalization.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 index with configurable parameters.
        
        Args:
            k1: Parameter BM25 untuk term frequency saturation (default: 1.5)
            b: Parameter BM25 untuk document length normalization (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.preprocessor = TextPreprocessor()
        self.bm25 = None  # type: Optional[BM25Okapi]
        self.documents: List[Document] = []
        self.tokenized_docs: List[List[str]] = []
        
        logger.info(f"BM25Index initialized with k1={k1}, b={b}")
        
    def build_index(self, documents: List[Document]) -> None:
        """Membangun BM25 index dari dokumen.
        
        Args:
            documents: List dokumen untuk diindeks
            
        Raises:
            Exception: If indexing fails
        """
        try:
            if BM25Okapi is None:
                logger.warning("BM25Okapi unavailable; skipping BM25 index build.")
                self.bm25 = None
                self.documents = documents or []
                self.tokenized_docs = []
                return
            
            self.documents = documents
            self.tokenized_docs = []
            
            logger.info(f"Building BM25 index for {len(documents)} documents")
            
            for doc in documents:
                # Gabungkan content dan metadata yang relevan
                text_content = doc.page_content
                
                # Tambahkan filename ke content untuk meningkatkan relevansi
                if 'filename' in (doc.metadata or {}):
                    filename = str(doc.metadata.get('filename', '')).replace('.md', '').replace('_', ' ')
                    text_content = f"{filename} {text_content}"
                
                # Preprocess dan tokenize
                tokens = self.preprocessor.preprocess(text_content)
                self.tokenized_docs.append(tokens)
            
            # Build BM25 index
            if self.tokenized_docs:
                self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
                logger.info("BM25 index built successfully")
            else:
                logger.warning("No tokenized documents available for BM25 indexing")
                
        except Exception as e:
            logger.error(f"Error building BM25 index: {str(e)}")
            self.bm25 = None
            raise
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Pencarian menggunakan BM25.
        
        Args:
            query: Query pencarian
            top_k: Jumlah hasil teratas (default: 10)
            
        Returns:
            List tuple (Document, BM25_score) sorted by relevance
            
        Raises:
            ValueError: If query is empty or invalid
        """
        try:
            if not self.bm25 or not self.documents:
                logger.warning("BM25 index not available")
                return []
            
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            
            # Preprocess query
            query_tokens = self.preprocessor.preprocess(query)
            if not query_tokens:
                logger.warning("Empty query after preprocessing")
                return []
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Create results with scores
            results = [(self.documents[i], float(score)) for i, score in enumerate(scores)]
            
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            top_results = results[:top_k]
            
            logger.info(f"BM25 search returned {len(top_results)} results for query: {query[:50]}...")
            return top_results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            return []
    
    def get_index_stats(self) -> dict:
        """Get statistics about the BM25 index.
        
        Returns:
            Dictionary containing index statistics
        """
        return {
            'total_documents': len(self.documents),
            'total_tokens': sum(len(tokens) for tokens in self.tokenized_docs),
            'average_doc_length': sum(len(tokens) for tokens in self.tokenized_docs) / len(self.tokenized_docs) if self.tokenized_docs else 0,
            'is_built': self.bm25 is not None,
            'parameters': {
                'k1': self.k1,
                'b': self.b
            }
        }
    
    def clear_index(self) -> None:
        """Clear the BM25 index and free memory."""
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        logger.info("BM25 index cleared")
    
    def is_indexed(self) -> bool:
        """Check if the index is built and ready for search.
        
        Returns:
            True if index is built, False otherwise
        """
        return self.bm25 is not None and len(self.documents) > 0