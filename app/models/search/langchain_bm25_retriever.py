"""Langchain BM25 Retriever Implementation.

This module provides BM25 retrieval functionality using Langchain's
BM25Retriever with optimized configuration for actuarial documents.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

logger = logging.getLogger(__name__)


class LangchainBM25Retriever:
    """Enhanced BM25 Retriever using Langchain with optimized configuration.
    
    This class provides BM25 retrieval functionality specifically optimized
    for actuarial documents with preprocessing and scoring enhancements.
    """
    
    def __init__(self, 
                 k1: float = 1.2,
                 b: float = 0.75,
                 preprocess_func: Optional[callable] = None,
                 k: int = 10) -> None:
        """Initialize Langchain BM25 Retriever.
        
        Args:
            k1: BM25 parameter controlling term frequency saturation (default: 1.2)
            b: BM25 parameter controlling document length normalization (default: 0.75)
            preprocess_func: Optional preprocessing function for documents
            k: Default number of documents to retrieve
        """
        self.k1 = k1
        self.b = b
        self.k = k
        self.preprocess_func = preprocess_func or self._default_preprocess
        self.retriever: Optional[BM25Retriever] = None
        self.documents: List[Document] = []
        self.is_built = False
        
        logger.info(f"LangchainBM25Retriever initialized with k1={k1}, b={b}, k={k}")
    
    def _default_preprocess(self, text: str) -> str:
        """Default preprocessing function for text.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        try:
            # Basic preprocessing: lowercase and strip
            processed = text.lower().strip()
            
            # Remove extra whitespace
            processed = ' '.join(processed.split())
            
            return processed
            
        except Exception as e:
            logger.warning(f"Error in text preprocessing: {e}")
            return text
    
    def build_retriever(self, documents: List[Document]) -> None:
        """Build BM25 retriever from documents.
        
        Args:
            documents: List of documents to index
            
        Raises:
            Exception: If retriever building fails
        """
        try:
            if not documents:
                logger.warning("No documents provided for BM25 retriever")
                return
            
            logger.info(f"Building BM25 retriever for {len(documents)} documents")
            
            # Preprocess documents
            processed_docs = []
            for doc in documents:
                try:
                    # Create a copy to avoid modifying original
                    processed_doc = Document(
                        page_content=self.preprocess_func(doc.page_content),
                        metadata=doc.metadata.copy()
                    )
                    processed_docs.append(processed_doc)
                except Exception as e:
                    logger.warning(f"Error preprocessing document: {e}")
                    processed_docs.append(doc)  # Use original if preprocessing fails
            
            # Create BM25 retriever with custom parameters
            self.retriever = BM25Retriever.from_documents(
                processed_docs,
                k=self.k,
                preprocess_func=self.preprocess_func
            )
            
            # Store original documents for metadata preservation
            self.documents = documents
            self.is_built = True
            
            logger.info("BM25 retriever built successfully")
            
        except Exception as e:
            logger.error(f"Error building BM25 retriever: {str(e)}")
            self.is_built = False
            raise
    
    def retrieve(self, 
                query: str, 
                k: Optional[int] = None,
                filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Retrieve documents using BM25.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (uses default if None)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved documents
            
        Raises:
            ValueError: If retriever is not built or query is invalid
        """
        # Validate inputs first
        if not self.is_built or not self.retriever:
            raise ValueError("BM25 retriever not built. Call build_retriever() first.")
        
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        try:
            # Set k for this retrieval
            retrieval_k = k or self.k
            original_k = self.retriever.k
            self.retriever.k = retrieval_k
            
            try:
                # Perform retrieval
                results = self.retriever.get_relevant_documents(query)
                
                # Apply metadata filtering if specified
                if filter_metadata and results:
                    filtered_results = []
                    for doc in results:
                        match = True
                        for key, value in filter_metadata.items():
                            if key not in doc.metadata or doc.metadata[key] != value:
                                match = False
                                break
                        if match:
                            filtered_results.append(doc)
                    results = filtered_results
                
                logger.info(f"BM25 retrieval returned {len(results)} documents for query: '{query[:50]}...'")
                return results
                
            finally:
                # Restore original k
                self.retriever.k = original_k
            
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {str(e)}")
            return []
    
    def retrieve_with_scores(self, 
                           query: str, 
                           k: Optional[int] = None,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Retrieve documents with BM25 scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            List of tuples (Document, BM25_score)
        """
        # Validate inputs first
        if not self.is_built or not self.retriever:
            raise ValueError("BM25 retriever not built. Call build_retriever() first.")
        
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        try:
            # Get documents without scores first
            documents = self.retrieve(query, k, filter_metadata)
            
            if not documents:
                return []
            
            # Calculate BM25 scores manually using the underlying BM25 object
            try:
                # Access the BM25 object from the retriever
                bm25_obj = self.retriever.vectorizer
                
                # Preprocess query
                processed_query = self.preprocess_func(query)
                query_tokens = processed_query.split()
                
                # Get scores for all documents
                all_scores = bm25_obj.get_scores(query_tokens)
                
                # Create document-score pairs
                doc_score_pairs = []
                for i, doc in enumerate(documents):
                    # Find the document index in the original corpus
                    doc_index = -1
                    for j, orig_doc in enumerate(self.documents):
                        if (doc.page_content == orig_doc.page_content and 
                            doc.metadata == orig_doc.metadata):
                            doc_index = j
                            break
                    
                    if doc_index >= 0 and doc_index < len(all_scores):
                        score = float(all_scores[doc_index])
                    else:
                        score = 0.0  # Default score if not found
                    
                    doc_score_pairs.append((doc, score))
                
                # Sort by score (descending)
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                
                logger.info(f"BM25 retrieval with scores returned {len(doc_score_pairs)} documents")
                return doc_score_pairs
                
            except Exception as score_error:
                logger.warning(f"Could not calculate BM25 scores: {score_error}")
                # Return documents with default scores
                return [(doc, 1.0) for doc in documents]
            
        except Exception as e:
            logger.error(f"Error in BM25 retrieval with scores: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics.
        
        Returns:
            Dictionary containing retriever statistics
        """
        return {
            'is_built': self.is_built,
            'document_count': len(self.documents),
            'k1': self.k1,
            'b': self.b,
            'default_k': self.k,
            'has_retriever': self.retriever is not None
        }