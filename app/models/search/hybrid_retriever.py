"""Hybrid Retriever Implementation.

This module provides hybrid retrieval functionality that combines
semantic embedding retrieval and BM25 retrieval with weighted scoring.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from .langchain_bm25_retriever import LangchainBM25Retriever
import numpy as np

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining semantic and BM25 retrieval.
    
    This class combines semantic embedding retrieval and BM25 retrieval
    with configurable weighting and advanced scoring mechanisms.
    """
    
    def __init__(self,
                 vector_store: VectorStore,
                 semantic_weight: float = 0.7,
                 bm25_weight: float = 0.3,
                 bm25_k1: float = 1.2,
                 bm25_b: float = 0.75,
                 normalize_scores: bool = True,
                 min_score_threshold: float = 0.0) -> None:
        """Initialize Hybrid Retriever.
        
        Args:
            vector_store: Vector store for semantic retrieval
            semantic_weight: Weight for semantic retrieval scores (0.0-1.0)
            bm25_weight: Weight for BM25 retrieval scores (0.0-1.0)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            normalize_scores: Whether to normalize scores before combining
            min_score_threshold: Minimum score threshold for results
        
        Raises:
            ValueError: If weights don't sum to 1.0 or are invalid
        """
        # Validate weights
        if not (0.0 <= semantic_weight <= 1.0 and 0.0 <= bm25_weight <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        
        if abs(semantic_weight + bm25_weight - 1.0) > 1e-6:
            logger.warning(f"Weights don't sum to 1.0: {semantic_weight + bm25_weight}. Normalizing...")
            total = semantic_weight + bm25_weight
            semantic_weight = semantic_weight / total
            bm25_weight = bm25_weight / total
        
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.normalize_scores = normalize_scores
        self.min_score_threshold = min_score_threshold
        
        # Initialize BM25 retriever
        self.bm25_retriever = LangchainBM25Retriever(
            k1=bm25_k1,
            b=bm25_b
        )
        
        self.is_built = False
        self.documents: List[Document] = []
        
        logger.info(f"HybridRetriever initialized with semantic_weight={semantic_weight}, "
                   f"bm25_weight={bm25_weight}")
    
    def build_index(self, documents: List[Document]) -> None:
        """Build indices for both semantic and BM25 retrieval.
        
        Args:
            documents: List of documents to index
            
        Raises:
            Exception: If index building fails
        """
        try:
            if not documents:
                logger.warning("No documents provided for hybrid retriever")
                return
            
            logger.info(f"Building hybrid index for {len(documents)} documents")
            
            # Store documents
            self.documents = documents
            
            # Build BM25 index
            self.bm25_retriever.build_retriever(documents)
            
            # Vector store should already be built/populated
            # We assume it contains the same documents
            
            self.is_built = True
            logger.info("Hybrid index built successfully")
            
        except Exception as e:
            logger.error(f"Error building hybrid index: {str(e)}")
            self.is_built = False
            raise
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization.
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            List of normalized scores
        """
        if not scores:
            return scores
        
        try:
            scores_array = np.array(scores)
            
            # Handle edge cases
            if len(scores) == 1:
                return [1.0]
            
            min_score = np.min(scores_array)
            max_score = np.max(scores_array)
            
            # Avoid division by zero
            if max_score == min_score:
                return [1.0] * len(scores)
            
            # Min-max normalization
            normalized = (scores_array - min_score) / (max_score - min_score)
            return normalized.tolist()
            
        except Exception as e:
            logger.warning(f"Error normalizing scores: {e}")
            return scores
    
    def _combine_results(self, 
                        semantic_results: List[Tuple[Document, float]],
                        bm25_results: List[Tuple[Document, float]],
                        k: int) -> List[Tuple[Document, float]]:
        """Combine semantic and BM25 results with weighted scoring.
        
        Args:
            semantic_results: Results from semantic retrieval
            bm25_results: Results from BM25 retrieval
            k: Number of final results to return
            
        Returns:
            Combined and ranked results
        """
        try:
            # Create document to score mappings
            semantic_scores = {}
            bm25_scores = {}
            all_docs = set()
            
            # Process semantic results
            for doc, score in semantic_results:
                doc_key = (doc.page_content, str(doc.metadata))
                semantic_scores[doc_key] = score
                all_docs.add((doc_key, doc))
            
            # Process BM25 results
            for doc, score in bm25_results:
                doc_key = (doc.page_content, str(doc.metadata))
                bm25_scores[doc_key] = score
                all_docs.add((doc_key, doc))
            
            if not all_docs:
                return []
            
            # Extract scores for normalization
            semantic_score_values = list(semantic_scores.values())
            bm25_score_values = list(bm25_scores.values())
            
            # Normalize scores if enabled
            if self.normalize_scores:
                if semantic_score_values:
                    normalized_semantic = self._normalize_scores(semantic_score_values)
                    semantic_keys = list(semantic_scores.keys())
                    semantic_scores = {key: score for key, score in zip(semantic_keys, normalized_semantic)}
                
                if bm25_score_values:
                    normalized_bm25 = self._normalize_scores(bm25_score_values)
                    bm25_keys = list(bm25_scores.keys())
                    bm25_scores = {key: score for key, score in zip(bm25_keys, normalized_bm25)}
            
            # Combine scores
            combined_results = []
            for doc_key, doc in all_docs:
                semantic_score = semantic_scores.get(doc_key, 0.0)
                bm25_score = bm25_scores.get(doc_key, 0.0)
                
                # Calculate weighted combined score
                combined_score = (
                    self.semantic_weight * semantic_score + 
                    self.bm25_weight * bm25_score
                )
                
                # Apply minimum score threshold
                if combined_score >= self.min_score_threshold:
                    combined_results.append((doc, combined_score))
            
            # Sort by combined score (descending)
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            # Fallback: return semantic results if available
            if semantic_results:
                return semantic_results[:k]
            return bm25_results[:k] if bm25_results else []
    
    def retrieve(self, 
                query: str, 
                k: int = 10,
                semantic_k: Optional[int] = None,
                bm25_k: Optional[int] = None,
                filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Retrieve documents using hybrid approach.
        
        Args:
            query: Search query
            k: Number of final documents to return
            semantic_k: Number of documents to retrieve from semantic search
            bm25_k: Number of documents to retrieve from BM25 search
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        try:
            if not self.is_built:
                logger.warning("Hybrid retriever not built")
                return []
            
            if not query or not isinstance(query, str):
                logger.warning("Invalid query provided")
                return []
            
            # Set retrieval counts (retrieve more initially for better combination)
            semantic_k = semantic_k or min(k * 2, 50)
            bm25_k = bm25_k or min(k * 2, 50)
            
            logger.info(f"Hybrid retrieval for query: '{query[:50]}...' (k={k})")
            
            # Perform semantic retrieval
            semantic_results = []
            try:
                semantic_docs = self.vector_store.similarity_search_with_score(
                    query, k=semantic_k
                )
                semantic_results = [(doc, float(score)) for doc, score in semantic_docs]
                logger.info(f"Semantic retrieval returned {len(semantic_results)} documents")
            except Exception as e:
                logger.warning(f"Semantic retrieval failed: {e}")
            
            # Perform BM25 retrieval
            bm25_results = []
            try:
                bm25_results = self.bm25_retriever.retrieve_with_scores(
                    query, k=bm25_k, filter_metadata=filter_metadata
                )
                logger.info(f"BM25 retrieval returned {len(bm25_results)} documents")
            except Exception as e:
                logger.warning(f"BM25 retrieval failed: {e}")
            
            # Combine results
            combined_results = self._combine_results(semantic_results, bm25_results, k)
            
            # Extract documents
            final_docs = [doc for doc, score in combined_results]
            
            logger.info(f"Hybrid retrieval returned {len(final_docs)} final documents")
            return final_docs
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []
    
    def retrieve_with_scores(self, 
                           query: str, 
                           k: int = 10,
                           semantic_k: Optional[int] = None,
                           bm25_k: Optional[int] = None,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Retrieve documents with hybrid scores.
        
        Args:
            query: Search query
            k: Number of final documents to return
            semantic_k: Number of documents to retrieve from semantic search
            bm25_k: Number of documents to retrieve from BM25 search
            filter_metadata: Optional metadata filters
            
        Returns:
            List of tuples (Document, hybrid_score)
        """
        try:
            if not self.is_built:
                logger.warning("Hybrid retriever not built")
                return []
            
            # Set retrieval counts
            semantic_k = semantic_k or min(k * 2, 50)
            bm25_k = bm25_k or min(k * 2, 50)
            
            # Perform retrievals
            semantic_results = []
            try:
                semantic_docs = self.vector_store.similarity_search_with_score(
                    query, k=semantic_k
                )
                semantic_results = [(doc, float(score)) for doc, score in semantic_docs]
            except Exception as e:
                logger.warning(f"Semantic retrieval failed: {e}")
            
            bm25_results = []
            try:
                bm25_results = self.bm25_retriever.retrieve_with_scores(
                    query, k=bm25_k, filter_metadata=filter_metadata
                )
            except Exception as e:
                logger.warning(f"BM25 retrieval failed: {e}")
            
            # Combine and return results with scores
            return self._combine_results(semantic_results, bm25_results, k)
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval with scores: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid retriever statistics.
        
        Returns:
            Dictionary containing retriever statistics
        """
        bm25_stats = self.bm25_retriever.get_stats() if self.bm25_retriever else {}
        
        return {
            'is_built': self.is_built,
            'document_count': len(self.documents),
            'semantic_weight': self.semantic_weight,
            'bm25_weight': self.bm25_weight,
            'normalize_scores': self.normalize_scores,
            'min_score_threshold': self.min_score_threshold,
            'bm25_stats': bm25_stats,
            'has_vector_store': self.vector_store is not None
        }