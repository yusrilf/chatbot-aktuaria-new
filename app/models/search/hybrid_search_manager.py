"""Hybrid search manager combining semantic and BM25 search with adaptive weighting.

This module provides the main HybridSearchManager class that combines
semantic search results with BM25 keyword search using adaptive weighting
based on query classification.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

from .search_result import SearchResult
from .bm25_index import BM25Index
from .result_combiner import ResultCombiner, CombinedResult
from ...models.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)


class HybridSearchManager:
    """Manager untuk hybrid search yang menggabungkan semantic dan BM25 dengan adaptive weighting.
    
    This class manages the combination of semantic search and BM25 keyword search,
    with adaptive weighting based on query type classification.
    """
    
    def __init__(self, 
                 semantic_weight: float = 0.85,
                 bm25_weight: float = 0.15,
                 bm25_k1: float = 1.2,
                 bm25_b: float = 0.6,
                 enable_adaptive_weighting: bool = True) -> None:
        """Initialize hybrid search manager.
        
        Args:
            semantic_weight: Bobot default untuk skor semantik (0-1)
            bm25_weight: Bobot default untuk skor BM25 (0-1)
            bm25_k1: Parameter BM25 k1
            bm25_b: Parameter BM25 b
            enable_adaptive_weighting: Enable adaptive weighting berdasarkan query type
            
        Raises:
            ValueError: If weights don't sum to approximately 1.0
        """
        self.default_semantic_weight = semantic_weight
        self.default_bm25_weight = bm25_weight
        self.bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)
        self.is_indexed = False
        self.enable_adaptive_weighting = enable_adaptive_weighting
        
        # Initialize query classifier for adaptive weighting
        if self.enable_adaptive_weighting:
            try:
                self.query_classifier = QueryClassifier()
                logger.info("Adaptive weighting enabled with query classifier")
            except Exception as e:
                logger.warning(f"Failed to initialize query classifier: {e}. Falling back to fixed weights.")
                self.query_classifier = None
                self.enable_adaptive_weighting = False
        else:
            self.query_classifier = None
            logger.info("Using fixed weighting scheme")
        
        # Validasi bobot default
        if abs(semantic_weight + bm25_weight - 1.0) > 0.001:
            logger.warning(f"Default weights don't sum to 1.0: semantic={semantic_weight}, bm25={bm25_weight}")
    
    def build_index(self, documents: List[Document]) -> None:
        """Membangun index untuk BM25.
        
        Args:
            documents: List dokumen untuk diindeks
            
        Raises:
            Exception: If indexing fails
        """
        try:
            logger.info(f"Building hybrid search index for {len(documents)} documents")
            self.bm25_index.build_index(documents)
            self.is_indexed = True
            logger.info("Hybrid search index built successfully")
            
        except Exception as e:
            logger.error(f"Error building hybrid search index: {str(e)}")
            self.is_indexed = False
            raise
    
    def hybrid_search(self, 
                     query: str,
                     semantic_results: List[Tuple[Document, float]],
                     top_k: int = 10,
                     apply_score_threshold: bool = True) -> List[SearchResult]:
        """Melakukan hybrid search dengan menggabungkan semantic dan BM25 menggunakan adaptive weighting.
        
        Args:
            query: Query pencarian
            semantic_results: Hasil semantic search dengan skor
            top_k: Jumlah hasil teratas
            
        Returns:
            List SearchResult yang sudah diurutkan berdasarkan combined score
        """
        try:
            if not self.is_indexed:
                logger.warning("BM25 index not available, falling back to semantic only")
                return self._create_semantic_only_results(query, semantic_results)
            
            # Determine adaptive weights based on query type
            semantic_weight, bm25_weight = self._get_adaptive_weights(query)
            
            # Get BM25 results with higher initial count for better filtering
            from ...config import config
            initial_k = max(top_k * config.RETRIEVAL_MULTIPLIER, config.INITIAL_RETRIEVAL_K)
            bm25_results = self.bm25_index.search(query, top_k=initial_k)
            
            if not bm25_results:
                logger.warning("No BM25 results, falling back to semantic only")
                return self._create_semantic_only_results(query, semantic_results)
            
            # Combine results with adaptive weights
            combined_results = ResultCombiner.combine_results(semantic_results, bm25_results, 
                                                             semantic_weight, bm25_weight)
            
            # Apply score threshold filtering if enabled
            if apply_score_threshold:
                filtered_results = []
                for result in combined_results:
                    if result.combined_score >= config.MIN_RELEVANCE_SCORE:
                        filtered_results.append(result)
                    else:
                        logger.debug(f"Filtered out result with low score: {result.combined_score:.3f} < {config.MIN_RELEVANCE_SCORE}")
                
                logger.info(f"After score filtering: {len(filtered_results)}/{len(combined_results)} results remain")
                combined_results = filtered_results
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Convert to SearchResult format and add rank positions
            search_results = []
            for i, result in enumerate(combined_results[:top_k]):
                search_result = SearchResult(
                    content=result.document.page_content,
                    score=result.combined_score,
                    source=result.document.metadata.get('filename', 'unknown'),
                    metadata={
                        'semantic_score': result.semantic_score,
                        'bm25_score': result.bm25_score,
                        'combined_score': result.combined_score,
                        'semantic_weight': semantic_weight,
                        'bm25_weight': bm25_weight,
                        **result.document.metadata
                    },
                    rank=i + 1
                )
                search_results.append(search_result)
            
            logger.info(f"Hybrid search returned {len(search_results)} results with weights: semantic={semantic_weight:.3f}, bm25={bm25_weight:.3f}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return self._create_semantic_only_results(query, semantic_results)
    
    def _get_adaptive_weights(self, query: str) -> Tuple[float, float]:
        """Mendapatkan adaptive weights berdasarkan query type.
        
        Args:
            query: Query pencarian
            
        Returns:
            Tuple[float, float]: (semantic_weight, bm25_weight)
        """
        if not self.enable_adaptive_weighting or not self.query_classifier:
            return self.default_semantic_weight, self.default_bm25_weight
        
        try:
            # Classify query
            query_type, confidence = self.query_classifier.classify_query(query)
            
            # Get adaptive weights
            weights = self.query_classifier.get_adaptive_weights(query_type, confidence)
            
            return weights['semantic'], weights['bm25']
            
        except Exception as e:
            logger.error(f"Error getting adaptive weights: {e}")
            return self.default_semantic_weight, self.default_bm25_weight
    
    def _create_semantic_only_results(self, query: str, semantic_results: List[Tuple[Document, float]]) -> List[SearchResult]:
        """Fallback ke semantic search saja.
        
        Args:
            query: Original query
            semantic_results: Semantic search results
            
        Returns:
            List of SearchResult objects
        """
        results = []
        for i, (doc, score) in enumerate(semantic_results):
            search_result = SearchResult(
                content=doc.page_content,
                score=score,
                source=doc.metadata.get('filename', 'unknown'),
                metadata={
                    'semantic_score': score,
                    'bm25_score': 0.0,
                    'combined_score': score,
                    'fallback_mode': 'semantic_only',
                    **doc.metadata
                },
                rank=i + 1
            )
            results.append(search_result)
        return results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Mendapatkan statistik search index.
        
        Returns:
            Dictionary containing search statistics
        """
        stats = {
            'is_indexed': self.is_indexed,
            'default_semantic_weight': self.default_semantic_weight,
            'default_bm25_weight': self.default_bm25_weight,
            'adaptive_weighting_enabled': self.enable_adaptive_weighting,
            'query_classifier_available': self.query_classifier is not None
        }
        
        if self.is_indexed:
            stats.update(self.bm25_index.get_index_stats())
        
        return stats
    
    def clear_index(self) -> None:
        """Clear all indexes and free memory."""
        self.bm25_index.clear_index()
        self.is_indexed = False
        logger.info("Hybrid search index cleared")