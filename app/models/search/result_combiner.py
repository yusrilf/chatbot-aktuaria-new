"""Result combination utilities for hybrid search.

This module provides utilities for combining and normalizing
search results from different search methods.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class CombinedResult:
    """Represents a combined search result with multiple scores.
    
    Attributes:
        document: The source document
        semantic_score: Normalized semantic similarity score
        bm25_score: Normalized BM25 relevance score
        combined_score: Weighted combination of both scores
    """
    
    document: Document
    semantic_score: float
    bm25_score: float
    combined_score: float
    
    @property
    def score(self) -> float:
        """Alias for combined_score for compatibility."""
        return self.combined_score


class ResultCombiner:
    """Utility class for combining and normalizing search results.
    
    This class provides methods to combine semantic and BM25 search results
    with proper score normalization and weighting.
    """
    
    @staticmethod
    def combine_results(semantic_results: List[Tuple[Document, float]],
                       bm25_results: List[Tuple[Document, float]],
                       semantic_weight: float,
                       bm25_weight: float) -> List[CombinedResult]:
        """Menggabungkan hasil semantic dan BM25 dengan normalisasi skor dan adaptive weights.
        
        Args:
            semantic_results: List of (Document, semantic_score) tuples
            bm25_results: List of (Document, bm25_score) tuples
            semantic_weight: Weight for semantic scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            
        Returns:
            List of CombinedResult objects with normalized and weighted scores
        """
        try:
            # Normalize scores
            semantic_scores = ResultCombiner._normalize_scores([score for _, score in semantic_results])
            bm25_scores = ResultCombiner._normalize_scores([score for _, score in bm25_results])
            
            # Create document to score mappings
            semantic_map = {}
            for i, (doc, _) in enumerate(semantic_results):
                doc_id = ResultCombiner._get_document_id(doc)
                semantic_map[doc_id] = (doc, semantic_scores[i])
            
            bm25_map = {}
            for i, (doc, _) in enumerate(bm25_results):
                doc_id = ResultCombiner._get_document_id(doc)
                bm25_map[doc_id] = (doc, bm25_scores[i])
            
            # Combine results
            combined_results = []
            all_doc_ids = set(semantic_map.keys()) | set(bm25_map.keys())
            
            for doc_id in all_doc_ids:
                semantic_score = semantic_map.get(doc_id, (None, 0.0))[1]
                bm25_score = bm25_map.get(doc_id, (None, 0.0))[1]
                
                # Get document (prefer semantic result)
                doc = semantic_map.get(doc_id, bm25_map.get(doc_id))[0]
                
                # Calculate combined score with adaptive weights
                combined_score = (
                    semantic_weight * semantic_score + 
                    bm25_weight * bm25_score
                )
                
                result = CombinedResult(
                    document=doc,
                    semantic_score=semantic_score,
                    bm25_score=bm25_score,
                    combined_score=combined_score
                )
                combined_results.append(result)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            # Fallback to semantic results only
            return ResultCombiner._create_semantic_fallback(semantic_results)
    
    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """Normalisasi skor menggunakan min-max normalization.
        
        Semakin tinggi score, semakin relevan (similarity/kedekatan tinggi).
        
        Args:
            scores: List of raw scores to normalize
            
        Returns:
            List of normalized scores in range [0.1, 1.0]
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        
        # Handle case where all scores are the same
        if len(set(scores)) == 1:
            return [0.8] * len(scores)  # Default good score
        
        # Use min-max normalization to preserve relative order
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        # Avoid division by zero
        if max_score == min_score:
            return [0.8] * len(scores)
        
        # Min-max normalization: (x - min) / (max - min)
        normalized = (scores_array - min_score) / (max_score - min_score)
        
        # Scale to range [0.1, 1.0] to maintain meaningful differences
        # Higher original scores get higher normalized scores
        normalized = 0.1 + (normalized * 0.9)
        
        return normalized.tolist()
    
    @staticmethod
    def _get_document_id(doc: Document) -> str:
        """Generate unique ID untuk dokumen.
        
        Args:
            doc: Document to generate ID for
            
        Returns:
            Unique document identifier string
        """
        # Gunakan kombinasi filename dan hash dari content
        filename = doc.metadata.get('filename', 'unknown')
        content_hash = hash(doc.page_content[:100])  # Hash dari 100 karakter pertama
        return f"{filename}_{content_hash}"
    
    @staticmethod
    def _create_semantic_fallback(semantic_results: List[Tuple[Document, float]]) -> List[CombinedResult]:
        """Create fallback results using only semantic scores.
        
        Args:
            semantic_results: List of semantic search results
            
        Returns:
            List of CombinedResult objects with only semantic scores
        """
        results = []
        for doc, score in semantic_results:
            result = CombinedResult(
                document=doc,
                semantic_score=score,
                bm25_score=0.0,
                combined_score=score
            )
            results.append(result)
        return results
    
    @staticmethod
    def calculate_score_statistics(results: List[CombinedResult]) -> Dict[str, float]:
        """Calculate statistics for combined results.
        
        Args:
            results: List of combined results
            
        Returns:
            Dictionary with score statistics
        """
        if not results:
            return {}
        
        semantic_scores = [r.semantic_score for r in results]
        bm25_scores = [r.bm25_score for r in results]
        combined_scores = [r.combined_score for r in results]
        
        return {
            'semantic_mean': np.mean(semantic_scores),
            'semantic_std': np.std(semantic_scores),
            'bm25_mean': np.mean(bm25_scores),
            'bm25_std': np.std(bm25_scores),
            'combined_mean': np.mean(combined_scores),
            'combined_std': np.std(combined_scores),
            'total_results': len(results)
        }