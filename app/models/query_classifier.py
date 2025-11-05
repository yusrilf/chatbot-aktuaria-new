"""Query Classifier for Actuarial Chatbot

This module provides classification of user queries into different categories
to enable adaptive weighting in hybrid search.
"""

import re
from typing import Dict, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enum for different types of actuarial queries"""
    CALCULATION = "calculation"  # Mathematical calculations, formulas
    CONCEPT = "concept"          # Theoretical concepts, definitions
    REGULATION = "regulation"    # PSAK, regulations, compliance
    GENERAL = "general"          # General questions, mixed content

class QueryClassifier:
    """Classifier to determine the type of actuarial query"""
    
    def __init__(self):
        """Initialize the query classifier with keyword patterns"""
        self.calculation_keywords = {
            'indonesian': [
                'hitung', 'kalkulasi', 'rumus', 'formula', 'perhitungan',
                'pvfb', 'pvdbo', 'present value', 'future benefit',
                'discount rate', 'tingkat diskonto', 'mortality rate',
                'tingkat kematian', 'withdrawal rate', 'tingkat keluar',
                'salary increase', 'kenaikan gaji', 'actuarial gain',
                'keuntungan aktuaria', 'actuarial loss', 'kerugian aktuaria',
                'service cost', 'biaya jasa', 'interest cost', 'biaya bunga',
                'sensitivity analysis', 'analisis sensitivitas',
                'multiple decrement', 'decremen ganda', 'tabel mortalitas',
                'mortality table', 'yield curve', 'kurva hasil'
            ],
            'english': [
                'calculate', 'computation', 'formula', 'equation',
                'pvfb', 'pvdbo', 'present value', 'future benefit',
                'discount', 'mortality', 'withdrawal', 'turnover',
                'salary growth', 'actuarial gain', 'actuarial loss',
                'service cost', 'interest cost', 'sensitivity',
                'multiple decrement', 'mortality table', 'yield curve'
            ]
        }
        
        self.concept_keywords = {
            'indonesian': [
                'definisi', 'pengertian', 'konsep', 'teori', 'prinsip',
                'apa itu', 'bagaimana', 'mengapa', 'jelaskan',
                'imbalan kerja', 'employee benefit', 'pensiun', 'pension',
                'aktuaria', 'actuarial', 'valuasi', 'valuation',
                'asumsi', 'assumption', 'metode', 'method',
                'projected unit credit', 'puc', 'entry age normal',
                'defined benefit', 'defined contribution',
                'vested benefit', 'unvested benefit'
            ],
            'english': [
                'definition', 'concept', 'theory', 'principle',
                'what is', 'how', 'why', 'explain',
                'employee benefit', 'pension', 'retirement',
                'actuarial', 'valuation', 'assumption', 'method',
                'projected unit credit', 'entry age normal',
                'defined benefit', 'defined contribution',
                'vested', 'unvested'
            ]
        }
        
        self.regulation_keywords = {
            'indonesian': [
                'psak', 'psak 24', 'psak 219', 'standar akuntansi',
                'peraturan', 'regulasi', 'compliance', 'kepatuhan',
                'disclosure', 'pengungkapan', 'laporan keuangan',
                'financial statement', 'audit', 'auditor',
                'ojk', 'otoritas jasa keuangan', 'bank indonesia',
                'undang-undang', 'peraturan pemerintah', 'pp',
                'peraturan menteri', 'permen', 'surat edaran',
                'ketentuan', 'persyaratan', 'standar'
            ],
            'english': [
                'psak', 'accounting standard', 'regulation',
                'compliance', 'disclosure', 'financial statement',
                'audit', 'auditor', 'ojk', 'bank indonesia',
                'law', 'government regulation', 'ministerial regulation',
                'circular letter', 'requirement', 'standard'
            ]
        }
    
    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify a query into one of the predefined types
        
        Args:
            query (str): The user query to classify
            
        Returns:
            Tuple[QueryType, float]: The predicted query type and confidence score
        """
        try:
            query_lower = query.lower().strip()
            
            # Calculate scores for each category
            calculation_score = self._calculate_keyword_score(
                query_lower, self.calculation_keywords
            )
            concept_score = self._calculate_keyword_score(
                query_lower, self.concept_keywords
            )
            regulation_score = self._calculate_keyword_score(
                query_lower, self.regulation_keywords
            )
            
            # Determine the category with highest score
            scores = {
                QueryType.CALCULATION: calculation_score,
                QueryType.CONCEPT: concept_score,
                QueryType.REGULATION: regulation_score
            }
            
            max_score = max(scores.values())
            
            # If no clear category, classify as general
            if max_score < 0.1:
                return QueryType.GENERAL, 0.5
            
            # Find the category with maximum score
            predicted_type = max(scores, key=scores.get)
            confidence = max_score
            
            logger.info(f"Query classified as {predicted_type.value} with confidence {confidence:.3f}")
            
            return predicted_type, confidence
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return QueryType.GENERAL, 0.5
    
    def _calculate_keyword_score(self, query: str, keyword_dict: Dict[str, List[str]]) -> float:
        """
        Calculate keyword matching score for a category
        
        Args:
            query (str): The query text
            keyword_dict (Dict[str, List[str]]): Keywords for the category
            
        Returns:
            float: Normalized score between 0 and 1
        """
        total_matches = 0
        total_keywords = 0
        
        for lang, keywords in keyword_dict.items():
            for keyword in keywords:
                total_keywords += 1
                # Use word boundaries for exact matching
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, query):
                    total_matches += 1
        
        # Normalize score
        if total_keywords == 0:
            return 0.0
        
        return total_matches / total_keywords
    
    def get_adaptive_weights(self, query_type: QueryType, confidence: float) -> Dict[str, float]:
        """
        Get adaptive weights based on query type and confidence
        
        Args:
            query_type (QueryType): The classified query type
            confidence (float): Classification confidence
            
        Returns:
            Dict[str, float]: Semantic and BM25 weights
        """
        # Base weights (85% semantic, 15% BM25)
        base_semantic = 0.85
        base_bm25 = 0.15
        
        # Adjust weights based on query type
        if query_type == QueryType.CALCULATION:
            # Calculations benefit more from semantic understanding
            semantic_weight = min(0.90, base_semantic + 0.05 * confidence)
            bm25_weight = 1.0 - semantic_weight
            
        elif query_type == QueryType.REGULATION:
            # Regulations benefit from keyword matching (PSAK numbers, etc.)
            bm25_weight = min(0.25, base_bm25 + 0.10 * confidence)
            semantic_weight = 1.0 - bm25_weight
            
        elif query_type == QueryType.CONCEPT:
            # Concepts benefit from semantic understanding
            semantic_weight = min(0.88, base_semantic + 0.03 * confidence)
            bm25_weight = 1.0 - semantic_weight
            
        else:  # GENERAL
            # Use base weights for general queries
            semantic_weight = base_semantic
            bm25_weight = base_bm25
        
        logger.info(f"Adaptive weights for {query_type.value}: semantic={semantic_weight:.3f}, bm25={bm25_weight:.3f}")
        
        return {
            'semantic': semantic_weight,
            'bm25': bm25_weight
        }