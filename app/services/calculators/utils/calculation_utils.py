"""Utility functions for actuarial calculations."""

import logging
import re
from typing import Dict, Any, List, Tuple
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class CalculationUtils:
    """Utility class for calculation-related helper functions."""
    
    @staticmethod
    def extract_keywords(question: str) -> List[str]:
        """Extract keywords from question for document retrieval.
        
        Args:
            question: User's calculation question
            
        Returns:
            List of extracted keywords
        """
        try:
            # Simple keyword extraction
            words = re.findall(r'\b\w+\b', question.lower())
            # Filter out common words
            stop_words = {'dan', 'atau', 'yang', 'adalah', 'untuk', 'dengan', 'pada', 'di', 'ke', 'dari'}
            keywords = [word for word in words if len(word) > 2 and word not in stop_words]
            return keywords[:10]  # Limit to 10 keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    @staticmethod
    def identify_calculation_type(question: str) -> str:
        """Identify the type of calculation from the question.
        
        Args:
            question: User's calculation question
            
        Returns:
            Type of calculation identified
        """
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['pv', 'present value', 'nilai sekarang']):
            return 'present_value'
        elif any(word in question_lower for word in ['psak', '219', 'imbalan']):
            return 'psak219'
        elif any(word in question_lower for word in ['asumsi', 'assumption', 'rate']):
            return 'assumptions'
        else:
            return 'general'
    
    @staticmethod
    def extract_source_info(source_documents: List[Document], session_id: str) -> List[Dict[str, Any]]:
        """Extract source information from documents.
        
        Args:
            source_documents: List of source documents
            session_id: Current session ID
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        seen_sources = set()
        
        for doc in source_documents:
            metadata = doc.metadata
            source_key = f"{metadata.get('filename', 'unknown')}_{metadata.get('chunk_id', 0)}"
            
            if source_key not in seen_sources:
                sources.append({
                    'filename': metadata.get('filename', 'Unknown'),
                    'doc_type': metadata.get('doc_type', 'general'),
                    'chunk_id': metadata.get('chunk_id', 0),
                    'headers': {k: v for k, v in metadata.items() if k.startswith('Header')},
                    'preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'session_id': metadata.get('session_id')
                })
                seen_sources.add(source_key)
        
        return sources
    
    @staticmethod
    def calculate_confidence(relevant_docs_with_scores: List[tuple]) -> float:
        """Calculate confidence score based on similarity scores.
        
        Args:
            relevant_docs_with_scores: List of (document, score) tuples
             
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not relevant_docs_with_scores:
            return 0.0
        
        scores = [score for _, score in relevant_docs_with_scores]
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-1 range
        confidence = min(avg_score, 1.0)
        return round(confidence, 3)
        try:
            question_lower = question.lower()
            
            if any(term in question_lower for term in ['present value', 'pv', 'discounted']):
                return 'present_value'
            elif any(term in question_lower for term in ['annuity', 'periodic payment']):
                return 'annuity'
            elif any(term in question_lower for term in ['mortality', 'death', 'survival']):
                return 'mortality'
            elif any(term in question_lower for term in ['benefit', 'pension', 'retirement']):
                return 'benefit'
            else:
                return 'general'
                
        except Exception as e:
            logger.error(f"Error identifying calculation type: {e}")
            return 'general'
    
    @staticmethod
    def extract_source_info(source_documents: List[Document], session_id: str) -> List[Dict[str, Any]]:
        """Extract source information from documents.
        
        Args:
            source_documents: List of source documents
            session_id: Session identifier
            
        Returns:
            List of source information dictionaries
        """
        try:
            sources = []
            for doc in source_documents:
                source_info = {
                    'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata,
                    'session_id': session_id
                }
                sources.append(source_info)
            return sources
            
        except Exception as e:
            logger.error(f"Error extracting source info: {e}")
            return []
    
    @staticmethod
    def calculate_confidence(relevant_docs_with_scores: List[tuple]) -> float:
        """Calculate confidence score based on document relevance.
        
        Args:
            relevant_docs_with_scores: List of (document, score) tuples
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if not relevant_docs_with_scores:
                return 0.0
                
            # Calculate average score
            total_score = sum(score for _, score in relevant_docs_with_scores)
            avg_score = total_score / len(relevant_docs_with_scores)
            
            # Normalize to 0-1 range
            confidence = min(max(avg_score, 0.0), 1.0)
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    @staticmethod
    def generate_calculation_summary(exec_results: Dict[str, Any], question: str, extracted: Dict[str, Any]) -> str:
        """Generate summary of calculation results.
        
        Args:
            exec_results: Execution results dictionary
            question: Original question
            extracted: Extracted data
            
        Returns:
            Summary string
        """
        try:
            if not exec_results:
                return "Tidak ada hasil perhitungan yang tersedia."
                
            # Count successful steps
            successful_steps = sum(1 for result in exec_results.values() 
                                 if isinstance(result, dict) and result.get('success', False))
            
            if successful_steps > 0:
                return f"Perhitungan berhasil diselesaikan dengan {successful_steps} langkah. " \
                       f"Hasil detail tersedia pada calculation_steps."
            else:
                return f"Perhitungan selesai dengan {len(exec_results)} langkah. Detail hasil tersedia pada calculation_steps."
                
        except Exception as e:
            logger.error(f"Error generating calculation summary: {e}")
            return "Error dalam menghasilkan ringkasan perhitungan."