#!/usr/bin/env python3
"""Helper functions for chat_service_original.py to support enhanced context information.

This module provides utility functions for confidence level assessment,
response quality evaluation, and error classification.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_confidence_level(confidence: float) -> str:
    """Get human-readable confidence level.
    
    Args:
        confidence: Confidence score between 0 and 1
        
    Returns:
        Human-readable confidence level string
        
    Example:
        >>> get_confidence_level(0.85)
        'high'
        >>> get_confidence_level(0.45)
        'low'
    """
    try:
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid confidence value: {confidence}, error: {e}")
        return "unknown"


def assess_tot_response_quality(confidence: float, reasoning_steps: List[Any]) -> str:
    """Assess Tree of Thought response quality.
    
    Args:
        confidence: Confidence score between 0 and 1
        reasoning_steps: List of reasoning steps from ToT processing
        
    Returns:
        Quality assessment string
        
    Example:
        >>> assess_tot_response_quality(0.8, ["step1", "step2", "step3"])
        'excellent'
        >>> assess_tot_response_quality(0.3, ["step1"])
        'poor'
    """
    try:
        steps_count = len(reasoning_steps) if reasoning_steps else 0
        
        if confidence >= 0.7 and steps_count >= 3:
            return "excellent"
        elif confidence >= 0.5 and steps_count >= 2:
            return "good"
        elif confidence >= 0.3 and steps_count >= 1:
            return "fair"
        else:
            return "poor"
    except (TypeError, ValueError) as e:
        logger.warning(f"Error assessing response quality: {e}")
        return "unknown"


def classify_error_type(error_msg: str) -> str:
    """Classify error type based on error message.
    
    Args:
        error_msg: Error message string
        
    Returns:
        Classified error type string
        
    Example:
        >>> classify_error_type("Connection timeout occurred")
        'timeout'
        >>> classify_error_type("OpenAI API error")
        'api_error'
    """
    try:
        if not error_msg or not isinstance(error_msg, str):
            return "unknown"
        
        error_lower = error_msg.lower()
        
        if "timeout" in error_lower or "time" in error_lower:
            return "timeout"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        elif "api" in error_lower or "openai" in error_lower:
            return "api_error"
        elif "memory" in error_lower:
            return "memory_error"
        elif "processing" in error_lower or "tot" in error_lower:
            return "processing_error"
        elif "retrieval" in error_lower or "document" in error_lower:
            return "retrieval_error"
        else:
            return "general_error"
    except Exception as e:
        logger.error(f"Error classifying error type: {e}")
        return "unknown"


def count_sources_by_type(sources: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count sources by type.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Dictionary with source type counts
        
    Example:
        >>> sources = [{"type": "document"}, {"type": "document"}, {"type": "web"}]
        >>> count_sources_by_type(sources)
        {'document': 2, 'web': 1}
    """
    try:
        type_counts = {}
        
        if not sources or not isinstance(sources, list):
            return type_counts
        
        for source in sources:
            if isinstance(source, dict):
                source_type = source.get('type', 'unknown')
                type_counts[source_type] = type_counts.get(source_type, 0) + 1
            else:
                logger.warning(f"Invalid source format: {source}")
                type_counts['invalid'] = type_counts.get('invalid', 0) + 1
        
        return type_counts
    except Exception as e:
        logger.error(f"Error counting sources by type: {e}")
        return {}


def extract_relevance_scores(sources: List[Dict[str, Any]]) -> List[float]:
    """Extract relevance scores from sources.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        List of relevance scores
        
    Example:
        >>> sources = [{"score": 0.8}, {"score": 0.6}, {"relevance": 0.9}]
        >>> extract_relevance_scores(sources)
        [0.8, 0.6, 0.9]
    """
    try:
        scores = []
        
        if not sources or not isinstance(sources, list):
            return scores
        
        for source in sources:
            if isinstance(source, dict):
                # Try different score field names
                score = source.get('score') or source.get('relevance') or source.get('similarity')
                if score is not None and isinstance(score, (int, float)):
                    scores.append(float(score))
        
        return scores
    except Exception as e:
        logger.error(f"Error extracting relevance scores: {e}")
        return []


def create_enhanced_context_info(
    confidence: float,
    reasoning_steps: List[Any],
    doc_retrieval_info: Dict[str, Any],
    sources: List[Dict[str, Any]] = None,
    mode: str = "default",
    has_error: bool = False,
    error_msg: str = None
) -> Dict[str, Any]:
    """Create enhanced context information for API responses.
    
    Args:
        confidence: Confidence score
        reasoning_steps: List of reasoning steps
        doc_retrieval_info: Document retrieval metadata
        sources: List of source documents (optional)
        mode: Processing mode
        has_error: Whether an error occurred
        error_msg: Error message if any
        
    Returns:
        Enhanced context information dictionary
    """
    try:
        sources = sources or []
        
        context_info = {
            'retrieval_metadata': {
                'documents_retrieved': len(doc_retrieval_info.get('relevant_sections', [])),
                'confidence_level': get_confidence_level(confidence),
                'processing_mode': mode,
                'response_quality': assess_tot_response_quality(confidence, reasoning_steps)
            },
            'reasoning_context': {
                'reasoning_steps_count': len(reasoning_steps) if reasoning_steps else 0,
                'paths_evaluated': doc_retrieval_info.get('total_paths_generated', 0),
                'verification_used': False,  # Not implemented in original service
                'query_expansion_used': doc_retrieval_info.get('query_expansion_enabled', False)
            },
            'document_context': {
                'strategies_used': doc_retrieval_info.get('strategies_used', 0),
                'context_enhancement_ratio': doc_retrieval_info.get('context_enhancement_ratio', 1.0),
                'relevant_sections_found': len(doc_retrieval_info.get('relevant_sections', [])),
                'question_analysis': doc_retrieval_info.get('question_type', 'unknown')
            },
            'error_context': {
                'has_error': has_error,
                'fallback_used': 'fallback' in mode.lower() if mode else False,
                'error_type': classify_error_type(error_msg) if error_msg else None
            }
        }
        
        # Add source-specific context if sources available
        if sources:
            context_info['source_context'] = {
                'source_types': list(set([src.get('type', 'unknown') for src in sources])),
                'source_count_by_type': count_sources_by_type(sources),
                'relevance_scores': extract_relevance_scores(sources)
            }
        
        return context_info
    except Exception as e:
        logger.error(f"Error creating enhanced context info: {e}")
        return {
            'retrieval_metadata': {'error': 'Failed to create context info'},
            'error_context': {'has_error': True, 'error_type': 'context_creation_error'}
        }