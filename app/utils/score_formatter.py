"""Score formatting utilities for consistent display across the application.

This module provides utilities to format relevance scores in a user-friendly way,
avoiding scientific notation for very small values.
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)


def format_score(score: Union[float, int], precision: int = 4) -> Union[float, str]:
    """Format score untuk readability - hindari scientific notation.
    
    Score menunjukkan similarity/relevance: semakin tinggi semakin relevan.
    Range normal: 0.1 - 1.0
    
    Args:
        score: Raw score value to format
        precision: Number of decimal places for formatting (default: 4)
        
    Returns:
        Union[float, str]: Formatted score. Returns string for very small numbers to avoid scientific notation.
        
    Examples:
        >>> format_score(0.85)
        0.85
        >>> format_score(0.1234)
        0.1234
        >>> format_score(4.947159e-05)  # Legacy small scores
        '0.000049'
    """
    try:
        import math
        
        if not isinstance(score, (int, float)):
            logger.warning(f"Invalid score type: {type(score)}, returning 0.1")
            return 0.1
            
        if math.isnan(score) or math.isinf(score):
            logger.warning(f"Invalid score value: {score}. Using default.")
            return 0.1
            
        # Handle edge cases
        if score == 0:
            return 0.0
        if score >= 1.0:
            return round(score, precision)
            
        # For very small scores (legacy), use string formatting to avoid scientific notation
        if score < 0.001:
            # Format directly as string with fixed decimal places to avoid scientific notation
            formatted_str = f"{score:.6f}"
            return formatted_str
        else:
            # For normal scores (0.001 - 1.0), use standard rounding
            return round(score, precision)
            
    except Exception as e:
        logger.error(f"Error formatting score {score}: {e}")
        return 0.1  # Return minimum meaningful score instead of 0.0


def format_score_with_relevance(score: Union[float, int],
                               precision: int = 4) -> tuple[Union[float, str], str]:
    """Format score and determine relevance category based on normalized scores.
    
    Args:
        score: Raw score value to format
        precision: Number of decimal places for formatting (default: 4)
        
    Returns:
        Tuple of (formatted_score, relevance_category)
        
    Examples:
        >>> format_score_with_relevance(0.85)
        (0.85, 'high')
        >>> format_score_with_relevance(4.947159e-05)
        ('0.000049', 'low')
    """
    try:
        from app.config import HIGH_RELEVANCE_THRESHOLD, MEDIUM_RELEVANCE_THRESHOLD
        
        formatted_score = format_score(score, precision)
        
        # Use original score for relevance determination to avoid string comparison
        # Determine relevance category based on normalized score thresholds
        if score >= HIGH_RELEVANCE_THRESHOLD:  # 0.7 for normalized scores
            relevance = 'high'
        elif score >= MEDIUM_RELEVANCE_THRESHOLD:  # 0.5 for normalized scores
            relevance = 'medium'
        else:
            relevance = 'low'
            
        return formatted_score, relevance
        
    except Exception as e:
        logger.error(f"Error formatting score with relevance: {e}")
        return format_score(score, precision), 'low'


def format_scores_batch(scores: list[Union[float, int]], 
                       precision: int = 4) -> list[float]:
    """Format multiple scores in batch for efficiency.
    
    Args:
        scores: List of raw score values to format
        precision: Number of decimal places for formatting (default: 4)
        
    Returns:
        List of formatted scores
        
    Examples:
        >>> format_scores_batch([0.85, 4.947159e-05, 0.65])
        [0.85, 0.000049, 0.65]
    """
    try:
        return [format_score(score, precision) for score in scores]
    except Exception as e:
        logger.error(f"Error formatting scores batch: {e}")
        return [0.0] * len(scores)


def is_scientific_notation(value: Union[str, float]) -> bool:
    """Check if a value is in scientific notation format.
    
    Args:
        value: Value to check (string or float)
        
    Returns:
        True if value is in scientific notation, False otherwise
        
    Examples:
        >>> is_scientific_notation("4.947159e-05")
        True
        >>> is_scientific_notation(0.85)
        False
        >>> is_scientific_notation(4.947159e-05)
        True
    """
    try:
        if isinstance(value, str):
            # Check if string contains 'e' or 'E' followed by digits
            import re
            pattern = r'[eE][+-]?\d+'
            return bool(re.search(pattern, value))
        elif isinstance(value, (int, float)):
            # Convert to string and check for scientific notation pattern
            str_value = str(value)
            import re
            pattern = r'[eE][+-]?\d+'
            return bool(re.search(pattern, str_value))
        return False
    except Exception:
        return False