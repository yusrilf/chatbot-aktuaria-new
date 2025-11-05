#!/usr/bin/env python3
"""Helper utilities for the actuarial chatbot application.

This module provides common utility functions for logging, file operations,
API responses, and data validation.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

import os
import logging
import json
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Configure module logger
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Raises:
        AttributeError: If log_level is not a valid logging level
    """
    try:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('app.log'),
                logging.StreamHandler()
            ]
        )
        logger.info(f"Logging configured with level: {log_level.upper()}")
    except AttributeError as e:
        logger.error(f"Invalid log level: {log_level}")
        raise


def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key format.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        True if the API key format is valid, False otherwise
        
    Note:
        This only validates the format, not the actual validity of the key
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # OpenAI API keys start with 'sk-' and are typically longer than 20 characters
    return api_key.startswith('sk-') and len(api_key) > 20


def get_file_size(file_path: str) -> str:
    """Get human readable file size.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Human readable file size string (e.g., "1.5 MB")
        
    Note:
        Returns "Unknown" if file doesn't exist or can't be accessed
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return "File not found"
            
        size = os.path.getsize(file_path)
        
        # Convert bytes to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
        
    except (OSError, IOError) as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return "Unknown"


def create_response(success: bool, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
    """Create standardized API response.
    
    Args:
        success: Whether the operation was successful
        message: Response message
        data: Optional data to include in response
        
    Returns:
        Standardized response dictionary
        
    Example:
        >>> create_response(True, "Operation successful", {"id": 123})
        {
            'success': True,
            'message': 'Operation successful',
            'timestamp': '2025-01-05T10:30:00.123456',
            'data': {'id': 123}
        }
    """
    response = {
        'success': success,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    
    if data is not None:
        response['data'] = data
    
    return response


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem storage
        
    Note:
        Removes or replaces unsafe characters and limits length
    """
    if not filename or not isinstance(filename, str):
        return "unnamed_file"
    
    # Remove or replace unsafe characters with underscores
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple consecutive underscores and trim
    filename = re.sub(r'_+', '_', filename).strip('_')
    
    # Limit filename length (preserve extension)
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
    
    return f"{name}{ext}" if ext else name


def format_number(number: Union[int, float], decimal_places: int = 2) -> str:
    """Format number with Indonesian locale formatting.
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places to show
        
    Returns:
        Formatted number string using Indonesian locale (comma as decimal separator)
        
    Example:
        >>> format_number(1234.56, 2)
        '1.234,56'
        
    Note:
        Uses Indonesian number formatting where:
        - Thousands separator: . (dot)
        - Decimal separator: , (comma)
    """
    try:
        if not isinstance(number, (int, float)):
            logger.warning(f"Invalid number type: {type(number)}")
            return str(number)
            
        # Format with standard locale first
        formatted = f"{number:,.{decimal_places}f}"
        
        # Convert to Indonesian format
        # Replace comma (thousands) with X temporarily
        # Replace dot (decimal) with comma
        # Replace X with dot
        return formatted.replace(',', 'X').replace('.', ',').replace('X', '.')
        
    except (ValueError, TypeError) as e:
        logger.error(f"Error formatting number {number}: {e}")
        return str(number)


def validate_files(files: Optional[List[Any]]) -> bool:
    """Validate uploaded files list.
    
    Args:
        files: List of uploaded file objects
        
    Returns:
        True if files list is valid and contains at least one file with a name
        
    Note:
        Checks that files exist and have non-empty filenames
    """
    if not files:
        return False
        
    try:
        # Check if any file has a non-empty filename
        return any(hasattr(f, 'filename') and f.filename and f.filename.strip() for f in files)
    except (AttributeError, TypeError) as e:
        logger.error(f"Error validating files: {e}")
        return False
