#!/usr/bin/env python3
"""Singleton Metaclass Implementation.

This module provides thread-safe singleton metaclass for preventing
duplicate initialization of heavy services.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

import threading
import logging
from typing import Dict, Any, Type, TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar('T')

class SingletonMeta(type):
    """
    Thread-safe singleton metaclass implementation.
    
    This metaclass ensures that only one instance of a class exists
    across the entire application, preventing duplicate initialization
    of heavy services.
    """
    _instances: Dict[Type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """
        Create or return existing singleton instance.
        
        Args:
            *args: Positional arguments for class constructor
            **kwargs: Keyword arguments for class constructor
            
        Returns:
            Singleton instance of the class
        """
        if cls not in cls._instances:
            with cls._lock:
                # Double-check locking pattern
                if cls not in cls._instances:
                    logger.info(f"Creating singleton instance of {cls.__name__}")
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
                else:
                    logger.debug(f"Returning existing singleton instance of {cls.__name__}")
        else:
            logger.debug(f"Returning existing singleton instance of {cls.__name__}")
            
        return cls._instances[cls]
    
    @classmethod
    def clear_instances(cls):
        """
        Clear all singleton instances (useful for testing).
        """
        with cls._lock:
            logger.info("Clearing all singleton instances")
            cls._instances.clear()
    
    @classmethod
    def get_instance(cls, class_type: Type[T]) -> Optional[T]:
        """
        Get existing singleton instance without creating new one.
        
        Args:
            class_type: Class type to get instance for
            
        Returns:
            Existing singleton instance or None
        """
        return cls._instances.get(class_type)