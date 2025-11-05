#!/usr/bin/env python3
"""Singleton and Service Decorators.

This module provides decorators for singleton services and lazy initialization.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

import logging
import threading
from functools import wraps
from typing import Any, Callable, TypeVar, Type

from .service_manager import get_service_manager

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Thread lock for preventing race conditions
_service_creation_lock = threading.Lock()

def singleton_service(service_name: str):
    """
    Decorator to register a class as a singleton service.
    
    This decorator automatically registers the decorated class
    as a singleton service in the global service manager.
    
    Args:
        service_name: Name to register the service under
        
    Returns:
        Decorated class with singleton behavior
        
    Example:
        @singleton_service("vector_store")
        class VectorStoreManager:
            pass
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Call original constructor
            original_init(self, *args, **kwargs)
            
            # Register in service manager with thread safety
            with _service_creation_lock:
                service_manager = get_service_manager()
                if not service_manager.is_service_initialized(service_name):
                    service_manager.register_service(service_name, self)
                    logger.info(f"Registered singleton service: {service_name}")
        
        cls.__init__ = new_init
        return cls
    
    return decorator

def lazy_initialization(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for lazy initialization of expensive operations.
    
    This decorator ensures that expensive initialization is only
    performed when actually needed, improving startup performance.
    
    Args:
        func: Function to decorate with lazy initialization
        
    Returns:
        Decorated function with lazy initialization
        
    Example:
        @lazy_initialization
        def initialize_vector_store(self):
            # Expensive initialization code
            pass
    """
    _initialized = False
    _result = None
    _init_lock = threading.Lock()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal _initialized, _result
        
        if not _initialized:
            with _init_lock:
                # Double-check locking pattern
                if not _initialized:
                    logger.info(f"Lazy initializing: {func.__name__}")
                    _result = func(*args, **kwargs)
                    _initialized = True
                    logger.debug(f"Lazy initialization completed: {func.__name__}")
        else:
            logger.debug(f"Using cached result: {func.__name__}")
            
        return _result
    
    return wrapper

def get_or_create_service(service_class: Type[T], service_name: str, *args, **kwargs) -> T:
    """
    Get existing service or create new singleton instance.
    
    This function checks if a service already exists in the service manager.
    If it exists, returns the existing instance. Otherwise, creates a new
    singleton instance and registers it with thread safety.
    
    Args:
        service_class: Class to instantiate
        service_name: Name to register service under
        *args: Arguments for class constructor
        **kwargs: Keyword arguments for class constructor
        
    Returns:
        Service instance (existing or newly created)
        
    Example:
        vector_store = get_or_create_service(
            VectorStoreManager, 
            "vector_store",
            config=config
        )
    """
    service_manager = get_service_manager()
    
    # Check if service already exists
    existing_service = service_manager.get_service(service_name)
    if existing_service is not None:
        logger.debug(f"Using existing service: {service_name}")
        return existing_service
    
    # Create new service instance with thread safety
    with _service_creation_lock:
        # Double-check locking pattern
        existing_service = service_manager.get_service(service_name)
        if existing_service is not None:
            logger.debug(f"Using existing service (double-check): {service_name}")
            return existing_service
            
        logger.info(f"Creating new service: {service_name}")
        try:
            instance = service_class(*args, **kwargs)
            service_manager.register_service(service_name, instance)
            logger.info(f"Successfully created and registered service: {service_name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create service {service_name}: {e}")
            raise