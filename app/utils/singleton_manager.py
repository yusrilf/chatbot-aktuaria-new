#!/usr/bin/env python3
"""Singleton Manager for Actuarial Chatbot Services.

This module provides backward compatibility imports for singleton functionality.
The actual implementation has been moved to the singleton package.

Author: AI Assistant
Date: 2025-01-05
Version: 2.0.0
"""

# Import all functionality from the new singleton package
from .singleton import (
    SingletonMeta,
    ServiceManager,
    get_service_manager,
    singleton_service,
    lazy_initialization,
    get_or_create_service
)

# Maintain backward compatibility
__all__ = [
    'SingletonMeta',
    'ServiceManager',
    'get_service_manager',
    'singleton_service',
    'lazy_initialization',
    'get_or_create_service'
]