#!/usr/bin/env python3
"""Singleton Package for Actuarial Chatbot Services.

This package provides singleton pattern implementation and service management
for preventing duplicate initialization of heavy services.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

from .singleton_meta import SingletonMeta
from .service_manager import ServiceManager, get_service_manager
from .decorators import (
    singleton_service,
    lazy_initialization,
    get_or_create_service
)

__all__ = [
    'SingletonMeta',
    'ServiceManager',
    'get_service_manager',
    'singleton_service',
    'lazy_initialization',
    'get_or_create_service'
]