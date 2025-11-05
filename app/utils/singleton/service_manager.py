#!/usr/bin/env python3
"""Service Manager for Centralized Service Registration.

This module provides centralized service management for the application,
allowing services to be registered and retrieved by name.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ServiceManager:
    """
    Centralized service manager for registering and retrieving services.
    
    This class provides a registry for application services, allowing
    them to be accessed by name throughout the application.
    """
    
    def __init__(self):
        """Initialize the service manager."""
        self._services: Dict[str, Any] = {}
        logger.debug("ServiceManager initialized")
    
    def register_service(self, name: str, service_instance: Any) -> None:
        """
        Register a service instance with a given name.
        
        Args:
            name: Service name for registration
            service_instance: Service instance to register
            
        Raises:
            ValueError: If service name already exists
        """
        if name in self._services:
            logger.warning(f"Service '{name}' already registered, overwriting")
        
        self._services[name] = service_instance
        logger.info(f"Service '{name}' registered successfully")
    
    def get_service(self, name: str) -> Optional[Any]:
        """
        Retrieve a service by name.
        
        Args:
            name: Service name to retrieve
            
        Returns:
            Service instance or None if not found
        """
        service = self._services.get(name)
        if service is None:
            logger.warning(f"Service '{name}' not found")
        else:
            logger.debug(f"Retrieved service '{name}'")
        return service
    
    def is_service_initialized(self, name: str) -> bool:
        """
        Check if a service is initialized and registered.
        
        Args:
            name: Service name to check
            
        Returns:
            True if service exists, False otherwise
        """
        exists = name in self._services
        logger.debug(f"Service '{name}' exists: {exists}")
        return exists
    
    def get_all_services(self) -> Dict[str, Any]:
        """
        Get all registered services.
        
        Returns:
            Dictionary of all registered services
        """
        logger.debug(f"Retrieved {len(self._services)} services")
        return self._services.copy()
    
    def clear_services(self) -> None:
        """
        Clear all registered services (useful for testing).
        """
        service_count = len(self._services)
        self._services.clear()
        logger.info(f"Cleared {service_count} services")

# Global service manager instance
_global_service_manager = ServiceManager()

def get_service_manager() -> ServiceManager:
    """
    Get the global service manager instance.
    
    Returns:
        Global ServiceManager instance
    """
    return _global_service_manager