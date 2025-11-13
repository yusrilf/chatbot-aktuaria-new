"""Application factory for actuarial chatbot Flask application.

This module contains the application factory pattern implementation
for creating and configuring the Flask application.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

from flask import Flask
from flask_cors import CORS
from typing import Optional
import logging

from app.config import config
from app.utils.helpers import setup_logging
from app.utils.singleton_manager import get_service_manager


# Import blueprints
from app.routes.health_routes import health_bp
from app.routes.chat_routes import chat_bp
from app.routes.document_routes import document_bp
from app.routes.api_routes import api_bp
from app.routes.langgraph_routes import langgraph_bp
from app.api.document_endpoints import document_bp as api_document_bp

logger = logging.getLogger(__name__)


def create_app(config_object: Optional[object] = None) -> Flask:
    """Create and configure Flask application.
    
    Args:
        config_object: Configuration object to use (defaults to app.config)
        
    Returns:
        Configured Flask application instance
    """
    # Setup logging first
    setup_logging(config.LOG_LEVEL)
    logger.info("Creating Flask application")
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure app
    if config_object is None:
        config_object = config
    app.config.from_object(config_object)
    
    # Setup CORS
    CORS(app)  # Allow all origins (for development)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register root route
    register_root_route(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Initialize services
    initialize_services()
    
    logger.info("Flask application created successfully")
    return app


def register_blueprints(app: Flask) -> None:
    """Register all blueprints with the Flask application.
    
    Args:
        app: Flask application instance
    """
    logger.info("Registering blueprints")
    
    # Register route blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(document_bp)
    app.register_blueprint(langgraph_bp)

    # Register API blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(api_document_bp)
    
    logger.info("All blueprints registered successfully")


def register_root_route(app: Flask) -> None:
    """Register root route with the Flask application.
    
    Args:
        app: Flask application instance
    """
    logger.info("Registering root route")
    
    @app.route('/')
    def root():
        """Root endpoint providing API information."""
        return {
            "name": "Actuarial Chatbot API",
            "version": "1.1.0",
            "status": "running",
            "description": "AI-powered actuarial chatbot with PSAK 219 expertise",
            "endpoints": {
                "health": "/health",
                "chat": "/ask, /askproject",
                "langgraph": "/langgraph/ask, /langgraph/health, /langgraph/info",
                "documents": "/input-docs, /documents/*",
                "storage": "/api/storage/*",
                "api_docs": "See README.MD for complete API documentation"
            },
            "documentation": "https://github.com/your-repo/chatbot-aktuaria"
        }
    
    @app.route('/robots.txt')
    @app.route('/robots<path:filename>.txt')
    def robots_txt(filename=None):
        """Handle robots.txt requests (including security scanner patterns)."""
        return "User-agent: *\nDisallow: /api/\nDisallow: /admin/\nAllow: /health\n", 200, {'Content-Type': 'text/plain'}
    
    logger.info("Root route registered successfully")


def register_error_handlers(app: Flask) -> None:
    """Register error handlers with the Flask application.
    
    Args:
        app: Flask application instance
    """
    logger.info("Registering error handlers")
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return {"error": "Endpoint not found"}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {str(error)}")
        return {"error": "Internal server error"}, 500
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 errors."""
        return {"error": "Bad request"}, 400
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 errors."""
        return {"error": "Method not allowed"}, 405
    
    logger.info("Error handlers registered successfully")


def initialize_services() -> None:
    """Initialize application services.
    
    This function ensures that all required services are properly
    initialized using the singleton manager.
    """
    logger.info("Initializing application services")
    
    try:
        # Get service manager
        service_manager = get_service_manager()
        
        # Services will be initialized lazily when first accessed
        # This is handled by the singleton manager
        
        logger.info("Application services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        raise


def ensure_full_initialization() -> None:
    """Ensure all services are fully initialized.
    
    This function can be called to force initialization of all services
    that might be needed during application runtime.
    """
    logger.info("Ensuring full service initialization")
    
    try:
        from app.models.embeddings import VectorStoreManager
        from app.services.chat.chat_service import ActuarialChatService
        from app.utils.singleton_manager import get_or_create_service
        
        # Force initialization of key services
        get_or_create_service(VectorStoreManager, 'vector_store_manager')
        get_or_create_service(ActuarialChatService, 'chat_service')
        
        logger.info("Full service initialization completed")
        
    except Exception as e:
        logger.error(f"Error in full initialization: {str(e)}")
        raise