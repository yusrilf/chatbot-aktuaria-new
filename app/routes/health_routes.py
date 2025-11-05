"""Health check routes for actuarial chatbot application.

This module contains health check and system status endpoints with graceful degradation.

Author: AI Assistant
Date: 2025-01-04
Version: 1.1.0
"""

from flask import Blueprint, jsonify
from typing import Dict, Any
import logging
import time

from app.utils.helpers import create_response
from app.config import config

logger = logging.getLogger(__name__)

# Create blueprint
health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """Health check endpoint with graceful degradation.
    
    This endpoint always returns 200 OK to prevent Azure from marking
    the deployment as failed, even if some services are unavailable.
    
    Returns:
        Dict containing health status and system information
    """
    health_data = {
        "status": "healthy",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "services": {},
        "warnings": []
    }
    
    # Check VectorStoreManager availability
    try:
        from app.utils.singleton_manager import get_or_create_service
        from app.models.embeddings import VectorStoreManager
        
        vector_store_manager = get_or_create_service(
            VectorStoreManager, 
            'vector_store_manager'
        )
        
        if vector_store_manager and hasattr(vector_store_manager, 'vectorstore') and vector_store_manager.vectorstore:
            # Try to get collection info as health check
            try:
                collection = vector_store_manager.chroma_client.get_collection(config.COLLECTION_NAME)
                doc_count = collection.count() if collection else 0
                health_data["services"]["vector_store"] = "operational"
                health_data["document_count"] = doc_count
            except Exception as e:
                logger.warning(f"ChromaDB collection check failed: {str(e)}")
                health_data["services"]["vector_store"] = "degraded"
                health_data["document_count"] = 0
                health_data["warnings"].append("ChromaDB collection unavailable")
        else:
            logger.warning("VectorStoreManager not properly initialized")
            health_data["services"]["vector_store"] = "unavailable"
            health_data["document_count"] = 0
            health_data["warnings"].append("VectorStore not initialized")
            
    except Exception as e:
        logger.warning(f"VectorStoreManager check failed: {str(e)}")
        health_data["services"]["vector_store"] = "error"
        health_data["document_count"] = 0
        health_data["warnings"].append(f"VectorStore error: {str(e)}")
    
    # Check ChatService availability
    try:
        from app.services.chat.chat_service import ActuarialChatService
        # Don't initialize, just check if class is importable
        health_data["services"]["chat_service"] = "available"
    except Exception as e:
        logger.warning(f"ChatService check failed: {str(e)}")
        health_data["services"]["chat_service"] = "error"
        health_data["warnings"].append(f"ChatService error: {str(e)}")
    
    # Always return 200 OK for Azure health checks
    return create_response(
        success=True,
        message="Service is running" + (" with warnings" if health_data["warnings"] else ""),
        data=health_data
    ), 200


@health_bp.route('/health/detailed', methods=['GET'])
def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check endpoint that may return error codes.
    
    This endpoint provides more detailed diagnostics and may return
    error codes for monitoring systems that need granular status.
    
    Returns:
        Dict containing detailed health status
    """
    health_data = {
        "status": "healthy",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "services": {},
        "errors": [],
        "warnings": []
    }
    
    has_critical_errors = False
    
    # Detailed VectorStoreManager check
    try:
        from app.utils.singleton_manager import get_or_create_service
        from app.models.embeddings import VectorStoreManager
        
        vector_store_manager = get_or_create_service(
            VectorStoreManager, 
            'vector_store_manager'
        )
        
        if vector_store_manager and hasattr(vector_store_manager, 'vectorstore') and vector_store_manager.vectorstore:
            try:
                # Test ChromaDB operations
                collection = vector_store_manager.chroma_client.get_collection(config.COLLECTION_NAME)
                doc_count = collection.count() if collection else 0
                
                # Test embedding functionality
                if hasattr(vector_store_manager, 'embedding_service'):
                    health_data["services"]["vector_store"] = {
                        "status": "operational",
                        "document_count": doc_count,
                        "embedding_service": "available",
                        "chroma_path": getattr(config, 'CHROMA_DB_PATH', 'unknown')
                    }
                else:
                    health_data["services"]["vector_store"] = {
                        "status": "degraded",
                        "document_count": doc_count,
                        "embedding_service": "unavailable"
                    }
                    health_data["warnings"].append("Embedding service not available")
                    
            except Exception as e:
                logger.error(f"ChromaDB operations failed: {str(e)}")
                health_data["services"]["vector_store"] = {
                    "status": "error",
                    "error": str(e)
                }
                health_data["errors"].append(f"ChromaDB error: {str(e)}")
                has_critical_errors = True
        else:
            health_data["services"]["vector_store"] = {
                "status": "unavailable",
                "error": "VectorStore not initialized"
            }
            health_data["errors"].append("VectorStore initialization failed")
            has_critical_errors = True
            
    except Exception as e:
        logger.error(f"VectorStoreManager initialization failed: {str(e)}")
        health_data["services"]["vector_store"] = {
            "status": "critical_error",
            "error": str(e)
        }
        health_data["errors"].append(f"VectorStore critical error: {str(e)}")
        has_critical_errors = True
    
    # Detailed ChatService check
    try:
        from app.services.chat.chat_service import ActuarialChatService
        health_data["services"]["chat_service"] = {
            "status": "available",
            "class_importable": True
        }
    except Exception as e:
        logger.error(f"ChatService import failed: {str(e)}")
        health_data["services"]["chat_service"] = {
            "status": "error",
            "error": str(e)
        }
        health_data["errors"].append(f"ChatService error: {str(e)}")
    
    # Determine overall status
    if has_critical_errors:
        health_data["status"] = "unhealthy"
        return create_response(
            success=False,
            message="Service has critical errors",
            data=health_data
        ), 503
    elif health_data["warnings"]:
        health_data["status"] = "degraded"
        return create_response(
            success=True,
            message="Service is running with warnings",
            data=health_data
        ), 200
    else:
        return create_response(
            success=True,
            message="All services are healthy",
            data=health_data
        ), 200


@health_bp.route('/documents/stats', methods=['GET'])
def get_document_stats() -> Dict[str, Any]:
    """Get document statistics with graceful degradation.
    
    Returns:
        Dict containing document statistics or fallback data
    """
    try:
        from app.utils.singleton_manager import get_or_create_service
        from app.models.embeddings import VectorStoreManager
        
        vector_store_manager = get_or_create_service(
            VectorStoreManager, 
            'vector_store_manager'
        )
        
        if vector_store_manager and hasattr(vector_store_manager, 'vectorstore') and vector_store_manager.vectorstore:
            try:
                collection_info = vector_store_manager.get_collection_info()
                doc_count = collection_info.get('count', 0)
                
                return create_response(
                    success=True,
                    message="Document statistics retrieved",
                    data={
                        "total_documents": doc_count,
                        "status": "active",
                        "collection_info": collection_info
                    }
                )
            except Exception as e:
                logger.warning(f"Error getting collection info: {str(e)}")
                return create_response(
                    success=True,
                    message="Document statistics unavailable",
                    data={
                        "total_documents": 0,
                        "status": "degraded",
                        "error": str(e)
                    }
                )
        else:
            return create_response(
                success=True,
                message="Document service not initialized",
                data={
                    "total_documents": 0,
                    "status": "unavailable"
                }
            )
        
    except Exception as e:
        logger.error(f"Error in document stats: {str(e)}")
        return create_response(
            success=True,
            message="Document statistics service error",
            data={
                "total_documents": 0,
                "status": "error",
                "error": str(e)
            }
        )