"""LangGraph routes for actuarial chatbot application.

This module contains endpoints for the LangGraph-based RAG workflow.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any
import logging

from app.utils.helpers import create_response
from app.utils.singleton_manager import get_or_create_service
from app.services.langgraph_service import LangGraphRAGService

logger = logging.getLogger(__name__)

# Create blueprint
langgraph_bp = Blueprint('langgraph', __name__, url_prefix='/langgraph')


@langgraph_bp.route('/ask', methods=['POST'])
def langgraph_ask() -> Dict[str, Any]:
    """Process a question through the LangGraph RAG workflow.

    Request body:
        {
            "question": "Your question here"
        }

    Returns:
        JSON response with answer and metadata
    """
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return create_response(
                success=False,
                message="No data provided in request body",
                data=None
            ), 400

        question = data.get('question', '').strip()

        if not question:
            return create_response(
                success=False,
                message="Question is required",
                data=None
            ), 400

        logger.info(f"Received LangGraph query: {question[:100]}...")

        # Get or create LangGraph service
        try:
            langgraph_service = get_or_create_service(
                LangGraphRAGService,
                'langgraph_service'
            )
        except Exception as e:
            logger.error(f"Error initializing LangGraph service: {str(e)}")
            return create_response(
                success=False,
                message="LangGraph service initialization failed",
                data={"error": str(e)}
            ), 503

        # Process query
        result = langgraph_service.query(question)

        if result.get('success'):
            return create_response(
                success=True,
                message="Query processed successfully",
                data={
                    "answer": result.get('answer'),
                    "question": result.get('question'),
                    "metadata": {
                        "service": "langgraph",
                        "workflow": "rag_with_grading"
                    }
                }
            ), 200
        else:
            return create_response(
                success=False,
                message="Error processing query",
                data={"error": result.get('error')}
            ), 500

    except Exception as e:
        logger.error(f"Unexpected error in langgraph_ask: {str(e)}")
        return create_response(
            success=False,
            message="Internal server error",
            data={"error": str(e)}
        ), 500


@langgraph_bp.route('/health', methods=['GET'])
def langgraph_health() -> Dict[str, Any]:
    """Check health of the LangGraph service.

    Returns:
        JSON response with health status
    """
    try:
        # Try to get existing service (don't create if not exists)
        from app.utils.singleton_manager import get_service_manager
        service_manager = get_service_manager()

        if 'langgraph_service' not in service_manager._services:
            return create_response(
                success=True,
                message="LangGraph service not initialized yet",
                data={
                    "status": "not_initialized",
                    "note": "Service will be initialized on first request"
                }
            ), 200

        # Get existing service
        langgraph_service = service_manager.get_service('langgraph_service')

        if not langgraph_service:
            return create_response(
                success=True,
                message="LangGraph service not available",
                data={"status": "unavailable"}
            ), 200

        # Perform health check
        health_result = langgraph_service.health_check()

        status_code = 200 if health_result.get('status') in ['healthy', 'degraded'] else 503

        return create_response(
            success=True,
            message=f"LangGraph service is {health_result.get('status')}",
            data=health_result
        ), status_code

    except Exception as e:
        logger.error(f"Error in langgraph health check: {str(e)}")
        return create_response(
            success=False,
            message="Health check failed",
            data={
                "status": "error",
                "error": str(e)
            }
        ), 500


@langgraph_bp.route('/info', methods=['GET'])
def langgraph_info() -> Dict[str, Any]:
    """Get information about the LangGraph service.

    Returns:
        JSON response with service information
    """
    return create_response(
        success=True,
        message="LangGraph RAG service information",
        data={
            "name": "LangGraph RAG Service",
            "version": "1.0.0",
            "description": "RAG workflow with document grading and query rewriting",
            "endpoints": {
                "/langgraph/ask": "POST - Process a question through RAG workflow",
                "/langgraph/health": "GET - Check service health",
                "/langgraph/info": "GET - Get service information"
            },
            "workflow": {
                "steps": [
                    "1. Generate query or respond directly",
                    "2. Retrieve relevant documents (if needed)",
                    "3. Grade document relevance",
                    "4. Rewrite question (if documents not relevant)",
                    "5. Generate final answer"
                ]
            },
            "features": [
                "Adaptive retrieval (only retrieves when needed)",
                "Document relevance grading",
                "Query rewriting for better results",
                "Pinecone vector store integration"
            ]
        }
    ), 200
