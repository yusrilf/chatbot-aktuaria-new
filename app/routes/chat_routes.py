"""Chat and conversation routes for actuarial chatbot application.

This module contains endpoints for chat interactions, conversation history,
and related chat functionalities.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any, List
import logging
import traceback

from app.services.enhanced_chat_service import EnhancedActuarialChatService
from app.utils.singleton_manager import get_or_create_service
from app.utils.helpers import create_response

logger = logging.getLogger(__name__)

# Create blueprint
chat_bp = Blueprint('chat', __name__)


def get_chat_service() -> EnhancedActuarialChatService:
    """Get or create chat service instance.
    
    Returns:
        EnhancedActuarialChatService instance
    """
    return get_or_create_service(EnhancedActuarialChatService, 'enhanced_chat_service')


@chat_bp.route('/askproject', methods=['POST'])
async def ask_project() -> Dict[str, Any]:
    """Handle project-specific questions.
    
    Returns:
        Dict containing response to project question
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return create_response(
                success=False,
                message="Question is required"
            ), 400
            
        question = data['question']
        session_id = data.get('session_id', 'default')
        
        chat_svc = get_chat_service()
        result = await chat_svc.ask_project(question, session_id)
        
        return create_response(
            success=True,
            message="Question processed successfully",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error in ask_project: {str(e)}")
        logger.error(traceback.format_exc())
        return create_response(
            success=False,
            message=f"Error processing question: {str(e)}"
        ), 500


@chat_bp.route('/ask', methods=['POST'])
async def ask_question() -> Dict[str, Any]:
    """Handle general questions.
    
    Returns:
        Dict containing response to general question
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return create_response(
                success=False,
                message="Question is required"
            ), 400
            
        question = data['question']
        session_id = data.get('session_id', 'default')
        
        chat_svc = get_chat_service()
        result = await chat_svc.ask_question(question, session_id)
        
        return create_response(
            success=True,
            message="Question processed successfully",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        logger.error(traceback.format_exc())
        return create_response(
            success=False,
            message=f"Error processing question: {str(e)}"
        ), 500


@chat_bp.route('/askdeep', methods=['POST'])
async def ask_deep() -> Dict[str, Any]:
    """Handle deep session-only RAG questions.
    
    Hanya mengambil konteks dari dokumen yang diunggah pada session_id
    pengguna spesifik, tanpa membaca global.
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return create_response(
                success=False,
                message="Question is required"
            ), 400
        
        question = data['question']
        session_id = data.get('session_id', 'default')
        
        chat_svc = get_chat_service()
        result = await chat_svc.ask_deep(question, session_id)
        
        return create_response(
            success=True,
            message="Deep session question processed",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error in ask_deep: {str(e)}")
        logger.error(traceback.format_exc())
        return create_response(
            success=False,
            message=f"Error processing askdeep: {str(e)}"
        ), 500


@chat_bp.route('/datastory', methods=['POST'])
def datastory() -> Dict[str, Any]:
    """Handle data story generation.
    
    Returns:
        Dict containing generated data story
    """
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return create_response(
                success=False,
                message="Data is required"
            ), 400
            
        data_content = data['data']
        session_id = data.get('session_id', 'default')
        
        chat_svc = get_chat_service()
        result = chat_svc.datastory(data_content, session_id)
        
        return create_response(
            success=True,
            message="Data story generated successfully",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error in datastory: {str(e)}")
        logger.error(traceback.format_exc())
        return create_response(
            success=False,
            message=f"Error generating data story: {str(e)}"
        ), 500


@chat_bp.route('/conversation/history', methods=['GET'])
def get_conversation_history() -> Dict[str, Any]:
    """Get conversation history.
    
    Returns:
        Dict containing conversation history
    """
    try:
        session_id = request.args.get('session_id', 'default')
        limit = int(request.args.get('limit', 10))
        
        chat_svc = get_chat_service()
        history = chat_svc.get_conversation_history(session_id, limit=limit)
        
        return create_response(
            success=True,
            message="Conversation history retrieved",
            data={"history": history}
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        return create_response(
            success=False,
            message=f"Error getting conversation history: {str(e)}"
        ), 500


@chat_bp.route('/conversation/clear', methods=['POST'])
def clear_conversation() -> Dict[str, Any]:
    """Clear conversation history.
    
    Returns:
        Dict containing clear operation result
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        
        chat_svc = get_chat_service()
        success = chat_svc.clear_memory(session_id)
        
        if success:
            return create_response(
                success=True,
                message="Conversation cleared successfully"
            )
        else:
            return create_response(
                success=False,
                message="Failed to clear conversation"
            ), 500
            
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        return create_response(
            success=False,
            message=f"Error clearing conversation: {str(e)}"
        ), 500