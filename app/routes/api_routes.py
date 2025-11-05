"""API routes for additional endpoints.

This module contains API routes that don't fit into other specific categories.
"""

from flask import Blueprint, jsonify, request
import os
import shutil
import tempfile
from typing import Dict, Any, List

from app.utils.helpers import create_response
import logging
from app.services.chat.chat_service import ActuarialChatService
from app.models.embeddings import VectorStoreManager
from app.services.parser.session_manager import SessionManager as ParserSessionManager
from app.utils.singleton_manager import get_or_create_service

# Initialize logger
logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/version', methods=['GET'])
def get_version():
    """Get API version information.
    
    Returns:
        JSON response with version information
    """
    try:
        return jsonify(create_response(
            success=True,
            message="API version retrieved",
            data={
                "version": "1.0.0",
                "api_name": "PSAK219 Chatbot API",
                "description": "API for PSAK219 document processing and Q&A"
            }
        ))
    except Exception as e:
        logger.error(f"Error getting version: {str(e)}")
        return jsonify(create_response(
            success=False,
            message="Error getting version",
            data={'error': str(e)}
        )), 500

@api_bp.route('/status', methods=['GET'])
def get_status():
    """Get API status information.
    
    Returns:
        JSON response with API status
    """
    try:
        return jsonify(create_response(
            success=True,
            message="API is operational",
            data={
                "status": "operational",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        ))
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify(create_response(
            success=False,
            message="Error getting status",
            data={'error': str(e)}
        )), 500


@api_bp.route('/storage/refresh', methods=['POST'])
def refresh_storage() -> Dict[str, Any]:
    """Refresh and clear all storage components.
    
    This endpoint clears:
    - Session memories and chat history
    - Vector store collections
    - Temporary files
    - Parser session data
    - Cache data
    
    Returns:
        JSON response with refresh operation results
    """
    try:
        refresh_results = []
        
        # 1. Clear chat service session memories
        try:
            chat_service = get_or_create_service(ActuarialChatService, 'chat_service')
            success = chat_service.clear_memory()  # Clear all sessions
            refresh_results.append({
                'component': 'chat_sessions',
                'status': 'success' if success else 'failed',
                'message': 'All chat session memories cleared' if success else 'Failed to clear chat sessions'
            })
        except Exception as e:
            logger.error(f"Error clearing chat sessions: {str(e)}")
            refresh_results.append({
                'component': 'chat_sessions',
                'status': 'error',
                'message': f'Error clearing chat sessions: {str(e)}'
            })
        
        # 2. Clear vector store collections
        try:
            vector_store = get_or_create_service(VectorStoreManager, 'vector_store_manager')
            success = vector_store.delete_collection()
            refresh_results.append({
                'component': 'vector_store',
                'status': 'success' if success else 'failed',
                'message': 'Vector store collection deleted' if success else 'Failed to delete vector store collection'
            })
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            refresh_results.append({
                'component': 'vector_store',
                'status': 'error',
                'message': f'Error clearing vector store: {str(e)}'
            })
        
        # 3. Clear parser session data
        try:
            parser_session = ParserSessionManager()
            parser_session.clear_session()
            refresh_results.append({
                'component': 'parser_sessions',
                'status': 'success',
                'message': 'Parser session data cleared'
            })
        except Exception as e:
            logger.error(f"Error clearing parser sessions: {str(e)}")
            refresh_results.append({
                'component': 'parser_sessions',
                'status': 'error',
                'message': f'Error clearing parser sessions: {str(e)}'
            })
        
        # 4. Clear temporary files
        try:
            temp_dirs_cleared = 0
            temp_files_cleared = 0
            
            # Clear system temp directory files related to our app
            temp_dir = tempfile.gettempdir()
            for filename in os.listdir(temp_dir):
                if any(prefix in filename for prefix in ['global_', 'psak_', 'auto_global_']):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            temp_files_cleared += 1
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            temp_dirs_cleared += 1
                    except Exception as file_error:
                        logger.warning(f"Could not remove temp file {file_path}: {file_error}")
            
            # Clear uploaded_documents directory
            uploaded_docs_path = os.path.join(os.getcwd(), 'uploaded_documents')
            if os.path.exists(uploaded_docs_path):
                for filename in os.listdir(uploaded_docs_path):
                    file_path = os.path.join(uploaded_docs_path, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            temp_files_cleared += 1
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            temp_dirs_cleared += 1
                    except Exception as file_error:
                        logger.warning(f"Could not remove uploaded file {file_path}: {file_error}")
            
            refresh_results.append({
                'component': 'temporary_files',
                'status': 'success',
                'message': f'Cleared {temp_files_cleared} files and {temp_dirs_cleared} directories'
            })
        except Exception as e:
            logger.error(f"Error clearing temporary files: {str(e)}")
            refresh_results.append({
                'component': 'temporary_files',
                'status': 'error',
                'message': f'Error clearing temporary files: {str(e)}'
            })
        
        # 5. Clear data directory (optional - be careful with this)
        data_path = os.path.join(os.getcwd(), 'data')
        if os.path.exists(data_path):
            try:
                # Only clear specific subdirectories, not the entire data folder
                vectorstore_path = os.path.join(data_path, 'vectorstore')
                if os.path.exists(vectorstore_path):
                    shutil.rmtree(vectorstore_path)
                    refresh_results.append({
                        'component': 'vectorstore_data',
                        'status': 'success',
                        'message': 'Vectorstore data directory cleared'
                    })
            except Exception as e:
                logger.error(f"Error clearing vectorstore data: {str(e)}")
                refresh_results.append({
                    'component': 'vectorstore_data',
                    'status': 'error',
                    'message': f'Error clearing vectorstore data: {str(e)}'
                })
        
        # Count successful operations
        successful_ops = sum(1 for result in refresh_results if result['status'] == 'success')
        total_ops = len(refresh_results)
        
        logger.info(f"Storage refresh completed: {successful_ops}/{total_ops} operations successful")
        
        return jsonify(create_response(
            success=successful_ops > 0,
            message=f"Storage refresh completed: {successful_ops}/{total_ops} operations successful",
            data={
                'refresh_results': refresh_results,
                'summary': {
                    'total_operations': total_ops,
                    'successful_operations': successful_ops,
                    'failed_operations': total_ops - successful_ops
                }
            }
        ))
        
    except Exception as e:
        logger.error(f"Error in refresh_storage: {str(e)}")
        return jsonify(create_response(
            success=False,
            message=f"Error refreshing storage: {str(e)}",
            data={'error': str(e)}
        )), 500


@api_bp.route('/storage/clear-sessions', methods=['POST'])
def clear_sessions() -> Dict[str, Any]:
    """Clear only session memories and chat history.
    
    Accepts session_id via:
    - JSON body: {"session_id": "your_session_id"}
    - Query parameter: ?session_id=your_session_id
    
    Returns:
        JSON response with session clearing results
    """
    try:
        # Support both JSON body and query parameters
        data = {}
        if request.is_json and request.get_json():
            data = request.get_json()
        session_id = data.get('session_id') or request.args.get('session_id')  # Optional: clear specific session
        
        # Clear chat service sessions
        chat_service = get_or_create_service(ActuarialChatService, 'chat_service')
        success = chat_service.clear_memory(session_id)
        
        message = f"Session {session_id} cleared" if session_id else "All sessions cleared"
        
        return jsonify(create_response(
            success=success,
            message=message if success else f"Failed to clear sessions",
            data={
                'session_id': session_id,
                'cleared_all': session_id is None
            }
        ))
        
    except Exception as e:
        logger.error(f"Error clearing sessions: {str(e)}")
        return jsonify(create_response(
            success=False,
            message=f"Error clearing sessions: {str(e)}",
            data={'error': str(e)}
        )), 500


@api_bp.route('/storage/status', methods=['GET'])
def get_storage_status() -> Dict[str, Any]:
    """Get storage status information.
    
    Returns:
        JSON response with storage status details
    """
    try:
        status_info = {}
        
        # Check chat service sessions
        try:
            chat_service = get_or_create_service(ActuarialChatService, 'chat_service')
            session_count = len(getattr(chat_service.session_manager, 'session_memories', {}))
            status_info['chat_sessions'] = {
                'active_sessions': session_count,
                'status': 'healthy'
            }
        except Exception as e:
            status_info['chat_sessions'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check vector store
        try:
            vector_store = get_or_create_service(VectorStoreManager, 'vector_store_manager')
            # Try to get collection info
            collection_info = vector_store.get_collection_info()
            status_info['vector_store'] = {
                'status': 'healthy',
                'collection_info': collection_info
            }
        except Exception as e:
            status_info['vector_store'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check temporary files
        try:
            temp_dir = tempfile.gettempdir()
            temp_files = [f for f in os.listdir(temp_dir) 
                         if any(prefix in f for prefix in ['global_', 'psak_', 'auto_global_'])]
            
            uploaded_docs_path = os.path.join(os.getcwd(), 'uploaded_documents')
            uploaded_files = []
            if os.path.exists(uploaded_docs_path):
                uploaded_files = os.listdir(uploaded_docs_path)
            
            status_info['temporary_files'] = {
                'temp_files_count': len(temp_files),
                'uploaded_files_count': len(uploaded_files),
                'status': 'healthy'
            }
        except Exception as e:
            status_info['temporary_files'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check data directory
        try:
            data_path = os.path.join(os.getcwd(), 'data')
            vectorstore_path = os.path.join(data_path, 'vectorstore')
            
            status_info['data_storage'] = {
                'data_dir_exists': os.path.exists(data_path),
                'vectorstore_dir_exists': os.path.exists(vectorstore_path),
                'status': 'healthy'
            }
        except Exception as e:
            status_info['data_storage'] = {
                'status': 'error',
                'error': str(e)
            }
        
        return jsonify(create_response(
            success=True,
            message="Storage status retrieved",
            data=status_info
        ))
        
    except Exception as e:
        logger.error(f"Error getting storage status: {str(e)}")
        return jsonify(create_response(
            success=False,
            message=f"Error getting storage status: {str(e)}",
            data={'error': str(e)}
        )), 500