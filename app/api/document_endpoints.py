#!/usr/bin/env python3
"""
Document API Endpoints for Actuarial Chatbot

This module provides REST API endpoints for document upload, parsing, and retrieval
functionality integrated with the PSAK219 document parser.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError

# Import document service
from ..services.documents.document_session_service import (
    get_document_service,
    upload_and_parse_document,
    query_document_data
)

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
document_bp = Blueprint('api_document', __name__, url_prefix='/api/documents')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.md', '.txt', '.pdf', '.docx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_file_size(file_content: bytes) -> bool:
    """Check if file size is within limits"""
    return len(file_content) <= MAX_FILE_SIZE


@document_bp.errorhandler(BadRequest)
def handle_bad_request(error):
    """Handle bad request errors"""
    return jsonify({
        'success': False,
        'error': 'Bad Request',
        'message': str(error.description)
    }), 400


@document_bp.errorhandler(NotFound)
def handle_not_found(error):
    """Handle not found errors"""
    return jsonify({
        'success': False,
        'error': 'Not Found',
        'message': str(error.description)
    }), 404


@document_bp.errorhandler(InternalServerError)
def handle_internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


@document_bp.route('/upload', methods=['POST'])
def upload_document():
    """Upload a document for parsing"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            raise BadRequest('No file provided')
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            raise BadRequest('No file selected')
        
        # Validate filename
        if not allowed_file(file.filename):
            raise BadRequest(f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}')
        
        # Read file content
        file_content = file.read()
        
        # Validate file size
        if not validate_file_size(file_content):
            raise BadRequest(f'File size exceeds maximum limit of {MAX_FILE_SIZE // (1024*1024)}MB')
        
        # Get document type from form data
        document_type = request.form.get('document_type', 'PSAK219')
        
        # Secure filename
        filename = secure_filename(file.filename)
        
        # Upload document
        service = get_document_service()
        document_id = service.upload_document(file_content, filename, document_type)
        
        logger.info(f"Document uploaded successfully: {document_id}")
        
        return jsonify({
            'success': True,
            'message': 'Document uploaded successfully',
            'data': {
                'document_id': document_id,
                'filename': filename,
                'document_type': document_type,
                'upload_timestamp': datetime.now().isoformat()
            }
        }), 201
        
    except BadRequest:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise InternalServerError(f"Failed to upload document: {str(e)}")


@document_bp.route('/parse/<document_id>', methods=['POST'])
def parse_document(document_id: str):
    """Parse an uploaded document"""
    try:
        service = get_document_service()
        
        # Check if document exists
        if document_id not in service.document_registry:
            raise NotFound(f'Document not found: {document_id}')
        
        # Parse document
        parsed_data = service.parse_document(document_id)
        
        logger.info(f"Document parsed successfully: {document_id}")
        
        return jsonify({
            'success': True,
            'message': 'Document parsed successfully',
            'data': {
                'document_id': document_id,
                'parsing_timestamp': datetime.now().isoformat(),
                'session_key': service.document_registry[document_id].session_key,
                'parsed_sections': list(parsed_data.keys()) if parsed_data else []
            }
        }), 200
        
    except NotFound:
        raise
    except Exception as e:
        logger.error(f"Error parsing document {document_id}: {str(e)}")
        raise InternalServerError(f"Failed to parse document: {str(e)}")


@document_bp.route('/upload-and-parse', methods=['POST'])
def upload_and_parse():
    """Upload and immediately parse a document"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            raise BadRequest('No file provided')
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            raise BadRequest('No file selected')
        
        # Validate filename
        if not allowed_file(file.filename):
            raise BadRequest(f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}')
        
        # Read file content
        file_content = file.read()
        
        # Validate file size
        if not validate_file_size(file_content):
            raise BadRequest(f'File size exceeds maximum limit of {MAX_FILE_SIZE // (1024*1024)}MB')
        
        # Secure filename
        filename = secure_filename(file.filename)
        
        # Get session_id from form data if provided
        session_id = request.form.get('session_id')
        
        # Upload and parse document with session_id
        service = get_document_service()
        document_id = service.upload_document(file_content, filename)
        
        # Parse document with session_id if provided
        if session_id:
            # Parse with session_id for PSAK219 data storage
            metadata = service.document_registry.get(document_id)
            if metadata:
                # Process the document with session_id for PSAK219 parsing
                from ..services.document_processor import DocumentProcessor
                processor = DocumentProcessor()
                processor.process_markdown_file(metadata.file_path, session_id, metadata.original_filename)
                
                # Update parsing status
                metadata.parsing_status = "completed"
                service._save_document_registry()
        else:
            # Parse without session_id (legacy behavior)
            service.parse_document(document_id)
        
        # Get document summary
        summary = service.get_document_summary(document_id)
        
        logger.info(f"Document uploaded and parsed successfully: {document_id} with session_id: {session_id}")
        
        return jsonify({
            'success': True,
            'message': 'Document uploaded and parsed successfully',
            'data': {
                'document_id': document_id,
                'filename': filename,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
        }), 201
        
    except BadRequest:
        raise
    except Exception as e:
        logger.error(f"Error uploading and parsing document: {str(e)}")
        raise InternalServerError(f"Failed to upload and parse document: {str(e)}")


@document_bp.route('/list', methods=['GET'])
def list_documents():
    """Get list of all uploaded documents"""
    try:
        service = get_document_service()
        documents = service.get_document_list()
        
        return jsonify({
            'success': True,
            'message': f'Retrieved {len(documents)} documents',
            'data': {
                'documents': documents,
                'total_count': len(documents)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise InternalServerError(f"Failed to list documents: {str(e)}")


@document_bp.route('/<document_id>', methods=['GET'])
def get_document_info(document_id: str):
    """Get information about a specific document"""
    try:
        service = get_document_service()
        
        # Check if document exists
        if document_id not in service.document_registry:
            raise NotFound(f'Document not found: {document_id}')
        
        # Get document summary
        summary = service.get_document_summary(document_id)
        
        return jsonify({
            'success': True,
            'message': 'Document information retrieved successfully',
            'data': summary
        }), 200
        
    except NotFound:
        raise
    except Exception as e:
        logger.error(f"Error getting document info for {document_id}: {str(e)}")
        raise InternalServerError(f"Failed to get document information: {str(e)}")


@document_bp.route('/<document_id>/data', methods=['GET'])
def get_document_data(document_id: str):
    """Get parsed data from a document"""
    try:
        service = get_document_service()
        
        # Check if document exists
        if document_id not in service.document_registry:
            raise NotFound(f'Document not found: {document_id}')
        
        # Get optional key parameter
        key = request.args.get('key')
        
        # Get document data
        data = service.get_document_data(document_id, key)
        
        return jsonify({
            'success': True,
            'message': 'Document data retrieved successfully',
            'data': {
                'document_id': document_id,
                'key': key,
                'content': data
            }
        }), 200
        
    except NotFound:
        raise
    except ValueError as e:
        raise BadRequest(str(e))
    except Exception as e:
        logger.error(f"Error getting document data for {document_id}: {str(e)}")
        raise InternalServerError(f"Failed to get document data: {str(e)}")


@document_bp.route('/<document_id>/search', methods=['GET'])
def search_document(document_id: str):
    """Search within a specific document"""
    try:
        # Get query parameter
        query = request.args.get('q')
        if not query:
            raise BadRequest('Query parameter "q" is required')
        
        # Get optional threshold parameter
        threshold = int(request.args.get('threshold', 70))
        
        service = get_document_service()
        
        # Check if document exists
        if document_id not in service.document_registry:
            raise NotFound(f'Document not found: {document_id}')
        
        # Search document
        results = service.search_document_data(document_id, query, threshold)
        
        return jsonify({
            'success': True,
            'message': f'Search completed for document {document_id}',
            'data': {
                'document_id': document_id,
                'query': query,
                'threshold': threshold,
                'results': results,
                'result_count': len(results)
            }
        }), 200
        
    except NotFound:
        raise
    except BadRequest:
        raise
    except ValueError as e:
        raise BadRequest(str(e))
    except Exception as e:
        logger.error(f"Error searching document {document_id}: {str(e)}")
        raise InternalServerError(f"Failed to search document: {str(e)}")


@document_bp.route('/search', methods=['GET'])
def search_all_documents():
    """Search across all documents"""
    try:
        # Get query parameter
        query = request.args.get('q')
        if not query:
            raise BadRequest('Query parameter "q" is required')
        
        # Get optional threshold parameter
        threshold = int(request.args.get('threshold', 70))
        
        # Search all documents
        results = query_document_data(query)
        
        # Count total results
        total_results = sum(len(doc_results) for doc_results in results.values())
        
        return jsonify({
            'success': True,
            'message': f'Search completed across all documents',
            'data': {
                'query': query,
                'threshold': threshold,
                'results': results,
                'documents_searched': len(results),
                'total_results': total_results
            }
        }), 200
        
    except BadRequest:
        raise
    except Exception as e:
        logger.error(f"Error searching all documents: {str(e)}")
        raise InternalServerError(f"Failed to search documents: {str(e)}")


@document_bp.route('/<document_id>/export', methods=['GET'])
def export_document(document_id: str):
    """Export document data as JSON"""
    try:
        service = get_document_service()
        
        # Check if document exists
        if document_id not in service.document_registry:
            raise NotFound(f'Document not found: {document_id}')
        
        # Get document data
        data = service.get_document_data(document_id)
        metadata = service.document_registry[document_id]
        
        # Prepare export data
        export_data = {
            'document_metadata': {
                'document_id': document_id,
                'original_filename': metadata.original_filename,
                'document_type': metadata.document_type,
                'upload_timestamp': metadata.upload_timestamp,
                'export_timestamp': datetime.now().isoformat()
            },
            'parsed_data': data
        }
        
        return jsonify({
            'success': True,
            'message': 'Document data exported successfully',
            'data': export_data
        }), 200
        
    except NotFound:
        raise
    except ValueError as e:
        raise BadRequest(str(e))
    except Exception as e:
        logger.error(f"Error exporting document {document_id}: {str(e)}")
        raise InternalServerError(f"Failed to export document: {str(e)}")


@document_bp.route('/<document_id>', methods=['DELETE'])
def delete_document(document_id: str):
    """Delete a document and its data"""
    try:
        service = get_document_service()
        
        # Check if document exists
        if document_id not in service.document_registry:
            raise NotFound(f'Document not found: {document_id}')
        
        # Delete document
        success = service.delete_document(document_id)
        
        if success:
            logger.info(f"Document deleted successfully: {document_id}")
            return jsonify({
                'success': True,
                'message': 'Document deleted successfully',
                'data': {
                    'document_id': document_id,
                    'deletion_timestamp': datetime.now().isoformat()
                }
            }), 200
        else:
            raise InternalServerError('Failed to delete document')
        
    except NotFound:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise InternalServerError(f"Failed to delete document: {str(e)}")


@document_bp.route('/list-session-documents', methods=['GET'])
def list_session_documents():
    """List documents for a specific session with optional global documents.
    
    Query parameters:
    - session_id: Session ID to filter documents (required)
    - include_global: Whether to include global documents (default: true)
    
    Returns:
        JSON response with document list
    """
    try:
        # Get query parameters
        session_id = request.args.get('session_id')
        if not session_id:
            raise BadRequest('session_id parameter is required')
        
        include_global = request.args.get('include_global', 'true').lower() == 'true'
        
        # Use VectorStoreManager instead of DocumentSessionService to access vector store documents
        from app.models.embeddings.vector_store_manager import VectorStoreManager
        from app.utils.singleton_manager import get_service_manager
        
        service_manager = get_service_manager()
        vector_store_manager = service_manager.get_service('vector_store_manager')
        
        if not vector_store_manager:
            # Create new instance if not found
            vector_store_manager = VectorStoreManager()
        
        # Get documents for session using vector store manager
        documents = vector_store_manager.list_documents_for_session(session_id, include_global)
        
        return jsonify({
            'success': True,
            'message': f'Retrieved {len(documents)} documents for session {session_id}',
            'data': {
                'session_id': session_id,
                'include_global': include_global,
                'documents': documents,
                'total_count': len(documents)
            }
        }), 200
        
    except BadRequest:
        raise
    except Exception as e:
        logger.error(f"Error getting document info for list-session-documents: {str(e)}")
        raise InternalServerError(f"Failed to get document information: {str(e)}")


@document_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        service = get_document_service()
        document_count = len(service.document_registry)
        
        return jsonify({
            'success': True,
            'message': 'Document service is healthy',
            'data': {
                'service_status': 'healthy',
                'document_count': document_count,
                'timestamp': datetime.now().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Document service is unhealthy',
            'error': str(e)
        }), 503


@document_bp.route('/psak219/session/<session_id>', methods=['GET'])
def get_psak219_by_session(session_id: str):
    """Get all PSAK219 parsed data for a specific session ID"""
    try:
        service = get_document_service()
        
        # Get PSAK219 data from session
        psak219_data = service.get_psak219_data_by_session(session_id)
        
        if not psak219_data:
            return jsonify({
                'success': True,
                'message': 'No PSAK219 documents found for this session',
                'data': {
                    'session_id': session_id,
                    'documents': [],
                    'total_documents': 0
                }
            }), 200
        
        # Format response
        return jsonify({
            'success': True,
            'message': f'Retrieved {len(psak219_data)} PSAK219 documents',
            'data': {
                'session_id': session_id,
                'total_documents': len(psak219_data),
                'documents': psak219_data,
                'retrieved_at': datetime.now().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting PSAK219 data for session {session_id}: {str(e)}")
        raise InternalServerError(f"Failed to retrieve PSAK219 data: {str(e)}")


@document_bp.route('/psak219/search/<session_id>', methods=['POST'])
def search_psak219_by_session(session_id: str):
    """Search PSAK219 data within a specific session"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'query' not in data:
            raise BadRequest('Query parameter is required in request body')
        
        query = data['query']
        threshold = data.get('threshold', 70)
        
        # Validate threshold
        if not isinstance(threshold, int) or threshold < 0 or threshold > 100:
            threshold = 70
        
        service = get_document_service()
        
        # Search PSAK219 data
        search_results = service.search_psak219_by_session(session_id, query, threshold)
        
        # Format response
        return jsonify({
            'success': True,
            'message': f'Search completed for session {session_id}',
            'data': {
                'session_id': session_id,
                'query': query,
                'threshold': threshold,
                'total_matches': len(search_results),
                'results': search_results,
                'searched_at': datetime.now().isoformat()
            }
        }), 200
        
    except BadRequest:
        raise
    except Exception as e:
        logger.error(f"Error searching PSAK219 data in session {session_id}: {str(e)}")
        raise InternalServerError(f"Failed to search PSAK219 data: {str(e)}")


@document_bp.route('/debug/session-keys', methods=['GET'])
def debug_session_keys():
    """Debug endpoint to view all session keys"""
    try:
        service = get_document_service()
        session_keys = service.session_manager.get_all_session_keys()
        
        return jsonify({
            'success': True,
            'data': {
                'session_keys': session_keys,
                'total_keys': len(session_keys)
            },
            'message': f'Found {len(session_keys)} session keys'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting session keys: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting session keys: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500


if __name__ == '__main__':
    # Test the endpoints
    from flask import Flask
    
    app = Flask(__name__)
    app.register_blueprint(document_bp)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Document API endpoints registered:")
    for rule in app.url_map.iter_rules():
        if rule.endpoint.startswith('document.'):
            print(f"  {rule.methods} {rule.rule} -> {rule.endpoint}")
    
    print("\nTo test the API, run the Flask application and use these endpoints:")
    print("  POST /api/documents/upload - Upload a document")
    print("  POST /api/documents/upload-and-parse - Upload and parse a document")
    print("  GET /api/documents/list - List all documents")
    print("  GET /api/documents/<id> - Get document info")
    print("  GET /api/documents/<id>/data - Get document data")
    print("  GET /api/documents/<id>/search?q=query - Search within document")
    print("  GET /api/documents/search?q=query - Search all documents")
    print("  GET /api/documents/psak219/session/<session_id> - Get PSAK219 data by session")
    print("  POST /api/documents/psak219/search/<session_id> - Search PSAK219 data by session")
    print("  DELETE /api/documents/<id> - Delete document")