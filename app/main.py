from flask import Flask, request, jsonify
import os
import logging
from typing import List
import traceback
import uuid
from app.config import config
from app.services.document_processor import DocumentProcessor
from app.services.chat_service import ActuarialChatService
from app.models.embeddings import VectorStoreManager
from app.utils.helpers import setup_logging, validate_files, validate_openai_key, create_response, get_file_size
from flask_cors import CORS

# Setup logging
setup_logging(config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config)

# Initialize services
document_processor = DocumentProcessor()
chat_service = ActuarialChatService()
vector_store_manager = VectorStoreManager()

def before_first_request():
    """Initialize app before first request"""
    logger.info("Initializing Actuarial Chatbot API")
    
    # Validate OpenAI API key
    if not validate_openai_key(config.OPENAI_API_KEY):
        logger.error("Invalid or missing OpenAI API key")
        raise ValueError("Invalid OpenAI API key")
    
    # Create necessary directories
    os.makedirs('data/documents', exist_ok=True)
    os.makedirs('data/vectorstore', exist_ok=True)
    
    logger.info("App initialized successfully")

before_first_request()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if services are working
        stats = chat_service.get_system_stats()
        
        return jsonify(create_response(
            success=True,
            message="Service is healthy",
            data={
                'status': 'healthy',
                'stats': stats
            }
        ))
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify(create_response(
            success=False,
            message="Service is unhealthy",
            data={'error': str(e)}
        )), 500

@app.route('/input-docs', methods=['POST'])
def input_documents():
    """Process and store markdown documents"""
    try:
        # Check if files are provided
        if 'files' not in request.files:
            return jsonify(create_response(
                success=False,
                message="No files provided"
            )), 400
        
        files = request.files.getlist('files')
        if not validate_files(files):
            return jsonify(create_response(
                success=False,
                message="No files selected"
            )), 400
        
        processed_files = []
        total_chunks = 0
        
        # Process each file
        for file in files:
            if file and file.filename.lower().endswith('.md'):
                # Save file temporarily
                filename = file.filename
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                temp_path = os.path.join('data/documents', unique_filename)
                #temp_path = os.path.join('data/documents', filename)
                file.save(temp_path)
                
                # Validate and process file
                if document_processor.validate_file(temp_path):
                    documents = document_processor.process_markdown_file(temp_path)
                    
                    if documents:
                        # Add to vector store
                        success = vector_store_manager.add_documents(documents)
                        
                        if success:
                            processed_files.append({
                                'filename': filename,
                                'chunks': len(documents),
                                'size': get_file_size(temp_path),
                                'status': 'success'
                            })
                            total_chunks += len(documents)
                        else:
                            processed_files.append({
                                'filename': filename,
                                'status': 'failed_to_store'
                            })
                    else:
                        processed_files.append({
                            'filename': filename,
                            'status': 'failed_to_process'
                        })
                else:
                    processed_files.append({
                        'filename': filename,
                        'status': 'invalid_file'
                    })
            else:
                processed_files.append({
                    'filename': file.filename if file else 'unknown',
                    'status': 'invalid_format'
                })
        
        return jsonify(create_response(
            success=True,
            message=f"Processed {len(processed_files)} files with {total_chunks} total chunks",
            data={
                'processed_files': processed_files,
                'total_chunks': total_chunks
            }
        ))
        
    except Exception as e:
        logger.exception("Error occurred during ...")  # ini otomatis log traceback
        return jsonify(create_response(
            success=False,
            message="Error processing documents",
            data={'error': str(e)}
        )), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question to the chatbot"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify(create_response(
                success=False,
                message="Question is required"
            )), 400
        
        question = data['question'].strip()
        session_id = data.get('session_id', 'default')
        
        if not question:
            return jsonify(create_response(
                success=False,
                message="Question cannot be empty"
            )), 400
        
        # Process question
        result = chat_service.ask_question(question, session_id)
        
        return jsonify(create_response(
            success=True,
            message="Question processed successfully",
            data=result
        ))
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify(create_response(
            success=False,
            message="Error processing question",
            data={'error': str(e)}
        )), 500

@app.route('/conversation/history', methods=['GET'])
def get_conversation_history():
    """Get conversation history"""
    try:
        session_id = request.args.get('session_id', 'default')
        history = chat_service.get_conversation_history(session_id)
        
        return jsonify(create_response(
            success=True,
            message="Conversation history retrieved",
            data={'history': history, 'session_id': session_id}
        ))
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        return jsonify(create_response(
            success=False,
            message="Error getting conversation history",
            data={'error': str(e)}
        )), 500

@app.route('/conversation/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation memory"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')
        
        success = chat_service.clear_memory(session_id)
        
        if success:
            return jsonify(create_response(
                success=True,
                message="Conversation memory cleared",
                data={'session_id': session_id}
            ))
        else:
            return jsonify(create_response(
                success=False,
                message="Failed to clear conversation memory"
            )), 500
            
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        return jsonify(create_response(
            success=False,
            message="Error clearing conversation",
            data={'error': str(e)}
        )), 500

@app.route('/documents/stats', methods=['GET'])
def get_document_stats():
    """Get document statistics"""
    try:
        stats = chat_service.get_system_stats()
        collection_info = vector_store_manager.get_collection_info()
        
        return jsonify(create_response(
            success=True,
            message="Document statistics retrieved",
            data={
                'collection_info': collection_info,
                'system_stats': stats
            }
        ))
        
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        return jsonify(create_response(
            success=False,
            message="Error getting document statistics",
            data={'error': str(e)}
        )), 500

@app.route('/documents/search', methods=['POST'])
def search_documents():
    """Search documents by similarity"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify(create_response(
                success=False,
                message="Search query is required"
            )), 400
        
        query = data['query'].strip()
        k = data.get('k', config.TOP_K_RESULTS)
        
        if not query:
            return jsonify(create_response(
                success=False,
                message="Search query cannot be empty"
            )), 400
        
        # Search documents
        results = vector_store_manager.similarity_search_with_score(query, k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score)
            })
        
        return jsonify(create_response(
            success=True,
            message=f"Found {len(formatted_results)} similar documents",
            data={
                'results': formatted_results,
                'query': query,
                'total_results': len(formatted_results)
            }
        ))
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return jsonify(create_response(
            success=False,
            message="Error searching documents",
            data={'error': str(e)}
        )), 500

@app.route('/documents/reset', methods=['POST'])
def reset_documents():
    """Reset/clear all documents from vector store"""
    try:
        success = vector_store_manager.delete_collection()
        
        if success:
            return jsonify(create_response(
                success=True,
                message="All documents have been cleared from the vector store"
            ))
        else:
            return jsonify(create_response(
                success=False,
                message="Failed to clear documents"
            )), 500
            
    except Exception as e:
        logger.error(f"Error resetting documents: {str(e)}")
        return jsonify(create_response(
            success=False,
            message="Error resetting documents",
            data={'error': str(e)}
        )), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify(create_response(
        success=False,
        message="Endpoint not found"
    )), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify(create_response(
        success=False,
        message="Internal server error"
    )), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=config.FLASK_DEBUG, host='0.0.0.0', port=port)
    CORS(app)