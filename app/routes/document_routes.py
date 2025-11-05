"""Document processing routes for actuarial chatbot application.

This module contains endpoints for document upload, processing,
search, and management functionalities.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any, List, Optional, Union
import logging
import os
import tempfile
import traceback
import time
import uuid
from werkzeug.utils import secure_filename

from app.services.document_processor import DocumentProcessor
# Parallel processing removed - using direct document processing
# JSON parsing functionality removed - files will be processed directly
from app.models.embeddings import VectorStoreManager
from app.services.integrated_document_processor import IntegratedDocumentProcessor
from app.services.global_docs_preprocessor import GlobalDocsPreprocessor, ChunkStrategy
from app.utils.singleton_manager import get_or_create_service
from app.utils.helpers import (
    validate_files, 
    get_file_size,
    create_response
)


def _filter_metadata_for_response(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Filter metadata to only include the desired fields in the correct format.
    
    Args:
        metadata: Original metadata dictionary
        
    Returns:
        Filtered metadata with only the desired fields
    """
    # Define the desired fields
    desired_fields = [
        "doc_name", "doc_type", "domain", "scope", "section_heading", 
        "section_order", "keywords", "difficulty", "last_updated", 
        "version", "related_regulations", "numbers"
    ]
    
    filtered = {}
    
    for field in desired_fields:
        if field in metadata:
            value = metadata[field]
            
            # Handle special cases for formatting
            if field == "keywords" and isinstance(value, str):
                # Convert comma-separated string back to list
                try:
                    filtered[field] = [k.strip() for k in value.split(',') if k.strip()]
                except:
                    filtered[field] = []
            elif field == "related_regulations" and isinstance(value, str):
                # Convert comma-separated string back to list
                try:
                    filtered[field] = [r.strip() for r in value.split(',') if r.strip()]
                except:
                    filtered[field] = []
            elif field == "numbers" and isinstance(value, str):
                # Try to parse JSON string back to dict
                try:
                    import json
                    filtered[field] = json.loads(value)
                except:
                    filtered[field] = {}
            else:
                filtered[field] = value
        else:
            # Set default values for missing fields
            if field == "keywords":
                filtered[field] = []
            elif field == "related_regulations":
                filtered[field] = []
            elif field == "numbers":
                filtered[field] = {}
            elif field == "section_order":
                filtered[field] = 0
            elif field == "section_heading":
                filtered[field] = "Root"
            else:
                filtered[field] = None
    
    return filtered

logger = logging.getLogger(__name__)

# Create blueprint
document_bp = Blueprint('document', __name__)

# Global service instances (lazy initialization)
document_processor = None


def get_document_processor() -> DocumentProcessor:
    """Get or create document processor instance.
    
    Returns:
        DocumentProcessor instance
    """
    global document_processor
    if document_processor is None:
        document_processor = DocumentProcessor()
    return document_processor


# Parallel processor function removed - using direct document processing


# Route duplikat dihapus - menggunakan route auto-detection di bawah


@document_bp.route('/documents/search', methods=['POST'])
def search_documents() -> Dict[str, Any]:
    """Search documents in vector store.
    
    Returns:
        Dict containing search results
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return create_response(
                success=False,
                message="Query is required"
            ), 400
            
        query = data['query']
        k = data.get('k', 5)
        
        vector_store_manager = get_or_create_service(
            VectorStoreManager, 
            'vector_store_manager'
        )
        
        # Perform search with session_id and fallback to global
        session_id = data.get('session_id', 'global')
        results_with_scores = vector_store_manager.similarity_search_with_score(
            query=query,
            session_id=session_id,
            k=k,
            allow_fallback_to_global=True,
            session_required=False
        )
        
        # Format results with filtered metadata and scores
        formatted_results = []
        for doc, score in results_with_scores:
            # Filter metadata to only include the desired fields
            filtered_metadata = _filter_metadata_for_response(doc.metadata)
            formatted_results.append({
                "content": doc.page_content,
                "metadata": filtered_metadata,
                "score": score  # Include the actual similarity score
            })
        
        return create_response(
            success=True,
            message=f"Found {len(formatted_results)} results",
            data={
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        return create_response(
            success=False,
            message=f"Error searching documents: {str(e)}"
        ), 500


@document_bp.route('/documents/reset', methods=['POST'])
def reset_documents() -> Dict[str, Any]:
    """Reset document vector store.
    
    Returns:
        Dict containing reset operation result
    """
    try:
        vector_store_manager = get_or_create_service(
            VectorStoreManager, 
            'vector_store_manager'
        )
        
        # Reset vector store
        vector_store_manager.reset_vector_store()
        
        return create_response(
            success=True,
            message="Document store reset successfully",
            data={"status": "reset_complete"}
        )
        
    except Exception as e:
        logger.error(f"Error resetting documents: {str(e)}")
        return create_response(
            success=False,
            message=f"Error resetting documents: {str(e)}"
        ), 500


@document_bp.route('/upload-global-knowledge', methods=['POST'])
def upload_global_knowledge():
    """Upload and process global knowledge documents.
    
    This endpoint processes documents as general knowledge that will be
    chunked and embedded directly into the vector store without JSON parsing.
    Suitable for general documentation, guides, and reference materials.
    
    Returns:
        JSON response with processing results
    """
    try:
        # Validate request
        if 'files' not in request.files:
            return create_response(
                success=False,
                message="No files provided"
            ), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return create_response(
                success=False,
                message="No valid files selected"
            ), 400
        
        # Get session ID from request
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        
        logger.info(f"Processing {len(files)} global knowledge files (Session: {session_id})")
        
        # Initialize document processor for global knowledge
        document_processor = get_document_processor()
        
        results = []
        total_chunks = 0
        
        # Process each file as global knowledge
        for file in files:
            try:
                start_time = time.time()
                
                # Save file temporarily
                temp_path = os.path.join('/tmp', f"global_{session_id}_{file.filename}")
                file.save(temp_path)
                
                # Process document directly (no JSON parsing)
                documents = document_processor.process_document(
                    file_path=temp_path,
                    session_id=session_id
                )
                
                processing_time = time.time() - start_time
                chunks_created = len(documents) if documents else 0
                total_chunks += chunks_created
                
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'document_type': 'global_knowledge',
                    'chunks_created': chunks_created,
                    'processing_time': processing_time
                })
                
                logger.info(f"Processed {file.filename}: {chunks_created} chunks in {processing_time:.2f}s")
                
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                    
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e),
                    'document_type': 'global_knowledge'
                })
        
        # Calculate success statistics
        successful_files = sum(1 for r in results if r['success'])
        
        return create_response(
            success=True,
            message=f"Processed {successful_files}/{len(files)} global knowledge files successfully",
            data={
                'session_id': session_id,
                'total_files': len(files),
                'successful_files': successful_files,
                'total_chunks_created': total_chunks,
                'results': results
            }
        )
        
    except Exception as e:
        logger.error(f"Error in upload_global_knowledge: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return create_response(
            success=False,
            message=f"Error processing global knowledge: {str(e)}"
        ), 500


# PSAK documents upload endpoint removed - JSON parsing functionality deprecated


@document_bp.route('/input-docs', methods=['POST'])
def input_docs():
    """Upload and process documents with support for multiple file types.
    
    This endpoint processes uploaded documents including .md, .json, .txt, and other formats.
    All files are processed directly without JSON parsing - JSON files from backend team
    will be handled as regular documents.
    
    Returns:
        JSON response with processing results
    """
    try:
        # Validate request
        if 'files' not in request.files:
            return create_response(
                success=False,
                message="No files provided"
            ), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return create_response(
                success=False,
                message="No valid files selected"
            ), 400
        
        # Get session ID from request
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        
        logger.info(f"Processing {len(files)} documents (Session: {session_id})")
        
        # Initialize document processor
        document_processor = get_document_processor()
        
        results = []
        total_chunks = 0
        
        # Process all files with direct processing (supports .md, .json, .txt, etc.)
        for file in files:
            try:
                start_time = time.time()
                
                # Validate file extension
                allowed_extensions = {'.md', '.json', '.txt', '.csv', '.tsv'}
                file_ext = os.path.splitext(file.filename)[1].lower()
                
                if file_ext not in allowed_extensions:
                    logger.warning(f"Unsupported file type: {file.filename} ({file_ext})")
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': f'Unsupported file type: {file_ext}. Supported: {", ".join(allowed_extensions)}',
                        'document_type': 'unknown',
                        'processing_method': 'direct_processing'
                    })
                    continue
                
                # Save file temporarily
                temp_path = os.path.join('/tmp', f"doc_{session_id}_{file.filename}")
                file.save(temp_path)
                
                # Read file content for processing
                with open(temp_path, 'rb') as f:
                    file_content = f.read()
                
                # Process document directly to get chunks
                if file_ext == '.json':
                    # For JSON files, use the JSON processor directly
                    result = document_processor.process_document(
                        file_content=file_content,
                        filename=file.filename,
                        request_id=session_id
                    )
                    documents = result.get('documents', [])
                    chunks_created = len(documents)
                else:
                    # For markdown/text files, get the documents from processing
                    documents = document_processor.process_markdown_file(
                        file_path=temp_path,
                        session_id=session_id,
                        original_filename=file.filename
                    )
                    chunks_created = len(documents)
                    result = {
                        'success': True,
                        'chunks_created': chunks_created,
                        'documents': documents
                    }
                
                # Store documents in vector store if processing was successful
                stored_in_vectorstore = False
                if documents and result.get('success', False):
                    # Get vector store manager
                    vector_store_manager = get_or_create_service(
                        VectorStoreManager, 
                        'vector_store_manager'
                    )
                    
                    # Add documents to vector store using document manager
                    success = vector_store_manager.document_manager.add_documents(documents)
                    stored_in_vectorstore = success
                    
                    if success:
                        logger.info(f"Successfully stored {file.filename} with {chunks_created} chunks in vector store")
                    else:
                        logger.error(f"Failed to store {file.filename} in vector store")
                        result['success'] = False
                        result['error'] = "Failed to store in vector store"
                
                processing_time = time.time() - start_time
                total_chunks += chunks_created
                
                # Determine document type based on extension
                doc_type = 'json_document' if file_ext == '.json' else 'text_document'
                
                results.append({
                    'filename': file.filename,
                    'success': result.get('success', False),
                    'document_type': doc_type,
                    'file_extension': file_ext,
                    'processing_method': 'direct_processing',
                    'chunks_created': chunks_created,
                    'processing_time': processing_time,
                    'stored_in_vectorstore': stored_in_vectorstore,
                    'error': result.get('error') if not result.get('success', False) else None
                })
                
                if result.get('success', False):
                    logger.info(f"Processed {file.filename}: {chunks_created} chunks in {processing_time:.2f}s")
                else:
                    logger.error(f"Failed to process {file.filename}: {result.get('error', 'Unknown error')}")
                
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                    
            except Exception as e:
                logger.error(f"Error processing document {file.filename}: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e),
                    'document_type': 'unknown',
                    'processing_method': 'direct_processing'
                })
        
        # Calculate success statistics
        successful_files = sum(1 for r in results if r['success'])
        psak_successful = sum(1 for r in results if r['success'] and r['document_type'] == 'psak_document')
        global_successful = sum(1 for r in results if r['success'] and r['document_type'] == 'global_knowledge')
        json_successful = sum(1 for r in results if r['success'] and r['document_type'] == 'json_document')
        
        return create_response(
            success=True,
            message=f"Auto-processed {successful_files}/{len(files)} documents successfully",
            data={
                'session_id': session_id,
                'total_files': len(files),
                'successful_files': successful_files,
                'psak_files_processed': psak_successful,
                'global_files_processed': global_successful,
                'json_files_processed': json_successful,
                'total_chunks_created': total_chunks,
                'results': results
            }
        )
        
    except Exception as e:
        logger.error(f"Error in upload_auto: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return create_response(
            success=False,
            message=f"Error in auto-processing: {str(e)}"
        ), 500


def _is_psak_document(filename: str) -> bool:
    """Determine if a document is a PSAK document based on filename patterns.
    
    Args:
        filename: Name of the file to analyze
        
    Returns:
        True if the file appears to be a PSAK document, False otherwise
    """
    filename_lower = filename.lower()
    
    # PSAK document indicators
    psak_indicators = [
        'psak', 'psak219', 'psak_219', 'imbalan_kerja', 'employee_benefit',
        'actuarial', 'aktuaria', 'benefit', 'pension', 'laporan_aktuaria',
        'actuarial_report', 'valuation', 'valuasi', 'liability', 'liabilitas'
    ]
    
    # Check for PSAK indicators in filename
    for indicator in psak_indicators:
        if indicator in filename_lower:
            return True
    
    # Check file extensions that typically contain structured data
    structured_extensions = ['.json', '.xlsx', '.xls', '.csv']
    for ext in structured_extensions:
        if filename_lower.endswith(ext):
            return True
    
    return False


@document_bp.route('/inputglobaldocs', methods=['POST'])
def input_global_docs():
    """Process all documents from the global_docs directory with global session_id.
    
    This endpoint processes all markdown files from the sample_docs/global_docs directory
    and stores them with a global session_id for universal access across all sessions.
    Documents are stored in the same VectorStoreManager used for search functionality.
    
    Returns:
        JSON response with processing results
    """
    try:
        # Define the global_docs directory path
        global_docs_path = "/Users/yusril/Desktop/project/Trae chatbot-aktuaria/sample_docs/global_docs"
        
        # Check if directory exists
        if not os.path.exists(global_docs_path):
            return create_response(
                success=False,
                message=f"Global docs directory not found: {global_docs_path}"
            ), 404
        
        # Get all markdown files from the directory
        markdown_files = []
        for filename in os.listdir(global_docs_path):
            if filename.endswith('.md'):
                file_path = os.path.join(global_docs_path, filename)
                if os.path.isfile(file_path):
                    markdown_files.append(file_path)
        
        if not markdown_files:
            return create_response(
                success=False,
                message="No markdown files found in global_docs directory"
            ), 404
        
        # Use global session_id for universal access
        session_id = "global"
        
        logger.info(f"Processing {len(markdown_files)} global docs files with session_id: {session_id}")
        
        # Initialize document processor AND vector store manager
        document_processor = get_document_processor()
        vector_store_manager = get_or_create_service(
            VectorStoreManager, 
            'vector_store_manager'
        )
        
        results = []
        total_chunks = 0
        successful_files = 0
        
        # Process each markdown file
        for file_path in markdown_files:
            try:
                start_time = time.time()
                filename = os.path.basename(file_path)
                
                logger.info(f"Processing global doc: {filename}")
                
                # Process document using DocumentProcessor to get chunks
                documents = document_processor.process_markdown_file(
                    file_path=file_path,
                    session_id=session_id,
                    original_filename=filename
                )
                
                # Store documents in VectorStoreManager (same as search endpoint)
                if documents:
                    # Add documents to vector store using document manager
                    success = vector_store_manager.document_manager.add_documents(documents)
                    
                    if success:
                        chunks_created = len(documents)
                        total_chunks += chunks_created
                        successful_files += 1
                        
                        processing_time = time.time() - start_time
                        
                        results.append({
                            'filename': filename,
                            'file_path': file_path,
                            'success': True,
                            'document_type': 'global_knowledge',
                            'chunks_created': chunks_created,
                            'processing_time': round(processing_time, 2),
                            'session_id': session_id,
                            'stored_in_vectorstore': True
                        })
                        
                        logger.info(f"Successfully processed and stored {filename}: {chunks_created} chunks in {processing_time:.2f}s")
                    else:
                        processing_time = time.time() - start_time
                        results.append({
                            'filename': filename,
                            'file_path': file_path,
                            'success': False,
                            'error': "Failed to store documents in vector store",
                            'document_type': 'global_knowledge',
                            'session_id': session_id,
                            'processing_time': round(processing_time, 2),
                            'stored_in_vectorstore': False
                        })
                        
                        logger.error(f"Failed to store {filename} in vector store")
                else:
                    processing_time = time.time() - start_time
                    results.append({
                        'filename': filename,
                        'file_path': file_path,
                        'success': False,
                        'error': "No documents generated from processing",
                        'document_type': 'global_knowledge',
                        'session_id': session_id,
                        'processing_time': round(processing_time, 2),
                        'stored_in_vectorstore': False
                    })
                    
                    logger.error(f"No documents generated for {filename}")
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Error processing {filename}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                results.append({
                    'filename': os.path.basename(file_path),
                    'file_path': file_path,
                    'success': False,
                    'error': str(e),
                    'document_type': 'global_knowledge',
                    'session_id': session_id,
                    'processing_time': round(processing_time, 2),
                    'stored_in_vectorstore': False
                })
        
        # Calculate processing statistics
        total_files = len(markdown_files)
        failed_files = total_files - successful_files
        
        response_data = {
            'session_id': session_id,
            'global_docs_path': global_docs_path,
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'total_chunks_created': total_chunks,
            'processing_summary': {
                'success_rate': round((successful_files / total_files) * 100, 2) if total_files > 0 else 0,
                'average_chunks_per_file': round(total_chunks / successful_files, 2) if successful_files > 0 else 0
            },
            'results': results
        }
        
        if successful_files == total_files:
            message = f"Successfully processed all {total_files} global docs files"
            success = True
        elif successful_files > 0:
            message = f"Processed {successful_files}/{total_files} global docs files with {failed_files} failures"
            success = True
        else:
            message = f"Failed to process any of the {total_files} global docs files"
            success = False
        
        return create_response(
            success=success,
            message=message,
            data=response_data
        ), 200 if success else 500
        
    except Exception as e:
        logger.error(f"Error in input_global_docs: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return create_response(
            success=False,
            message="Internal server error while processing global docs",
            data={'error': str(e)}
        ), 500


def _preprocess_global_content(content: str, filename: str, chunk_strategy: str, max_chunk_size: int) -> tuple[str, dict]:
    """Advanced preprocessing for global documents with comprehensive analysis.
    
    Args:
        content: Raw document content
        filename: Original filename
        chunk_strategy: Chunking strategy to apply
        max_chunk_size: Maximum chunk size
        
    Returns:
        Tuple of (processed_content, metadata)
    """
    try:
        from datetime import datetime
        
        # Initialize the advanced preprocessor
        preprocessor = GlobalDocsPreprocessor({
            'max_file_size': 15 * 1024 * 1024,  # 15MB for global docs
            'min_content_length': 100,
            'preserve_code_blocks': True,
            'normalize_whitespace': True,
            'extract_links': True,
            'quality_threshold': 0.7
        })
        
        # Determine chunking strategy based on content
        strategy = ChunkStrategy.ADAPTIVE
        if 'tutorial' in filename.lower() or 'guide' in filename.lower():
            strategy = ChunkStrategy.SECTION_BASED
        elif 'reference' in filename.lower() or 'api' in filename.lower():
            strategy = ChunkStrategy.SEMANTIC
        
        # Process the document
        result = preprocessor.preprocess_document(
            content=content,
            filename=filename,
            chunk_strategy=strategy,
            max_chunk_size=max_chunk_size
        )
        
        if not result.success:
            logger.error(f"Preprocessing failed for {filename}: {result.error_message}")
            return content, {
                'original_filename': filename,
                'preprocessing_failed': True,
                'error': result.error_message,
                'processed_at': datetime.now().isoformat()
            }
        
        # Log preprocessing results
        logger.info(f"Successfully preprocessed {filename}: "
                   f"quality={result.quality_score:.2f}, "
                   f"sections={len(result.sections)}")
        
        # Enhanced metadata with preprocessing results
        metadata = {
            'original_filename': filename,
            'processed_at': datetime.now().isoformat(),
            'preprocessing_version': '2.0.0',
            'success': True,
            'quality_score': result.quality_score,
            'sections': [
                {
                    'title': section.title,
                    'level': section.level,
                    'word_count': section.word_count,
                    'line_range': [section.line_start, section.line_end]
                }
                for section in result.sections
            ],
            'preprocessing_stats': {
                'original_length': len(content),
                'processed_length': len(result.processed_content),
                'sections_found': len(result.sections),
                'quality_level': result.metadata.get('quality_metrics', {}).get('quality_level', 'unknown')
            },
            'chunking_config': {
                'strategy': chunk_strategy,
                'max_chunk_size': max_chunk_size,
                'actual_chunks': len(result.chunks),
                'estimated_chunks': max(1, len(result.processed_content) // max_chunk_size)
            },
            'chunks_info': [
                {
                    'chunk_id': chunk.chunk_id,
                    'section': chunk.section,
                    'order': chunk.order,
                    'token_count': chunk.token_count,
                    'char_count': len(chunk.content),
                    'has_front_matter': bool(chunk.front_matter),
                    'metadata_keys': list(chunk.metadata.keys())
                }
                for chunk in result.chunks
            ]
        }
        
        # Merge additional metadata from preprocessor
        if hasattr(result, 'metadata') and result.metadata:
            metadata.update(result.metadata)
        
        # Store chunks for later processing
        metadata['semantic_chunks'] = result.chunks
        
        return result.processed_content, metadata
        
    except Exception as e:
        error_msg = f"Error in advanced preprocessing for {filename}: {str(e)}"
        logger.error(error_msg)
        # Return original content with minimal metadata on error
        return content, {
            'original_filename': filename,
            'preprocessing_error': str(e),
            'processed_at': datetime.now().isoformat()
        }


@document_bp.route('/generate-global-docs-report', methods=['GET'])
def generate_global_docs_report():
    """Generate a comprehensive Markdown report of all global documents stored in the vector database.
    
    This endpoint creates a detailed MD file showing all global documents with their metadata,
    content previews, and statistics. The report is saved to the project root directory.
    
    Returns:
        JSON response with report generation results and file path
    """
    try:
        logger.info("Starting global docs report generation")
        
        # Initialize vector store manager
        vector_store_manager = get_or_create_service(
            VectorStoreManager, 
            'vector_store_manager'
        )
        
        # Get all global documents using session_id filter
        global_docs = vector_store_manager.list_documents_for_session(
            session_id="global", 
            include_global=True
        )
        
        # Filter only global documents (session_id = 'global')
        global_only_docs = [doc for doc in global_docs if doc.get('session_id') == 'global']
        
        logger.info(f"Found {len(global_only_docs)} global documents to include in report")
        
        # Generate report content
        report_content = _generate_markdown_report(global_only_docs)
        
        # Save report to file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"global_docs_report_{timestamp}.md"
        report_path = os.path.join(os.getcwd(), report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Global docs report saved to: {report_path}")
        
        return create_response(
            success=True,
            message=f"Global docs report generated successfully",
            data={
                'report_file': report_filename,
                'report_path': report_path,
                'total_global_docs': len(global_only_docs),
                'timestamp': timestamp
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating global docs report: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return create_response(
            success=False,
            message=f"Error generating global docs report: {str(e)}"
        ), 500


def _generate_markdown_report(global_docs: List[Dict[str, Any]]) -> str:
    """Generate comprehensive Markdown report content for global documents.
    
    Args:
        global_docs: List of global document dictionaries
        
    Returns:
        Formatted Markdown report content
    """
    from datetime import datetime
    
    # Report header
    report_lines = [
        "# Global Documents Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Global Documents:** {len(global_docs)}",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        f"- **Total Documents:** {len(global_docs)}",
        f"- **Session ID:** global",
        f"- **Storage Type:** Vector Database (ChromaDB)",
        "",
    ]
    
    # Group documents by filename for better organization
    docs_by_filename = {}
    for doc in global_docs:
        filename = doc.get('filename', 'Unknown')
        if filename not in docs_by_filename:
            docs_by_filename[filename] = []
        docs_by_filename[filename].append(doc)
    
    report_lines.extend([
        f"- **Unique Files:** {len(docs_by_filename)}",
        f"- **Total Chunks:** {len(global_docs)}",
        "",
        "---",
        "",
        "## Document Details",
        ""
    ])
    
    # Generate detailed report for each file
    for filename, docs in sorted(docs_by_filename.items()):
        report_lines.extend([
            f"### 📄 {filename}",
            "",
            f"**Chunks:** {len(docs)}",
            ""
        ])
        
        # Show metadata from first chunk (representative)
        if docs:
            first_doc = docs[0]
            metadata = first_doc.get('metadata', {})
            
            # Extract key metadata fields
            doc_type = metadata.get('doc_type', 'N/A')
            domain = metadata.get('domain', 'N/A')
            scope = metadata.get('scope', 'N/A')
            difficulty = metadata.get('difficulty', 'N/A')
            version = metadata.get('version', 'N/A')
            last_updated = metadata.get('last_updated', 'N/A')
            
            report_lines.extend([
                "**Metadata:**",
                f"- Document Type: `{doc_type}`",
                f"- Domain: `{domain}`",
                f"- Scope: `{scope}`",
                f"- Difficulty: `{difficulty}`",
                f"- Version: `{version}`",
                f"- Last Updated: `{last_updated}`",
                ""
            ])
            
            # Show keywords if available
            keywords = metadata.get('keywords', [])
            if keywords:
                if isinstance(keywords, str):
                    keywords = keywords.split(',')
                keywords_str = ', '.join([f"`{k.strip()}`" for k in keywords[:10]])  # Limit to 10 keywords
                report_lines.extend([
                    f"**Keywords:** {keywords_str}",
                    ""
                ])
            
            # Show related regulations if available
            related_regs = metadata.get('related_regulations', [])
            if related_regs:
                if isinstance(related_regs, str):
                    related_regs = related_regs.split(',')
                regs_str = ', '.join([f"`{r.strip()}`" for r in related_regs[:5]])  # Limit to 5 regulations
                report_lines.extend([
                    f"**Related Regulations:** {regs_str}",
                    ""
                ])
        
        # Show content previews for each chunk
        report_lines.extend([
            "**Content Chunks:**",
            ""
        ])
        
        for i, doc in enumerate(docs[:5], 1):  # Limit to first 5 chunks per file
            content_preview = doc.get('content_preview', 'No preview available')
            # Clean up content preview
            content_preview = content_preview.replace('\n', ' ').strip()
            if len(content_preview) > 200:
                content_preview = content_preview[:200] + "..."
            
            report_lines.extend([
                f"**Chunk {i}:**",
                f"```",
                content_preview,
                f"```",
                ""
            ])
        
        if len(docs) > 5:
            report_lines.extend([
                f"*... and {len(docs) - 5} more chunks*",
                ""
            ])
        
        report_lines.extend([
            "---",
            ""
        ])
    
    # Footer
    report_lines.extend([
        "",
        "## Technical Information",
        "",
        "- **Vector Store:** ChromaDB",
        "- **Session Filter:** `session_id = 'global'`",
        "- **Document Manager:** VectorStoreManager",
        "- **Generated by:** Trae AI Chatbot Aktuaria",
        "",
        "---",
        "",
        "*This report was automatically generated from the vector database.*"
    ])
    
    return '\n'.join(report_lines)