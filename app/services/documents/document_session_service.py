"""Refactored Document Session Service using modular components.

This module provides the main document session service functionality
using smaller, modular components for better maintainability.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import uuid

# Import the document parser
from app.services.parser import (
    PSAK219DocumentParser,
    SessionManager as ParserSessionManager,
    get_session_manager,
    get_session_variable,
    search_value,
    parse_psak219_document
)
# Enhanced PSAK219 parser removed - using direct document processing

# Import modular components
from .document_metadata import DocumentMetadata
from .document_registry import DocumentRegistry
from .document_search import DocumentSearchService
from .document_export import DocumentExportService

logger = logging.getLogger(__name__)

class DocumentSessionService:
    """Main document session service using modular components.
    
    This service integrates the PSAK219 document parser with the existing
    chatbot system, providing session management and document retrieval
    capabilities for enhanced RAG performance.
    """
    
    def __init__(self, upload_directory: str = "data/uploaded_documents"):
        """Initialize the document session service.
        
        Args:
            upload_directory: Directory for storing uploaded documents
        """
        self.upload_directory = Path(upload_directory)
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.session_manager = get_session_manager()
        # Enhanced PSAK219 parser removed - using direct document processing
        self.parser = None
        
        # Initialize modular components
        self.registry = DocumentRegistry(self.upload_directory)
        self.search_service = DocumentSearchService(self.registry, self.session_manager)
        self.export_service = DocumentExportService(
            self.registry, self.search_service, self.session_manager
        )
        
        logger.info(f"DocumentSessionService initialized with {self.registry.get_document_count()} documents")
    
    @property
    def document_registry(self):
        """Provide access to the document registry for backward compatibility."""
        return self.registry
    
    def upload_document(self, file_content: bytes, filename: str, 
                      document_type: str = "PSAK219") -> str:
        """Upload a new document to the system.
        
        Args:
            file_content: Binary content of the file
            filename: Original filename
            document_type: Type of document
            
        Returns:
            Document ID of the uploaded document
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_count = self.registry.get_document_count()
            safe_filename = f"doc_{timestamp}_{doc_count}.md"
            file_path = self.upload_directory / safe_filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Create metadata
            metadata = DocumentMetadata.create_new(
                filename, str(file_path), document_type
            )
            
            # Add to registry
            self.registry.add_document(metadata)
            
            logger.info(f"Document uploaded: {metadata.document_id} ({filename})")
            return metadata.document_id
        
        except Exception as e:
            logger.error(f"Error uploading document {filename}: {str(e)}")
            raise
    
    def parse_document(self, document_id: str) -> Dict[str, Any]:
        """Parse an uploaded document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with parsing results
        """
        try:
            metadata = self.registry.get_document(document_id)
            if not metadata:
                raise ValueError(f"Document not found: {document_id}")
            
            # Read file content
            with open(metadata.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse document
            session_key = f"doc_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            parsing_result = self.parser.parse_document(content, session_key)
            
            if parsing_result.get('success', False):
                # Update metadata
                metadata.mark_parsing_completed(session_key)
                self.registry.update_document(document_id, metadata)
                
                logger.info(f"Document parsed successfully: {document_id}")
                return {
                    'success': True,
                    'document_id': document_id,
                    'session_key': session_key,
                    'message': 'Document parsed successfully'
                }
            else:
                # Mark as failed
                error_msg = parsing_result.get('error', 'Unknown parsing error')
                metadata.mark_parsing_failed(error_msg)
                self.registry.update_document(document_id, metadata)
                
                return {
                    'success': False,
                    'document_id': document_id,
                    'error': error_msg
                }
        
        except Exception as e:
            logger.error(f"Error parsing document {document_id}: {str(e)}")
            # Update metadata with error
            metadata = self.registry.get_document(document_id)
            if metadata:
                metadata.mark_parsing_failed(str(e))
                self.registry.update_document(document_id, metadata)
            raise
    
    # Delegate methods to modular components
    def get_document_data(self, document_id: str, key: Optional[str] = None) -> Any:
        """Get data from a specific document."""
        return self.search_service.get_document_data(document_id, key)
    
    def search_document_data(self, document_id: str, query: str, 
                           threshold: int = 70) -> List[Dict[str, Any]]:
        """Search data within a specific document."""
        return self.search_service.search_document(document_id, query, threshold)
    
    def search_all_documents(self, query: str, 
                           threshold: int = 70) -> Dict[str, List[Dict[str, Any]]]:
        """Search across all parsed documents."""
        return self.search_service.search_all_documents(query, threshold)
    
    def get_document_list(self, include_global: bool = False) -> List[Dict[str, Any]]:
        """Get list of uploaded documents.
        
        Args:
            include_global: Whether to include global documents in the list
            
        Returns:
            List of document information dictionaries
        """
        all_docs = [doc.get_info_dict() for doc in self.registry.list_documents()]
        
        if include_global:
            return all_docs
        else:
            # Filter out global documents (documents without session_id or with is_global=True)
            return [
                doc for doc in all_docs 
                if not doc.get('is_global', False) and doc.get('session_id')
            ]
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its data."""
        try:
            metadata = self.registry.get_document(document_id)
            if not metadata:
                return False
            
            # Remove from session manager if parsed
            if metadata.session_key and metadata.session_key in self.session_manager._session_data:
                del self.session_manager._session_data[metadata.session_key]
            
            # Remove from registry (this also deletes the file)
            return self.registry.remove_document(document_id)
        
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise
    
    def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Get a summary of key information from a document."""
        return self.export_service.export_document_summary(document_id)
    
    def export_document_data(self, document_id: str, 
                           output_path: Union[str, Path]) -> None:
        """Export all data from a document to a file."""
        self.export_service.export_document_data(document_id, output_path)
    
    # Session and system management methods
    def get_psak219_data_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get PSAK219 data associated with a session."""
        return self.search_service.search_psak219_by_session(session_id, "", threshold=0)
    
    def search_psak219_by_session(self, session_id: str, query: str, 
                                 threshold: int = 70) -> List[Dict[str, Any]]:
        """Search PSAK219 documents by session ID."""
        return self.search_service.search_psak219_by_session(session_id, query, threshold)
    
    def get_session_variable(self, key: str) -> Any:
        """Get a session variable from the global session manager."""
        try:
            return self.session_manager.get_session_variable(key)
        except Exception as e:
            logger.error(f"Error getting session variable {key}: {str(e)}")
            return None
    
    def set_session_variable(self, key: str, value: Any) -> None:
        """Set a session variable in the global session manager."""
        try:
            self.session_manager.set_session_variable(key, value)
        except Exception as e:
            logger.error(f"Error setting session variable {key}: {str(e)}")
            raise
    
    def get_sources_info(self, session_id: str) -> List[Dict[str, Any]]:
        """Get information about available sources for a session."""
        try:
            sources = []
            for metadata in self.registry.list_parsed_documents():
                if metadata.document_type == "PSAK219":
                    sources.append({
                        'document_id': metadata.document_id,
                        'filename': metadata.original_filename,
                        'type': metadata.document_type,
                        'status': metadata.parsing_status
                    })
            return sources
        except Exception as e:
            logger.error(f"Error getting sources info: {str(e)}")
            return []
    
    def list_documents_for_session(self, session_id: str, include_global: bool = True) -> List[Dict[str, Any]]:
        """List all documents for a specific session.
        
        Args:
            session_id: Session ID to filter documents
            include_global: Whether to include global documents
            
        Returns:
            List of document information dictionaries
        """
        try:
            documents = []
            
            # Get session-specific documents
            for metadata in self.registry.list_documents():
                if metadata.session_key and session_id in metadata.session_key:
                    documents.append({
                        'filename': metadata.original_filename,
                        'session_id': session_id,
                        'is_global': False,
                        'content_preview': f"Document: {metadata.original_filename}",
                        'metadata': {
                            'document_id': metadata.document_id,
                            'document_type': metadata.document_type,
                            'parsing_status': metadata.parsing_status
                        }
                    })
            
            # Add global documents if requested
            if include_global:
                for metadata in self.registry.list_documents():
                    if not metadata.session_key:  # Global documents don't have session keys
                        documents.append({
                            'filename': metadata.original_filename,
                            'session_id': 'global',
                            'is_global': True,
                            'content_preview': f"Global document: {metadata.original_filename}",
                            'metadata': {
                                'document_id': metadata.document_id,
                                'document_type': metadata.document_type,
                                'parsing_status': metadata.parsing_status
                            }
                        })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents for session {session_id}: {str(e)}")
            return []

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            return {
                'registry_stats': self.registry.get_registry_stats(),
                'session_stats': {
                    'active_sessions': len(self.session_manager._session_data)
                },
                'upload_directory': str(self.upload_directory)
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {'error': str(e)}

# Global service instance
_global_document_service = None

def get_document_service() -> DocumentSessionService:
    """Get or create the global document service instance."""
    global _global_document_service
    if _global_document_service is None:
        _global_document_service = DocumentSessionService()
    return _global_document_service

def upload_and_parse_document(file_content: bytes, filename: str) -> str:
    """Convenience function to upload and parse a document."""
    service = get_document_service()
    document_id = service.upload_document(file_content, filename)
    service.parse_document(document_id)
    return document_id

def query_document_data(query: str, document_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to query document data."""
    service = get_document_service()
    if document_id:
        results = service.search_document_data(document_id, query)
        return {'document_id': document_id, 'results': results}
    else:
        results = service.search_all_documents(query)
        return {'all_documents': results}