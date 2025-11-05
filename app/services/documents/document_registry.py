"""Document registry management.

This module handles the persistent storage and management of document metadata
for the actuarial chatbot system.
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .document_metadata import DocumentMetadata

logger = logging.getLogger(__name__)

class DocumentRegistry:
    """Manages the registry of uploaded and parsed documents.
    
    This class handles persistent storage of document metadata,
    allowing the system to track all uploaded documents across sessions.
    """
    
    def __init__(self, upload_directory: Path):
        """Initialize the document registry.
        
        Args:
            upload_directory: Directory where documents and registry are stored
        """
        self.upload_directory = upload_directory
        self.registry_file = upload_directory / "document_registry.json"
        self.documents: Dict[str, DocumentMetadata] = {}
        
        # Ensure upload directory exists
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"DocumentRegistry initialized with {len(self.documents)} documents")
    
    def _load_registry(self) -> None:
        """Load document registry from persistent storage."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                
                for doc_id, doc_data in registry_data.items():
                    self.documents[doc_id] = DocumentMetadata.from_dict(doc_data)
                
                logger.info(f"Loaded {len(self.documents)} documents from registry")
            else:
                logger.info("No existing document registry found")
        
        except Exception as e:
            logger.error(f"Error loading document registry: {str(e)}")
            # Initialize empty registry on error
            self.documents = {}
    
    def _save_registry(self) -> None:
        """Save document registry to persistent storage."""
        try:
            registry_data = {}
            for doc_id, metadata in self.documents.items():
                registry_data[doc_id] = metadata.to_dict()
            
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved registry with {len(self.documents)} documents")
        
        except Exception as e:
            logger.error(f"Error saving document registry: {str(e)}")
            raise
    
    def add_document(self, metadata: DocumentMetadata) -> None:
        """Add a document to the registry.
        
        Args:
            metadata: Document metadata to add
        """
        self.documents[metadata.document_id] = metadata
        self._save_registry()
        logger.info(f"Added document to registry: {metadata.document_id}")
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document metadata if found, None otherwise
        """
        return self.documents.get(document_id)
    
    def update_document(self, document_id: str, metadata: DocumentMetadata) -> None:
        """Update document metadata.
        
        Args:
            document_id: Document identifier
            metadata: Updated metadata
        """
        if document_id in self.documents:
            self.documents[document_id] = metadata
            self._save_registry()
            logger.info(f"Updated document metadata: {document_id}")
        else:
            raise ValueError(f"Document not found: {document_id}")
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the registry.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if document was removed, False if not found
        """
        if document_id in self.documents:
            metadata = self.documents[document_id]
            
            # Delete physical file if it exists
            try:
                file_path = Path(metadata.file_path)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not delete file {metadata.file_path}: {str(e)}")
            
            # Remove from registry
            del self.documents[document_id]
            self._save_registry()
            logger.info(f"Removed document from registry: {document_id}")
            return True
        
        return False
    
    def list_documents(self) -> List[DocumentMetadata]:
        """Get list of all documents in registry.
        
        Returns:
            List of all document metadata
        """
        return list(self.documents.values())
    
    def list_parsed_documents(self) -> List[DocumentMetadata]:
        """Get list of successfully parsed documents.
        
        Returns:
            List of parsed document metadata
        """
        return [doc for doc in self.documents.values() if doc.is_parsed()]
    
    def get_document_count(self) -> int:
        """Get total number of documents in registry.
        
        Returns:
            Number of documents
        """
        return len(self.documents)
    
    def get_parsed_document_count(self) -> int:
        """Get number of successfully parsed documents.
        
        Returns:
            Number of parsed documents
        """
        return len(self.list_parsed_documents())
    
    def document_exists(self, document_id: str) -> bool:
        """Check if a document exists in the registry.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if document exists
        """
        return document_id in self.documents
    
    def get_documents_by_type(self, document_type: str) -> List[DocumentMetadata]:
        """Get documents by type.
        
        Args:
            document_type: Type of documents to retrieve
            
        Returns:
            List of documents of specified type
        """
        return [doc for doc in self.documents.values() 
                if doc.document_type == document_type]
    
    def get_registry_stats(self) -> Dict[str, int]:
        """Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        total = len(self.documents)
        parsed = self.get_parsed_document_count()
        failed = len([doc for doc in self.documents.values() 
                     if doc.parsing_status == "failed"])
        pending = len([doc for doc in self.documents.values() 
                      if doc.parsing_status == "pending"])
        
        return {
            'total_documents': total,
            'parsed_documents': parsed,
            'failed_documents': failed,
            'pending_documents': pending
        }