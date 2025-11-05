"""Document metadata management.

This module defines the DocumentMetadata dataclass and related utilities
for managing document information in the actuarial chatbot system.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for uploaded documents.
    
    Attributes:
        document_id: Unique identifier for the document
        original_filename: Original name of the uploaded file
        file_path: Path where the document is stored
        document_type: Type of document (e.g., 'PSAK219')
        upload_timestamp: ISO timestamp when document was uploaded
        parsing_status: Status of document parsing ('pending', 'completed', 'failed')
        session_key: Key for accessing parsed data in session manager
        error_message: Error message if parsing failed
    """
    document_id: str
    original_filename: str
    file_path: str
    document_type: str
    upload_timestamp: str
    parsing_status: str
    session_key: Optional[str] = None
    error_message: Optional[str] = None
    
    @classmethod
    def create_new(cls, filename: str, file_path: str, 
                   document_type: str = "PSAK219") -> 'DocumentMetadata':
        """Create new document metadata with generated ID and timestamp.
        
        Args:
            filename: Original filename
            file_path: Path where document is stored
            document_type: Type of document
            
        Returns:
            New DocumentMetadata instance
        """
        document_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        logger.info(f"Creating new document metadata: {document_id} for {filename}")
        
        return cls(
            document_id=document_id,
            original_filename=filename,
            file_path=file_path,
            document_type=document_type,
            upload_timestamp=timestamp,
            parsing_status="pending"
        )
    
    def mark_parsing_completed(self, session_key: str) -> None:
        """Mark document parsing as completed.
        
        Args:
            session_key: Key for accessing parsed data
        """
        self.parsing_status = "completed"
        self.session_key = session_key
        self.error_message = None
        logger.info(f"Document {self.document_id} parsing completed")
    
    def mark_parsing_failed(self, error_message: str) -> None:
        """Mark document parsing as failed.
        
        Args:
            error_message: Error message describing the failure
        """
        self.parsing_status = "failed"
        self.error_message = error_message
        self.session_key = None
        logger.error(f"Document {self.document_id} parsing failed: {error_message}")
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary for serialization.
        
        Returns:
            Dictionary representation of metadata
        """
        return {
            'document_id': self.document_id,
            'original_filename': self.original_filename,
            'file_path': self.file_path,
            'document_type': self.document_type,
            'upload_timestamp': self.upload_timestamp,
            'parsing_status': self.parsing_status,
            'session_key': self.session_key,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DocumentMetadata':
        """Create DocumentMetadata from dictionary.
        
        Args:
            data: Dictionary containing metadata
            
        Returns:
            DocumentMetadata instance
        """
        return cls(**data)
    
    def is_parsed(self) -> bool:
        """Check if document has been successfully parsed.
        
        Returns:
            True if document is parsed and ready for use
        """
        return self.parsing_status == "completed" and self.session_key is not None
    
    def get_info_dict(self) -> dict:
        """Get basic document information for API responses.
        
        Returns:
            Dictionary with basic document information
        """
        return {
            'document_id': self.document_id,
            'original_filename': self.original_filename,
            'document_type': self.document_type,
            'upload_timestamp': self.upload_timestamp,
            'parsing_status': self.parsing_status,
            'error_message': self.error_message
        }