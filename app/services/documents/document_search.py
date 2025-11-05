"""Document search service.

This module provides search functionality across documents
using the PSAK219 document parser's search capabilities.
"""

import logging
from typing import Dict, List, Any, Optional

from .document_registry import DocumentRegistry
from .document_metadata import DocumentMetadata
from ..parser import SessionManager

logger = logging.getLogger(__name__)

class DocumentSearchService:
    """Service for searching across documents and their content.
    
    This service provides unified search functionality across all
    uploaded and parsed documents in the system.
    """
    
    def __init__(self, registry: DocumentRegistry, session_manager: SessionManager):
        """Initialize the document search service.
        
        Args:
            registry: Document registry for metadata management
            session_manager: Session manager for accessing parsed data
        """
        self.registry = registry
        self.session_manager = session_manager
        logger.info("DocumentSearchService initialized")
    
    def search_document(self, document_id: str, query: str, 
                       threshold: int = 70) -> List[Dict[str, Any]]:
        """Search within a specific document.
        
        Args:
            document_id: Document identifier
            query: Search query
            threshold: Similarity threshold for fuzzy matching
            
        Returns:
            List of search results
            
        Raises:
            ValueError: If document not found or not parsed
        """
        try:
            metadata = self.registry.get_document(document_id)
            if not metadata:
                raise ValueError(f"Document not found: {document_id}")
            
            if not metadata.is_parsed():
                raise ValueError(f"Document not parsed yet: {document_id}")
            
            # Switch to the document's session
            old_session = self.session_manager._last_parsed_document
            self.session_manager._last_parsed_document = metadata.session_key
            
            try:
                results = self.session_manager.search_value(query, threshold)
                logger.info(f"Found {len(results)} results in document {document_id} for query: '{query}'")
                return results
            finally:
                # Restore original session
                self.session_manager._last_parsed_document = old_session
        
        except Exception as e:
            logger.error(f"Error searching document {document_id}: {str(e)}")
            raise
    
    def search_all_documents(self, query: str, 
                           threshold: int = 70) -> Dict[str, List[Dict[str, Any]]]:
        """Search across all parsed documents.
        
        Args:
            query: Search query
            threshold: Similarity threshold for fuzzy matching
            
        Returns:
            Dictionary mapping document IDs to their search results
        """
        try:
            all_results = {}
            parsed_documents = self.registry.list_parsed_documents()
            
            logger.info(f"Searching {len(parsed_documents)} documents for query: '{query}'")
            
            for metadata in parsed_documents:
                try:
                    results = self.search_document(metadata.document_id, query, threshold)
                    if results:
                        all_results[metadata.document_id] = results
                except Exception as e:
                    logger.warning(f"Error searching document {metadata.document_id}: {str(e)}")
            
            logger.info(f"Found results in {len(all_results)} documents")
            return all_results
        
        except Exception as e:
            logger.error(f"Error searching all documents: {str(e)}")
            raise
    
    def search_by_document_type(self, document_type: str, query: str, 
                               threshold: int = 70) -> Dict[str, List[Dict[str, Any]]]:
        """Search within documents of a specific type.
        
        Args:
            document_type: Type of documents to search
            query: Search query
            threshold: Similarity threshold for fuzzy matching
            
        Returns:
            Dictionary mapping document IDs to their search results
        """
        try:
            all_results = {}
            documents = self.registry.get_documents_by_type(document_type)
            parsed_documents = [doc for doc in documents if doc.is_parsed()]
            
            logger.info(f"Searching {len(parsed_documents)} {document_type} documents for query: '{query}'")
            
            for metadata in parsed_documents:
                try:
                    results = self.search_document(metadata.document_id, query, threshold)
                    if results:
                        all_results[metadata.document_id] = results
                except Exception as e:
                    logger.warning(f"Error searching document {metadata.document_id}: {str(e)}")
            
            return all_results
        
        except Exception as e:
            logger.error(f"Error searching {document_type} documents: {str(e)}")
            raise
    
    def get_document_data(self, document_id: str, key: Optional[str] = None) -> Any:
        """Get specific data from a document.
        
        Args:
            document_id: Document identifier
            key: Specific data key to retrieve (optional)
            
        Returns:
            Document data or specific value if key provided
            
        Raises:
            ValueError: If document not found or not parsed
        """
        try:
            metadata = self.registry.get_document(document_id)
            if not metadata:
                raise ValueError(f"Document not found: {document_id}")
            
            if not metadata.is_parsed():
                raise ValueError(f"Document not parsed yet: {document_id}")
            
            # Switch to the document's session
            old_session = self.session_manager._last_parsed_document
            self.session_manager._last_parsed_document = metadata.session_key
            
            try:
                if key:
                    # Get specific value
                    result = self.session_manager.get_session_variable(key)
                else:
                    # Get all session data
                    session_data = self.session_manager._session_data.get(
                        metadata.session_key, {}
                    )
                    result = session_data.get('parsed_data', {})
                
                return result
            finally:
                # Restore original session
                self.session_manager._last_parsed_document = old_session
        
        except Exception as e:
            logger.error(f"Error getting document data for {document_id}: {str(e)}")
            raise
    
    def search_psak219_by_session(self, session_id: str, query: str, 
                                 threshold: int = 70) -> List[Dict[str, Any]]:
        """Search PSAK219 documents by session ID.
        
        Args:
            session_id: Session identifier
            query: Search query
            threshold: Similarity threshold
            
        Returns:
            List of search results
        """
        try:
            # Find documents associated with this session
            results = []
            for metadata in self.registry.list_parsed_documents():
                if (metadata.document_type == "PSAK219" and 
                    metadata.session_key and 
                    session_id in metadata.session_key):
                    
                    try:
                        doc_results = self.search_document(
                            metadata.document_id, query, threshold
                        )
                        results.extend(doc_results)
                    except Exception as e:
                        logger.warning(f"Error searching document {metadata.document_id}: {str(e)}")
            
            logger.info(f"Found {len(results)} PSAK219 results for session {session_id}")
            return results
        
        except Exception as e:
            logger.error(f"Error searching PSAK219 by session {session_id}: {str(e)}")
            return []
    
    def get_search_suggestions(self, partial_query: str, 
                              limit: int = 10) -> List[str]:
        """Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested search terms
        """
        try:
            suggestions = set()
            
            # Common PSAK219 search terms
            common_terms = [
                "discount rate", "salary increase", "retirement age", 
                "current service cost", "present value", "actuarial assumptions",
                "employee data", "company information", "valuation period",
                "mortality table", "turnover rate", "disability rate"
            ]
            
            # Filter common terms that match partial query
            partial_lower = partial_query.lower()
            for term in common_terms:
                if partial_lower in term.lower():
                    suggestions.add(term)
            
            # Limit results
            return list(suggestions)[:limit]
        
        except Exception as e:
            logger.error(f"Error getting search suggestions: {str(e)}")
            return []