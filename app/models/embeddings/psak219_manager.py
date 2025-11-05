"""PSAK219 Manager for handling PSAK219-specific operations.

This module manages PSAK219 document search and metadata operations
for the vector store.
"""

from langchain_core.documents import Document
from typing import Dict, List, Optional, Tuple, Any
import logging

from app.config import config

logger = logging.getLogger(__name__)

class PSAK219Manager:
    """Manages PSAK219-specific operations for the vector store."""
    
    def __init__(self, vector_store_manager):
        """Initialize PSAK219Manager with reference to VectorStoreManager.
        
        Args:
            vector_store_manager: Reference to the main VectorStoreManager instance
        """
        self.vector_store_manager = vector_store_manager
    
    def search_psak219_documents(
        self, 
        session_id: str = None, 
        company_name: str = None, 
        period: str = None,
        allow_fallback_to_global: bool = True
    ) -> List[Tuple[Document, float]]:
        """Search for PSAK219 documents with optional filters and session-based fallback.
        
        Args:
            session_id: Session ID for filtering
            company_name: Company name for filtering
            period: Period for filtering
            allow_fallback_to_global: Whether to fallback to global documents if no session docs found
            
        Returns:
            List of (Document, score) tuples for PSAK219 documents
        """
        try:
            results = []
            
            # First, try session-specific PSAK219 documents if session_id provided
            if session_id:
                logger.info(f"Searching session-specific PSAK219 documents for session {session_id}")
                
                # Build filter for session-specific PSAK219 documents
                # Use multiple filter approaches to ensure we find PSAK219 documents
                session_filter = {'session_id': session_id}
                
                # Add company filter if provided
                if company_name:
                    session_filter['psak219_company'] = company_name
                
                # Add period filter if provided
                if period:
                    session_filter['psak219_period'] = period
                
                # Search session-specific documents with larger retrieval count
                session_results = self.vector_store_manager.vectorstore.similarity_search_with_score(
                    query="PSAK219",  # Use PSAK219 query to improve relevance
                    k=2000,    # Increased number to get all matching docs
                    filter=session_filter
                )
                
                logger.info(f"Raw session search returned {len(session_results)} documents")
                
                # Filter results to include PSAK219 documents using multiple criteria
                psak219_session_results = []
                for doc, score in session_results:
                    metadata = doc.metadata
                    
                    # Check multiple PSAK219 indicators
                    is_psak219 = (
                        metadata.get('psak219_session_key') is not None or
                        metadata.get('is_psak219_document') == True or
                        metadata.get('has_parsed_json') == True or
                        metadata.get('document_type') == 'PSAK219' or
                        'psak219' in metadata.get('filename', '').lower()
                    )
                    
                    if is_psak219:
                        psak219_session_results.append((doc, score))
                        logger.debug(f"Found PSAK219 doc: {metadata.get('filename')} with session_id: {metadata.get('session_id')}")
                
                logger.info(f"Found {len(psak219_session_results)} session-specific PSAK219 documents")
                results.extend(psak219_session_results)
            
            # If no session results and fallback is allowed, search global PSAK219 documents
            if not results and allow_fallback_to_global:
                logger.info("No session-specific PSAK219 documents found, searching global documents")
                
                # Build filter for global PSAK219 documents
                global_filter = {'session_id': 'global'}  # Fixed: use session_id='global' instead of is_global=True
                
                # Add company filter if provided
                if company_name:
                    global_filter['psak219_company'] = company_name
                
                # Add period filter if provided
                if period:
                    global_filter['psak219_period'] = period
                
                # Search global documents with larger retrieval count
                global_results = self.vector_store_manager.vectorstore.similarity_search_with_score(
                    query="PSAK219",  # Use PSAK219 query to improve relevance
                    k=2000,    # Increased number to get all matching docs
                    filter=global_filter
                )
                
                logger.info(f"Raw global search returned {len(global_results)} documents")
                
                # Filter results to include PSAK219 documents using multiple criteria
                psak219_global_results = []
                for doc, score in global_results:
                    metadata = doc.metadata
                    
                    # Check multiple PSAK219 indicators
                    is_psak219 = (
                        metadata.get('psak219_session_key') is not None or
                        metadata.get('is_psak219_document') == True or
                        metadata.get('has_parsed_json') == True or
                        metadata.get('document_type') == 'PSAK219' or
                        'psak219' in metadata.get('filename', '').lower()
                    )
                    
                    if is_psak219:
                        psak219_global_results.append((doc, score))
                        logger.debug(f"Found global PSAK219 doc: {metadata.get('filename')}")
                
                logger.info(f"Found {len(psak219_global_results)} global PSAK219 documents")
                results.extend(psak219_global_results)
            
            logger.info(f"Total PSAK219 documents found: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching PSAK219 documents: {str(e)}")
            return []
    
    def get_psak219_metadata_summary(self, session_id: str = None, allow_fallback_to_global: bool = True) -> Dict[str, Any]:
        """Get summary of PSAK219 metadata available in the vector store.
        
        Args:
            session_id: Optional session ID for filtering
            allow_fallback_to_global: Whether to fallback to global documents if no session docs found
            
        Returns:
            Dictionary containing summary of available PSAK219 metadata
        """
        try:
            # Get all PSAK219 documents with session filtering and fallback
            psak219_docs = self.search_psak219_documents(
                session_id=session_id, 
                allow_fallback_to_global=allow_fallback_to_global
            )
            
            # Initialize summary structure
            summary = {
                'total_documents': len(psak219_docs),
                'companies': set(),
                'periods': set(),
                'sections_available': set(),
                'documents': []
            }
            
            # Process each document to extract metadata
            for doc, score in psak219_docs:
                meta = doc.metadata
                
                # Collect unique values
                if meta.get('psak219_company'):
                    summary['companies'].add(meta['psak219_company'])
                if meta.get('psak219_period'):
                    summary['periods'].add(meta['psak219_period'])
                if meta.get('psak219_sections'):
                    sections = meta['psak219_sections'].split(',') if isinstance(meta['psak219_sections'], str) else []
                    summary['sections_available'].update(sections)
                
                # Document info
                summary['documents'].append({
                    'filename': meta.get('filename'),
                    'company': meta.get('psak219_company'),
                    'period': meta.get('psak219_period'),
                    'session_key': meta.get('psak219_session_key'),
                    'score': score
                })
            
            # Convert sets to lists for JSON serialization
            summary['companies'] = list(summary['companies'])
            summary['periods'] = list(summary['periods'])
            summary['sections_available'] = list(summary['sections_available'])
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting PSAK219 metadata summary: {str(e)}")
            return {'error': str(e)}
    
    def get_psak219_companies(self, session_id: str = None, allow_fallback_to_global: bool = True) -> List[str]:
        """Get list of unique companies in PSAK219 documents.
        
        Args:
            session_id: Optional session ID for filtering
            allow_fallback_to_global: Whether to fallback to global documents if no session docs found
            
        Returns:
            List of unique company names
        """
        try:
            summary = self.get_psak219_metadata_summary(session_id, allow_fallback_to_global)
            return summary.get('companies', [])
        except Exception as e:
            logger.error(f"Error getting PSAK219 companies: {str(e)}")
            return []
    
    def get_psak219_periods(self, session_id: str = None, allow_fallback_to_global: bool = True) -> List[str]:
        """Get list of unique periods in PSAK219 documents.
        
        Args:
            session_id: Optional session ID for filtering
            allow_fallback_to_global: Whether to fallback to global documents if no session docs found
            
        Returns:
            List of unique periods
        """
        try:
            summary = self.get_psak219_metadata_summary(session_id, allow_fallback_to_global)
            return summary.get('periods', [])
        except Exception as e:
            logger.error(f"Error getting PSAK219 periods: {str(e)}")
            return []
    
    def get_psak219_sections(self, session_id: str = None, allow_fallback_to_global: bool = True) -> List[str]:
        """Get list of available sections in PSAK219 documents.
        
        Args:
            session_id: Optional session ID for filtering
            allow_fallback_to_global: Whether to fallback to global documents if no session docs found
            
        Returns:
            List of available sections
        """
        try:
            summary = self.get_psak219_metadata_summary(session_id, allow_fallback_to_global)
            return summary.get('sections_available', [])
        except Exception as e:
            logger.error(f"Error getting PSAK219 sections: {str(e)}")
            return []
    
    def search_psak219_by_company_and_period(
        self, 
        company_name: str, 
        period: str, 
        session_id: str = None,
        allow_fallback_to_global: bool = True
    ) -> List[Tuple[Document, float]]:
        """Search PSAK219 documents by specific company and period.
        
        Args:
            company_name: Company name to search for
            period: Period to search for
            session_id: Optional session ID for filtering
            allow_fallback_to_global: Whether to fallback to global documents if no session docs found
            
        Returns:
            List of (Document, score) tuples matching the criteria
        """
        try:
            return self.search_psak219_documents(
                session_id=session_id,
                company_name=company_name,
                period=period,
                allow_fallback_to_global=allow_fallback_to_global
            )
        except Exception as e:
            logger.error(f"Error searching PSAK219 by company and period: {str(e)}")
            return []
    
    def get_psak219_document_count(self, session_id: str = None, allow_fallback_to_global: bool = True) -> int:
        """Get total count of PSAK219 documents.
        
        Args:
            session_id: Optional session ID for filtering
            allow_fallback_to_global: Whether to fallback to global documents if no session docs found
            
        Returns:
            Total count of PSAK219 documents
        """
        try:
            psak219_docs = self.search_psak219_documents(
                session_id=session_id, 
                allow_fallback_to_global=allow_fallback_to_global
            )
            return len(psak219_docs)
        except Exception as e:
            logger.error(f"Error getting PSAK219 document count: {str(e)}")
            return 0