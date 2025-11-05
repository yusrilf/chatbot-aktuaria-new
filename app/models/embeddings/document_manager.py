"""Document Manager for handling document operations.

This module manages document addition, removal, and listing operations
for the vector store.
"""

import traceback
from langchain_core.documents import Document
from typing import Dict, List, Optional, Any
import logging
import os

from app.config import config

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manages document operations for the vector store."""
    
    def __init__(self, vector_store_manager):
        """Initialize DocumentManager with reference to VectorStoreManager.
        
        Args:
            vector_store_manager: Reference to the main VectorStoreManager instance
        """
        self.vector_store_manager = vector_store_manager
    
    def add_global_documents_from_folder(self, folder_path: str) -> None:
        """Add all documents from a folder as global documents.
        
        Args:
            folder_path: Path to the folder containing documents
        """
        try:
            if not os.path.exists(folder_path):
                logger.warning(f"Folder {folder_path} does not exist")
                return
            
            processed_count = 0
            from app.services.document_processor import DocumentProcessor
            dp = DocumentProcessor()
            session_id = 'global'
            
            for filename in os.listdir(folder_path):
                # Process both .txt and .md files for global documents
                if filename.endswith(('.txt', '.md')):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        # Chunk the file content using DocumentProcessor before upsert
                        documents = dp.process_markdown_file(
                            file_path=file_path,
                            session_id=session_id,
                            original_filename=filename
                        )
                        
                        if documents:
                            # Add chunked documents (with Pinecone-safe metadata slimming inside add_documents)
                            self.add_documents(documents)
                            processed_count += 1
                            logger.info(f"Added global document (chunked): {filename} with {len(documents)} chunks")
                        else:
                            logger.warning(f"No chunks generated for {filename}, skipping")
                    except Exception as e:
                        logger.error(f"Error processing file {filename}: {str(e)}")
            
            logger.info(f"Added {processed_count} global documents from {folder_path}")
        except Exception as e:
            logger.error(f"Error adding global documents from folder: {str(e)}")
    
    def add_global_document(self, doc_name: str, doc_content: str) -> None:
        """Add a single global document.
        
        Args:
            doc_name: Name of the document
            doc_content: Content of the document
        """
        try:
            if not self._document_exists(doc_name):
                doc = Document(
                    page_content=doc_content,
                    metadata={
                        'filename': doc_name, 
                        'is_global': True,
                        'session_id': 'global'  # Ensure global documents have session_id='global'
                    }
                )
                self.vector_store_manager.vectorstore.add_documents([doc])
                logger.info(f"Added global document: {doc_name}")
            else:
                logger.info(f"Document {doc_name} already exists, skipping")
        except Exception as e:
            logger.error(f"Error adding global document {doc_name}: {str(e)}")
    
    def _document_exists(self, doc_name: str) -> bool:
        """Check if a document already exists in the vector store.
        
        Args:
            doc_name: Name of the document to check
            
        Returns:
            bool: True if document exists, False otherwise
        """
        try:
            # Prefer direct collection query when Chroma is available
            if getattr(self.vector_store_manager, 'chroma_client', None):
                try:
                    collection = self.vector_store_manager.chroma_client.get_collection(config.COLLECTION_NAME)
                    docs = collection.get(where={'filename': doc_name}, limit=config.DOCUMENT_LISTING_K)
                    count = len(docs.get('ids', []))
                    logger.debug(f"Document existence check via collection.get for '{doc_name}': {count} matches")
                    return count > 0
                except Exception as ce:
                    logger.warning(f"Collection.get failed for existence check, falling back: {ce}")
            # Fallback to similarity_search with a non-empty placeholder query
            results = self.vector_store_manager.vectorstore.similarity_search(
                query=" ",  # avoid empty embedding edge-case
                k=config.DOCUMENT_LISTING_K,  # Use configurable K for document listing
                filter={'filename': doc_name}
            )
            return len(results) > 0
        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            return False
    
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
            # Prefer direct collection query when Chroma is available
            if getattr(self.vector_store_manager, 'chroma_client', None):
                try:
                    collection = self.vector_store_manager.chroma_client.get_collection(config.COLLECTION_NAME)
                    # Session-specific documents
                    sess = collection.get(where={'session_id': session_id}, limit=config.DOCUMENT_LISTING_K)
                    logger.debug(f"Session '{session_id}' listing via collection.get returned {len(sess.get('ids', []))} docs")
                    for content, meta in zip(sess.get('documents', []), sess.get('metadatas', [])):
                        preview = content[:200] + '...' if len(content) > 200 else content
                        documents.append({
                            'filename': meta.get('filename', 'Unknown'),
                            'session_id': meta.get('session_id'),
                            'is_global': meta.get('is_global', False),
                            'content_preview': preview,
                            'metadata': meta
                        })
                    # Global documents
                    if include_global:
                        glob = collection.get(where={'session_id': 'global'}, limit=config.DOCUMENT_LISTING_K)
                        logger.debug(f"Global listing via collection.get returned {len(glob.get('ids', []))} docs")
                        for content, meta in zip(glob.get('documents', []), glob.get('metadatas', [])):
                            preview = content[:200] + '...' if len(content) > 200 else content
                            documents.append({
                                'filename': meta.get('filename', 'Unknown'),
                                'session_id': 'global',
                                'is_global': True,
                                'content_preview': preview,
                                'metadata': meta
                            })
                    return documents
                except Exception as ce:
                    logger.warning(f"Collection.get failed for listing, falling back: {ce}")
            # Fallback to similarity_search with a non-empty placeholder query
            session_results = self.vector_store_manager.vectorstore.similarity_search(
                query=" ",
                k=config.DOCUMENT_LISTING_K,  # Use configurable K for document listing
                filter={'session_id': session_id}
            )
            for doc in session_results:
                documents.append({
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'session_id': doc.metadata.get('session_id'),
                    'is_global': doc.metadata.get('is_global', False),
                    'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata
                })
            if include_global:
                global_results = self.vector_store_manager.vectorstore.similarity_search(
                    query=" ",
                    k=config.DOCUMENT_LISTING_K,  # Use configurable K for document listing
                    filter={'session_id': 'global'}  # Fixed: use session_id='global' instead of is_global=True
                )
                for doc in global_results:
                    documents.append({
                        'filename': doc.metadata.get('filename', 'Unknown'),
                        'session_id': 'global',  # Fixed: set session_id to 'global'
                        'is_global': True,
                        'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                        'metadata': doc.metadata
                    })
            return documents
        except Exception as e:
            logger.error(f"Error listing documents for session {session_id}: {str(e)}")
            return []
    
    def _slim_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Slim metadata to comply with Pinecone's 40KB per-vector limit.
        
        Keeps essential fields and truncates long strings/lists to avoid oversize payloads.
        """
        # Essential keys to retain
        allowed_keys = {
            'filename', 'session_id', 'is_global', 'document_type', 'doc_type',
            'domain', 'scope', 'version', 'last_updated',
            'chunk_id', 'section_heading', 'section_order', 'header_preview', 'chunk_size',
            'source'
        }
        slim: Dict[str, Any] = {}
        for key, value in metadata.items():
            if key not in allowed_keys:
                continue
            # Normalize and cap sizes
            if isinstance(value, str):
                # Cap string length to 2048 chars
                slim[key] = value[:2048]
            elif isinstance(value, (int, float, bool)):
                slim[key] = value
            elif isinstance(value, (list, tuple)):
                # Keep first 20 items, join into comma-separated string capped to 2048 chars
                joined = ', '.join(str(item) for item in value[:20])
                slim[key] = joined[:2048]
            else:
                # Convert other types to string capped to 1024 chars
                try:
                    s = str(value)
                    slim[key] = s[:1024]
                except Exception:
                    # Skip values that cannot be stringified
                    pass
        return slim
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return False
            
            # Pinecone-specific metadata slimming to avoid oversize errors
            backend = str(getattr(config, 'VECTOR_BACKEND', 'chroma')).lower()
            docs_to_add: List[Document] = []
            if backend == 'pinecone':
                for doc in documents:
                    meta = doc.metadata or {}
                    slim_meta = self._slim_metadata_for_pinecone(meta)
                    docs_to_add.append(Document(page_content=doc.page_content, metadata=slim_meta))
            else:
                docs_to_add = documents
            
            # Add documents to vectorstore
            self.vector_store_manager.vectorstore.add_documents(docs_to_add)
            
            # Rebuild hybrid index with new documents
            self._rebuild_hybrid_index()
            
            logger.info(f"Successfully added {len(docs_to_add)} documents")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Error adding documents: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Handle collection ID mismatch error
            if "collection" in error_msg and "does not exist" in error_msg:
                logger.warning("Collection ID mismatch detected, attempting to reinitialize vectorstore...")
                
                # Try to reinitialize vectorstore
                if hasattr(self.vector_store_manager, 'reinitialize_vectorstore'):
                    if self.vector_store_manager.reinitialize_vectorstore():
                        logger.info("Vectorstore reinitialized, retrying document addition...")
                        try:
                            # Retry adding documents
                            self.vector_store_manager.vectorstore.add_documents(docs_to_add)
                            self._rebuild_hybrid_index()
                            logger.info(f"Successfully added {len(docs_to_add)} documents after reinitializing")
                            return True
                        except Exception as retry_e:
                            logger.error(f"Failed to add documents even after reinitializing: {str(retry_e)}")
                            return False
                    else:
                        logger.error("Failed to reinitialize vectorstore")
                        return False
                else:
                    logger.error("Reinitialize method not available")
                    return False
            
            return False
    
    def _rebuild_hybrid_index(self) -> None:
        """Rebuild the hybrid search index after adding new documents.
        
        Note: This method is kept for backward compatibility but does nothing
        since we've switched to pure vector search mode.
        """
        try:
            # Check if hybrid search manager is available
            if (hasattr(self.vector_store_manager, 'hybrid_search_manager') and 
                self.vector_store_manager.hybrid_search_manager is not None):
                # Get all documents for re-indexing
                all_docs = self.vector_store_manager._get_all_documents_from_vectorstore()
                if all_docs:
                    self.vector_store_manager.hybrid_search_manager.build_index(all_docs)
                    logger.info(f"Hybrid index rebuilt with {len(all_docs)} documents")
            else:
                logger.debug("Hybrid search manager not available - using pure vector search mode")
        except Exception as e:
            logger.error(f"Error rebuilding hybrid index: {str(e)}")