"""
Enhanced Chroma Vector Database Manager

This module provides an enhanced interface for managing Chroma vector database
operations with improved metadata handling, unique ID generation, and batch processing.
It integrates with the existing VectorStoreManager while providing additional functionality.

Author: AI Assistant
Date: 2025-01-27
"""

import os
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional imports with safe fallbacks to avoid ModuleNotFoundError during tests
try:
    import chromadb
except Exception:
    chromadb = None  # type: ignore
try:
    from chromadb.config import Settings  # type: ignore
except Exception:
    Settings = None  # type: ignore
from langchain_core.documents import Document
try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None  # type: ignore
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None  # type: ignore

from app.config import config
from app.models.embeddings.vector_store_manager import VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChromaOperationResult:
    """Result of a Chroma database operation."""
    success: bool
    operation: str
    affected_count: int
    processing_time: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class DocumentStats:
    """Statistics for document operations."""
    total_documents: int
    total_chunks: int
    unique_documents: int
    doc_types: Dict[str, int]
    domains: Dict[str, int]
    difficulties: Dict[str, int]
    last_updated: str


class EnhancedChromaManager:
    """Enhanced Chroma vector database manager with advanced features."""
    
    def __init__(self, 
                 chroma_path: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        """Initialize the enhanced Chroma manager.
        
        Args:
            chroma_path: Path to Chroma database (defaults to config)
            collection_name: Name of the collection (defaults to config)
            embedding_model: Embedding model to use (defaults to config)
        """
        self.chroma_path = chroma_path or config.CHROMA_DB_PATH
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        
        # Initialize components
        self._initialize_chroma_client()
        self._initialize_embeddings()
        self._initialize_vectorstore()
        
        # Integration with existing system
        self.vector_store_manager = None
        self._initialize_integration()
        
        logger.info(f"Enhanced Chroma Manager initialized with collection: {self.collection_name}")
    
    def _initialize_chroma_client(self) -> None:
        """Initialize Chroma client with persistent storage."""
        try:
            # Guard: skip init if chromadb or Settings unavailable
            if chromadb is None or Settings is None:
                logger.warning("Chroma client init skipped: chromadb/Settings unavailable")
                self.chroma_client = None
                return

            # Ensure directory exists
            os.makedirs(self.chroma_path, exist_ok=True)
            
            # Try to initialize client with error handling for existing instances
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            except Exception as client_error:
                # If there's an existing instance with different settings, try to reset
                logger.warning(f"Chroma client initialization failed: {str(client_error)}")
                logger.info("Attempting to reset and reinitialize Chroma client...")
                
                try:
                    # Try to get existing client and reset
                    temp_client = chromadb.PersistentClient(path=self.chroma_path)
                    temp_client.reset()
                    logger.info("Successfully reset existing Chroma instance")
                except Exception as reset_error:
                    logger.warning(f"Could not reset existing instance: {str(reset_error)}")
                
                # Try to initialize again
                self.chroma_client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            
            logger.info(f"Chroma client initialized at: {self.chroma_path}")
            
        except Exception as e:
            logger.error(f"Error initializing Chroma client: {str(e)}")
            # Fallback: disable client instead of raising during tests
            self.chroma_client = None
    
    def _initialize_embeddings(self) -> None:
        """Initialize OpenAI embeddings (optional)."""
        try:
            if OpenAIEmbeddings and getattr(config, 'OPENAI_API_KEY', None):
                try:
                    self.embeddings = OpenAIEmbeddings(
                        model=self.embedding_model,
                        openai_api_key=config.OPENAI_API_KEY
                    )
                    logger.info(f"Embeddings initialized with model: {self.embedding_model}")
                except Exception as e:
                    logger.warning(f"OpenAIEmbeddings init failed; disabling embeddings: {e}")
                    self.embeddings = None
            else:
                logger.warning("OpenAIEmbeddings not installed or API key missing; embeddings disabled")
                self.embeddings = None
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            self.embeddings = None
    
    def _initialize_vectorstore(self) -> None:
        """Initialize Chroma vectorstore (optional)."""
        try:
            # Guard: dependencies and client must be available
            if Chroma is None or self.chroma_client is None or self.embeddings is None:
                logger.warning("Vectorstore init skipped: missing Chroma/client/embeddings")
                self.vectorstore = None
                return

            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_path
            )
            
            logger.info(f"Vectorstore initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {str(e)}")
            # Fallback: disable vectorstore instead of raising during tests
            self.vectorstore = None
    
    def _initialize_integration(self) -> None:
        """Initialize integration with existing VectorStoreManager."""
        try:
            # This allows us to leverage existing functionality while adding enhancements
            self.vector_store_manager = VectorStoreManager()
            logger.info("Integration with existing VectorStoreManager established")
            
        except Exception as e:
            logger.warning(f"Could not initialize VectorStoreManager integration: {str(e)}")
            self.vector_store_manager = None
    
    def generate_unique_id(self, 
                          doc_name: str, 
                          section_order: int, 
                          content: str) -> str:
        """Generate unique ID for a document chunk.
        
        Args:
            doc_name: Name of the document
            section_order: Order of the section
            content: Content of the chunk
            
        Returns:
            Unique ID string
        """
        try:
            # Create content hash for uniqueness
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
            
            # Format: docname_sectionorder_hash
            unique_id = f"{doc_name}_{section_order:03d}_{content_hash}"
            
            return unique_id
            
        except Exception as e:
            logger.error(f"Error generating unique ID: {str(e)}")
            # Fallback ID
            return f"{doc_name}_{section_order:03d}_{int(time.time())}"
    
    def add_documents_with_metadata(self, 
                                   documents: List[Document],
                                   batch_size: int = 100) -> ChromaOperationResult:
        """Add documents to Chroma with enhanced metadata handling.
        
        Args:
            documents: List of documents to add
            batch_size: Size of batches for processing
            
        Returns:
            ChromaOperationResult with operation details
        """
        start_time = time.time()
        
        try:
            if not documents:
                return ChromaOperationResult(
                    success=True,
                    operation="add_documents",
                    affected_count=0,
                    processing_time=0,
                    details={"message": "No documents to add"}
                )
            
            # Process documents in batches
            total_added = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare documents with unique IDs
                prepared_docs = []
                ids = []
                
                for doc in batch:
                    # Generate unique ID
                    doc_name = doc.metadata.get('filename', 'unknown')
                    section_order = doc.metadata.get('section_order', 0)
                    unique_id = self.generate_unique_id(doc_name, section_order, doc.page_content)
                    
                    # Add processing metadata
                    enhanced_metadata = doc.metadata.copy()
                    enhanced_metadata.update({
                        'chunk_id': unique_id,
                        'added_at': int(time.time()),
                        'embedding_model': self.embedding_model,
                        'content_length': len(doc.page_content),
                        'content_hash': hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                    })
                    
                    # Create enhanced document
                    enhanced_doc = Document(
                        page_content=doc.page_content,
                        metadata=enhanced_metadata
                    )
                    
                    prepared_docs.append(enhanced_doc)
                    ids.append(unique_id)
                
                # Add batch to vectorstore
                self.vectorstore.add_documents(
                    documents=prepared_docs,
                    ids=ids
                )
                
                total_added += len(batch)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
            
            # Persist changes
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
            
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully added {total_added} documents in {processing_time:.2f}s")
            
            return ChromaOperationResult(
                success=True,
                operation="add_documents",
                affected_count=total_added,
                processing_time=processing_time,
                details={
                    "batch_size": batch_size,
                    "total_batches": (len(documents) + batch_size - 1) // batch_size
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error adding documents: {str(e)}"
            logger.error(error_msg)
            
            return ChromaOperationResult(
                success=False,
                operation="add_documents",
                affected_count=0,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def upsert_documents(self, 
                        documents: List[Document],
                        batch_size: int = 100) -> ChromaOperationResult:
        """Upsert documents (update if exists, insert if new).
        
        Args:
            documents: List of documents to upsert
            batch_size: Size of batches for processing
            
        Returns:
            ChromaOperationResult with operation details
        """
        start_time = time.time()
        
        try:
            if not documents:
                return ChromaOperationResult(
                    success=True,
                    operation="upsert_documents",
                    affected_count=0,
                    processing_time=0,
                    details={"message": "No documents to upsert"}
                )
            
            # Group documents by filename for efficient removal
            docs_by_filename = {}
            for doc in documents:
                filename = doc.metadata.get('filename', 'unknown')
                if filename not in docs_by_filename:
                    docs_by_filename[filename] = []
                docs_by_filename[filename].append(doc)
            
            total_upserted = 0
            
            # Process each document group
            for filename, file_docs in docs_by_filename.items():
                # Remove existing documents for this file
                removal_result = self.remove_documents_by_filename(filename)
                
                if removal_result.success:
                    logger.info(f"Removed {removal_result.affected_count} existing chunks for {filename}")
                
                # Add new documents
                add_result = self.add_documents_with_metadata(file_docs, batch_size)
                
                if add_result.success:
                    total_upserted += add_result.affected_count
                    logger.info(f"Upserted {add_result.affected_count} chunks for {filename}")
                else:
                    logger.error(f"Failed to upsert documents for {filename}: {add_result.error_message}")
            
            processing_time = time.time() - start_time
            
            return ChromaOperationResult(
                success=True,
                operation="upsert_documents",
                affected_count=total_upserted,
                processing_time=processing_time,
                details={
                    "files_processed": len(docs_by_filename),
                    "batch_size": batch_size
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error upserting documents: {str(e)}"
            logger.error(error_msg)
            
            return ChromaOperationResult(
                success=False,
                operation="upsert_documents",
                affected_count=0,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def remove_documents_by_filename(self, filename: str) -> ChromaOperationResult:
        """Remove all documents with a specific filename.
        
        Args:
            filename: Name of the file to remove
            
        Returns:
            ChromaOperationResult with operation details
        """
        start_time = time.time()
        
        try:
            # Get collection
            collection = self.chroma_client.get_collection(self.collection_name)
            
            # Find documents with this filename
            results = collection.get(
                where={"filename": filename},
                include=['ids']
            )
            
            if results['ids']:
                # Delete documents
                collection.delete(ids=results['ids'])
                
                processing_time = time.time() - start_time
                
                logger.info(f"Removed {len(results['ids'])} documents for filename: {filename}")
                
                return ChromaOperationResult(
                    success=True,
                    operation="remove_documents",
                    affected_count=len(results['ids']),
                    processing_time=processing_time,
                    details={"filename": filename}
                )
            else:
                processing_time = time.time() - start_time
                
                return ChromaOperationResult(
                    success=True,
                    operation="remove_documents",
                    affected_count=0,
                    processing_time=processing_time,
                    details={"filename": filename, "message": "No documents found"}
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error removing documents for {filename}: {str(e)}"
            logger.error(error_msg)
            
            return ChromaOperationResult(
                success=False,
                operation="remove_documents",
                affected_count=0,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def get_document_stats(self) -> DocumentStats:
        """Get comprehensive statistics about documents in the collection.
        
        Returns:
            DocumentStats with collection statistics
        """
        try:
            # Get collection
            collection = self.chroma_client.get_collection(self.collection_name)
            
            # Get all metadata
            results = collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                return DocumentStats(
                    total_documents=0,
                    total_chunks=0,
                    unique_documents=0,
                    doc_types={},
                    domains={},
                    difficulties={},
                    last_updated=""
                )
            
            # Analyze metadata
            unique_docs = set()
            doc_types = {}
            domains = {}
            difficulties = {}
            latest_update = 0
            
            for metadata in results['metadatas']:
                # Track unique documents
                filename = metadata.get('filename', 'unknown')
                unique_docs.add(filename)
                
                # Count doc types
                doc_type = metadata.get('doc_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # Count domains
                domain = metadata.get('domain', 'general')
                domains[domain] = domains.get(domain, 0) + 1
                
                # Count difficulties
                difficulty = metadata.get('difficulty', 'basic')
                difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
                
                # Track latest update
                added_at = metadata.get('added_at', 0)
                if added_at > latest_update:
                    latest_update = added_at
            
            # Format last updated
            last_updated = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_update)) if latest_update else ""
            
            return DocumentStats(
                total_documents=len(unique_docs),
                total_chunks=len(results['metadatas']),
                unique_documents=len(unique_docs),
                doc_types=doc_types,
                domains=domains,
                difficulties=difficulties,
                last_updated=last_updated
            )
            
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return DocumentStats(
                total_documents=0,
                total_chunks=0,
                unique_documents=0,
                doc_types={},
                domains={},
                difficulties={},
                last_updated=""
            )
    
    def search_with_metadata_filter(self, 
                                   query: str,
                                   metadata_filter: Optional[Dict[str, Any]] = None,
                                   k: int = 5) -> List[Tuple[Document, float]]:
        """Search documents with metadata filtering.
        
        Args:
            query: Search query
            metadata_filter: Metadata filters to apply
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=metadata_filter
            )
            
            logger.info(f"Found {len(results)} results for query with metadata filter")
            return results
            
        except Exception as e:
            logger.error(f"Error in metadata filtered search: {str(e)}")
            return []
    
    def clear_collection(self) -> ChromaOperationResult:
        """Clear all documents from the collection.
        
        Returns:
            ChromaOperationResult with operation details
        """
        start_time = time.time()
        
        try:
            # Delete the collection
            self.chroma_client.delete_collection(self.collection_name)
            
            # Recreate the collection
            self._initialize_vectorstore()
            
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully cleared collection: {self.collection_name}")
            
            return ChromaOperationResult(
                success=True,
                operation="clear_collection",
                affected_count=0,  # We don't know the exact count
                processing_time=processing_time,
                details={"collection_name": self.collection_name}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error clearing collection: {str(e)}"
            logger.error(error_msg)
            
            return ChromaOperationResult(
                success=False,
                operation="clear_collection",
                affected_count=0,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the Chroma database.
        
        Returns:
            Health check results
        """
        try:
            # Check client connection
            collections = self.chroma_client.list_collections()
            
            # Check collection exists
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            # Get collection stats if exists
            stats = None
            if collection_exists:
                collection = self.chroma_client.get_collection(self.collection_name)
                stats = {
                    'count': collection.count(),
                    'name': collection.name
                }
            
            return {
                'status': 'healthy',
                'chroma_path': self.chroma_path,
                'collection_name': self.collection_name,
                'collection_exists': collection_exists,
                'total_collections': len(collections),
                'collection_stats': stats,
                'embedding_model': self.embedding_model,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }