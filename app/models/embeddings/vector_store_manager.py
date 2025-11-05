"""Vector Store Manager for ChromaDB and embeddings management.

This module handles the core vector store initialization and management.
"""

import traceback
# Chroma imports moved to runtime branch to avoid import-time errors
# Chroma imports moved to runtime branch to avoid import-time errors
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None
# Chroma imports moved to runtime branch to avoid import-time errors
from langchain_core.documents import Document
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
try:
    import cohere
except Exception:
    cohere = None
import sqlite3
import shutil
from pathlib import Path

from app.config import config
from app.models.hybrid_search import HybridSearchManager, SearchResult
from app.utils.singleton_manager import SingletonMeta, lazy_initialization, get_service_manager
from .document_manager import DocumentManager
from .search_manager import SearchManager
from .psak219_manager import PSAK219Manager

logger = logging.getLogger(__name__)

class VectorStoreManager(metaclass=SingletonMeta):
    """Singleton VectorStore Manager to prevent duplicate initialization."""
    
    def __init__(self):
        """Initialize VectorStoreManager singleton."""
        # Only initialize once per singleton instance
        if hasattr(self, '_initialized'):
            return
            
        logger.info("Initializing VectorStoreManager singleton")
        
        # Initialize embeddings and clients (guarded)
        self.embeddings = None
        try:
            if OpenAIEmbeddings is not None and getattr(config, 'OPENAI_API_KEY', None):
                self.embeddings = OpenAIEmbeddings(
                    model=getattr(config, 'EMBEDDING_MODEL', 'text-embedding-3-small'),
                    openai_api_key=config.OPENAI_API_KEY
                )
            else:
                logger.warning("OpenAIEmbeddings unavailable or API key missing; embeddings disabled.")
        except Exception as e:
            logger.warning(f"OpenAIEmbeddings initialization failed: {str(e)}")
            self.embeddings = None
        
        self.cohere_client = None
        try:
            if getattr(config, 'COHERE_API_KEY', None):
                self.cohere_client = cohere.Client(config.COHERE_API_KEY)
            else:
                logger.warning("Cohere API key missing; reranking disabled.")
        except Exception as e:
            logger.warning(f"Cohere client init failed: {str(e)}")
            self.cohere_client = None
        
        self.chroma_client = None
        self.vectorstore = None
        
        # Hybrid Search Manager disabled - using pure vector search
        self.hybrid_search_manager = None
        
        # Initialize core components with error handling
        self._initialize_vectorstore_with_fallback()
        
        # Initialize component managers only if vectorstore is available
        if self.vectorstore is not None:
            self.document_manager = DocumentManager(self)
            self.search_manager = SearchManager(self)
            self.psak219_manager = PSAK219Manager(self)
        else:
            # Initialize with None to prevent attribute errors
            self.document_manager = None
            self.search_manager = None
            self.psak219_manager = None
            logger.warning("Component managers not initialized due to vectorstore failure")
        
        # Register with service manager
        get_service_manager().register_service('vector_store_manager', self)
        
        # Mark as initialized
        self._initialized = True
        
        logger.info("VectorStoreManager singleton initialized successfully")

    def ensure_full_initialization(self) -> None:
        """Ensure all components are fully initialized (call when needed)."""
        try:
            if self.vectorstore is None:
                logger.warning("Cannot ensure full initialization: vectorstore is None")
                return
                
            if not hasattr(self, '_global_docs_initialized'):
                self._initialize_global_documents()
            if not hasattr(self, '_hybrid_index_initialized'):
                self._initialize_hybrid_index()
        except Exception as e:
            logger.error(f"Error during full initialization: {str(e)}")

    def _initialize_vectorstore_with_fallback(self) -> None:
        """Initialize ChromaDB vector store with fallback mechanisms."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self._initialize_vectorstore()
                logger.info("Vector store initialized successfully")
                return
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e).lower()
                
                logger.error(f"Vector store initialization attempt {retry_count} failed: {str(e)}")
                
                # Handle specific ChromaDB schema errors
                if "no such column" in error_msg or "sqlite3.operationalerror" in error_msg:
                    logger.warning("Detected ChromaDB schema incompatibility, attempting database reset...")
                    if self._reset_chroma_database():
                        logger.info("Database reset successful, retrying initialization...")
                        continue
                
                # Handle permission errors
                elif "permission denied" in error_msg or "read-only" in error_msg:
                    logger.warning("Detected permission issues, attempting to fix...")
                    if self._fix_permissions():
                        logger.info("Permission fix attempted, retrying initialization...")
                        continue
                
                # Handle corrupted database
                elif "database is locked" in error_msg or "database disk image is malformed" in error_msg:
                    logger.warning("Detected database corruption, attempting recovery...")
                    if self._recover_database():
                        logger.info("Database recovery attempted, retrying initialization...")
                        continue
                
                if retry_count >= max_retries:
                    logger.error(f"Failed to initialize vector store after {max_retries} attempts")
                    self._initialize_fallback_mode()
                    break

    def _initialize_vectorstore(self) -> None:
        """Initialize vector store based on configured backend."""
        try:
            # Require embeddings; otherwise trigger fallback mode
            if self.embeddings is None:
                raise RuntimeError("Embeddings unavailable")
            
            if str(getattr(config, 'VECTOR_BACKEND', 'chroma')).lower() == 'pinecone':
                # Initialize Pinecone backend
                from pinecone import Pinecone, ServerlessSpec
                from langchain_pinecone import PineconeVectorStore

                pc = Pinecone(api_key=config.PINECONE_API_KEY)
                existing = {i.name for i in pc.list_indexes()}

                if config.PINECONE_INDEX_NAME not in existing:
                    pc.create_index(
                        name=config.PINECONE_INDEX_NAME,
                        dimension=config.PINECONE_DIMENSION,
                        metric=config.PINECONE_METRIC,
                        spec=ServerlessSpec(
                            cloud=config.PINECONE_CLOUD,
                            region=config.PINECONE_REGION
                        )
                    )

                # LangChain Pinecone vectorstore
                self.vectorstore = PineconeVectorStore(
                    index_name=config.PINECONE_INDEX_NAME,
                    namespace=config.PINECONE_NAMESPACE,
                    embedding=self.embeddings,
                    pinecone_api_key=config.PINECONE_API_KEY,
                )

                # Test Pinecone connectivity
                self._test_pinecone_connection()
            else:
                # Initialize ChromaDB backend
                import chromadb
                from chromadb.config import Settings
                from langchain_chroma import Chroma
                os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)

                self.chroma_client = chromadb.PersistentClient(
                    path=config.CHROMA_DB_PATH,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )

                self.vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name=config.COLLECTION_NAME,
                    embedding_function=self.embeddings
                )

                # Test the connection (Chroma)
                self._test_vectorstore_connection()
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _test_pinecone_connection(self) -> None:
        """Test Pinecone vectorstore connection."""
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            index = pc.Index(config.PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            total = stats.get('total_vector_count', None)
            logger.info(f"Pinecone index '{config.PINECONE_INDEX_NAME}' reachable, vectors: {total}")
        except Exception as e:
            logger.error(f"Pinecone connection test failed: {str(e)}")
            raise

    def _test_vectorstore_connection(self) -> None:
        """Test vectorstore connection to ensure it's working properly."""
        try:
            # Try to get collection info
            if self.chroma_client:
                collections = self.chroma_client.list_collections()
                logger.info(f"Available collections: {[c.name for c in collections]}")
                
            # Try a simple operation
            if self.vectorstore:
                # This will create the collection if it doesn't exist
                collection_info = self.get_collection_info()
                logger.info(f"Collection test successful: {collection_info}")
                
        except Exception as e:
            logger.error(f"Vectorstore connection test failed: {str(e)}")
            raise

    def _reset_chroma_database(self) -> bool:
        """Reset ChromaDB database to fix schema issues."""
        try:
            db_path = Path(config.CHROMA_DB_PATH)
            
            if db_path.exists():
                logger.info(f"Removing existing ChromaDB at {db_path}")
                shutil.rmtree(db_path)
                
            # Recreate directory
            os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
            logger.info("ChromaDB database reset completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB database: {str(e)}")
            return False

    def _fix_permissions(self) -> bool:
        """Fix permission issues with ChromaDB directory."""
        try:
            db_path = Path(config.CHROMA_DB_PATH)
            
            # Create directory if it doesn't exist
            os.makedirs(db_path, exist_ok=True)
            
            # Try to fix permissions (Unix-like systems)
            if hasattr(os, 'chmod'):
                os.chmod(db_path, 0o755)
                
            # Try to create a test file
            test_file = db_path / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            
            logger.info("Permission fix completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fix permissions: {str(e)}")
            return False

    def _recover_database(self) -> bool:
        """Attempt to recover corrupted database."""
        try:
            db_path = Path(config.CHROMA_DB_PATH)
            
            # Look for SQLite database files
            sqlite_files = list(db_path.glob("*.sqlite*"))
            
            for sqlite_file in sqlite_files:
                try:
                    # Try to open and check the database
                    conn = sqlite3.connect(str(sqlite_file))
                    conn.execute("PRAGMA integrity_check;")
                    conn.close()
                    logger.info(f"Database {sqlite_file} passed integrity check")
                    
                except sqlite3.Error as e:
                    logger.warning(f"Database {sqlite_file} failed integrity check: {str(e)}")
                    # Remove corrupted file
                    sqlite_file.unlink()
                    logger.info(f"Removed corrupted database file: {sqlite_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to recover database: {str(e)}")
            return False

    def _initialize_fallback_mode(self) -> None:
        """Initialize in fallback mode when ChromaDB fails."""
        logger.warning("Initializing in fallback mode - ChromaDB unavailable")
        
        self.chroma_client = None
        self.vectorstore = None
        
        # Create mock objects to prevent crashes
        class MockVectorStore:
            def similarity_search(self, *args, **kwargs):
                logger.warning("VectorStore unavailable - returning empty results")
                return []
                
            def add_documents(self, *args, **kwargs):
                logger.warning("VectorStore unavailable - cannot add documents")
                return []
        
        self.vectorstore = MockVectorStore()

    @lazy_initialization
    def _initialize_global_documents(self) -> None:
        """Lazy initialization of global documents from specified folder."""
        try:
            if self.document_manager is None:
                logger.warning("Cannot initialize global documents: document_manager is None")
                return
                
            folder_path = 'sample_docs/global_docs'  # Path folder dokumen global
            # Panggil fungsi untuk menambahkan semua dokumen global
            self.document_manager.add_global_documents_from_folder(folder_path)
            logger.info("Global documents initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing global documents: {str(e)}")

    def _initialize_hybrid_index(self) -> None:
        """Deprecated: Hybrid search index initialization disabled.
        
        This method is kept for backward compatibility but does nothing
        since we've switched to pure vector search.
        """
        logger.info("Hybrid search index initialization skipped (pure vector search mode)")

    def _get_all_documents_from_vectorstore(self) -> List[Document]:
        """Mengambil semua dokumen dari vectorstore untuk indexing."""
        try:
            # Get all documents without filtering
            results = self.vectorstore.similarity_search(
                query="",  # Empty query to get all docs
                k=100     # Reduced number for faster indexing
            )
            return results
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            return []
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from vectorstore for hybrid indexing.
        
        Returns:
            List[Document]: List of all documents in the vectorstore
        """
        try:
            logger.info("Retrieving all documents for hybrid indexing...")
            documents = self._get_all_documents_from_vectorstore()
            logger.info(f"Retrieved {len(documents)} documents for hybrid indexing")
            return documents
        except Exception as e:
            logger.error(f"Error in get_all_documents: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection or Pinecone index."""
        try:
            if str(getattr(config, 'VECTOR_BACKEND', 'chroma')).lower() == 'pinecone':
                from pinecone import Pinecone
                pc = Pinecone(api_key=config.PINECONE_API_KEY)
                index = pc.Index(config.PINECONE_INDEX_NAME)
                stats = index.describe_index_stats()
                return {
                    'name': config.PINECONE_INDEX_NAME,
                    'count': stats.get('total_vector_count', 0)
                }
            else:
                collection = self.chroma_client.get_collection(config.COLLECTION_NAME)
                return {
                    'name': collection.name,
                    'count': collection.count()
                }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {'error': str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the current collection."""
        try:
            self.chroma_client.delete_collection(config.COLLECTION_NAME)
            logger.info(f"Collection {config.COLLECTION_NAME} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def reinitialize_vectorstore(self) -> bool:
        """Reinitialize the vectorstore to fix collection ID mismatch issues.
        
        Returns:
            bool: True if reinitialization was successful, False otherwise
        """
        try:
            logger.info("Reinitializing vectorstore to fix collection ID mismatch...")
            
            # Close existing connections
            if hasattr(self, 'vectorstore') and self.vectorstore:
                self.vectorstore = None
            
            if hasattr(self, 'chroma_client') and self.chroma_client:
                self.chroma_client = None
            
            # Reinitialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=config.CHROMA_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            # Reinitialize Langchain Chroma vectorstore
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=config.COLLECTION_NAME,
                embedding_function=self.embeddings
            )
            
            # Test the connection
            self._test_vectorstore_connection()
            
            logger.info("Vectorstore reinitialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error reinitializing vectorstore: {str(e)}")
            return False

    def hybrid_similarity_search_with_score(
        self,
        query: str,
        session_id: str = None,
        k: int = None,
        filter: Optional[Dict] = None,
        session_required: bool = True,
        allow_fallback_to_global: bool = False,
        return_placeholder_on_empty: bool = True,
        use_hybrid: bool = True
    ) -> List[Tuple[Document, float]]:
        """Hybrid similarity search delegated to search manager.
        
        Args:
            query: Query pencarian
            session_id: ID sesi
            k: Jumlah hasil yang diinginkan
            filter: Filter tambahan
            session_required: Apakah session_id wajib
            allow_fallback_to_global: Fallback ke dokumen global
            return_placeholder_on_empty: Return placeholder jika kosong
            use_hybrid: Gunakan hybrid search (True) atau semantic saja (False)
            
        Returns:
            List tuple (Document, combined_score)
        """
        return self.search_manager.hybrid_similarity_search_with_score(
            query=query,
            session_id=session_id,
            k=k,
            filter=filter,
            session_required=session_required,
            allow_fallback_to_global=allow_fallback_to_global,
            return_placeholder_on_empty=return_placeholder_on_empty,
            use_hybrid=use_hybrid
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        session_id: str = None,
        k: int = None,
        filter: Optional[Dict] = None,
        session_required: bool = True,
        allow_fallback_to_global: bool = False,
        return_placeholder_on_empty: bool = True
    ) -> List[Tuple[Document, float]]:
        """Similarity search with scores delegated to search manager.
        
        Args:
            query: Search query
            session_id: Session ID for filtering
            k: Number of results to return
            filter: Additional filters
            session_required: Whether session_id is required
            allow_fallback_to_global: Whether to fallback to global documents
            return_placeholder_on_empty: Whether to return placeholder on empty results
            
        Returns:
            List of (Document, score) tuples
        """
        return self.search_manager.similarity_search_with_score(
            query=query,
            session_id=session_id,
            k=k,
            filter=filter,
            session_required=session_required,
            allow_fallback_to_global=allow_fallback_to_global,
            return_placeholder_on_empty=return_placeholder_on_empty
        )
    
    def search_psak219_documents(
        self,
        session_id: str,
        company_name: Optional[str] = None,
        period: Optional[str] = None,
        allow_fallback_to_global: bool = True
    ) -> List[Tuple[Document, float]]:
        """Search PSAK219 documents delegated to PSAK219 manager.
        
        Args:
            session_id: Session ID for filtering
            company_name: Optional company name filter
            period: Optional period filter
            allow_fallback_to_global: Whether to fallback to global documents
            
        Returns:
            List of (Document, score) tuples
        """
        return self.psak219_manager.search_psak219_documents(
            session_id=session_id,
            company_name=company_name,
            period=period,
            allow_fallback_to_global=allow_fallback_to_global
        )
    
    def list_documents_for_session(self, session_id: str, include_global: bool = True) -> List[Dict[str, Any]]:
        """List documents for session delegated to document manager.
        
        Args:
            session_id: Session ID to filter documents
            include_global: Whether to include global documents
            
        Returns:
            List of document information dictionaries
        """
        return self.document_manager.list_documents_for_session(session_id, include_global)

    def debug_print_all_metadata(self) -> None:
        """Print all metadata in current collection for debugging."""
        try:
            collection = self.chroma_client.get_collection(config.COLLECTION_NAME)
            docs = collection.get()
            for i in range(len(docs['ids'])):
                print(f"{i+1}. metadata: {docs['metadatas'][i]}")
        except Exception as e:
            logger.error(f"Error fetching metadata: {str(e)}")