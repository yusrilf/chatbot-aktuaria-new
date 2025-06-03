import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Optional
import logging
import os

from app.config import config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        self.chroma_client = None
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize ChromaDB vector store"""
        try:
            # Ensure directory exists
            os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=config.CHROMA_DB_PATH
            )
            
            # Initialize Langchain Chroma vectorstore
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=config.COLLECTION_NAME,
                embedding_function=self.embeddings
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to vector store"""
        try:
            if not documents:
                logger.warning("No documents to add")
                return False
            
            # Log metadata dari setiap dokumen untuk debugging
            for i, doc in enumerate(documents):
                logger.info(f"Document {i} metadata: {doc.metadata}")
        
            # Add documents to vector store
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, session_id: str, k: int = None) -> List[Document]:
        """Search for similar documents"""
        try:
            k = k or config.TOP_K_RESULTS
            
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
            )
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, session_id: str, k: int = None) -> List[tuple]:
        """Search for similar documents with similarity scores"""
        try:
            k = k or config.TOP_K_RESULTS
            
            # DIAGNOSA: Test apakah filter ChromaDB bekerja
            #logger.info(f"Testing ChromaDB filter for session_id: '{session_id}'")
            
            # Test 1: Dengan filter
            metadata_filter = {"session_id": session_id}
            results_filtered = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=metadata_filter
            )
            
            # Test 2: Tanpa filter (untuk comparison)
            results_no_filter = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k * 3  # Ambil lebih banyak untuk manual filter
            )
            
            #logger.info(f"Results with ChromaDB filter: {len(results_filtered)}")
            #logger.info(f"Results without filter: {len(results_no_filter)}")
            
            # Cek apakah filter benar-benar bekerja
            filter_working = True
            if len(results_filtered) > 0:
                # Cek apakah ada hasil yang tidak sesuai session_id
                for doc, score in results_filtered:
                    if doc.metadata.get('session_id') != session_id:
                        filter_working = False
                        logger.warning("ChromaDB filter NOT working - found non-matching session_id")
                        break
            
            # Jika filter tidak bekerja, gunakan manual filtering
            if not filter_working or len(results_filtered) == 0:
                logger.info("Using manual filtering approach")
                
                # Manual filter dari hasil tanpa filter
                manual_filtered = [
                    (doc, score) for doc, score in results_no_filter
                    if doc.metadata.get('session_id') == session_id
                ]
                
                # Sort by score dan ambil top k
                manual_filtered.sort(key=lambda x: x[1])  # Sort by score (ascending = better)
                results = manual_filtered[:k]
                
                logger.info(f"Manual filtering found {len(results)} matching documents")
            else:
                logger.info("ChromaDB filter working correctly")
                results = results_filtered
            
            # Debug log hasil final
            logger.info(f"Final results count: {len(results)}")
            for i, (doc, score) in enumerate(results):
                doc_session_id = doc.metadata.get('session_id')
                filename = doc.metadata.get('filename', 'N/A')
                logger.info(f"Final Result {i}: filename={filename}, session_id='{doc_session_id}', score={score}")
            
            # Apply similarity threshold
            filtered_results = [
                (doc, score) for doc, score in results
                if score >= config.SIMILARITY_THRESHOLD
            ]
            
            logger.info(f"Found {len(filtered_results)} documents above threshold ({config.SIMILARITY_THRESHOLD})")
        
            #logger.info(filtered_results)
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching documents with score: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            return []
    
    def get_collection_info(self) -> dict:
        """Get information about the current collection"""
        try:
            collection = self.chroma_client.get_collection(config.COLLECTION_NAME)
            return {
                'name': collection.name,
                'count': collection.count(),
                'metadata': collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            self.chroma_client.delete_collection(config.COLLECTION_NAME)
            self._initialize_vectorstore()
            logger.info("Collection deleted and reinitialized")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False