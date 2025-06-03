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
            
            # Add documents to vector store
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents"""
        try:
            k = k or config.TOP_K_RESULTS
            results = self.vectorstore.similarity_search(
                query=query,
                k=k
            )
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Search for similar documents with similarity scores"""
        try:
            k = k or config.TOP_K_RESULTS
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by similarity threshold
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= config.SIMILARITY_THRESHOLD
            ]
            
            logger.info(f"Found {len(filtered_results)} relevant documents above threshold")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching documents with score: {str(e)}")
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