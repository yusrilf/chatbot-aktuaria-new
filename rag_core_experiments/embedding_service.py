"""Embedding Service - Core functionality for generating embeddings and managing vector store

This module provides simplified embedding generation and vector storage.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb

from config import config
from chunker import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings and managing vector store."""

    def __init__(
        self,
        embedding_model: str = None,
        chroma_path: str = None,
        collection_name: str = None
    ):
        """Initialize embedding service.

        Args:
            embedding_model: OpenAI embedding model name
            chroma_path: Path to ChromaDB
            collection_name: Name of collection
        """
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.chroma_path = chroma_path or config.CHROMA_DB_PATH
        self.collection_name = collection_name or config.COLLECTION_NAME

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=config.OPENAI_API_KEY
        )

        # Initialize ChromaDB
        os.makedirs(self.chroma_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)

        # Initialize vectorstore
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

        logger.info(
            f"EmbeddingService initialized: "
            f"model={self.embedding_model}, "
            f"collection={self.collection_name}"
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to vector store.

        Args:
            chunks: List of DocumentChunk objects
        """
        try:
            logger.info(f"Adding {len(chunks)} chunks to vector store")

            # Convert chunks to Document format
            documents = []
            ids = []

            for chunk in chunks:
                doc = Document(
                    page_content=chunk.content,
                    metadata=chunk.metadata
                )
                documents.append(doc)
                ids.append(chunk.chunk_id)

            # Add to vectorstore
            self.vectorstore.add_documents(documents=documents, ids=ids)

            logger.info(f"Successfully added {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results
            filter_dict: Metadata filters

        Returns:
            List of (Document, score) tuples
        """
        try:
            k = k or config.TOP_K_RESULTS

            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )

            logger.info(f"Found {len(results)} results for query")
            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise

    def hybrid_search(
        self,
        query: str,
        k: int = None,
        semantic_weight: float = None,
        bm25_weight: float = None
    ) -> List[Tuple[Document, float]]:
        """Perform hybrid search (semantic + BM25).

        Note: This is a simplified version. Full BM25 implementation
        would require the BM25 index from the original system.

        Args:
            query: Search query
            k: Number of results
            semantic_weight: Weight for semantic search
            bm25_weight: Weight for BM25 search

        Returns:
            List of (Document, score) tuples
        """
        k = k or config.TOP_K_RESULTS
        semantic_weight = semantic_weight or config.SEMANTIC_WEIGHT

        # For now, just use semantic search
        # In production, you'd combine with BM25
        logger.info("Running semantic search (BM25 not implemented in core version)")
        results = self.similarity_search(query, k=k)

        # Apply semantic weight to scores
        weighted_results = [
            (doc, score * semantic_weight) for doc, score in results
        ]

        return weighted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection stats
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()

            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model,
                'chroma_path': self.chroma_path
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'total_chunks': 0, 'error': str(e)}

    def clear_collection(self) -> None:
        """Clear all documents from collection."""
        try:
            self.chroma_client.delete_collection(self.collection_name)

            # Recreate vectorstore
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )

            logger.info("Collection cleared successfully")

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
