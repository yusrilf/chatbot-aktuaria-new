"""Semantic Retriever for CoT Retrieval System.

This module implements semantic retrieval on filtered document subsets,
retrieving top M chunks per document for better context coverage.

Author: AI Assistant
Date: 2025-01-04
Version: 1.1.0 - Added caching and parallel processing optimization
"""

import logging
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from app.services.retrieval.document_filter import FilterResult
from app.services.retrieval.query_preprocessor import ExtractedEntities
from app.services.retrieval.cache_manager import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Container for retrieved document chunk."""
    content: str
    metadata: Dict[str, Any]
    score: float
    document_name: str
    section_heading: Optional[str] = None
    chunk_index: Optional[int] = None


@dataclass
class SemanticRetrievalResult:
    """Result from semantic retrieval."""
    chunks: List[RetrievedChunk]
    total_chunks: int
    documents_searched: List[str]
    retrieval_strategy: str
    query_used: str
    search_metadata: Dict[str, Any]


class SemanticRetriever:
    """Performs semantic retrieval on filtered document subsets."""
    
    def __init__(self, vector_store_manager=None, search_manager=None):
        """
        Initialize the semantic retriever.
        
        Args:
            vector_store_manager: Vector store manager for document retrieval
            search_manager: Search manager for enhanced retrieval
        """
        self.vector_store_manager = vector_store_manager
        self.search_manager = search_manager
        self.default_chunks_per_doc = 2
        self.max_total_chunks = 10
        
        # Initialize cache manager for performance optimization
        self.cache_manager = CacheManager(
            max_cache_size=1000,  # Cache up to 1000 queries
            default_ttl=300.0     # 5 minutes TTL
        )
        
        # Thread pool for parallel processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        logger.info("SemanticRetriever initialized with caching and parallel processing")
        logger.info(f"Cache settings: max_size=1000, TTL=300s")
    
    def retrieve_from_subset(
        self,
        query: str,
        entities: ExtractedEntities,
        filter_result: FilterResult,
        session_id: Optional[str] = None,
        chunks_per_doc: int = 2,
        max_total_chunks: int = 10,
        complexity_level: Optional[str] = None
    ) -> SemanticRetrievalResult:
        """
        Perform semantic retrieval on filtered document subset.
        
        Args:
            query: User query
            entities: Extracted entities from query
            filter_result: Result from document filtering
            session_id: Optional session ID
            chunks_per_doc: Maximum chunks per document
            max_total_chunks: Maximum total chunks to retrieve
            complexity_level: Query complexity level ('simple', 'moderate', 'complex')
            
        Returns:
            SemanticRetrievalResult with retrieved chunks
        """
        try:
            logger.info(f"Starting semantic retrieval for {len(filter_result.filtered_files)} documents")
            
            # Apply complexity-based limits if provided
            if complexity_level:
                complexity_limits = {
                    'simple': {'chunks_per_doc': 2, 'max_total_chunks': 5},
                    'moderate': {'chunks_per_doc': 3, 'max_total_chunks': 10},
                    'complex': {'chunks_per_doc': 4, 'max_total_chunks': 15}
                }
                
                if complexity_level in complexity_limits:
                    limits = complexity_limits[complexity_level]
                    chunks_per_doc = min(chunks_per_doc, limits['chunks_per_doc'])
                    max_total_chunks = min(max_total_chunks, limits['max_total_chunks'])
                    
                    logger.info(f"Applied complexity-based limits for {complexity_level}: "
                               f"chunks_per_doc={chunks_per_doc}, max_total_chunks={max_total_chunks}")
            
            # Prepare search query
            search_query = self._prepare_search_query(query, entities)
            
            # Perform retrieval based on available managers
            if self.search_manager:
                result = self._retrieve_with_search_manager(
                    search_query, filter_result, session_id, 
                    chunks_per_doc, max_total_chunks
                )
            elif self.vector_store_manager:
                result = self._retrieve_with_vector_store(
                    search_query, filter_result, session_id,
                    chunks_per_doc, max_total_chunks
                )
            else:
                logger.warning("No retrieval manager available, using fallback")
                result = self._fallback_retrieval(search_query, filter_result)
            
            logger.info(f"Semantic retrieval completed: {len(result.chunks)} chunks retrieved "
                       f"(complexity: {complexity_level or 'not specified'})")
            return result
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {str(e)}")
            return self._create_error_result(query, filter_result, str(e))
    
    def __del__(self):
        """
        Cleanup resources when the retriever is destroyed.
        """
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
                logger.debug("Thread pool shutdown completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring performance.
        
        Returns:
            Dictionary containing cache statistics
        """
        try:
            return {
                'query_cache_size': len(self.cache_manager.query_cache.cache),
                'embedding_cache_size': len(self.cache_manager.embedding_cache.cache),
                'query_cache_hits': getattr(self.cache_manager.query_cache, 'hits', 0),
                'query_cache_misses': getattr(self.cache_manager.query_cache, 'misses', 0),
                'embedding_cache_hits': getattr(self.cache_manager.embedding_cache, 'hits', 0),
                'embedding_cache_misses': getattr(self.cache_manager.embedding_cache, 'misses', 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}
    
    def clear_cache(self):
        """
        Clear all cached data.
        """
        try:
            self.cache_manager.clear_all_caches()
            logger.info("All caches cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def _prepare_search_query(self, query: str, entities: ExtractedEntities) -> str:
        """Prepare enhanced search query from original query and entities."""
        try:
            # Start with normalized query
            enhanced_query = entities.normalized_query
            
            # Add important entities
            query_parts = [enhanced_query]
            
            # Add PSAK references
            if entities.psak_references:
                query_parts.extend(entities.psak_references)
            
            # Add PUC references
            if entities.puc_references:
                query_parts.extend(entities.puc_references)
            
            # Add key technical terms
            if entities.technical_terms:
                # Add most relevant technical terms (limit to avoid query bloat)
                key_terms = entities.technical_terms[:5]
                query_parts.extend(key_terms)
            
            # Combine into enhanced query
            enhanced_query = " ".join(query_parts)
            
            logger.debug(f"Enhanced query: {enhanced_query}")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error preparing search query: {str(e)}")
            return query
    
    def _retrieve_with_search_manager(
        self,
        query: str,
        filter_result: FilterResult,
        session_id: Optional[str],
        chunks_per_doc: int,
        max_total_chunks: int
    ) -> SemanticRetrievalResult:
        """Retrieve using search manager."""
        try:
            logger.info("Using search manager for retrieval")
            
            # Create metadata filter for selected documents
            metadata_filter = {
                "filename": {"$in": filter_result.filtered_files}
            }
            
            # Perform search with document filter
            search_results = self.search_manager.search_documents(
                query=query,
                session_id=session_id,
                limit=max_total_chunks,
                use_hybrid=True,
                enable_batch_optimization=True,
                metadata_filter=metadata_filter
            )
            
            # Convert search results to retrieved chunks
            chunks = self._convert_search_results_to_chunks(
                search_results, chunks_per_doc
            )
            
            return SemanticRetrievalResult(
                chunks=chunks,
                total_chunks=len(chunks),
                documents_searched=filter_result.filtered_files,
                retrieval_strategy="search_manager",
                query_used=query,
                search_metadata={
                    "metadata_filter": metadata_filter,
                    "chunks_per_doc": chunks_per_doc,
                    "max_total_chunks": max_total_chunks
                }
            )
            
        except Exception as e:
            logger.error(f"Error with search manager retrieval: {str(e)}")
            # Fallback to vector store if available
            if self.vector_store_manager:
                return self._retrieve_with_vector_store(
                    query, filter_result, session_id, chunks_per_doc, max_total_chunks
                )
            else:
                return self._fallback_retrieval(query, filter_result)
    
    def _search_document_with_cache(
        self,
        query: str,
        doc_name: str,
        doc_filter: Dict[str, Any],
        session_id: Optional[str],
        chunks_per_doc: int
    ) -> List[Tuple[Any, float]]:
        """
        Search a single document with caching support.
        
        Args:
            query: Search query
            doc_name: Document name
            doc_filter: Document filter
            session_id: Session identifier
            chunks_per_doc: Number of chunks per document
            
        Returns:
            List of (document, score) tuples
        """
        # Check cache first
        cache_key_data = {**doc_filter, 'chunks_per_doc': chunks_per_doc}
        cached_result = self.cache_manager.get_query_result(query, cache_key_data, session_id or "none")
        
        if cached_result is not None:
            logger.debug(f"Using cached result for document: {doc_name}")
            return cached_result
        
        # Perform dual query: search both user session and global session
        doc_results = []
        
        # First, try to search in user session if session_id is provided
        if session_id:
            try:
                user_results = self.vector_store_manager.similarity_search_with_score(
                    query=query,
                    k=chunks_per_doc,
                    filter=doc_filter,
                    session_id=session_id
                )
                doc_results.extend(user_results)
                logger.debug(f"Found {len(user_results)} chunks in user session {session_id} for {doc_name}")
            except Exception as e:
                logger.debug(f"Error searching user session {session_id} for {doc_name}: {str(e)}")
        
        # Then, search in global session to ensure global documents are included
        try:
            global_results = self.vector_store_manager.similarity_search_with_score(
                query=query,
                k=chunks_per_doc,
                filter=doc_filter,
                session_id="global"
            )
            doc_results.extend(global_results)
            logger.debug(f"Found {len(global_results)} chunks in global session for {doc_name}")
        except Exception as e:
            logger.debug(f"Error searching global session for {doc_name}: {str(e)}")
        
        # Remove duplicates based on content hash and keep best scores
        seen_content = {}
        unique_results = []
        for doc_result, score in doc_results:
            content_hash = hash(doc_result.page_content[:200])
            if content_hash not in seen_content or score > seen_content[content_hash][1]:
                seen_content[content_hash] = (doc_result, score)
        
        # Convert back to list and sort by score
        doc_results = list(seen_content.values())
        doc_results.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to chunks_per_doc
        doc_results = doc_results[:chunks_per_doc]
        
        # Cache the result
        self.cache_manager.cache_query_result(query, cache_key_data, session_id or "none", doc_results)
        
        return doc_results
    
    def _retrieve_with_vector_store(
        self,
        query: str,
        filter_result: FilterResult,
        session_id: Optional[str],
        chunks_per_doc: int,
        max_total_chunks: int
    ) -> SemanticRetrievalResult:
        """Retrieve using vector store manager with parallel processing and caching."""
        try:
            logger.info(f"Using vector store manager for retrieval with session_id: {session_id}")
            logger.info(f"Filtered documents: {filter_result.filtered_files}")
            
            all_chunks = []
            chunks_per_document = {}
            
            # Prepare tasks for parallel execution
            search_tasks = []
            
            # Retrieve from each filtered document using parallel processing
            for doc in filter_result.filtered_files:
                try:
                    # Handle both Document objects and string filenames
                    if hasattr(doc, 'filename'):
                        doc_name = doc.filename
                        doc_filter = {"filename": doc_name}
                    elif hasattr(doc, 'name'):
                        doc_name = doc.name
                        doc_filter = {"filename": doc_name}
                    elif isinstance(doc, str):
                        doc_name = doc
                        doc_filter = {"filename": doc_name}
                    else:
                        logger.warning(f"Unknown document type: {type(doc)}")
                        continue
                    
                    # Submit task to thread pool for parallel processing
                    future = self.thread_pool.submit(
                        self._search_document_with_cache,
                        query, doc_name, doc_filter, session_id, chunks_per_doc
                    )
                    search_tasks.append((future, doc_name))
                    
                except Exception as e:
                    logger.error(f"Error preparing search task for document {doc}: {str(e)}")
                    continue
            
            # Collect results from parallel tasks
            for future, doc_name in search_tasks:
                try:
                    doc_results = future.result(timeout=10.0)  # 10 second timeout per document
                    
                    logger.debug(f"Found {len(doc_results)} chunks for document {doc_name}")
                    
                    # Convert to retrieved chunks
                    doc_chunks = []
                    for doc_result, score in doc_results:
                        # Filter out placeholder error messages
                        if self._is_placeholder_content(doc_result.page_content, doc_result.metadata):
                            logger.info(f"Filtering out placeholder content from {doc_name}")
                            continue
                            
                        chunk = RetrievedChunk(
                            content=doc_result.page_content,
                            metadata=doc_result.metadata,
                            score=score,
                            document_name=doc_name,
                            section_heading=doc_result.metadata.get('section_heading'),
                            chunk_index=doc_result.metadata.get('chunk_index')
                        )
                        doc_chunks.append(chunk)
                    
                    chunks_per_document[doc_name] = len(doc_chunks)
                    all_chunks.extend(doc_chunks)
                    
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Timeout searching document: {doc_name}")
                except Exception as e:
                    logger.error(f"Error retrieving from document {doc_name}: {str(e)}")
                    continue
            
            logger.info(f"Total chunks retrieved: {len(all_chunks)}")
            
            # Sort by score and limit total chunks
            all_chunks.sort(key=lambda x: x.score, reverse=True)
            limited_chunks = all_chunks[:max_total_chunks]
            
            return SemanticRetrievalResult(
                chunks=limited_chunks,
                total_chunks=len(limited_chunks),
                documents_searched=[doc.filename if hasattr(doc, 'filename') else 
                                  doc.name if hasattr(doc, 'name') else 
                                  str(doc) for doc in filter_result.filtered_files],
                retrieval_strategy="vector_store_parallel_cached",
                query_used=query,
                search_metadata={
                    "chunks_per_document": chunks_per_document,
                    "total_documents_searched": len(filter_result.filtered_files),
                    "chunks_per_doc_limit": chunks_per_doc,
                    "max_total_chunks": max_total_chunks,
                    "session_id": session_id,
                    "cache_enabled": True,
                    "parallel_processing": True
                }
            )
            
        except Exception as e:
            logger.error(f"Error with vector store retrieval: {str(e)}")
            return self._fallback_retrieval(query, filter_result)
    
    def _convert_search_results_to_chunks(
        self, 
        search_results: List[Any], 
        chunks_per_doc: int
    ) -> List[RetrievedChunk]:
        """Convert search manager results to retrieved chunks."""
        try:
            chunks = []
            chunks_per_document = {}
            
            for result in search_results:
                # Extract document name and content
                doc_name = result.get('metadata', {}).get('filename', 'unknown')
                content = result.get('content', '')
                
                # Filter out placeholder error messages
                if self._is_placeholder_content(content, result.get('metadata', {})):
                    logger.info(f"Filtering out placeholder content from {doc_name}")
                    continue
                
                # Check chunks per document limit
                current_count = chunks_per_document.get(doc_name, 0)
                if current_count >= chunks_per_doc:
                    continue
                
                # Create retrieved chunk
                chunk = RetrievedChunk(
                    content=content,
                    metadata=result.get('metadata', {}),
                    score=result.get('score', 0.0),
                    document_name=doc_name,
                    section_heading=result.get('metadata', {}).get('section_heading'),
                    chunk_index=result.get('metadata', {}).get('chunk_index')
                )
                
                chunks.append(chunk)
                chunks_per_document[doc_name] = current_count + 1
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error converting search results: {str(e)}")
            return []
    
    def _is_placeholder_content(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Check if content is a placeholder error message."""
        try:
            # Check for placeholder metadata flag
            if metadata.get('is_placeholder', False):
                return True
            
            # Check for specific placeholder filenames
            filename = metadata.get('filename', '')
            if filename in ['no_results_placeholder', 'error_placeholder']:
                return True
            
            # Check for common error message patterns
            error_patterns = [
                'tidak ada dokumen yang relevan',
                'maaf, tidak ada dokumen',
                'terjadi kesalahan dalam pencarian',
                'silakan coba dengan kata kunci yang berbeda',
                'no relevant documents found',
                'error in search'
            ]
            
            content_lower = content.lower()
            for pattern in error_patterns:
                if pattern in content_lower:
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking placeholder content: {e}")
            return False
    
    def _fallback_retrieval(
        self, 
        query: str, 
        filter_result: FilterResult
    ) -> SemanticRetrievalResult:
        """Fallback retrieval when no managers are available."""
        try:
            logger.warning("Using fallback retrieval - no vector store or search manager available")
            
            # Create dummy chunks for filtered documents
            chunks = []
            for i, doc_name in enumerate(filter_result.filtered_files[:5]):
                chunk = RetrievedChunk(
                    content=f"Fallback content for {doc_name} - query: {query}",
                    metadata={"filename": doc_name, "fallback": True},
                    score=0.5 - (i * 0.1),  # Decreasing scores
                    document_name=doc_name,
                    section_heading="Fallback Section"
                )
                chunks.append(chunk)
            
            return SemanticRetrievalResult(
                chunks=chunks,
                total_chunks=len(chunks),
                documents_searched=filter_result.filtered_files,
                retrieval_strategy="fallback",
                query_used=query,
                search_metadata={"fallback_reason": "No retrieval managers available"}
            )
            
        except Exception as e:
            logger.error(f"Error in fallback retrieval: {str(e)}")
            return self._create_error_result(query, filter_result, str(e))
    
    def _create_error_result(
        self, 
        query: str, 
        filter_result: FilterResult, 
        error: str
    ) -> SemanticRetrievalResult:
        """Create error result for failed retrieval."""
        return SemanticRetrievalResult(
            chunks=[],
            total_chunks=0,
            documents_searched=filter_result.filtered_files,
            retrieval_strategy="error",
            query_used=query,
            search_metadata={"error": error}
        )
    
    def get_chunks_by_document(
        self, 
        retrieval_result: SemanticRetrievalResult
    ) -> Dict[str, List[RetrievedChunk]]:
        """Group retrieved chunks by document."""
        try:
            chunks_by_doc = {}
            
            for chunk in retrieval_result.chunks:
                doc_name = chunk.document_name
                if doc_name not in chunks_by_doc:
                    chunks_by_doc[doc_name] = []
                chunks_by_doc[doc_name].append(chunk)
            
            return chunks_by_doc
            
        except Exception as e:
            logger.error(f"Error grouping chunks by document: {str(e)}")
            return {}
    
    def get_retrieval_statistics(
        self, 
        retrieval_result: SemanticRetrievalResult
    ) -> Dict[str, Any]:
        """Get statistics about retrieval result."""
        try:
            chunks_by_doc = self.get_chunks_by_document(retrieval_result)
            
            stats = {
                "total_chunks": retrieval_result.total_chunks,
                "documents_with_chunks": len(chunks_by_doc),
                "documents_searched": len(retrieval_result.documents_searched),
                "average_score": sum(chunk.score for chunk in retrieval_result.chunks) / len(retrieval_result.chunks) if retrieval_result.chunks else 0,
                "chunks_per_document": {doc: len(chunks) for doc, chunks in chunks_by_doc.items()},
                "retrieval_strategy": retrieval_result.retrieval_strategy
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating retrieval statistics: {str(e)}")
            return {"error": str(e)}