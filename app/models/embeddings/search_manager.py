"""Search Manager for handling search operations.

This module manages similarity search, hybrid search, and document reranking
operations for the vector store.
"""

import traceback
from langchain_core.documents import Document
from typing import Dict, List, Optional, Tuple, Any
import logging

from app.config import config
from app.models.search.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

class SearchManager:
    """Manages search operations for the vector store."""
    
    def __init__(self, vector_store_manager):
        """Initialize SearchManager with reference to VectorStoreManager.
        
        Args:
            vector_store_manager: Reference to the main VectorStoreManager instance
        """
        self.vector_store_manager = vector_store_manager
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self._hybrid_initialized = False
    
    def _initialize_hybrid_retriever(self) -> bool:
        """Initialize hybrid retriever if not already done.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if self._hybrid_initialized and self.hybrid_retriever:
                return True
            
            # Get vector store from vector_store_manager
            vector_store = getattr(self.vector_store_manager, 'vector_store', None)
            if not vector_store:
                logger.warning("Vector store not available for hybrid retriever")
                return False
            
            # Initialize hybrid retriever with configurable weights
            semantic_weight = getattr(config, 'HYBRID_SEMANTIC_WEIGHT', 0.7)
            bm25_weight = getattr(config, 'HYBRID_BM25_WEIGHT', 0.3)
            
            self.hybrid_retriever = HybridRetriever(
                vector_store=vector_store,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                bm25_k1=getattr(config, 'BM25_K1', 1.2),
                bm25_b=getattr(config, 'BM25_B', 0.75),
                normalize_scores=getattr(config, 'HYBRID_NORMALIZE_SCORES', True),
                min_score_threshold=getattr(config, 'HYBRID_MIN_SCORE_THRESHOLD', 0.0)
            )
            
            # Build index with available documents
            try:
                # Get documents from vector store manager
                if hasattr(self.vector_store_manager, 'get_all_documents'):
                    documents = self.vector_store_manager.get_all_documents()
                    if documents:
                        self.hybrid_retriever.build_index(documents)
                        logger.info(f"Hybrid retriever built with {len(documents)} documents")
                    else:
                        logger.warning("No documents available for hybrid retriever")
                else:
                    logger.warning("Cannot get documents for hybrid retriever initialization")
            except Exception as build_error:
                logger.warning(f"Could not build hybrid index: {build_error}")
            
            self._hybrid_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing hybrid retriever: {str(e)}")
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
        use_hybrid: bool = True  # Now defaults to True for hybrid search
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid similarity search combining semantic and BM25 retrieval with reranking.
        
        Args:
            query: Query pencarian
            session_id: ID sesi
            k: Jumlah hasil yang diinginkan
            filter: Filter tambahan
            session_required: Apakah session_id wajib
            allow_fallback_to_global: Fallback ke dokumen global
            return_placeholder_on_empty: Return placeholder jika kosong
            use_hybrid: Whether to use hybrid retrieval (True) or pure vector search (False)
            
        Returns:
            List tuple (Document, score)
        """
        try:
            k = k or config.DEFAULT_K
            results = []
            
            # Use hybrid search if enabled and available
            hybrid_available = (use_hybrid and self.hybrid_retriever and 
                              hasattr(self.hybrid_retriever, 'is_built') and 
                              self.hybrid_retriever.is_built)
            
            if hybrid_available:
                try:
                    logger.info(f"Using hybrid retrieval for query: '{query[:50]}...'")
                    
                    # Perform hybrid retrieval
                    results = self.hybrid_retriever.retrieve_with_scores(
                        query=query,
                        k=k * config.HYBRID_SEARCH_MULTIPLIER,  # Use configurable multiplier
                        filter_metadata=filter
                    )
                    
                    # Convert to expected format if needed
                    if results:
                        logger.info(f"Hybrid retrieval returned {len(results)} results")
                    else:
                        logger.warning("Hybrid retrieval returned no results, falling back to vector search")
                        raise Exception("No hybrid results")
                        
                except Exception as hybrid_error:
                    logger.warning(f"Hybrid retrieval failed: {hybrid_error}, falling back to vector search")
                    use_hybrid = False
                    results = []
            else:
                logger.info("Hybrid retrieval not available, using vector search")
                use_hybrid = False
            
            # Fallback to pure semantic search
            if not use_hybrid:
                logger.info(f"Using pure vector search for query: '{query[:50]}...'")
                results = self.similarity_search_with_score(
                    query=query,
                    session_id=session_id,
                    k=k * config.HYBRID_SEARCH_MULTIPLIER,  # Use configurable multiplier
                    filter=filter,
                    session_required=session_required,
                    allow_fallback_to_global=allow_fallback_to_global,
                    return_placeholder_on_empty=return_placeholder_on_empty
                )
            
            # Apply reranking if enabled and we have results
            if results and config.ENABLE_RERANKING:
                # Normalize scores before reranking for consistency
                logger.info(f"Normalizing {len(results)} scores before reranking")
                results = self._normalize_similarity_scores(results)
                results = self.rerank_documents(query, results, top_n=k)
            else:
                # Normalize scores even without reranking
                if results:
                    logger.info(f"Normalizing {len(results)} scores (no reranking)")
                    results = self._normalize_similarity_scores(results)
                
                # Apply dynamic retrieval or traditional k-based filtering
                if config.ENABLE_DYNAMIC_RETRIEVAL:
                    # Dynamic retrieval: return all documents above threshold
                    dynamic_results = []
                    for doc, score in results:
                        if score >= config.DYNAMIC_RETRIEVAL_THRESHOLD:
                            dynamic_results.append((doc, score))
                    
                    # Ensure we have minimum and maximum document limits
                    if len(dynamic_results) < config.DYNAMIC_RETRIEVAL_MIN_DOCS:
                        # If we have fewer than minimum, take top k documents regardless of threshold
                        logger.info(f"Dynamic retrieval found {len(dynamic_results)} docs < minimum {config.DYNAMIC_RETRIEVAL_MIN_DOCS}, using top {k} instead")
                        results = results[:k]
                    elif len(dynamic_results) > config.DYNAMIC_RETRIEVAL_MAX_DOCS:
                        # If we have more than maximum, limit to max docs
                        logger.info(f"Dynamic retrieval found {len(dynamic_results)} docs > maximum {config.DYNAMIC_RETRIEVAL_MAX_DOCS}, limiting to {config.DYNAMIC_RETRIEVAL_MAX_DOCS}")
                        results = dynamic_results[:config.DYNAMIC_RETRIEVAL_MAX_DOCS]
                    else:
                        # Perfect range, use all dynamic results
                        logger.info(f"Dynamic retrieval found {len(dynamic_results)} docs above threshold {config.DYNAMIC_RETRIEVAL_THRESHOLD}")
                        results = dynamic_results
                else:
                    # Traditional k-based limiting
                    results = results[:k]
            
            search_type = "hybrid" if use_hybrid else "vector"
            logger.info(f"{search_type.capitalize()} search returned {len(results)} final results")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid similarity search: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            if return_placeholder_on_empty:
                placeholder_doc = Document(
                    page_content="Maaf, terjadi kesalahan dalam pencarian dokumen. Silakan coba lagi.",
                    metadata={
                        'filename': 'error_placeholder',
                        'session_id': session_id,
                        'error': str(e)
                    }
                )
                return [(placeholder_doc, 0.0)]
            
            return []
    
    def similarity_search(self, query: str, session_id: str, k: int = None) -> List[Document]:
        """Perform similarity search without scores.
        
        Args:
            query: Search query
            session_id: Session ID for filtering
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        try:
            k = k or config.DEFAULT_K
            
            # Get documents with scores and extract just the documents
            results_with_scores = self.similarity_search_with_score(
                query=query,
                session_id=session_id,
                k=k
            )
            
            return [doc for doc, score in results_with_scores]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def rerank_documents(self, query: str, docs_with_scores: List[tuple], top_n: int = None) -> List[tuple]:
        """Rerank documents using Cohere reranker.
        
        Args:
            query: Original search query
            docs_with_scores: List of (Document, score) tuples
            top_n: Number of top results to return after reranking
            
        Returns:
            List of reranked (Document, score) tuples
        """
        try:
            if not docs_with_scores:
                return []
            
            top_n = top_n or config.DEFAULT_K
            
            # Prepare documents for reranking
            documents = [doc.page_content for doc, _ in docs_with_scores]
            
            # Use Cohere reranker
            rerank_response = self.vector_store_manager.cohere_client.rerank(
                model=config.COHERE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents))
            )
            
            # Create reranked results
            reranked_results = []
            for result in rerank_response.results:
                original_doc, original_score = docs_with_scores[result.index]
                # Use rerank score as the new score
                reranked_results.append((original_doc, result.relevance_score))
            
            logger.info(f"Reranked {len(reranked_results)} documents")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in document reranking: {str(e)}")
            # Return original results if reranking fails
            return docs_with_scores[:top_n] if top_n else docs_with_scores
    
    def _normalize_similarity_scores(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Normalize similarity scores to ensure consistent 0-1 range.
        
        ChromaDB returns different score types (cosine similarity, L2 distance, etc.)
        This method normalizes them to a consistent 0-1 range where higher = more relevant.
        
        Args:
            results: List of (Document, raw_score) tuples
            
        Returns:
            List of (Document, normalized_score) tuples
        """
        if not results:
            return results
        
        try:
            scores = [score for _, score in results]
            
            # Handle edge cases
            if len(scores) == 1:
                # Single result gets high relevance score
                return [(results[0][0], 0.85)]
            
            # Check if scores are already in good range (0-1)
            min_score = min(scores)
            max_score = max(scores)
            
            # If scores are very small (< 0.01), they're likely distance metrics
            # Convert distance to similarity: similarity = 1 / (1 + distance)
            if max_score < 0.01:
                logger.info("Converting distance scores to similarity scores")
                normalized_results = []
                for doc, score in results:
                    # Convert distance to similarity, then normalize to 0.1-1.0 range
                    similarity = 1.0 / (1.0 + score)
                    # Scale to meaningful range
                    normalized_score = 0.1 + (similarity * 0.9)
                    normalized_results.append((doc, normalized_score))
                return normalized_results
            
            # If scores are already in reasonable range (0.1-1.0), keep them
            if min_score >= 0.1 and max_score <= 1.0:
                logger.debug("Scores already in good range, keeping original")
                return results
            
            # Otherwise, apply min-max normalization
            if max_score == min_score:
                # All scores are identical
                normalized_results = [(doc, 0.8) for doc, _ in results]
            else:
                # Min-max normalization to 0.1-1.0 range
                normalized_results = []
                for doc, score in results:
                    normalized = (score - min_score) / (max_score - min_score)
                    # Scale to 0.1-1.0 to maintain meaningful differences
                    final_score = 0.1 + (normalized * 0.9)
                    normalized_results.append((doc, final_score))
            
            logger.info(f"Normalized scores: original range [{min_score:.4f}, {max_score:.4f}] -> "
                       f"new range [{min([s for _, s in normalized_results]):.4f}, {max([s for _, s in normalized_results]):.4f}]")
            
            return normalized_results
            
        except Exception as e:
            logger.error(f"Error normalizing scores: {e}")
            # Return original results if normalization fails
            return results

    def similarity_search_with_score(
        self,
        query: str,
        session_id: str = None,
        k: int = None,
        filter: Optional[Dict] = None,
        session_required: bool = True,
        allow_fallback_to_global: bool = False,
        return_placeholder_on_empty: bool = True,
        apply_score_threshold: bool = True
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores.
        
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
        try:
            k = k or config.DEFAULT_K
            # Use higher initial retrieval count for better filtering
            initial_k = max(k * config.RETRIEVAL_MULTIPLIER, config.INITIAL_RETRIEVAL_K)
            
            # Search from both session-specific and global documents, then combine results
            all_results = []
            
            # Build base filter
            base_filter = {}
            if filter:
                base_filter.update(filter)
            
            # 1. Search session-specific documents if session_id provided and required
            session_results = []
            if session_id and session_required:
                # Build session filter - avoid $and operator as it's not supported
                session_filter = {"session_id": session_id}
                
                # Apply additional filters by performing layered search
                if base_filter:
                    # Merge base_filter into session_filter directly
                    session_filter.update(base_filter)
                
                logger.info(f"Searching session-specific documents for session {session_id} with filter: {session_filter}")
                try:
                    session_results = self.vector_store_manager.vectorstore.similarity_search_with_score(
                        query=query,
                        k=initial_k,  # Use higher initial k
                        filter=session_filter
                    )
                    logger.info(f"Found {len(session_results)} session-specific results")
                except Exception as e:
                    logger.warning(f"Session search failed: {str(e)}, trying without additional filters")
                    # Fallback to simple session filter if complex filter fails
                    try:
                        session_results = self.vector_store_manager.vectorstore.similarity_search_with_score(
                            query=query,
                            k=initial_k,
                            filter={"session_id": session_id}
                        )
                        logger.info(f"Found {len(session_results)} session-specific results (fallback)")
                    except Exception as e2:
                        logger.error(f"Session search completely failed: {str(e2)}")
                        session_results = []
                
                all_results.extend(session_results)
            
            # 2. Search global documents if fallback is allowed
            global_results = []
            if allow_fallback_to_global:
                # Build global filter - avoid $and operator as it's not supported
                global_filter = {"session_id": "global"}
                
                # Apply additional filters by performing layered search
                if base_filter:
                    # Merge base_filter into global_filter directly
                    global_filter.update(base_filter)
                
                logger.info(f"Searching global documents with filter: {global_filter}")
                try:
                    global_results = self.vector_store_manager.vectorstore.similarity_search_with_score(
                        query=query,
                        k=initial_k,  # Use higher initial k
                        filter=global_filter
                    )
                    logger.info(f"Found {len(global_results)} global results")
                except Exception as e:
                    logger.warning(f"Global search failed: {str(e)}, trying without additional filters")
                    # Fallback to simple global filter if complex filter fails
                    try:
                        global_results = self.vector_store_manager.vectorstore.similarity_search_with_score(
                            query=query,
                            k=initial_k,
                            filter={"session_id": "global"}
                        )
                        logger.info(f"Found {len(global_results)} global results (fallback)")
                    except Exception as e2:
                        logger.error(f"Global search completely failed: {str(e2)}")
                        global_results = []
                
                all_results.extend(global_results)
            
            # 3. Combine and deduplicate results based on content hash to avoid duplicates
            seen_content = set()
            results = []
            for doc, score in all_results:
                # Create a simple hash of content to detect duplicates
                content_hash = hash(doc.page_content[:200])  # Use first 200 chars for hash
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    results.append((doc, score))
            
            logger.info(f"Combined results: {len(session_results)} session + {len(global_results)} global = {len(all_results)} total, {len(results)} after deduplication")
            
            # Normalize scores first to ensure consistency
            if results:
                logger.info(f"Normalizing {len(results)} scores before filtering")
                results = self._normalize_similarity_scores(results)
            
            # Apply dynamic retrieval or traditional k-based filtering
            if config.ENABLE_DYNAMIC_RETRIEVAL:
                # Dynamic retrieval: return all documents above threshold
                dynamic_results = []
                for doc, score in results:
                    # Debug: log all scores for analysis
                    logger.debug(f"Document score: {score:.4f} (dynamic threshold: {config.DYNAMIC_RETRIEVAL_THRESHOLD}) - {doc.metadata.get('filename', 'unknown')}")
                    if score >= config.DYNAMIC_RETRIEVAL_THRESHOLD:
                        dynamic_results.append((doc, score))
                    else:
                        logger.debug(f"Filtered out document '{doc.metadata.get('filename', 'unknown')}' with score: {score:.4f} < {config.DYNAMIC_RETRIEVAL_THRESHOLD}")
                
                # Ensure we have minimum and maximum document limits
                if len(dynamic_results) < config.DYNAMIC_RETRIEVAL_MIN_DOCS:
                    # If we have fewer than minimum, take top k documents regardless of threshold
                    logger.info(f"Dynamic retrieval found {len(dynamic_results)} docs < minimum {config.DYNAMIC_RETRIEVAL_MIN_DOCS}, using top {k} instead")
                    results = results[:k]
                elif len(dynamic_results) > config.DYNAMIC_RETRIEVAL_MAX_DOCS:
                    # If we have more than maximum, limit to max docs
                    logger.info(f"Dynamic retrieval found {len(dynamic_results)} docs > maximum {config.DYNAMIC_RETRIEVAL_MAX_DOCS}, limiting to {config.DYNAMIC_RETRIEVAL_MAX_DOCS}")
                    results = dynamic_results[:config.DYNAMIC_RETRIEVAL_MAX_DOCS]
                else:
                    # Perfect range, use all dynamic results
                    logger.info(f"Dynamic retrieval found {len(dynamic_results)} docs above threshold {config.DYNAMIC_RETRIEVAL_THRESHOLD}")
                    results = dynamic_results
            else:
                # Traditional score threshold filtering
                if apply_score_threshold and config.MIN_RELEVANCE_SCORE > 0:
                    filtered_results = []
                    for doc, score in results:
                        # Debug: log all scores for analysis
                        logger.debug(f"Document score: {score:.4f} (threshold: {config.MIN_RELEVANCE_SCORE}) - {doc.metadata.get('filename', 'unknown')}")
                        if score >= config.MIN_RELEVANCE_SCORE:
                            filtered_results.append((doc, score))
                        else:
                            logger.info(f"Filtered out document '{doc.metadata.get('filename', 'unknown')}' with score: {score:.4f} < {config.MIN_RELEVANCE_SCORE}")
                    
                    logger.info(f"After score filtering: {len(filtered_results)}/{len(results)} results remain (threshold: {config.MIN_RELEVANCE_SCORE})")
                    results = filtered_results
                else:
                    logger.info(f"Score threshold filtering disabled (apply_score_threshold={apply_score_threshold}, MIN_RELEVANCE_SCORE={config.MIN_RELEVANCE_SCORE})")
                
                # Limit to requested k after filtering
                if len(results) > k:
                    results = results[:k]
                    logger.info(f"Limited results to top {k} documents")
            
            # Debug: log final results with metadata
            logger.info(f"Final result count: {len(results)} for query: '{query[:50]}...'")
            if results:
                logger.debug(f"Final results metadata: {[doc.metadata.get('filename', 'unknown') for doc, _ in results]}")
            
            # Return placeholder if no results and requested
            if not results and return_placeholder_on_empty:
                placeholder_doc = Document(
                    page_content="Maaf, tidak ada dokumen yang relevan ditemukan untuk pertanyaan Anda. Silakan coba dengan kata kunci yang berbeda.",
                    metadata={
                        'filename': 'no_results_placeholder',
                        'session_id': session_id,
                        'is_placeholder': True
                    }
                )
                return [(placeholder_doc, 0.0)]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            if return_placeholder_on_empty:
                error_doc = Document(
                    page_content="Terjadi kesalahan dalam pencarian. Silakan coba lagi.",
                    metadata={
                        'filename': 'error_placeholder',
                        'session_id': session_id,
                        'error': str(e)
                    }
                )
                return [(error_doc, 0.0)]
            
            return []

    def search_documents(
        self,
        query: str,
        limit: int = 10,
        session_id: str = None,
        use_hybrid: bool = True,
        enable_batch_optimization: bool = True,
        metadata_filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimized document search with batch processing and caching.
        
        Args:
            query: Search query
            limit: Maximum number of results
            session_id: Session identifier for filtering
            use_hybrid: Whether to use hybrid search
            enable_batch_optimization: Enable batch optimization for better performance
            metadata_filter: Optional metadata filter for document filtering
            
        Returns:
            List of search result dictionaries
        """
        try:
            import time
            start_time = time.time()
            
            # Use optimized hybrid search with reduced parameters
            if enable_batch_optimization:
                # Strategy 1: Optimized hybrid search with smart caching
                results = self._optimized_batch_search(
                    query=query,
                    limit=limit,
                    session_id=session_id,
                    use_hybrid=use_hybrid,
                    metadata_filter=metadata_filter
                )
            else:
                # Strategy 2: Standard search (fallback)
                results = self._standard_search(
                    query=query,
                    limit=limit,
                    session_id=session_id,
                    use_hybrid=use_hybrid,
                    metadata_filter=metadata_filter
                )
            
            elapsed_time = time.time() - start_time
            
            # Format results for consistency
            formatted_results = []
            for item in results:
                if isinstance(item, tuple) and len(item) == 2:
                    doc, score = item
                    formatted_results.append({
                        "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                        "score": float(score) if score is not None else 0.0
                    })
                elif isinstance(item, dict):
                    formatted_results.append(item)
                else:
                    # Handle other formats
                    formatted_results.append({
                        "content": str(item),
                        "metadata": {},
                        "score": 0.0
                    })
            
            logger.info(f"Optimized search returned {len(formatted_results)} results in {elapsed_time:.3f}s")
            return formatted_results[:limit]  # Ensure we don't exceed limit
            
        except Exception as e:
            logger.error(f"Error in optimized document search: {e}")
            return []

    def _optimized_batch_search(
        self,
        query: str,
        limit: int,
        session_id: str = None,
        use_hybrid: bool = True,
        metadata_filter: Dict[str, Any] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform optimized batch search with smart session + global combination.
        
        Args:
            query: Search query
            limit: Maximum results
            session_id: Session identifier
            use_hybrid: Use hybrid search
            metadata_filter: Optional metadata filter for document filtering
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            all_results = []
            
            # Step 1: Session-specific search (if session_id provided)
            if session_id and session_id != 'global':
                try:
                    session_limit = min(limit // 2, 6)  # Limit session results for efficiency
                    session_results = self.hybrid_similarity_search_with_score(
                        query=query,
                        session_id=session_id,
                        k=session_limit,
                        filter=metadata_filter,
                        session_required=True,
                        allow_fallback_to_global=False,
                        use_hybrid=use_hybrid
                    )
                    
                    if session_results:
                        all_results.extend(session_results)
                        logger.debug(f"Session search returned {len(session_results)} results")
                        
                except Exception as e:
                    logger.warning(f"Session search failed: {e}")
            
            # Step 2: Global search for additional context
            try:
                remaining_limit = limit - len(all_results)
                if remaining_limit > 0:
                    global_limit = min(remaining_limit, 8)  # Limit global results
                    global_results = self.hybrid_similarity_search_with_score(
                        query=query,
                        session_id='global',
                        k=global_limit,
                        filter=metadata_filter,
                        session_required=True,
                        allow_fallback_to_global=False,
                        use_hybrid=use_hybrid
                    )
                    
                    if global_results:
                        all_results.extend(global_results)
                        logger.debug(f"Global search returned {len(global_results)} results")
                        
            except Exception as e:
                logger.warning(f"Global search failed: {e}")
            
            # Step 3: Smart deduplication and ranking
            if all_results:
                deduplicated_results = self._smart_deduplicate_results(all_results)
                return deduplicated_results[:limit]
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in optimized batch search: {e}")
            return []

    def _standard_search(
        self,
        query: str,
        limit: int,
        session_id: str = None,
        use_hybrid: bool = True,
        metadata_filter: Dict[str, Any] = None
    ) -> List[Tuple[Document, float]]:
        """
        Standard search method (fallback).
        
        Args:
            query: Search query
            limit: Maximum results
            session_id: Session identifier
            use_hybrid: Use hybrid search
            metadata_filter: Optional metadata filter for document filtering
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            return self.hybrid_similarity_search_with_score(
                query=query,
                session_id=session_id or 'global',
                k=limit,
                filter=metadata_filter,
                session_required=False,
                allow_fallback_to_global=True,
                use_hybrid=use_hybrid
            )
        except Exception as e:
            logger.error(f"Error in standard search: {e}")
            return []

    def _smart_deduplicate_results(
        self, 
        results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        Smart deduplication of search results based on content similarity.
        
        Args:
            results: List of (Document, score) tuples
            
        Returns:
            Deduplicated list of (Document, score) tuples
        """
        try:
            if not results:
                return results
            
            seen_content_keys = set()
            deduplicated = []
            
            for doc, score in results:
                # Create content key for similarity check (first 150 chars)
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                content_key = content[:150].lower().strip()
                
                # Skip if we've seen very similar content
                if content_key not in seen_content_keys:
                    seen_content_keys.add(content_key)
                    deduplicated.append((doc, score))
            
            # Sort by score (lower is better for similarity scores)
            deduplicated.sort(key=lambda x: x[1])
            
            logger.debug(f"Deduplicated {len(results)} results to {len(deduplicated)} unique results")
            return deduplicated
            
        except Exception as e:
            logger.error(f"Error in smart deduplication: {e}")
            return results