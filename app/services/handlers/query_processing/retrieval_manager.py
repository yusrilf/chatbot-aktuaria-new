"""Retrieval Manager for Document Search and Context Preparation.

This module handles document retrieval, search planning, and context preparation
for different types of queries in the actuarial chatbot.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from langchain_core.documents import Document
from app.utils.performance_monitor import get_global_monitor

logger = logging.getLogger(__name__)

class RetrievalManager:
    """Manages document retrieval and context preparation operations."""
    
    def __init__(self, vector_store_manager=None, document_session_service=None):
        """Initialize RetrievalManager.
        
        Args:
            vector_store_manager: Vector store manager for document retrieval
            document_session_service: Service for managing document sessions
        """
        self.vector_store_manager = vector_store_manager
        self.document_session_service = document_session_service
        
        # Initialize performance monitor
        self.performance_monitor = get_global_monitor()
        
        logger.info("RetrievalManager initialized")
    
    def generate_retrieval_plan(self, 
                              question: str, 
                              session_id: str, 
                              refined_query: Optional[str] = None, 
                              include_global: bool = True) -> Dict[str, Any]:
        """Generate a simplified retrieval plan using pure vector search.
        
        Args:
            question: The question to generate plan for
            session_id: Session identifier
            refined_query: Optional refined query string
            include_global: Whether to include global documents
            
        Returns:
            Dictionary containing simplified retrieval plan
        """
        try:
            plan = {
                'original_question': question,
                'refined_query': refined_query or question,
                'session_id': session_id,
                'include_global': include_global,
                'search_strategies': [],
                'dual_query_enabled': False  # Simplified approach
            }
            
            # Primary strategy: Enhanced vector search with session + global combination
            plan['search_strategies'].append({
                'type': 'vector_search',
                'query': refined_query or question,
                'k': 15,  # Increased from 10 to get more results for better coverage
                'rerank': True,
                'description': 'Enhanced vector search combining session-specific and global documents'
            })
            
            # Check if this is a PSAK219-related question
            is_psak219_question = any(keyword in question.lower() for keyword in 
                  ['psak', 'aktuaria', 'current service cost', 'benefit', 'karyawan', 
                   'actuarial', 'pension', 'employee', 'service cost', 'liability'])
            
            if is_psak219_question:
                # Add specialized PSAK219 search strategy
                plan['search_strategies'].append({
                    'type': 'psak219_search',
                    'query': refined_query or question,
                    'k': 5,
                    'rerank': False,
                    'description': 'PSAK219-specific document search'
                })
                logger.info("Added PSAK219 search strategy for actuarial question")
            
            # Check for session-specific documents
            session_doc_count = 0
            if self.document_session_service:
                try:
                    session_docs = self.document_session_service.list_documents_for_session(session_id)
                    session_doc_count = len(session_docs) if session_docs else 0
                    
                    if session_doc_count > 0:
                        logger.info(f"Found {session_doc_count} session-specific documents")
                        # Additional session-focused search for high document count sessions
                        if session_doc_count > 3:
                            plan['search_strategies'].append({
                                 'type': 'session_search',
                                 'query': refined_query or question,
                                 'k': 4,
                                 'rerank': False,
                                 'description': 'Additional session-specific search'
                             })
                except Exception as e:
                    logger.warning(f"Could not check session documents: {e}")
            
            plan['session_document_count'] = session_doc_count
            plan['is_psak219_question'] = is_psak219_question
            
            logger.info(f"Generated comprehensive retrieval plan with {len(plan['search_strategies'])} strategies")
            logger.info(f"Session documents: {session_doc_count}, PSAK219 question: {is_psak219_question}")
            return plan
            
        except Exception as e:
            logger.error(f"Error generating retrieval plan: {str(e)}")
            return {
                'original_question': question,
                'refined_query': refined_query or question,
                'session_id': session_id,
                'include_global': include_global,
                'search_strategies': [{
                    'type': 'hybrid_search',
                    'query': refined_query or question,
                    'k': 5,
                    'rerank': False
                }],
                'error': str(e)
            }
    
    def retrieve_documents(self, plan: Dict[str, Any], session_id: str) -> List[Tuple[Document, float]]:
        """Retrieve documents based on the retrieval plan.
        
        Args:
            plan: Retrieval plan dictionary
            session_id: Session identifier
            
        Returns:
            List of tuples containing (Document, score)
        """
        try:
            # Start performance monitoring
            self.performance_monitor.start_timer("retrieval_total")
            
            all_docs = []
            seen_content = set()
            
            for strategy in plan.get('search_strategies', []):
                try:
                    # Monitor each search strategy
                    strategy_type = strategy.get('type', 'unknown')
                    self.performance_monitor.start_timer(f"search_strategy_{strategy_type}")
                    
                    docs = self._execute_search_strategy(strategy, session_id)
                    
                    self.performance_monitor.end_timer(f"search_strategy_{strategy_type}")
                    
                    # Deduplicate documents
                    for doc, score in docs:
                        content_hash = hash(doc.page_content[:200])  # Use first 200 chars for dedup
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            all_docs.append((doc, score))
                            
                except Exception as e:
                    self.performance_monitor.end_timer(f"search_strategy_{strategy_type}")
                    logger.error(f"Error executing search strategy {strategy['type']}: {str(e)}")
                    continue
            
            # Sort by score (higher is better)
            all_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Dynamic retrieval: limit based on configuration
            from app.config import config
            
            if config.ENABLE_DYNAMIC_RETRIEVAL:
                # Filter by threshold and apply dynamic limits
                high_quality_docs = [
                    (doc, score) for doc, score in all_docs 
                    if score >= config.DYNAMIC_RETRIEVAL_THRESHOLD
                ]
                
                logger.info(f"Found {len(high_quality_docs)} documents above threshold {config.DYNAMIC_RETRIEVAL_THRESHOLD} out of {len(all_docs)} total")
                
                if len(high_quality_docs) >= config.DYNAMIC_RETRIEVAL_MIN_DOCS:
                    # Use dynamic retrieval: return all high-quality docs up to max limit
                    final_docs = high_quality_docs[:config.DYNAMIC_RETRIEVAL_MAX_DOCS]
                    logger.info(f"Dynamic retrieval: returning {len(final_docs)} documents above threshold {config.DYNAMIC_RETRIEVAL_THRESHOLD}")
                else:
                    # Fallback to traditional k-based limiting - ensure we return at least some documents
                    max_docs = max(10, config.DYNAMIC_RETRIEVAL_MIN_DOCS)  # Ensure minimum documents
                    final_docs = all_docs[:max_docs]
                    logger.info(f"Dynamic retrieval fallback: returning top {len(final_docs)} documents (below minimum threshold)")
            else:
                # Traditional approach: limit to top results
                max_docs = 10
                final_docs = all_docs[:max_docs]
                logger.info(f"Traditional retrieval: returning top {len(final_docs)} documents")
            
            self.performance_monitor.end_timer("retrieval_total")
            
            logger.info(f"Retrieved {len(final_docs)} documents from {len(plan.get('search_strategies', []))} strategies")
            return final_docs
            
        except Exception as e:
            self.performance_monitor.end_timer("retrieval_total")
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def _execute_search_strategy(
        self, 
        strategy: Dict[str, Any], 
        query: str, 
        session_id: str = None
    ) -> List[Tuple[Document, float]]:
        """
        Execute search strategy with optimized batch processing.
        
        Args:
            strategy: Search strategy configuration
            query: Search query
            session_id: Session identifier
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            strategy_type = strategy.get('type', 'vector_search')
            k = strategy.get('k', 10)
            
            logger.info(f"Executing optimized {strategy_type} strategy with k={k}")
            
            # Use optimized search manager for better performance
            if hasattr(self.vector_store_manager, 'search_manager'):
                search_manager = self.vector_store_manager.search_manager
                
                # Check if optimized search is available
                if hasattr(search_manager, 'search_documents'):
                    logger.info("Using optimized batch search from SearchManager")
                    
                    # Convert to expected format
                    use_hybrid = strategy_type == 'hybrid_search'
                    formatted_results = search_manager.search_documents(
                        query=query,
                        limit=k,
                        session_id=session_id,
                        use_hybrid=use_hybrid,
                        enable_batch_optimization=True
                    )
                    
                    # Convert back to (Document, score) tuples
                    results = []
                    for item in formatted_results:
                        doc = Document(
                            page_content=item.get('content', ''),
                            metadata=item.get('metadata', {})
                        )
                        score = item.get('score', 0.0)
                        results.append((doc, score))
                    
                    logger.info(f"Optimized search returned {len(results)} results")
                    return results
            
            # Fallback to original implementation
            logger.info("Using standard search implementation")
            return self._execute_standard_search_strategy(strategy, query, session_id)
            
        except Exception as e:
            logger.error(f"Error in optimized search strategy execution: {e}")
            # Fallback to standard implementation
            return self._execute_standard_search_strategy(strategy, query, session_id)

    def _execute_standard_search_strategy(
        self, 
        strategy: Dict[str, Any], 
        query: str, 
        session_id: str = None
    ) -> List[Tuple[Document, float]]:
        """
        Execute standard search strategy (original implementation).
        
        Args:
            strategy: Search strategy configuration
            query: Search query
            session_id: Session identifier
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            strategy_type = strategy.get('type', 'vector_search')
            k = strategy.get('k', 10)
            
            if strategy_type == 'vector_search':
                # Enhanced vector search with two-step approach
                logger.info(f"Executing enhanced vector search for session {session_id}")
                
                # Step 1: Search session-specific documents
                session_results = self.vector_store_manager.similarity_search_with_score(
                    query=query,
                    session_id=session_id,
                    k=k,
                    session_required=True,  # Only session documents
                    allow_fallback_to_global=False,  # No fallback in first step
                    return_placeholder_on_empty=False  # Don't return placeholder for fallback logic
                )
                
                logger.info(f"Session-specific search returned {len(session_results)} documents")
                
                # Step 2: Always search global documents to enrich results
                # Calculate remaining slots for global documents
                remaining_k = max(k - len(session_results), k // 2)  # At least half of k for global
                
                if remaining_k > 0:
                    logger.info(f"Searching global documents for additional context (k={remaining_k})")
                    global_results = self.vector_store_manager.similarity_search_with_score(
                        query=query,
                        session_id="global",  # Explicitly search global
                        k=remaining_k,
                        session_required=True,  # Only global documents
                        allow_fallback_to_global=False,  # No fallback needed
                        return_placeholder_on_empty=False  # Don't return placeholder
                    )
                    logger.info(f"Global search returned {len(global_results)} documents")
                    
                    # Combine and sort by relevance score (lower is better)
                    combined_results = session_results + global_results
                    combined_results.sort(key=lambda x: x[1])  # Sort by score (ascending)
                    
                    # Limit to requested k and log final count
                    final_results = combined_results[:k]
                    logger.info(f"Combined search returned {len(final_results)} total documents "
                              f"({len(session_results)} session + {len(global_results)} global)")
                    return final_results
                
                # Fallback: return session results if no global search needed
                return session_results
            
            elif strategy_type == 'hybrid_search':
                # Enhanced hybrid search with two-step approach (same as vector_search)
                logger.info(f"Executing enhanced hybrid search for session {session_id}")
                
                # Step 1: Search session-specific documents
                session_results = self.vector_store_manager.hybrid_similarity_search_with_score(
                    query=query,
                    session_id=session_id,
                    k=k,
                    session_required=True,  # Only session documents
                    allow_fallback_to_global=False,  # No fallback in first step
                    return_placeholder_on_empty=False,  # Don't return placeholder for fallback logic
                    use_hybrid=True
                )
                
                logger.info(f"Session-specific hybrid search returned {len(session_results)} documents")
                
                # Step 2: Always search global documents to enrich results
                # Calculate remaining slots for global documents
                remaining_k = max(k - len(session_results), k // 2)  # At least half of k for global
                
                if remaining_k > 0:
                    logger.info(f"Searching global documents with hybrid for additional context (k={remaining_k})")
                    global_results = self.vector_store_manager.hybrid_similarity_search_with_score(
                        query=query,
                        session_id="global",  # Explicitly search global
                        k=remaining_k,
                        session_required=True,  # Only global documents
                        allow_fallback_to_global=False,  # No fallback needed
                        return_placeholder_on_empty=False,  # Don't return placeholder
                        use_hybrid=True
                    )
                    logger.info(f"Global hybrid search returned {len(global_results)} documents")
                    
                    # Combine and sort by relevance score (lower is better)
                    combined_results = session_results + global_results
                    combined_results.sort(key=lambda x: x[1])  # Sort by score (ascending)
                    
                    # Limit to requested k and log final count
                    final_results = combined_results[:k]
                    logger.info(f"Combined hybrid search returned {len(final_results)} total documents "
                              f"({len(session_results)} session + {len(global_results)} global)")
                    return final_results
                
                # Fallback: return session results if no global search needed
                return session_results
            
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}, using vector search")
                return self.vector_store_manager.similarity_search_with_score(
                    query=query,
                    session_id=session_id,
                    k=k
                )
                
        except Exception as e:
            logger.error(f"Error executing search strategy {strategy_type}: {e}")
            return []

    def prepare_context(self, relevant_docs: List[Tuple[Document, float]]) -> str:
        """Prepare context string from relevant documents.
        
        Args:
            relevant_docs: List of tuples containing (Document, score)
            
        Returns:
            Formatted context string
        """
        if not relevant_docs:
            return "Tidak ada dokumen relevan yang ditemukan."
        
        try:
            context_parts = []
            
            for i, (doc, score) in enumerate(relevant_docs, 1):
                # Get document metadata
                metadata = doc.metadata or {}
                source = metadata.get('source', 'Unknown')
                
                # Format document content
                content = doc.page_content.strip()
                if len(content) > 1000:
                    content = content[:1000] + "..."
                
                # Add document to context
                context_part = f"[Dokumen {i}] (Sumber: {source}, Skor: {score:.3f})\n{content}\n"
                context_parts.append(context_part)
            
            context = "\n".join(context_parts)
            
            logger.info(f"Prepared context from {len(relevant_docs)} documents ({len(context)} characters)")
            return context
            
        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            return "Terjadi kesalahan dalam mempersiapkan konteks dokumen."
    
    def get_chat_history_string(self, session_memories: Dict[str, Any], session_id: str) -> str:
        """Get chat history as a formatted string.
        
        Args:
            session_memories: Dictionary of session memories
            session_id: Session identifier
            
        Returns:
            Formatted chat history string
        """
        try:
            if session_id not in session_memories:
                return "Tidak ada riwayat percakapan."
            
            memory = session_memories[session_id]
            messages = memory.chat_memory.messages
            
            if not messages:
                return "Tidak ada riwayat percakapan."
            
            # Format recent messages (last 10)
            recent_messages = messages[-10:]
            history_parts = []
            
            for msg in recent_messages:
                if hasattr(msg, 'type'):
                    if msg.type == 'human':
                        history_parts.append(f"User: {msg.content}")
                    elif msg.type == 'ai':
                        history_parts.append(f"Assistant: {msg.content}")
                else:
                    # Fallback for different message formats
                    history_parts.append(f"Message: {str(msg)}")
            
            history = "\n".join(history_parts)
            
            logger.info(f"Retrieved chat history for session {session_id} ({len(recent_messages)} messages)")
            return history
            
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return "Terjadi kesalahan dalam mengambil riwayat percakapan."
    
    def _is_psak219_relevant(self, question: str) -> bool:
        """Check if question is relevant to PSAK219.
        
        Args:
            question: Question to check
            
        Returns:
            True if PSAK219 relevant, False otherwise
        """
        psak219_keywords = [
            'psak219', 'psak 219', 'imbalan kerja', 'employee benefit',
            'actuarial', 'aktuaria', 'pension', 'pensiun', 'benefit',
            'kewajiban', 'obligation', 'valuation', 'valuasi'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in psak219_keywords)
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get statistics about retrieval operations.
        
        Returns:
            Dictionary containing retrieval statistics
        """
        # This could be expanded to track actual statistics
        return {
            'vector_store_available': self.vector_store_manager is not None,
            'document_service_available': self.document_session_service is not None,
            'supported_strategies': ['hybrid_search', 'session_search', 'psak219_search', 'actuarial_search', 'standard_search']
        }