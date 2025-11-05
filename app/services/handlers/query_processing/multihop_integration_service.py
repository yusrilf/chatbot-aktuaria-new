"""Multihop Integration Service.

This module integrates multihop retrieval functionality with existing
chat services and provides a unified interface for multihop operations.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel

# Import multihop components
from .multihop_retrieval_manager import MultihopRetrievalManager
from .context_synthesizer import ContextSynthesizer
from .query_refinement_service import QueryRefinementService

logger = logging.getLogger(__name__)

class MultihopIntegrationService:
    """Service that integrates multihop retrieval with existing systems.
    
    This service provides a unified interface for multihop retrieval
    and seamlessly integrates with existing chat and retrieval services.
    """
    
    def __init__(self,
                 retrieval_manager,
                 query_expansion_service,
                 llm: BaseLanguageModel,
                 enable_multihop: bool = True,
                 multihop_config: Optional[Dict[str, Any]] = None):
        """Initialize MultihopIntegrationService.
        
        Args:
            retrieval_manager: Existing retrieval manager
            query_expansion_service: Existing query expansion service
            llm: Language model for multihop operations
            enable_multihop: Whether to enable multihop retrieval
            multihop_config: Configuration for multihop operations
        """
        self.retrieval_manager = retrieval_manager
        self.query_expansion_service = query_expansion_service
        self.llm = llm
        self.enable_multihop = enable_multihop
        
        # Default multihop configuration
        self.config = {
            'max_hops': 3,
            'confidence_threshold': 0.75,
            'min_docs_per_hop': 3,
            'max_docs_per_hop': 8,
            'max_context_length': 4000,
            'overlap_threshold': 0.8,
            'max_query_length': 100,
            'enable_query_decomposition': True,
            'enable_context_synthesis': True,
            'fallback_to_single_hop': True
        }
        
        # Update with provided config
        if multihop_config:
            self.config.update(multihop_config)
        
        # Initialize multihop components
        self._init_multihop_components()
        
        logger.info(f"MultihopIntegrationService initialized (multihop: {enable_multihop})")
    
    def _init_multihop_components(self) -> None:
        """Initialize multihop retrieval components."""
        try:
            if self.enable_multihop:
                # Initialize MultihopRetrievalManager
                self.multihop_manager = MultihopRetrievalManager(
                    retrieval_manager=self.retrieval_manager,
                    query_expansion_service=self.query_expansion_service,
                    llm=self.llm,
                    max_hops=self.config['max_hops'],
                    confidence_threshold=self.config['confidence_threshold'],
                    min_docs_per_hop=self.config['min_docs_per_hop'],
                    max_docs_per_hop=self.config['max_docs_per_hop']
                )
                
                # Initialize ContextSynthesizer
                if self.config['enable_context_synthesis']:
                    self.context_synthesizer = ContextSynthesizer(
                        llm=self.llm,
                        max_context_length=self.config['max_context_length'],
                        overlap_threshold=self.config['overlap_threshold']
                    )
                else:
                    self.context_synthesizer = None
                
                # Initialize QueryRefinementService
                self.query_refinement = QueryRefinementService(
                    llm=self.llm,
                    query_expansion_service=self.query_expansion_service,
                    max_query_length=self.config['max_query_length']
                )
                
                logger.info("Multihop components initialized successfully")
            else:
                self.multihop_manager = None
                self.context_synthesizer = None
                self.query_refinement = None
                logger.info("Multihop disabled - using single hop retrieval only")
                
        except Exception as e:
            logger.error(f"Error initializing multihop components: {str(e)}")
            # Disable multihop on initialization failure
            self.enable_multihop = False
            self.multihop_manager = None
            self.context_synthesizer = None
            self.query_refinement = None
            
            if self.config['fallback_to_single_hop']:
                logger.warning("Falling back to single hop retrieval")
            else:
                raise
    
    def retrieve_documents(self, 
                          query: str,
                          session_id: str,
                          context_type: str = "aktuaria",
                          force_single_hop: bool = False) -> Dict[str, Any]:
        """Retrieve documents using multihop or single hop strategy.
        
        Args:
            query: User query
            session_id: Session identifier
            context_type: Context type for retrieval
            force_single_hop: Force single hop retrieval
            
        Returns:
            Dictionary containing retrieval results
        """
        start_time = time.time()
        
        try:
            # Determine retrieval strategy
            use_multihop = (
                self.enable_multihop and 
                not force_single_hop and 
                self._should_use_multihop(query)
            )
            
            if use_multihop:
                logger.info(f"Using multihop retrieval for query: {query[:50]}...")
                result = self._perform_multihop_retrieval(
                    query=query,
                    session_id=session_id,
                    context_type=context_type
                )
            else:
                logger.info(f"Using single hop retrieval for query: {query[:50]}...")
                result = self._perform_single_hop_retrieval(
                    query=query,
                    session_id=session_id,
                    context_type=context_type
                )
            
            # Add timing and metadata
            result['total_retrieval_time'] = time.time() - start_time
            result['retrieval_strategy'] = 'multihop' if use_multihop else 'single_hop'
            result['session_id'] = session_id
            
            logger.info(f"Retrieval completed: {result['retrieval_strategy']} "
                       f"({result['total_retrieval_time']:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            
            # Fallback to single hop if multihop fails
            if use_multihop and self.config['fallback_to_single_hop']:
                logger.warning("Multihop failed, falling back to single hop")
                try:
                    return self._perform_single_hop_retrieval(
                        query=query,
                        session_id=session_id,
                        context_type=context_type
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {str(fallback_error)}")
            
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'session_id': session_id,
                'retrieval_strategy': 'failed',
                'total_retrieval_time': time.time() - start_time
            }
    
    def _should_use_multihop(self, query: str) -> bool:
        """Determine if multihop retrieval should be used.
        
        Args:
            query: User query
            
        Returns:
            True if multihop should be used
        """
        if not self.multihop_manager:
            return False
        
        # Criteria for using multihop:
        # 1. Query is complex (multiple concepts)
        # 2. Query asks for comprehensive information
        # 3. Query contains comparison or analysis terms
        
        query_lower = query.lower()
        
        # Check for complexity indicators
        complexity_indicators = [
            'bagaimana', 'mengapa', 'jelaskan', 'bandingkan', 'analisis',
            'perbedaan', 'hubungan', 'pengaruh', 'dampak', 'faktor',
            'komprehensif', 'detail', 'lengkap', 'menyeluruh'
        ]
        
        has_complexity = any(indicator in query_lower for indicator in complexity_indicators)
        
        # Check for multiple concepts
        actuarial_concepts = [
            'mortalitas', 'morbiditas', 'cadangan', 'premi', 'valuasi',
            'kewajiban', 'risiko', 'asuransi', 'pensiun', 'anuitas'
        ]
        
        concept_count = sum(1 for concept in actuarial_concepts if concept in query_lower)
        has_multiple_concepts = concept_count >= 2
        
        # Check query length (longer queries often need multihop)
        is_long_query = len(query.split()) >= 5
        
        # Decision logic
        use_multihop = has_complexity or has_multiple_concepts or is_long_query
        
        logger.debug(f"Multihop decision for '{query[:30]}...': {use_multihop} "
                    f"(complexity: {has_complexity}, concepts: {concept_count}, "
                    f"length: {len(query.split())} words)")
        
        return use_multihop
    
    def _perform_multihop_retrieval(self, 
                                   query: str,
                                   session_id: str,
                                   context_type: str) -> Dict[str, Any]:
        """Perform multihop retrieval.
        
        Args:
            query: User query
            session_id: Session identifier
            context_type: Context type
            
        Returns:
            Multihop retrieval results
        """
        try:
            # Check if query should be decomposed first
            if self.config['enable_query_decomposition'] and len(query.split()) > 8:
                decomposition_result = self.query_refinement.decompose_complex_query(query)
                
                if decomposition_result['success'] and len(decomposition_result['sub_queries']) > 1:
                    # Process sub-queries sequentially
                    return self._process_decomposed_queries(
                        decomposition_result['sub_queries'],
                        session_id,
                        context_type,
                        original_query=query
                    )
            
            # Standard multihop retrieval
            multihop_result = self.multihop_manager.multihop_retrieve(
                query=query,
                session_id=session_id,
                context_type=context_type
            )
            
            if not multihop_result['success']:
                return multihop_result
            
            # Synthesize context if enabled
            if self.context_synthesizer:
                synthesis_result = self.context_synthesizer.synthesize_multihop_context(
                    multihop_result
                )
                
                if synthesis_result['success']:
                    multihop_result['synthesized_context'] = synthesis_result['synthesized_context']
                    multihop_result['context_metadata'] = synthesis_result['context_metadata']
                    multihop_result['synthesis_time'] = synthesis_result['synthesis_time']
            
            return multihop_result
            
        except Exception as e:
            logger.error(f"Error in multihop retrieval: {str(e)}")
            raise
    
    def _perform_single_hop_retrieval(self, 
                                     query: str,
                                     session_id: str,
                                     context_type: str) -> Dict[str, Any]:
        """Perform single hop retrieval using existing system.
        
        Args:
            query: User query
            session_id: Session identifier
            context_type: Context type
            
        Returns:
            Single hop retrieval results
        """
        try:
            # Use existing retrieval manager
            plan = self.retrieval_manager.generate_retrieval_plan(
                question=query,
                session_id=session_id,
                include_global=True
            )
            
            documents = self.retrieval_manager.retrieve_documents(
                plan=plan,
                session_id=session_id
            )
            
            # Format result to match multihop structure
            result = {
                'success': True,
                'original_query': query,
                'total_hops': 1,
                'hop_results': [{
                    'hop_number': 1,
                    'query': query,
                    'documents': documents,
                    'document_count': len(documents),
                    'average_confidence': sum(score for _, score in documents) / len(documents) if documents else 0.0,
                    'retrieval_plan': plan
                }],
                'final_documents': documents,
                'total_documents': len(documents),
                'confidence_score': sum(score for _, score in documents) / len(documents) if documents else 0.0,
                'retrieval_time': 0.0  # Will be set by caller
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single hop retrieval: {str(e)}")
            raise
    
    def _process_decomposed_queries(self, 
                                   sub_queries: List[str],
                                   session_id: str,
                                   context_type: str,
                                   original_query: str) -> Dict[str, Any]:
        """Process decomposed sub-queries.
        
        Args:
            sub_queries: List of sub-queries
            session_id: Session identifier
            context_type: Context type
            original_query: Original complex query
            
        Returns:
            Combined results from sub-queries
        """
        try:
            logger.info(f"Processing {len(sub_queries)} decomposed sub-queries")
            
            all_documents = []
            all_hop_results = []
            seen_content_hashes = set()
            
            for i, sub_query in enumerate(sub_queries, 1):
                logger.info(f"Processing sub-query {i}/{len(sub_queries)}: {sub_query[:50]}...")
                
                # Retrieve for sub-query
                sub_result = self.multihop_manager.multihop_retrieve(
                    original_query=sub_query,
                    session_id=session_id,
                    context_type=context_type
                )
                
                if sub_result['success']:
                    # Add documents with deduplication
                    for doc, score in sub_result['final_documents']:
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content_hashes:
                            seen_content_hashes.add(content_hash)
                            all_documents.append((doc, score))
                    
                    # Add hop results with sub-query context
                    for hop_result in sub_result['hop_results']:
                        hop_result['sub_query_index'] = i
                        hop_result['sub_query'] = sub_query
                        all_hop_results.append(hop_result)
            
            # Sort final documents by score
            all_documents.sort(key=lambda x: x[1], reverse=True)
            final_documents = all_documents[:15]  # Top 15 documents
            
            # Calculate overall confidence
            overall_confidence = sum(score for _, score in final_documents) / len(final_documents) if final_documents else 0.0
            
            result = {
                'success': True,
                'original_query': original_query,
                'sub_queries': sub_queries,
                'total_hops': len(all_hop_results),
                'hop_results': all_hop_results,
                'final_documents': final_documents,
                'total_documents': len(final_documents),
                'confidence_score': overall_confidence,
                'retrieval_time': 0.0,  # Will be set by caller
                'decomposition_used': True
            }
            
            # Synthesize context if enabled
            if self.context_synthesizer:
                synthesis_result = self.context_synthesizer.synthesize_multihop_context(result)
                
                if synthesis_result['success']:
                    result['synthesized_context'] = synthesis_result['synthesized_context']
                    result['context_metadata'] = synthesis_result['context_metadata']
                    result['synthesis_time'] = synthesis_result['synthesis_time']
            
            logger.info(f"Decomposed query processing completed: {len(final_documents)} final documents")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing decomposed queries: {str(e)}")
            raise
    
    def get_retrieval_context(self, 
                             retrieval_result: Dict[str, Any],
                             max_length: int = 4000) -> str:
        """Extract context from retrieval results for response generation.
        
        Args:
            retrieval_result: Results from retrieve_documents
            max_length: Maximum context length
            
        Returns:
            Context string for response generation
        """
        try:
            # Use synthesized context if available
            if 'synthesized_context' in retrieval_result:
                context = retrieval_result['synthesized_context']
                if len(context) <= max_length:
                    return context
                else:
                    return context[:max_length - 3] + "..."
            
            # Fallback to document concatenation
            final_documents = retrieval_result.get('final_documents', [])
            
            if not final_documents:
                return "Tidak ada dokumen relevan ditemukan."
            
            context_parts = []
            current_length = 0
            
            for i, (doc, score) in enumerate(final_documents[:8], 1):  # Top 8 documents
                filename = doc.metadata.get('filename', f'Dokumen {i}')
                content = doc.page_content
                
                # Estimate space needed
                header = f"\n\n[{filename}]\n"
                needed_space = len(header) + len(content)
                
                if current_length + needed_space > max_length:
                    # Add partial content if space allows
                    remaining_space = max_length - current_length - len(header) - 3
                    if remaining_space > 100:  # Minimum useful content
                        partial_content = content[:remaining_space] + "..."
                        context_parts.append(f"{header}{partial_content}")
                    break
                
                context_parts.append(f"{header}{content}")
                current_length += needed_space
            
            return "".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error extracting retrieval context: {str(e)}")
            return f"Error dalam ekstraksi konteks: {str(e)}"
    
    def _perform_single_hop_retrieval(self, query: str, session_id: str, context_type: str) -> Dict[str, Any]:
        """Perform single hop retrieval.
        
        Args:
            query: User query
            session_id: Session identifier
            context_type: Context type
            
        Returns:
            Single hop retrieval results
        """
        try:
            # Use existing retrieval manager
            documents = self.retrieval_manager.retrieve_documents(
                query=query,
                session_id=session_id,
                context_type=context_type
            )
            
            # Format as consistent result structure
            if isinstance(documents, list):
                # Convert to (document, score) tuples if needed
                formatted_docs = []
                for doc in documents:
                    if isinstance(doc, tuple):
                        formatted_docs.append(doc)
                    else:
                        formatted_docs.append((doc, 1.0))  # Default score
                
                return {
                    'success': True,
                    'final_documents': formatted_docs,
                    'total_documents': len(formatted_docs),
                    'hop_count': 1,
                    'retrieval_strategy': 'single_hop'
                }
            else:
                # Handle dictionary response
                return {
                    'success': documents.get('success', True),
                    'final_documents': documents.get('documents', []),
                    'total_documents': len(documents.get('documents', [])),
                    'hop_count': 1,
                    'retrieval_strategy': 'single_hop'
                }
                
        except Exception as e:
            logger.error(f"Single hop retrieval failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'final_documents': [],
                'total_documents': 0,
                'hop_count': 1,
                'retrieval_strategy': 'single_hop'
            }
    
    def _format_single_hop_context(self, retrieval_result: Dict[str, Any]) -> str:
        """Format single hop retrieval results as context.
        
        Args:
            retrieval_result: Single hop retrieval results
            
        Returns:
            Formatted context string
        """
        try:
            documents = retrieval_result.get('final_documents', [])
            
            if not documents:
                return "Tidak ada dokumen relevan ditemukan."
            
            return self._format_documents_as_context(documents)
            
        except Exception as e:
            logger.error(f"Error formatting single hop context: {e}")
            return f"Error dalam format konteks: {str(e)}"
    
    def extract_context_from_results(self, retrieval_result: Dict[str, Any]) -> str:
        """Extract context from retrieval results.
        
        Args:
            retrieval_result: Results from retrieval operation
            
        Returns:
            Extracted context string
        """
        return self.get_retrieval_context(retrieval_result)
    
    def extract_context(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Extract context for a query using appropriate retrieval strategy.
        
        Args:
            query: User query
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing extracted context
        """
        try:
            strategy = self.determine_retrieval_strategy(query)
            
            if strategy == "multihop":
                # Use multihop retrieval
                results = self.retrieve_with_multihop(
                    query=query,
                    session_id=session_id or 'default'
                )
                
                # Format context
                context = self.get_retrieval_context(results)
                
                return {
                    'strategy': 'multihop',
                    'context': context,
                    'hop_count': results.get('hop_count', 1),
                    'total_documents': len(results.get('final_documents', [])),
                    'session_id': session_id
                }
            else:
                # Use single hop retrieval
                documents = self._perform_single_hop_retrieval(query, session_id or 'default', 'aktuaria')
                
                return {
                    'strategy': 'single_hop',
                    'context': self._format_single_hop_context(documents),
                    'hop_count': 1,
                    'total_documents': len(documents.get('final_documents', [])),
                    'session_id': session_id
                }
                
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return {
                'strategy': 'error',
                'context': 'Error occurred during context extraction',
                'error': str(e),
                'session_id': session_id
            }
    
    def determine_retrieval_strategy(self, query: str) -> str:
        """Determine which retrieval strategy to use.
        
        Args:
            query: User query
            
        Returns:
            Strategy name ('multihop' or 'single_hop')
        """
        if self._should_use_multihop(query):
            return "multihop"
        else:
            return "single_hop"
    
    def extract_context_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context from retrieval result.
        
        Args:
            result: Retrieval result dictionary
            
        Returns:
            Dictionary containing extracted context
        """
        try:
            documents = result.get('documents', [])
            
            # Extract text content
            context_texts = []
            document_sources = []
            key_topics = set()
            
            for doc_item in documents:
                if isinstance(doc_item, tuple):
                    doc, score = doc_item
                else:
                    doc = doc_item
                    score = 1.0
                    
                context_texts.append(doc.page_content)
                
                # Format score untuk readability - hindari scientific notation
                formatted_score = float(f"{score:.6f}") if score < 0.001 else round(score, 4)
                
                document_sources.append({
                    'filename': doc.metadata.get('filename', 'unknown'),
                    'page': doc.metadata.get('page', 1),
                    'document_type': doc.metadata.get('document_type', 'unknown'),
                    'score': formatted_score
                })
                
                # Extract topics
                if 'topic' in doc.metadata:
                    key_topics.add(doc.metadata['topic'])
            
            # Combine context
            context_text = '\n\n'.join(context_texts)
            
            # Calculate confidence score
            confidence_score = result.get('overall_confidence', 0.0)
            if not confidence_score and documents:
                # Calculate from document scores
                scores = [item[1] if isinstance(item, tuple) else 1.0 for item in documents]
                confidence_score = sum(scores) / len(scores) if scores else 0.0
            
            return {
                'context_text': context_text,
                'key_topics': list(key_topics),
                'document_sources': document_sources,
                'confidence_score': confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return {
                'context_text': '',
                'key_topics': [],
                'document_sources': [],
                'confidence_score': 0.0
            }
    
    def _filter_documents_by_type(self, documents: List, document_types: List[str]) -> List:
        """Filter documents by specified types.
        
        Args:
            documents: List of documents (tuples or Document objects)
            document_types: List of allowed document types
            
        Returns:
            Filtered list of documents
        """
        try:
            filtered_docs = []
            
            for doc_item in documents:
                if isinstance(doc_item, tuple):
                    doc, score = doc_item
                else:
                    doc = doc_item
                    score = 1.0
                
                doc_type = doc.metadata.get('document_type')
                if doc_type in document_types:
                    filtered_docs.append((doc, score) if isinstance(doc_item, tuple) else doc)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error filtering documents: {e}")
            return documents
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get statistics about integration service.
        
        Returns:
            Dictionary containing statistics
        """
        stats = {
            'multihop_enabled': self.enable_multihop,
            'config': self.config.copy(),
            'components': {
                'multihop_manager': self.multihop_manager is not None,
                'context_synthesizer': self.context_synthesizer is not None,
                'query_refinement': self.query_refinement is not None
            }
        }
        
        # Add component-specific statistics
        if self.multihop_manager:
            stats['multihop_manager_stats'] = self.multihop_manager.get_multihop_statistics()
        
        if self.context_synthesizer:
            stats['context_synthesizer_stats'] = self.context_synthesizer.get_synthesis_statistics()
        
        if self.query_refinement:
            stats['query_refinement_stats'] = self.query_refinement.get_refinement_statistics()
        
        return stats
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update multihop configuration.
        
        Args:
            new_config: New configuration values
        """
        self.config.update(new_config)
        logger.info(f"Multihop configuration updated: {new_config}")
        
        # Reinitialize components if necessary
        if self.enable_multihop:
            self._init_multihop_components()
    
    def enable_multihop_retrieval(self) -> None:
        """Enable multihop retrieval."""
        if not self.enable_multihop:
            self.enable_multihop = True
            self._init_multihop_components()
            logger.info("Multihop retrieval enabled")
    
    def disable_multihop_retrieval(self) -> None:
        """Disable multihop retrieval."""
        if self.enable_multihop:
            self.enable_multihop = False
            self.multihop_manager = None
            self.context_synthesizer = None
            self.query_refinement = None
            logger.info("Multihop retrieval disabled")
    
    def retrieve_with_multihop(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform multihop retrieval for a query.
        
        Args:
            query: User query
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing retrieval results
        """
        try:
            # Extract parameters early to avoid reference errors
            session_id = kwargs.get('session_id', 'default')
            context_type = kwargs.get('context_type', 'aktuaria')
            max_hops = kwargs.get('max_hops', self.config.get('max_hops', 3))
            max_docs_per_hop = kwargs.get('max_docs_per_hop', self.config.get('max_docs_per_hop', 5))
            
            # Force multihop for testing or if explicitly requested
            force_multihop = kwargs.get('force_multihop', False)
            
            # Check if multihop should be used
            should_use_multihop = force_multihop or (self.enable_multihop and self.multihop_manager and self._should_use_multihop(query))
            
            if not should_use_multihop:
                 logger.info("Falling back to single hop retrieval")
                 return self._perform_single_hop_fallback(
                     query=query,
                     session_id=session_id,
                     context_type=context_type,
                     **kwargs
                 )
            
            results = self.multihop_manager.multihop_retrieve(
                query=query,
                session_id=session_id,
                context_type=context_type
            )
            
            # Filter documents by type if specified
            documents = results.get('documents', [])
            if 'document_types' in kwargs and kwargs['document_types']:
                documents = self._filter_documents_by_type(documents, kwargs['document_types'])
                results['documents'] = documents
            
            # Add metadata and ensure consistent structure
            results['multihop_enabled'] = True
            results['config'] = {
                'max_hops': max_hops,
                'max_docs_per_hop': max_docs_per_hop
            }
            
            # Ensure 'documents' field exists for backward compatibility
            if 'final_documents' in results and 'documents' not in results:
                results['documents'] = results['final_documents']
            
            # Add strategy metadata
            results['strategy_used'] = 'multihop'
            results['execution_metadata'] = {
                'total_hops': results.get('total_hops', 1),
                'overall_confidence': results.get('overall_confidence', 0.0),
                'execution_time': results.get('execution_time', 0.0),
                'session_id': session_id,
                'multihop_enabled': True,
                'fallback_used': False,
                'config': results['config']
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multihop retrieval: {e}")
            # Prepare kwargs without session_id and context_type to avoid duplication
            fallback_kwargs = {k: v for k, v in kwargs.items() if k not in ['session_id', 'context_type']}
            return self._perform_single_hop_fallback(
                query=query,
                session_id=session_id,
                context_type=context_type,
                **fallback_kwargs
            )
    
    def _perform_single_hop_fallback(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform single hop retrieval as fallback.
        
        Args:
            query: User query
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing single hop results
        """
        try:
            session_id = kwargs.get('session_id', 'fallback')
            context_type = kwargs.get('context_type', 'aktuaria')
            
            result = self._perform_single_hop_retrieval(
                query=query,
                session_id=session_id,
                context_type=context_type
            )
            
            # Filter documents by type if specified
            documents = result.get('documents', [])
            if 'document_types' in kwargs and kwargs['document_types']:
                documents = self._filter_documents_by_type(documents, kwargs['document_types'])
                result['documents'] = documents
            
            result['fallback_used'] = True
            result['strategy_used'] = 'single_hop'
            
            # Ensure 'documents' field exists for backward compatibility
            if 'final_documents' in result and 'documents' not in result:
                result['documents'] = result['final_documents']
            
            result['execution_metadata'] = {
                'total_hops': 1,
                'overall_confidence': result.get('confidence', 0.0),
                'execution_time': result.get('execution_time', 0.0),
                'session_id': session_id,
                'multihop_enabled': False,
                'fallback_used': True,
                'config': {}
            }
            
            return result
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'documents': [],
                'fallback_used': True,
                'strategy_used': 'single_hop',
                'execution_metadata': {
                    'total_hops': 0,
                    'overall_confidence': 0.0,
                    'execution_time': 0.0,
                    'session_id': kwargs.get('session_id', 'fallback'),
                    'multihop_enabled': False,
                    'fallback_used': True,
                    'config': {}
                },
                'original_query': query
            }
    
    def process_decomposed_query(self, query: str, decomposed_queries: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Process a list of decomposed queries.
        
        Args:
            decomposed_queries: List of sub-queries
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing processed results
        """
        try:
            # If no decomposed queries provided, decompose the original query
            if not decomposed_queries:
                # Enhanced decomposition logic
                import re
                
                # First try splitting on conjunctions
                decomposed_queries = re.split(r'\s+(?:dan|atau|serta|dengan|untuk)\s+', query.lower())
                decomposed_queries = [q.strip() for q in decomposed_queries if q.strip()]
                
                # If still only one query, try splitting on actuarial terms
                if len(decomposed_queries) <= 1:
                    actuarial_terms = ['mortalitas', 'morbiditas', 'premi', 'cadangan', 'teknis', 'valuasi', 'asuransi', 'pensiun']
                    found_terms = [term for term in actuarial_terms if term in query.lower()]
                    
                    if len(found_terms) > 1:
                        # Create sub-queries for each found term
                        decomposed_queries = [f"{term} aktuaria" for term in found_terms]
                    else:
                        decomposed_queries = [query]  # Fallback to original query
            
            all_results = []
            all_documents = []
            session_id = kwargs.get('session_id', 'decomposed')
            context_type = kwargs.get('context_type', 'aktuaria')
            
            for i, sub_query in enumerate(decomposed_queries):
                logger.info(f"Processing sub-query {i+1}/{len(decomposed_queries)}: {sub_query}")
                
                # Prepare kwargs without session_id and context_type to avoid duplication
                sub_kwargs = {k: v for k, v in kwargs.items() if k not in ['session_id', 'context_type']}
                
                # Retrieve for each sub-query
                sub_results = self.retrieve_with_multihop(
                    query=sub_query,
                    session_id=f"{session_id}_sub_{i+1}",
                    context_type=context_type,
                    **sub_kwargs
                )
                
                all_results.append({
                    'query': sub_query,
                    'results': sub_results
                })
                
                # Collect documents
                if sub_results.get('success') and 'final_documents' in sub_results:
                    all_documents.extend(sub_results['final_documents'])
            
            # Deduplicate documents
            unique_documents = self._deduplicate_documents(all_documents)
            
            # Synthesize final context if synthesizer is available
            combined_query = " ".join(decomposed_queries)
            final_context = ""
            
            if self.context_synthesizer and unique_documents:
                try:
                    synthesis_result = self.context_synthesizer.synthesize_multihop_context({
                        'original_query': combined_query,
                        'final_documents': unique_documents,
                        'hop_results': all_results
                    })
                    
                    if synthesis_result.get('success'):
                        final_context = synthesis_result.get('synthesized_context', '')
                except Exception as e:
                    logger.warning(f"Context synthesis failed: {e}")
                    final_context = self._format_documents_as_context(unique_documents)
            else:
                final_context = self._format_documents_as_context(unique_documents)
            
            # Format synthesis result
            synthesis_result = {
                'synthesized_context': final_context,
                'key_topics': [],
                'confidence_score': 0.8
            }
            
            # Extract key topics from documents
            key_topics = set()
            for doc_item in unique_documents:
                if isinstance(doc_item, tuple):
                    doc, score = doc_item
                else:
                    doc = doc_item
                if 'topic' in doc.metadata:
                    key_topics.add(doc.metadata['topic'])
            
            synthesis_result['key_topics'] = list(key_topics)
            
            return {
                'success': True,
                'sub_query_results': all_results,
                'combined_documents': unique_documents,
                'synthesis_result': synthesis_result,
                'total_sub_queries': len(decomposed_queries),
                'total_unique_documents': len(unique_documents),
                # Keep backward compatibility
                'decomposed_queries': decomposed_queries,
                'sub_results': all_results,
                'final_documents': unique_documents,
                'final_context': final_context,
                'total_documents': len(unique_documents)
            }
            
        except Exception as e:
            logger.error(f"Error processing decomposed query: {e}")
            return {
                'success': False,
                'error': str(e),
                'sub_query_results': [],
                'combined_documents': [],
                'synthesis_result': {
                    'synthesized_context': 'Error occurred during query processing',
                    'key_topics': [],
                    'confidence_score': 0.0
                },
                'total_sub_queries': len(decomposed_queries) if decomposed_queries else 0,
                'total_unique_documents': 0,
                # Keep backward compatibility
                'decomposed_queries': decomposed_queries,
                'sub_results': [],
                'final_documents': [],
                'final_context': 'Error occurred during query processing'
            }
    
    def _deduplicate_documents(self, documents: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Remove duplicate documents based on content similarity.
        
        Args:
            documents: List of (document, score) tuples
            
        Returns:
            Deduplicated list of documents
        """
        if not documents:
            return []
        
        unique_docs = []
        seen_hashes = set()
        
        for doc, score in documents:
            # Create hash from first 200 characters of content
            content_hash = hash(doc.page_content[:200])
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append((doc, score))
        
        # Sort by score descending
        unique_docs.sort(key=lambda x: x[1], reverse=True)
        
        return unique_docs
    
    def _format_documents_as_context(self, documents: List[Tuple[Document, float]]) -> str:
        """Format documents as context string.
        
        Args:
            documents: List of (document, score) tuples
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "Tidak ada dokumen relevan ditemukan."
        
        context_parts = []
        
        for i, (doc, score) in enumerate(documents[:8], 1):  # Top 8 documents
            filename = doc.metadata.get('filename', f'Dokumen {i}')
            content = doc.page_content
            
            context_parts.append(f"\n\n[{filename}]\n{content}")
        
        return "".join(context_parts)