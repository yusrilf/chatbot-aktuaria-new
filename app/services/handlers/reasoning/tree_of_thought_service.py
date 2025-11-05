"""Tree of Thought Service - Main Orchestrator.

Service utama yang mengintegrasikan Query Expansion, Multi-Path Reasoning,
dan Path Evaluation untuk implementasi Tree of Thought dalam RAG pipeline.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.llms.base import LLM
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import komponen ToT
from .query_expansion_service import QueryExpansionService
from .multipath_generator import MultipathReasoningGenerator
from .evaluation_layer import EvaluationLayer
from .validation_layer import ValidationLayer
from .tot_document_reader import ToTDocumentReader

# Import performance monitoring
from app.config import config
from app.utils.performance_monitor import get_global_monitor, performance_timer

logger = logging.getLogger(__name__)

class TreeOfThoughtService:
    """Main orchestrator untuk Tree of Thought implementation."""
    
    def __init__(self, 
                 llm: LLM,
                 search_manager: Any = None,
                 enable_parallel_processing: bool = True):
        """
        Initialize Tree of Thought Service.
        
        Args:
            llm: Language model untuk semua komponen ToT
            search_manager: Search manager untuk retrieval dokumen
            enable_parallel_processing: Enable parallel processing untuk performance
        """
        self.llm = llm
        self.search_manager = search_manager
        self.enable_parallel_processing = enable_parallel_processing
        
        # Initialize performance monitor
        self.performance_monitor = get_global_monitor()
        
        # Initialize komponen ToT
        self._init_tot_components()
        
        # Configuration
        self.config = {
            "max_expanded_queries": 4,
            "max_reasoning_paths": 4,
            "min_confidence_threshold": 60,
            "enable_query_expansion": True,
            "enable_multipath_reasoning": True,
            "enable_batch_rag": True,  # Enable batch RAG processing
            "batch_rag_timeout": 15,   # Timeout for batch operations
            "parallel_workers": 4,     # Max parallel workers
            "retrieval_limit_per_query": 3,  # Limit documents per query
            "max_combined_query_length": 500,  # Max length for combined queries
        }
        
        logger.info("Tree of Thought Service initialized successfully")
    
    def _init_tot_components(self) -> None:
        """Initialize semua komponen Tree of Thought."""
        try:
            self.query_expander = QueryExpansionService(self.llm)
            self.multipath_generator = MultipathReasoningGenerator(self.llm)
            self.path_evaluator = EvaluationLayer(self.llm)
            self.validation_layer = ValidationLayer()
            self.tot_document_reader = ToTDocumentReader()
            
            logger.info("All ToT components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ToT components: {e}")
            raise
    
    def process_with_tree_of_thought(self, 
                                   question: str,
                                   context: str = "",
                                   chat_history: str = "",
                                   session_id: str = "") -> Dict[str, Any]:
        """
        Main method untuk memproses pertanyaan dengan Tree of Thought.
        
        Args:
            question: Pertanyaan dari user
            context: Konteks dokumen yang sudah diretrieve
            chat_history: Riwayat percakapan
            session_id: ID session untuk tracking
            
        Returns:
            Dict berisi hasil lengkap Tree of Thought processing
        """
        # Start timing for total ToT processing
        tot_timer = self.performance_monitor.start_timer("tot_processing_with_tree_of_thought")
        
        try:
            logger.info(f"Processing question with ToT: {question[:100]}...")
            
            # Step 1: Query Expansion (jika enabled)
            expansion_timer = self.performance_monitor.start_timer("tot_query_expansion_step")
            expanded_queries = self._expand_query_if_enabled(question)
            self.performance_monitor.stop_timer(expansion_timer)
            
            # Step 2: Enhanced Context Retrieval (jika ada search_manager)
            context_timer = self.performance_monitor.start_timer("tot_context_enhancement")
            enhanced_context = self._enhance_context_retrieval(expanded_queries, context)
            self.performance_monitor.stop_timer(context_timer)
            
            # Step 3: Multi-Path Reasoning Generation
            reasoning_timer = self.performance_monitor.start_timer("tot_reasoning_generation")
            reasoning_paths = self._generate_reasoning_paths(question, enhanced_context, chat_history)
            self.performance_monitor.stop_timer(reasoning_timer)
            
            # Step 4: Path Evaluation & Selection
            evaluation_timer = self.performance_monitor.start_timer("tot_path_evaluation_step")
            evaluation_result = self._evaluate_and_select_best_path(question, reasoning_paths)
            self.performance_monitor.stop_timer(evaluation_timer)
            
            # Step 5: Prepare Final Result
            final_result = self._prepare_final_result(
                question=question,
                expanded_queries=expanded_queries,
                reasoning_paths=reasoning_paths,
                evaluation_result=evaluation_result,
                session_id=session_id
            )
            
            # Stop total timer and add performance metrics
            self.performance_monitor.stop_timer(tot_timer)
            
            # Add performance metrics only if enabled in config
            if config.PERFORMANCE_METRICS_IN_RESPONSE:
                final_result["performance_metrics"] = self.performance_monitor.get_metrics()
            
            logger.info(f"ToT processing completed successfully for session {session_id}")
            return final_result
            
        except Exception as e:
            # Stop timer on error
            self.performance_monitor.stop_timer(tot_timer)
            logger.error(f"Error in ToT processing: {e}")
            
            # Return fallback result with error info
            fallback_result = self._get_fallback_result(question, context, chat_history)
            fallback_result["error"] = str(e)
            
            # Add performance metrics only if enabled in config
            if config.PERFORMANCE_METRICS_IN_RESPONSE:
                fallback_result["performance_metrics"] = self.performance_monitor.get_metrics()
            return fallback_result
    
    def _expand_query_if_enabled(self, question: str) -> Dict[str, Any]:
        """Expand query jika feature enabled."""
        if not self.config["enable_query_expansion"]:
            return {
                "success": False,
                "expanded_queries": [question],
                "total_queries": 1,
                "expansion_disabled": True
            }
        
        try:
            return self.query_expander.expand_query(question)
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return {
                "success": False,
                "expanded_queries": [question],
                "total_queries": 1,
                "expansion_failed": True
            }
    
    @performance_timer("tot_enhance_context_retrieval")
    def _enhance_context_retrieval(self, 
                                 expanded_queries: Dict[str, Any], 
                                 original_context: str) -> str:
        """Enhance context retrieval menggunakan batch RAG untuk semua sinonim sekaligus."""
        if not self.search_manager or not expanded_queries.get("success", False):
            return original_context
        
        # Start timing for context enhancement
        enhancement_timer = self.performance_monitor.start_timer("tot_context_enhancement_processing")
        
        try:
            queries = expanded_queries.get("expanded_queries", [])
            
            # Start timing for ToT document reader
            tot_reader_timer = self.performance_monitor.start_timer("tot_document_reader")
            
            # Use ToT Document Reader for enhanced retrieval
            enhanced_result = self.tot_document_reader.process_question_with_tot(
                question=queries[0] if queries else "",
                session_id="tot_enhancement",
                expanded_queries=queries,
                existing_context=original_context
            )
            
            # Stop ToT reader timer
            self.performance_monitor.stop_timer(tot_reader_timer)
            
            if enhanced_result and enhanced_result.get('success', False):
                self.performance_monitor.stop_timer(enhancement_timer)
                return enhanced_result.get('enhanced_context', original_context)
            
            # OPTIMIZED: Batch RAG approach - combine all synonyms into one search
            try:
                batch_rag_timer = self.performance_monitor.start_timer("batch_rag_retrieval")
                
                # Method 1: Smart query combination for batch RAG
                batch_context = self._batch_rag_retrieval(queries, original_context)
                
                self.performance_monitor.stop_timer(batch_rag_timer)
                
                if batch_context != original_context:
                    logger.info(f"Batch RAG successful with {len(queries)} queries")
                    self.performance_monitor.stop_timer(enhancement_timer)
                    return batch_context
            except Exception as e:
                self.performance_monitor.stop_timer(batch_rag_timer)
                logger.warning(f"Batch RAG failed: {e}")
    
            # Method 2: Parallel retrieval as fallback
            if self.enable_parallel_processing and len(queries) > 1:
                try:
                    with self.performance_monitor.time_operation("parallel_rag_retrieval"):
                        parallel_context = self._parallel_rag_retrieval(queries, original_context)
                        if parallel_context != original_context:
                            logger.info(f"Parallel RAG successful with {len(queries)} queries")
                            return parallel_context
                except Exception as e:
                    logger.warning(f"Parallel RAG failed, using sequential fallback: {e}")
            
            # Method 3: Sequential fallback (original method, but optimized)
            with self.performance_monitor.time_operation("sequential_rag_retrieval"):
                enhanced_contexts = []
                
                # Retrieve untuk setiap expanded query (limit untuk performance)
                for query in queries[:self.config["max_expanded_queries"]]:
                    try:
                        # Gunakan search_manager untuk retrieve dokumen
                        search_results = self.search_manager.search_documents(
                            query=query,
                            limit=self.config.get("retrieval_limit_per_query", 3)
                        )
                        
                        if search_results:
                            query_context = "\n".join([doc.get("content", "") for doc in search_results])
                            enhanced_contexts.append(query_context)
                            
                    except Exception as e:
                        logger.warning(f"Failed to retrieve for query '{query}': {e}")
                        continue
                
                # Combine original context dengan enhanced contexts
                all_contexts = [original_context] + enhanced_contexts
                combined_context = "\n\n--- ENHANCED CONTEXT ---\n\n".join(
                    [ctx for ctx in all_contexts if ctx.strip()]
                )
                
                logger.info(f"Enhanced context from {len(queries)} expanded queries")
                return combined_context
            
        except Exception as e:
            logger.warning(f"Context enhancement failed: {e}")
            return original_context
    
    def _batch_rag_retrieval(self, queries: List[str], original_context: str) -> str:
        """
        Perform optimized batch RAG retrieval with parallel processing for 2 synonyms.
        
        Args:
            queries: List of expanded queries/synonyms (optimized for 2 synonyms)
            original_context: Original context
            
        Returns:
            Enhanced context from batch retrieval
        """
        try:
            if not queries or not self.config.get("enable_batch_rag", True):
                return original_context
            
            import time
            import concurrent.futures
            start_time = time.time()
            
            # Limit to 3 queries max (1 original + 2 synonyms)
            limited_queries = queries[:3]
            
            # Strategy 1: Parallel retrieval for each query (faster than sequential)
            if len(limited_queries) > 1 and self.config.get("enable_parallel_rag", True):
                try:
                    parallel_context = self._parallel_batch_retrieval(limited_queries, original_context)
                    if parallel_context != original_context:
                        elapsed_time = time.time() - start_time
                        logger.info(f"Parallel batch RAG successful with {len(limited_queries)} queries in {elapsed_time:.2f}s")
                        return parallel_context
                except Exception as e:
                    logger.warning(f"Parallel batch RAG failed, using combined approach: {e}")
            
            # Strategy 2: Smart combined query (fallback)
            combined_query = self._create_optimized_combined_query(limited_queries)
            
            # Validate combined query length
            if len(combined_query) > self.config.get("max_combined_query_length", 400):
                logger.warning(f"Combined query too long ({len(combined_query)} chars), truncating")
                combined_query = combined_query[:self.config.get("max_combined_query_length", 400)]
            
            # Single RAG call with optimized parameters
            search_results = self.search_manager.search_documents(
                query=combined_query,
                limit=min(12, len(limited_queries) * 4)  # Reduced from 15 to 12 for efficiency
            )
            
            elapsed_time = time.time() - start_time
            
            # Check timeout (reduced from 15s to 10s)
            if elapsed_time > self.config.get("batch_rag_timeout", 10):
                logger.warning(f"Batch RAG timeout ({elapsed_time:.2f}s), using fallback")
                return original_context
            
            if search_results:
                # Extract and combine content with smart deduplication
                unique_contents = self._deduplicate_and_rank_content(search_results)
                
                if unique_contents:
                    batch_context = "\n\n".join(unique_contents)
                    
                    # Combine with original context
                    if original_context.strip():
                        combined_context = f"{original_context}\n\n--- OPTIMIZED BATCH CONTEXT ---\n\n{batch_context}"
                    else:
                        combined_context = batch_context
                    
                    logger.info(f"Optimized batch RAG retrieved {len(unique_contents)} unique documents in {elapsed_time:.2f}s")
                    return combined_context
            
            logger.info(f"Batch RAG completed in {elapsed_time:.2f}s but no results found")
            return original_context
            
        except Exception as e:
            logger.error(f"Error in optimized batch RAG retrieval: {e}")
            return original_context

    def _parallel_batch_retrieval(self, queries: List[str], original_context: str) -> str:
        """
        Perform parallel RAG retrieval for multiple queries simultaneously.
        
        Args:
            queries: List of queries (max 3 for efficiency)
            original_context: Original context
            
        Returns:
            Enhanced context from parallel retrieval
        """
        try:
            import concurrent.futures
            import time
            
            start_time = time.time()
            all_results = []
            
            # Use optimized number of workers (max 3 for 2 synonyms + 1 original)
            max_workers = min(3, len(queries))
            timeout = self.config.get("parallel_rag_timeout", 8)  # Reduced timeout
            
            # Use ThreadPoolExecutor for parallel retrieval
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all retrieval tasks with optimized parameters
                future_to_query = {
                    executor.submit(
                        self._optimized_single_query_retrieval, 
                        query, 
                        limit=4  # Reduced from default to speed up
                    ): query 
                    for query in queries
                }
                
                # Collect results as they complete with timeout
                try:
                    for future in concurrent.futures.as_completed(future_to_query, timeout=timeout):
                        query = future_to_query[future]
                        try:
                            result = future.result()
                            if result:
                                all_results.extend(result)
                                logger.debug(f"Parallel retrieval for '{query[:30]}...' returned {len(result)} docs")
                        except Exception as e:
                            logger.warning(f"Parallel retrieval failed for query '{query[:30]}...': {e}")
                            
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Parallel retrieval timeout after {timeout}s")
                    # Cancel remaining futures
                    for future in future_to_query:
                        future.cancel()
            
            elapsed_time = time.time() - start_time
            
            if all_results:
                # Smart deduplication and ranking
                unique_contents = self._deduplicate_and_rank_content(all_results)
                
                if unique_contents:
                    parallel_context = "\n\n".join(unique_contents)
                    
                    # Combine with original context
                    if original_context.strip():
                        combined_context = f"{original_context}\n\n--- PARALLEL RAG CONTEXT ---\n\n{parallel_context}"
                    else:
                        combined_context = parallel_context
                    
                    logger.info(f"Parallel RAG retrieved {len(unique_contents)} unique documents in {elapsed_time:.2f}s")
                    return combined_context
            
            logger.info(f"Parallel RAG completed in {elapsed_time:.2f}s but no results found")
            return original_context
            
        except Exception as e:
            logger.error(f"Error in parallel batch retrieval: {e}")
            return original_context

    def _optimized_single_query_retrieval(self, query: str, limit: int = 4) -> List[Dict[str, Any]]:
        """
        Optimized single query retrieval with reduced parameters for speed.
        
        Args:
            query: Search query
            limit: Maximum number of results (reduced for speed)
            
        Returns:
            List of search results
        """
        try:
            if not self.search_manager:
                return []
            
            # Use search manager with optimized parameters
            search_results = self.search_manager.search_documents(
                query=query,
                limit=limit
            )
            
            return search_results if search_results else []
            
        except Exception as e:
            logger.error(f"Error in optimized single query retrieval: {e}")
            return []

    def _deduplicate_and_rank_content(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        Smart deduplication and ranking of search results content.
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            List of unique, ranked content strings
        """
        try:
            seen_content = set()
            unique_contents = []
            content_scores = {}
            
            for doc in search_results:
                content = doc.get("content", "").strip()
                if not content:
                    continue
                
                # Simple content similarity check (first 100 chars)
                content_key = content[:100].lower()
                
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_contents.append(content)
                    
                    # Score based on content length and relevance indicators
                    score = len(content) * 0.1
                    if any(keyword in content.lower() for keyword in ['psak', 'aktuaria', 'asuransi', 'keuangan']):
                        score += 10
                    
                    content_scores[content] = score
            
            # Sort by score (descending) and limit to top results
            sorted_contents = sorted(unique_contents, key=lambda x: content_scores.get(x, 0), reverse=True)
            
            # Return top 8 results for efficiency
            return sorted_contents[:8]
            
        except Exception as e:
            logger.error(f"Error in content deduplication and ranking: {e}")
            return [doc.get("content", "") for doc in search_results[:8] if doc.get("content")]

    def _create_optimized_combined_query(self, queries: List[str]) -> str:
        """
        Create an optimized combined query from 2 synonyms + 1 original (max 3 queries).
        
        Args:
            queries: List of query variations (max 3)
            
        Returns:
            Optimized combined query
        """
        try:
            if not queries:
                return ""
            
            if len(queries) == 1:
                return queries[0]
            
            # For 2-3 queries, use smart combination
            # Strategy: Use longest query as base, add unique keywords from others
            base_query = max(queries, key=len)
            
            # Early return if base query is comprehensive enough
            if len(base_query) > 200:  # Reduced threshold for efficiency
                logger.debug(f"Base query sufficient: '{base_query[:50]}...'")
                return base_query
            
            # Extract unique meaningful keywords from other queries
            all_words = set()
            stop_words = {'yang', 'adalah', 'dengan', 'untuk', 'dari', 'pada', 'dalam', 
                         'bagaimana', 'apa', 'cara', 'dan', 'atau', 'ini', 'itu', 'di', 'ke',
                         'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
            
            for query in queries:
                if query != base_query:  # Skip base query
                    words = [word.lower().strip() for word in query.split() 
                            if len(word) > 2 and word.lower() not in stop_words]
                    all_words.update(words)
            
            # Add unique terms not in base query (limit to 5 for efficiency)
            unique_terms = [term for term in list(all_words)[:10] 
                           if term not in base_query.lower()][:5]
            
            if unique_terms:
                combined_query = f"{base_query} {' '.join(unique_terms)}"
            else:
                combined_query = base_query
            
            # Ensure query doesn't exceed max length (reduced to 300 for efficiency)
            max_length = self.config.get("max_combined_query_length", 300)
            if len(combined_query) > max_length:
                combined_query = combined_query[:max_length].rsplit(' ', 1)[0]
            
            logger.debug(f"Optimized combined query: '{combined_query[:50]}...' from {len(queries)} queries")
            return combined_query
            
        except Exception as e:
            logger.error(f"Error creating optimized combined query: {e}")
            return queries[0] if queries else ""

    def _parallel_rag_retrieval(self, queries: List[str], original_context: str) -> str:
        """
        Perform parallel RAG retrieval for multiple queries using ThreadPoolExecutor.
        
        Args:
            queries: List of expanded queries
            original_context: Original context
            
        Returns:
            Enhanced context from parallel retrieval
        """
        try:
            import concurrent.futures
            import time
            
            start_time = time.time()
            enhanced_contexts = []
            
            # Use configured number of workers
            max_workers = min(self.config.get("parallel_workers", 4), len(queries))
            timeout = self.config.get("batch_rag_timeout", 15)
            
            # Use ThreadPoolExecutor for parallel retrieval
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all retrieval tasks
                future_to_query = {
                    executor.submit(self._single_query_retrieval, query): query 
                    for query in queries[:self.config["max_expanded_queries"]]
                }
                
                # Collect results as they complete with timeout
                try:
                    for future in concurrent.futures.as_completed(future_to_query, timeout=timeout):
                        query = future_to_query[future]
                        try:
                            context = future.result(timeout=5)  # Individual task timeout
                            if context and context.strip():
                                enhanced_contexts.append(context)
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"Timeout for query '{query[:50]}...'")
                            continue
                        except Exception as e:
                            logger.warning(f"Parallel retrieval failed for query '{query[:50]}...': {e}")
                            continue
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Parallel RAG overall timeout ({timeout}s)")
            
            elapsed_time = time.time() - start_time
            
            # Combine all contexts with deduplication
            if enhanced_contexts:
                # Remove duplicate contexts
                seen_contexts = set()
                unique_contexts = []
                
                for context in enhanced_contexts:
                    context_hash = hash(context[:200])  # Use first 200 chars for dedup
                    if context_hash not in seen_contexts:
                        seen_contexts.add(context_hash)
                        unique_contexts.append(context)
                
                if unique_contexts:
                    all_contexts = [original_context] + unique_contexts
                    combined_context = "\n\n--- PARALLEL ENHANCED CONTEXT ---\n\n".join(
                        [ctx for ctx in all_contexts if ctx.strip()]
                    )
                    
                    logger.info(f"Parallel RAG completed in {elapsed_time:.2f}s with {len(unique_contexts)} unique contexts")
                    return combined_context
            
            logger.info(f"Parallel RAG completed in {elapsed_time:.2f}s but no valid contexts found")
            return original_context
            
        except Exception as e:
            logger.error(f"Error in parallel RAG retrieval: {e}")
            return original_context
    
    def _single_query_retrieval(self, query: str) -> str:
        """
        Perform single query retrieval for parallel processing.
        
        Args:
            query: Single query to retrieve
            
        Returns:
            Context from single query retrieval
        """
        try:
            # Use configured limit per query
            limit = self.config.get("retrieval_limit_per_query", 3)
            
            search_results = self.search_manager.search_documents(
                query=query,
                limit=limit
            )
            
            if search_results:
                # Extract content and filter empty results
                contents = [doc.get("content", "").strip() for doc in search_results]
                valid_contents = [content for content in contents if content]
                
                if valid_contents:
                    return "\n\n".join(valid_contents)
            
            return ""
            
        except Exception as e:
            logger.warning(f"Single query retrieval failed for '{query[:50]}...': {e}")
            return ""

    def _generate_reasoning_paths(self, 
                                question: str, 
                                context: str, 
                                chat_history: str) -> Dict[str, Any]:
        """Generate multiple reasoning paths."""
        if not self.config["enable_multipath_reasoning"]:
            # Single path fallback
            return {
                "reasoning_paths": [{
                    "path_type": "single",
                    "reasoning_steps": ["Analisis pertanyaan", "Cari informasi", "Berikan jawaban"],
                    "answer": f"Berdasarkan konteks: {context[:200]}...",
                    "confidence_score": 70,
                    "key_points": ["Jawaban langsung"],
                    "sources_used": ["konteks"]
                }],
                "multipath_disabled": True
            }
        
        try:
            return self.multipath_generator.generate_reasoning_paths(
                question=question,
                context=context,
                chat_history=chat_history
            )
        except Exception as e:
            logger.warning(f"Multi-path reasoning failed: {e}")
            return self._get_fallback_reasoning_paths(question, context)
    
    def _evaluate_and_select_best_path(self, 
                                     question: str, 
                                     reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate dan select best reasoning path with validation."""
        reasoning_paths = reasoning_result.get("reasoning_paths", [])
        
        if not self.config.get("enable_path_evaluation", True) or len(reasoning_paths) <= 1:
            # Return first path sebagai best
            best_path = reasoning_paths[0] if reasoning_paths else {}
            return {
                "best_path": best_path,
                "evaluation_disabled": True,
                "path_scores": [],
                "total_paths_evaluated": len(reasoning_paths)
            }
        
        try:
            # Evaluate paths with validation integration
            evaluated_paths = self._evaluate_paths_with_validation(question, reasoning_paths)
            
            return self.path_evaluator.evaluate_reasoning_paths(
                original_question=question,
                reasoning_paths=evaluated_paths
            )
        except Exception as e:
            logger.warning(f"Path evaluation failed: {e}")
            return self._get_fallback_evaluation(reasoning_paths)
    
    def _evaluate_paths_with_validation(self, question: str, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate and rank generated paths with validation."""
        try:
            evaluated_paths = []
            
            for i, path in enumerate(paths):
                try:
                    # Validate response using validation layer
                    response_text = path.get("answer", "")
                    validation_result = self.validation_layer.validate_response(
                        response=response_text,
                        question=question
                    )
                    
                    # Get original confidence score
                    original_score = path.get("confidence_score", 0.0)
                    validation_score = validation_result.get("overall_score", 0.0)
                    
                    # Weighted combination: 70% original, 30% validation
                    combined_score = (original_score * 0.7) + (validation_score * 0.3)
                    
                    # Apply penalty if validation fails
                    if not validation_result.get("is_valid", True):
                        combined_score *= 0.8  # 20% penalty for invalid responses
                    
                    # Add validation results to path
                    path["validation"] = validation_result
                    path["original_confidence_score"] = original_score
                    path["confidence_score"] = combined_score
                    path["path_id"] = f"path_{i+1}"
                    path["is_valid"] = validation_result.get("is_valid", True)
                    
                    evaluated_paths.append(path)
                    
                except Exception as e:
                    logger.warning(f"Failed to validate path {i+1}: {e}")
                    # Add path with default validation
                    path["validation"] = {"is_valid": False, "error": str(e)}
                    path["path_id"] = f"path_{i+1}"
                    path["is_valid"] = False
                    evaluated_paths.append(path)
            
            # Sort by score (descending), prioritizing valid responses
            evaluated_paths.sort(
                key=lambda x: (x.get("is_valid", False), x.get("confidence_score", 0.0)), 
                reverse=True
            )
            
            return evaluated_paths
            
        except Exception as e:
            logger.error(f"Error in path validation: {e}")
            return paths  # Return original paths if validation fails
    
    def _prepare_final_result(self, 
                            question: str,
                            expanded_queries: Dict[str, Any],
                            reasoning_paths: Dict[str, Any],
                            evaluation_result: Dict[str, Any],
                            session_id: str) -> Dict[str, Any]:
        """Prepare final comprehensive result."""
        best_path = evaluation_result.get("best_path", {})
        
        return {
            "success": True,
            "session_id": session_id,
            "original_question": question,
            
            # Query Expansion Results
            "query_expansion": {
                "enabled": self.config["enable_query_expansion"],
                "success": expanded_queries.get("success", False),
                "expanded_queries": expanded_queries.get("expanded_queries", []),
                "total_queries": expanded_queries.get("total_queries", 1)
            },
            
            # Multi-Path Reasoning Results
            "multipath_reasoning": {
                "enabled": self.config["enable_multipath_reasoning"],
                "total_paths_generated": len(reasoning_paths.get("reasoning_paths", [])),
                "path_types": reasoning_paths.get("path_types", []),
                "avg_confidence": reasoning_paths.get("avg_confidence", 0)
            },
            
            # Path Evaluation Results
            "path_evaluation": {
                "enabled": self.config["enable_path_evaluation"],
                "total_paths_evaluated": evaluation_result.get("total_paths_evaluated", 0),
                "best_path_type": best_path.get("path_type", "unknown"),
                "best_path_score": best_path.get("evaluation_score", 0)
            },
            
            # Final Answer
            "final_answer": {
                "answer": best_path.get("answer", "Jawaban tidak tersedia"),
                "reasoning_steps": best_path.get("reasoning_steps", []),
                "confidence_score": best_path.get("confidence_score", 0),
                "key_points": best_path.get("key_points", []),
                "sources_used": best_path.get("sources_used", []),
                "path_type": best_path.get("path_type", "unknown")
            },
            
            # Metadata
            "metadata": {
                "processing_timestamp": self._get_timestamp(),
                "tot_config_used": self.config.copy(),
                "fallback_used": evaluation_result.get("fallback_used", False),
                "performance_stats": self._get_performance_stats(reasoning_paths, evaluation_result)
            }
        }
    
    def _get_fallback_result(self, question: str, context: str, chat_history: str) -> Dict[str, Any]:
        """Fallback result jika ToT processing gagal."""
        logger.info("Using fallback ToT result")
        
        fallback_answer = f"Berdasarkan konteks yang tersedia: {context[:300]}..."
        
        return {
            "success": False,
            "original_question": question,
            "fallback_used": True,
            "final_answer": {
                "answer": fallback_answer,
                "reasoning_steps": ["Analisis pertanyaan", "Gunakan konteks tersedia", "Berikan jawaban"],
                "confidence_score": 60,
                "key_points": ["Fallback response"],
                "sources_used": ["konteks"],
                "path_type": "fallback"
            },
            "error_info": "ToT processing failed, using fallback method",
            "metadata": {
                "processing_timestamp": self._get_timestamp(),
                "tot_config_used": self.config.copy()
            }
        }
    
    def _get_fallback_reasoning_paths(self, question: str, context: str) -> Dict[str, Any]:
        """Fallback reasoning paths."""
        return {
            "reasoning_paths": [{
                "path_type": "fallback",
                "reasoning_steps": ["Analisis pertanyaan", "Gunakan konteks", "Berikan jawaban"],
                "answer": f"Berdasarkan konteks: {context[:200]}...",
                "confidence_score": 65,
                "key_points": ["Fallback reasoning"],
                "sources_used": ["konteks"]
            }],
            "fallback_used": True
        }
    
    def _get_fallback_evaluation(self, reasoning_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback evaluation."""
        best_path = reasoning_paths[0] if reasoning_paths else {}
        return {
            "best_path": best_path,
            "evaluation_fallback": True,
            "path_scores": [],
            "total_paths_evaluated": len(reasoning_paths)
        }
    
    def _get_performance_stats(self, reasoning_result: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "paths_generated": len(reasoning_result.get("reasoning_paths", [])),
            "paths_evaluated": evaluation_result.get("total_paths_evaluated", 0),
            "best_path_confidence": evaluation_result.get("best_path", {}).get("confidence_score", 0),
            "avg_path_confidence": reasoning_result.get("avg_confidence", 0)
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update ToT configuration."""
        try:
            self.config.update(new_config)
            logger.info(f"ToT configuration updated: {new_config}")
        except Exception as e:
            logger.error(f"Error updating ToT config: {e}")
    
    def process_with_enhanced_document_retrieval(self, query: str, context: str = "", 
                                                       session_id: str = "", 
                                                       task_type: str = "general",
                                                       additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process query with enhanced document retrieval using ToT approach
        
        Args:
            query: User's question
            context: Initial context
            session_id: Session identifier
            task_type: Type of task (general, calculation, theory)
            additional_context: Additional context information
            
        Returns:
            Enhanced processing result with document-aware retrieval
        """
        try:
            import time
            start_time = time.time()
            logger.info(f"Starting enhanced ToT processing for query: {query[:100]}...")
            
            # Step 1: Analyze question and determine document requirements
            document_analysis = self.tot_document_reader.analyze_question(query)
            
            # Step 2: Generate retrieval strategies based on question analysis
            retrieval_strategies = self.tot_document_reader.generate_retrieval_strategies(
                question_analysis=document_analysis,
                session_id=session_id
            )
            
            # Step 3: Execute multi-path document retrieval
            enhanced_context_result = self.tot_document_reader.execute_multipath_retrieval(
                strategies=retrieval_strategies,
                session_id=session_id,
                existing_context=context
            )
            
            # Step 4: Use enhanced context for ToT processing
            enhanced_context = enhanced_context_result.get('enhanced_context', context)
            
            # Step 5: Process with standard ToT using enhanced context
            tot_result = self.process_with_tree_of_thought(
                question=query,
                context=enhanced_context,
                session_id=session_id
            )
            
            # Step 6: Enhance result with document retrieval metadata
            if tot_result.get('success', False):
                tot_result['document_retrieval'] = {
                    'strategies_used': len(retrieval_strategies.get('strategies', [])),
                    'context_enhancement_ratio': len(enhanced_context) / max(len(context), 1),
                    'relevant_sections': enhanced_context_result.get('relevant_sections', []),
                    'document_analysis': document_analysis
                }
            
            return tot_result
            
        except Exception as e:
            logger.error(f"Error in enhanced document retrieval processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_used": True
            }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current ToT configuration."""
        return self.config.copy()

    @performance_timer("tot_process_question")
    def process_question(self, 
                        question: str, 
                        context: str = "", 
                        session_id: str = None) -> Dict[str, Any]:
        """
        Process question menggunakan Tree of Thought approach.
        
        Args:
            question: Pertanyaan yang akan diproses
            context: Context tambahan
            session_id: Session ID untuk tracking
            
        Returns:
            Dictionary berisi hasil processing dengan confidence score
        """
        try:
            with self.performance_monitor.time_operation("tot_full_pipeline"):
                logger.info(f"Processing question with ToT: {question[:100]}...")
                
                # Step 1: Query Expansion
                expanded_queries = {}
                if self.config["enable_query_expansion"]:
                    with self.performance_monitor.time_operation("tot_query_expansion"):
                        expanded_queries = self.query_expansion_service.expand_query(question)
                        logger.info(f"Query expansion result: {expanded_queries.get('success', False)}")
                
                # Step 2: Enhanced Context Retrieval
                enhanced_context = context
                if self.search_manager and expanded_queries.get("success", False):
                    with self.performance_monitor.time_operation("tot_context_retrieval"):
                        enhanced_context = self._enhance_context_retrieval(expanded_queries, context)
                        logger.info(f"Enhanced context length: {len(enhanced_context)} chars")
                
                # Step 3: Multi-path Reasoning
                reasoning_paths = []
                if self.config["enable_multipath_reasoning"]:
                    with self.performance_monitor.time_operation("tot_multipath_reasoning"):
                        reasoning_paths = self._generate_reasoning_paths(
                            question, enhanced_context, expanded_queries
                        )
                        logger.info(f"Generated {len(reasoning_paths)} reasoning paths")
                
                # Step 4: Path Evaluation dan Selection
                final_result = {}
                if reasoning_paths:
                    with self.performance_monitor.time_operation("tot_path_evaluation"):
                        final_result = self._evaluate_and_select_best_path(
                            question, reasoning_paths, enhanced_context
                        )
                else:
                    # Fallback ke single path
                    logger.warning("No reasoning paths generated, using fallback")
                    final_result = self._fallback_single_path(question, enhanced_context)
                
                # Log performance summary
                self.performance_monitor.log_performance_summary()
                
                # Add metadata
                final_result.update({
                    "session_id": session_id,
                    "expanded_queries_count": len(expanded_queries.get("expanded_queries", [])),
                    "reasoning_paths_count": len(reasoning_paths),
                    "enhanced_context_length": len(enhanced_context)
                })
                
                logger.info(f"ToT processing completed with confidence: {final_result.get('confidence', 0)}")
                return final_result
                
        except Exception as e:
            logger.error(f"Error in ToT processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
                "confidence": 0,
                "session_id": session_id
            }