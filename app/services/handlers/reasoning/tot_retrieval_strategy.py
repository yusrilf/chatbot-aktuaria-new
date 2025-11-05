"""Tree of Thought Retrieval Strategy.

Strategy untuk menggabungkan retrieval dari global documents dan session documents
menggunakan pendekatan Tree of Thought untuk optimasi retrieval.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class RetrievalPath(Enum):
    """Enum untuk jenis path retrieval."""
    GLOBAL_ONLY = "global_only"
    SESSION_ONLY = "session_only"
    HYBRID_BALANCED = "hybrid_balanced"
    GLOBAL_PRIORITY = "global_priority"
    SESSION_PRIORITY = "session_priority"
    CONTEXTUAL_ADAPTIVE = "contextual_adaptive"

class RetrievalMode(Enum):
    """Enum untuk mode retrieval."""
    FAST = "fast"  # Quick retrieval dengan minimal processing
    BALANCED = "balanced"  # Balance antara speed dan quality
    COMPREHENSIVE = "comprehensive"  # Thorough retrieval dengan full analysis
    ADAPTIVE = "adaptive"  # Adaptive berdasarkan question complexity

@dataclass
class RetrievalResult:
    """Result dari retrieval operation."""
    documents: List[Dict[str, Any]]
    source_type: str  # 'global', 'session', 'hybrid'
    retrieval_path: RetrievalPath
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]
    section_mappings: Optional[List[Any]] = None

@dataclass
class RetrievalStrategy:
    """Strategy untuk retrieval operation."""
    path: RetrievalPath
    weight_global: float
    weight_session: float
    max_documents: int
    enable_reranking: bool
    enable_section_mapping: bool
    timeout_seconds: float
    metadata: Dict[str, Any]

class ToTRetrievalStrategy:
    """Tree of Thought Retrieval Strategy untuk document retrieval."""
    
    def __init__(self, 
                 vector_store_manager=None,
                 question_analyzer=None,
                 document_section_mapper=None,
                 llm=None):
        """
        Initialize ToT Retrieval Strategy.
        
        Args:
            vector_store_manager: Vector store manager
            question_analyzer: Question analyzer instance
            document_section_mapper: Document section mapper instance
            llm: Language model untuk analysis
        """
        self.vector_store_manager = vector_store_manager
        self.question_analyzer = question_analyzer
        self.document_section_mapper = document_section_mapper
        self.llm = llm
        
        # Strategy configurations
        self.strategy_configs = {
            RetrievalPath.GLOBAL_ONLY: {
                "weight_global": 1.0,
                "weight_session": 0.0,
                "max_documents": 15,
                "enable_reranking": True,
                "timeout": 10.0
            },
            RetrievalPath.SESSION_ONLY: {
                "weight_global": 0.0,
                "weight_session": 1.0,
                "max_documents": 10,
                "enable_reranking": True,
                "timeout": 5.0
            },
            RetrievalPath.HYBRID_BALANCED: {
                "weight_global": 0.6,
                "weight_session": 0.4,
                "max_documents": 20,
                "enable_reranking": True,
                "timeout": 15.0
            },
            RetrievalPath.GLOBAL_PRIORITY: {
                "weight_global": 0.8,
                "weight_session": 0.2,
                "max_documents": 18,
                "enable_reranking": True,
                "timeout": 12.0
            },
            RetrievalPath.SESSION_PRIORITY: {
                "weight_global": 0.3,
                "weight_session": 0.7,
                "max_documents": 15,
                "enable_reranking": True,
                "timeout": 8.0
            },
            RetrievalPath.CONTEXTUAL_ADAPTIVE: {
                "weight_global": 0.5,  # Will be adjusted dynamically
                "weight_session": 0.5,  # Will be adjusted dynamically
                "max_documents": 25,
                "enable_reranking": True,
                "timeout": 20.0
            }
        }
        
        # Mode configurations
        self.mode_configs = {
            RetrievalMode.FAST: {
                "enable_section_mapping": False,
                "enable_parallel_retrieval": True,
                "max_concurrent_operations": 3,
                "enable_advanced_reranking": False
            },
            RetrievalMode.BALANCED: {
                "enable_section_mapping": True,
                "enable_parallel_retrieval": True,
                "max_concurrent_operations": 2,
                "enable_advanced_reranking": True
            },
            RetrievalMode.COMPREHENSIVE: {
                "enable_section_mapping": True,
                "enable_parallel_retrieval": False,
                "max_concurrent_operations": 1,
                "enable_advanced_reranking": True
            },
            RetrievalMode.ADAPTIVE: {
                "enable_section_mapping": True,  # Will be adjusted
                "enable_parallel_retrieval": True,  # Will be adjusted
                "max_concurrent_operations": 2,  # Will be adjusted
                "enable_advanced_reranking": True  # Will be adjusted
            }
        }
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        
        # Configuration
        self.config = {
            "default_mode": RetrievalMode.BALANCED,
            "default_path": RetrievalPath.HYBRID_BALANCED,
            "enable_performance_tracking": True,
            "enable_adaptive_optimization": True,
            "max_retrieval_attempts": 3,
            "fallback_timeout": 30.0
        }
        
        logger.info("ToT Retrieval Strategy initialized")
    
    async def execute_retrieval(self, 
                              question_analysis: Dict[str, Any],
                              session_id: str,
                              mode: Optional[RetrievalMode] = None,
                              path: Optional[RetrievalPath] = None) -> RetrievalResult:
        """
        Execute retrieval menggunakan Tree of Thought approach.
        
        Args:
            question_analysis: Hasil analisis pertanyaan
            session_id: Session ID
            mode: Retrieval mode (optional)
            path: Retrieval path (optional)
            
        Returns:
            RetrievalResult object
        """
        start_time = time.time()
        
        try:
            logger.info(f"Executing ToT retrieval for session {session_id}")
            
            # Determine optimal strategy
            strategy = await self._determine_optimal_strategy(
                question_analysis, session_id, mode, path
            )
            
            # Execute retrieval dengan strategy yang dipilih
            result = await self._execute_strategy(strategy, question_analysis, session_id)
            
            # Track performance
            if self.config["enable_performance_tracking"]:
                self._track_performance(strategy, result, time.time() - start_time)
            
            logger.info(f"ToT retrieval completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in ToT retrieval execution: {e}")
            return await self._get_fallback_result(question_analysis, session_id, time.time() - start_time)
    
    async def _determine_optimal_strategy(self, 
                                        question_analysis: Dict[str, Any],
                                        session_id: str,
                                        mode: Optional[RetrievalMode] = None,
                                        path: Optional[RetrievalPath] = None) -> RetrievalStrategy:
        """
        Determine optimal retrieval strategy berdasarkan question analysis.
        
        Args:
            question_analysis: Question analysis results
            session_id: Session ID
            mode: Override mode
            path: Override path
            
        Returns:
            RetrievalStrategy object
        """
        try:
            # Use provided mode/path atau determine automatically
            selected_mode = mode or await self._select_retrieval_mode(question_analysis)
            selected_path = path or await self._select_retrieval_path(question_analysis, session_id)
            
            # Get base configuration
            path_config = self.strategy_configs[selected_path].copy()
            mode_config = self.mode_configs[selected_mode].copy()
            
            # Apply adaptive adjustments
            if selected_path == RetrievalPath.CONTEXTUAL_ADAPTIVE:
                path_config = await self._adjust_adaptive_weights(
                    path_config, question_analysis, session_id
                )
            
            if selected_mode == RetrievalMode.ADAPTIVE:
                mode_config = await self._adjust_adaptive_mode(
                    mode_config, question_analysis
                )
            
            # Create strategy
            strategy = RetrievalStrategy(
                path=selected_path,
                weight_global=path_config["weight_global"],
                weight_session=path_config["weight_session"],
                max_documents=path_config["max_documents"],
                enable_reranking=path_config["enable_reranking"],
                enable_section_mapping=mode_config["enable_section_mapping"],
                timeout_seconds=path_config["timeout"],
                metadata={
                    "mode": selected_mode.value,
                    "path": selected_path.value,
                    "mode_config": mode_config,
                    "path_config": path_config,
                    "question_complexity": question_analysis.get('complexity_score', 0.5)
                }
            )
            
            logger.info(f"Selected strategy: {selected_path.value} with {selected_mode.value} mode")
            return strategy
            
        except Exception as e:
            logger.error(f"Error determining optimal strategy: {e}")
            # Fallback to default strategy
            return self._get_default_strategy()
    
    async def _select_retrieval_mode(self, question_analysis: Dict[str, Any]) -> RetrievalMode:
        """
        Select retrieval mode berdasarkan question analysis.
        
        Args:
            question_analysis: Question analysis results
            
        Returns:
            Selected RetrievalMode
        """
        try:
            complexity_score = question_analysis.get('complexity_score', 0.5)
            question_type = question_analysis.get('question_type', 'theory')
            
            # Simple questions -> Fast mode
            if complexity_score < 0.3 and question_type in ['data_lookup', 'simple_calculation']:
                return RetrievalMode.FAST
            
            # Complex questions -> Comprehensive mode
            elif complexity_score > 0.8 or question_type in ['comparison', 'multi_step_calculation']:
                return RetrievalMode.COMPREHENSIVE
            
            # Variable complexity -> Adaptive mode
            elif question_analysis.get('has_multiple_domains', False):
                return RetrievalMode.ADAPTIVE
            
            # Default -> Balanced mode
            else:
                return RetrievalMode.BALANCED
                
        except Exception as e:
            logger.error(f"Error selecting retrieval mode: {e}")
            return self.config["default_mode"]
    
    async def _select_retrieval_path(self, 
                                   question_analysis: Dict[str, Any],
                                   session_id: str) -> RetrievalPath:
        """
        Select retrieval path berdasarkan question analysis dan session context.
        
        Args:
            question_analysis: Question analysis results
            session_id: Session ID
            
        Returns:
            Selected RetrievalPath
        """
        try:
            # Check session document availability
            has_session_docs = await self._check_session_documents(session_id)
            
            # Question characteristics
            question_type = question_analysis.get('question_type', 'theory')
            primary_domain = question_analysis.get('primary_domain', 'general')
            
            # No session documents -> Global only
            if not has_session_docs:
                return RetrievalPath.GLOBAL_ONLY
            
            # Session-specific questions -> Session priority
            if question_analysis.get('is_session_specific', False):
                return RetrievalPath.SESSION_PRIORITY
            
            # General theoretical questions -> Global priority
            if question_type == 'theory' and primary_domain == 'general':
                return RetrievalPath.GLOBAL_PRIORITY
            
            # Calculation questions -> Hybrid balanced (need both examples and formulas)
            if question_type in ['calculation', 'multi_step_calculation']:
                return RetrievalPath.HYBRID_BALANCED
            
            # Complex multi-domain questions -> Contextual adaptive
            if question_analysis.get('has_multiple_domains', False):
                return RetrievalPath.CONTEXTUAL_ADAPTIVE
            
            # Default -> Hybrid balanced
            return RetrievalPath.HYBRID_BALANCED
            
        except Exception as e:
            logger.error(f"Error selecting retrieval path: {e}")
            return self.config["default_path"]
    
    async def _check_session_documents(self, session_id: str) -> bool:
        """
        Check apakah ada session documents yang tersedia.
        
        Args:
            session_id: Session ID
            
        Returns:
            True jika ada session documents
        """
        try:
            if not self.vector_store_manager:
                return False
            
            # Check session vector store
            session_docs = await self._get_session_document_count(session_id)
            return session_docs > 0
            
        except Exception as e:
            logger.error(f"Error checking session documents: {e}")
            return False
    
    async def _get_session_document_count(self, session_id: str) -> int:
        """
        Get jumlah dokumen dalam session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Number of session documents
        """
        try:
            # Implementation depends on vector store manager interface
            if hasattr(self.vector_store_manager, 'get_session_document_count'):
                return await self.vector_store_manager.get_session_document_count(session_id)
            else:
                # Fallback: try to retrieve sample documents
                sample_docs = await self._retrieve_session_documents(
                    "sample query", session_id, max_docs=1
                )
                return len(sample_docs) if sample_docs else 0
                
        except Exception as e:
            logger.error(f"Error getting session document count: {e}")
            return 0
    
    async def _adjust_adaptive_weights(self, 
                                     config: Dict[str, Any],
                                     question_analysis: Dict[str, Any],
                                     session_id: str) -> Dict[str, Any]:
        """
        Adjust weights untuk contextual adaptive path.
        
        Args:
            config: Base configuration
            question_analysis: Question analysis
            session_id: Session ID
            
        Returns:
            Adjusted configuration
        """
        try:
            # Factors untuk weight adjustment
            complexity_score = question_analysis.get('complexity_score', 0.5)
            is_session_specific = question_analysis.get('is_session_specific', False)
            question_type = question_analysis.get('question_type', 'theory')
            
            # Base weights
            weight_global = 0.5
            weight_session = 0.5
            
            # Adjust berdasarkan session specificity
            if is_session_specific:
                weight_session += 0.3
                weight_global -= 0.3
            
            # Adjust berdasarkan question type
            if question_type == 'theory':
                weight_global += 0.2
                weight_session -= 0.2
            elif question_type in ['calculation', 'procedure']:
                # Balance untuk examples dan formulas
                pass  # Keep balanced
            
            # Adjust berdasarkan complexity
            if complexity_score > 0.7:
                # Complex questions benefit from more comprehensive search
                weight_global += 0.1
                weight_session += 0.1
                config["max_documents"] = min(config["max_documents"] + 5, 30)
            
            # Normalize weights
            total_weight = weight_global + weight_session
            if total_weight > 0:
                weight_global /= total_weight
                weight_session /= total_weight
            
            # Ensure weights are within bounds
            weight_global = max(0.1, min(0.9, weight_global))
            weight_session = max(0.1, min(0.9, weight_session))
            
            config["weight_global"] = weight_global
            config["weight_session"] = weight_session
            
            logger.info(f"Adaptive weights: global={weight_global:.2f}, session={weight_session:.2f}")
            return config
            
        except Exception as e:
            logger.error(f"Error adjusting adaptive weights: {e}")
            return config
    
    async def _adjust_adaptive_mode(self, 
                                  config: Dict[str, Any],
                                  question_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust mode configuration untuk adaptive mode.
        
        Args:
            config: Base mode configuration
            question_analysis: Question analysis
            
        Returns:
            Adjusted mode configuration
        """
        try:
            complexity_score = question_analysis.get('complexity_score', 0.5)
            question_type = question_analysis.get('question_type', 'theory')
            
            # Adjust berdasarkan complexity
            if complexity_score < 0.3:
                # Simple questions -> Fast mode settings
                config["enable_section_mapping"] = False
                config["max_concurrent_operations"] = 3
                config["enable_advanced_reranking"] = False
            elif complexity_score > 0.8:
                # Complex questions -> Comprehensive mode settings
                config["enable_section_mapping"] = True
                config["max_concurrent_operations"] = 1
                config["enable_advanced_reranking"] = True
            
            # Adjust berdasarkan question type
            if question_type in ['calculation', 'multi_step_calculation']:
                config["enable_section_mapping"] = True  # Need precise sections
                config["enable_advanced_reranking"] = True  # Need good ranking
            
            return config
            
        except Exception as e:
            logger.error(f"Error adjusting adaptive mode: {e}")
            return config
    
    async def _execute_strategy(self, 
                              strategy: RetrievalStrategy,
                              question_analysis: Dict[str, Any],
                              session_id: str) -> RetrievalResult:
        """
        Execute retrieval strategy.
        
        Args:
            strategy: Retrieval strategy
            question_analysis: Question analysis
            session_id: Session ID
            
        Returns:
            RetrievalResult object
        """
        start_time = time.time()
        
        try:
            # Execute retrieval berdasarkan path
            if strategy.path == RetrievalPath.GLOBAL_ONLY:
                documents = await self._retrieve_global_documents(
                    question_analysis, strategy.max_documents
                )
                source_type = "global"
                
            elif strategy.path == RetrievalPath.SESSION_ONLY:
                documents = await self._retrieve_session_documents(
                    question_analysis.get('original_question', ''), 
                    session_id, 
                    strategy.max_documents
                )
                source_type = "session"
                
            else:
                # Hybrid retrieval
                documents = await self._retrieve_hybrid_documents(
                    question_analysis, session_id, strategy
                )
                source_type = "hybrid"
            
            # Apply reranking jika enabled
            if strategy.enable_reranking and documents:
                documents = await self._rerank_documents(
                    documents, question_analysis, strategy
                )
            
            # Apply section mapping jika enabled
            section_mappings = None
            if strategy.enable_section_mapping and self.document_section_mapper:
                section_mappings = self.document_section_mapper.map_question_to_sections(
                    question_analysis, documents, session_id
                )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                documents, question_analysis, strategy
            )
            
            processing_time = time.time() - start_time
            
            result = RetrievalResult(
                documents=documents,
                source_type=source_type,
                retrieval_path=strategy.path,
                confidence_score=confidence_score,
                processing_time=processing_time,
                metadata={
                    "strategy": strategy.metadata,
                    "document_count": len(documents),
                    "has_section_mappings": section_mappings is not None,
                    "reranking_applied": strategy.enable_reranking
                },
                section_mappings=section_mappings
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            processing_time = time.time() - start_time
            return await self._get_fallback_result(question_analysis, session_id, processing_time)
    
    async def _retrieve_global_documents(self, 
                                       question_analysis: Dict[str, Any],
                                       max_docs: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents dari global vector store.
        
        Args:
            question_analysis: Question analysis
            max_docs: Maximum number of documents
            
        Returns:
            List of retrieved documents
        """
        try:
            if not self.vector_store_manager:
                return []
            
            query = question_analysis.get('original_question', '')
            
            # Use vector store manager untuk global retrieval
            if hasattr(self.vector_store_manager, 'retrieve_global_documents'):
                documents = await self.vector_store_manager.retrieve_global_documents(
                    query, max_docs
                )
            else:
                # Fallback method
                documents = await self._fallback_global_retrieval(query, max_docs)
            
            logger.info(f"Retrieved {len(documents)} global documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving global documents: {e}")
            return []
    
    async def _retrieve_session_documents(self, 
                                        query: str,
                                        session_id: str,
                                        max_docs: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents dari session vector store.
        
        Args:
            query: Search query
            session_id: Session ID
            max_docs: Maximum number of documents
            
        Returns:
            List of retrieved documents
        """
        try:
            if not self.vector_store_manager:
                return []
            
            # Use vector store manager untuk session retrieval
            if hasattr(self.vector_store_manager, 'retrieve_session_documents'):
                documents = await self.vector_store_manager.retrieve_session_documents(
                    query, session_id, max_docs
                )
            else:
                # Fallback method
                documents = await self._fallback_session_retrieval(query, session_id, max_docs)
            
            logger.info(f"Retrieved {len(documents)} session documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving session documents: {e}")
            return []
    
    async def _retrieve_hybrid_documents(self, 
                                       question_analysis: Dict[str, Any],
                                       session_id: str,
                                       strategy: RetrievalStrategy) -> List[Dict[str, Any]]:
        """
        Retrieve documents menggunakan hybrid approach.
        
        Args:
            question_analysis: Question analysis
            session_id: Session ID
            strategy: Retrieval strategy
            
        Returns:
            Combined list of documents
        """
        try:
            query = question_analysis.get('original_question', '')
            
            # Calculate document counts berdasarkan weights
            total_docs = strategy.max_documents
            global_docs_count = int(total_docs * strategy.weight_global)
            session_docs_count = int(total_docs * strategy.weight_session)
            
            # Ensure minimum counts
            global_docs_count = max(1, global_docs_count)
            session_docs_count = max(1, session_docs_count)
            
            # Parallel retrieval untuk efficiency
            mode_config = strategy.metadata.get('mode_config', {})
            if mode_config.get('enable_parallel_retrieval', True):
                # Parallel execution
                global_task = self._retrieve_global_documents(question_analysis, global_docs_count)
                session_task = self._retrieve_session_documents(query, session_id, session_docs_count)
                
                global_docs, session_docs = await asyncio.gather(
                    global_task, session_task, return_exceptions=True
                )
                
                # Handle exceptions
                if isinstance(global_docs, Exception):
                    logger.error(f"Global retrieval failed: {global_docs}")
                    global_docs = []
                if isinstance(session_docs, Exception):
                    logger.error(f"Session retrieval failed: {session_docs}")
                    session_docs = []
            else:
                # Sequential execution
                global_docs = await self._retrieve_global_documents(question_analysis, global_docs_count)
                session_docs = await self._retrieve_session_documents(query, session_id, session_docs_count)
            
            # Combine dan deduplicate documents
            combined_docs = self._combine_and_deduplicate_documents(
                global_docs, session_docs, strategy
            )
            
            # Limit to max documents
            if len(combined_docs) > strategy.max_documents:
                combined_docs = combined_docs[:strategy.max_documents]
            
            logger.info(f"Hybrid retrieval: {len(global_docs)} global + {len(session_docs)} session = {len(combined_docs)} combined")
            return combined_docs
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            # Fallback to global only
            return await self._retrieve_global_documents(question_analysis, strategy.max_documents)
    
    def _combine_and_deduplicate_documents(self, 
                                         global_docs: List[Dict[str, Any]],
                                         session_docs: List[Dict[str, Any]],
                                         strategy: RetrievalStrategy) -> List[Dict[str, Any]]:
        """
        Combine dan deduplicate documents dari global dan session.
        
        Args:
            global_docs: Global documents
            session_docs: Session documents
            strategy: Retrieval strategy
            
        Returns:
            Combined and deduplicated documents
        """
        try:
            # Mark source untuk tracking
            for doc in global_docs:
                doc['retrieval_source'] = 'global'
                doc['source_weight'] = strategy.weight_global
            
            for doc in session_docs:
                doc['retrieval_source'] = 'session'
                doc['source_weight'] = strategy.weight_session
            
            # Combine documents
            all_docs = global_docs + session_docs
            
            # Deduplicate berdasarkan content similarity atau source
            seen_sources = set()
            deduplicated_docs = []
            
            for doc in all_docs:
                doc_id = doc.get('source', '') + str(hash(doc.get('content', '')[:100]))
                if doc_id not in seen_sources:
                    seen_sources.add(doc_id)
                    deduplicated_docs.append(doc)
            
            # Sort berdasarkan relevance score dan source weight
            deduplicated_docs.sort(
                key=lambda x: (x.get('score', 0.5) * x.get('source_weight', 0.5)), 
                reverse=True
            )
            
            return deduplicated_docs
            
        except Exception as e:
            logger.error(f"Error combining and deduplicating documents: {e}")
            return global_docs + session_docs  # Simple fallback
    
    async def _rerank_documents(self, 
                              documents: List[Dict[str, Any]],
                              question_analysis: Dict[str, Any],
                              strategy: RetrievalStrategy) -> List[Dict[str, Any]]:
        """
        Rerank documents berdasarkan relevance.
        
        Args:
            documents: Documents to rerank
            question_analysis: Question analysis
            strategy: Retrieval strategy
            
        Returns:
            Reranked documents
        """
        try:
            if not documents:
                return documents
            
            mode_config = strategy.metadata.get('mode_config', {})
            enable_advanced = mode_config.get('enable_advanced_reranking', True)
            
            if enable_advanced and self.llm:
                # Advanced reranking dengan LLM
                reranked_docs = await self._advanced_rerank_with_llm(
                    documents, question_analysis
                )
            else:
                # Simple reranking berdasarkan scores
                reranked_docs = self._simple_rerank_by_scores(
                    documents, question_analysis
                )
            
            logger.info(f"Reranked {len(documents)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            return documents  # Return original order on error
    
    async def _advanced_rerank_with_llm(self, 
                                      documents: List[Dict[str, Any]],
                                      question_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Advanced reranking menggunakan LLM.
        
        Args:
            documents: Documents to rerank
            question_analysis: Question analysis
            
        Returns:
            Reranked documents
        """
        try:
            # Implementation untuk LLM-based reranking
            # This would involve creating prompts dan getting relevance scores
            
            # For now, fallback to simple reranking
            return self._simple_rerank_by_scores(documents, question_analysis)
            
        except Exception as e:
            logger.error(f"Error in advanced reranking: {e}")
            return documents
    
    def _simple_rerank_by_scores(self, 
                               documents: List[Dict[str, Any]],
                               question_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simple reranking berdasarkan existing scores dan factors.
        
        Args:
            documents: Documents to rerank
            question_analysis: Question analysis
            
        Returns:
            Reranked documents
        """
        try:
            keywords = question_analysis.get('keywords', [])
            
            # Calculate enhanced scores
            for doc in documents:
                base_score = doc.get('score', 0.5)
                source_weight = doc.get('source_weight', 0.5)
                
                # Keyword bonus
                content = doc.get('content', '').lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
                keyword_bonus = (keyword_matches / len(keywords)) * 0.2 if keywords else 0
                
                # Source type bonus
                source_bonus = 0.1 if doc.get('retrieval_source') == 'session' else 0
                
                # Calculate final score
                enhanced_score = (base_score * source_weight) + keyword_bonus + source_bonus
                doc['enhanced_score'] = min(enhanced_score, 1.0)
            
            # Sort by enhanced score
            documents.sort(key=lambda x: x.get('enhanced_score', 0.5), reverse=True)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in simple reranking: {e}")
            return documents
    
    def _calculate_confidence_score(self, 
                                  documents: List[Dict[str, Any]],
                                  question_analysis: Dict[str, Any],
                                  strategy: RetrievalStrategy) -> float:
        """
        Calculate confidence score untuk retrieval result.
        
        Args:
            documents: Retrieved documents
            question_analysis: Question analysis
            strategy: Retrieval strategy
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        try:
            if not documents:
                return 0.0
            
            # Base confidence dari document scores
            doc_scores = [doc.get('score', 0.5) for doc in documents]
            avg_doc_score = sum(doc_scores) / len(doc_scores)
            
            # Document count factor
            count_factor = min(len(documents) / 10, 1.0)  # Optimal around 10 docs
            
            # Strategy confidence
            strategy_confidence = {
                RetrievalPath.GLOBAL_ONLY: 0.8,
                RetrievalPath.SESSION_ONLY: 0.7,
                RetrievalPath.HYBRID_BALANCED: 0.9,
                RetrievalPath.GLOBAL_PRIORITY: 0.85,
                RetrievalPath.SESSION_PRIORITY: 0.75,
                RetrievalPath.CONTEXTUAL_ADAPTIVE: 0.95
            }.get(strategy.path, 0.8)
            
            # Keyword coverage factor
            keywords = question_analysis.get('keywords', [])
            if keywords:
                covered_keywords = set()
                for doc in documents:
                    content = doc.get('content', '').lower()
                    for kw in keywords:
                        if kw.lower() in content:
                            covered_keywords.add(kw)
                keyword_coverage = len(covered_keywords) / len(keywords)
            else:
                keyword_coverage = 1.0
            
            # Calculate final confidence
            confidence = (
                avg_doc_score * 0.4 +
                count_factor * 0.2 +
                strategy_confidence * 0.2 +
                keyword_coverage * 0.2
            )
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _track_performance(self, 
                         strategy: RetrievalStrategy,
                         result: RetrievalResult,
                         total_time: float) -> None:
        """
        Track performance metrics.
        
        Args:
            strategy: Used strategy
            result: Retrieval result
            total_time: Total processing time
        """
        try:
            strategy_key = f"{strategy.path.value}_{strategy.metadata.get('mode', 'unknown')}"
            
            performance_data = {
                'processing_time': total_time,
                'document_count': len(result.documents),
                'confidence_score': result.confidence_score,
                'timestamp': time.time()
            }
            
            self.performance_history[strategy_key].append(performance_data)
            
            # Keep only recent history (last 100 entries)
            if len(self.performance_history[strategy_key]) > 100:
                self.performance_history[strategy_key] = self.performance_history[strategy_key][-100:]
            
        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
    
    async def _get_fallback_result(self, 
                                 question_analysis: Dict[str, Any],
                                 session_id: str,
                                 processing_time: float) -> RetrievalResult:
        """
        Get fallback result jika terjadi error.
        
        Args:
            question_analysis: Question analysis
            session_id: Session ID
            processing_time: Processing time so far
            
        Returns:
            Fallback RetrievalResult
        """
        try:
            # Try simple global retrieval as fallback
            fallback_docs = await self._retrieve_global_documents(question_analysis, 5)
            
            return RetrievalResult(
                documents=fallback_docs,
                source_type="global_fallback",
                retrieval_path=RetrievalPath.GLOBAL_ONLY,
                confidence_score=0.3,
                processing_time=processing_time,
                metadata={
                    "fallback_used": True,
                    "document_count": len(fallback_docs)
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback result: {e}")
            return RetrievalResult(
                documents=[],
                source_type="empty_fallback",
                retrieval_path=RetrievalPath.GLOBAL_ONLY,
                confidence_score=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _get_default_strategy(self) -> RetrievalStrategy:
        """
        Get default retrieval strategy.
        
        Returns:
            Default RetrievalStrategy
        """
        try:
            default_path = self.config["default_path"]
            default_mode = self.config["default_mode"]
            
            path_config = self.strategy_configs[default_path]
            mode_config = self.mode_configs[default_mode]
            
            return RetrievalStrategy(
                path=default_path,
                weight_global=path_config["weight_global"],
                weight_session=path_config["weight_session"],
                max_documents=path_config["max_documents"],
                enable_reranking=path_config["enable_reranking"],
                enable_section_mapping=mode_config["enable_section_mapping"],
                timeout_seconds=path_config["timeout"],
                metadata={
                    "mode": default_mode.value,
                    "path": default_path.value,
                    "is_default": True
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating default strategy: {e}")
            # Hard-coded fallback
            return RetrievalStrategy(
                path=RetrievalPath.HYBRID_BALANCED,
                weight_global=0.6,
                weight_session=0.4,
                max_documents=15,
                enable_reranking=True,
                enable_section_mapping=True,
                timeout_seconds=15.0,
                metadata={"fallback": True}
            )
    
    async def _fallback_global_retrieval(self, query: str, max_docs: int) -> List[Dict[str, Any]]:
        """
        Fallback method untuk global retrieval.
        
        Args:
            query: Search query
            max_docs: Maximum documents
            
        Returns:
            List of documents
        """
        try:
            # Implementation depends on available interfaces
            # This is a placeholder untuk actual implementation
            return []
            
        except Exception as e:
            logger.error(f"Error in fallback global retrieval: {e}")
            return []
    
    async def _fallback_session_retrieval(self, 
                                        query: str,
                                        session_id: str,
                                        max_docs: int) -> List[Dict[str, Any]]:
        """
        Fallback method untuk session retrieval.
        
        Args:
            query: Search query
            session_id: Session ID
            max_docs: Maximum documents
            
        Returns:
            List of documents
        """
        try:
            # Implementation depends on available interfaces
            # This is a placeholder untuk actual implementation
            return []
            
        except Exception as e:
            logger.error(f"Error in fallback session retrieval: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Performance statistics
        """
        try:
            stats = {}
            
            for strategy_key, history in self.performance_history.items():
                if not history:
                    continue
                
                times = [entry['processing_time'] for entry in history]
                confidences = [entry['confidence_score'] for entry in history]
                doc_counts = [entry['document_count'] for entry in history]
                
                stats[strategy_key] = {
                    'total_executions': len(history),
                    'avg_processing_time': sum(times) / len(times),
                    'avg_confidence': sum(confidences) / len(confidences),
                    'avg_document_count': sum(doc_counts) / len(doc_counts),
                    'min_processing_time': min(times),
                    'max_processing_time': max(times)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration.
        
        Args:
            new_config: New configuration values
        """
        try:
            self.config.update(new_config)
            logger.info(f"ToT Retrieval Strategy config updated: {new_config}")
        except Exception as e:
            logger.error(f"Error updating config: {e}")
    
    def clear_performance_history(self) -> None:
        """
        Clear performance history.
        """
        try:
            self.performance_history.clear()
            logger.info("Performance history cleared")
        except Exception as e:
            logger.error(f"Error clearing performance history: {e}")