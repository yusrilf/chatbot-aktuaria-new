"""
Chain of Thought (CoT) Orchestrator Service

This service orchestrates the complete CoT retrieval pipeline:
1. Query preprocessing and entity extraction
2. Document orchestration and selection
3. Document filtering based on orchestrator recommendations
4. Semantic retrieval on filtered subset
5. Chunk reranking for relevance
6. Context assembly and organization
7. LLM verification of final answer

Author: Assistant
Date: 2024
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import asyncio
import traceback

from .query_preprocessor import QueryPreprocessor, ExtractedEntities
from .document_orchestrator import DocumentOrchestrator, OrchestrationResult
from .document_filter import DocumentFilter, FilterResult
from .semantic_retriever import SemanticRetriever, SemanticRetrievalResult, RetrievedChunk
from .reranker import ChunkReranker, RerankingResult, RerankingScore
from .context_assembler import ContextAssembler, AssembledContext
from .llm_verifier import LLMVerifier, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class CoTResult:
    """Complete Chain of Thought result"""
    query: str
    processed_query: ExtractedEntities
    orchestration: OrchestrationResult
    filtering: FilterResult
    retrieval: SemanticRetrievalResult
    reranking: RerankingResult
    context: AssembledContext
    verification: Optional[VerificationResult] = None
    
    # Performance metrics
    total_time: float = 0.0
    preprocessing_time: float = 0.0
    orchestration_time: float = 0.0
    filtering_time: float = 0.0
    retrieval_time: float = 0.0
    reranking_time: float = 0.0
    assembly_time: float = 0.0
    verification_time: float = 0.0
    
    # Quality metrics
    documents_considered: int = 0
    documents_selected: int = 0
    chunks_retrieved: int = 0
    chunks_reranked: int = 0
    final_context_length: int = 0


class CoTOrchestrator:
    """
    Main Chain of Thought orchestrator that coordinates all retrieval components
    """
    
    def __init__(
        self,
        llm_service,
        vector_store_manager,
        search_manager=None,
        enable_verification: bool = True,
        max_context_length: int = 8000,
        max_documents: int = 50,
        debug_mode: bool = False
    ):
        """
        Initialize CoT orchestrator
        
        Args:
            llm_service: LLM service for orchestration and verification
            vector_store_manager: Vector store manager for retrieval
            search_manager: Optional search manager for fallback
            enable_verification: Whether to enable LLM verification
            max_context_length: Maximum context length for assembly
            max_documents: Maximum number of documents to process (default: 50)
            debug_mode: Enable debug logging
        """
        self.llm_service = llm_service
        self.vector_store_manager = vector_store_manager
        self.search_manager = search_manager
        self.enable_verification = enable_verification
        self.max_context_length = max_context_length
        self.max_documents = max_documents
        self.debug_mode = debug_mode
        
        # Complexity-based chunk limits for performance optimization
        self.chunk_limits = {
            'simple': 5,     # Simple queries need fewer chunks
            'moderate': 10,  # Moderate queries need balanced chunks
            'complex': 15    # Complex queries need more comprehensive chunks
        }
        
        # Initialize components
        self.query_preprocessor = QueryPreprocessor()
        self.document_orchestrator = DocumentOrchestrator(llm_service)
        self.document_filter = DocumentFilter()
        self.semantic_retriever = SemanticRetriever(
            vector_store_manager, search_manager
        )
        self.chunk_reranker = ChunkReranker(llm_service)
        self.context_assembler = ContextAssembler(max_context_length)
        
        if enable_verification:
            self.llm_verifier = LLMVerifier(llm_service)
        else:
            self.llm_verifier = None
            
        logger.info(f"CoT Orchestrator initialized with verification: {enable_verification}, "
                   f"chunk_limits: {self.chunk_limits}")
    
    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        available_documents: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> CoTResult:
        """
        Process query through Chain of Thought pipeline with async optimizations
        
        Args:
            query: User query
            session_id: Session identifier
            available_documents: Available documents for processing
            chat_history: Previous conversation history
            
        Returns:
            CoTResult with complete processing information
        """
        start_time = time.time()
        logger.info(f"=== DEBUG: Starting CoT process_query ===")
        logger.info(f"Query: {query}")
        logger.info(f"Available documents count: {len(available_documents) if available_documents else 0}")
        
        # Apply document limit early to reduce processing overhead
        if available_documents and len(available_documents) > self.max_documents:
            logger.info(f"Limiting documents from {len(available_documents)} to {self.max_documents} for performance")
            # Prioritize session documents first, then global documents
            session_docs = [doc for doc in available_documents if not doc.get('is_global', False)]
            global_docs = [doc for doc in available_documents if doc.get('is_global', False)]
            
            # Take up to max_documents, prioritizing session documents
            limited_docs = session_docs[:self.max_documents//2] + global_docs[:self.max_documents//2]
            available_documents = limited_docs[:self.max_documents]
            logger.info(f"Document limit applied: using {len(available_documents)} documents")
        
        try:
            # Step 1: Query Preprocessing (critical path)
            step_start = time.time()
            logger.info(f"=== DEBUG: Step 1 - Query Preprocessing ===")
            processed_query = await self._preprocess_query(query, chat_history)
            preprocessing_time = time.time() - step_start
            logger.info(f"Query preprocessing completed in {preprocessing_time:.2f}s")
            
            # Step 2 & 3: Run Sequential Context Reading and Document Orchestration concurrently
            step_start = time.time()
            logger.info(f"=== DEBUG: Steps 2-3 - Concurrent Sequential Reading & Orchestration ===")
            
            # Run these operations concurrently as they can be independent
            sequential_task = asyncio.create_task(
                self._sequential_document_reading(processed_query, available_documents, session_id)
            )
            orchestration_task = asyncio.create_task(
                self._orchestrate_documents(processed_query, available_documents, None)  # Start without sequential context
            )
            
            # Wait for both to complete
            sequential_context, initial_orchestration = await asyncio.gather(
                sequential_task, orchestration_task
            )
            
            # Re-run orchestration with sequential context if needed (lightweight operation)
            orchestration = await self._orchestrate_documents(
                processed_query, available_documents, sequential_context
            )
            
            concurrent_time = time.time() - step_start
            logger.info(f"Concurrent operations completed in {concurrent_time:.2f}s")
            
            # Step 4: Document Filtering with complexity analysis (critical path)
            step_start = time.time()
            logger.info(f"=== DEBUG: Step 4 - Document Filtering ===")
            
            # Analyze query complexity for early filtering
            complexity_level = self._analyze_query_complexity(processed_query)
            logger.info(f"Query complexity determined: {complexity_level}")
            
            filtering = await self._filter_documents(
                orchestration, available_documents, session_id, complexity_level
            )
            filtering_time = time.time() - step_start
            logger.info(f"Document filtering completed in {filtering_time:.2f}s")
            
            # Step 5: Semantic Retrieval with complexity optimization (critical path)
            step_start = time.time()
            logger.info(f"=== DEBUG: Step 5 - Semantic Retrieval ===")
            
            retrieval = await self._retrieve_semantically(
                processed_query, filtering.filtered_files, session_id, complexity_level
            )
            retrieval_time = time.time() - step_start
            logger.info(f"Semantic retrieval completed in {retrieval_time:.2f}s")
            logger.info(f"Retrieved {len(retrieval.chunks)} chunks")
            
            # Step 6 & 7: Run Reranking and Context Assembly preparation concurrently
            step_start = time.time()
            logger.info(f"=== DEBUG: Steps 6-7 - Concurrent Reranking & Context Prep ===")
            
            # Start reranking task
            reranking_task = asyncio.create_task(
                self._rerank_chunks_by_source(processed_query, retrieval.chunks, sequential_context)
            )
            
            # Prepare context assembly metadata concurrently (non-critical)
            context_prep_task = asyncio.create_task(
                self._prepare_context_metadata(session_id, sequential_context)
            )
            
            # Wait for reranking (critical) and context prep (optimization)
            reranking, context_metadata = await asyncio.gather(
                reranking_task, context_prep_task
            )
            
            reranking_time = time.time() - step_start
            logger.info(f"Concurrent reranking and context prep completed in {reranking_time:.2f}s")
            
            # Step 8: Final Context Assembly (critical path)
            step_start = time.time()
            context = await self._assemble_context(
                reranking, session_id, sequential_context, context_metadata
            )
            assembly_time = time.time() - step_start
            
            # Optional Verification (will be done after LLM generates answer)
            verification_time = 0.0
            
            total_time = time.time() - start_time
            
            # Create result
            result = CoTResult(
                query=query,
                processed_query=processed_query,
                orchestration=orchestration,
                filtering=filtering,
                retrieval=retrieval,
                reranking=reranking,
                context=context,
                total_time=total_time,
                preprocessing_time=preprocessing_time,
                orchestration_time=concurrent_time,  # Combined time for concurrent operations
                filtering_time=filtering_time,
                retrieval_time=retrieval_time,
                reranking_time=reranking_time,
                assembly_time=assembly_time,
                verification_time=verification_time,
                documents_considered=len(available_documents or []),
                documents_selected=len(filtering.filtered_files),
                chunks_retrieved=len(retrieval.chunks),
                chunks_reranked=len(reranking.reranked_chunks),
                final_context_length=len(context.full_context)
            )
            
            if self.debug_mode:
                self._log_debug_info(result)
                
            logger.info(f"CoT processing completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in CoT processing: {str(e)}")
            raise
    
    async def verify_answer(
        self,
        answer: str,
        cot_result: CoTResult
    ) -> VerificationResult:
        """
        Verify LLM answer against retrieved context
        
        Args:
            answer: Generated answer to verify
            cot_result: CoT result with context
            
        Returns:
            VerificationResult with citation analysis
        """
        if not self.llm_verifier:
            logger.warning("LLM verifier not enabled")
            return None
            
        start_time = time.time()
        
        try:
            verification = self.llm_verifier.verify_answer(
                answer, cot_result.context.full_context, cot_result.query
            )
            
            verification_time = time.time() - start_time
            cot_result.verification = verification
            cot_result.verification_time = verification_time
            
            logger.info(f"Answer verification completed in {verification_time:.2f}s")
            return verification
            
        except Exception as e:
            logger.error(f"Error in answer verification: {str(e)}")
            return None
    
    async def _preprocess_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> ExtractedEntities:
        """Preprocess query for entity extraction and normalization"""
        try:
            return self.query_preprocessor.preprocess_query(query)
        except Exception as e:
            logger.error(f"Query preprocessing failed: {str(e)}")
            # Return basic processed query as fallback
            return ExtractedEntities(
                psak_references=[],
                puc_references=[],
                numbers=[],
                years=[],
                technical_terms=[],
                normalized_query=query.lower().strip(),
                original_query=query
            )
    
    def _convert_to_document_metadata(self, documents: List[Dict[str, Any]]) -> List[Any]:
        """Convert dictionary format documents to DocumentMetadata objects.
        
        Args:
            documents: List of document dictionaries from _get_available_documents
            
        Returns:
            List of DocumentMetadata objects for orchestrator
        """
        from .document_orchestrator import DocumentMetadata
        
        converted_docs = []
        for doc in documents:
            try:
                # Extract metadata from the document
                metadata = doc.get('metadata', {})
                
                # Create DocumentMetadata object with required fields
                doc_metadata = DocumentMetadata(
                    filename=doc.get('filename', 'unknown'),
                    doc_type=metadata.get('doc_type', doc.get('doc_type', 'unknown')),
                    keywords=self._extract_keywords_from_metadata(metadata),
                    domain=metadata.get('domain', 'general'),
                    difficulty=metadata.get('difficulty', 'medium'),
                    version=metadata.get('version'),
                    title=metadata.get('section_heading', metadata.get('title')),
                    description=doc.get('content_preview', '')[:200] if doc.get('content_preview') else None
                )
                converted_docs.append(doc_metadata)
                
            except Exception as e:
                logger.warning(f"Failed to convert document {doc.get('filename', 'unknown')}: {str(e)}")
                continue
                
        logger.info(f"Converted {len(converted_docs)} documents to DocumentMetadata format")
        return converted_docs
    
    def _extract_keywords_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract keywords from document metadata.
        
        Args:
            metadata: Document metadata dictionary
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Get keywords from various metadata fields
        if 'keywords' in metadata:
            if isinstance(metadata['keywords'], list):
                keywords.extend(metadata['keywords'])
            elif isinstance(metadata['keywords'], str):
                keywords.extend([k.strip() for k in metadata['keywords'].split(',') if k.strip()])
        
        # Add technical terms from other fields
        for field in ['technical_terms', 'related_regulations', 'psak_references']:
            if field in metadata and metadata[field]:
                if isinstance(metadata[field], list):
                    keywords.extend(metadata[field])
                elif isinstance(metadata[field], str):
                    keywords.extend([k.strip() for k in metadata[field].split(',') if k.strip()])
        
        # Add document type and domain as keywords
        if metadata.get('doc_type'):
            keywords.append(metadata['doc_type'])
        if metadata.get('domain'):
            keywords.append(metadata['domain'])
            
        return list(set(keywords))  # Remove duplicates

    async def _orchestrate_documents(
        self,
        processed_query: ExtractedEntities,
        available_documents: Optional[List[Dict[str, Any]]],
        sequential_context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationResult:
        """Orchestrate document selection using LLM"""
        try:
            # Convert dictionary format to DocumentMetadata objects
            if available_documents:
                metadata_docs = self._convert_to_document_metadata(available_documents)
                logger.info(f"Converted {len(metadata_docs)} documents for orchestration")
            else:
                metadata_docs = []
                logger.warning("No available documents provided for orchestration")
            
            return self.document_orchestrator.orchestrate_documents(
                query=processed_query.original_query,
                entities=processed_query,
                available_documents=metadata_docs
            )
        except Exception as e:
            logger.error(f"Document orchestration failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return fallback orchestration
            return OrchestrationResult(
                selected_files=[],
                reasoning="Document orchestration failed, using empty selection",
                confidence_score=0.1,
                fallback_used=True,
                metadata_used=[]
            )
    
    async def _filter_documents(
        self,
        orchestration: OrchestrationResult,
        available_documents: Optional[List[Dict[str, Any]]],
        session_id: Optional[str],
        complexity_level: Optional[str] = None
    ) -> FilterResult:
        """Filter documents based on orchestration recommendations and complexity"""
        try:
            # Convert available_documents to DocumentMetadata format
            if available_documents:
                converted_docs = self._convert_to_document_metadata(available_documents)
            else:
                converted_docs = []
                
            return self.document_filter.apply_filtering(
                orchestration_result=orchestration,
                available_documents=converted_docs,
                session_id=session_id,
                complexity_level=complexity_level
            )
        except Exception as e:
            logger.error(f"Document filtering failed: {str(e)}")
            traceback.print_exc()
            # Return fallback filtering
            return FilterResult(
                filtered_files=[],
                filter_criteria={},
                total_available=len(available_documents or []),
                total_filtered=0,
                fallback_applied=True,
                reasoning="Document filtering failed, using empty selection"
            )
    
    async def _retrieve_semantically(
        self,
        processed_query: ExtractedEntities,
        selected_documents: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        complexity_level: Optional[str] = None
    ) -> SemanticRetrievalResult:
        """Perform semantic retrieval on filtered documents with complexity optimization"""
        try:
            # Create a FilterResult for the semantic retriever
            filter_result = FilterResult(
                filtered_files=selected_documents,
                filter_criteria={},
                total_available=len(selected_documents),
                total_filtered=len(selected_documents),
                fallback_applied=False,
                reasoning="Using provided documents"
            )
            
            # Use complexity-aware retrieval if available
            if hasattr(self.semantic_retriever, 'retrieve') and complexity_level:
                return self.semantic_retriever.retrieve(
                    query=processed_query.original_query,
                    filter_result=filter_result,
                    session_id=session_id,
                    complexity_level=complexity_level
                )
            else:
                # Fallback to original method
                return self.semantic_retriever.retrieve_from_subset(
                    query=processed_query.original_query,
                    entities=processed_query,
                    filter_result=filter_result,
                    session_id=session_id
                )
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {str(e)}")
            # Return empty retrieval result
            return SemanticRetrievalResult(
                chunks=[],
                total_chunks=0,
                documents_searched=[],
                retrieval_strategy="fallback",
                query_used=processed_query.original_query,
                search_metadata={"error": str(e)}
            )
    
    async def _rerank_chunks(
        self,
        processed_query: ExtractedEntities,
        chunks: List[Any]
    ) -> RerankingResult:
        """Rerank chunks for relevance with complexity-based limits"""
        try:
            # Determine query complexity
            complexity = self._analyze_query_complexity(processed_query)
            chunk_limit = self.chunk_limits.get(complexity, 10)
            
            logger.info(f"Query complexity: {complexity}, chunk limit: {chunk_limit}")
            
            return self.chunk_reranker.rerank_chunks(
                query=processed_query.original_query,
                chunks=chunks,
                top_k=chunk_limit,  # Use complexity-based limit
                min_score_threshold=0.001  # Very low threshold for debugging
            )
        except Exception as e:
            logger.error(f"Chunk reranking failed: {str(e)}")
            # Return chunks without reranking
            return RerankingResult(
                reranked_chunks=[],
                scores=[],
                reranking_method="fallback",
                original_count=len(chunks),
                final_count=0,
                metadata={"error": str(e)}
            )
    
    async def _assemble_context(
        self,
        reranking: RerankingResult,
        session_id: Optional[str] = None,
        sequential_context: Optional[Dict[str, Any]] = None,
        context_metadata: Optional[Dict[str, Any]] = None
    ) -> AssembledContext:
        """Assemble final context from reranked chunks with optional metadata"""
        try:
            # Assemble context from reranked chunks with metadata
            return self.context_assembler.assemble_context(
                reranking_result=reranking,
                session_id=session_id,
                include_readme=True,
                context_metadata=context_metadata
            )
        except Exception as e:
            logger.error(f"Context assembly failed: {e}")
            return AssembledContext(
                full_context="",
                sections=[],
                total_length=0,
                chunk_count=0,
                document_count=0,
                has_readme=False,
                assembly_metadata={"error": str(e)}
            )
    
    def _log_debug_info(self, result: CoTResult):
        """Log detailed debug information"""
        logger.debug("=== CoT Processing Debug Info ===")
        logger.debug(f"Query: {result.query}")
        logger.debug(f"Processed entities: {result.processed_query.technical_terms}")
        logger.debug(f"Documents considered: {result.documents_considered}")
        logger.debug(f"Documents selected: {result.documents_selected}")
        logger.debug(f"Chunks retrieved: {result.chunks_retrieved}")
        logger.debug(f"Chunks reranked: {result.chunks_reranked}")
        logger.debug(f"Final context length: {result.final_context_length}")
        logger.debug(f"Total processing time: {result.total_time:.2f}s")
        logger.debug("=== End Debug Info ===")
    
    def get_performance_stats(self, result: CoTResult) -> Dict[str, Any]:
        """Get performance statistics from CoT result"""
        return {
            "total_time": result.total_time,
            "quality_metrics": {
                "documents_considered": result.documents_considered,
                "documents_selected": result.documents_selected,
                "chunks_retrieved": result.chunks_retrieved,
                "chunks_reranked": result.chunks_reranked,
                "final_context_length": result.final_context_length,
                "selection_ratio": (
                    result.documents_selected / max(result.documents_considered, 1)
                )
            }
        }
    
    async def _sequential_document_reading(
        self,
        processed_query: ExtractedEntities,
        available_documents: Optional[List[Dict[str, Any]]],
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Sequential reading of README.md → INDEX.md → JSON actuarial report
        
        Args:
            processed_query: Processed query with entities
            available_documents: Available documents list
            session_id: Session ID for filtering
            
        Returns:
            Dictionary with sequential context from all three sources
        """
        try:
            logger.info("Starting sequential document reading")
            sequential_context = {
                "readme_context": None,
                "index_context": None,
                "json_report_context": None,
                "reading_plan": None
            }
            
            if not available_documents:
                logger.warning("No available documents for sequential reading")
                return sequential_context
            
            # Step 1: Read README.md for foundational knowledge
            readme_docs = [doc for doc in available_documents if 'README.md' in doc.get('filename', '')]
            if readme_docs:
                logger.info("Reading README.md for foundational knowledge")
                readme_context = await self._read_specific_document(
                    readme_docs[0], processed_query, "foundational_knowledge"
                )
                sequential_context["readme_context"] = readme_context
            else:
                logger.warning("README.md not found in available documents")
            
            # Step 2: Read INDEX.md for calculation guidance
            index_docs = [doc for doc in available_documents if 'INDEX.md' in doc.get('filename', '')]
            if index_docs:
                logger.info("Reading INDEX.md for calculation guidance")
                index_context = await self._read_specific_document(
                    index_docs[0], processed_query, "calculation_guidance"
                )
                sequential_context["index_context"] = index_context
            else:
                logger.warning("INDEX.md not found in available documents")
            
            # Step 3: Identify and read JSON actuarial report
            json_docs = [doc for doc in available_documents 
                        if doc.get('session_id') == session_id and 
                        ('.json' in doc.get('filename', '').lower() or 
                         'actuarial' in doc.get('filename', '').lower() or
                         'laporan' in doc.get('filename', '').lower())]
            
            if json_docs:
                logger.info(f"Reading JSON actuarial report: {json_docs[0].get('filename')}")
                json_context = await self._read_specific_document(
                    json_docs[0], processed_query, "actuarial_data"
                )
                sequential_context["json_report_context"] = json_context
            else:
                logger.info("No JSON actuarial report found in session documents")
            
            # Step 4: Generate retrieval plan based on sequential reading
            sequential_context["reading_plan"] = await self._generate_retrieval_plan(
                processed_query, sequential_context
            )
            
            logger.info("Sequential document reading completed")
            return sequential_context
            
        except Exception as e:
            logger.error(f"Error in sequential document reading: {str(e)}")
            return {
                "readme_context": None,
                "index_context": None,
                "json_report_context": None,
                "reading_plan": None,
                "error": str(e)
            }
    
    async def _read_specific_document(
        self,
        document: Dict[str, Any],
        processed_query: ExtractedEntities,
        purpose: str
    ) -> Dict[str, Any]:
        """
        Read a specific document with targeted retrieval
        
        Args:
            document: Document metadata
            processed_query: Processed query
            purpose: Purpose of reading (foundational_knowledge, calculation_guidance, actuarial_data)
            
        Returns:
            Context extracted from the document
        """
        try:
            # Create targeted search based on purpose
            if purpose == "foundational_knowledge":
                search_query = f"PSAK 219 basic concepts definitions {' '.join(processed_query.technical_terms)}"
            elif purpose == "calculation_guidance":
                search_query = f"actuarial calculation steps methodology {' '.join(processed_query.technical_terms)}"
            elif purpose == "actuarial_data":
                search_query = f"employee benefits data calculations {' '.join(processed_query.technical_terms)}"
            else:
                search_query = processed_query.original_query
            
            # Perform targeted retrieval on this specific document
            filename = document.get('filename', '')
            
            # Use semantic search with filename filter
            results = self.vector_store_manager.vectorstore.similarity_search(
                query=search_query,
                k=5,  # Get top 5 chunks from this document
                filter={'filename': filename}
            )
            
            if not results:
                logger.warning(f"No content found in {filename}")
                return {"content": "", "chunks": [], "purpose": purpose}
            
            # Extract and combine content
            content_chunks = []
            for doc in results:
                content_chunks.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 1.0  # Default score for sequential reading
                })
            
            combined_content = "\n\n".join([chunk["content"] for chunk in content_chunks])
            
            return {
                "content": combined_content,
                "chunks": content_chunks,
                "purpose": purpose,
                "document": filename,
                "chunk_count": len(content_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error reading document {document.get('filename', 'unknown')}: {str(e)}")
            return {"content": "", "chunks": [], "purpose": purpose, "error": str(e)}
    
    async def _generate_retrieval_plan(
        self,
        processed_query: ExtractedEntities,
        sequential_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a clear retrieval plan based on sequential reading
        
        Args:
            processed_query: Processed query with entities
            sequential_context: Context from sequential reading
            
        Returns:
            Retrieval plan with strategy and priorities
        """
        try:
            plan = {
                "strategy": "sequential_enhanced",
                "priorities": [],
                "metadata_filters": {},
                "reranking_strategy": "multi_source",
                "context_assembly": "layered"
            }
            
            # Determine priorities based on available context
            if sequential_context.get("readme_context"):
                plan["priorities"].append({
                    "source": "README.md",
                    "weight": 0.3,
                    "purpose": "foundational_knowledge"
                })
            
            if sequential_context.get("index_context"):
                plan["priorities"].append({
                    "source": "INDEX.md", 
                    "weight": 0.4,
                    "purpose": "calculation_guidance"
                })
            
            if sequential_context.get("json_report_context"):
                plan["priorities"].append({
                    "source": "JSON_report",
                    "weight": 0.3,
                    "purpose": "actuarial_data"
                })
            
            # Set metadata filters based on query entities
            entities = processed_query.technical_terms
            if any(term in entities for term in ["PSAK", "219", "aktuaria", "actuarial"]):
                plan["metadata_filters"]["keywords"] = ["PSAK 219", "actuarial", "employee benefits"]
            
            if any(term in entities for term in ["perhitungan", "calculation", "hitung"]):
                plan["metadata_filters"]["document_type"] = ["calculation_guide", "methodology"]
            
            logger.info(f"Generated retrieval plan with {len(plan['priorities'])} sources")
            return plan
            
        except Exception as e:
            logger.error(f"Error generating retrieval plan: {str(e)}")
            return {
                "strategy": "fallback",
                "priorities": [],
                "metadata_filters": {},
                "error": str(e)
            }
    
    async def _rerank_chunks_by_source(
        self,
        processed_query: ExtractedEntities,
        chunks: List[Any],
        sequential_context: Dict[str, Any]
    ) -> RerankingResult:
        """
        Rerank chunks by source with proper threshold validation
        
        Args:
            processed_query: Processed query
            chunks: Retrieved chunks
            sequential_context: Context from sequential reading
            
        Returns:
            RerankingResult with source-based reranking
        """
        try:
            logger.info(f"=== DEBUG: Starting source-based reranking ===")
            logger.info(f"Input chunks count: {len(chunks)}")
            logger.info(f"Chunks types: {[type(chunk).__name__ for chunk in chunks[:3]]}")  # Show first 3 types
            logger.info(f"Query: {processed_query.original_query}")
            
            print(f"RERANK_BY_SOURCE: Starting with {len(chunks)} chunks")
            
            # If no chunks, return empty result
            if not chunks:
                logger.warning("No chunks to rerank")
                return RerankingResult(
                    reranked_chunks=[],
                    scores=[],
                    reranking_method="source_based",
                    original_count=0,
                    final_count=0,
                    metadata={"reason": "No chunks provided"}
                )
            
            # Group chunks by source
            source_groups = {
                "readme": [],
                "index": [],
                "json_report": [],
                "other": []
            }
            
            for chunk in chunks:
                # Handle different chunk types safely - check if it has metadata attribute
                if hasattr(chunk, 'metadata'):
                    filename = chunk.metadata.get('filename', '').lower()
                elif hasattr(chunk, 'document_name'):
                    filename = chunk.document_name.lower()
                elif isinstance(chunk, str):
                    # String chunks don't have filename info
                    filename = 'unknown'
                else:
                    filename = str(chunk).lower()
                    
                if 'readme.md' in filename:
                    source_groups["readme"].append(chunk)
                elif 'index.md' in filename:
                    source_groups["index"].append(chunk)
                elif '.json' in filename or 'actuarial' in filename or 'laporan' in filename:
                    source_groups["json_report"].append(chunk)
                else:
                    source_groups["other"].append(chunk)
            
            # Log source distribution
            for source, source_chunks in source_groups.items():
                if source_chunks:
                    logger.info(f"Source {source}: {len(source_chunks)} chunks")
            
            # Rerank each source separately
            all_reranked_chunks = []
            all_scores = []
            source_scores = {}
            
            for source, source_chunks in source_groups.items():
                if not source_chunks:
                    source_scores[source] = 0
                    continue
                
                try:
                    # Convert chunks to RetrievedChunk format if needed
                    converted_chunks = []
                    for i, chunk in enumerate(source_chunks):
                        try:
                            if isinstance(chunk, RetrievedChunk):
                                converted_chunks.append(chunk)
                            elif isinstance(chunk, str):
                                # Handle string chunks - this is likely the source of the error
                                logger.warning(f"Found string chunk in source {source}: {chunk[:100]}...")
                                converted_chunk = RetrievedChunk(
                                    content=chunk,
                                    metadata={'source': source, 'type': 'string_chunk'},
                                    score=0.3,  # Lower score for string chunks
                                    document_name=f"{source}_document",
                                    section_heading=None,
                                    chunk_index=i
                                )
                                converted_chunks.append(converted_chunk)
                            elif isinstance(chunk, dict):
                                # Handle dictionary chunks
                                content = chunk.get('content', chunk.get('page_content', str(chunk)))
                                metadata = chunk.get('metadata', {})
                                score = chunk.get('score', 0.5)
                                document_name = chunk.get('document_name', 
                                                         chunk.get('filename', metadata.get('filename', 'unknown')))
                                section_heading = chunk.get('section_heading')
                                chunk_index = chunk.get('chunk_index', i)
                                
                                converted_chunk = RetrievedChunk(
                                    content=content,
                                    metadata=metadata,
                                    score=score,
                                    document_name=document_name,
                                    section_heading=section_heading,
                                    chunk_index=chunk_index
                                )
                                converted_chunks.append(converted_chunk)
                            elif hasattr(chunk, 'content'):
                                # Handle objects with content attribute - safe extraction
                                try:
                                    if hasattr(chunk, 'content'):
                                        content = chunk.content
                                    elif hasattr(chunk, 'page_content'):
                                        content = chunk.page_content
                                    else:
                                        content = str(chunk)
                                except Exception as e:
                                    logger.warning(f"Error extracting content from chunk: {e}")
                                    content = str(chunk)
                                    
                                metadata = getattr(chunk, 'metadata', {})
                                score = getattr(chunk, 'score', 0.5)
                                document_name = getattr(chunk, 'doc_name', 
                                                       getattr(chunk, 'document_name', 'unknown'))
                                section_heading = getattr(chunk, 'section_heading', None)
                                chunk_index = getattr(chunk, 'chunk_id', 
                                                     getattr(chunk, 'chunk_index', i))
                                
                                converted_chunk = RetrievedChunk(
                                    content=content,
                                    metadata=metadata,
                                    score=score,
                                    document_name=document_name,
                                    section_heading=section_heading,
                                    chunk_index=chunk_index
                                )
                                converted_chunks.append(converted_chunk)
                            elif hasattr(chunk, 'page_content'):
                                # Handle Document objects
                                content = chunk.page_content
                                metadata = getattr(chunk, 'metadata', {})
                                score = getattr(chunk, 'score', 0.5)
                                document_name = metadata.get('filename', 'unknown')
                                section_heading = metadata.get('section_heading')
                                chunk_index = metadata.get('chunk_index', i)
                                
                                converted_chunk = RetrievedChunk(
                                    content=content,
                                    metadata=metadata,
                                    score=score,
                                    document_name=document_name,
                                    section_heading=section_heading,
                                    chunk_index=chunk_index
                                )
                                converted_chunks.append(converted_chunk)
                            else:
                                # Fallback for completely unknown types
                                logger.warning(f"Unknown chunk type {type(chunk)} in source {source}, converting to string")
                                content = str(chunk)
                                metadata = {'source': source, 'type': str(type(chunk))}
                                score = 0.2  # Very low score for unknown types
                                document_name = f"{source}_document"
                                section_heading = None
                                chunk_index = i
                                
                                converted_chunk = RetrievedChunk(
                                    content=content,
                                    metadata=metadata,
                                    score=score,
                                    document_name=document_name,
                                    section_heading=section_heading,
                                    chunk_index=chunk_index
                                )
                                converted_chunks.append(converted_chunk)
                        except Exception as chunk_error:
                            logger.error(f"Error converting chunk {i} in source {source}: {str(chunk_error)}")
                            logger.error(f"Chunk type: {type(chunk)}, Chunk: {str(chunk)[:200]}...")
                            # Create a fallback chunk to prevent complete failure
                            fallback_chunk = RetrievedChunk(
                                content=f"Error processing chunk: {str(chunk_error)}",
                                metadata={'source': source, 'error': str(chunk_error)},
                                score=0.1,
                                document_name=f"{source}_error",
                                section_heading=None,
                                chunk_index=i
                            )
                            converted_chunks.append(fallback_chunk)
                    
                    # Use existing reranker for each source with converted chunks
                    logger.info(f"=== DEBUG: Reranking source {source} ===")
                    logger.info(f"Converted chunks count: {len(converted_chunks)}")
                    logger.info(f"Query: {processed_query.original_query}")
                    
                    # Determine query complexity for chunk limits
                    complexity = self._analyze_query_complexity(processed_query)
                    source_chunk_limit = max(3, self.chunk_limits.get(complexity, 10) // 3)  # Distribute across sources
                    
                    logger.info(f"Source {source} complexity: {complexity}, chunk limit: {source_chunk_limit}")
                    
                    source_reranking = self.chunk_reranker.rerank_chunks(
                        query=processed_query.original_query,
                        chunks=converted_chunks,
                        top_k=source_chunk_limit,  # Use complexity-based limit per source
                        min_score_threshold=0.001  # Very low threshold for debugging
                    )
                    
                    logger.info(f"Reranking result type: {type(source_reranking)}")
                    logger.info(f"Reranked chunks count: {len(source_reranking.reranked_chunks) if source_reranking.reranked_chunks else 0}")
                    logger.info(f"Scores count: {len(source_reranking.scores) if source_reranking.scores else 0}")
                    
                    # Extract reranked chunks and scores
                    reranked_chunks = source_reranking.reranked_chunks
                    scores = source_reranking.scores
                    
                    # Apply additional threshold validation if needed
                    threshold = 0.001  # Very low threshold for debugging
                    filtered_results = []
                    
                    logger.info(f"Source {source}: Processing {len(reranked_chunks)} reranked chunks with {len(scores)} scores")
                    
                    for i, chunk in enumerate(reranked_chunks):
                        if i < len(scores):
                            score_value = scores[i].score
                            logger.info(f"Chunk {i}: score={score_value}, threshold={threshold}")
                            if score_value >= threshold:
                                filtered_results.append((chunk, scores[i]))
                            else:
                                logger.info(f"Chunk {i} filtered out: score {score_value} < threshold {threshold}")
                        else:
                            logger.warning(f"No score available for chunk {i}")
                    
                    # Add to overall results
                    all_reranked_chunks.extend([chunk for chunk, _ in filtered_results])
                    all_scores.extend([score for _, score in filtered_results])
                    source_scores[source] = len(filtered_results)
                    
                    logger.info(f"Source {source}: {len(filtered_results)}/{len(source_chunks)} chunks passed threshold")
                    
                except Exception as e:
                    logger.error(f"Error reranking source {source}: {str(e)}")
                    # Include original chunks with default scores
                    for chunk in source_chunks:
                        all_reranked_chunks.append(chunk)
                        all_scores.append(RerankingScore(
                            score=0.5,
                            explanation=f"Fallback score for {source}",
                            confidence=0.3,
                            chunk_index=len(all_scores)
                        ))
                    source_scores[source] = len(source_chunks)
            
            # Sort by score across all sources
            if all_scores:
                combined_results = list(zip(all_reranked_chunks, all_scores))
                combined_results.sort(key=lambda x: x[1].score, reverse=True)
                
                final_chunks = [chunk for chunk, _ in combined_results]
                final_scores = [score for _, score in combined_results]
            else:
                final_chunks = []
                final_scores = []
            
            logger.info(f"Source-based reranking completed: {len(final_chunks)} total chunks")
            
            return RerankingResult(
                reranked_chunks=final_chunks,
                scores=final_scores,
                reranking_method="source_based",
                original_count=len(chunks),
                final_count=len(final_chunks),
                metadata={
                    "source_scores": source_scores,
                    "threshold_applied": 0.001,  # Use very low threshold for debugging
                    "multi_source_reranking": True,
                    "complexity_based_limits": True
                }
            )
            
        except Exception as e:
            logger.error(f"Error in source-based reranking: {str(e)}")
            traceback.print_exc()
            # Fallback to original reranking
            return await self._rerank_chunks(processed_query, chunks)
    
    def _analyze_query_complexity(self, processed_query: ExtractedEntities) -> str:
        """
        Analyze query complexity to determine appropriate chunk limits
        
        Args:
            processed_query: Processed query with extracted entities
            
        Returns:
            Complexity level: 'simple', 'moderate', or 'complex'
        """
        try:
            complexity_score = 0
            
            # Count different types of entities
            entity_counts = {
                'psak_refs': len(processed_query.psak_references),
                'puc_refs': len(processed_query.puc_references),
                'numbers': len(processed_query.numbers),
                'years': len(processed_query.years),
                'technical_terms': len(processed_query.technical_terms)
            }
            
            # Calculate complexity based on entity diversity and count
            total_entities = sum(entity_counts.values())
            entity_types = sum(1 for count in entity_counts.values() if count > 0)
            
            # Query length factor
            query_length = len(processed_query.original_query.split())
            
            # Complexity scoring
            if total_entities <= 2 and entity_types <= 1 and query_length <= 10:
                complexity = 'simple'
            elif total_entities <= 5 and entity_types <= 3 and query_length <= 20:
                complexity = 'moderate'
            else:
                complexity = 'complex'
            
            # Special cases for actuarial queries
            actuarial_keywords = ['perhitungan', 'calculation', 'aktuaria', 'actuarial', 'PSAK 219']
            if any(keyword.lower() in processed_query.original_query.lower() for keyword in actuarial_keywords):
                if complexity == 'simple':
                    complexity = 'moderate'  # Actuarial queries need more context
            
            logger.info(f"Query complexity analysis: {complexity} "
                       f"(entities: {total_entities}, types: {entity_types}, length: {query_length})")
            
            return complexity
            
        except Exception as e:
            logger.error(f"Error analyzing query complexity: {str(e)}")
            return 'moderate'  # Default to moderate complexity
    
    async def _prepare_context_metadata(
        self, 
        session_id: Optional[str], 
        sequential_context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Prepare context assembly metadata concurrently (non-critical optimization)
        
        Args:
            session_id: Session identifier
            sequential_context: Sequential reading context
            
        Returns:
            Dictionary with context metadata for assembly optimization
        """
        try:
            metadata = {
                'session_id': session_id,
                'has_sequential_context': sequential_context is not None,
                'sequential_context_length': len(sequential_context) if sequential_context else 0,
                'timestamp': time.time(),
                'optimization_hints': {
                    'prioritize_session_docs': session_id is not None,
                    'include_sequential_summary': sequential_context is not None
                }
            }
            
            # Add any additional metadata preparation that can be done concurrently
            if sequential_context:
                # Extract key themes from sequential context for better assembly
                # Handle both string and dict types for sequential_context
                if isinstance(sequential_context, dict):
                    context_text = sequential_context.get('content', str(sequential_context))
                else:
                    context_text = str(sequential_context)
                metadata['sequential_themes'] = self._extract_context_themes(context_text)
            
            logger.debug(f"Context metadata prepared: {len(metadata)} items")
            return metadata
            
        except Exception as e:
            logger.error(f"Error preparing context metadata: {str(e)}")
            return {'error': str(e), 'fallback': True}
    
    def _extract_context_themes(self, context: str) -> List[str]:
        """
        Extract key themes from sequential context for better assembly
        
        Args:
            context: Sequential context string
            
        Returns:
            List of key themes/topics
        """
        try:
            # Simple theme extraction based on common actuarial terms
            themes = []
            actuarial_terms = [
                'PSAK', 'aktuaria', 'actuarial', 'perhitungan', 'calculation',
                'asuransi', 'insurance', 'klaim', 'claim', 'premi', 'premium',
                'cadangan', 'reserve', 'risiko', 'risk'
            ]
            
            context_lower = context.lower()
            for term in actuarial_terms:
                if term.lower() in context_lower:
                    themes.append(term)
            
            return themes[:5]  # Limit to top 5 themes
            
        except Exception as e:
            logger.error(f"Error extracting context themes: {str(e)}")
            return []