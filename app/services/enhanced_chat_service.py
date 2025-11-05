"""
Enhanced Chat Service with Chain of Thought (CoT) Orchestration

This service replaces the old chat service implementation with a new CoT-based
approach that provides better document selection and retrieval orchestration.

Key improvements:
1. Query preprocessing and entity extraction
2. LLM-based document orchestration
3. Filtered semantic retrieval
4. Context reranking and assembly
5. Answer verification

Author: Assistant
Date: 2024
"""

import logging
import traceback
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    from langchain_openai import ChatOpenAI
except Exception as _import_err:
    ChatOpenAI = None
    logging.getLogger(__name__).warning(
        "Optional dependency 'langchain_openai' not available. Using safe fallbacks for tests."
    )
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from app.config import config
from app.models.embeddings import VectorStoreManager
from app.services.retrieval.cot_orchestrator import CoTOrchestrator, CoTResult
from app.services.complexity_detector import ComplexityDetector, ComplexityLevel

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Structured chat response"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    session_id: str
    mode: str
    processing_time: float
    cot_metrics: Optional[Dict[str, Any]] = None
    verification_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class EnhancedActuarialChatService:
    """
    Enhanced Actuarial Chat Service with CoT orchestration
    """
    
    def __init__(self):
        """Initialize enhanced chat service with CoT components"""
        # Core LLM and vector store (with safe fallback)
        self.llm = None
        self.llm_available = False
        try:
            if ChatOpenAI is not None and getattr(config, "OPENAI_API_KEY", None):
                self.llm = ChatOpenAI(
                    model=getattr(config, "OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.1,
                    api_key=config.OPENAI_API_KEY
                )
                self.llm_available = True
            else:
                logger.warning("LLM not initialized: ChatOpenAI missing or API key not set.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
        
        self.vector_store_manager = VectorStoreManager()
        
        # Session memory management
        self.session_memories = {}
        
        # Initialize complexity detector for conditional verification
        self.complexity_detector = ComplexityDetector()
        
        # Initialize CoT orchestrator with fallback if LLM unavailable
        self.cot_orchestrator = None
        try:
            self.cot_orchestrator = CoTOrchestrator(
                llm_service=self.llm if self.llm_available else None,
                vector_store_manager=self.vector_store_manager,
                enable_verification=True,  # Will be controlled dynamically
                max_context_length=8000,
                debug_mode=config.DEBUG if hasattr(config, 'DEBUG') else False
            )
        except Exception as e:
            logger.warning(f"CoT orchestrator not initialized: {e}")
        
        # Setup fallback chains for non-CoT scenarios
        self._setup_fallback_chains()
        
        logger.info("Enhanced Actuarial Chat Service initialized with CoT orchestration and complexity detection")
    
    def _setup_fallback_chains(self):
        """Setup fallback chains for external questions and errors"""
        try:
            # External question chain (for general actuarial discussions)
            external_prompt = PromptTemplate(
                input_variables=["question", "chat_history"],
                template=self._get_external_prompt_template()
            )
            
            self.external_qa_chain = LLMChain(
                llm=self.llm,
                prompt=external_prompt,
                verbose=False
            )
            
            logger.info("Fallback chains setup successfully")
            
        except Exception as e:
            logger.error(f"Error setting up fallback chains: {str(e)}")
            raise
    
    def _get_external_prompt_template(self) -> str:
        """Get prompt template for external actuarial questions"""
        return """
Anda adalah asisten ahli aktuaria yang berpengalaman. Jawab pertanyaan berikut dengan:

1. **Penjelasan yang jelas dan terstruktur**
2. **Contoh praktis jika relevan**
3. **Referensi ke standar aktuaria yang berlaku**
4. **Saran implementasi jika diperlukan**

**Riwayat Percakapan:**
{chat_history}

**Pertanyaan:**
{question}

**Jawaban:**
Berikan jawaban yang komprehensif dan mudah dipahami, dengan fokus pada aspek praktis dan teoritis yang relevan.
"""
    
    def _get_cot_answer_prompt_template(self) -> str:
        """Get prompt template for generating final answer with CoT context"""
        return """
Anda adalah asisten ahli aktuaria. Berdasarkan konteks yang telah dikurasi melalui proses Chain of Thought, berikan jawaban yang akurat dan komprehensif.

**KONTEKS YANG TELAH DIKURASI:**
{context}

**RIWAYAT PERCAKAPAN:**
{chat_history}

**PERTANYAAN:**
{question}

**INSTRUKSI:**
1. Gunakan HANYA informasi dari konteks yang diberikan
2. Jika informasi tidak cukup, nyatakan dengan jelas
3. Berikan jawaban yang terstruktur dan mudah dipahami
4. Sertakan referensi spesifik ke dokumen sumber
5. Jika ada konflik informasi, jelaskan dan berikan rekomendasi

**JAWABAN:**
"""
    
    async def ask_project(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Main entry point for project-specific questions using CoT orchestration
        
        Args:
            question: User question
            session_id: Session identifier
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing project question with CoT for session {session_id}")
            
            # Ensure session memory
            self._ensure_session_memory(session_id)
            chat_history = self._get_chat_history_string(session_id)
            
            # Analyze question complexity for conditional verification
            complexity_analysis = self.complexity_detector.analyze_complexity(question)
            logger.info(f"Question complexity: {complexity_analysis.level.value} "
                       f"(confidence: {complexity_analysis.confidence:.2f}) "
                       f"- Verification: {'enabled' if complexity_analysis.requires_verification else 'disabled'}")
            
            # Temporarily set verification based on complexity
            original_verification_setting = self.cot_orchestrator.enable_verification
            self.cot_orchestrator.enable_verification = complexity_analysis.requires_verification
            
            # Get available documents for the session
            available_documents = await self._get_available_documents(session_id)
            
            # Process query through CoT orchestration
            cot_result = await self.cot_orchestrator.process_query(
                query=question,
                session_id=session_id,
                available_documents=available_documents,
                chat_history=self._parse_chat_history(chat_history)
            )
            
            # Generate final answer using assembled context
            final_answer = await self._generate_final_answer(
                question, cot_result.context.full_context, chat_history
            )
            
            # Verify answer only if complexity analysis indicates it's needed
            verification_result = None
            if complexity_analysis.requires_verification:
                logger.info("Performing verification for complex question")
                verification_result = await self.cot_orchestrator.verify_answer(
                    final_answer, cot_result
                )
            else:
                logger.info("Skipping verification for simple question - performance optimization")
            
            # Restore original verification setting
            self.cot_orchestrator.enable_verification = original_verification_setting
            
            # Save to memory
            self._save_to_memory(session_id, question, final_answer)
            
            # Extract sources from CoT result
            sources = self._extract_sources_from_cot(cot_result)
            
            # Calculate confidence
            confidence = self._calculate_cot_confidence(cot_result, verification_result)
            
            processing_time = time.time() - start_time
            
            # Create response with complexity analysis metadata
            response = ChatResponse(
                answer=final_answer,
                sources=sources,
                confidence=confidence,
                session_id=session_id,
                mode='cot_orchestrated',
                processing_time=processing_time,
                cot_metrics=self.cot_orchestrator.get_performance_stats(cot_result),
                verification_result=self._format_verification_result(verification_result)
            )
            
            # Add complexity analysis to response
            if hasattr(response, 'cot_metrics') and response.cot_metrics:
                response.cot_metrics['complexity_analysis'] = {
                    'level': complexity_analysis.level.value,
                    'confidence': complexity_analysis.confidence,
                    'verification_used': complexity_analysis.requires_verification,
                    'reasoning': complexity_analysis.reasoning,
                    'keywords_found': complexity_analysis.keywords_found
                }
            
            logger.info(f"CoT processing completed in {processing_time:.2f}s "
                       f"(verification: {'used' if complexity_analysis.requires_verification else 'skipped'})")
            return self._format_response(response)
            
        except Exception as e:
            logger.error(f"Error in CoT ask_project: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fallback to external handling
            return await self._handle_fallback_question(question, session_id, str(e))
    
    async def ask_question(self, question: str, session_id: str) -> Dict[str, Any]:
        """Handle general questions with fallback when CoT unavailable."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing general actuarial question for session {session_id}")
            
            # Ensure session memory
            self._ensure_session_memory(session_id)
            chat_history = self._get_chat_history_string(session_id)
            
            # Perform document retrieval first
            try:
                # Search for relevant documents using vector store
                relevant_docs_with_scores = self.vector_store_manager.similarity_search_with_score(
                    query=question,
                    k=10,  # Get more documents initially
                    session_id=session_id,
                    session_required=False,  # Allow global documents
                    allow_fallback_to_global=True
                )
                
                logger.info(f"Retrieved {len(relevant_docs_with_scores)} documents for question")
                
                # Extract documents and format sources
                sources = []
                context_parts = []
                
                for i, (doc, score) in enumerate(relevant_docs_with_scores):
                    # Add to context (limit to MAX_CONTEXT_LENGTH)
                    snippet = (doc.page_content or "")[:config.MAX_CONTEXT_LENGTH]
                    context_parts.append(f"[DOKUMEN {i+1}] (Skor: {score:.3f})\n{snippet}")
                    
                    # Add to sources
                    source_info = {
                        'content': doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        'metadata': doc.metadata,
                        'score': score
                    }
                    sources.append(source_info)
                
                context = "\n\n".join(context_parts)
                
                # If we have relevant documents, use them in the prompt
                if context_parts:
                    enhanced_prompt = f"""Berdasarkan dokumen yang relevan berikut:

{context}
Pertanyaan: {question}

Berikan jawaban yang komprehensif berdasarkan dokumen di atas. Jika informasi tidak cukup dalam dokumen, tambahkan pengetahuan umum aktuaria yang relevan."""
                    
                    result = self.external_qa_chain.run(
                        question=enhanced_prompt,
                        chat_history=chat_history  # pass separately to avoid duplication
                    )
                    
                    mode = 'rag_enhanced'
                    confidence = min(0.9, 0.5 + (len(sources) * 0.1))  # Higher confidence with more sources
                else:
                    # Fallback to external chain without documents
                    result = self.external_qa_chain.run(
                        question=question,
                        chat_history=chat_history
                    )
                    mode = 'external_actuarial'
                    confidence = 0.7
                    
            except Exception as retrieval_error:
                logger.warning(f"Document retrieval failed: {retrieval_error}")
                # Fallback to external chain without documents
                result = self.external_qa_chain.run(
                    question=question,
                    chat_history=chat_history
                )
                sources = []
                mode = 'external_actuarial'
                confidence = 0.7
            
            # Save to memory
            self._save_to_memory(session_id, question, result)
            
            processing_time = time.time() - start_time
            
            response = ChatResponse(
                answer=result,
                sources=sources,
                confidence=confidence,
                session_id=session_id,
                mode=mode,
                processing_time=processing_time
            )
            
            return self._format_response(response)
            
        except Exception as e:
            logger.error(f"Error in ask_question: {str(e)}")
            return await self._handle_fallback_question(question, session_id, str(e))
    
    async def ask_deep(self, question: str, session_id: str) -> Dict[str, Any]:
        """Answer using strictly session-only RAG context (no global fallback).
        
        Args:
            question: Pertanyaan pengguna
            session_id: ID sesi pengguna spesifik
        
        Returns:
            Dict berisi jawaban, sumber, dan metadata.
        """
        start_time = time.time()
        try:
            logger.info(f"Processing deep session-only query for session {session_id}")
            self._ensure_session_memory(session_id)
            
            # Strict session-only retrieval, disable global fallback
            results = self.vector_store_manager.similarity_search_with_score(
                query=question,
                session_id=session_id,
                k=getattr(config, 'DEFAULT_K', 5),
                session_required=True,
                allow_fallback_to_global=False,
                return_placeholder_on_empty=False
            )
            
            sources = []
            context_chunks = []
            for doc, score in results:
                meta = getattr(doc, 'metadata', {}) or {}
                sources.append({
                    'filename': meta.get('filename') or meta.get('source') or 'unknown',
                    'score': float(score) if isinstance(score, (float, int)) else 0.0,
                    'session_id': meta.get('session_id'),
                    'metadata': {k: v for k, v in (meta.items() if isinstance(meta, dict) else []) if k != 'session_id'}
                })
                # Only add context from the same session
                if meta.get('session_id') == session_id:
                    try:
                        context_chunks.append(getattr(doc, 'page_content', '')[:1000])
                    except Exception:
                        # Defensive: skip malformed docs
                        continue
            
            context = "\n\n".join(context_chunks)
            
            # Build the final answer
            if self.llm_available and self.llm and context:
                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=(
                        "Gunakan hanya konteks berikut dari sesi pengguna untuk menjawab.\n"
                        "Konteks:\n{context}\n\n"
                        "Pertanyaan:\n{question}\n\n"
                        "Jawaban:"
                    )
                )
                chain = LLMChain(llm=self.llm, prompt=prompt, verbose=False)
                try:
                    final_answer = chain.run(context=context, question=question)
                except Exception as e:
                    logger.warning(f"LLMChain failed in ask_deep, using fallback: {e}")
                    final_answer = f"Mode fallback: {len(sources)} sumber dari sesi {session_id}."
            else:
                final_answer = (
                    f"Mode fallback: {len(sources)} sumber ditemukan dari sesi {session_id}. "
                    + ("Konteks diproses tanpa LLM." if context else "Tidak ada konteks ditemukan.")
                )
            
            # Save conversation memory
            self._save_to_memory(session_id, question, final_answer)
            processing_time = time.time() - start_time
            
            response = ChatResponse(
                answer=final_answer,
                sources=sources,
                confidence=0.6 if sources else 0.3,
                session_id=session_id,
                mode='session_only_rag',
                processing_time=processing_time
            )
            return self._format_response(response)
            
        except Exception as e:
            logger.error(f"Error in ask_deep: {e}")
            logger.error(traceback.format_exc())
            return self._create_error_response(f"ask_deep failed: {e}", session_id=session_id)

    async def _get_available_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get available documents for the session including global documents
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of document metadata dictionaries
        """
        try:
            # Get all documents for session (includes both session-specific and global)
            all_documents = self.vector_store_manager.list_documents_for_session(
                session_id=session_id, 
                include_global=True
            )
            
            # Format documents for CoT orchestrator
            formatted_docs = []
            for doc in all_documents:
                formatted_docs.append({
                    'filename': doc.get('filename', 'unknown'),
                    'doc_type': 'global' if doc.get('is_global', False) else 'session',
                    'session_id': doc.get('session_id'),
                    'is_global': doc.get('is_global', False),
                    'metadata': doc.get('metadata', {}),
                    'content_preview': doc.get('content_preview', '')
                })
            
            # Ensure README.md and INDEX.md are prioritized if they exist
            readme_docs = [doc for doc in formatted_docs if 'README.md' in doc['filename']]
            index_docs = [doc for doc in formatted_docs if 'INDEX.md' in doc['filename']]
            other_docs = [doc for doc in formatted_docs if 'README.md' not in doc['filename'] and 'INDEX.md' not in doc['filename']]
            
            # Prioritize README.md and INDEX.md at the beginning
            prioritized_docs = readme_docs + index_docs + other_docs
            
            logger.info(f"Found {len(prioritized_docs)} available documents for session {session_id}")
            logger.info(f"README.md documents: {len(readme_docs)}, INDEX.md documents: {len(index_docs)}")
            
            return prioritized_docs
            
        except Exception as e:
            logger.error(f"Error getting available documents: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def _generate_final_answer(
        self, 
        question: str, 
        context: str, 
        chat_history: str
    ) -> str:
        """
        Generate final answer using assembled context
        
        Args:
            question: Original question
            context: Assembled context from CoT
            chat_history: Chat history string
            
        Returns:
            Generated answer string
        """
        try:
            prompt = PromptTemplate(
                input_variables=["question", "context", "chat_history"],
                template=self._get_cot_answer_prompt_template()
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=False)
            
            answer = chain.run(
                question=question,
                context=context,
                chat_history=chat_history
            )
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            return "Maaf, terjadi kesalahan saat menghasilkan jawaban final."
    
    def _extract_sources_from_cot(self, cot_result: CoTResult) -> List[Dict[str, Any]]:
        """Extract and format sources from CoT result with robust error handling"""
        sources = []
        seen_sources = set()
        
        try:
            # Debug logging untuk struktur CoT result
            logger.info(f"=== DEBUG: CoT Result Structure ===")
            logger.info(f"CoT result type: {type(cot_result)}")
            logger.info(f"Has reranking: {hasattr(cot_result, 'reranking')}")
            
            if not hasattr(cot_result, 'reranking') or not cot_result.reranking:
                logger.warning("CoT result has no reranking data")
                return []
            
            logger.info(f"Reranking type: {type(cot_result.reranking)}")
            logger.info(f"Has reranked_chunks: {hasattr(cot_result.reranking, 'reranked_chunks')}")
            
            if not hasattr(cot_result.reranking, 'reranked_chunks'):
                logger.warning("Reranking has no reranked_chunks attribute")
                return []
            
            chunks = cot_result.reranking.reranked_chunks
            if not chunks:
                logger.warning("No reranked chunks available")
                return []
                
            logger.info(f"Reranked chunks count: {len(chunks)}")
            logger.info(f"Reranked chunks type: {type(chunks)}")
            
            # Debug first few chunks with enhanced type checking
            for i, chunk in enumerate(chunks[:3]):
                logger.info(f"Chunk {i} type: {type(chunk)}")
                logger.info(f"Chunk {i} attributes: {dir(chunk)}")
                
                # Safe attribute checking
                has_content = hasattr(chunk, 'content')
                has_page_content = hasattr(chunk, 'page_content')
                has_metadata = hasattr(chunk, 'metadata')
                
                logger.info(f"Chunk {i}: has_content={has_content}, has_page_content={has_page_content}, has_metadata={has_metadata}")
                
                if isinstance(chunk, str):
                    logger.info(f"Chunk {i}: string value={chunk[:100]}...")
                elif isinstance(chunk, dict):
                    logger.info(f"Chunk {i}: dict keys={list(chunk.keys())}")
                    if hasattr(chunk, '__dict__'):
                        logger.info(f"Chunk {i}: dict={chunk.__dict__}")
                    else:
                        logger.info(f"Chunk {i}: value={str(chunk)[:200]}")
                else:
                    # For objects, try to get string representation safely
                    try:
                        chunk_str = str(chunk)[:200]
                        logger.info(f"Chunk {i}: object string={chunk_str}")
                    except Exception as e:
                        logger.warning(f"Chunk {i}: could not convert to string: {e}")
            
            logger.info(f"Extracting sources from {len(chunks)} reranked chunks")
            
            # Extract from reranked chunks with enhanced error handling
            for i, chunk in enumerate(chunks):
                try:
                    logger.debug(f"Processing chunk {i}: {type(chunk)}")
                    
                    # Initialize variables
                    content = None
                    metadata = {}
                    document_name = "Unknown"
                    score = 0.0
                    
                    # Enhanced chunk type handling with defensive programming
                    if chunk is None:
                        logger.warning(f"Chunk {i}: is None, skipping")
                        continue
                    
                    # Handle RetrievedChunk objects properly
                    elif hasattr(chunk, 'metadata') and hasattr(chunk, 'content'):
                        # This is a RetrievedChunk object
                        logger.debug(f"Chunk {i}: RetrievedChunk object")
                        try:
                            metadata = getattr(chunk, 'metadata', {})
                            content = getattr(chunk, 'content', '')
                            document_name = getattr(chunk, 'document_name', metadata.get('filename', 'Unknown'))
                            score = getattr(chunk, 'score', 0.0)
                            logger.debug(f"Chunk {i}: document_name={document_name}, score={score}")
                        except Exception as e:
                            logger.error(f"Chunk {i}: Error extracting RetrievedChunk attributes: {e}")
                            continue
                        
                    elif hasattr(chunk, 'page_content'):
                        # This is a Document object
                        logger.debug(f"Chunk {i}: Document object")
                        try:
                            metadata = getattr(chunk, 'metadata', {})
                            content = getattr(chunk, 'page_content', '')
                            document_name = metadata.get('filename', 'Unknown')
                            score = getattr(chunk, 'relevance_score', 0.0)
                            logger.debug(f"Chunk {i}: document_name={document_name}, score={score}")
                        except Exception as e:
                            logger.error(f"Chunk {i}: Error extracting Document attributes: {e}")
                            continue
                        
                    elif isinstance(chunk, str):
                        # Handle string chunks - skip them as they don't have proper metadata
                        logger.warning(f"Chunk {i}: Skipping string chunk: {chunk[:100]}...")
                        continue
                        
                    elif isinstance(chunk, dict):
                        # Handle dictionary chunks
                        logger.debug(f"Chunk {i}: Dictionary chunk")
                        try:
                            content = chunk.get('content', chunk.get('page_content', ''))
                            metadata = chunk.get('metadata', {})
                            document_name = chunk.get('document_name', chunk.get('filename', metadata.get('filename', 'Unknown')))
                            score = chunk.get('score', chunk.get('relevance_score', 0.0))
                            logger.debug(f"Chunk {i}: document_name={document_name}, score={score}")
                        except Exception as e:
                            logger.error(f"Chunk {i}: Error extracting dict attributes: {e}")
                            continue
                            
                    else:
                        # Unknown chunk type - try to handle gracefully
                        logger.warning(f"Chunk {i}: Unknown chunk type: {type(chunk)}")
                        logger.warning(f"Chunk {i}: Available attributes: {dir(chunk)}")
                        
                        # Try to extract content safely
                        try:
                            if hasattr(chunk, 'content'):
                                content = getattr(chunk, 'content', '')
                                metadata = getattr(chunk, 'metadata', {})
                                document_name = getattr(chunk, 'document_name', 'Unknown')
                                score = getattr(chunk, 'score', 0.0)
                            else:
                                # Last resort - convert to string
                                content = str(chunk)
                                metadata = {'source': 'unknown', 'type': str(type(chunk))}
                                document_name = 'Unknown'
                                score = 0.0
                        except Exception as e:
                            logger.error(f"Chunk {i}: Error in fallback extraction: {e}")
                            continue
                    
                    # Validate extracted data
                    if not content:
                        logger.warning(f"Chunk {i}: No content extracted, skipping")
                        continue
                    
                    if not isinstance(metadata, dict):
                        logger.warning(f"Chunk {i}: Invalid metadata type {type(metadata)}, using empty dict")
                        metadata = {}
                    
                    # Create unique source key
                    chunk_id = metadata.get('chunk_id', metadata.get('chunk_index', i))
                    source_key = f"{document_name}_{chunk_id}"
                    logger.debug(f"Chunk {i}: source_key={source_key}")
                    
                    if source_key not in seen_sources:
                        # Create full content for AI processing instead of preview
                        try:
                            # Use full content for AI processing - no truncation
                            full_content = content.strip() if content else "Content not available"
                            
                            # Add document metadata for better context
                            section_info = metadata.get('section_heading', metadata.get('section', 'General'))
                            if section_info and section_info != 'Unknown Section':
                                document_display = f"{document_name} - {section_info}"
                            else:
                                document_display = document_name
                                
                            # Format for AI processing: "filename: full content"
                            ai_ready_content = f"{document_display}: {full_content}"
                            
                        except Exception as e:
                            logger.warning(f"Chunk {i}: Error creating full content: {e}")
                            ai_ready_content = f"{document_name}: Content processing error"
                        
                        # Ensure session_id is included from the CoT result context
                        session_id = metadata.get('session_id') or getattr(cot_result, 'session_id', None)
                        
                        # Create source entry with full content for AI processing
                        try:
                            source_entry = {
                                'document_name': str(document_name) if document_name else 'Unknown',
                                'full_content': str(full_content) if full_content else 'No content available',
                                'ai_ready_format': str(ai_ready_content),  # New field for AI processing
                                'section': str(section_info) if section_info else 'General',
                                'relevance_score': float(score) if isinstance(score, (int, float)) else 0.0,
                                'metadata': dict(metadata) if isinstance(metadata, dict) else {},
                                'chunk_id': str(chunk_id) if chunk_id is not None else str(i),
                                'session_id': str(session_id) if session_id else None,
                                # Keep preview for backward compatibility if needed
                                'preview': str(full_content[:200] + "..." if len(full_content) > 200 else full_content)
                            }
                            
                            # Validate source entry
                            if source_entry['document_name'] and source_entry['full_content']:
                                sources.append(source_entry)
                                seen_sources.add(source_key)
                                logger.debug(f"Added source with full content: {source_entry['document_name']}")
                            else:
                                logger.warning(f"Chunk {i}: Invalid source entry, skipping")
                                
                        except Exception as e:
                            logger.error(f"Chunk {i}: Error creating source entry: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    logger.error(f"Chunk type: {type(chunk)}")
                    logger.error(f"Chunk attributes: {dir(chunk) if hasattr(chunk, '__dict__') else 'No attributes'}")
                    continue
            
            logger.info(f"Successfully extracted {len(sources)} sources from CoT result")
            return sources
            
        except Exception as e:
            logger.error(f"Critical error in _extract_sources_from_cot: {e}")
            logger.error(f"CoT result type: {type(cot_result)}")
            logger.error(f"CoT result attributes: {dir(cot_result) if hasattr(cot_result, '__dict__') else 'No attributes'}")
            return []
    
    def _calculate_cot_confidence(
        self, 
        cot_result: CoTResult, 
        verification_result: Optional[Any] = None
    ) -> float:
        """Calculate confidence score based on CoT metrics and verification"""
        try:
            base_confidence = 0.5
            
            # Factor in orchestration confidence
            if hasattr(cot_result.orchestration, 'confidence'):
                base_confidence += cot_result.orchestration.confidence * 0.2
            
            # Factor in filtering confidence
            if hasattr(cot_result.filtering, 'confidence'):
                base_confidence += cot_result.filtering.confidence * 0.1
            
            # Factor in number of relevant chunks
            if cot_result.chunks_retrieved > 0:
                base_confidence += min(cot_result.chunks_retrieved / 10, 0.2)
            
            # Factor in verification result
            if verification_result and hasattr(verification_result, 'overall_score'):
                base_confidence += verification_result.overall_score * 0.1
            
            return min(base_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating CoT confidence: {str(e)}")
            return 0.5
    
    def _format_verification_result(self, verification_result: Optional[Any]) -> Optional[Dict[str, Any]]:
        """Format verification result for response"""
        if not verification_result:
            return {
                'passed': False,
                'score': 0.0,
                'notes': ['Verification not performed'],
                'citation_issues': 0
            }
            
        try:
            # Handle VerificationResult object
            if hasattr(verification_result, 'is_valid'):
                # Safely extract notes from issues
                issues = getattr(verification_result, 'issues', [])
                notes = []
                for issue in issues[:3]:  # Top 3 issues
                    try:
                        if hasattr(issue, 'suggestion'):
                            notes.append(issue.suggestion)
                        elif isinstance(issue, str):
                            notes.append(issue)
                        elif isinstance(issue, dict):
                            notes.append(issue.get('suggestion', str(issue)))
                        else:
                            notes.append(str(issue))
                    except Exception as e:
                        logger.warning(f"Error extracting issue note: {e}")
                        notes.append(f"Issue extraction error: {str(e)}")
                
                return {
                    'passed': verification_result.is_valid,
                    'score': getattr(verification_result, 'confidence_score', 0.0),
                    'citation_coverage': getattr(verification_result, 'citation_coverage', 0.0),
                    'citation_issues': len(issues),
                    'verified_claims': len(getattr(verification_result, 'verified_claims', [])),
                    'total_claims': getattr(verification_result, 'total_claims', 0),
                    'notes': notes
                }
            
            # Handle dict format
            elif isinstance(verification_result, dict):
                return {
                    'passed': verification_result.get('is_valid', False),
                    'score': verification_result.get('confidence_score', 0.0),
                    'citation_coverage': verification_result.get('citation_coverage', 0.0),
                    'citation_issues': len(verification_result.get('issues', [])),
                    'notes': verification_result.get('verification_notes', ['No verification details available'])
                }
            
            # Fallback for unknown format
            else:
                logger.warning(f"Unknown verification result format: {type(verification_result)}")
                return {
                    'passed': False,
                    'score': 0.0,
                    'notes': ['Unknown verification format'],
                    'citation_issues': 0
                }
                
        except Exception as e:
            logger.error(f"Error formatting verification result: {str(e)}")
            return {
                'passed': False,
                'score': 0.0,
                'notes': [f'Verification formatting error: {str(e)}'],
                'citation_issues': 0
            }
    
    async def _handle_fallback_question(
        self, 
        question: str, 
        session_id: str, 
        error_msg: str
    ) -> Dict[str, Any]:
        """Handle fallback when CoT processing fails"""
        try:
            logger.warning(f"Using fallback for session {session_id}: {error_msg}")
            
            # Ensure session memory
            self._ensure_session_memory(session_id)
            chat_history = self._get_chat_history_string(session_id)
            
            # Use external chain as fallback
            result = self.external_qa_chain.run(
                question=question,
                chat_history=chat_history
            )
            
            # Save to memory
            self._save_to_memory(session_id, question, result)
            
            response = ChatResponse(
                answer=result,
                sources=[],
                confidence=0.4,  # Lower confidence for fallback
                session_id=session_id,
                mode='fallback_external',
                processing_time=0.0,
                error=f"CoT processing failed: {error_msg}"
            )
            
            return self._format_response(response)
            
        except Exception as e:
            logger.error(f"Fallback handling also failed: {str(e)}")
            return self._create_error_response(session_id, str(e))
    
    def _format_response(self, response: ChatResponse) -> Dict[str, Any]:
        """Format ChatResponse to dictionary with enhanced context information"""
        result = {
            'answer': response.answer,
            'sources': response.sources,
            'confidence': response.confidence,
            'session_id': response.session_id,
            'mode': response.mode,
            'processing_time': response.processing_time
        }
        
        # Add enhanced context information
        context_info = {
            'retrieval_metadata': {
                'documents_retrieved': len(response.sources),
                'confidence_level': self._get_confidence_level(response.confidence),
                'processing_mode': response.mode,
                'response_quality': self._assess_response_quality(response)
            }
        }
        
        if response.cot_metrics:
            result['cot_metrics'] = response.cot_metrics
            # Add CoT-specific context information
            context_info['reasoning_context'] = {
                'reasoning_steps_count': response.cot_metrics.get('reasoning_steps_count', 0),
                'paths_evaluated': response.cot_metrics.get('paths_evaluated', 0),
                'verification_used': response.cot_metrics.get('verification_enabled', False),
                'query_expansion_used': response.cot_metrics.get('query_expansion_used', False)
            }
            
        if response.verification_result:
            result['verification'] = response.verification_result
            # Add verification context
            context_info['verification_context'] = {
                'verification_score': response.verification_result.get('score', 0),
                'verification_passed': response.verification_result.get('passed', False),
                'verification_notes': response.verification_result.get('notes', [])
            }
            
        if response.error:
            result['error'] = response.error
            context_info['error_context'] = {
                'has_error': True,
                'fallback_used': 'fallback' in response.mode,
                'error_type': self._classify_error_type(response.error)
            }
        else:
            context_info['error_context'] = {
                'has_error': False,
                'fallback_used': False,
                'error_type': None
            }
        
        # Add document context if sources available
        if response.sources:
            context_info['document_context'] = {
                'source_types': list(set([src.get('type', 'unknown') for src in response.sources])),
                'source_count_by_type': self._count_sources_by_type(response.sources),
                'relevance_scores': [src.get('score', 0) for src in response.sources if 'score' in src]
            }
        
        result['context_info'] = context_info
        
        return result
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get human-readable confidence level"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _assess_response_quality(self, response: ChatResponse) -> str:
        """Assess overall response quality"""
        if response.error:
            return "error"
        
        if response.confidence >= 0.7 and len(response.sources) > 0:
            return "excellent"
        elif response.confidence >= 0.5 and len(response.sources) > 0:
            return "good"
        elif response.confidence >= 0.3:
            return "fair"
        else:
            return "poor"
    
    def _classify_error_type(self, error_msg: str) -> str:
        """Classify error type based on error message"""
        if not error_msg:
            return "unknown"
        
        error_lower = error_msg.lower()
        if "timeout" in error_lower or "time" in error_lower:
            return "timeout"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        elif "api" in error_lower or "openai" in error_lower:
            return "api_error"
        elif "memory" in error_lower:
            return "memory_error"
        elif "processing" in error_lower or "cot" in error_lower:
            return "processing_error"
        else:
            return "general_error"
    
    def _count_sources_by_type(self, sources: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count sources by type"""
        type_counts = {}
        for source in sources:
            source_type = source.get('type', 'unknown')
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
        return type_counts
    
    def _create_error_response(self, error_msg: str, session_id: str = None) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'answer': 'Maaf, terjadi kesalahan saat memproses pertanyaan Anda.',
            'sources': [],
            'confidence': 0.0,
            'session_id': session_id,
            'error': error_msg,
            'mode': 'error'
        }
    
    def _ensure_session_memory(self, session_id: str):
        """Ensure session memory exists"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = ConversationBufferWindowMemory(
                k=getattr(config, 'MEMORY_WINDOW_SIZE', 6),  # configurable window size
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

    def _prune_session_memory(self, session_id: str) -> None:
        """Prune stored messages to keep only the most recent window.
        This prevents unbounded growth of ChatMessageHistory which can crash the server.
        """
        try:
            memory = self.session_memories.get(session_id)
            if not memory:
                return
            max_turns = getattr(config, 'MEMORY_WINDOW_SIZE', 6)
            max_messages = max_turns * 2  # each turn = human + ai
            messages = getattr(memory.chat_memory, 'messages', None)
            if isinstance(messages, list) and len(messages) > max_messages:
                # Keep only the most recent messages in place
                memory.chat_memory.messages = messages[-max_messages:]
        except Exception as e:
            logger.warning(f"Memory pruning failed for session {session_id}: {str(e)}")
    
    def _get_chat_history_string(self, session_id: str) -> str:
        """Format chat history as string with last k turns"""
        try:
            if session_id not in self.session_memories:
                return ""
            memory = self.session_memories[session_id]
            history = memory.chat_memory.messages
            if not history:
                return ""
            
            # Use configurable window (last k turns => 2*k messages)
            max_turns = getattr(config, 'MEMORY_WINDOW_SIZE', 6)
            max_messages = max_turns * 2
            history = history[-max_messages:]
            
            # Format messages
            formatted_history = []
            for msg in history:
                role = "User" if getattr(msg, 'type', '') == 'human' else "Assistant"
                formatted_history.append(f"{role}: {msg.content}")
            return "\n".join(formatted_history)
            
        except Exception as e:
            logger.warning(f"Error formatting chat history: {str(e)}")
            return ""
    
    def _parse_chat_history(self, chat_history: str) -> List[Dict[str, str]]:
        """Parse chat history string to list of messages"""
        try:
            messages = []
            for line in chat_history.split('\n'):
                if ': ' in line:
                    role, content = line.split(': ', 1)
                    messages.append({
                        'role': role.lower(),
                        'content': content
                    })
            return messages
        except Exception as e:
            logger.error(f"Error parsing chat history: {str(e)}")
            return []
    
    def _save_to_memory(self, session_id: str, question: str, answer: str):
        """Save interaction to session memory"""
        try:
            if session_id in self.session_memories:
                memory = self.session_memories[session_id]
                memory.save_context(
                    {"input": question},
                    {"answer": answer}  # Changed from "output" to "answer" to match output_key
                )
                # Prune after saving to prevent unbounded growth
                self._prune_session_memory(session_id)
                logger.debug(f"Successfully saved to memory for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving to memory: {str(e)}")
            logger.error(f"Session ID: {session_id}, Question: {question[:100]}..., Answer type: {type(answer)}")
    
    def clear_memory(self, session_id: str = None) -> bool:
        """Clear conversation memory for a specific session or all sessions.
        
        Args:
            session_id: Session identifier, if None clears all sessions
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if session_id:
                if session_id in self.session_memories:
                    del self.session_memories[session_id]
                    logger.info(f"Cleared memory for session: {session_id}")
                    return True
            else:
                self.session_memories.clear()
                return True
            return False
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False
    
    def get_conversation_history(self, session_id: str = None, random_sample: bool = False, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier, defaults to 'default' if None
            random_sample: Whether to return random sample of conversations (not implemented)
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation dictionaries with question, answer, and timestamp
        """
        try:
            # Use default session if none provided
            if not session_id:
                session_id = 'default'
            
            # Ensure session memory exists
            self._ensure_session_memory(session_id)
            
            # Get memory for the session
            if session_id not in self.session_memories:
                logger.warning(f"No memory found for session_id: {session_id}")
                return []
            
            memory = self.session_memories[session_id]
            messages = memory.chat_memory.messages
            
            if not messages:
                logger.info(f"No messages found for session: {session_id}")
                return []
            
            # Only iterate through recent subset to avoid heavy processing
            recent_messages = messages[-(limit * 2):]
            
            # Convert messages to conversation format
            conversations = []
            current_conversation = {}
            
            for message in recent_messages:
                if hasattr(message, 'type') and hasattr(message, 'content'):
                    if message.type == 'human':
                        # Save previous conversation if exists
                        if current_conversation and 'answer' in current_conversation:
                            conversations.append(current_conversation)
                        
                        # Start new conversation
                        current_conversation = {
                            'question': message.content,
                            'timestamp': getattr(message, 'timestamp', 'unknown')
                        }
                    elif message.type == 'ai' and current_conversation:
                        # Complete current conversation
                        current_conversation['answer'] = message.content
            
            # Add the last conversation if it exists and is complete
            if current_conversation and 'answer' in current_conversation:
                conversations.append(current_conversation)
            
            # Apply limit (get most recent conversations)
            conversations = conversations[-limit:] if len(conversations) > limit else conversations
            
            logger.info(f"Retrieved {len(conversations)} conversations for session: {session_id}")
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting conversation history for session {session_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            return {
                'active_sessions': len(self.session_memories),
                'cot_orchestrator_enabled': True,
                'verification_enabled': self.cot_orchestrator.enable_verification,
                'max_context_length': self.cot_orchestrator.max_context_length,
                'model': config.OPENAI_MODEL
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {}