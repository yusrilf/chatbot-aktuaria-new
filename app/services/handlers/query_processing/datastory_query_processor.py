"""Datastory Query Processor for Document Analysis Questions.

This module handles questions that require analysis of uploaded documents
and datastory generation based on document content.
"""

import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.documents import Document

from app.config import config
from app.utils.preview_extractor import extract_meaningful_preview, get_document_summary_info
from app.utils.performance_monitor import get_global_monitor

logger = logging.getLogger(__name__)

class DatastoryQueryProcessor:
    """Processes questions requiring document analysis and datastory generation."""
    
    def __init__(self, 
                 llm: BaseLanguageModel, 
                 prompt_manager=None, 
                 response_parser=None,
                 retrieval_manager=None):
        """Initialize DatastoryQueryProcessor.
        
        Args:
            llm: Language model for generating responses
            prompt_manager: Manager for prompt templates
            response_parser: Parser for response formatting
            retrieval_manager: Manager for document retrieval
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.response_parser = response_parser
        self.retrieval_manager = retrieval_manager
        
        # Initialize performance monitor
        self.performance_monitor = get_global_monitor()
        
        # Initialize datastory chain
        self._init_datastory_chain()
        
        logger.info("DatastoryQueryProcessor initialized")
    
    def _init_datastory_chain(self):
        """Initialize the datastory processing chain."""
        try:
            if self.prompt_manager and hasattr(self.prompt_manager, 'get_datastory_prompt'):
                datastory_prompt = self.prompt_manager.get_datastory_prompt()
            else:
                # Fallback prompt template
                datastory_prompt = PromptTemplate(
                    input_variables=["question", "context", "chat_history"],
                    template="""
Anda adalah asisten AI yang ahli dalam analisis dokumen dan pembuatan datastory.

Riwayat Percakapan:
{chat_history}

Konteks Dokumen:
{context}

Pertanyaan: {question}

Instruksi:
1. Analisis dokumen yang relevan dalam konteks
2. Berikan jawaban yang komprehensif berdasarkan informasi dokumen
3. Jika memungkinkan, buat datastory yang menarik dan informatif
4. Sertakan referensi ke dokumen sumber
5. Jika informasi tidak cukup, jelaskan keterbatasan dan sarankan langkah selanjutnya

Jawaban:
"""
                )
            
            self.datastory_chain = LLMChain(
                llm=self.llm,
                prompt=datastory_prompt,
                verbose=False
            )
            
            logger.info("Datastory chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing datastory chain: {str(e)}")
            # Create a minimal fallback chain
            fallback_prompt = PromptTemplate(
                input_variables=["question", "context"],
                template="""
Berdasarkan dokumen yang tersedia:
{context}

Pertanyaan: {question}

Jawaban: Saya akan menganalisis dokumen untuk menjawab pertanyaan Anda.
"""
            )
            self.datastory_chain = LLMChain(
                llm=self.llm,
                prompt=fallback_prompt,
                verbose=False
            )
    
    def handle_datastory_question(self, 
                                question: str, 
                                session_id: str, 
                                chat_history: str = "") -> Dict[str, Any]:
        """Handle datastory questions requiring document analysis.
        
        Args:
            question: The question to process
            session_id: Session identifier
            chat_history: Previous chat history
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Start performance monitoring
            self.performance_monitor.start_timer("datastory_question_total")
            
            logger.info(f"Processing datastory question for session {session_id}")
            
            # Step 1: Generate retrieval plan
            self.performance_monitor.start_timer("retrieval_planning")
            
            if self.retrieval_manager:
                retrieval_plan = self.retrieval_manager.generate_retrieval_plan(
                    question=question,
                    session_id=session_id,
                    include_global=True
                )
            else:
                retrieval_plan = None
                
            self.performance_monitor.end_timer("retrieval_planning")
            
            # Step 2: Document retrieval
            self.performance_monitor.start_timer("document_retrieval")
            
            if self.retrieval_manager and retrieval_plan:
                # Retrieve relevant documents
                relevant_docs = self.retrieval_manager.retrieve_documents(
                    plan=retrieval_plan,
                    session_id=session_id
                )
                
                # Prepare context from documents
                context = self.retrieval_manager.prepare_context(relevant_docs)
                
            else:
                logger.warning("Retrieval manager not available, using empty context")
                context = "Tidak ada dokumen yang tersedia untuk analisis."
                relevant_docs = []
                
            self.performance_monitor.end_timer("document_retrieval")
            
            # Step 3: LLM processing
            self.performance_monitor.start_timer("llm_inference")
            
            # Prepare input for the chain
            chain_input = {
                "question": question,
                "context": context,
                "chat_history": chat_history or "Tidak ada riwayat percakapan sebelumnya."
            }
            
            # Generate response using the chain
            raw_response = self.datastory_chain.run(chain_input)
            
            self.performance_monitor.end_timer("llm_inference")
            
            # Step 4: Response parsing
            self.performance_monitor.start_timer("response_parsing")
            
            # Parse response if parser is available
            if self.response_parser:
                try:
                    parsed_response = self.response_parser.parse_datastory_response(
                        raw_response, 
                        relevant_docs
                    )
                except Exception as parse_error:
                    logger.warning(f"Response parsing failed: {str(parse_error)}")
                    parsed_response = self._create_fallback_response(
                        raw_response, 
                        relevant_docs
                    )
            else:
                parsed_response = self._create_fallback_response(
                    raw_response, 
                    relevant_docs
                )
                
            self.performance_monitor.end_timer("response_parsing")
            self.performance_monitor.end_timer("datastory_question_total")
            
            # Get performance metrics
            performance_metrics = self.performance_monitor.get_summary()
            
            # Add metadata
            result = {
                'response': parsed_response,
                'session_id': session_id,
                'question_type': 'datastory',
                'processing_method': 'document_analysis',
                'documents_used': len(relevant_docs),
                'success': True,
                'error': None
            }
            
            # Add performance metrics only if enabled in config
            if config.PERFORMANCE_METRICS_IN_RESPONSE:
                result['performance_metrics'] = performance_metrics
            
            logger.info(f"Datastory question processed successfully for session {session_id} "
                       f"using {len(relevant_docs)} documents")
            return result
            
        except Exception as e:
            self.performance_monitor.end_timer("datastory_question_total")
            error_msg = f"Error processing datastory question: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Return error response
            return {
                'response': {
                    'answer': "Maaf, terjadi kesalahan dalam menganalisis dokumen. Silakan coba lagi atau periksa apakah dokumen sudah diupload dengan benar.",
                    'confidence': 0.0,
                    'sources': [],
                    'type': 'datastory',
                    'error': True
                },
                'session_id': session_id,
                'question_type': 'datastory',
                'processing_method': 'error_fallback',
                'documents_used': 0,
                'success': False,
                'error': error_msg
            }
    
    def _create_fallback_response(self, 
                                raw_response: str, 
                                relevant_docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """Create fallback response structure.
        
        Args:
            raw_response: Raw response from LLM
            relevant_docs: List of relevant documents with scores
            
        Returns:
            Formatted response dictionary
        """
        try:
            # Extract sources from documents
            sources = []
            for doc, score in relevant_docs:
                metadata = doc.metadata or {}
                # Create meaningful preview from document content (extended length for 2-3 sentences)
                preview = extract_meaningful_preview(doc.page_content, max_length=200)
                
                # Format score untuk readability - hindari scientific notation
                formatted_score = float(f"{score:.6f}") if score < 0.001 else round(score, 4)
                
                source_info = {
                    'source': metadata.get('source', 'Unknown'),
                    'score': formatted_score,
                    'preview': preview,  # Meaningful preview with reduced length
                    'chunk_id': metadata.get('chunk_id', None),
                    'session_id': session_id
                }
                sources.append(source_info)
            
            # Calculate confidence based on document availability and scores
            if relevant_docs:
                avg_score = sum(score for _, score in relevant_docs) / len(relevant_docs)
                confidence = min(0.9, max(0.5, avg_score))
            else:
                confidence = 0.3
            
            return {
                'answer': raw_response,
                'confidence': confidence,
                'sources': sources,
                'type': 'datastory',
                'metadata': {
                    'documents_analyzed': len(relevant_docs),
                    'analysis_type': 'document_based',
                    'has_context': len(relevant_docs) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating fallback response: {str(e)}")
            return {
                'answer': raw_response,
                'confidence': 0.5,
                'sources': [],
                'type': 'datastory',
                'error': str(e)
            }
    
    def is_datastory_question(self, question: str) -> bool:
        """Determine if a question requires datastory analysis.
        
        Args:
            question: Question to analyze
            
        Returns:
            True if question requires datastory analysis, False otherwise
        """
        try:
            # Keywords that indicate datastory/document analysis questions
            datastory_keywords = [
                'analisis', 'analyze', 'ringkasan', 'summary', 'rangkuman',
                'dokumen', 'document', 'file', 'data', 'informasi',
                'berdasarkan dokumen', 'dari file', 'dalam dokumen',
                'ceritakan', 'jelaskan tentang', 'apa isi', 'bagaimana',
                'tren', 'trend', 'pola', 'pattern', 'insight',
                'datastory', 'data story', 'visualisasi', 'grafik'
            ]
            
            # Keywords that indicate specific calculations (not datastory)
            calculation_keywords = [
                'hitung', 'calculate', 'kalkulasi', 'rumus', 'formula',
                'berapa', 'how much', 'nilai', 'value', 'hasil perhitungan'
            ]
            
            question_lower = question.lower()
            
            # Check for datastory keywords
            has_datastory_keywords = any(keyword in question_lower for keyword in datastory_keywords)
            
            # Check for calculation keywords (which might override datastory)
            has_calculation_keywords = any(keyword in question_lower for keyword in calculation_keywords)
            
            # Question is datastory if it has datastory keywords but not primarily calculation
            is_datastory = has_datastory_keywords and not has_calculation_keywords
            
            logger.debug(f"Question classification - Datastory: {is_datastory}, "
                        f"Datastory keywords: {has_datastory_keywords}, "
                        f"Calculation keywords: {has_calculation_keywords}")
            
            return is_datastory
            
        except Exception as e:
            logger.error(f"Error classifying datastory question: {str(e)}")
            return False
    
    def generate_datastory_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate a summary datastory from documents.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Dictionary containing datastory summary
        """
        try:
            if not documents:
                return {
                    'summary': 'Tidak ada dokumen yang tersedia untuk dianalisis.',
                    'key_insights': [],
                    'document_count': 0,
                    'confidence': 0.0
                }
            
            # Prepare context from documents
            context_parts = []
            for i, doc in enumerate(documents[:5], 1):  # Limit to first 5 docs
                content = doc.page_content[:500]  # Limit content length
                metadata = doc.metadata or {}
                source = metadata.get('source', f'Document {i}')
                context_parts.append(f"[{source}]: {content}")
            
            context = "\n\n".join(context_parts)
            
            # Generate summary using LLM
            summary_prompt = PromptTemplate(
                input_variables=["context"],
                template="""
Berdasarkan dokumen-dokumen berikut, buatlah ringkasan datastory yang informatif:

{context}

Buatlah:
1. Ringkasan utama dari semua dokumen
2. 3-5 insight kunci yang dapat diambil
3. Pola atau tren yang teridentifikasi

Ringkasan Datastory:
"""
            )
            
            summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
            summary_response = summary_chain.run({"context": context})
            
            return {
                'summary': summary_response,
                'key_insights': self._extract_insights(summary_response),
                'document_count': len(documents),
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Error generating datastory summary: {str(e)}")
            return {
                'summary': f'Terjadi kesalahan dalam membuat ringkasan: {str(e)}',
                'key_insights': [],
                'document_count': len(documents) if documents else 0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _extract_insights(self, summary_text: str) -> List[str]:
        """Extract key insights from summary text.
        
        Args:
            summary_text: Summary text to extract insights from
            
        Returns:
            List of key insights
        """
        try:
            # Simple extraction based on common patterns
            insights = []
            lines = summary_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if any(marker in line.lower() for marker in ['insight', 'penting', 'kunci', 'utama', 'tren']):
                    if len(line) > 10:  # Filter out very short lines
                        insights.append(line)
            
            # If no specific insights found, take meaningful sentences
            if not insights:
                sentences = summary_text.split('.')
                for sentence in sentences[:3]:  # Take first 3 sentences
                    sentence = sentence.strip()
                    if len(sentence) > 20:
                        insights.append(sentence + '.')
            
            return insights[:5]  # Limit to 5 insights
            
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return []
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the processor.
        
        Returns:
            Dictionary containing processor information
        """
        return {
            'processor_type': 'datastory_query',
            'capabilities': [
                'document_analysis',
                'datastory_generation',
                'content_summarization',
                'insight_extraction',
                'multi_document_synthesis'
            ],
            'requirements': [
                'uploaded_documents',
                'document_context',
                'retrieval_system'
            ],
            'llm_available': self.llm is not None,
            'prompt_manager_available': self.prompt_manager is not None,
            'response_parser_available': self.response_parser is not None,
            'retrieval_manager_available': self.retrieval_manager is not None
        }