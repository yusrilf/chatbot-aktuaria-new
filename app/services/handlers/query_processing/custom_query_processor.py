"""Custom Query Processor for Actuarial Chatbot.

This module handles custom questions requiring specialized analysis
with document retrieval and context-aware processing.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from app.config import config
from app.services.handlers.reasoning.query_expansion_service import QueryExpansionService
from app.utils.performance_monitor import get_global_monitor

logger = logging.getLogger(__name__)

class CustomQueryProcessor:
    """Processor for custom actuarial questions with document retrieval."""
    
    def __init__(self, 
                 llm=None, 
                 retrieval_manager=None, 
                 prompt_manager=None, 
                 response_parser=None):
        """Initialize the custom query processor.
        
        Args:
            llm: Language model instance
            retrieval_manager: Manager for document retrieval operations
            prompt_manager: Manager for handling prompts
            response_parser: Parser for processing responses
        """
        self.llm = llm
        self.retrieval_manager = retrieval_manager
        self.prompt_manager = prompt_manager
        self.response_parser = response_parser
        
        # Initialize performance monitor
        self.performance_monitor = get_global_monitor()
        
        # Initialize query expansion service
        self.query_expansion_service = QueryExpansionService(llm=llm) if llm else None
        self.enable_query_expansion = True  # Can be configured
        
        # Initialize chains
        self._initialize_chains()
        
        logger.info("CustomQueryProcessor initialized")

    def _initialize_chains(self):
        """Initialize the custom question processing chains."""
        try:
            if self.prompt_manager and hasattr(self.prompt_manager, 'get_custom_prompt'):
                custom_prompt = self.prompt_manager.get_custom_prompt()
            else:
                # Fallback prompt template
                custom_prompt = PromptTemplate(
                    input_variables=["question", "context", "chat_history"],
                    template="""
Anda adalah asisten AI yang ahli dalam analisis aktuaria dan asuransi.

Riwayat Percakapan:
{chat_history}

Konteks Dokumen:
{context}

Pertanyaan: {question}

Instruksi:
1. Analisis pertanyaan dengan cermat
2. Gunakan konteks dokumen yang relevan
3. Berikan jawaban yang komprehensif dan akurat
4. Sertakan referensi ke dokumen sumber jika ada
5. Jika informasi tidak cukup, jelaskan keterbatasan

Jawaban:
"""
                )
            
            self.custom_chain = LLMChain(
                llm=self.llm,
                prompt=custom_prompt,
                verbose=False
            )
            
            logger.info("Custom question chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing custom chain: {str(e)}")
            # Create a minimal fallback chain
            fallback_prompt = PromptTemplate(
                input_variables=["question"],
                template="Pertanyaan: {question}\n\nJawaban: Maaf, saya mengalami kesulitan teknis. Silakan coba lagi nanti."
            )
            self.custom_chain = LLMChain(
                llm=self.llm,
                prompt=fallback_prompt,
                verbose=False
            )

    def handle_custom_question(self, 
                             question: str, 
                             session_id: str, 
                             chat_history: str = "") -> Dict[str, Any]:
        """Handle custom questions with document retrieval and analysis.
        
        Args:
            question: The question to process
            session_id: Session identifier
            chat_history: Previous chat history
            
        Returns:
            Dictionary containing response and metadata
        """
        # Initialize timer variables
        custom_question_timer = None
        query_expansion_timer = None
        document_retrieval_timer = None
        llm_inference_timer = None
        response_parsing_timer = None
        
        try:
            # Start performance monitoring
            custom_question_timer = self.performance_monitor.start_timer("custom_question_total")
            
            logger.info(f"Processing custom question for session {session_id}")
            
            # Step 1: Query expansion (if enabled)
            expanded_query = question
            if self.enable_query_expansion and self.query_expansion_service:
                query_expansion_timer = self.performance_monitor.start_timer("query_expansion")
                try:
                    expanded_query = self.query_expansion_service.expand_query(question)
                    logger.info(f"Query expanded from '{question}' to '{expanded_query}'")
                except Exception as e:
                    logger.warning(f"Query expansion failed: {str(e)}")
                    expanded_query = question
                finally:
                    if query_expansion_timer:
                        self.performance_monitor.stop_timer(query_expansion_timer)
                        query_expansion_timer = None
            
            # Step 2: Document retrieval
            document_retrieval_timer = self.performance_monitor.start_timer("document_retrieval")
            relevant_docs = []
            context = ""
            
            if self.retrieval_manager:
                try:
                    # Create comprehensive retrieval plan for expanded query
                    retrieval_plan = {
                        'original_question': question,
                        'refined_query': expanded_query,
                        'session_id': session_id,
                        'include_global': True,
                        'search_strategies': [
                            {
                                'type': 'hybrid_search',
                                'query': expanded_query,
                                'k': 8,  # Increased from 5 for better coverage
                                'rerank': True,
                                'description': 'Primary hybrid search with expanded query'
                            },
                            {
                                'type': 'vector_search',
                                'query': question,  # Original question as fallback
                                'k': 5,
                                'rerank': False,
                                'description': 'Fallback vector search with original query'
                            }
                        ]
                    }
                    
                    # Use expanded query for retrieval
                    relevant_docs = self.retrieval_manager.retrieve_documents(
                        retrieval_plan, 
                        session_id=session_id
                    )
                    
                    # Format context from retrieved documents
                    context_parts = []
                    for i, (doc, score) in enumerate(relevant_docs):
                        context_parts.append(f"[DOKUMEN {i+1}] (Skor: {score:.3f})\n{doc.page_content}")
                    
                    context = "\n\n".join(context_parts)
                    logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
                    
                except Exception as e:
                    logger.warning(f"Document retrieval failed: {str(e)}")
                    context = "Tidak ada dokumen relevan yang ditemukan."
            else:
                context = "Sistem retrieval tidak tersedia."
            
            if document_retrieval_timer:
                self.performance_monitor.stop_timer(document_retrieval_timer)
                document_retrieval_timer = None
            
            # Step 3: LLM processing
            llm_inference_timer = self.performance_monitor.start_timer("llm_inference")
            
            # Prepare input for the chain
            chain_input = {
                "question": question,
                "context": context,
                "chat_history": chat_history or "Tidak ada riwayat percakapan sebelumnya."
            }
            
            # Generate response using the chain
            raw_response = self.custom_chain.run(chain_input)
            
            if llm_inference_timer:
                self.performance_monitor.stop_timer(llm_inference_timer)
                llm_inference_timer = None
            
            # Step 4: Response parsing
            response_parsing_timer = self.performance_monitor.start_timer("response_parsing")
            
            # Parse response if parser is available
            if self.response_parser:
                try:
                    parsed_response = self.response_parser.parse_custom_response(raw_response, relevant_docs, session_id)
                except Exception as parse_error:
                    logger.warning(f"Response parsing failed: {str(parse_error)}")
                    parsed_response = {
                        'answer': raw_response,
                        'confidence': 0.8,
                        'sources': [doc[0].metadata.get('source', 'Unknown') for doc in relevant_docs],
                        'reasoning': 'Standard custom processing'
                    }
            else:
                parsed_response = {
                    'answer': raw_response,
                    'confidence': 0.8,
                    'sources': [doc[0].metadata.get('source', 'Unknown') for doc in relevant_docs],
                    'reasoning': 'Standard custom processing'
                }
            
            if response_parsing_timer:
                self.performance_monitor.stop_timer(response_parsing_timer)
                response_parsing_timer = None
                
            if custom_question_timer:
                self.performance_monitor.stop_timer(custom_question_timer)
                custom_question_timer = None
            
            # Get performance metrics
            performance_metrics = self.performance_monitor.get_summary()
            
            # Format final response
            response = {
                'answer': parsed_response['answer'],
                'sources': parsed_response.get('sources', []),
                'confidence': parsed_response.get('confidence', 0.8),
                'reasoning': parsed_response.get('reasoning', 'Custom question processing'),
                'session_id': session_id,
                'query_type': 'custom',
                'documents_retrieved': len(relevant_docs),
                'expanded_query': expanded_query if expanded_query != question else None
            }
            
            # Add performance metrics only if enabled in config
            if config.PERFORMANCE_METRICS_IN_RESPONSE:
                response['performance_metrics'] = performance_metrics
            
            logger.info(f"Custom question processed successfully for session {session_id}")
            return response
            
        except Exception as e:
            # Stop any active timers on error
            if custom_question_timer:
                self.performance_monitor.stop_timer(custom_question_timer)
            if query_expansion_timer:
                self.performance_monitor.stop_timer(query_expansion_timer)
            if document_retrieval_timer:
                self.performance_monitor.stop_timer(document_retrieval_timer)
            if llm_inference_timer:
                self.performance_monitor.stop_timer(llm_inference_timer)
            if response_parsing_timer:
                self.performance_monitor.stop_timer(response_parsing_timer)
                
            logger.error(f"Error handling custom question: {str(e)}")
            return {
                'answer': f"Maaf, terjadi kesalahan dalam memproses pertanyaan custom Anda: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'query_type': 'custom',
                'error': str(e),
                'performance_metrics': self.performance_monitor.get_summary()
            }

    def get_capabilities(self) -> Dict[str, Any]:
        """Get processor capabilities and configuration.
        
        Returns:
            Dictionary containing processor capabilities
        """
        return {
            'processor_type': 'custom',
            'supports_query_expansion': self.enable_query_expansion and self.query_expansion_service is not None,
            'supports_document_retrieval': self.retrieval_manager is not None,
            'supports_response_parsing': self.response_parser is not None,
            'llm_available': self.llm is not None,
            'performance_monitoring': True
        }