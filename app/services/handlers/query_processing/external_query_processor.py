"""External Query Processor for General and External Questions.

This module handles external/general questions that don't require specific
document context or PSAK219 knowledge.
"""

import logging
import traceback
from typing import Dict, Any, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel

logger = logging.getLogger(__name__)

class ExternalQueryProcessor:
    """Processes external and general questions without document context."""
    
    def __init__(self, llm: BaseLanguageModel, prompt_manager=None, response_parser=None):
        """Initialize ExternalQueryProcessor.
        
        Args:
            llm: Language model for generating responses
            prompt_manager: Manager for prompt templates
            response_parser: Parser for response formatting
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.response_parser = response_parser
        
        # Initialize external question chain
        self._init_external_chain()
        
        logger.info("ExternalQueryProcessor initialized")
    
    def _init_external_chain(self):
        """Initialize the external question processing chain."""
        try:
            if self.prompt_manager and hasattr(self.prompt_manager, 'get_external_prompt'):
                external_prompt = self.prompt_manager.get_external_prompt()
            else:
                # Fallback prompt template
                external_prompt = PromptTemplate(
                    input_variables=["question", "chat_history"],
                    template="""
Anda adalah asisten AI yang membantu menjawab pertanyaan umum.

Riwayat Percakapan:
{chat_history}

Pertanyaan: {question}

Jawaban: Berikan jawaban yang informatif dan membantu untuk pertanyaan ini. Jika pertanyaan terkait aktuaria atau asuransi, berikan penjelasan dasar yang mudah dipahami.
"""
                )
            
            self.external_chain = LLMChain(
                llm=self.llm,
                prompt=external_prompt,
                verbose=False
            )
            
            logger.info("External question chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing external chain: {str(e)}")
            # Create a minimal fallback chain
            fallback_prompt = PromptTemplate(
                input_variables=["question"],
                template="Pertanyaan: {question}\n\nJawaban: Maaf, saya mengalami kesulitan teknis. Silakan coba lagi nanti."
            )
            self.external_chain = LLMChain(
                llm=self.llm,
                prompt=fallback_prompt,
                verbose=False
            )
    
    def handle_external_question(self, 
                               question: str, 
                               session_id: str, 
                               chat_history: str = "") -> Dict[str, Any]:
        """Handle external/general questions.
        
        Args:
            question: The question to process
            session_id: Session identifier
            chat_history: Previous chat history
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            logger.info(f"Processing external question for session {session_id}")
            
            # Prepare input for the chain
            chain_input = {
                "question": question,
                "chat_history": chat_history or "Tidak ada riwayat percakapan sebelumnya."
            }
            
            # Generate response using the chain
            raw_response = self.external_chain.run(chain_input)
            
            # Parse response if parser is available
            if self.response_parser:
                try:
                    parsed_response = self.response_parser.parse_external_response(raw_response)
                except Exception as parse_error:
                    logger.warning(f"Response parsing failed: {str(parse_error)}")
                    parsed_response = {
                        'answer': raw_response,
                        'confidence': 0.7,
                        'sources': [],
                        'type': 'external'
                    }
            else:
                parsed_response = {
                    'answer': raw_response,
                    'confidence': 0.8,
                    'sources': [],
                    'type': 'external'
                }
            
            # Add metadata
            result = {
                'response': parsed_response,
                'session_id': session_id,
                'question_type': 'external',
                'processing_method': 'llm_chain',
                'success': True,
                'error': None
            }
            
            logger.info(f"External question processed successfully for session {session_id}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing external question: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Return error response
            return {
                'response': {
                    'answer': "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi atau hubungi administrator.",
                    'confidence': 0.0,
                    'sources': [],
                    'type': 'external',
                    'error': True
                },
                'session_id': session_id,
                'question_type': 'external',
                'processing_method': 'error_fallback',
                'success': False,
                'error': error_msg
            }
    
    def is_external_question(self, question: str) -> bool:
        """Determine if a question should be handled as external.
        
        Args:
            question: Question to analyze
            
        Returns:
            True if question is external, False otherwise
        """
        try:
            # Keywords that indicate document-specific questions
            document_keywords = [
                'dokumen', 'file', 'upload', 'analisis dokumen',
                'dalam dokumen', 'berdasarkan dokumen', 'dari file'
            ]
            
            # Keywords that indicate PSAK219 specific questions
            psak219_keywords = [
                'psak219', 'psak 219', 'imbalan kerja', 'employee benefit',
                'actuarial valuation', 'valuasi aktuaria', 'kewajiban imbalan'
            ]
            
            question_lower = question.lower()
            
            # Check if question contains document-specific keywords
            has_document_keywords = any(keyword in question_lower for keyword in document_keywords)
            
            # Check if question contains PSAK219 keywords
            has_psak219_keywords = any(keyword in question_lower for keyword in psak219_keywords)
            
            # Question is external if it doesn't contain document or PSAK219 keywords
            is_external = not (has_document_keywords or has_psak219_keywords)
            
            logger.debug(f"Question classification - External: {is_external}, "
                        f"Document keywords: {has_document_keywords}, "
                        f"PSAK219 keywords: {has_psak219_keywords}")
            
            return is_external
            
        except Exception as e:
            logger.error(f"Error classifying question: {str(e)}")
            # Default to external if classification fails
            return True
    
    def get_external_response_template(self) -> str:
        """Get template for external responses.
        
        Returns:
            Template string for external responses
        """
        return """
Berdasarkan pengetahuan umum saya, berikut adalah jawaban untuk pertanyaan Anda:

{answer}

Catatan: Ini adalah jawaban umum. Jika Anda memiliki dokumen spesifik atau memerlukan analisis PSAK219, 
silakan upload dokumen atau ajukan pertanyaan yang lebih spesifik.
"""
    
    def format_external_response(self, answer: str, confidence: float = 0.8) -> Dict[str, Any]:
        """Format external response with standard structure.
        
        Args:
            answer: The answer text
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Formatted response dictionary
        """
        try:
            template = self.get_external_response_template()
            formatted_answer = template.format(answer=answer)
            
            return {
                'answer': formatted_answer,
                'confidence': confidence,
                'sources': [],
                'type': 'external',
                'metadata': {
                    'processing_type': 'external_llm',
                    'requires_documents': False,
                    'is_general_knowledge': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error formatting external response: {str(e)}")
            return {
                'answer': answer,
                'confidence': confidence,
                'sources': [],
                'type': 'external',
                'error': str(e)
            }
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the processor.
        
        Returns:
            Dictionary containing processor information
        """
        return {
            'processor_type': 'external_query',
            'capabilities': [
                'general_questions',
                'basic_actuarial_concepts',
                'insurance_fundamentals',
                'mathematical_concepts'
            ],
            'limitations': [
                'no_document_analysis',
                'no_specific_calculations',
                'general_knowledge_only'
            ],
            'llm_available': self.llm is not None,
            'prompt_manager_available': self.prompt_manager is not None,
            'response_parser_available': self.response_parser is not None
        }