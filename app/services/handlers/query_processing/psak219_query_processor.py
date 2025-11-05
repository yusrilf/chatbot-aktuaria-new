"""PSAK219 Query Processor for Actuarial and Employee Benefit Questions.

This module handles questions specifically related to PSAK219 (Employee Benefits),
actuarial calculations, and pension-related queries.
"""

import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.documents import Document
from app.utils.preview_extractor import extract_meaningful_preview, get_document_summary_info

logger = logging.getLogger(__name__)

class PSAK219QueryProcessor:
    """Processes PSAK219 and actuarial-specific questions."""
    
    def __init__(self, 
                 llm: BaseLanguageModel, 
                 prompt_manager=None, 
                 response_parser=None,
                 retrieval_manager=None,
                 psak219_manager=None):
        """Initialize PSAK219QueryProcessor.
        
        Args:
            llm: Language model for generating responses
            prompt_manager: Manager for prompt templates
            response_parser: Parser for response formatting
            retrieval_manager: Manager for document retrieval
            psak219_manager: Specialized PSAK219 manager
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.response_parser = response_parser
        self.retrieval_manager = retrieval_manager
        self.psak219_manager = psak219_manager
        
        # Initialize PSAK219 processing chains
        self._init_psak219_chains()
        
        logger.info("PSAK219QueryProcessor initialized")
    
    def _init_psak219_chains(self):
        """Initialize PSAK219-specific processing chains."""
        try:
            # PSAK219 analysis chain
            if self.prompt_manager and hasattr(self.prompt_manager, 'get_psak219_prompt'):
                psak219_prompt = self.prompt_manager.get_psak219_prompt()
            else:
                psak219_prompt = PromptTemplate(
                    input_variables=["question", "context", "chat_history"],
                    template="""
Anda adalah ahli aktuaria yang sangat berpengalaman dalam PSAK219 (Imbalan Kerja).

Riwayat Percakapan:
{chat_history}

Konteks PSAK219 dan Data Aktuaria:
{context}

Pertanyaan PSAK219: {question}

Instruksi:
1. Analisis pertanyaan dalam konteks PSAK219 dan standar aktuaria
2. Berikan jawaban yang akurat berdasarkan standar PSAK219
3. Jika melibatkan perhitungan aktuaria, jelaskan metodologi yang digunakan
4. Sertakan referensi ke paragraf PSAK219 yang relevan jika ada
5. Berikan contoh praktis jika memungkinkan
6. Jelaskan implikasi untuk pelaporan keuangan

Jawaban Ahli PSAK219:
"""
                )
            
            self.psak219_chain = LLMChain(
                llm=self.llm,
                prompt=psak219_prompt,
                verbose=False
            )
            
            # Actuarial calculation chain
            actuarial_prompt = PromptTemplate(
                input_variables=["question", "context", "calculation_data"],
                template="""
Anda adalah aktuaris bersertifikat yang ahli dalam perhitungan imbalan kerja.

Data dan Konteks:
{context}

Data Perhitungan:
{calculation_data}

Pertanyaan Aktuaria: {question}

Instruksi:
1. Identifikasi jenis perhitungan aktuaria yang diperlukan
2. Tentukan asumsi aktuaria yang relevan
3. Lakukan perhitungan step-by-step sesuai standar aktuaria
4. Berikan interpretasi hasil dalam konteks PSAK219
5. Jelaskan sensitivitas terhadap perubahan asumsi

Hasil Perhitungan Aktuaria:
"""
            )
            
            self.actuarial_chain = LLMChain(
                llm=self.llm,
                prompt=actuarial_prompt,
                verbose=False
            )
            
            logger.info("PSAK219 processing chains initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing PSAK219 chains: {str(e)}")
            # Create minimal fallback chains
            fallback_prompt = PromptTemplate(
                input_variables=["question", "context"],
                template="""
Pertanyaan PSAK219: {question}
Konteks: {context}

Jawaban: Saya akan menganalisis pertanyaan PSAK219 Anda berdasarkan informasi yang tersedia.
"""
            )
            self.psak219_chain = LLMChain(llm=self.llm, prompt=fallback_prompt)
            self.actuarial_chain = self.psak219_chain
    
    def handle_psak219_question(self, 
                              question: str, 
                              session_id: str, 
                              chat_history: str = "") -> Dict[str, Any]:
        """Handle PSAK219-specific questions.
        
        Args:
            question: The PSAK219 question to process
            session_id: Session identifier
            chat_history: Previous chat history
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            logger.info(f"Processing PSAK219 question for session {session_id}")
            
            # Classify PSAK219 question type
            question_type = self._classify_psak219_question(question)
            
            # Retrieve PSAK219-specific documents
            if self.retrieval_manager:
                # Generate specialized retrieval plan for PSAK219
                retrieval_plan = self._generate_psak219_retrieval_plan(
                    question, session_id, question_type
                )
                
                # Retrieve relevant documents
                relevant_docs = self.retrieval_manager.retrieve_documents(
                    plan=retrieval_plan,
                    session_id=session_id
                )
                
                # Prepare PSAK219-specific context
                context = self._prepare_psak219_context(relevant_docs, question_type)
                
            else:
                logger.warning("Retrieval manager not available for PSAK219 processing")
                context = "Tidak ada dokumen PSAK219 yang tersedia untuk analisis."
                relevant_docs = []
            
            # Get additional PSAK219 data if manager is available
            psak219_data = self._get_psak219_specific_data(question, question_type)
            
            # Process based on question type
            if question_type == 'actuarial_calculation':
                response = self._handle_actuarial_calculation(
                    question, context, psak219_data, relevant_docs
                )
            elif question_type == 'standard_interpretation':
                response = self._handle_standard_interpretation(
                    question, context, chat_history, relevant_docs
                )
            elif question_type == 'valuation_method':
                response = self._handle_valuation_method(
                    question, context, psak219_data, relevant_docs
                )
            else:
                response = self._handle_general_psak219(
                    question, context, chat_history, relevant_docs
                )
            
            # Add PSAK219-specific metadata
            result = {
                'response': response,
                'session_id': session_id,
                'question_type': 'psak219',
                'psak219_subtype': question_type,
                'processing_method': 'psak219_specialized',
                'documents_used': len(relevant_docs),
                'psak219_data_used': psak219_data is not None,
                'success': True,
                'error': None
            }
            
            logger.info(f"PSAK219 question processed successfully for session {session_id} "
                       f"(type: {question_type}, docs: {len(relevant_docs)})")
            return result
            
        except Exception as e:
            error_msg = f"Error processing PSAK219 question: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Return error response
            return {
                'response': {
                    'answer': "Maaf, terjadi kesalahan dalam memproses pertanyaan PSAK219. Silakan coba lagi atau konsultasikan dengan ahli aktuaria.",
                    'confidence': 0.0,
                    'sources': [],
                    'type': 'psak219',
                    'error': True
                },
                'session_id': session_id,
                'question_type': 'psak219',
                'processing_method': 'error_fallback',
                'success': False,
                'error': error_msg
            }
    
    def _classify_psak219_question(self, question: str) -> str:
        """Classify the type of PSAK219 question.
        
        Args:
            question: Question to classify
            
        Returns:
            PSAK219 question type
        """
        try:
            question_lower = question.lower()
            
            # Actuarial calculation keywords
            calc_keywords = [
                'hitung', 'calculate', 'valuasi', 'valuation', 'dbo',
                'defined benefit obligation', 'service cost', 'interest cost',
                'actuarial gain', 'actuarial loss', 'present value'
            ]
            
            # Standard interpretation keywords
            standard_keywords = [
                'psak219', 'standar', 'standard', 'paragraf', 'paragraph',
                'ketentuan', 'requirement', 'pengakuan', 'recognition',
                'pengukuran', 'measurement', 'penyajian', 'presentation'
            ]
            
            # Valuation method keywords
            method_keywords = [
                'metode', 'method', 'pendekatan', 'approach', 'asumsi',
                'assumption', 'discount rate', 'mortality', 'turnover',
                'projected unit credit', 'puc'
            ]
            
            # Reporting keywords
            reporting_keywords = [
                'laporan', 'report', 'disclosure', 'pengungkapan',
                'catatan', 'notes', 'neraca', 'balance sheet',
                'laba rugi', 'income statement', 'oci'
            ]
            
            # Check for each type
            if any(keyword in question_lower for keyword in calc_keywords):
                return 'actuarial_calculation'
            elif any(keyword in question_lower for keyword in method_keywords):
                return 'valuation_method'
            elif any(keyword in question_lower for keyword in reporting_keywords):
                return 'reporting_disclosure'
            elif any(keyword in question_lower for keyword in standard_keywords):
                return 'standard_interpretation'
            else:
                return 'general_psak219'
                
        except Exception as e:
            logger.error(f"Error classifying PSAK219 question: {str(e)}")
            return 'general_psak219'
    
    def _generate_psak219_retrieval_plan(self, 
                                       question: str, 
                                       session_id: str, 
                                       question_type: str) -> Dict[str, Any]:
        """Generate specialized retrieval plan for PSAK219 questions.
        
        Args:
            question: The question
            session_id: Session identifier
            question_type: Type of PSAK219 question
            
        Returns:
            Specialized retrieval plan
        """
        try:
            plan = {
                'original_question': question,
                'refined_query': question,
                'session_id': session_id,
                'question_type': question_type,
                'search_strategies': []
            }
            
            # Add PSAK219-specific search
            plan['search_strategies'].append({
                'type': 'psak219_search',
                'query': question,
                'k': 5,
                'focus': question_type
            })
            
            # Add actuarial document search if calculation-related
            if question_type == 'actuarial_calculation':
                plan['search_strategies'].append({
                    'type': 'actuarial_search',
                    'query': question,
                    'k': 3,
                    'focus': 'calculations'
                })
            
            # Add standard document search
            plan['search_strategies'].append({
                'type': 'standard_search',
                'query': question,
                'k': 3,
                'focus': 'psak219_standard'
            })
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating PSAK219 retrieval plan: {str(e)}")
            return {
                'original_question': question,
                'session_id': session_id,
                'search_strategies': [{
                    'type': 'hybrid_search',
                    'query': question,
                    'k': 5
                }]
            }
    
    def _prepare_psak219_context(self, 
                               relevant_docs: List[Tuple[Document, float]], 
                               question_type: str) -> str:
        """Prepare PSAK219-specific context from documents.
        
        Args:
            relevant_docs: Relevant documents with scores
            question_type: Type of PSAK219 question
            
        Returns:
            Formatted PSAK219 context
        """
        try:
            if not relevant_docs:
                return "Tidak ada dokumen PSAK219 yang relevan ditemukan."
            
            context_parts = []
            
            for i, (doc, score) in enumerate(relevant_docs, 1):
                metadata = doc.metadata or {}
                source = metadata.get('source', 'Unknown')
                
                # Check if document is PSAK219-related
                is_psak219 = any(keyword in source.lower() for keyword in 
                               ['psak219', 'psak 219', 'employee benefit', 'imbalan kerja'])
                
                content = doc.page_content.strip()
                if len(content) > 800:  # Longer content for PSAK219
                    content = content[:800] + "..."
                
                # Format with PSAK219 emphasis
                doc_type = "[PSAK219]" if is_psak219 else "[Aktuaria]"
                context_part = f"{doc_type} Dokumen {i} (Sumber: {source}, Relevansi: {score:.3f})\n{content}\n"
                context_parts.append(context_part)
            
            context = "\n".join(context_parts)
            
            # Add question type specific guidance
            if question_type == 'actuarial_calculation':
                context += "\n\n[PANDUAN] Fokus pada data numerik, asumsi aktuaria, dan metodologi perhitungan."
            elif question_type == 'standard_interpretation':
                context += "\n\n[PANDUAN] Fokus pada ketentuan standar, paragraf spesifik, dan interpretasi resmi."
            
            return context
            
        except Exception as e:
            logger.error(f"Error preparing PSAK219 context: {str(e)}")
            return "Terjadi kesalahan dalam mempersiapkan konteks PSAK219."
    
    def _get_psak219_specific_data(self, question: str, question_type: str) -> Optional[Dict[str, Any]]:
        """Get PSAK219-specific data from specialized manager.
        
        Args:
            question: The question
            question_type: Type of question
            
        Returns:
            PSAK219-specific data or None
        """
        try:
            if not self.psak219_manager:
                return None
            
            # Get relevant PSAK219 data based on question type
            if question_type == 'actuarial_calculation':
                return self.psak219_manager.get_calculation_data(question)
            elif question_type == 'standard_interpretation':
                return self.psak219_manager.get_standard_references(question)
            elif question_type == 'valuation_method':
                return self.psak219_manager.get_method_guidance(question)
            else:
                return self.psak219_manager.get_general_data(question)
                
        except Exception as e:
            logger.error(f"Error getting PSAK219 specific data: {str(e)}")
            return None
    
    def _handle_actuarial_calculation(self, 
                                    question: str, 
                                    context: str, 
                                    psak219_data: Optional[Dict[str, Any]],
                                    relevant_docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """Handle actuarial calculation questions.
        
        Args:
            question: The calculation question
            context: Document context
            psak219_data: PSAK219-specific data
            relevant_docs: Relevant documents
            
        Returns:
            Formatted response
        """
        try:
            # Prepare calculation data
            calculation_data = "Tidak ada data perhitungan spesifik."
            if psak219_data:
                calculation_data = str(psak219_data)
            
            # Generate response
            chain_input = {
                "question": question,
                "context": context,
                "calculation_data": calculation_data
            }
            
            raw_response = self.actuarial_chain.run(chain_input)
            
            return self._format_psak219_response(
                raw_response, relevant_docs, 'actuarial_calculation'
            )
            
        except Exception as e:
            logger.error(f"Error handling actuarial calculation: {str(e)}")
            return self._create_psak219_error_response('actuarial_calculation', str(e))
    
    def _handle_standard_interpretation(self, 
                                      question: str, 
                                      context: str, 
                                      chat_history: str,
                                      relevant_docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """Handle standard interpretation questions.
        
        Args:
            question: The interpretation question
            context: Document context
            chat_history: Chat history
            relevant_docs: Relevant documents
            
        Returns:
            Formatted response
        """
        try:
            chain_input = {
                "question": question,
                "context": context,
                "chat_history": chat_history
            }
            
            raw_response = self.psak219_chain.run(chain_input)
            
            return self._format_psak219_response(
                raw_response, relevant_docs, 'standard_interpretation'
            )
            
        except Exception as e:
            logger.error(f"Error handling standard interpretation: {str(e)}")
            return self._create_psak219_error_response('standard_interpretation', str(e))
    
    def _handle_valuation_method(self, 
                               question: str, 
                               context: str, 
                               psak219_data: Optional[Dict[str, Any]],
                               relevant_docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """Handle valuation method questions.
        
        Args:
            question: The method question
            context: Document context
            psak219_data: PSAK219-specific data
            relevant_docs: Relevant documents
            
        Returns:
            Formatted response
        """
        try:
            # Add method-specific context
            method_context = context
            if psak219_data:
                method_context += f"\n\n[DATA METODE]\n{psak219_data}"
            
            chain_input = {
                "question": question,
                "context": method_context,
                "chat_history": "Pertanyaan tentang metode valuasi aktuaria."
            }
            
            raw_response = self.psak219_chain.run(chain_input)
            
            return self._format_psak219_response(
                raw_response, relevant_docs, 'valuation_method'
            )
            
        except Exception as e:
            logger.error(f"Error handling valuation method: {str(e)}")
            return self._create_psak219_error_response('valuation_method', str(e))
    
    def _handle_general_psak219(self, 
                              question: str, 
                              context: str, 
                              chat_history: str,
                              relevant_docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """Handle general PSAK219 questions.
        
        Args:
            question: The general question
            context: Document context
            chat_history: Chat history
            relevant_docs: Relevant documents
            
        Returns:
            Formatted response
        """
        try:
            chain_input = {
                "question": question,
                "context": context,
                "chat_history": chat_history
            }
            
            raw_response = self.psak219_chain.run(chain_input)
            
            return self._format_psak219_response(
                raw_response, relevant_docs, 'general_psak219'
            )
            
        except Exception as e:
            logger.error(f"Error handling general PSAK219: {str(e)}")
            return self._create_psak219_error_response('general_psak219', str(e))
    
    def _format_psak219_response(self, 
                               raw_response: str, 
                               relevant_docs: List[Tuple[Document, float]], 
                               response_type: str) -> Dict[str, Any]:
        """Format PSAK219 response with specialized metadata.
        
        Args:
            raw_response: Raw LLM response
            relevant_docs: Relevant documents
            response_type: Type of PSAK219 response
            
        Returns:
            Formatted response dictionary
        """
        try:
            # Extract sources with PSAK219 emphasis
            sources = []
            for doc, score in relevant_docs:
                metadata = doc.metadata or {}
                source = metadata.get('source', 'Unknown')
                
                is_psak219 = any(keyword in source.lower() for keyword in 
                               ['psak219', 'psak 219', 'employee benefit'])
                
                # Create meaningful preview from document content (extended length for 2-3 sentences)
                preview = extract_meaningful_preview(doc.page_content, max_length=200)
                
                # Format score untuk readability - hindari scientific notation
                formatted_score = float(f"{score:.6f}") if score < 0.001 else round(score, 4)
                
                source_info = {
                    'source': source,
                    'score': formatted_score,
                    'is_psak219_standard': is_psak219,
                    'preview': preview,  # Replaced 'page' with 'preview'
                    'paragraph': metadata.get('paragraph', None),
                    'relevance': 'high' if score > 0.8 else 'medium' if score > 0.6 else 'low',
                    'session_id': session_id
                }
                sources.append(source_info)
            
            # Calculate confidence with PSAK219 boost
            base_confidence = sum(score for _, score in relevant_docs) / len(relevant_docs) if relevant_docs else 0.3
            psak219_boost = 0.1 if any(s['is_psak219_standard'] for s in sources) else 0
            confidence = min(0.95, base_confidence + psak219_boost)
            
            return {
                'answer': raw_response,
                'confidence': confidence,
                'sources': sources,
                'type': 'psak219',
                'metadata': {
                    'psak219_subtype': response_type,
                    'documents_used': len(relevant_docs),
                    'psak219_standards_found': sum(1 for s in sources if s['is_psak219_standard']),
                    'actuarial_focus': response_type in ['actuarial_calculation', 'valuation_method']
                }
            }
            
        except Exception as e:
            logger.error(f"Error formatting PSAK219 response: {str(e)}")
            return self._create_psak219_error_response(response_type, str(e))
    
    def _create_psak219_error_response(self, response_type: str, error_msg: str) -> Dict[str, Any]:
        """Create PSAK219 error response.
        
        Args:
            response_type: Type of PSAK219 processing
            error_msg: Error message
            
        Returns:
            Error response dictionary
        """
        return {
            'answer': f"Maaf, terjadi kesalahan dalam memproses pertanyaan PSAK219 ({response_type}). Silakan konsultasikan dengan ahli aktuaria atau coba lagi dengan pertanyaan yang lebih spesifik.",
            'confidence': 0.0,
            'sources': [],
            'type': 'psak219',
            'error': True,
            'error_message': error_msg,
            'metadata': {
                'psak219_subtype': response_type,
                'processing_failed': True
            }
        }
    
    def is_psak219_question(self, question: str) -> bool:
        """Determine if a question is PSAK219-related.
        
        Args:
            question: Question to analyze
            
        Returns:
            True if PSAK219-related, False otherwise
        """
        try:
            psak219_keywords = [
                'psak219', 'psak 219', 'imbalan kerja', 'employee benefit',
                'actuarial', 'aktuaria', 'aktuaris', 'actuary',
                'pension', 'pensiun', 'benefit obligation', 'kewajiban imbalan',
                'dbo', 'defined benefit', 'service cost', 'interest cost',
                'actuarial gain', 'actuarial loss', 'valuation', 'valuasi',
                'discount rate', 'mortality', 'turnover', 'puc',
                'projected unit credit', 'oci', 'other comprehensive income'
            ]
            
            question_lower = question.lower()
            return any(keyword in question_lower for keyword in psak219_keywords)
            
        except Exception as e:
            logger.error(f"Error checking PSAK219 relevance: {str(e)}")
            return False
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the PSAK219 processor.
        
        Returns:
            Dictionary containing processor information
        """
        return {
            'processor_type': 'psak219_query',
            'specialization': 'PSAK219 Employee Benefits',
            'capabilities': [
                'actuarial_calculations',
                'standard_interpretation',
                'valuation_methods',
                'reporting_guidance',
                'assumption_analysis'
            ],
            'question_types': [
                'actuarial_calculation',
                'standard_interpretation',
                'valuation_method',
                'reporting_disclosure',
                'general_psak219'
            ],
            'llm_available': self.llm is not None,
            'prompt_manager_available': self.prompt_manager is not None,
            'response_parser_available': self.response_parser is not None,
            'retrieval_manager_available': self.retrieval_manager is not None,
            'psak219_manager_available': self.psak219_manager is not None
        }