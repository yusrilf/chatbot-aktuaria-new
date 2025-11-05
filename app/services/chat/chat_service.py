"""Refactored ActuarialChatService using modular components.

This module provides the main chat service functionality using smaller,
modular components for better maintainability and testing.
"""

from typing import List, Dict, Any, Optional
import logging
import traceback
import time
# Safe optional import for ChatOpenAI to avoid ModuleNotFoundError during tests
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  # type: ignore

from app.config import config
from app.models.embeddings import VectorStoreManager
from app.services.prompts.prompt_manager import PromptManager
from app.services.parsers.response_parser import ResponseParser
from app.services.calculators.calculation_handler import CalculationHandler
from app.services.handlers.query_handler import QueryHandler
from app.services.documents.document_session_service import DocumentSessionService
from app.utils.singleton_manager import get_or_create_service
from app.utils.performance_monitor import PerformanceMonitor, get_global_monitor

from .intent_classifier import IntentClassifier
from .session_manager import SessionManager
from .psak219_handler import PSAK219Handler

logger = logging.getLogger(__name__)

class ActuarialChatService:
    """Main actuarial chatbot service using modular components."""
    
    def __init__(self):
        """Initialize the actuarial chat service with all components."""
        logger.info("Initializing ActuarialChatService")
        
        # Initialize performance monitor
        self.performance_monitor = get_global_monitor()
        
        # Core LLM and vector store (safe initialization)
        self.llm = None
        try:
            if ChatOpenAI is not None and getattr(config, 'OPENAI_API_KEY', None):
                self.llm = ChatOpenAI(
                    model=getattr(config, 'OPENAI_MODEL', 'gpt-4o-mini'),
                    temperature=0.1,
                    api_key=config.OPENAI_API_KEY
                )
            else:
                logger.warning("LLM not initialized: ChatOpenAI missing or API key not set.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            self.llm = None
        
        # Use singleton VectorStoreManager
        self.vector_store_manager = get_or_create_service(
            VectorStoreManager, 
            'vector_store_manager'
        )
        
        # Initialize modular components (allow llm=None safely)
        self.prompt_manager = PromptManager()
        self.response_parser = ResponseParser(llm=self.llm)
        self.document_session_service = DocumentSessionService()
        
        # Initialize new modular components
        self.session_manager = SessionManager()
        self.intent_classifier = IntentClassifier(self.llm, self.response_parser)
        self.psak219_handler = PSAK219Handler(self.document_session_service)
        
        # Initialize existing handlers
        self.calculation_handler = CalculationHandler(
            llm=self.llm,
            vector_store_manager=self.vector_store_manager,
            response_parser=self.response_parser,
            prompt_manager=self.prompt_manager
        )
        self.query_handler = QueryHandler(
            llm=self.llm,
            vector_store_manager=self.vector_store_manager,
            document_session_service=self.document_session_service,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser
        )
        
        logger.info("ActuarialChatService initialized with modular components")

    def ask_project(self, question: str, session_id: str) -> Dict[str, Any]:
        """Main entry point with intent classification and routing.
        
        Args:
            question: User question
            session_id: Session identifier
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Start overall timing
        overall_timer = self.performance_monitor.start_timer("ask_project_total")
        
        try:
            logger.info(f"Processing project question for session {session_id}")
            
            # Time: Session setup
            session_timer = self.performance_monitor.start_timer("session_setup")
            self.session_manager.ensure_session_memory(session_id)
            chat_history = self.session_manager.get_chat_history_string(session_id)
            self.performance_monitor.stop_timer(session_timer)
            
            # Time: Intent classification
            intent_timer = self.performance_monitor.start_timer("intent_classification")
            classification = self.intent_classifier.classify_intent(
                question, session_id, chat_history
            )
            intent = classification["intent"]
            complexity = classification["complexity"]
            self.performance_monitor.stop_timer(intent_timer)
            
            logger.info(f"Classified intent: {intent}, complexity: {complexity}")
            
            # Time: Main processing
            processing_timer = self.performance_monitor.start_timer(f"processing_{intent}")
            if intent == "calculation":
                result = self._handle_calculation_flow(question, session_id, classification)
            else:
                # All non-calculation questions go to theory flow
                result = self._handle_theory_flow(question, session_id, classification)
            self.performance_monitor.stop_timer(processing_timer)
            
            # Time: Memory update
            memory_timer = self.performance_monitor.start_timer("memory_update")
            self.session_manager.add_to_memory(session_id, question, result.get('answer', ''))
            self.performance_monitor.stop_timer(memory_timer)
            
            # Stop overall timing and log performance summary
            total_time = self.performance_monitor.stop_timer(overall_timer)
            
            # Add timing information to result only if enabled in config
            if config.PERFORMANCE_METRICS_IN_RESPONSE:
                result['performance_metrics'] = {
                    'total_time': total_time,
                    'intent': intent,
                    'complexity': complexity
                }
            
            # Log detailed performance breakdown (always log for debugging)
            if config.PERFORMANCE_LOGGING_ENABLED:
                self._log_performance_breakdown(session_id, total_time)
            
            return result
                
        except Exception as e:
            self.performance_monitor.stop_timer(overall_timer)
            logger.error(f"Error in ask_project: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'answer': 'Maaf, terjadi kesalahan saat memproses pertanyaan Anda.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e),
                'mode': 'error'
            }

    def _handle_theory_flow(self, question: str, session_id: str, 
                           classification: Dict[str, Any]) -> Dict[str, Any]:
        """Handle theory questions with optimized approach based on complexity.
        
        Args:
            question: User question
            session_id: Session identifier
            classification: Intent classification result
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing theory flow for session {session_id}, complexity: {classification.get('complexity')}")
            
            complexity = classification.get('complexity', 'moderate')
            
            # Time: PSAK219 check
            psak_timer = self.performance_monitor.start_timer("psak219_check")
            
            # For simple questions, prioritize direct retrieval without heavy COT
            if complexity == 'simple':
                # First try direct PSAK219 data access
                if self.intent_classifier.is_psak219_data_question(question, session_id):
                    psak219_result = self.psak219_handler.get_direct_psak219_answer(question, session_id)
                    if psak219_result:
                        self.performance_monitor.stop_timer(psak_timer)
                        psak219_result['classification'] = classification
                        psak219_result['mode'] = 'theory_direct'
                        return psak219_result
            
            self.performance_monitor.stop_timer(psak_timer)
            
            # Time: Query processing
            query_timer = self.performance_monitor.start_timer("query_processing")
            
            # Route to appropriate processor based on complexity and question type
            if complexity in ['simple', 'moderate']:
                # Use custom processor for document-based questions
                result = self.query_handler.handle_custom_question(question, session_id)
            else:
                # Use datastory processor for complex analysis
                result = self.query_handler.handle_datastory_question(question, session_id)
            
            self.performance_monitor.stop_timer(query_timer)
            
            # Add classification info to result
            result['classification'] = classification
            result['mode'] = 'theory_processed'
            
            return result
            
        except Exception as e:
            logger.error(f"Error in theory flow: {str(e)}")
            return {
                'answer': 'Maaf, terjadi kesalahan saat memproses pertanyaan teori Anda.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e),
                'mode': 'theory_error'
            }

    def _handle_calculation_flow(self, question: str, session_id: str, 
                                classification: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculation questions using COT planning and step-by-step RAG.
        
        Args:
            question: User question
            session_id: Session identifier
            classification: Intent classification result
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            complexity = classification.get("complexity", "moderate")
            logger.info(f"Processing calculation question with complexity: {complexity}")
            
            # Ensure session memory
            self.session_manager.ensure_session_memory(session_id)
            
            # Get chat history for context
            chat_history = self.session_manager.get_chat_history_string(session_id)
            
            # Delegate to calculation handler
            result = self.calculation_handler.handle_calculation_question(
                question=question,
                session_id=session_id,
                classification=classification,
                chat_history=chat_history
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in calculation flow: {str(e)}")
            return {
                'answer': 'Maaf, terjadi kesalahan saat memproses pertanyaan perhitungan Anda.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e),
                'mode': 'calculation_error'
            }

    # Legacy methods for backward compatibility
    def ask_question(self, question: str, session_id: str) -> Dict[str, Any]:
        """Process a general actuarial question (legacy method)."""
        try:
            logger.info(f"Processing general actuarial question for session {session_id}")
            return self.query_handler.handle_external_question(question, session_id)
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return self._create_error_response(session_id, str(e), 'error')
    
    def datastory(self, data: str, session_id: str) -> Dict[str, Any]:
        """Process datastory/discussion questions (legacy method)."""
        try:
            logger.info(f"Processing datastory question for session {session_id}")
            return self.query_handler.handle_datastory_question(data, session_id)
        except Exception as e:
            logger.error(f"Error processing datastory: {str(e)}")
            return self._create_error_response(session_id, str(e), 'error')
    
    def handle_custom_question(self, question: str, session_id: str, 
                              use_cot: bool = False) -> Dict[str, Any]:
        """Handle custom questions (legacy method)."""
        try:
            return self.query_handler.handle_custom_question(question, session_id, use_cot)
        except Exception as e:
            logger.error(f"Error handling custom question: {str(e)}")
            return self._create_error_response(session_id, str(e), 'error')
    
    def handle_calculation_flow(self, question: str, session_id: str, 
                               complexity: Optional[str] = None) -> Dict[str, Any]:
        """Handle calculation flow (legacy method)."""
        try:
            self.session_manager.ensure_session_memory(session_id)
            chat_history = self.session_manager.get_chat_history_string(session_id)
            return self.calculation_handler.handle_calculation_flow(
                question, session_id, complexity, chat_history
            )
        except Exception as e:
            logger.error(f"Error in calculation flow: {str(e)}")
            return self._create_error_response(session_id, str(e), 'error_calculation')
    
    # Session and system management methods
    def clear_memory(self, session_id: Optional[str] = None) -> bool:
        """Clear conversation memory."""
        return self.session_manager.clear_memory(session_id)
    
    def get_conversation_history(self, session_id: Optional[str] = None, 
                               random_sample: bool = False, 
                               limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.session_manager.get_conversation_history(session_id, random_sample, limit)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            return {
                'vector_store': {
                    'document_count': self.vector_store_manager.get_document_count(),
                    'collection_info': self.vector_store_manager.get_collection_info()
                },
                'sessions': {
                    'active_sessions': self.session_manager.get_session_count(),
                    'session_ids': self.session_manager.get_session_ids()
                },
                'psak219': {
                    'available_documents': len(self.psak219_handler.get_available_psak219_documents())
                }
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {'error': str(e)}
    
    def get_sources_info(self, session_id: str) -> List[Dict[str, Any]]:
        """Get information about available sources."""
        try:
            return self.document_session_service.get_sources_info(session_id)
        except Exception as e:
            logger.error(f"Error getting sources info: {str(e)}")
            return []
    
    def _create_error_response(self, session_id: str, error: str, mode: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'answer': 'Maaf, terjadi kesalahan saat memproses pertanyaan Anda.',
            'sources': [],
            'confidence': 0.0,
            'session_id': session_id,
            'error': error,
            'mode': mode
        }

    def _log_performance_breakdown(self, session_id: str, total_time: float) -> None:
        """Log detailed performance breakdown for analysis.
        
        Args:
            session_id: Session identifier
            total_time: Total processing time
        """
        try:
            # Get performance summary from monitor
            summary = self.performance_monitor.get_summary()
            
            logger.info(f"=== PERFORMANCE BREAKDOWN - Session {session_id} ===")
            logger.info(f"Total Time: {total_time:.3f}s")
            
            # Log individual component timings
            for operation, timing_data in summary.items():
                if isinstance(timing_data, dict) and 'total_time' in timing_data:
                    avg_time = timing_data.get('avg_time', 0)
                    count = timing_data.get('count', 0)
                    total_op_time = timing_data.get('total_time', 0)
                    percentage = (total_op_time / total_time * 100) if total_time > 0 else 0
                    
                    logger.info(f"  {operation}: {total_op_time:.3f}s ({percentage:.1f}%) - "
                              f"avg: {avg_time:.3f}s, count: {count}")
            
            # Identify bottlenecks (operations taking >20% of total time)
            bottlenecks = []
            for operation, timing_data in summary.items():
                if isinstance(timing_data, dict) and 'total_time' in timing_data:
                    total_op_time = timing_data.get('total_time', 0)
                    percentage = (total_op_time / total_time * 100) if total_time > 0 else 0
                    if percentage > 20:
                        bottlenecks.append((operation, percentage, total_op_time))
            
            if bottlenecks:
                logger.warning("PERFORMANCE BOTTLENECKS DETECTED:")
                for operation, percentage, op_time in sorted(bottlenecks, key=lambda x: x[1], reverse=True):
                    logger.warning(f"  🔴 {operation}: {op_time:.3f}s ({percentage:.1f}%)")
            
            logger.info("=== END PERFORMANCE BREAKDOWN ===")
            
        except Exception as e:
            logger.error(f"Error logging performance breakdown: {str(e)}")