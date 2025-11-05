"""Main actuarial chat service combining all components."""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import local components
from .query_processor import QueryProcessor
from .prompt_manager import PromptManager
from .calculation_flow import CalculationFlowHandler
from .response_generator import ResponseGenerator
from .memory_manager import MemoryManager

# Import external dependencies
from app.services.calculators.calculation_handler import CalculationHandler
from app.services.calculators.calculation_utils import CalculationUtils

logger = logging.getLogger(__name__)

class ActuarialChatService:
    """Main actuarial chat service with modular architecture."""
    
    def __init__(self, llm, vector_store_manager, memory_window_size: int = 10):
        """Initialize actuarial chat service.
        
        Args:
            llm: Language model instance
            vector_store_manager: Vector store manager instance
            memory_window_size: Size of conversation memory window
        """
        try:
            self.llm = llm
            self.vector_store_manager = vector_store_manager
            
            # Initialize components
            self.memory_manager = MemoryManager(memory_window_size)
            self.prompt_manager = PromptManager(llm)
            self.query_processor = QueryProcessor(llm, vector_store_manager)
            self.calculation_flow = CalculationFlowHandler(llm, vector_store_manager, self.query_processor)
            self.response_generator = ResponseGenerator(llm, self.prompt_manager)
            
            # Initialize calculation components
            self.calculation_handler = CalculationHandler(llm, vector_store_manager)
            self.calculation_utils = CalculationUtils()
            
            logger.info("ActuarialChatService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ActuarialChatService: {e}")
            raise
    
    def process_question(self, question: str, session_id: str, 
                        document_type: str = "general") -> Dict[str, Any]:
        """Process user question and generate response.
        
        Args:
            question: User question
            session_id: Session identifier
            document_type: Type of document to search
            
        Returns:
            Response dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing question for session {session_id}: {question[:100]}...")
            
            # Get chat history
            chat_history = self.memory_manager.get_chat_history(session_id, "string")
            
            # Refine query with context
            refined_query = self.query_processor.refine_query_with_readme(
                question, chat_history
            )
            
            # Determine processing approach
            processing_approach = self._determine_processing_approach(question)
            
            if processing_approach == "calculation":
                return self._process_calculation_question(
                    question, refined_query, session_id, chat_history
                )
            elif processing_approach == "external":
                return self._process_external_question(
                    question, refined_query, session_id, document_type
                )
            else:
                return self._process_general_question(
                    question, refined_query, session_id, document_type
                )
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return self._generate_error_response(str(e))
    
    def process_calculation_request(self, question: str, session_id: str, 
                                  complexity: str = None) -> Dict[str, Any]:
        """Process calculation-specific requests.
        
        Args:
            question: User question
            session_id: Session identifier
            complexity: Calculation complexity level
            
        Returns:
            Calculation response dictionary
        """
        try:
            logger.info(f"Processing calculation request for session {session_id}")
            
            # Get chat history
            chat_history = self.memory_manager.get_chat_history(session_id, "string")
            
            # Execute calculation flow
            calculation_result = self.calculation_flow.handle_calculation_flow(
                question, session_id, chat_history, complexity
            )
            
            # Generate response
            response = self.response_generator.generate_response(
                question, "", calculation_result, "calculation"
            )
            
            # Add to memory
            self.memory_manager.add_message_pair(
                session_id, question, response.get("response", "")
            )
            
            return {
                "status": "success",
                "response": response,
                "calculation_result": calculation_result,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing calculation request: {e}")
            return self._generate_error_response(str(e))
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information and statistics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information dictionary
        """
        try:
            return self.memory_manager.get_session_info(session_id)
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {"error": str(e)}
    
    def clear_session(self, session_id: str) -> Dict[str, Any]:
        """Clear session memory.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Operation result dictionary
        """
        try:
            self.memory_manager.clear_session_memory(session_id)
            return {
                "status": "success",
                "message": f"Session {session_id} cleared successfully"
            }
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics and health information.
        
        Returns:
            Service statistics dictionary
        """
        try:
            memory_stats = self.memory_manager.get_memory_stats()
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "memory_stats": memory_stats,
                "components": {
                    "memory_manager": "active",
                    "prompt_manager": "active",
                    "query_processor": "active",
                    "calculation_flow": "active",
                    "response_generator": "active"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _determine_processing_approach(self, question: str) -> str:
        """Determine the best processing approach for the question.
        
        Args:
            question: User question
            
        Returns:
            Processing approach (calculation, external, general)
        """
        try:
            question_lower = question.lower()
            
            # Check for calculation keywords
            calc_keywords = [
                'hitung', 'calculate', 'kalkulasi', 'perhitungan',
                'present value', 'pv', 'nilai sekarang',
                'liability', 'kewajiban', 'reserve', 'cadangan'
            ]
            
            if any(keyword in question_lower for keyword in calc_keywords):
                return "calculation"
            
            # Check for external document keywords
            external_keywords = [
                'regulasi', 'regulation', 'peraturan', 'standar',
                'psak', 'ifrs', 'sak', 'ojk'
            ]
            
            if any(keyword in question_lower for keyword in external_keywords):
                return "external"
            
            return "general"
            
        except Exception as e:
            logger.error(f"Error determining processing approach: {e}")
            return "general"
    
    def _process_calculation_question(self, question: str, refined_query: str, 
                                    session_id: str, chat_history: str) -> Dict[str, Any]:
        """Process calculation-related questions.
        
        Args:
            question: Original question
            refined_query: Refined query
            session_id: Session identifier
            chat_history: Chat history context
            
        Returns:
            Calculation response dictionary
        """
        try:
            # Execute calculation flow
            calculation_result = self.calculation_flow.handle_calculation_flow(
                refined_query, session_id, chat_history
            )
            
            # Retrieve relevant documents
            docs = self.query_processor.retrieve_for_document(
                "actuarial", refined_query, session_id, k=5
            )
            
            # Generate context from documents
            context = self._build_context_from_docs([doc for doc, _ in docs])
            
            # Generate response
            response = self.response_generator.generate_response(
                question, context, calculation_result, "calculation"
            )
            
            # Format final response
            final_response = self.response_generator.format_response_with_sources(
                response.get("response", ""),
                self.calculation_utils.extract_source_info([doc for doc, _ in docs], session_id),
                calculation_result.get("confidence", 0.0)
            )
            
            # Add to memory
            self.memory_manager.add_message_pair(
                session_id, question, final_response["response"]
            )
            
            return {
                "status": "success",
                "type": "calculation",
                "response": final_response,
                "calculation_result": calculation_result,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error processing calculation question: {e}")
            return self._generate_error_response(str(e))
    
    def _process_external_question(self, question: str, refined_query: str, 
                                 session_id: str, document_type: str) -> Dict[str, Any]:
        """Process external document questions.
        
        Args:
            question: Original question
            refined_query: Refined query
            session_id: Session identifier
            document_type: Document type to search
            
        Returns:
            External response dictionary
        """
        try:
            # Retrieve relevant documents
            docs = self.query_processor.retrieve_for_document(
                "external", refined_query, session_id, k=4
            )
            
            # Generate context
            context = self._build_context_from_docs([doc for doc, _ in docs])
            
            # Generate response
            response = self.response_generator.generate_response(
                question, context, None, "external"
            )
            
            # Format final response
            final_response = self.response_generator.format_response_with_sources(
                response.get("response", ""),
                self.calculation_utils.extract_source_info([doc for doc, _ in docs], session_id),
                self.calculation_utils.calculate_confidence(docs)
            )
            
            # Add to memory
            self.memory_manager.add_message_pair(
                session_id, question, final_response["response"]
            )
            
            return {
                "status": "success",
                "type": "external",
                "response": final_response,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error processing external question: {e}")
            return self._generate_error_response(str(e))
    
    def _process_general_question(self, question: str, refined_query: str, 
                                session_id: str, document_type: str) -> Dict[str, Any]:
        """Process general questions.
        
        Args:
            question: Original question
            refined_query: Refined query
            session_id: Session identifier
            document_type: Document type to search
            
        Returns:
            General response dictionary
        """
        try:
            # Retrieve relevant documents
            docs = self.query_processor.retrieve_for_document(
                document_type, refined_query, session_id, k=4
            )
            
            # Generate context
            context = self._build_context_from_docs([doc for doc, _ in docs])
            
            # Generate response
            response = self.response_generator.generate_response(
                question, context, None, "general"
            )
            
            # Format final response
            final_response = self.response_generator.format_response_with_sources(
                response.get("response", ""),
                self.calculation_utils.extract_source_info([doc for doc, _ in docs], session_id),
                self.calculation_utils.calculate_confidence(docs)
            )
            
            # Add to memory
            self.memory_manager.add_message_pair(
                session_id, question, final_response["response"]
            )
            
            return {
                "status": "success",
                "type": "general",
                "response": final_response,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error processing general question: {e}")
            return self._generate_error_response(str(e))
    
    def _build_context_from_docs(self, documents: List[Any]) -> str:
        """Build context string from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Context string
        """
        try:
            if not documents:
                return ""
            
            context_parts = []
            for i, doc in enumerate(documents[:5], 1):  # Limit to 5 docs
                content = doc.page_content[:500]  # Limit content length
                context_parts.append(f"Document {i}: {content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building context from docs: {e}")
            return ""
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate standardized error response.
        
        Args:
            error_message: Error message
            
        Returns:
            Error response dictionary
        """
        return {
            "status": "error",
            "type": "error",
            "response": {
                "response": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi.",
                "sources": [],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.0,
                    "source_count": 0,
                    "error": error_message
                }
            },
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }