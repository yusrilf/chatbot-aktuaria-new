"""Main Query Handler for Actuarial Chatbot.

This module provides the main QueryHandler class that coordinates
different types of query processing operations.
"""

import logging
from typing import Dict, Any, Optional

from .external_query_processor import ExternalQueryProcessor
from .datastory_query_processor import DatastoryQueryProcessor
from .custom_query_processor import CustomQueryProcessor
from .retrieval_manager import RetrievalManager
from .psak219_query_processor import PSAK219QueryProcessor

logger = logging.getLogger(__name__)

class QueryHandler:
    """Main handler for different types of actuarial queries."""
    
    def __init__(self, 
                 llm=None, 
                 vector_store_manager=None, 
                 document_session_service=None, 
                 prompt_manager=None, 
                 response_parser=None):
        """Initialize the query handler with required components.
        
        Args:
            llm: Language model instance
            vector_store_manager: Vector store manager for document retrieval
            document_session_service: Service for managing document sessions
            prompt_manager: Manager for handling prompts
            response_parser: Parser for processing responses
        """
        self.llm = llm
        self.vector_store_manager = vector_store_manager
        self.document_session_service = document_session_service
        self.prompt_manager = prompt_manager
        self.response_parser = response_parser
        
        # Initialize specialized processors
        self.retrieval_manager = RetrievalManager(
            vector_store_manager=vector_store_manager,
            document_session_service=document_session_service
        )
        
        self.external_processor = ExternalQueryProcessor(
            llm=llm,
            prompt_manager=prompt_manager,
            response_parser=response_parser
        )
        
        self.datastory_processor = DatastoryQueryProcessor(
            llm=llm,
            retrieval_manager=self.retrieval_manager,
            prompt_manager=prompt_manager,
            response_parser=response_parser
        )
        
        self.custom_processor = CustomQueryProcessor(
            llm=llm,
            retrieval_manager=self.retrieval_manager,
            prompt_manager=prompt_manager,
            response_parser=response_parser
        )
        
        self.psak219_processor = PSAK219QueryProcessor(
            llm=llm,
            prompt_manager=prompt_manager,
            response_parser=response_parser,
            retrieval_manager=self.retrieval_manager
        )
        
        # Session memory management
        self.session_memories = {}
        
        logger.info("QueryHandler initialized with all specialized processors")
    
    def handle_external_question(self, question: str, session_id: str) -> Dict[str, Any]:
        """Handle external/general actuarial questions.
        
        Args:
            question: The question to process
            session_id: Session identifier
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            logger.info(f"Processing external question for session {session_id}")
            
            # Check if this is a PSAK219 question that can be answered directly
            if self.document_session_service and self.psak219_processor.is_psak219_question(question):
                try:
                    psak219_result = self.psak219_processor.handle_psak219_question(question, session_id)
                    if psak219_result and psak219_result.get('success', False):
                        logger.info(f"Returning direct PSAK219 answer for session {session_id}")
                        return psak219_result
                except Exception as e:
                    logger.warning(f"PSAK219 processing failed, falling back to external: {str(e)}")
            
            # Ensure session memory
            self._ensure_session_memory(session_id)
            
            # Get chat history for external processing
            memory = self.session_memories.get(session_id)
            chat_history = ""
            if memory and hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
                # Format messages as string
                history_parts = []
                for msg in memory.chat_memory.messages:
                    if hasattr(msg, 'type'):
                        if msg.type == 'human':
                            history_parts.append(f"Human: {msg.content}")
                        elif msg.type == 'ai':
                            history_parts.append(f"Assistant: {msg.content}")
                chat_history = "\n".join(history_parts)
            
            # Process with external query processor
            return self.external_processor.handle_external_question(question, session_id, chat_history)
            
        except Exception as e:
            logger.error(f"Error handling external question: {str(e)}")
            return {
                'answer': f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}",
                'sources': [],
                'session_id': session_id,
                'error': str(e)
            }
    
    def handle_datastory_question(self, data: str, session_id: str) -> Dict[str, Any]:
        """Handle data story questions with provided data.
        
        Args:
            data: The data to analyze
            session_id: Session identifier
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            logger.info(f"Processing datastory question for session {session_id}")
            
            # Ensure session memory
            self._ensure_session_memory(session_id)
            
            # Process with datastory query processor
            return self.datastory_processor.process_question(data, session_id, self.session_memories)
            
        except Exception as e:
            logger.error(f"Error handling datastory question: {str(e)}")
            return {
                'answer': f"Maaf, terjadi kesalahan dalam memproses data story Anda: {str(e)}",
                'sources': [],
                'session_id': session_id,
                'error': str(e)
            }
    
    def handle_custom_question(self, question: str, session_id: str, use_cot: bool = False) -> Dict[str, Any]:
        """Handle custom questions with optional chain-of-thought reasoning.
        
        Args:
            question: The question to process
            session_id: Session identifier
            use_cot: Whether to use chain-of-thought reasoning
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            logger.info(f"Processing custom question for session {session_id} (CoT: {use_cot})")
            
            # Ensure session memory
            self._ensure_session_memory(session_id)
            
            # Process with custom query processor
            # Get chat history from memory object
            memory = self.session_memories.get(session_id)
            chat_history = ""
            if memory and hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
                # Format messages as string
                history_parts = []
                for msg in memory.chat_memory.messages:
                    if hasattr(msg, 'type'):
                        if msg.type == 'human':
                            history_parts.append(f"Human: {msg.content}")
                        elif msg.type == 'ai':
                            history_parts.append(f"Assistant: {msg.content}")
                chat_history = "\n".join(history_parts)
            
            return self.custom_processor.handle_custom_question(
                question, session_id, chat_history
            )
            
        except Exception as e:
            logger.error(f"Error handling custom question: {str(e)}")
            return {
                'answer': f"Maaf, terjadi kesalahan dalam memproses pertanyaan custom Anda: {str(e)}",
                'sources': [],
                'session_id': session_id,
                'error': str(e)
            }
    
    def _ensure_session_memory(self, session_id: str) -> None:
        """Ensure session memory exists for the given session.
        
        Args:
            session_id: Session identifier
        """
        if session_id not in self.session_memories:
            from langchain.memory import ConversationBufferWindowMemory
            self.session_memories[session_id] = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 exchanges
                return_messages=True,
                memory_key="chat_history"
            )
            logger.info(f"Created new session memory for session {session_id}")
    
    def get_session_memory(self, session_id: str) -> Optional[Any]:
        """Get session memory for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session memory object or None if not found
        """
        return self.session_memories.get(session_id)
    
    def clear_session_memory(self, session_id: str) -> bool:
        """Clear session memory for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if memory was cleared, False if session not found
        """
        if session_id in self.session_memories:
            del self.session_memories[session_id]
            logger.info(f"Cleared session memory for session {session_id}")
            return True
        return False
    
    def get_active_sessions(self) -> list:
        """Get list of active session IDs.
        
        Returns:
            List of active session IDs
        """
        return list(self.session_memories.keys())
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions.
        
        Returns:
            Dictionary containing session statistics
        """
        return {
            'total_sessions': len(self.session_memories),
            'active_sessions': list(self.session_memories.keys()),
            'memory_usage': {
                session_id: len(memory.chat_memory.messages) 
                for session_id, memory in self.session_memories.items()
            }
        }