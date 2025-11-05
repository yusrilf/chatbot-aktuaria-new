"""Session management module for chat service.

This module handles session memory, chat history, and conversation state management.
"""

from typing import Dict, List, Any, Optional
import logging
import random
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages chat sessions, memory, and conversation history."""
    
    def __init__(self):
        """Initialize the session manager."""
        self.session_memories: Dict[str, ConversationBufferMemory] = {}
        
    def ensure_session_memory(self, session_id: str) -> None:
        """Ensure session memory exists for the given session ID.
        
        Args:
            session_id: Session identifier
        """
        if session_id not in self.session_memories:
            logger.info(f"Creating new session memory for session: {session_id}")
            self.session_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
    
    def get_chat_history_string(self, session_id: str) -> str:
        """Get chat history as a formatted string.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted chat history string
        """
        try:
            self.ensure_session_memory(session_id)
            memory = self.session_memories[session_id]
            
            # Get messages from memory
            messages = memory.chat_memory.messages
            
            if not messages:
                return ""
            
            # Format messages as string
            history_parts = []
            for message in messages[-10:]:  # Last 10 messages
                if hasattr(message, 'type'):
                    if message.type == 'human':
                        history_parts.append(f"User: {message.content}")
                    elif message.type == 'ai':
                        history_parts.append(f"Assistant: {message.content}")
            
            return "\n".join(history_parts)
            
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return ""
    
    def clear_memory(self, session_id: Optional[str] = None) -> bool:
        """Clear conversation memory for a session or all sessions.
        
        Args:
            session_id: Session to clear, or None to clear all sessions
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if session_id:
                if session_id in self.session_memories:
                    self.session_memories[session_id].clear()
                    logger.info(f"Cleared memory for session: {session_id}")
                    return True
                else:
                    logger.warning(f"Session {session_id} not found")
                    return False
            else:
                # Clear all sessions
                for sid in self.session_memories:
                    self.session_memories[sid].clear()
                logger.info("Cleared all session memories")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False
    
    def get_conversation_history(self, session_id: Optional[str] = None, 
                               random_sample: bool = False, 
                               limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier, or None for default session
            random_sample: Whether to return random sample of conversations
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation dictionaries
        """
        try:
            if not session_id:
                session_id = 'default'
                
            self.ensure_session_memory(session_id)
            memory = self.session_memories[session_id]
            
            # Get messages from memory
            messages = memory.chat_memory.messages
            
            if not messages:
                return []
            
            # Convert messages to conversation format
            conversations = []
            current_conversation = {}
            
            for message in messages:
                if hasattr(message, 'type'):
                    if message.type == 'human':
                        if current_conversation:  # Save previous conversation
                            conversations.append(current_conversation)
                        current_conversation = {
                            'question': message.content,
                            'timestamp': getattr(message, 'timestamp', 'unknown')
                        }
                    elif message.type == 'ai' and current_conversation:
                        current_conversation['answer'] = message.content
            
            # Add the last conversation if it exists
            if current_conversation and 'answer' in current_conversation:
                conversations.append(current_conversation)
            
            # Apply sampling and limit
            if random_sample and len(conversations) > limit:
                conversations = random.sample(conversations, limit)
            else:
                conversations = conversations[-limit:]  # Get most recent
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    def add_to_memory(self, session_id: str, question: str, answer: str) -> None:
        """Add a question-answer pair to session memory.
        
        Args:
            session_id: Session identifier
            question: User question
            answer: Assistant answer
        """
        try:
            self.ensure_session_memory(session_id)
            memory = self.session_memories[session_id]
            
            # Add to memory
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            
            logger.debug(f"Added conversation to memory for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error adding to memory: {str(e)}")
    
    def get_session_count(self) -> int:
        """Get the number of active sessions.
        
        Returns:
            Number of active sessions
        """
        return len(self.session_memories)
    
    def get_session_ids(self) -> List[str]:
        """Get list of active session IDs.
        
        Returns:
            List of session identifiers
        """
        return list(self.session_memories.keys())
    
    def get_memory(self, session_id: str) -> Optional[ConversationBufferMemory]:
        """Get memory object for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationBufferMemory object or None if not found
        """
        self.ensure_session_memory(session_id)
        return self.session_memories.get(session_id)