"""Memory management and session handling for actuarial chat service."""

import logging
import time
from typing import Dict, Any, List, Optional
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages conversation memory and session state."""
    
    def __init__(self, memory_window_size: int = 10):
        """Initialize memory manager.
        
        Args:
            memory_window_size: Number of conversation turns to keep in memory
        """
        self.memory_window_size = memory_window_size
        self.session_memories: Dict[str, ConversationBufferWindowMemory] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
    
    def get_or_create_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get existing memory or create new one for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationBufferWindowMemory instance
        """
        try:
            if session_id not in self.session_memories:
                self.session_memories[session_id] = ConversationBufferWindowMemory(
                    k=self.memory_window_size,
                    return_messages=True
                )
                self.session_metadata[session_id] = {
                    "created_at": self._get_current_timestamp(),
                    "message_count": 0,
                    "last_activity": self._get_current_timestamp()
                }
                logger.info(f"Created new memory for session {session_id}")
            
            # Update last activity
            self.session_metadata[session_id]["last_activity"] = self._get_current_timestamp()
            
            return self.session_memories[session_id]
            
        except Exception as e:
            logger.error(f"Error getting/creating memory for session {session_id}: {e}")
            # Return a new memory as fallback
            return ConversationBufferWindowMemory(
                k=self.memory_window_size,
                return_messages=True
            )
    
    def add_message_pair(self, session_id: str, human_message: str, ai_message: str):
        """Add human-AI message pair to session memory.
        
        Args:
            session_id: Session identifier
            human_message: Human message text
            ai_message: AI response text
        """
        try:
            memory = self.get_or_create_memory(session_id)
            
            # Add messages to memory
            memory.chat_memory.add_user_message(human_message)
            memory.chat_memory.add_ai_message(ai_message)
            
            # Update metadata
            if session_id in self.session_metadata:
                self.session_metadata[session_id]["message_count"] += 2
                self.session_metadata[session_id]["last_activity"] = self._get_current_timestamp()
            
            logger.debug(f"Added message pair to session {session_id}")
            
        except Exception as e:
            logger.error(f"Error adding message pair to session {session_id}: {e}")
    
    def get_chat_history(self, session_id: str, format_type: str = "string") -> Any:
        """Get chat history for session.
        
        Args:
            session_id: Session identifier
            format_type: Format type ('string', 'messages', 'dict')
            
        Returns:
            Chat history in requested format
        """
        try:
            memory = self.get_or_create_memory(session_id)
            
            if format_type == "string":
                return self._format_history_as_string(memory)
            elif format_type == "messages":
                return memory.chat_memory.messages
            elif format_type == "dict":
                return self._format_history_as_dict(memory)
            else:
                logger.warning(f"Unknown format type: {format_type}, defaulting to string")
                return self._format_history_as_string(memory)
                
        except Exception as e:
            logger.error(f"Error getting chat history for session {session_id}: {e}")
            return "" if format_type == "string" else []
    
    def clear_session_memory(self, session_id: str):
        """Clear memory for specific session.
        
        Args:
            session_id: Session identifier
        """
        try:
            if session_id in self.session_memories:
                self.session_memories[session_id].clear()
                logger.info(f"Cleared memory for session {session_id}")
            
            if session_id in self.session_metadata:
                self.session_metadata[session_id]["message_count"] = 0
                self.session_metadata[session_id]["last_activity"] = self._get_current_timestamp()
                
        except Exception as e:
            logger.error(f"Error clearing memory for session {session_id}: {e}")
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information and metadata.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information dictionary
        """
        try:
            info = {
                "session_id": session_id,
                "exists": session_id in self.session_memories,
                "metadata": self.session_metadata.get(session_id, {})
            }
            
            if session_id in self.session_memories:
                memory = self.session_memories[session_id]
                info["message_count"] = len(memory.chat_memory.messages)
                info["memory_window_size"] = self.memory_window_size
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting session info for {session_id}: {e}")
            return {
                "session_id": session_id,
                "exists": False,
                "error": str(e)
            }
    
    def cleanup_old_sessions(self, max_inactive_hours: int = 24):
        """Clean up old inactive sessions.
        
        Args:
            max_inactive_hours: Maximum hours of inactivity before cleanup
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (max_inactive_hours * 3600)
            
            sessions_to_remove = []
            
            for session_id, metadata in self.session_metadata.items():
                last_activity = metadata.get("last_activity", 0)
                if last_activity < cutoff_time:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                if session_id in self.session_memories:
                    del self.session_memories[session_id]
                if session_id in self.session_metadata:
                    del self.session_metadata[session_id]
                logger.info(f"Cleaned up inactive session {session_id}")
            
            if sessions_to_remove:
                logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")
                
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs.
        
        Returns:
            List of active session IDs
        """
        try:
            return list(self.session_memories.keys())
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Memory statistics dictionary
        """
        try:
            stats = {
                "total_sessions": len(self.session_memories),
                "memory_window_size": self.memory_window_size,
                "sessions": {}
            }
            
            for session_id, memory in self.session_memories.items():
                stats["sessions"][session_id] = {
                    "message_count": len(memory.chat_memory.messages),
                    "metadata": self.session_metadata.get(session_id, {})
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}
    
    def _format_history_as_string(self, memory: ConversationBufferWindowMemory) -> str:
        """Format chat history as string.
        
        Args:
            memory: ConversationBufferWindowMemory instance
            
        Returns:
            Formatted chat history string
        """
        try:
            messages = memory.chat_memory.messages
            if not messages:
                return ""
            
            history_parts = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    history_parts.append(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    history_parts.append(f"Assistant: {message.content}")
            
            return "\n".join(history_parts)
            
        except Exception as e:
            logger.error(f"Error formatting history as string: {e}")
            return ""
    
    def _format_history_as_dict(self, memory: ConversationBufferWindowMemory) -> List[Dict[str, str]]:
        """Format chat history as list of dictionaries.
        
        Args:
            memory: ConversationBufferWindowMemory instance
            
        Returns:
            List of message dictionaries
        """
        try:
            messages = memory.chat_memory.messages
            formatted_messages = []
            
            for message in messages:
                if isinstance(message, HumanMessage):
                    formatted_messages.append({
                        "role": "human",
                        "content": message.content
                    })
                elif isinstance(message, AIMessage):
                    formatted_messages.append({
                        "role": "assistant",
                        "content": message.content
                    })
            
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Error formatting history as dict: {e}")
            return []
    
    def _get_current_timestamp(self) -> float:
        """Get current timestamp.
        
        Returns:
            Current timestamp as float
        """
        return time.time()