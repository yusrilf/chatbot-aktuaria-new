"""Session Manager for PSAK219 Document Parser.

This module manages session data for parsed documents, providing
storage, retrieval, and search capabilities across multiple sessions.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Import fuzzywuzzy with fallback
try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    logger.warning("fuzzywuzzy not available, using basic string matching")



def _basic_similarity(text1: str, text2: str) -> int:
    """Basic string similarity fallback when fuzzywuzzy is not available.
    
    Args:
        text1: First string to compare
        text2: Second string to compare
        
    Returns:
        Similarity score (0-100)
    """
    if not text1 or not text2:
        return 0
    
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    
    # Exact match
    if text1_lower == text2_lower:
        return 100
    
    # Substring match
    if text1_lower in text2_lower or text2_lower in text1_lower:
        return 80
    
    # Word overlap
    words1 = set(text1_lower.split())
    words2 = set(text2_lower.split())
    overlap = len(words1.intersection(words2))
    total_words = len(words1.union(words2))
    
    if total_words > 0:
        return int((overlap / total_words) * 60)
    
    return 0



def _get_similarity_score(text1: str, text2: str) -> int:
    """Get similarity score using fuzzywuzzy or fallback method.
    
    Args:
        text1: First string to compare
        text2: Second string to compare
        
    Returns:
        Similarity score (0-100)
    """
    if FUZZYWUZZY_AVAILABLE:
        return fuzz.partial_ratio(text1, text2)
    else:
        return _basic_similarity(text1, text2)


class SessionManager:
    """Manages session data for parsed PSAK219 documents.
    
    This class provides centralized storage and retrieval of parsed document data,
    enabling cross-document search and session management capabilities.
    """
    
    def __init__(self):
        """Initialize the session manager."""
        self._session_data: Dict[str, Dict[str, Any]] = {}
        self._parsing_history: List[Dict[str, Any]] = []
        self._last_parsed_document: Optional[str] = None  # Track last parsed document session
        logger.info("SessionManager initialized")
    
    def store_parsed_data(self, document_path: str, parsed_data: Dict[str, Any]) -> None:
        """Store parsed data from a document.
        
        Args:
            document_path: Path to the parsed document
            parsed_data: Dictionary containing parsed document data
        """
        try:
            session_key = parsed_data.get('session_key', document_path)
            
            self._session_data[session_key] = {
                'document_path': document_path,
                'parsed_data': parsed_data,
                'timestamp': datetime.now().isoformat(),
                'session_key': session_key
            }
            
            # Add to parsing history
            self._parsing_history.append({
                'session_key': session_key,
                'document_path': document_path,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            })
            
            # Update last parsed document
            self._last_parsed_document = session_key
            
            logger.info(f"Stored parsed data for session: {session_key}")
            
        except Exception as e:
            logger.error(f"Error storing parsed data: {str(e)}")
            # Add failed parsing to history
            self._parsing_history.append({
                'session_key': session_key if 'session_key' in locals() else 'unknown',
                'document_path': document_path,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    def get_session_variable(self, key: str) -> Any:
        """Get a session variable by key.
        
        Args:
            key: The key to search for across all sessions
            
        Returns:
            The value associated with the key, or None if not found
        """
        try:
            # Search across all sessions for the key
            for session_key, session_data in self._session_data.items():
                parsed_data = session_data.get('parsed_data', {})
                
                # Check if key exists directly in parsed data
                if key in parsed_data:
                    logger.debug(f"Found key '{key}' in session {session_key}")
                    return parsed_data[key]
                
                # Check nested structures
                for section_name, section_data in parsed_data.items():
                    if isinstance(section_data, dict) and key in section_data:
                        logger.debug(f"Found key '{key}' in section '{section_name}' of session {session_key}")
                        return section_data[key]
            
            logger.warning(f"Key '{key}' not found in any session")
            return None
            
        except Exception as e:
            logger.error(f"Error getting session variable '{key}': {str(e)}")
            return None
    
    def search_value(self, query: str, threshold: int = 70) -> List[Dict[str, Any]]:
        """Search for values across all sessions using fuzzy matching.
        
        Args:
            query: Search query string
            threshold: Minimum similarity threshold (0-100)
            
        Returns:
            List of matching results with metadata
        """
        try:
            results = []
            
            for session_key, session_data in self._session_data.items():
                parsed_data = session_data.get('parsed_data', {})
                
                # Search in all string values
                matches = self._search_in_data(parsed_data, query, threshold, session_key)
                results.extend(matches)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            logger.info(f"Found {len(results)} matches for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for value '{query}': {str(e)}")
            return []
    
    def _search_in_data(self, data: Any, query: str, threshold: int, 
                       session_key: str, path: str = "") -> List[Dict[str, Any]]:
        """Recursively search for query in data structure.
        
        Args:
            data: Data to search in
            query: Search query
            threshold: Similarity threshold
            session_key: Current session key
            path: Current path in data structure
            
        Returns:
            List of matching results
        """
        results = []
        
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check key name similarity
                    key_similarity = _get_similarity_score(query.lower(), key.lower())
                    if key_similarity >= threshold:
                        results.append({
                            'session_key': session_key,
                            'path': current_path,
                            'key': key,
                            'value': value,
                            'similarity': key_similarity,
                            'match_type': 'key'
                        })
                    
                    # Recursively search in value
                    results.extend(self._search_in_data(
                        value, query, threshold, session_key, current_path
                    ))
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    current_path = f"{path}[{i}]" if path else f"[{i}]"
                    results.extend(self._search_in_data(
                        item, query, threshold, session_key, current_path
                    ))
            
            elif isinstance(data, str):
                # Check string value similarity
                value_similarity = _get_similarity_score(query.lower(), data.lower())
                if value_similarity >= threshold:
                    results.append({
                        'session_key': session_key,
                        'path': path,
                        'value': data,
                        'similarity': value_similarity,
                        'match_type': 'value'
                    })
            
            elif isinstance(data, (int, float)):
                # Check numeric values (convert to string for comparison)
                str_data = str(data)
                value_similarity = _get_similarity_score(query.lower(), str_data.lower())
                if value_similarity >= threshold:
                    results.append({
                        'session_key': session_key,
                        'path': path,
                        'value': data,
                        'similarity': value_similarity,
                        'match_type': 'numeric'
                    })
        
        except Exception as e:
            logger.warning(f"Error searching in data at path '{path}': {str(e)}")
        
        return results
    
    def get_all_session_keys(self) -> List[str]:
        """Get all session keys."""
        return list(self._session_data.keys())
    
    def get_parsing_history(self) -> List[Dict[str, Any]]:
        """Get parsing history."""
        return self._parsing_history.copy()
    
    def clear_session(self) -> None:
        """Clear all session data."""
        try:
            self._session_data.clear()
            self._parsing_history.clear()
            logger.info("All session data cleared")
        except Exception as e:
            logger.error(f"Error clearing session data: {str(e)}")
            raise
    
    def set_session_variable(self, key: str, value: Any) -> None:
        """Set a session variable in the most recent session.
        
        Args:
            key: Variable key
            value: Variable value
        """
        try:
            if not self._session_data:
                # Create a default session if none exists
                default_session = {
                    'document_path': 'default',
                    'parsed_data': {},
                    'timestamp': datetime.now().isoformat(),
                    'session_key': 'default'
                }
                self._session_data['default'] = default_session
            
            # Get the most recent session
            latest_session_key = max(self._session_data.keys(), 
                                   key=lambda k: self._session_data[k]['timestamp'])
            
            self._session_data[latest_session_key]['parsed_data'][key] = value
            logger.debug(f"Set session variable '{key}' in session {latest_session_key}")
            
        except Exception as e:
            logger.error(f"Error setting session variable '{key}': {str(e)}")
            raise
    
    def get_session_data_by_key(self, key: str) -> Any:
        """Get session data by session key.
        
        Args:
            key: Session key
            
        Returns:
            Session data or None if not found
        """
        try:
            return self._session_data.get(key)
        except Exception as e:
            logger.error(f"Error getting session data for key '{key}': {str(e)}")
            return None
    
    def get_psak219_data_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get PSAK219 data associated with a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of PSAK219 data entries for the session
        """
        try:
            results = []
            
            # Search for sessions that match the session_id
            for session_key, session_data in self._session_data.items():
                if session_id in session_key or session_key == session_id:
                    parsed_data = session_data.get('parsed_data', {})
                    
                    # Extract relevant PSAK219 data
                    psak219_entry = {
                        'session_key': session_key,
                        'document_path': session_data.get('document_path', ''),
                        'timestamp': session_data.get('timestamp', ''),
                        'company_info': parsed_data.get('company_info', {}),
                        'employee_data': parsed_data.get('employee_data', {}),
                        'actuarial_assumptions': parsed_data.get('actuarial_assumptions', {}),
                        'financial_results': parsed_data.get('financial_results', {}),
                        'sensitivity_analysis': parsed_data.get('sensitivity_analysis', {})
                    }
                    
                    results.append(psak219_entry)
            
            logger.info(f"Found {len(results)} PSAK219 entries for session: {session_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error getting PSAK219 data for session {session_id}: {str(e)}")
            return []
    
    def search_psak219_by_session(self, session_id: str, query: str, 
                                 threshold: int = 70) -> List[Dict[str, Any]]:
        """Search PSAK219 data within a specific session.
        
        Args:
            session_id: Session identifier
            query: Search query
            threshold: Similarity threshold
            
        Returns:
            List of matching PSAK219 data entries
        """
        try:
            results = []
            
            # Get PSAK219 data for the session
            session_data_list = self.get_psak219_data_by_session(session_id)
            
            for session_entry in session_data_list:
                # Search within this session's data
                matches = self._search_in_data(
                    session_entry, query, threshold, session_entry['session_key']
                )
                
                # Add session context to matches
                for match in matches:
                    match['session_id'] = session_id
                    match['document_path'] = session_entry.get('document_path', '')
                
                results.extend(matches)
            
            # Sort by similarity
            results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            logger.info(f"Found {len(results)} matches for query '{query}' in session {session_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching PSAK219 data in session {session_id}: {str(e)}")
            return []
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about current sessions.
        
        Returns:
            Dictionary with session statistics
        """
        try:
            total_sessions = len(self._session_data)
            successful_parses = len([h for h in self._parsing_history if h.get('status') == 'success'])
            failed_parses = len([h for h in self._parsing_history if h.get('status') == 'failed'])
            
            return {
                'total_sessions': total_sessions,
                'successful_parses': successful_parses,
                'failed_parses': failed_parses,
                'success_rate': successful_parses / (successful_parses + failed_parses) if (successful_parses + failed_parses) > 0 else 0,
                'session_keys': list(self._session_data.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting session stats: {str(e)}")
            return {'error': str(e)}