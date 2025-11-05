"""PSAK219 Document Parser Module.

This module provides modular components for parsing PSAK219 actuarial documents.
It includes data models, session management, extraction logic, and the main parser.
"""

from .data_models import (
    CompanyInfo,
    EmployeeData,
    ActuarialAssumptions,
    FinancialResults
)

from .session_manager import SessionManager
from .extraction_engine import ExtractionEngine
from .psak219_parser import PSAK219DocumentParser

# Global session manager instance
_global_session_manager = SessionManager()

def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    return _global_session_manager

def get_session_variable(key: str):
    """Get a session variable from the global session manager."""
    return _global_session_manager.get_session_variable(key)

def search_value(query: str, threshold: int = 70):
    """Search for values across all sessions."""
    return _global_session_manager.search_value(query, threshold)

def parse_psak219_document(file_path):
    """Parse a PSAK219 document using the global parser."""
    parser = PSAK219DocumentParser(_global_session_manager)
    return parser.parse_document(file_path)

__all__ = [
    'CompanyInfo',
    'EmployeeData', 
    'ActuarialAssumptions',
    'FinancialResults',
    'SessionManager',
    'ExtractionEngine',
    'PSAK219DocumentParser',
    'get_session_manager',
    'get_session_variable',
    'search_value',
    'parse_psak219_document'
]