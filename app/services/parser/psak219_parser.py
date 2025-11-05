"""Main PSAK219 Document Parser.

This module provides the main parser class that orchestrates the extraction
of structured data from PSAK219 actuarial documents using modular components.
"""

import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from .session_manager import SessionManager
from .extraction_engine import ExtractionEngine
from .data_models import CompanyInfo, EmployeeData, ActuarialAssumptions, FinancialResults

logger = logging.getLogger(__name__)


class PSAK219DocumentParser:
    """Main parser class for PSAK219 actuarial documents.
    
    This class orchestrates the parsing process by using modular components
    for extraction, session management, and data storage.
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        """Initialize the PSAK219 document parser.
        
        Args:
            session_manager: Optional session manager instance
        """
        self.session_manager = session_manager or SessionManager()
        self.extraction_engine = ExtractionEngine()
        logger.info("PSAK219DocumentParser initialized")
    
    def parse_document(self, file_path: Union[str, Path], 
                     session_key: Optional[str] = None) -> Dict[str, Any]:
        """Parse a PSAK219 document and extract structured data.
        
        Args:
            file_path: Path to the document file
            session_key: Optional session key for storing parsed data
            
        Returns:
            Dictionary containing parsed data and metadata
        """
        try:
            file_path = Path(file_path)
            
            # Generate session key if not provided
            if not session_key:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_key = f"psak219_{file_path.stem}_{timestamp}"
            
            logger.info(f"Starting to parse document: {file_path}")
            
            # Read document content
            if isinstance(file_path, str) and not Path(file_path).exists():
                # Treat as content string if file doesn't exist
                content = file_path
                document_path = "content_string"
            else:
                # Read from file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                document_path = str(file_path)
            
            # Extract structured data using extraction engine
            company_info = self.extraction_engine.extract_company_info(content)
            employee_data = self.extraction_engine.extract_employee_data(content)
            actuarial_assumptions = self.extraction_engine.extract_actuarial_assumptions(content)
            financial_results = self.extraction_engine.extract_financial_results(content)
            sensitivity_analysis = self.extraction_engine.extract_sensitivity_analysis(content)
            
            # Compile parsed data
            parsed_data = {
                'session_key': session_key,
                'document_path': document_path,
                'parsing_timestamp': datetime.now().isoformat(),
                'company_info': asdict(company_info),
                'employee_data': asdict(employee_data),
                'actuarial_assumptions': asdict(actuarial_assumptions),
                'financial_results': asdict(financial_results),
                'sensitivity_analysis': sensitivity_analysis,
                'metadata': {
                    'parser_version': '2.0.0',
                    'content_length': len(content),
                    'extraction_summary': {
                        'company_info_complete': company_info.is_complete(),
                        'employee_count': employee_data.get_total_employees(),
                        'assumptions_complete': actuarial_assumptions.is_complete(),
                        'has_financial_data': any([
                            financial_results.present_value_obligation_current,
                            financial_results.current_service_cost_current
                        ]),
                        'has_sensitivity_analysis': bool(
                            sensitivity_analysis.get('discount_rate_sensitivity') or 
                            sensitivity_analysis.get('salary_increase_sensitivity')
                        )
                    }
                }
            }
            
            # Store in session manager
            self.session_manager.store_parsed_data(document_path, parsed_data)
            
            logger.info(f"Successfully parsed document: {document_path}")
            logger.info(f"Session key: {session_key}")
            
            return {
                'success': True,
                'session_key': session_key,
                'document_path': document_path,
                'parsed_data': parsed_data,
                'summary': self._generate_parsing_summary(parsed_data)
            }
            
        except FileNotFoundError:
            error_msg = f"Document file not found: {file_path}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'file_not_found'
            }
        
        except UnicodeDecodeError as e:
            error_msg = f"Error reading document (encoding issue): {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'encoding_error'
            }
        
        except Exception as e:
            error_msg = f"Error parsing document {file_path}: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'parsing_error'
            }
    
    def _generate_parsing_summary(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the parsing results.
        
        Args:
            parsed_data: The parsed document data
            
        Returns:
            Dictionary with parsing summary
        """
        try:
            metadata = parsed_data.get('metadata', {})
            extraction_summary = metadata.get('extraction_summary', {})
            
            company_info = parsed_data.get('company_info', {})
            employee_data = parsed_data.get('employee_data', {})
            financial_results = parsed_data.get('financial_results', {})
            
            return {
                'document_info': {
                    'company_name': company_info.get('nama_perusahaan', 'Unknown'),
                    'valuation_period': company_info.get('periode_valuasi', 'Unknown'),
                    'actuary_name': company_info.get('nama_aktuaris', 'Unknown')
                },
                'data_completeness': {
                    'company_info_complete': extraction_summary.get('company_info_complete', False),
                    'has_employee_data': extraction_summary.get('employee_count', 0) > 0,
                    'assumptions_complete': extraction_summary.get('assumptions_complete', False),
                    'has_financial_data': extraction_summary.get('has_financial_data', False),
                    'has_sensitivity_analysis': extraction_summary.get('has_sensitivity_analysis', False)
                },
                'key_metrics': {
                    'total_employees': employee_data.get('permanent_count', 0) + employee_data.get('contract_count', 0),
                    'present_value_obligation': financial_results.get('present_value_obligation_current', 0),
                    'current_service_cost': financial_results.get('current_service_cost_current', 0)
                },
                'parsing_metadata': {
                    'session_key': parsed_data.get('session_key'),
                    'parsing_timestamp': parsed_data.get('parsing_timestamp'),
                    'content_length': metadata.get('content_length', 0),
                    'parser_version': metadata.get('parser_version', 'unknown')
                }
            }
        
        except Exception as e:
            logger.warning(f"Error generating parsing summary: {str(e)}")
            return {
                'error': 'Could not generate summary',
                'session_key': parsed_data.get('session_key', 'unknown')
            }
    
    def export_to_json(self, session_key: str, output_path: Union[str, Path], 
                      pretty_print: bool = True) -> None:
        """Export parsed data to JSON file.
        
        Args:
            session_key: Session key of the data to export
            output_path: Path for the output JSON file
            pretty_print: Whether to format JSON with indentation
        """
        try:
            session_data = self.session_manager.get_session_data_by_key(session_key)
            
            if not session_data:
                raise ValueError(f"No data found for session key: {session_key}")
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if pretty_print:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(session_data, f, ensure_ascii=False)
            
            logger.info(f"Exported session data to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            raise
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get statistics about parsing operations.
        
        Returns:
            Dictionary with parsing statistics
        """
        try:
            session_stats = self.session_manager.get_session_stats()
            parsing_history = self.session_manager.get_parsing_history()
            
            # Calculate additional statistics
            recent_parses = [h for h in parsing_history 
                           if datetime.fromisoformat(h['timestamp']) > 
                           datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)]
            
            return {
                'session_statistics': session_stats,
                'parsing_history_count': len(parsing_history),
                'recent_parses_today': len(recent_parses),
                'available_sessions': session_stats.get('session_keys', []),
                'engine_info': {
                    'extraction_patterns': len(self.extraction_engine._compiled_patterns),
                    'parser_version': '2.0.0'
                }
            }
        
        except Exception as e:
            logger.error(f"Error getting parsing statistics: {str(e)}")
            return {'error': str(e)}
    
    def search_across_sessions(self, query: str, threshold: int = 70) -> List[Dict[str, Any]]:
        """Search for data across all parsed sessions.
        
        Args:
            query: Search query string
            threshold: Minimum similarity threshold (0-100)
            
        Returns:
            List of search results with session context
        """
        try:
            return self.session_manager.search_value(query, threshold)
        except Exception as e:
            logger.error(f"Error searching across sessions: {str(e)}")
            return []
    
    def get_session_data(self, session_key: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific session.
        
        Args:
            session_key: Session identifier
            
        Returns:
            Session data or None if not found
        """
        try:
            return self.session_manager.get_session_data_by_key(session_key)
        except Exception as e:
            logger.error(f"Error getting session data: {str(e)}")
            return None
    
    def clear_all_sessions(self) -> None:
        """Clear all session data."""
        try:
            self.session_manager.clear_session()
            logger.info("All session data cleared")
        except Exception as e:
            logger.error(f"Error clearing sessions: {str(e)}")
            raise
    
    def validate_document_format(self, content: str) -> Dict[str, Any]:
        """Validate if document content appears to be a valid PSAK219 document.
        
        Args:
            content: Document content to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'is_valid': False,
                'confidence_score': 0,
                'found_sections': [],
                'missing_sections': [],
                'recommendations': []
            }
            
            # Check for key PSAK219 indicators
            psak219_indicators = [
                ('company_info', ['nama perusahaan', 'periode valuasi']),
                ('employee_data', ['jumlah karyawan', 'rata-rata usia']),
                ('actuarial_assumptions', ['tingkat diskonto', 'kenaikan gaji']),
                ('financial_results', ['nilai kini kewajiban', 'biaya jasa kini']),
                ('sensitivity_analysis', ['sensitivitas', 'analisis sensitivitas'])
            ]
            
            content_lower = content.lower()
            found_sections = []
            
            for section_name, keywords in psak219_indicators:
                section_found = any(keyword in content_lower for keyword in keywords)
                if section_found:
                    found_sections.append(section_name)
            
            # Calculate confidence score
            confidence_score = (len(found_sections) / len(psak219_indicators)) * 100
            
            validation_results.update({
                'is_valid': confidence_score >= 60,  # At least 60% of sections found
                'confidence_score': confidence_score,
                'found_sections': found_sections,
                'missing_sections': [section for section, _ in psak219_indicators 
                                   if section not in found_sections]
            })
            
            # Generate recommendations
            if confidence_score < 60:
                validation_results['recommendations'].append(
                    "Document may not be a complete PSAK219 actuarial report"
                )
            if 'company_info' not in found_sections:
                validation_results['recommendations'].append(
                    "Missing company information section"
                )
            if 'financial_results' not in found_sections:
                validation_results['recommendations'].append(
                    "Missing financial results section"
                )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating document format: {str(e)}")
            return {
                'is_valid': False,
                'error': str(e),
                'confidence_score': 0
            }