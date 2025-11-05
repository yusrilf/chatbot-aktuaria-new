"""Document export service.

This module provides functionality to export document data
in various formats for analysis and reporting.
"""

import json
import logging
from typing import Dict, List, Any, Union, Optional
from pathlib import Path
from datetime import datetime

from .document_registry import DocumentRegistry
from .document_search import DocumentSearchService
from ..parser import SessionManager

logger = logging.getLogger(__name__)

class DocumentExportService:
    """Service for exporting document data in various formats.
    
    This service provides functionality to export parsed document data
    for analysis, reporting, and integration with other systems.
    """
    
    def __init__(self, registry: DocumentRegistry, 
                 search_service: DocumentSearchService,
                 session_manager: SessionManager):
        """Initialize the document export service.
        
        Args:
            registry: Document registry for metadata management
            search_service: Search service for data retrieval
            session_manager: Session manager for accessing parsed data
        """
        self.registry = registry
        self.search_service = search_service
        self.session_manager = session_manager
        logger.info("DocumentExportService initialized")
    
    def export_document_data(self, document_id: str, 
                           output_path: Union[str, Path],
                           format_type: str = "json") -> None:
        """Export all data from a specific document.
        
        Args:
            document_id: Document identifier
            output_path: Path where to save the exported data
            format_type: Export format ('json', 'csv', 'txt')
            
        Raises:
            ValueError: If document not found or format not supported
        """
        try:
            metadata = self.registry.get_document(document_id)
            if not metadata:
                raise ValueError(f"Document not found: {document_id}")
            
            if not metadata.is_parsed():
                raise ValueError(f"Document not parsed yet: {document_id}")
            
            # Get all document data
            document_data = self.search_service.get_document_data(document_id)
            
            # Prepare export data
            export_data = {
                'metadata': metadata.to_dict(),
                'export_timestamp': datetime.now().isoformat(),
                'document_data': document_data
            }
            
            output_path = Path(output_path)
            
            if format_type.lower() == "json":
                self._export_as_json(export_data, output_path)
            elif format_type.lower() == "txt":
                self._export_as_text(export_data, output_path)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            logger.info(f"Exported document {document_id} to {output_path}")
        
        except Exception as e:
            logger.error(f"Error exporting document {document_id}: {str(e)}")
            raise
    
    def _export_as_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data as JSON format.
        
        Args:
            data: Data to export
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error exporting as JSON: {str(e)}")
            raise
    
    def _export_as_text(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export data as readable text format.
        
        Args:
            data: Data to export
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write metadata
                f.write("DOCUMENT EXPORT REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                metadata = data.get('metadata', {})
                f.write(f"Document ID: {metadata.get('document_id', 'N/A')}\n")
                f.write(f"Original Filename: {metadata.get('original_filename', 'N/A')}\n")
                f.write(f"Document Type: {metadata.get('document_type', 'N/A')}\n")
                f.write(f"Upload Timestamp: {metadata.get('upload_timestamp', 'N/A')}\n")
                f.write(f"Export Timestamp: {data.get('export_timestamp', 'N/A')}\n\n")
                
                # Write document data
                f.write("DOCUMENT DATA\n")
                f.write("-" * 30 + "\n")
                
                document_data = data.get('document_data', {})
                self._write_nested_dict(f, document_data, indent=0)
        
        except Exception as e:
            logger.error(f"Error exporting as text: {str(e)}")
            raise
    
    def _write_nested_dict(self, file_handle, data: Any, indent: int = 0) -> None:
        """Write nested dictionary data to text file.
        
        Args:
            file_handle: File handle to write to
            data: Data to write
            indent: Current indentation level
        """
        indent_str = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    file_handle.write(f"{indent_str}{key}:\n")
                    self._write_nested_dict(file_handle, value, indent + 1)
                else:
                    file_handle.write(f"{indent_str}{key}: {value}\n")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                file_handle.write(f"{indent_str}[{i}]:\n")
                self._write_nested_dict(file_handle, item, indent + 1)
        else:
            file_handle.write(f"{indent_str}{data}\n")
    
    def export_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Export a summary of key document information.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with document summary
            
        Raises:
            ValueError: If document not found or not parsed
        """
        try:
            metadata = self.registry.get_document(document_id)
            if not metadata:
                raise ValueError(f"Document not found: {document_id}")
            
            if not metadata.is_parsed():
                return {
                    'document_id': document_id,
                    'status': metadata.parsing_status,
                    'error': metadata.error_message
                }
            
            # Get key data points
            summary = {
                'document_id': document_id,
                'original_filename': metadata.original_filename,
                'status': 'completed',
                'company_name': self.search_service.get_document_data(
                    document_id, 'informasi_umum.nama_perusahaan'
                ),
                'valuation_period': self.search_service.get_document_data(
                    document_id, 'informasi_umum.periode_valuasi'
                ),
                'total_employees': {
                    'permanent': self.search_service.get_document_data(
                        document_id, 'data_karyawan.permanent_count'
                    ),
                    'contract': self.search_service.get_document_data(
                        document_id, 'data_karyawan.contract_count'
                    )
                },
                'key_assumptions': {
                    'discount_rate': self.search_service.get_document_data(
                        document_id, 'asumsi_aktuaria.discount_rate'
                    ),
                    'salary_increase_rate': self.search_service.get_document_data(
                        document_id, 'asumsi_aktuaria.salary_increase_rate'
                    ),
                    'retirement_age': self.search_service.get_document_data(
                        document_id, 'asumsi_aktuaria.normal_retirement_age'
                    )
                },
                'financial_highlights': {
                    'present_value_obligation': self.search_service.get_document_data(
                        document_id, 'hasil_keuangan.present_value_obligation_current'
                    ),
                    'current_service_cost': self.search_service.get_document_data(
                        document_id, 'hasil_keuangan.current_service_cost_current'
                    )
                }
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting document summary for {document_id}: {str(e)}")
            raise
    
    def export_all_documents_summary(self) -> List[Dict[str, Any]]:
        """Export summary of all documents.
        
        Returns:
            List of document summaries
        """
        try:
            summaries = []
            for metadata in self.registry.list_documents():
                try:
                    if metadata.is_parsed():
                        summary = self.export_document_summary(metadata.document_id)
                    else:
                        summary = {
                            'document_id': metadata.document_id,
                            'original_filename': metadata.original_filename,
                            'status': metadata.parsing_status,
                            'error': metadata.error_message
                        }
                    summaries.append(summary)
                except Exception as e:
                    logger.warning(f"Error getting summary for {metadata.document_id}: {str(e)}")
                    summaries.append({
                        'document_id': metadata.document_id,
                        'status': 'error',
                        'error': str(e)
                    })
            
            return summaries
        
        except Exception as e:
            logger.error(f"Error exporting all documents summary: {str(e)}")
            raise
    
    def export_search_results(self, query: str, threshold: int = 70,
                            output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Export search results across all documents.
        
        Args:
            query: Search query
            threshold: Similarity threshold
            output_path: Optional path to save results
            
        Returns:
            Dictionary with search results
        """
        try:
            search_results = self.search_service.search_all_documents(query, threshold)
            
            export_data = {
                'query': query,
                'threshold': threshold,
                'search_timestamp': datetime.now().isoformat(),
                'total_documents_searched': self.registry.get_parsed_document_count(),
                'documents_with_results': len(search_results),
                'results': search_results
            }
            
            if output_path:
                output_path = Path(output_path)
                self._export_as_json(export_data, output_path)
                logger.info(f"Search results exported to {output_path}")
            
            return export_data
        
        except Exception as e:
            logger.error(f"Error exporting search results: {str(e)}")
            raise