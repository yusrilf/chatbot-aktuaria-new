"""JSON Financial Report Processor for actuarial chatbot application.

This module provides specialized processing for JSON financial reports,
optimizing content extraction and chunking for RAG system.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from langchain_core.documents import Document
from datetime import datetime
import uuid

from app.models.embeddings import VectorStoreManager

logger = logging.getLogger(__name__)


class JSONFinancialProcessor:
    """Specialized processor for JSON financial reports.
    
    This class handles JSON financial reports with optimized chunking
    and metadata extraction for RAG system performance.
    """
    
    def __init__(self) -> None:
        """Initialize the JSON financial processor."""
        self.vector_store_manager = VectorStoreManager()
        
    def process_json_financial_report(
        self, 
        file_content: bytes, 
        filename: str, 
        session_id: str
    ) -> Dict[str, Any]:
        """Process JSON financial report for optimal RAG performance.
        
        Args:
            file_content: Binary content of JSON file
            filename: Original filename
            session_id: Session identifier
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            json.JSONDecodeError: If JSON is invalid
            Exception: For other processing errors
        """
        try:
            # Parse JSON content
            json_str = file_content.decode('utf-8')
            json_data = json.loads(json_str)
            
            logger.info(f"Processing JSON financial report: {filename}")
            
            # Extract and chunk content optimally
            documents = self._extract_and_chunk_content(
                json_data, 
                filename, 
                session_id
            )
            
            # Add documents to vector store
            if documents:
                success = self.vector_store_manager.document_manager.add_documents(documents)
                if success:
                    logger.info(f"Added {len(documents)} documents to vector store")
                else:
                    logger.error("Failed to add documents to vector store")
                    raise Exception("Failed to add documents to vector store")
            
            return {
                'success': True,
                'filename': filename,
                'chunks_created': len(documents),
                'documents': documents,  # Add documents to result
                'document_type': 'financial_report_json',
                'processed': True,
                'embedding': True
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'error': f'Invalid JSON format: {str(e)}',
                'chunks_created': 0,
                'documents': []  # Add empty documents list
            }
        except Exception as e:
            logger.error(f"Error processing JSON financial report {filename}: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'error': str(e),
                'chunks_created': 0,
                'documents': []  # Add empty documents list
            }
    
    def _extract_and_chunk_content(
        self, 
        json_data: Dict[str, Any], 
        filename: str, 
        session_id: str
    ) -> List[Document]:
        """Extract and chunk content from JSON financial report.
        
        Args:
            json_data: Parsed JSON data
            filename: Original filename
            session_id: Session identifier
            
        Returns:
            List of Document objects with optimized chunks
        """
        documents = []
        
        try:
            # Extract main report data - handle multiple JSON structures
            if 'laporan_valuasi_aktuaria' in json_data:
                # Structured actuarial report
                report_data = json_data['laporan_valuasi_aktuaria']
                
                # Process different sections with specialized chunking
                documents.extend(self._process_general_info(report_data, filename, session_id))
                documents.extend(self._process_employee_data(report_data, filename, session_id))
                documents.extend(self._process_financial_assumptions(report_data, filename, session_id))
                documents.extend(self._process_actuarial_results(report_data, filename, session_id))
                documents.extend(self._process_calculation_details(report_data, filename, session_id))
            else:
                # Generic JSON processing - flatten and chunk any JSON structure
                documents.extend(self._process_generic_json(json_data, filename, session_id))
                
            logger.info(f"Extracted {len(documents)} chunks from {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting content from {filename}: {str(e)}")
            return []
    
    def _process_general_info(
        self, 
        report_data: Dict[str, Any], 
        filename: str, 
        session_id: str
    ) -> List[Document]:
        """Process general information section.
        
        Args:
            report_data: Report data dictionary
            filename: Original filename
            session_id: Session identifier
            
        Returns:
            List of Document objects for general info
        """
        documents = []
        
        try:
            # Company and report information
            if 'informasi_umum' in report_data:
                info = report_data['informasi_umum']
                
                content_parts = []
                content_parts.append(f"LAPORAN VALUASI AKTUARIA - {report_data.get('nama_perusahaan', 'N/A')}")
                content_parts.append(f"Nomor Laporan: {report_data.get('nomor_laporan', 'N/A')}")
                content_parts.append(f"Periode Valuasi: {report_data.get('periode_valuasi', 'N/A')}")
                
                for key, value in info.items():
                    if isinstance(value, dict) and 'label' in value and 'value' in value:
                        content_parts.append(f"{value['label']}: {value['value']}")
                
                content = "\n".join(content_parts)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'chunk_id': str(uuid.uuid4()),
                        'session_id': session_id,
                        'filename': filename,
                        'source': filename,  # Add source field for test detection
                        'document_type': 'financial_report_json',
                        'section_type': 'general_information',
                        'section_name': 'Informasi Umum',
                        'content_type': 'company_info',
                        'timestamp': datetime.now().isoformat(),
                        'chunk_index': 0
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error processing general info: {str(e)}")
            
        return documents
    
    def _process_generic_json(
        self, 
        json_data: Dict[str, Any], 
        filename: str, 
        session_id: str
    ) -> List[Document]:
        """Process generic JSON structure by flattening and chunking.
        
        Args:
            json_data: JSON data dictionary
            filename: Original filename
            session_id: Session identifier
            
        Returns:
            List of Document objects for generic JSON
        """
        documents = []
        
        try:
            # Flatten JSON structure into readable text chunks
            flattened_content = self._flatten_json_to_text(json_data)
            
            # Split content into manageable chunks
            chunks = self._split_content_into_chunks(flattened_content, max_chunk_size=1000)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only create document if chunk has content
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'chunk_id': str(uuid.uuid4()),
                            'session_id': session_id,
                            'filename': filename,
                            'source': filename,  # Add source field for test detection
                            'document_type': 'generic_json',
                            'section_type': 'generic_content',
                            'section_name': f'JSON Content Part {i+1}',
                            'content_type': 'json_data',
                            'timestamp': datetime.now().isoformat(),
                            'chunk_index': i
                        }
                    )
                    documents.append(doc)
                    
        except Exception as e:
            logger.error(f"Error processing generic JSON: {str(e)}")
            
        return documents
    
    def _flatten_json_to_text(self, data: Any, prefix: str = "") -> str:
        """Recursively flatten JSON data into readable text.
        
        Args:
            data: JSON data to flatten
            prefix: Current key prefix for nested structures
            
        Returns:
            Flattened text representation
        """
        lines = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    lines.append(f"{current_prefix}:")
                    lines.append(self._flatten_json_to_text(value, current_prefix))
                else:
                    lines.append(f"{current_prefix}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_prefix = f"{prefix}[{i}]" if prefix else f"item_{i}"
                if isinstance(item, (dict, list)):
                    lines.append(f"{current_prefix}:")
                    lines.append(self._flatten_json_to_text(item, current_prefix))
                else:
                    lines.append(f"{current_prefix}: {item}")
        else:
            return str(data)
            
        return "\n".join(lines)
    
    def _split_content_into_chunks(self, content: str, max_chunk_size: int = 1000) -> List[str]:
        """Split content into chunks of specified maximum size.
        
        Args:
            content: Text content to split
            max_chunk_size: Maximum size per chunk
            
        Returns:
            List of content chunks
        """
        if len(content) <= max_chunk_size:
            return [content]
            
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
    
    def _process_employee_data(
        self, 
        report_data: Dict[str, Any], 
        filename: str, 
        session_id: str
    ) -> List[Document]:
        """Process employee data section.
        
        Args:
            report_data: Report data dictionary
            filename: Original filename
            session_id: Session identifier
            
        Returns:
            List of Document objects for employee data
        """
        documents = []
        
        try:
            if 'isi_laporan' in report_data and 'ii_ringkasan_data' in report_data['isi_laporan']:
                data_section = report_data['isi_laporan']['ii_ringkasan_data']
                
                # Process employee data tables
                if 'ii_1_data_karyawan' in data_section:
                    emp_data = data_section['ii_1_data_karyawan']
                    
                    content_parts = []
                    content_parts.append("DATA KARYAWAN")
                    content_parts.append("=" * 50)
                    
                    if 'table' in emp_data:
                        table = emp_data['table']
                        if 'headers' in table:
                            headers = table['headers']
                            content_parts.append(f"Kategori: {headers.get('permanent', 'N/A')} | {headers.get('contract', 'N/A')}")
                        
                        if 'rows' in table:
                            for row in table['rows']:
                                if isinstance(row, dict):
                                    desc = row.get('deskripsi', 'N/A')
                                    perm = row.get('permanent', 'N/A')
                                    cont = row.get('contract', 'N/A')
                                    content_parts.append(f"{desc}: Permanent={perm}, Contract={cont}")
                    
                    content = "\n".join(content_parts)
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'chunk_id': str(uuid.uuid4()),
                            'session_id': session_id,
                            'filename': filename,
                            'source': filename,  # Add source field for test detection
                            'document_type': 'financial_report_json',
                            'section_type': 'employee_data',
                            'section_name': 'Data Karyawan',
                            'content_type': 'statistical_data',
                            'timestamp': datetime.now().isoformat(),
                            'chunk_index': len(documents)
                        }
                    )
                    documents.append(doc)
                    
        except Exception as e:
            logger.error(f"Error processing employee data: {str(e)}")
            
        return documents
    
    def _process_financial_assumptions(
        self, 
        report_data: Dict[str, Any], 
        filename: str, 
        session_id: str
    ) -> List[Document]:
        """Process financial assumptions section.
        
        Args:
            report_data: Report data dictionary
            filename: Original filename
            session_id: Session identifier
            
        Returns:
            List of Document objects for financial assumptions
        """
        documents = []
        
        try:
            if 'isi_laporan' in report_data:
                isi_laporan = report_data['isi_laporan']
                
                # Process assumptions section
                if 'iii_asumsi_aktuaria' in isi_laporan:
                    assumptions = isi_laporan['iii_asumsi_aktuaria']
                    
                    content_parts = []
                    content_parts.append("ASUMSI AKTUARIA")
                    content_parts.append("=" * 50)
                    
                    # Process all assumption subsections
                    for key, value in assumptions.items():
                        if isinstance(value, dict):
                            if 'title' in value:
                                content_parts.append(f"\n{value['title']}")
                                content_parts.append("-" * len(value['title']))
                            
                            # Process content
                            if 'content' in value:
                                if isinstance(value['content'], list):
                                    for item in value['content']:
                                        if isinstance(item, str):
                                            content_parts.append(item)
                                        elif isinstance(item, dict):
                                            content_parts.append(self._format_dict_content(item))
                                elif isinstance(value['content'], str):
                                    content_parts.append(value['content'])
                                elif isinstance(value['content'], dict):
                                    content_parts.append(self._format_dict_content(value['content']))
                            
                            # Process tables if present
                            if 'table' in value:
                                table_content = self._format_table_content(value['table'])
                                if table_content:
                                    content_parts.append(table_content)
                    
                    if content_parts:
                        content = "\n".join(content_parts)
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                'chunk_id': str(uuid.uuid4()),
                                'session_id': session_id,
                                'filename': filename,
                                'source': filename,
                                'document_type': 'financial_report_json',
                                'section_type': 'financial_assumptions',
                                'section_name': 'Asumsi Aktuaria',
                                'content_type': 'assumptions_data',
                                'timestamp': datetime.now().isoformat(),
                                'chunk_index': 0
                            }
                        )
                        documents.append(doc)
                        
        except Exception as e:
            logger.error(f"Error processing financial assumptions: {str(e)}")
            
        return documents
    
    def _process_actuarial_results(
        self, 
        report_data: Dict[str, Any], 
        filename: str, 
        session_id: str
    ) -> List[Document]:
        """Process actuarial calculation results.
        
        Args:
            report_data: Report data dictionary
            filename: Original filename
            session_id: Session identifier
            
        Returns:
            List of Document objects for actuarial results
        """
        documents = []
        
        try:
            if 'isi_laporan' in report_data:
                isi_laporan = report_data['isi_laporan']
                
                # Process results sections
                result_sections = [
                    'iv_hasil_perhitungan',
                    'v_analisis_sensitivitas',
                    'vi_kesimpulan'
                ]
                
                for section_key in result_sections:
                    if section_key in isi_laporan:
                        section_data = isi_laporan[section_key]
                        
                        content_parts = []
                        section_title = section_key.replace('_', ' ').upper()
                        content_parts.append(section_title)
                        content_parts.append("=" * len(section_title))
                        
                        # Process section content
                        if isinstance(section_data, dict):
                            for key, value in section_data.items():
                                if isinstance(value, dict):
                                    if 'title' in value:
                                        content_parts.append(f"\n{value['title']}")
                                        content_parts.append("-" * len(value['title']))
                                    
                                    if 'content' in value:
                                        content_parts.append(self._format_content_recursive(value['content']))
                                    
                                    if 'table' in value:
                                        table_content = self._format_table_content(value['table'])
                                        if table_content:
                                            content_parts.append(table_content)
                                elif isinstance(value, (str, int, float)):
                                    content_parts.append(f"{key}: {value}")
                        
                        if len(content_parts) > 2:  # More than just title and separator
                            content = "\n".join(content_parts)
                            
                            doc = Document(
                                page_content=content,
                                metadata={
                                    'chunk_id': str(uuid.uuid4()),
                                    'session_id': session_id,
                                    'filename': filename,
                                    'source': filename,
                                    'document_type': 'financial_report_json',
                                    'section_type': 'actuarial_results',
                                    'section_name': section_title,
                                    'content_type': 'calculation_results',
                                    'timestamp': datetime.now().isoformat(),
                                    'chunk_index': len(documents)
                                }
                            )
                            documents.append(doc)
                            
        except Exception as e:
            logger.error(f"Error processing actuarial results: {str(e)}")
            
        return documents
    
    def _process_calculation_details(
        self, 
        report_data: Dict[str, Any], 
        filename: str, 
        session_id: str
    ) -> List[Document]:
        """Process detailed calculation data.
        
        Args:
            report_data: Report data dictionary
            filename: Original filename
            session_id: Session identifier
            
        Returns:
            List of Document objects for calculation details
        """
        documents = []
        
        try:
            # Process any remaining sections not covered by other methods
            if 'isi_laporan' in report_data:
                isi_laporan = report_data['isi_laporan']
                
                # Look for calculation details in various sections
                detail_sections = [
                    'i_pendahuluan',
                    'vii_lampiran',
                    'viii_appendix'
                ]
                
                for section_key in detail_sections:
                    if section_key in isi_laporan:
                        section_data = isi_laporan[section_key]
                        
                        content_parts = []
                        section_title = section_key.replace('_', ' ').upper()
                        content_parts.append(section_title)
                        content_parts.append("=" * len(section_title))
                        
                        # Process section content recursively
                        content_parts.append(self._format_content_recursive(section_data))
                        
                        if len(content_parts) > 2:  # More than just title and separator
                            content = "\n".join(content_parts)
                            
                            doc = Document(
                                page_content=content,
                                metadata={
                                    'chunk_id': str(uuid.uuid4()),
                                    'session_id': session_id,
                                    'filename': filename,
                                    'source': filename,
                                    'document_type': 'financial_report_json',
                                    'section_type': 'calculation_details',
                                    'section_name': section_title,
                                    'content_type': 'detailed_calculations',
                                    'timestamp': datetime.now().isoformat(),
                                    'chunk_index': len(documents)
                                }
                            )
                            documents.append(doc)
                            
        except Exception as e:
            logger.error(f"Error processing calculation details: {str(e)}")
            
        return documents
    
    def _format_content_recursive(self, content: Any) -> str:
        """Recursively format content of any type into readable text.
        
        Args:
            content: Content to format
            
        Returns:
            Formatted text string
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, (int, float)):
            return str(content)
        elif isinstance(content, list):
            formatted_items = []
            for item in content:
                formatted_items.append(self._format_content_recursive(item))
            return "\n".join(formatted_items)
        elif isinstance(content, dict):
            return self._format_dict_content(content)
        else:
            return str(content)
    
    def _format_dict_content(self, data: Dict[str, Any]) -> str:
        """Format dictionary content into readable text.
        
        Args:
            data: Dictionary to format
            
        Returns:
            Formatted text string
        """
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                if 'label' in value and 'value' in value:
                    lines.append(f"{value['label']}: {value['value']}")
                else:
                    lines.append(f"{key}:")
                    lines.append(self._format_content_recursive(value))
            elif isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {self._format_content_recursive(item)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    def _format_table_content(self, table: Dict[str, Any]) -> str:
        """Format table content into readable text.
        
        Args:
            table: Table data dictionary
            
        Returns:
            Formatted table text
        """
        lines = []
        
        if 'headers' in table:
            headers = table['headers']
            if isinstance(headers, dict):
                header_line = " | ".join(str(v) for v in headers.values())
                lines.append(header_line)
                lines.append("-" * len(header_line))
            elif isinstance(headers, list):
                header_line = " | ".join(str(h) for h in headers)
                lines.append(header_line)
                lines.append("-" * len(header_line))
        
        if 'rows' in table:
            for row in table['rows']:
                if isinstance(row, dict):
                    row_line = " | ".join(str(v) for v in row.values())
                    lines.append(row_line)
                elif isinstance(row, list):
                    row_line = " | ".join(str(item) for item in row)
                    lines.append(row_line)
        
        return "\n".join(lines)