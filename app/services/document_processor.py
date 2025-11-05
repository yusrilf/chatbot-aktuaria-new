"""Document processing service for actuarial chatbot application.

This module provides document processing capabilities including markdown parsing,
PSAK219 document handling, adaptive chunking, and metadata extraction.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

import os
import markdown
from typing import List, Dict, Any, Optional, Union
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
import re
import yaml
import json
from datetime import datetime

from app.config import config
from app.services.parser import PSAK219DocumentParser, SessionManager, _global_session_manager
# Enhanced PSAK219 parser removed - using direct document processing
from app.services.adaptive_chunking_strategy import adaptive_chunking_strategy
from app.services.json_financial_processor import JSONFinancialProcessor
from app.services.chunk_metadata_extractor import ChunkMetadataExtractor

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents for the actuarial chatbot system.
    
    This class handles document processing including markdown parsing,
    PSAK219 document analysis, adaptive chunking, and metadata extraction.
    """
    def __init__(self) -> None:
        """Initialize the document processor with default configurations.
        
        Sets up markdown splitters, text splitters, adaptive chunking strategy,
        and PSAK219 document parser.
        """
        # Default markdown splitter for hierarchical document structure
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )
        
        # Recursive text splitter with configurable chunk size and overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        # Initialize adaptive chunking strategy for intelligent document splitting
        self.adaptive_strategy = adaptive_chunking_strategy
        
        # Initialize Enhanced PSAK219 parser with global session manager
        self.session_manager = _global_session_manager
        # Enhanced PSAK219 parser removed - using direct document processing
        self.psak219_parser = None
        
        # Initialize JSON financial processor for JSON reports
        self.json_processor = JSONFinancialProcessor()
        
        # Initialize chunk metadata extractor for enhanced metadata
        self.metadata_extractor = ChunkMetadataExtractor()
        
        logger.info("DocumentProcessor initialized successfully")
    
    def process_document(self, file_content: bytes, filename: str, request_id: str) -> Dict[str, Any]:
        """Process a single uploaded document.
        
        Args:
            file_content: Binary content of the uploaded file
            filename: Original filename
            request_id: Request ID for tracking
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Check if file is JSON
            if filename.lower().endswith('.json'):
                logger.info(f"Processing JSON financial report: {filename}")
                return self.json_processor.process_json_financial_report(
                    file_content=file_content,
                    filename=filename,
                    session_id=request_id
                )
            
            # Process as markdown (existing logic)
            import tempfile
            import os
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.md', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Process the temporary file
                documents = self.process_markdown_file(
                    file_path=temp_file_path,
                    session_id=request_id,
                    original_filename=filename
                )
                
                return {
                    'success': True,
                    'filename': filename,
                    'chunks_created': len(documents),
                    'document_type': 'markdown',
                    'processed': True,
                    'embedding': True,  # Indicate embeddings were created
                    'processing_time': 0.0  # Will be calculated by caller
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            return {
                'success': False,
                'filename': filename,
                'error': str(e),
                'processing_time': 0.0
            }
    
    def process_markdown_file(
        self, 
        file_path: str, 
        session_id: str, 
        original_filename: Optional[str] = None
    ) -> List[Document]:
        """Process a single markdown file into documents with normalized metadata.
        
        Args:
            file_path: Path to the markdown file to process
            session_id: Session identifier for document processing
            original_filename: Original filename if different from file_path
            
        Returns:
            List of processed Document objects with metadata
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            UnicodeDecodeError: If file encoding is not UTF-8
            Exception: For other processing errors
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                raw_content = file.read()
            
            # 1) ambil front-matter YAML jika ada
            front_matter = {}
            content = raw_content
            if raw_content.strip().startswith("---"):
                # cari blok YAML di awal file
                parts = raw_content.split('---', 2)
                # parts[0] is empty, parts[1] is yaml, parts[2] is rest
                if len(parts) >= 3:
                    yaml_text = parts[1]
                    try:
                        front_matter = yaml.safe_load(yaml_text) or {}
                        content = parts[2]
                    except Exception as e:
                        logger.warning(f"YAML parse failed for {file_path}: {e}")
            
            # 2) normalize filename: pakai original_filename jika diberikan
            if original_filename:
                filename = original_filename
            else:
                filename = os.path.basename(file_path)
                # jika ada uuid_ prefix hapus
                filename = filename.split("_", 1)[-1] if "_" in filename else filename
            
            filename = filename.strip()
            
            # 3) tentukan document_type & folder
            # prioritas: front_matter -> filename pattern -> content heuristics -> default
            doc_type = front_matter.get('document_type') or self._extract_document_type(filename, content)
            # normalize some common names
            if isinstance(doc_type, str):
                document_type = doc_type.lower()
            else:
                document_type = 'general'
            
            # try get folder from front_matter
            folder = front_matter.get('folder') or front_matter.get('scope') or front_matter.get('domain')
            if not folder:
                # if filename contains known folder tokens (use config.FOLDERS if set)
                folders = getattr(config, 'FOLDERS', [])
                filename_lower = filename.lower()
                matched = next((f for f in folders if f.lower() in filename_lower), None)
                folder = matched or 'general'
            
            # PSAK219 Document Parsing Integration
            parsed_json_data = None
            is_psak219_document = False
            
            # Check if this is a PSAK219 document
            if self._is_psak219_document(filename, content, document_type):
                try:
                    logger.info(f"Detected PSAK219 document: {filename}")
                    is_psak219_document = True
                    
                    # Parse document to structured JSON using enhanced parser
                    if self.psak219_parser is not None:
                        parsed_json_data = self.psak219_parser.parse_document_enhanced(file_path)
                        
                        if parsed_json_data:
                            # Store parsed data in session
                            session_key = f"psak219_parsed_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            self.session_manager.set_session_variable(session_key, parsed_json_data)
                            
                            logger.info(f"PSAK219 data parsed and stored in session: {session_key}")
                            logger.info(f"Parsed sections: {list(parsed_json_data.keys())}")
                        else:
                            logger.warning(f"PSAK219 parsing returned empty data for: {filename}")
                    else:
                        logger.warning(f"PSAK219 parser not available, skipping enhanced parsing for: {filename}")
                        parsed_json_data = None
                        
                except Exception as e:
                    logger.error(f"Error parsing PSAK219 document {filename}: {str(e)}")
                    # Continue with normal processing even if PSAK219 parsing fails
                    parsed_json_data = None
            
            # 4) clean content and prepare for adaptive chunking
            content = self._clean_markdown_content(content)
            
            # Get optimal chunking configuration using adaptive strategy
            chunking_config = self.adaptive_strategy.get_optimal_chunking_config(content, file_path)
            
            # Create optimized splitters based on content analysis
            header_splitter, text_splitter = self.adaptive_strategy.create_optimized_splitters(chunking_config)
            
            # Use header splitter if available, otherwise use text splitter directly
            if header_splitter and chunking_config.preserve_headers:
                header_splits = header_splitter.split_text(content)
            else:
                # Split content directly with text splitter
                text_chunks = text_splitter.split_text(content)
                header_splits = [Document(page_content=chunk, metadata={}) for chunk in text_chunks]
            
            # 5) build Document list with normalized metadata
            documents = []
            global_chunk_idx = 0
            for i, split in enumerate(header_splits):
                # split may be string or Document-like
                if isinstance(split, str):
                    split_text = split
                    split_meta = {}
                else:
                    split_text = getattr(split, "page_content", str(split))
                    split_meta = getattr(split, "metadata", {}) or {}
                
                # base metadata
                base_meta = {
                    'source': file_path,
                    'filename': filename,
                    'document_type': document_type,   # normalized key
                    'doc_type': doc_type if isinstance(doc_type, str) else str(doc_type),
                    'folder': folder,
                    'session_id': session_id,
                    'is_psak219_document': is_psak219_document,
                    'has_parsed_json': parsed_json_data is not None,
                    'chunking_strategy': 'adaptive',
                    'optimal_chunk_size': chunking_config.base_chunk_size,
                    'chunk_overlap_ratio': chunking_config.overlap_ratio
                }
                
                # Add PSAK219 parsing metadata if available
                if parsed_json_data:
                    base_meta.update({
                        'psak219_session_key': session_key,
                        'psak219_sections': str(list(parsed_json_data.keys())),  # Convert list to string
                        'psak219_company': str(parsed_json_data.get('informasi_umum', {}).get('nama_perusahaan', '')),
                        'psak219_period': str(parsed_json_data.get('informasi_umum', {}).get('periode_valuasi', ''))
                    })
                # include any header metadata from splitter (Header1, Header2, etc.)
                for k, v in split_meta.items():
                    base_meta[k] = v
                
                # further split large chunks using adaptive text splitter
                if len(split_text) > chunking_config.max_chunk_size:
                    sub_chunks = text_splitter.split_text(split_text)
                    for j, sub_chunk in enumerate(sub_chunks):
                        # Extract enhanced metadata using ChunkMetadataExtractor
                        try:
                            enhanced_meta = self.metadata_extractor.extract_metadata(
                                content=sub_chunk,
                                chunk_id=str(global_chunk_idx),
                                doc_name=filename,
                                section_info=split_meta,
                                front_matter=front_matter
                            )
                            logger.debug(f"Enhanced metadata extracted for chunk {global_chunk_idx}: {enhanced_meta.to_dict()}")
                        except Exception as e:
                            logger.error(f"Error extracting enhanced metadata for chunk {global_chunk_idx}: {e}")
                            enhanced_meta = None
                        
                        # Combine with base metadata
                        chunk_meta = base_meta.copy()
                        if enhanced_meta:
                            chunk_meta.update(enhanced_meta.to_dict())
                        chunk_meta['chunk_id'] = global_chunk_idx
                        chunk_meta['sub_chunk_id'] = j
                        chunk_meta['header_preview'] = (split_text[:120] + '...') if len(split_text) > 120 else split_text
                        chunk_meta['chunk_size'] = len(sub_chunk)
                        
                        # Filter complex metadata before creating document
                        filtered_meta = self._filter_complex_metadata(chunk_meta)
                        documents.append(Document(page_content=sub_chunk, metadata=filtered_meta))
                        global_chunk_idx += 1
                else:
                    # Extract enhanced metadata using ChunkMetadataExtractor
                    try:
                        enhanced_meta = self.metadata_extractor.extract_metadata(
                            content=split_text,
                            chunk_id=str(global_chunk_idx),
                            doc_name=filename,
                            section_info=split_meta,
                            front_matter=front_matter
                        )
                        logger.debug(f"Enhanced metadata extracted for chunk {global_chunk_idx}: {enhanced_meta.to_dict()}")
                    except Exception as e:
                        logger.error(f"Error extracting enhanced metadata for chunk {global_chunk_idx}: {e}")
                        enhanced_meta = None
                    
                    # Combine with base metadata
                    chunk_meta = base_meta.copy()
                    if enhanced_meta:
                        chunk_meta.update(enhanced_meta.to_dict())
                    chunk_meta['chunk_id'] = global_chunk_idx
                    chunk_meta['header_preview'] = (split_text[:120] + '...') if len(split_text) > 120 else split_text
                    chunk_meta['chunk_size'] = len(split_text)
                    
                    # Filter complex metadata before creating document
                    filtered_meta = self._filter_complex_metadata(chunk_meta)
                    documents.append(Document(page_content=split_text, metadata=filtered_meta))
                    global_chunk_idx += 1
            
            # Validate chunks using adaptive strategy
            validation_result = self.adaptive_strategy.validate_chunks(documents, chunking_config)
            
            logger.info(
                f"Processed {filename} into {len(documents)} chunks using adaptive strategy "
                f"(base_size: {chunking_config.base_chunk_size}, overlap: {chunking_config.overlap_ratio:.2f})"
            )
            
            if validation_result.get('warnings'):
                logger.warning(f"Chunking validation warnings for {filename}: {validation_result['warnings']}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing markdown file {file_path}: {str(e)}")
            return []
    
    def _clean_markdown_content(self, content: str) -> str:
        """Remove unwanted headers, footers, and repeated lines from markdown.
        
        Args:
            content: Raw markdown content to clean
            
        Returns:
            Cleaned markdown content with noise removed
            
        Note:
            Removes page numbers, URLs, repeated headers, and common footer patterns
        """
        # Hapus footer/header umum
        lines = content.splitlines()
        clean_lines = []

        for line in lines:
            line = line.strip()

            # Skip empty or noise lines
            if not line or re.match(r'^(Page \d+ of \d+|Halaman \d+|^#+\s*$|^\*+$|^[-=]{3,})$', line):
                continue

            # Hapus URL tidak penting
            if re.match(r'https?://', line):
                continue

            clean_lines.append(line)

        cleaned_content = '\n'.join(clean_lines)

        # Optional: Hilangkan header/footer yang diulang (misal: "Asuransi ABC 2024")
        cleaned_content = re.sub(r"(Asuransi\s+\w+\s+\d{4})", "", cleaned_content)

        return cleaned_content.strip()

    def _filter_complex_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Filter complex metadata to only include simple types supported by ChromaDB.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Filtered metadata with only str, int, float, bool values
            
        Note:
            ChromaDB only supports simple metadata types. Complex types like lists,
            dicts, and objects are converted to strings or removed.
        """
        filtered = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to comma-separated strings
                try:
                    filtered[key] = ', '.join(str(item) for item in value)
                except Exception:
                    filtered[key] = str(value)
            elif isinstance(value, dict):
                # Convert dicts to JSON strings (truncated if too long)
                try:
                    import json
                    json_str = json.dumps(value, ensure_ascii=False)
                    if len(json_str) > 500:  # Truncate very long JSON
                        json_str = json_str[:497] + '...'
                    filtered[key] = json_str
                except Exception:
                    filtered[key] = str(value)
            else:
                # Convert other types to strings
                try:
                    str_value = str(value)
                    if len(str_value) > 500:  # Truncate very long strings
                        str_value = str_value[:497] + '...'
                    filtered[key] = str_value
                except Exception:
                    # Skip values that can't be converted to string
                    continue
        
        return filtered

    def process_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """Process multiple markdown files into a combined document list.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Combined list of processed Document objects from all files
            
        Note:
            Only processes files with .md extension
        """
        all_documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.md'):
                documents = self.process_markdown_file(file_path)
                all_documents.extend(documents)
            else:
                logger.warning(f"Skipping non-markdown file: {file_path}")
        
        logger.info(f"Processed {len(file_paths)} files into {len(all_documents)} total chunks")
        return all_documents
    
    def _extract_document_type(self, filename: str, content: str) -> str:
        """Extract document type from filename and content analysis.
        
        Args:
            filename: Name of the file to analyze
            content: Content of the file to analyze
            
        Returns:
            Document type classification (manual, financial_report, formula, regulation, general)
            
        Note:
            Uses pattern matching on both filename and content to determine document type
        """
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Check filename patterns
        if 'panduan' in filename_lower or 'manual' in filename_lower:
            return 'manual'
        elif 'laporan' in filename_lower or 'keuangan' in filename_lower:
            return 'financial_report'
        elif 'rumus' in filename_lower or 'formula' in filename_lower:
            return 'formula'
        elif 'regulasi' in filename_lower or 'peraturan' in filename_lower:
            return 'regulation'
        
        # Check content patterns
        if any(word in content_lower for word in ['rumus', 'formula', 'perhitungan']):
            return 'formula'
        elif any(word in content_lower for word in ['laporan keuangan', 'neraca', 'laba rugi']):
            return 'financial_report'
        elif any(word in content_lower for word in ['panduan', 'prosedur', 'langkah']):
            return 'manual'
        
        return 'general'
    
    def extract_tables_from_markdown(self, content: str) -> List[Dict[str, Any]]:
        """Extract tables from markdown content using regex patterns.
        
        Args:
            content: Markdown content containing tables
            
        Returns:
            List of dictionaries representing extracted tables with headers and rows
            
        Note:
            Parses markdown table format with pipe separators
        """
        tables = []
        table_pattern = r'\|(.+)\|\n\|(.+)\|\n(\|(.+)\|\n)+'
        
        matches = re.finditer(table_pattern, content, re.MULTILINE)
        
        for match in matches:
            table_text = match.group(0)
            lines = table_text.strip().split('\n')
            
            if len(lines) >= 3:  # Header + separator + at least one row
                headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]
                rows = []
                
                for line in lines[2:]:  # Skip header and separator
                    if '|' in line:
                        row = [cell.strip() for cell in line.split('|')[1:-1]]
                        if len(row) == len(headers):
                            rows.append(dict(zip(headers, row)))
                
                if rows:
                    tables.append({
                        'headers': headers,
                        'rows': rows,
                        'raw_text': table_text
                    })
        
        return tables
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if a file can be processed by the document processor.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid and can be processed, False otherwise
            
        Note:
            Checks file existence, markdown extension, and readability
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return False
            
            if not file_path.endswith('.md'):
                logger.error(f"File is not a markdown file: {file_path}")
                return False
            
            # Check if file is readable
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if len(content.strip()) == 0:
                    logger.error(f"File is empty: {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {str(e)}")
            return False
    
    def _is_psak219_document(self, filename: str, content: str, document_type: str) -> bool:
        """
        Detect if a document is a PSAK219 actuarial report based on filename and content.
        
        Args:
            filename (str): The document filename
            content (str): The document content
            document_type (str): The detected document type
            
        Returns:
            bool: True if this is likely a PSAK219 document
        """
        try:
            # Check filename patterns
            filename_lower = filename.lower()
            psak219_filename_patterns = [
                'psak219', 'psak_219', 'psak 219',
                'actuarial', 'aktuaria', 'aktuaris',
                'employee_benefit', 'employee benefit',
                'imbalan_kerja', 'imbalan kerja',
                'pension', 'pensiun'
            ]
            
            filename_match = any(pattern in filename_lower for pattern in psak219_filename_patterns)
            
            # Check content patterns (Indonesian and English)
            content_lower = content.lower()
            psak219_content_patterns = [
                'psak 219', 'psak219', 'psak_219',
                'imbalan kerja', 'employee benefit',
                'aktuaria', 'actuarial',
                'tingkat diskonto', 'discount rate',
                'kenaikan gaji', 'salary increase',
                'usia pensiun', 'retirement age',
                'kewajiban imbalan', 'benefit obligation',
                'biaya jasa', 'service cost',
                'analisis sensitivitas', 'sensitivity analysis'
            ]
            
            content_match_count = sum(1 for pattern in psak219_content_patterns if pattern in content_lower)
            
            # Decision logic: filename match OR multiple content matches
            is_psak219 = filename_match or content_match_count >= 3
            
            if is_psak219:
                logger.info(f"PSAK219 document detected - Filename match: {filename_match}, Content matches: {content_match_count}")
            
            return is_psak219
            
        except Exception as e:
            logger.error(f"Error detecting PSAK219 document: {str(e)}")
            return False