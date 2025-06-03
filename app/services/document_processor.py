import os
import markdown
from typing import List, Dict, Any
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
import re

from app.config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_markdown_file(self, file_path: str, session_id: str) -> List[Document]:
        """Process a single markdown file into documents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract metadata from filename and content
            filename = os.path.basename(file_path)
            doc_type = self._extract_document_type(filename, content)
            
            # Split by markdown headers first
            header_splits = self.markdown_splitter.split_text(content)
            
            # Further split large chunks
            documents = []
            for i, split in enumerate(header_splits):
                # Create metadata
                metadata = {
                    'source': file_path,
                    'filename': filename,
                    'doc_type': doc_type,
                    'session_id': session_id  # BARU: Simpan session_id
                }
                
                # Add header context to metadata
                if hasattr(split, 'metadata'):
                    metadata.update(split.metadata)
                
                # Split further if chunk is too large
                if len(split.page_content) > config.CHUNK_SIZE:
                    sub_chunks = self.text_splitter.split_text(split.page_content)
                    for j, sub_chunk in enumerate(sub_chunks):
                        doc_metadata = metadata.copy()
                        doc_metadata['sub_chunk_id'] = j
                        documents.append(Document(
                            page_content=sub_chunk,
                            metadata=doc_metadata
                        ))
                else:
                    documents.append(Document(
                        page_content=split.page_content,
                        metadata=metadata
                    ))
            
            logger.info(f"Processed {filename} into {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def process_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """Process multiple markdown files"""
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
        """Extract document type from filename and content"""
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
        """Extract tables from markdown content"""
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
        """Validate if file can be processed"""
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