#!/usr/bin/env python3
"""
Adaptive Chunking Strategy for Optimal Document Processing

This module provides intelligent chunking strategies that adapt to document type,
content complexity, and processing requirements to ensure optimal chunk sizes
for both storage efficiency and retrieval accuracy.

Features:
- Dynamic chunk size calculation based on document characteristics
- Content-aware chunking for different document types (PSAK219, general docs)
- Semantic boundary preservation
- Overlap optimization for context preservation
- Performance monitoring and adjustment

Author: AI Assistant
Date: 2025-01-27
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
from pathlib import Path

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document

from app.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Document type enumeration for chunking strategy selection"""
    PSAK219 = "psak219"
    GENERAL_MARKDOWN = "general_markdown"
    TECHNICAL_DOCUMENT = "technical_document"
    NARRATIVE_TEXT = "narrative_text"
    STRUCTURED_DATA = "structured_data"

@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy"""
    base_chunk_size: int
    max_chunk_size: int
    min_chunk_size: int
    overlap_ratio: float
    preserve_headers: bool
    semantic_boundaries: bool
    adaptive_sizing: bool
    
class ContentAnalyzer:
    """Analyzes document content to determine optimal chunking strategy"""
    
    def __init__(self):
        self.logger = logger
        
    def analyze_document(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Analyze document content to determine characteristics for chunking.
        
        Args:
            content: Document content as string
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document analysis results
        """
        try:
            analysis = {
                'document_type': self._detect_document_type(content, file_path),
                'content_length': len(content),
                'line_count': len(content.split('\n')),
                'avg_line_length': self._calculate_avg_line_length(content),
                'header_count': self._count_headers(content),
                'table_count': self._count_tables(content),
                'list_count': self._count_lists(content),
                'complexity_score': self._calculate_complexity_score(content),
                'has_structured_data': self._has_structured_data(content),
                'language_density': self._analyze_language_density(content)
            }
            
            self.logger.debug(f"Document analysis completed for {file_path}: {analysis}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing document {file_path}: {str(e)}")
            return self._get_default_analysis()
    
    def _detect_document_type(self, content: str, file_path: str) -> DocumentType:
        """Detect document type based on content and filename"""
        filename = os.path.basename(file_path).lower()
        
        # Check for PSAK219 indicators
        psak219_indicators = [
            'psak219', 'psak 219', 'actuarial', 'aktuaria',
            'employee benefit', 'imbalan kerja', 'valuasi'
        ]
        
        if any(indicator in filename for indicator in psak219_indicators):
            return DocumentType.PSAK219
        
        # Check content patterns
        if re.search(r'#+\s*(informasi\s+umum|employee\s+data|actuarial\s+assumptions)', content, re.IGNORECASE):
            return DocumentType.PSAK219
        
        # Check for technical document patterns
        technical_patterns = [
            r'```\w+',  # Code blocks
            r'\|.*\|.*\|',  # Tables
            r'\d+\.\s+\w+',  # Numbered lists
        ]
        
        if any(re.search(pattern, content) for pattern in technical_patterns):
            return DocumentType.TECHNICAL_DOCUMENT
        
        # Check for structured data
        if self._has_structured_data(content):
            return DocumentType.STRUCTURED_DATA
        
        return DocumentType.GENERAL_MARKDOWN
    
    def _calculate_avg_line_length(self, content: str) -> float:
        """Calculate average line length"""
        lines = [line for line in content.split('\n') if line.strip()]
        if not lines:
            return 0.0
        return statistics.mean(len(line) for line in lines)
    
    def _count_headers(self, content: str) -> int:
        """Count markdown headers"""
        return len(re.findall(r'^#+\s+', content, re.MULTILINE))
    
    def _count_tables(self, content: str) -> int:
        """Count markdown tables"""
        return len(re.findall(r'\|.*\|.*\|', content))
    
    def _count_lists(self, content: str) -> int:
        """Count lists (both ordered and unordered)"""
        ordered_lists = len(re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE))
        unordered_lists = len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE))
        return ordered_lists + unordered_lists
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate content complexity score (0-1)"""
        factors = {
            'avg_sentence_length': min(1.0, len(content.split('.')) / max(1, len(content.split()))),
            'special_chars_ratio': len(re.findall(r'[^\w\s]', content)) / max(1, len(content)),
            'number_density': len(re.findall(r'\d+', content)) / max(1, len(content.split())),
            'header_density': self._count_headers(content) / max(1, len(content.split('\n')))
        }
        
        return min(1.0, sum(factors.values()) / len(factors))
    
    def _has_structured_data(self, content: str) -> bool:
        """Check if content has structured data patterns"""
        patterns = [
            r'\{[^}]*\}',  # JSON-like structures
            r'\w+:\s*\w+',  # Key-value pairs
            r'\|.*\|.*\|',  # Tables
            r'^\s*-\s+\w+:',  # YAML-like lists
        ]
        
        return any(re.search(pattern, content, re.MULTILINE) for pattern in patterns)
    
    def _analyze_language_density(self, content: str) -> Dict[str, float]:
        """Analyze language characteristics for density calculation"""
        words = content.split()
        if not words:
            return {'word_density': 0.0, 'char_per_word': 0.0}
        
        return {
            'word_density': len(words) / max(1, len(content)),
            'char_per_word': len(content) / len(words)
        }
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when analysis fails"""
        return {
            'document_type': DocumentType.GENERAL_MARKDOWN,
            'content_length': 0,
            'line_count': 0,
            'avg_line_length': 0.0,
            'header_count': 0,
            'table_count': 0,
            'list_count': 0,
            'complexity_score': 0.5,
            'has_structured_data': False,
            'language_density': {'word_density': 0.0, 'char_per_word': 0.0}
        }

class AdaptiveChunkingStrategy:
    """Adaptive chunking strategy that optimizes chunk sizes based on content analysis"""
    
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        self.logger = logger
        
        # Define chunking configurations for different document types
        self.chunking_configs = {
            DocumentType.PSAK219: ChunkingConfig(
                base_chunk_size=1500,  # Larger chunks for structured actuarial data
                max_chunk_size=2500,
                min_chunk_size=800,
                overlap_ratio=0.15,  # 15% overlap for context preservation
                preserve_headers=True,
                semantic_boundaries=True,
                adaptive_sizing=True
            ),
            DocumentType.TECHNICAL_DOCUMENT: ChunkingConfig(
                base_chunk_size=1200,
                max_chunk_size=2000,
                min_chunk_size=600,
                overlap_ratio=0.12,
                preserve_headers=True,
                semantic_boundaries=True,
                adaptive_sizing=True
            ),
            DocumentType.STRUCTURED_DATA: ChunkingConfig(
                base_chunk_size=1000,
                max_chunk_size=1800,
                min_chunk_size=500,
                overlap_ratio=0.10,
                preserve_headers=True,
                semantic_boundaries=False,  # Preserve data structure
                adaptive_sizing=True
            ),
            DocumentType.GENERAL_MARKDOWN: ChunkingConfig(
                base_chunk_size=1000,
                max_chunk_size=1600,
                min_chunk_size=400,
                overlap_ratio=0.15,
                preserve_headers=True,
                semantic_boundaries=True,
                adaptive_sizing=True
            ),
            DocumentType.NARRATIVE_TEXT: ChunkingConfig(
                base_chunk_size=800,
                max_chunk_size=1400,
                min_chunk_size=300,
                overlap_ratio=0.20,  # Higher overlap for narrative flow
                preserve_headers=False,
                semantic_boundaries=True,
                adaptive_sizing=True
            )
        }
    
    def get_optimal_chunking_config(self, content: str, file_path: str) -> ChunkingConfig:
        """
        Determine optimal chunking configuration based on content analysis.
        
        Args:
            content: Document content
            file_path: Path to the document
            
        Returns:
            Optimal chunking configuration
        """
        try:
            analysis = self.content_analyzer.analyze_document(content, file_path)
            doc_type = analysis['document_type']
            
            # Get base configuration
            base_config = self.chunking_configs.get(doc_type, self.chunking_configs[DocumentType.GENERAL_MARKDOWN])
            
            if not base_config.adaptive_sizing:
                return base_config
            
            # Adapt configuration based on analysis
            adapted_config = self._adapt_config_to_content(base_config, analysis)
            
            self.logger.info(
                f"Optimal chunking config for {file_path}: "
                f"type={doc_type.value}, chunk_size={adapted_config.base_chunk_size}, "
                f"overlap={adapted_config.overlap_ratio:.2f}"
            )
            
            return adapted_config
            
        except Exception as e:
            self.logger.error(f"Error determining chunking config for {file_path}: {str(e)}")
            return self.chunking_configs[DocumentType.GENERAL_MARKDOWN]
    
    def _adapt_config_to_content(self, base_config: ChunkingConfig, analysis: Dict[str, Any]) -> ChunkingConfig:
        """
        Adapt base configuration based on content analysis.
        
        Args:
            base_config: Base chunking configuration
            analysis: Content analysis results
            
        Returns:
            Adapted chunking configuration
        """
        # Create a copy of base config
        adapted_config = ChunkingConfig(
            base_chunk_size=base_config.base_chunk_size,
            max_chunk_size=base_config.max_chunk_size,
            min_chunk_size=base_config.min_chunk_size,
            overlap_ratio=base_config.overlap_ratio,
            preserve_headers=base_config.preserve_headers,
            semantic_boundaries=base_config.semantic_boundaries,
            adaptive_sizing=base_config.adaptive_sizing
        )
        
        # Adjust based on content length
        content_length = analysis['content_length']
        if content_length > 50000:  # Very large documents
            adapted_config.base_chunk_size = min(
                adapted_config.max_chunk_size,
                int(adapted_config.base_chunk_size * 1.3)
            )
        elif content_length < 5000:  # Small documents
            adapted_config.base_chunk_size = max(
                adapted_config.min_chunk_size,
                int(adapted_config.base_chunk_size * 0.7)
            )
        
        # Adjust based on complexity
        complexity = analysis['complexity_score']
        if complexity > 0.7:  # High complexity
            adapted_config.overlap_ratio = min(0.25, adapted_config.overlap_ratio * 1.2)
        elif complexity < 0.3:  # Low complexity
            adapted_config.overlap_ratio = max(0.05, adapted_config.overlap_ratio * 0.8)
        
        # Adjust based on structure
        if analysis['has_structured_data']:
            adapted_config.semantic_boundaries = False
            adapted_config.overlap_ratio = max(0.05, adapted_config.overlap_ratio * 0.8)
        
        # Adjust based on header density
        header_density = analysis['header_count'] / max(1, analysis['line_count'])
        if header_density > 0.1:  # Many headers
            adapted_config.preserve_headers = True
            adapted_config.base_chunk_size = max(
                adapted_config.min_chunk_size,
                int(adapted_config.base_chunk_size * 0.9)
            )
        
        return adapted_config
    
    def create_optimized_splitters(self, config: ChunkingConfig) -> Tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
        """
        Create optimized text splitters based on chunking configuration.
        
        Args:
            config: Chunking configuration
            
        Returns:
            Tuple of (header_splitter, text_splitter)
        """
        try:
            # Create header splitter if headers should be preserved
            header_splitter = None
            if config.preserve_headers:
                header_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3"),
                        ("####", "Header 4"),
                    ]
                )
            
            # Calculate overlap size
            overlap_size = int(config.base_chunk_size * config.overlap_ratio)
            
            # Choose separators based on semantic boundaries preference
            if config.semantic_boundaries:
                separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
            else:
                separators = ["\n\n", "\n", " "]
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.base_chunk_size,
                chunk_overlap=overlap_size,
                length_function=len,
                separators=separators,
                keep_separator=True
            )
            
            self.logger.debug(
                f"Created splitters: chunk_size={config.base_chunk_size}, "
                f"overlap={overlap_size}, preserve_headers={config.preserve_headers}"
            )
            
            return header_splitter, text_splitter
            
        except Exception as e:
            self.logger.error(f"Error creating splitters: {str(e)}")
            # Fallback to default splitters
            return self._create_default_splitters()
    
    def _create_default_splitters(self) -> Tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
        """Create default splitters as fallback"""
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        return header_splitter, text_splitter
    
    def validate_chunks(self, chunks: List[Document], config: ChunkingConfig) -> Dict[str, Any]:
        """
        Validate generated chunks against configuration requirements.
        
        Args:
            chunks: List of generated document chunks
            config: Chunking configuration used
            
        Returns:
            Validation results and statistics
        """
        try:
            if not chunks:
                return {
                    'valid': False,
                    'error': 'No chunks generated',
                    'statistics': {}
                }
            
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            
            statistics_data = {
                'total_chunks': len(chunks),
                'avg_chunk_size': statistics.mean(chunk_sizes),
                'min_chunk_size': min(chunk_sizes),
                'max_chunk_size': max(chunk_sizes),
                'median_chunk_size': statistics.median(chunk_sizes),
                'chunks_within_range': sum(
                    1 for size in chunk_sizes 
                    if config.min_chunk_size <= size <= config.max_chunk_size
                ),
                'size_distribution': {
                    'small': sum(1 for size in chunk_sizes if size < config.min_chunk_size),
                    'optimal': sum(1 for size in chunk_sizes if config.min_chunk_size <= size <= config.max_chunk_size),
                    'large': sum(1 for size in chunk_sizes if size > config.max_chunk_size)
                }
            }
            
            # Calculate validation score
            optimal_ratio = statistics_data['chunks_within_range'] / len(chunks)
            valid = optimal_ratio >= 0.7  # At least 70% of chunks should be within optimal range
            
            return {
                'valid': valid,
                'optimal_ratio': optimal_ratio,
                'statistics': statistics_data
            }
            
        except Exception as e:
            self.logger.error(f"Error validating chunks: {str(e)}")
            return {
                'valid': False,
                'error': str(e),
                'statistics': {}
            }

# Global instance for easy access
adaptive_chunking_strategy = AdaptiveChunkingStrategy()