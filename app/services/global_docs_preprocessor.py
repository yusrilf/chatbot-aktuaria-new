"""Global Documents Preprocessing Service.

This module provides advanced preprocessing capabilities for global documentation
including content validation, metadata extraction, structure analysis, and
optimization for vector storage and retrieval.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import re
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

# Import the new semantic chunking components
from .semantic_document_chunker import SemanticDocumentChunker, DocumentChunk
from .chunk_metadata_extractor import ChunkMetadataExtractor

logger = logging.getLogger(__name__)


class ChunkStrategy(Enum):
    """Available chunking strategies for document processing."""
    ADAPTIVE = "adaptive"
    FIXED = "fixed"
    SEMANTIC = "semantic"
    SECTION_BASED = "section_based"


class ContentQuality(Enum):
    """Content quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class DocumentSection:
    """Represents a document section with metadata."""
    level: int
    title: str
    content: str
    line_start: int
    line_end: int
    word_count: int
    char_count: int
    subsections: List['DocumentSection']


@dataclass
class PreprocessingResult:
    """Result of document preprocessing operation."""
    success: bool
    processed_content: str
    metadata: Dict[str, Any]
    sections: List[DocumentSection]
    chunks: List[DocumentChunk]  # Added chunks to the result
    quality_score: float
    error_message: Optional[str] = None


class GlobalDocsPreprocessor:
    """Advanced preprocessor for global documentation files.
    
    Provides comprehensive preprocessing capabilities including:
    - Content validation and sanitization
    - Structure analysis and extraction
    - Metadata enrichment
    - Quality assessment
    - Semantic chunking with metadata
    - Optimization for vector storage
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the preprocessor with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Default configuration
        self.default_config = {
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'min_content_length': 50,
            'max_section_depth': 6,
            'preserve_code_blocks': True,
            'normalize_whitespace': True,
            'extract_links': True,
            'quality_threshold': 0.6,
            'chunk_size': 1000,
            'chunk_overlap': 100,
            'min_chunk_size': 200
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Initialize chunking components
        self.semantic_chunker = SemanticDocumentChunker(config={
            'max_chunk_size': self.config['chunk_size'],
            'overlap_size': self.config['chunk_overlap'],
            'min_chunk_size': self.config['min_chunk_size']
        })
        self.metadata_extractor = ChunkMetadataExtractor()
        
        self.logger.info(f"GlobalDocsPreprocessor initialized with config: {self.config}")
    
    def preprocess_document(
        self, 
        content: str, 
        filename: str,
        chunk_strategy: ChunkStrategy = ChunkStrategy.SEMANTIC,
        max_chunk_size: int = 1000
    ) -> PreprocessingResult:
        """Preprocess a global document with comprehensive analysis and semantic chunking.
        
        Args:
            content: Raw document content
            filename: Original filename
            chunk_strategy: Strategy for content chunking
            max_chunk_size: Maximum size for chunks
            
        Returns:
            PreprocessingResult with processed content, metadata, and chunks
        """
        try:
            self.logger.info(f"Starting preprocessing for {filename}")
            
            # Validate input
            validation_result = self._validate_content(content, filename)
            if not validation_result['valid']:
                return PreprocessingResult(
                    success=False,
                    processed_content="",
                    metadata={},
                    sections=[],
                    chunks=[],
                    quality_score=0.0,
                    error_message=validation_result['error']
                )
            
            # Initialize metadata
            metadata = self._initialize_metadata(filename, chunk_strategy, max_chunk_size)
            
            # Content normalization
            normalized_content = self._normalize_content(content)
            
            # Structure analysis
            sections = self._analyze_structure(normalized_content)
            
            # Content enhancement
            enhanced_content = self._enhance_content(normalized_content, sections)
            
            # Quality assessment
            quality_score = self._assess_quality(enhanced_content, sections)
            
            # Semantic chunking with metadata extraction
            chunks = []
            if chunk_strategy == ChunkStrategy.SEMANTIC:
                chunks = self.semantic_chunker.chunk_document(enhanced_content, filename)
                
                # Enhance chunks with additional metadata
                for chunk in chunks:
                    enhanced_metadata = self.metadata_extractor.extract_metadata(
                        content=chunk.content,
                        chunk_id=chunk.chunk_id,
                        doc_name=chunk.doc_name,
                        section_info={
                            'heading': chunk.section_heading,
                            'order': chunk.section_order,
                            'depth': chunk.heading_level
                        },
                        front_matter=chunk.front_matter or {}
                    )
                    # Store enhanced metadata in chunk for later use
                    chunk.enhanced_metadata = enhanced_metadata
                    
                self.logger.info(f"Created {len(chunks)} semantic chunks for {filename}")
            
            # Update metadata with analysis results
            metadata.update(self._generate_metadata(
                content, enhanced_content, sections, quality_score
            ))
            
            # Add chunking information to metadata
            metadata['chunking'] = {
                'strategy': chunk_strategy.value,
                'total_chunks': len(chunks),
                'chunk_size': max_chunk_size,
                'overlap': self.config['chunk_overlap']
            }
            
            self.logger.info(f"Preprocessing completed for {filename}: quality={quality_score:.2f}, chunks={len(chunks)}")
            
            return PreprocessingResult(
                success=True,
                processed_content=enhanced_content,
                metadata=metadata,
                sections=sections,
                chunks=chunks,
                quality_score=quality_score
            )
            
        except Exception as e:
            error_msg = f"Error preprocessing {filename}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return PreprocessingResult(
                success=False,
                processed_content="",
                metadata={},
                sections=[],
                chunks=[],
                quality_score=0.0,
                error_message=error_msg
            )
    
    def _validate_content(self, content: str, filename: str) -> Dict[str, Any]:
        """Validate document content for processing.
        
        Args:
            content: Document content to validate
            filename: Original filename
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Check content length
            if len(content) < self.config['min_content_length']:
                return {
                    'valid': False,
                    'error': f'Content too short: {len(content)} chars (min: {self.config["min_content_length"]})'
                }
            
            # Check file size
            if len(content.encode('utf-8')) > self.config['max_file_size']:
                return {
                    'valid': False,
                    'error': f'File too large: {len(content.encode("utf-8"))} bytes (max: {self.config["max_file_size"]})'
                }
            
            # Check for valid text content
            try:
                content.encode('utf-8')
            except UnicodeEncodeError as e:
                return {
                    'valid': False,
                    'error': f'Invalid text encoding: {str(e)}'
                }
            
            # Check for minimum structure (at least some text)
            if not content.strip():
                return {
                    'valid': False,
                    'error': 'Content is empty or contains only whitespace'
                }
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    def _initialize_metadata(
        self, 
        filename: str, 
        chunk_strategy: ChunkStrategy, 
        max_chunk_size: int
    ) -> Dict[str, Any]:
        """Initialize metadata structure for the document.
        
        Args:
            filename: Original filename
            chunk_strategy: Chunking strategy
            max_chunk_size: Maximum chunk size
            
        Returns:
            Initial metadata dictionary
        """
        return {
            'original_filename': filename,
            'processed_at': datetime.now().isoformat(),
            'preprocessing_version': '1.0.0',
            'processor_config': self.config.copy(),
            'chunking_config': {
                'strategy': chunk_strategy.value,
                'max_chunk_size': max_chunk_size
            },
            'content_hash': '',
            'processing_stats': {},
            'quality_metrics': {},
            'structure_analysis': {},
            'enhancement_applied': []
        }
    
    def _normalize_content(self, content: str) -> str:
        """Normalize document content for consistent processing.
        
        Args:
            content: Raw content to normalize
            
        Returns:
            Normalized content string
        """
        try:
            normalized = content
            
            # 1. Normalize line endings
            normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
            
            # 2. Handle whitespace normalization
            if self.config['normalize_whitespace']:
                # Remove excessive blank lines (max 2 consecutive)
                normalized = re.sub(r'\n\s*\n\s*\n+', '\n\n', normalized)
                # Normalize spaces and tabs within lines
                normalized = re.sub(r'[ \t]+', ' ', normalized)
                # Remove trailing whitespace from lines
                normalized = '\n'.join(line.rstrip() for line in normalized.split('\n'))
            
            # 3. Preserve code blocks if configured
            if self.config['preserve_code_blocks']:
                # Mark code blocks for preservation during further processing
                normalized = self._preserve_code_blocks(normalized)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing content: {str(e)}")
            return content  # Return original on error
    
    def _preserve_code_blocks(self, content: str) -> str:
        """Preserve code blocks during content processing.
        
        Args:
            content: Content with potential code blocks
            
        Returns:
            Content with preserved code blocks
        """
        # Find and preserve markdown code blocks
        code_block_pattern = r'```[\s\S]*?```'
        inline_code_pattern = r'`[^`\n]+`'
        
        # Replace with placeholders and store original blocks
        preserved_blocks = []
        
        def replace_code_block(match):
            block_id = f"__CODE_BLOCK_{len(preserved_blocks)}__"
            preserved_blocks.append(match.group(0))
            return block_id
        
        # Preserve code blocks
        content = re.sub(code_block_pattern, replace_code_block, content)
        content = re.sub(inline_code_pattern, replace_code_block, content)
        
        # Store preserved blocks in content for later restoration
        if preserved_blocks:
            content += f"\n\n<!-- PRESERVED_BLOCKS: {json.dumps(preserved_blocks)} -->"
        
        return content
    
    def _analyze_structure(self, content: str) -> List[DocumentSection]:
        """Analyze document structure and extract sections.
        
        Args:
            content: Normalized content to analyze
            
        Returns:
            List of DocumentSection objects
        """
        try:
            sections = []
            lines = content.split('\n')
            current_section = None
            line_number = 0
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                # Detect markdown headers
                if line_stripped.startswith('#'):
                    # Close previous section
                    if current_section:
                        current_section.line_end = i - 1
                        current_section.char_count = len(current_section.content)
                        current_section.word_count = len(current_section.content.split())
                        sections.append(current_section)
                    
                    # Start new section
                    level = len(line_stripped) - len(line_stripped.lstrip('#'))
                    title = line_stripped.lstrip('#').strip()
                    
                    current_section = DocumentSection(
                        level=level,
                        title=title,
                        content="",
                        line_start=i,
                        line_end=i,
                        word_count=0,
                        char_count=0,
                        subsections=[]
                    )
                
                elif current_section:
                    # Add content to current section
                    if line_stripped:  # Non-empty line
                        current_section.content += line + '\n'
            
            # Close final section
            if current_section:
                current_section.line_end = len(lines) - 1
                current_section.char_count = len(current_section.content)
                current_section.word_count = len(current_section.content.split())
                sections.append(current_section)
            
            # Build hierarchical structure
            sections = self._build_section_hierarchy(sections)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error analyzing structure: {str(e)}")
            return []
    
    def _build_section_hierarchy(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Build hierarchical structure from flat section list.
        
        Args:
            sections: Flat list of sections
            
        Returns:
            Hierarchically organized sections
        """
        if not sections:
            return []
        
        root_sections = []
        section_stack = []
        
        for section in sections:
            # Find parent level
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()
            
            if section_stack:
                # Add as subsection
                section_stack[-1].subsections.append(section)
            else:
                # Add as root section
                root_sections.append(section)
            
            section_stack.append(section)
        
        return root_sections
    
    def _enhance_content(self, content: str, sections: List[DocumentSection]) -> str:
        """Enhance content for better processing and retrieval.
        
        Args:
            content: Normalized content
            sections: Analyzed document sections
            
        Returns:
            Enhanced content string
        """
        try:
            enhanced = content
            
            # 1. Add section metadata as comments
            if sections:
                section_metadata = []
                for section in sections:
                    section_metadata.append(f"Section: {section.title} (Level {section.level})")
                
                metadata_comment = f"<!-- Document Structure: {'; '.join(section_metadata)} -->\n\n"
                enhanced = metadata_comment + enhanced
            
            # 2. Extract and preserve links if configured
            if self.config['extract_links']:
                enhanced = self._enhance_links(enhanced)
            
            # 3. Add content markers for better chunking
            enhanced = self._add_content_markers(enhanced, sections)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing content: {str(e)}")
            return content
    
    def _enhance_links(self, content: str) -> str:
        """Extract and enhance links in the content.
        
        Args:
            content: Content to process
            
        Returns:
            Content with enhanced links
        """
        # Find markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links_found = re.findall(link_pattern, content)
        
        if links_found:
            # Add links index at the end
            links_index = "\n\n<!-- Links Index:\n"
            for i, (text, url) in enumerate(links_found):
                links_index += f"{i+1}. {text}: {url}\n"
            links_index += "-->\n"
            
            content += links_index
        
        return content
    
    def _add_content_markers(self, content: str, sections: List[DocumentSection]) -> str:
        """Add content markers to improve chunking and retrieval.
        
        Args:
            content: Content to enhance
            sections: Document sections
            
        Returns:
            Content with added markers
        """
        # Add section boundaries for better chunking
        enhanced_lines = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            enhanced_lines.append(line)
            
            # Add section boundary markers
            if line.strip().startswith('#'):
                level = len(line.strip()) - len(line.strip().lstrip('#'))
                enhanced_lines.append(f"<!-- SECTION_BOUNDARY_LEVEL_{level} -->")
        
        return '\n'.join(enhanced_lines)
    
    def _assess_quality(self, content: str, sections: List[DocumentSection]) -> float:
        """Assess the quality of the processed content.
        
        Args:
            content: Processed content
            sections: Document sections
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            score = 0.0
            max_score = 0.0
            
            # 1. Structure quality (30%)
            structure_weight = 0.3
            max_score += structure_weight
            
            if sections:
                # Has structure
                score += structure_weight * 0.5
                
                # Good section distribution
                avg_section_length = sum(s.word_count for s in sections) / len(sections)
                if 50 <= avg_section_length <= 500:  # Reasonable section size
                    score += structure_weight * 0.3
                
                # Hierarchical structure
                has_hierarchy = any(s.subsections for s in sections)
                if has_hierarchy:
                    score += structure_weight * 0.2
            
            # 2. Content quality (40%)
            content_weight = 0.4
            max_score += content_weight
            
            # Content length
            word_count = len(content.split())
            if word_count >= 100:
                score += content_weight * 0.3
            
            # Content diversity (different types of content)
            has_lists = bool(re.search(r'^\s*[-*+]\s', content, re.MULTILINE))
            has_code = bool(re.search(r'```|`[^`]+`', content))
            has_links = bool(re.search(r'\[([^\]]+)\]\(([^)]+)\)', content))
            
            diversity_score = sum([has_lists, has_code, has_links]) / 3
            score += content_weight * 0.4 * diversity_score
            
            # Readability (simple heuristic)
            sentences = re.split(r'[.!?]+', content)
            if sentences:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                if 10 <= avg_sentence_length <= 25:  # Reasonable sentence length
                    score += content_weight * 0.3
            
            # 3. Technical quality (30%)
            technical_weight = 0.3
            max_score += technical_weight
            
            # No encoding issues (already validated)
            score += technical_weight * 0.4
            
            # Proper formatting
            has_proper_headers = bool(re.search(r'^#+\s+\w+', content, re.MULTILINE))
            if has_proper_headers:
                score += technical_weight * 0.3
            
            # Consistent formatting
            inconsistent_spacing = len(re.findall(r'\n\s*\n\s*\n\s*\n', content))
            if inconsistent_spacing < 3:  # Few formatting inconsistencies
                score += technical_weight * 0.3
            
            # Normalize score
            final_score = score / max_score if max_score > 0 else 0.0
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error assessing quality: {str(e)}")
            return 0.5  # Default moderate quality
    
    def _generate_metadata(
        self, 
        original_content: str, 
        processed_content: str, 
        sections: List[DocumentSection], 
        quality_score: float
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for the processed document.
        
        Args:
            original_content: Original document content
            processed_content: Processed document content
            sections: Analyzed sections
            quality_score: Calculated quality score
            
        Returns:
            Comprehensive metadata dictionary
        """
        try:
            # Content statistics
            original_stats = self._calculate_content_stats(original_content)
            processed_stats = self._calculate_content_stats(processed_content)
            
            # Section analysis
            section_analysis = {
                'total_sections': len(sections),
                'max_depth': max((s.level for s in sections), default=0),
                'sections_with_subsections': sum(1 for s in sections if s.subsections),
                'average_section_length': sum(s.word_count for s in sections) / len(sections) if sections else 0,
                'section_titles': [s.title for s in sections[:10]]  # First 10 titles
            }
            
            # Quality assessment
            quality_level = ContentQuality.EXCELLENT if quality_score >= 0.8 else \
                           ContentQuality.GOOD if quality_score >= 0.6 else \
                           ContentQuality.FAIR if quality_score >= 0.4 else \
                           ContentQuality.POOR
            
            return {
                'content_hash': hashlib.md5(processed_content.encode()).hexdigest(),
                'processing_stats': {
                    'original_stats': original_stats,
                    'processed_stats': processed_stats,
                    'compression_ratio': processed_stats['char_count'] / original_stats['char_count'] if original_stats['char_count'] > 0 else 1.0
                },
                'quality_metrics': {
                    'quality_score': quality_score,
                    'quality_level': quality_level.value,
                    'assessment_criteria': {
                        'structure_quality': 'good' if len(sections) > 0 else 'poor',
                        'content_diversity': 'good' if quality_score > 0.6 else 'fair',
                        'technical_quality': 'good' if quality_score > 0.5 else 'fair'
                    }
                },
                'structure_analysis': section_analysis,
                'enhancement_applied': [
                    'content_normalization',
                    'structure_analysis',
                    'quality_assessment',
                    'metadata_extraction'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating metadata: {str(e)}")
            return {'error': f'Metadata generation failed: {str(e)}'}
    
    def _calculate_content_stats(self, content: str) -> Dict[str, Any]:
        """Calculate basic statistics for content.
        
        Args:
            content: Content to analyze
            
        Returns:
            Dictionary with content statistics
        """
        lines = content.split('\n')
        words = content.split()
        
        return {
            'char_count': len(content),
            'word_count': len(words),
            'line_count': len(lines),
            'paragraph_count': len([line for line in lines if line.strip()]),
            'avg_words_per_line': len(words) / len(lines) if lines else 0,
            'avg_chars_per_word': len(content) / len(words) if words else 0
        }