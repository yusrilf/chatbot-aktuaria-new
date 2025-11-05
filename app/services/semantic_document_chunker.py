"""Semantic Document Chunker for Markdown with YAML Front-matter.

This module provides advanced semantic chunking capabilities for markdown documents
with YAML front-matter, respecting document structure and generating rich metadata
for optimal retrieval and embedding performance.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import re
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import json
import math

# Import enhanced metadata extractor
from .chunk_metadata_extractor import ChunkMetadataExtractor

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of document chunks."""
    FRONT_MATTER = "front_matter"
    HEADING = "heading"
    CONTENT = "content"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    LIST = "list"
    PARAGRAPH = "paragraph"


class DifficultyLevel(Enum):
    """Document difficulty levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    OVERVIEW = "overview"


@dataclass
class DocumentChunk:
    """Represents a semantic chunk of a document with rich metadata."""
    
    # Core content
    content: str
    chunk_id: str
    chunk_type: ChunkType
    
    # Document metadata
    doc_name: str
    section_heading: str
    section_order: int
    
    # Position metadata
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    
    # Content metrics
    token_count: int
    word_count: int
    char_count: int
    
    # Semantic metadata
    front_matter: Dict[str, Any]
    difficulty: Optional[str]
    estimated_reading: Optional[str]
    version: Optional[str]
    keywords: List[str]
    target_audience: List[str]
    
    # Structural metadata
    heading_level: int
    parent_section: Optional[str]
    subsections: List[str]
    
    # Processing metadata
    created_at: str
    chunk_hash: str
    overlap_with_previous: int
    overlap_with_next: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return asdict(self)
    
    def get_embedding_text(self) -> str:
        """Get text optimized for embedding (without metadata)."""
        return self.content.strip()
    
    def get_context_text(self) -> str:
        """Get text with contextual information for retrieval."""
        context_parts = []
        
        # Add section context
        if self.section_heading and self.section_heading != "Root":
            context_parts.append(f"Section: {self.section_heading}")
        
        # Add document context
        if self.doc_name:
            context_parts.append(f"Document: {self.doc_name}")
        
        # Add difficulty context
        if self.difficulty:
            context_parts.append(f"Difficulty: {self.difficulty}")
        
        # Combine with content
        context_prefix = " | ".join(context_parts)
        return f"{context_prefix}\n\n{self.content}" if context_parts else self.content


class SemanticDocumentChunker:
    """Advanced semantic chunker for markdown documents with YAML front-matter.
    
    Features:
    - YAML front-matter extraction and preservation
    - Heading-based semantic boundaries
    - Intelligent content splitting with overlap
    - Rich metadata generation
    - Token-aware chunking
    - Structure preservation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the semantic chunker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize metadata extractor
        self.metadata_extractor = ChunkMetadataExtractor()
        
        # Default configuration
        self.default_config = {
            'max_chunk_size': 1000,  # tokens
            'min_chunk_size': 100,   # tokens
            'overlap_size': 150,     # tokens
            'chars_per_token': 4,    # estimation ratio
            'preserve_code_blocks': True,
            'preserve_tables': True,
            'split_long_paragraphs': True,
            'include_front_matter_in_chunks': False,
            'max_heading_levels': 6,
            'min_section_size': 50   # tokens
        }
        
        # Merge configurations
        self.config = {**self.default_config, **self.config}
        
        # Compile regex patterns
        self._compile_patterns()
        
        self.logger.info(f"SemanticDocumentChunker initialized with config: {self.config}")
    
    def _compile_patterns(self):
        """Compile regex patterns for document parsing."""
        self.patterns = {
            'front_matter': re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL | re.MULTILINE),
            'heading': re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
            'inline_code': re.compile(r'`[^`\n]+`'),
            'table': re.compile(r'^\|.*\|$', re.MULTILINE),
            'list_item': re.compile(r'^\s*[-*+]\s+', re.MULTILINE),
            'numbered_list': re.compile(r'^\s*\d+\.\s+', re.MULTILINE),
            'paragraph_break': re.compile(r'\n\s*\n'),
            'sentence_end': re.compile(r'[.!?]+\s+')
        }
    
    def chunk_document(
        self, 
        content: str, 
        doc_name: str,
        max_chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Chunk a markdown document with semantic awareness.
        
        Args:
            content: Raw markdown content with YAML front-matter
            doc_name: Name of the document
            max_chunk_size: Maximum chunk size in tokens (overrides config)
            overlap_size: Overlap size in tokens (overrides config)
            
        Returns:
            List of DocumentChunk objects
        """
        try:
            self.logger.info(f"Starting semantic chunking for {doc_name}")
            
            # Use provided sizes or fall back to config
            max_size = max_chunk_size or self.config['max_chunk_size']
            overlap = overlap_size or self.config['overlap_size']
            
            # Parse document structure
            front_matter, main_content = self._extract_front_matter(content)
            sections = self._parse_document_structure(main_content)
            
            # Generate chunks
            chunks = []
            chunk_counter = 0
            
            for section_idx, section in enumerate(sections):
                section_chunks = self._chunk_section(
                    section=section,
                    doc_name=doc_name,
                    front_matter=front_matter,
                    section_order=section_idx,
                    max_chunk_size=max_size,
                    overlap_size=overlap,
                    chunk_counter_start=chunk_counter
                )
                
                chunks.extend(section_chunks)
                chunk_counter += len(section_chunks)
            
            # Add overlap information
            self._add_overlap_metadata(chunks, overlap)
            
            self.logger.info(f"Chunking completed for {doc_name}: {len(chunks)} chunks generated")
            
            return chunks
            
        except Exception as e:
            error_msg = f"Error chunking document {doc_name}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _extract_front_matter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML front-matter from document.
        
        Args:
            content: Raw document content
            
        Returns:
            Tuple of (front_matter_dict, remaining_content)
        """
        try:
            match = self.patterns['front_matter'].match(content)
            
            if match:
                yaml_content = match.group(1)
                remaining_content = content[match.end():]
                
                try:
                    front_matter = yaml.safe_load(yaml_content) or {}
                except yaml.YAMLError as e:
                    self.logger.warning(f"Failed to parse YAML front-matter: {e}")
                    front_matter = {'yaml_parse_error': str(e)}
                
                return front_matter, remaining_content
            else:
                return {}, content
                
        except Exception as e:
            self.logger.warning(f"Error extracting front-matter: {e}")
            return {}, content
    
    def _parse_document_structure(self, content: str) -> List[Dict[str, Any]]:
        """Parse document into structured sections based on headings.
        
        Args:
            content: Main document content (without front-matter)
            
        Returns:
            List of section dictionaries
        """
        try:
            sections = []
            lines = content.split('\n')
            current_section = None
            line_number = 0
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                # Check for heading
                heading_match = self.patterns['heading'].match(line)
                
                if heading_match:
                    # Save previous section
                    if current_section:
                        current_section['content'] = '\n'.join(current_section['content_lines'])
                        current_section['line_end'] = i - 1
                        current_section['char_end'] = sum(len(l) + 1 for l in lines[:i])
                        sections.append(current_section)
                    
                    # Start new section
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()
                    
                    current_section = {
                        'heading': title,
                        'level': level,
                        'line_start': i,
                        'line_end': i,
                        'char_start': sum(len(l) + 1 for l in lines[:i]),
                        'char_end': 0,
                        'content_lines': [],
                        'content': '',
                        'subsections': []
                    }
                
                elif current_section:
                    # Add content to current section
                    current_section['content_lines'].append(line)
                
                elif not current_section and line_stripped:
                    # Content before first heading - create root section
                    current_section = {
                        'heading': 'Root',
                        'level': 0,
                        'line_start': i,
                        'line_end': i,
                        'char_start': 0,
                        'char_end': 0,
                        'content_lines': [line],
                        'content': '',
                        'subsections': []
                    }
            
            # Save final section
            if current_section:
                current_section['content'] = '\n'.join(current_section['content_lines'])
                current_section['line_end'] = len(lines) - 1
                current_section['char_end'] = len(content)
                sections.append(current_section)
            
            # Build hierarchical structure
            sections = self._build_section_hierarchy(sections)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error parsing document structure: {e}")
            # Return single section with all content
            return [{
                'heading': 'Root',
                'level': 0,
                'line_start': 0,
                'line_end': len(content.split('\n')) - 1,
                'char_start': 0,
                'char_end': len(content),
                'content': content,
                'content_lines': content.split('\n'),
                'subsections': []
            }]
    
    def _build_section_hierarchy(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build hierarchical structure from flat section list.
        
        Args:
            sections: Flat list of sections
            
        Returns:
            Hierarchically organized sections
        """
        if not sections:
            return []
        
        # Track parent sections for hierarchy building
        for i, section in enumerate(sections):
            section['subsections'] = []
            
            # Find subsections (sections with higher level that come after this one)
            for j in range(i + 1, len(sections)):
                next_section = sections[j]
                
                # If next section has higher level, it's a subsection
                if next_section['level'] > section['level']:
                    section['subsections'].append(next_section['heading'])
                else:
                    # Stop when we reach same or lower level
                    break
        
        return sections
    
    def _chunk_section(
        self,
        section: Dict[str, Any],
        doc_name: str,
        front_matter: Dict[str, Any],
        section_order: int,
        max_chunk_size: int,
        overlap_size: int,
        chunk_counter_start: int
    ) -> List[DocumentChunk]:
        """Chunk a single section with semantic awareness.
        
        Args:
            section: Section dictionary
            doc_name: Document name
            front_matter: YAML front-matter
            section_order: Order of section in document
            max_chunk_size: Maximum chunk size in tokens
            overlap_size: Overlap size in tokens
            chunk_counter_start: Starting counter for chunk IDs
            
        Returns:
            List of chunks for this section
        """
        try:
            content = section['content'].strip()
            if not content:
                return []
            
            # Estimate token count
            estimated_tokens = self._estimate_tokens(content)
            
            # If section fits in one chunk, return as single chunk
            if estimated_tokens <= max_chunk_size:
                return [self._create_chunk(
                    content=content,
                    chunk_id=f"{doc_name}_chunk_{chunk_counter_start:03d}",
                    doc_name=doc_name,
                    section=section,
                    section_order=section_order,
                    front_matter=front_matter,
                    chunk_index=0,
                    total_chunks_in_section=1
                )]
            
            # Split large section into multiple chunks
            chunks = self._split_long_content(
                content=content,
                section=section,
                doc_name=doc_name,
                section_order=section_order,
                front_matter=front_matter,
                max_chunk_size=max_chunk_size,
                overlap_size=overlap_size,
                chunk_counter_start=chunk_counter_start
            )
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error chunking section {section.get('heading', 'Unknown')}: {e}")
            return []
    
    def _split_long_content(
        self,
        content: str,
        section: Dict[str, Any],
        doc_name: str,
        section_order: int,
        front_matter: Dict[str, Any],
        max_chunk_size: int,
        overlap_size: int,
        chunk_counter_start: int
    ) -> List[DocumentChunk]:
        """Split long content into multiple chunks with overlap.
        
        Args:
            content: Content to split
            section: Section metadata
            doc_name: Document name
            section_order: Section order
            front_matter: YAML front-matter
            max_chunk_size: Maximum chunk size
            overlap_size: Overlap size
            chunk_counter_start: Starting chunk counter
            
        Returns:
            List of chunks
        """
        try:
            chunks = []
            
            # Split by paragraphs first
            paragraphs = self.patterns['paragraph_break'].split(content)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            current_chunk_content = []
            current_chunk_tokens = 0
            chunk_index = 0
            
            for para_idx, paragraph in enumerate(paragraphs):
                para_tokens = self._estimate_tokens(paragraph)
                
                # If paragraph alone exceeds max size, split it further
                if para_tokens > max_chunk_size:
                    # Save current chunk if it has content
                    if current_chunk_content:
                        chunk_text = '\n\n'.join(current_chunk_content)
                        chunks.append(self._create_chunk(
                            content=chunk_text,
                            chunk_id=f"{doc_name}_chunk_{chunk_counter_start + chunk_index:03d}",
                            doc_name=doc_name,
                            section=section,
                            section_order=section_order,
                            front_matter=front_matter,
                            chunk_index=chunk_index,
                            total_chunks_in_section=-1  # Will be updated later
                        ))
                        chunk_index += 1
                        current_chunk_content = []
                        current_chunk_tokens = 0
                    
                    # Split large paragraph by sentences
                    para_chunks = self._split_paragraph_by_sentences(
                        paragraph, max_chunk_size, overlap_size
                    )
                    
                    for para_chunk in para_chunks:
                        chunks.append(self._create_chunk(
                            content=para_chunk,
                            chunk_id=f"{doc_name}_chunk_{chunk_counter_start + chunk_index:03d}",
                            doc_name=doc_name,
                            section=section,
                            section_order=section_order,
                            front_matter=front_matter,
                            chunk_index=chunk_index,
                            total_chunks_in_section=-1
                        ))
                        chunk_index += 1
                
                # If adding this paragraph would exceed max size, save current chunk
                elif current_chunk_tokens + para_tokens > max_chunk_size and current_chunk_content:
                    chunk_text = '\n\n'.join(current_chunk_content)
                    chunks.append(self._create_chunk(
                        content=chunk_text,
                        chunk_id=f"{doc_name}_chunk_{chunk_counter_start + chunk_index:03d}",
                        doc_name=doc_name,
                        section=section,
                        section_order=section_order,
                        front_matter=front_matter,
                        chunk_index=chunk_index,
                        total_chunks_in_section=-1
                    ))
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_content = self._get_overlap_content(current_chunk_content, overlap_size)
                    current_chunk_content = overlap_content + [paragraph]
                    current_chunk_tokens = sum(self._estimate_tokens(p) for p in current_chunk_content)
                
                else:
                    # Add paragraph to current chunk
                    current_chunk_content.append(paragraph)
                    current_chunk_tokens += para_tokens
            
            # Save final chunk
            if current_chunk_content:
                chunk_text = '\n\n'.join(current_chunk_content)
                chunks.append(self._create_chunk(
                    content=chunk_text,
                    chunk_id=f"{doc_name}_chunk_{chunk_counter_start + chunk_index:03d}",
                    doc_name=doc_name,
                    section=section,
                    section_order=section_order,
                    front_matter=front_matter,
                    chunk_index=chunk_index,
                    total_chunks_in_section=-1
                ))
            
            # Update total chunks count
            total_chunks = len(chunks)
            for chunk in chunks:
                # This is a bit hacky, but we need to update the metadata
                chunk.front_matter['total_chunks_in_section'] = total_chunks
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error splitting long content: {e}")
            return []
    
    def _split_paragraph_by_sentences(
        self, 
        paragraph: str, 
        max_chunk_size: int, 
        overlap_size: int
    ) -> List[str]:
        """Split a large paragraph by sentences.
        
        Args:
            paragraph: Paragraph to split
            max_chunk_size: Maximum chunk size in tokens
            overlap_size: Overlap size in tokens
            
        Returns:
            List of sentence-based chunks
        """
        try:
            sentences = self.patterns['sentence_end'].split(paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return [paragraph]
            
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self._estimate_tokens(sentence)
                
                if current_tokens + sentence_tokens > max_chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_sentences = self._get_sentence_overlap(current_chunk, overlap_size)
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(self._estimate_tokens(s) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            
            # Save final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks if chunks else [paragraph]
            
        except Exception as e:
            self.logger.error(f"Error splitting paragraph by sentences: {e}")
            return [paragraph]
    
    def _get_overlap_content(self, content_list: List[str], overlap_size: int) -> List[str]:
        """Get overlap content from the end of current chunk.
        
        Args:
            content_list: List of content pieces
            overlap_size: Desired overlap size in tokens
            
        Returns:
            List of content pieces for overlap
        """
        if not content_list:
            return []
        
        overlap_content = []
        overlap_tokens = 0
        
        # Work backwards from the end
        for content in reversed(content_list):
            content_tokens = self._estimate_tokens(content)
            
            if overlap_tokens + content_tokens <= overlap_size:
                overlap_content.insert(0, content)
                overlap_tokens += content_tokens
            else:
                break
        
        return overlap_content
    
    def _get_sentence_overlap(self, sentences: List[str], overlap_size: int) -> List[str]:
        """Get sentence-level overlap.
        
        Args:
            sentences: List of sentences
            overlap_size: Desired overlap size in tokens
            
        Returns:
            List of sentences for overlap
        """
        if not sentences:
            return []
        
        overlap_sentences = []
        overlap_tokens = 0
        
        # Work backwards from the end
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            
            if overlap_tokens + sentence_tokens <= overlap_size:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk(
        self,
        content: str,
        chunk_id: str,
        doc_name: str,
        section: Dict[str, Any],
        section_order: int,
        front_matter: Dict[str, Any],
        chunk_index: int,
        total_chunks_in_section: int
    ) -> DocumentChunk:
        """Create a DocumentChunk with rich metadata.
        
        Args:
            content: Chunk content
            chunk_id: Unique chunk identifier
            doc_name: Document name
            section: Section metadata
            section_order: Section order in document
            front_matter: YAML front-matter
            chunk_index: Index of chunk within section
            total_chunks_in_section: Total chunks in this section
            
        Returns:
            DocumentChunk object
        """
        try:
            # Calculate metrics
            word_count = len(content.split())
            char_count = len(content)
            token_count = self._estimate_tokens(content)
            
            # Extract enhanced metadata using ChunkMetadataExtractor
            enhanced_metadata = self.metadata_extractor.extract_metadata(
                content=content,
                chunk_id=chunk_id,
                doc_name=doc_name,
                section_info=section,
                front_matter=front_matter
            )
            
            # Extract metadata from front-matter (fallback for compatibility)
            difficulty = self._extract_difficulty(front_matter)
            estimated_reading = front_matter.get('estimated_reading')
            version = front_matter.get('version')
            keywords = front_matter.get('keywords', [])
            target_audience = front_matter.get('target_audience', [])
            
            # Ensure lists
            if isinstance(keywords, str):
                keywords = [keywords]
            if isinstance(target_audience, str):
                target_audience = [target_audience]
            
            # Create chunk hash
            chunk_hash = hashlib.md5(content.encode()).hexdigest()[:16]
            
            # Determine chunk type
            chunk_type = self._determine_chunk_type(content)
            
            # Create DocumentChunk with enhanced metadata
            chunk = DocumentChunk(
                content=content,
                chunk_id=chunk_id,
                chunk_type=chunk_type,
                doc_name=doc_name,
                section_heading=section['heading'],
                section_order=section_order,
                line_start=section['line_start'],
                line_end=section['line_end'],
                char_start=section['char_start'],
                char_end=section['char_end'],
                token_count=token_count,
                word_count=word_count,
                char_count=char_count,
                front_matter=front_matter,
                difficulty=difficulty,
                estimated_reading=estimated_reading,
                version=version,
                keywords=keywords,
                target_audience=target_audience,
                heading_level=section['level'],
                parent_section=None,  # Could be enhanced to track parent sections
                subsections=section.get('subsections', []),
                created_at=datetime.now().isoformat(),
                chunk_hash=chunk_hash,
                overlap_with_previous=0,  # Will be calculated later
                overlap_with_next=0       # Will be calculated later
            )
            
            # Attach enhanced metadata to chunk
            chunk.enhanced_metadata = enhanced_metadata
            
            return chunk
            
        except Exception as e:
            self.logger.error(f"Error creating chunk: {e}")
            raise
    
    def _extract_difficulty(self, front_matter: Dict[str, Any]) -> Optional[str]:
        """Extract difficulty level from front-matter.
        
        Args:
            front_matter: YAML front-matter dictionary
            
        Returns:
            Difficulty level string or None
        """
        difficulty = front_matter.get('difficulty')
        
        if isinstance(difficulty, list) and difficulty:
            return difficulty[0]  # Take first difficulty level
        elif isinstance(difficulty, str):
            return difficulty
        else:
            return None
    
    def _determine_chunk_type(self, content: str) -> ChunkType:
        """Determine the type of chunk based on content.
        
        Args:
            content: Chunk content
            
        Returns:
            ChunkType enum value
        """
        content_stripped = content.strip()
        
        # Check for code blocks
        if self.patterns['code_block'].search(content_stripped):
            return ChunkType.CODE_BLOCK
        
        # Check for tables
        if self.patterns['table'].search(content_stripped):
            return ChunkType.TABLE
        
        # Check for lists
        if (self.patterns['list_item'].search(content_stripped) or 
            self.patterns['numbered_list'].search(content_stripped)):
            return ChunkType.LIST
        
        # Check for headings
        if self.patterns['heading'].search(content_stripped):
            return ChunkType.HEADING
        
        # Default to content
        return ChunkType.CONTENT
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Simple estimation: chars / chars_per_token
        return max(1, math.ceil(len(text) / self.config['chars_per_token']))
    
    def _add_overlap_metadata(self, chunks: List[DocumentChunk], overlap_size: int):
        """Add overlap metadata to chunks.
        
        Args:
            chunks: List of chunks to update
            overlap_size: Configured overlap size
        """
        try:
            for i, chunk in enumerate(chunks):
                # Calculate overlap with previous chunk
                if i > 0:
                    prev_chunk = chunks[i - 1]
                    overlap_prev = self._calculate_overlap(prev_chunk.content, chunk.content)
                    chunk.overlap_with_previous = overlap_prev
                
                # Calculate overlap with next chunk
                if i < len(chunks) - 1:
                    next_chunk = chunks[i + 1]
                    overlap_next = self._calculate_overlap(chunk.content, next_chunk.content)
                    chunk.overlap_with_next = overlap_next
                    
        except Exception as e:
            self.logger.error(f"Error adding overlap metadata: {e}")
    
    def _calculate_overlap(self, text1: str, text2: str) -> int:
        """Calculate overlap between two text chunks.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Estimated overlap in tokens
        """
        try:
            # Simple approach: find common words at boundaries
            words1 = text1.split()
            words2 = text2.split()
            
            if not words1 or not words2:
                return 0
            
            # Check overlap at end of text1 and start of text2
            max_overlap = min(len(words1), len(words2), 50)  # Limit check to 50 words
            overlap_count = 0
            
            for i in range(1, max_overlap + 1):
                if words1[-i:] == words2[:i]:
                    overlap_count = i
            
            return overlap_count
            
        except Exception as e:
            self.logger.error(f"Error calculating overlap: {e}")
            return 0
    
    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Statistics dictionary
        """
        try:
            if not chunks:
                return {'error': 'No chunks provided'}
            
            # Basic stats
            total_chunks = len(chunks)
            total_tokens = sum(chunk.token_count for chunk in chunks)
            total_words = sum(chunk.word_count for chunk in chunks)
            total_chars = sum(chunk.char_count for chunk in chunks)
            
            # Size distribution
            token_counts = [chunk.token_count for chunk in chunks]
            avg_tokens = total_tokens / total_chunks
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            
            # Section distribution
            sections = {}
            for chunk in chunks:
                section = chunk.section_heading
                if section not in sections:
                    sections[section] = 0
                sections[section] += 1
            
            # Chunk types
            chunk_types = {}
            for chunk in chunks:
                chunk_type = chunk.chunk_type.value
                if chunk_type not in chunk_types:
                    chunk_types[chunk_type] = 0
                chunk_types[chunk_type] += 1
            
            # Difficulty distribution
            difficulties = {}
            for chunk in chunks:
                difficulty = chunk.difficulty or 'unknown'
                if difficulty not in difficulties:
                    difficulties[difficulty] = 0
                difficulties[difficulty] += 1
            
            return {
                'total_chunks': total_chunks,
                'total_tokens': total_tokens,
                'total_words': total_words,
                'total_chars': total_chars,
                'avg_tokens_per_chunk': round(avg_tokens, 2),
                'min_tokens_per_chunk': min_tokens,
                'max_tokens_per_chunk': max_tokens,
                'sections_distribution': sections,
                'chunk_types_distribution': chunk_types,
                'difficulty_distribution': difficulties,
                'documents_processed': len(set(chunk.doc_name for chunk in chunks))
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating chunking stats: {e}")
            return {'error': f'Failed to calculate stats: {str(e)}'}