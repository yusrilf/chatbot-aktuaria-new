"""Document Chunking - Core functionality for semantic chunking

This module provides simplified semantic chunking capabilities.
"""

import re
import hashlib
import math
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    content: str
    chunk_id: str
    metadata: Dict[str, Any]
    token_count: int
    chunk_index: int


class SemanticChunker:
    """Semantic document chunker with overlap support."""

    def __init__(
        self,
        max_chunk_size: int = None,
        overlap_size: int = None,
        chars_per_token: int = None
    ):
        """Initialize semantic chunker.

        Args:
            max_chunk_size: Maximum chunk size in tokens
            overlap_size: Overlap size in tokens
            chars_per_token: Characters per token ratio
        """
        self.max_chunk_size = max_chunk_size or config.CHUNK_SIZE
        self.overlap_size = overlap_size or config.CHUNK_OVERLAP
        self.chars_per_token = chars_per_token or config.CHARS_PER_TOKEN

        # Compile regex patterns
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.sentence_pattern = re.compile(r'[.!?]+\s+')

        logger.info(
            f"SemanticChunker initialized: "
            f"max_chunk_size={self.max_chunk_size}, "
            f"overlap={self.overlap_size}"
        )

    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_name: str
    ) -> List[DocumentChunk]:
        """Chunk a document with semantic awareness.

        Args:
            content: Document content
            metadata: Document metadata
            doc_name: Document name

        Returns:
            List of DocumentChunk objects
        """
        try:
            logger.info(f"Chunking document: {doc_name}")

            # Split by paragraphs
            paragraphs = self.paragraph_pattern.split(content)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            chunks = []
            current_chunk = []
            current_tokens = 0

            for para in paragraphs:
                para_tokens = self._estimate_tokens(para)

                # If paragraph alone exceeds max size, split by sentences
                if para_tokens > self.max_chunk_size:
                    # Save current chunk if it has content
                    if current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunks.append(self._create_chunk(
                            chunk_text, metadata, doc_name, len(chunks)
                        ))
                        current_chunk = []
                        current_tokens = 0

                    # Split large paragraph by sentences
                    sent_chunks = self._split_by_sentences(para)
                    for sent_chunk in sent_chunks:
                        chunks.append(self._create_chunk(
                            sent_chunk, metadata, doc_name, len(chunks)
                        ))

                # If adding paragraph exceeds max size, save current chunk
                elif current_tokens + para_tokens > self.max_chunk_size and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text, metadata, doc_name, len(chunks)
                    ))

                    # Start new chunk with overlap
                    overlap_content = self._get_overlap(current_chunk)
                    current_chunk = overlap_content + [para]
                    current_tokens = sum(self._estimate_tokens(p) for p in current_chunk)

                else:
                    # Add paragraph to current chunk
                    current_chunk.append(para)
                    current_tokens += para_tokens

            # Save final chunk
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text, metadata, doc_name, len(chunks)
                ))

            logger.info(f"Created {len(chunks)} chunks for {doc_name}")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise

    def _create_chunk(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_name: str,
        chunk_index: int
    ) -> DocumentChunk:
        """Create a document chunk with metadata.

        Args:
            content: Chunk content
            metadata: Document metadata
            doc_name: Document name
            chunk_index: Chunk index

        Returns:
            DocumentChunk object
        """
        # Generate unique chunk ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        chunk_id = f"{doc_name}_{chunk_index:03d}_{content_hash}"

        # Calculate token count
        token_count = self._estimate_tokens(content)

        # Combine metadata
        chunk_metadata = {
            **metadata,
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'chunk_size': token_count,
            'created_at': datetime.now().isoformat(),
        }

        return DocumentChunk(
            content=content,
            chunk_id=chunk_id,
            metadata=chunk_metadata,
            token_count=token_count,
            chunk_index=chunk_index
        )

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences when it's too large.

        Args:
            text: Text to split

        Returns:
            List of sentence-based chunks
        """
        sentences = self.sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)

            if current_tokens + sent_tokens > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                overlap = self._get_sentence_overlap(current_chunk)
                current_chunk = overlap + [sent]
                current_tokens = sum(self._estimate_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _get_overlap(self, paragraphs: List[str]) -> List[str]:
        """Get overlap paragraphs from end of current chunk.

        Args:
            paragraphs: List of paragraphs

        Returns:
            List of paragraphs for overlap
        """
        overlap = []
        overlap_tokens = 0

        for para in reversed(paragraphs):
            para_tokens = self._estimate_tokens(para)

            if overlap_tokens + para_tokens <= self.overlap_size:
                overlap.insert(0, para)
                overlap_tokens += para_tokens
            else:
                break

        return overlap

    def _get_sentence_overlap(self, sentences: List[str]) -> List[str]:
        """Get overlap sentences from end of current chunk.

        Args:
            sentences: List of sentences

        Returns:
            List of sentences for overlap
        """
        overlap = []
        overlap_tokens = 0

        for sent in reversed(sentences):
            sent_tokens = self._estimate_tokens(sent)

            if overlap_tokens + sent_tokens <= self.overlap_size:
                overlap.insert(0, sent)
                overlap_tokens += sent_tokens
            else:
                break

        return overlap

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        return max(1, math.ceil(len(text) / self.chars_per_token))
