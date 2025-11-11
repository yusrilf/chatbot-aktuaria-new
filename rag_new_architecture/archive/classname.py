from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from xml.dom.minidom import Document

@dataclass
class MarkdownTable:
    content: str  # The table content
    start_index: int  # Starting position in original text
    end_index: int  # Ending position in original text


@dataclass
class MarkdownCode:
    content: str  # The code content
    language: Optional[str]  # Programming language (if specified)
    start_index: int  # Starting position in original text
    end_index: int  # Ending position in original text


@dataclass
class MarkdownImage:
    alias: str  # Alt text or filename
    content: str  # Image path or data URL
    start_index: int  # Starting position in original text
    end_index: int  # Ending position in original text
    link: Optional[str]  # Link URL (if image is clickable)


@dataclass
class MarkdownDocument(Document):
    id: str  # Unique document ID
    content: str  # Full markdown content
    tables: List[MarkdownTable]  # Extracted tables
    code: List[MarkdownCode]  # Extracted code blocks
    images: List[MarkdownImage]  # Extracted images
    chunks: List[Chunk]  # type: ignore # Remaining text chunks
    metadata: Dict[str, Any]  # Additional metadata
