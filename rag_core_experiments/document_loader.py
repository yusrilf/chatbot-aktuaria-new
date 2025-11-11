"""Document Loading - Core functionality for loading markdown documents

This module provides simplified document loading capabilities.
"""

import os
import re
import yaml
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """Represents a loaded document."""
    content: str
    metadata: Dict[str, Any]
    filename: str
    front_matter: Dict[str, Any]


class DocumentLoader:
    """Simple document loader for markdown files."""

    def __init__(self):
        """Initialize document loader."""
        logger.info("DocumentLoader initialized")

    def load_document(self, file_path: str) -> LoadedDocument:
        """Load a single markdown document.

        Args:
            file_path: Path to the markdown file

        Returns:
            LoadedDocument object
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # Extract YAML front-matter
            front_matter, content = self._extract_front_matter(raw_content)

            # Extract filename
            filename = os.path.basename(file_path)

            # Create metadata
            metadata = {
                'filename': filename,
                'source': file_path,
                'doc_type': front_matter.get('document_type', 'general'),
                'domain': front_matter.get('domain', 'general'),
            }

            logger.info(f"Loaded document: {filename} ({len(content)} chars)")

            return LoadedDocument(
                content=content,
                metadata=metadata,
                filename=filename,
                front_matter=front_matter
            )

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise

    def load_documents(self, file_paths: List[str]) -> List[LoadedDocument]:
        """Load multiple documents.

        Args:
            file_paths: List of paths to markdown files

        Returns:
            List of LoadedDocument objects
        """
        documents = []
        for path in file_paths:
            try:
                doc = self.load_document(path)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def _extract_front_matter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML front-matter from markdown.

        Args:
            content: Raw markdown content

        Returns:
            Tuple of (front_matter_dict, remaining_content)
        """
        try:
            pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL | re.MULTILINE)
            match = pattern.match(content)

            if match:
                yaml_content = match.group(1)
                remaining = content[match.end():]

                try:
                    front_matter = yaml.safe_load(yaml_content) or {}
                except yaml.YAMLError as e:
                    logger.warning(f"YAML parse error: {e}")
                    front_matter = {}

                return front_matter, remaining
            else:
                return {}, content

        except Exception as e:
            logger.warning(f"Error extracting front-matter: {e}")
            return {}, content
