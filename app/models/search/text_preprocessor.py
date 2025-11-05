"""Text Preprocessor for Indonesian Text with Actuarial Domain Optimization.

This module provides text preprocessing functionality specifically optimized
for Indonesian actuarial documents and queries.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

import re
import logging
from typing import List, Set

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocessor untuk teks Indonesia dengan optimasi untuk domain aktuaria."""
    
    def __init__(self):
        """Initialize the text preprocessor with Indonesian stopwords and patterns."""
        # Stopwords bahasa Indonesia yang umum
        self.stopwords: Set[str] = {
            'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'untuk', 'dengan', 'dalam', 'adalah',
            'akan', 'telah', 'sudah', 'dapat', 'bisa', 'harus', 'juga', 'atau', 'serta',
            'ini', 'itu', 'tersebut', 'sebagai', 'oleh', 'karena', 'jika', 'apabila',
            'bahwa', 'agar', 'supaya', 'maka', 'sehingga', 'namun', 'tetapi', 'namun'
        }
        
        # Pattern untuk normalisasi teks
        self.patterns = {
            'numbers': re.compile(r'\d+[.,]?\d*'),  # Angka dengan koma/titik
            'punctuation': re.compile(r'[^\w\s]'),  # Tanda baca
            'whitespace': re.compile(r'\s+'),       # Multiple whitespace
            'currency': re.compile(r'rp\.?\s*\d+', re.IGNORECASE),  # Mata uang
        }
        
        logger.debug("TextPreprocessor initialized with Indonesian stopwords")
    
    def preprocess(self, text: str, preserve_numbers: bool = True) -> str:
        """
        Preprocess text for search optimization.
        
        Args:
            text: Input text to preprocess
            preserve_numbers: Whether to preserve numerical values
            
        Returns:
            Preprocessed text string
        """
        if not text:
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Preserve currency and important numbers if needed
            if preserve_numbers:
                # Replace currency with normalized form
                text = self.patterns['currency'].sub('rupiah', text)
            else:
                # Remove all numbers
                text = self.patterns['numbers'].sub('', text)
            
            # Remove punctuation but preserve spaces
            text = self.patterns['punctuation'].sub(' ', text)
            
            # Normalize whitespace
            text = self.patterns['whitespace'].sub(' ', text)
            
            # Remove extra spaces
            text = text.strip()
            
            logger.debug(f"Preprocessed text: {text[:100]}...")
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text to tokenize
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            List of tokens
        """
        try:
            # Preprocess first
            processed_text = self.preprocess(text)
            
            # Split into tokens
            tokens = processed_text.split()
            
            # Remove stopwords if requested
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.stopwords]
            
            # Filter out very short tokens
            tokens = [token for token in tokens if len(token) > 2]
            
            logger.debug(f"Tokenized into {len(tokens)} tokens")
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Input text to extract keywords from
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        try:
            # Tokenize without removing stopwords first
            all_tokens = self.tokenize(text, remove_stopwords=False)
            
            # Remove stopwords manually to get better control
            keywords = [token for token in all_tokens if token not in self.stopwords]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for keyword in keywords:
                if keyword not in seen:
                    seen.add(keyword)
                    unique_keywords.append(keyword)
            
            # Return top keywords
            result = unique_keywords[:max_keywords]
            logger.debug(f"Extracted {len(result)} keywords: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def add_stopwords(self, words: List[str]) -> None:
        """
        Add custom stopwords to the existing set.
        
        Args:
            words: List of words to add as stopwords
        """
        self.stopwords.update(words)
        logger.info(f"Added {len(words)} custom stopwords")
    
    def remove_stopwords(self, words: List[str]) -> None:
        """
        Remove words from the stopwords set.
        
        Args:
            words: List of words to remove from stopwords
        """
        for word in words:
            self.stopwords.discard(word)
        logger.info(f"Removed {len(words)} words from stopwords")