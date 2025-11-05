"""Preview Extractor Utility for Document Content.

This module provides utilities to extract meaningful preview content from documents,
filtering out markdown formatting, whitespace, and other non-informative characters.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_meaningful_preview(content: str, max_length: int = 100) -> str:
    """Extract meaningful preview from document content.
    
    Args:
        content: Raw document content
        max_length: Maximum length of preview (default: 100)
        
    Returns:
        Cleaned and meaningful preview text
        
    Note:
        Filters out markdown formatting, excessive whitespace, and non-informative content
    """
    try:
        if not content or not content.strip():
            return "[Konten kosong]"
        
        # Remove common markdown formatting
        cleaned = content
        
        # Remove markdown headers (# ## ###)
        cleaned = re.sub(r'^#{1,6}\s*', '', cleaned, flags=re.MULTILINE)
        
        # Remove markdown bold/italic (**text** *text*)
        cleaned = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', cleaned)
        
        # Remove remaining asterisks and markdown symbols
        cleaned = re.sub(r'\*+', '', cleaned)
        cleaned = re.sub(r'#+', '', cleaned)
        cleaned = re.sub(r'`+', '', cleaned)
        
        # Remove markdown links [text](url)
        cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)
        
        # Remove code blocks and inline code
        cleaned = re.sub(r'```[^`]*```', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
        
        # Remove table separators and formatting
        cleaned = re.sub(r'\|[-\s]*\|', '', cleaned)
        cleaned = re.sub(r'^\s*\|', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\|\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove excessive whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Skip if content is too short or only contains formatting characters
        if len(cleaned) < 3 or cleaned.strip() in ['|', '**', '```', '#', '-', '=', '1', '2', '3', '4', '5']:
            # Try to find meaningful content in the original by looking for words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
            if len(words) >= 2:
                # Create meaningful preview from words found
                meaningful_text = ' '.join(words[:8])  # Take first 8 words
                return meaningful_text[:max_length] + ("..." if len(meaningful_text) > max_length else "")
            
            # Try to find Indonesian/English text patterns
            text_matches = re.findall(r'[a-zA-Z][a-zA-Z\s]{5,}', content)
            if text_matches:
                best_match = max(text_matches, key=len)
                return best_match.strip()[:max_length] + ("..." if len(best_match.strip()) > max_length else "")
            
            # Last resort: look for any alphanumeric content
            alphanumeric = re.sub(r'[^a-zA-Z0-9\s]', ' ', content)
            alphanumeric = re.sub(r'\s+', ' ', alphanumeric).strip()
            if len(alphanumeric) > 5:
                return alphanumeric[:max_length] + ("..." if len(alphanumeric) > max_length else "")
            
            return "[Dokumen kosong atau tidak terbaca]"
        
        # Extract 2-3 meaningful sentences for better context
        sentences = re.split(r'[.!?]', cleaned)
        meaningful_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Meaningful sentence
                # Add sentence if it fits within max_length
                if current_length + len(sentence) + 2 <= max_length:  # +2 for punctuation and space
                    meaningful_sentences.append(sentence)
                    current_length += len(sentence) + 2
                    # Stop after 3 sentences or if we're close to max_length
                    if len(meaningful_sentences) >= 3 or current_length > max_length * 0.8:
                        break
                else:
                    # If adding full sentence exceeds limit, truncate it
                    remaining_space = max_length - current_length - 3  # -3 for "..."
                    if remaining_space > 20:  # Only add if there's meaningful space
                        meaningful_sentences.append(sentence[:remaining_space])
                    break
        
        if meaningful_sentences:
            result = ". ".join(meaningful_sentences)
            if not result.endswith(('.', '!', '?')):
                result += "..."
            else:
                result += "."
            return result
        
        # If no meaningful sentences found, return cleaned content
        if len(cleaned) <= max_length:
            return cleaned
        else:
            return cleaned[:max_length] + "..."
            
    except Exception as e:
        logger.error(f"Error extracting preview: {str(e)}")
        return "[Error mengekstrak preview]"

def extract_key_phrases(content: str, max_phrases: int = 3) -> list:
    """Extract key phrases from document content.
    
    Args:
        content: Raw document content
        max_phrases: Maximum number of key phrases to extract
        
    Returns:
        List of key phrases found in the content
        
    Note:
        Identifies important terms and phrases relevant to actuarial content
    """
    try:
        if not content:
            return []
        
        # Common actuarial and financial terms
        key_terms = [
            'psak 219', 'psak219', 'employee benefit', 'imbalan kerja',
            'discount rate', 'tingkat diskonto', 'salary increase', 'kenaikan gaji',
            'mortality table', 'tabel mortalitas', 'retirement age', 'usia pensiun',
            'present value', 'nilai sekarang', 'actuarial', 'aktuaria',
            'liability', 'liabilitas', 'obligation', 'kewajiban',
            'service cost', 'biaya jasa', 'interest cost', 'biaya bunga'
        ]
        
        content_lower = content.lower()
        found_phrases = []
        
        for term in key_terms:
            if term in content_lower and term not in found_phrases:
                found_phrases.append(term)
                if len(found_phrases) >= max_phrases:
                    break
        
        return found_phrases
        
    except Exception as e:
        logger.error(f"Error extracting key phrases: {str(e)}")
        return []

def get_document_summary_info(content: str, metadata: dict) -> str:
    """Get summary information about document content.
    
    Args:
        content: Document content
        metadata: Document metadata
        
    Returns:
        Summary information string
        
    Note:
        Combines meaningful preview with key metadata information
    """
    try:
        preview = extract_meaningful_preview(content, max_length=80)
        key_phrases = extract_key_phrases(content, max_phrases=2)
        
        summary_parts = [preview]
        
        if key_phrases:
            summary_parts.append(f"[Topik: {', '.join(key_phrases)}]")
        
        # Add document type if available
        doc_type = metadata.get('doc_type', '')
        if doc_type and doc_type != 'general':
            summary_parts.append(f"[Tipe: {doc_type}]")
        
        return " ".join(summary_parts)
        
    except Exception as e:
        logger.error(f"Error getting document summary: {str(e)}")
        return extract_meaningful_preview(content, max_length=100)