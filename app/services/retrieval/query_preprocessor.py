"""Query Preprocessor for CoT Retrieval Orchestration.

This module handles query normalization and entity extraction for improved
document retrieval in the Chain of Thought system.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import unicodedata

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    """Container for extracted entities from query."""
    psak_references: List[str]
    puc_references: List[str]
    numbers: List[float]
    years: List[int]
    technical_terms: List[str]
    normalized_query: str
    original_query: str


class QueryPreprocessor:
    """Preprocesses queries for better document retrieval."""
    
    def __init__(self):
        """Initialize the query preprocessor."""
        self.psak_patterns = [
            r'PSAK\s*(\d+)',
            r'psak\s*(\d+)',
            r'Pernyataan\s*Standar\s*Akuntansi\s*Keuangan\s*(\d+)',
            r'standar\s*akuntansi\s*(\d+)'
        ]
        
        self.puc_patterns = [
            r'PUC',
            r'puc',
            r'Projected\s*Unit\s*Credit',
            r'projected\s*unit\s*credit',
            r'metode\s*puc',
            r'metode\s*projected\s*unit\s*credit'
        ]
        
        self.technical_terms = [
            'aktuaria', 'actuarial', 'imbalan', 'kerja', 'pensiun', 'pension',
            'liabilitas', 'liability', 'aset', 'asset', 'asumsi', 'assumption',
            'diskonto', 'discount', 'mortalitas', 'mortality', 'morbiditas',
            'turnover', 'salary', 'gaji', 'kenaikan', 'inflasi', 'inflation',
            'present', 'value', 'nilai', 'sekarang', 'future', 'masa', 'depan',
            'benefit', 'manfaat', 'service', 'layanan', 'cost', 'biaya',
            'gain', 'loss', 'keuntungan', 'kerugian', 'remeasurement',
            'pengukuran', 'kembali', 'corridor', 'koridor'
        ]
        
        logger.info("QueryPreprocessor initialized")
    
    def preprocess_query(self, query: str) -> ExtractedEntities:
        """
        Preprocess query with normalization and entity extraction.
        
        Args:
            query: Raw user query
            
        Returns:
            ExtractedEntities object with all extracted information
        """
        try:
            logger.info(f"Preprocessing query: {query[:100]}...")
            
            # Step 1: Normalize query
            normalized = self._normalize_text(query)
            
            # Step 2: Extract entities
            psak_refs = self._extract_psak_references(query)
            puc_refs = self._extract_puc_references(query)
            numbers = self._extract_numbers(query)
            years = self._extract_years(query)
            tech_terms = self._extract_technical_terms(normalized)
            
            entities = ExtractedEntities(
                psak_references=psak_refs,
                puc_references=puc_refs,
                numbers=numbers,
                years=years,
                technical_terms=tech_terms,
                normalized_query=normalized,
                original_query=query
            )
            
            logger.info(f"Extracted entities: PSAK={len(psak_refs)}, "
                       f"PUC={len(puc_refs)}, numbers={len(numbers)}, "
                       f"years={len(years)}, tech_terms={len(tech_terms)}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error preprocessing query: {str(e)}")
            # Return minimal entities on error
            return ExtractedEntities(
                psak_references=[],
                puc_references=[],
                numbers=[],
                years=[],
                technical_terms=[],
                normalized_query=query,
                original_query=query
            )
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better processing.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        try:
            # Unicode normalization
            text = unicodedata.normalize('NFKD', text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Normalize common variations
            text = re.sub(r'psak\s*219', 'psak 219', text)
            text = re.sub(r'imbalan\s*kerja', 'imbalan kerja', text)
            text = re.sub(r'projected\s*unit\s*credit', 'projected unit credit', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error normalizing text: {str(e)}")
            return text
    
    def _extract_psak_references(self, text: str) -> List[str]:
        """Extract PSAK references from text."""
        try:
            psak_refs = []
            
            for pattern in self.psak_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        psak_num = match.group(1)
                        psak_ref = f"PSAK {psak_num}"
                        if psak_ref not in psak_refs:
                            psak_refs.append(psak_ref)
                    else:
                        # Pattern without capture group
                        psak_ref = match.group(0).upper()
                        if psak_ref not in psak_refs:
                            psak_refs.append(psak_ref)
            
            return psak_refs
            
        except Exception as e:
            logger.error(f"Error extracting PSAK references: {str(e)}")
            return []
    
    def _extract_puc_references(self, text: str) -> List[str]:
        """Extract PUC references from text."""
        try:
            puc_refs = []
            
            for pattern in self.puc_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if "PUC" not in puc_refs:
                        puc_refs.append("PUC")
                    if "Projected Unit Credit" not in puc_refs:
                        puc_refs.append("Projected Unit Credit")
                    break
            
            return puc_refs
            
        except Exception as e:
            logger.error(f"Error extracting PUC references: {str(e)}")
            return []
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text."""
        try:
            numbers = []
            
            # Pattern for numbers (including decimals and percentages)
            number_pattern = r'\b\d+(?:\.\d+)?(?:%|\s*persen)?\b'
            matches = re.finditer(number_pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    num_str = match.group(0)
                    # Remove percentage signs
                    num_str = re.sub(r'%|\s*persen', '', num_str)
                    num = float(num_str)
                    if num not in numbers:
                        numbers.append(num)
                except ValueError:
                    continue
            
            return sorted(numbers)
            
        except Exception as e:
            logger.error(f"Error extracting numbers: {str(e)}")
            return []
    
    def _extract_years(self, text: str) -> List[int]:
        """Extract years from text."""
        try:
            years = []
            
            # Pattern for years (1900-2100)
            year_pattern = r'\b(19\d{2}|20\d{2}|21\d{2})\b'
            matches = re.finditer(year_pattern, text)
            
            for match in matches:
                year = int(match.group(1))
                if year not in years:
                    years.append(year)
            
            return sorted(years)
            
        except Exception as e:
            logger.error(f"Error extracting years: {str(e)}")
            return []
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from normalized text."""
        try:
            found_terms = []
            
            for term in self.technical_terms:
                if term.lower() in text.lower():
                    if term not in found_terms:
                        found_terms.append(term)
            
            return found_terms
            
        except Exception as e:
            logger.error(f"Error extracting technical terms: {str(e)}")
            return []
    
    def create_search_keywords(self, entities: ExtractedEntities) -> List[str]:
        """
        Create search keywords from extracted entities.
        
        Args:
            entities: Extracted entities
            
        Returns:
            List of search keywords
        """
        try:
            keywords = []
            
            # Add PSAK references
            keywords.extend(entities.psak_references)
            
            # Add PUC references
            keywords.extend(entities.puc_references)
            
            # Add technical terms
            keywords.extend(entities.technical_terms)
            
            # Add years as strings
            keywords.extend([str(year) for year in entities.years])
            
            # Add normalized query words
            query_words = entities.normalized_query.split()
            keywords.extend([word for word in query_words if len(word) > 3])
            
            # Remove duplicates and return
            return list(set(keywords))
            
        except Exception as e:
            logger.error(f"Error creating search keywords: {str(e)}")
            return [entities.normalized_query]