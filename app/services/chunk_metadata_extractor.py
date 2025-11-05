"""Chunk Metadata Extractor for Enhanced Document Understanding.

This module provides advanced metadata extraction capabilities for document chunks,
enriching them with semantic, structural, and contextual information to improve
retrieval accuracy and context understanding.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import re
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import json
import math
from collections import Counter

logger = logging.getLogger(__name__)


class MetadataType(Enum):
    """Types of metadata that can be extracted."""
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    CONTEXTUAL = "contextual"
    QUANTITATIVE = "quantitative"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"


class ContentComplexity(Enum):
    """Content complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class ChunkMetadata:
    """Simplified and structured metadata for document chunks optimized for chatbot retrieval."""
    
    # Core structured fields for chatbot (as requested by user)
    doc_name: str
    doc_type: str = "teknis"  # Default for actuarial documents
    domain: str = "aktuaria"  # Default domain
    scope: str = "sains_aktuaria_imbalan_kerja"  # Default scope
    section_heading: str = ""  # Section title
    section_order: int = 0  # Section order (integer for simplicity)
    keywords: List[str] = None  # Keywords from frontmatter + extracted
    difficulty: str = "moderate"  # Difficulty level
    last_updated: str = ""  # Last update date
    version: str = "1.0"  # Document version
    related_regulations: List[str] = None  # Related regulations
    
    # Optional structured numerical data for financial queries
    numbers: Dict[str, Union[float, int]] = None  # Structured numerical data
    
    # Internal processing metadata (minimal, for system use)
    chunk_id: str = ""  # Internal chunk identifier
    extraction_timestamp: str = ""  # When metadata was extracted
    confidence_score: float = 0.0  # Extraction confidence
    
    def __post_init__(self):
        """Initialize default values for list and dict fields."""
        if self.keywords is None:
            self.keywords = []
        if self.related_regulations is None:
            self.related_regulations = []
        if self.numbers is None:
            self.numbers = {}
        
        # Set extraction timestamp if not provided
        if not self.extraction_timestamp:
            self.extraction_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding internal fields for clean output."""
        result = {
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "domain": self.domain,
            "scope": self.scope,
            "section_heading": self.section_heading,
            "section_order": self.section_order,
            "keywords": self.keywords,
            "difficulty": self.difficulty,
            "last_updated": self.last_updated,
            "version": self.version,
            "related_regulations": self.related_regulations,
        }
        
        # Only include numbers if it has content
        if self.numbers:
            result["numbers"] = self.numbers
            
        return result


class MetadataType(Enum):
    """Types of metadata that can be extracted."""
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    CONTEXTUAL = "contextual"
    QUANTITATIVE = "quantitative"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"


class ContentComplexity(Enum):
    """Content complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"



class ChunkMetadataExtractor:
    """Advanced metadata extractor for document chunks.
    
    Features:
    - Structural analysis (headings, hierarchy, position)
    - Semantic analysis (concepts, entities, topics)
    - Content complexity assessment
    - Regulatory compliance detection
    - Technical term extraction
    - Cross-reference analysis
    - Quality scoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the metadata extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Default configuration
        self.default_config = {
            'extract_named_entities': True,
            'calculate_readability': True,
            'analyze_complexity': True,
            'detect_regulatory_content': True,
            'extract_technical_terms': True,
            'analyze_cross_references': True,
            'min_concept_frequency': 2,
            'max_concepts_per_chunk': 10,
            'confidence_threshold': 0.7
        }
        
        # Merge configurations
        self.config = {**self.default_config, **self.config}
        
        # Initialize patterns and dictionaries
        self._initialize_patterns()
        self._initialize_domain_knowledge()
        
        self.logger.info(f"ChunkMetadataExtractor initialized with config: {self.config}")
    
    def _initialize_patterns(self):
        """Initialize regex patterns for content analysis."""
        self.patterns = {
            # Mathematical and formula patterns
            'formula': re.compile(r'[=+\-*/^()]\s*[\d\w\s+\-*/^()]+|∑|∏|∫|√|\$.*?\$', re.IGNORECASE),
            'percentage': re.compile(r'\d+(?:\.\d+)?%'),
            'currency': re.compile(r'Rp\.?\s*[\d,.]+ |USD\s*[\d,.]+|\$[\d,.]+'),
            
            # Numerical patterns for extraction
            'discount_rate': re.compile(r'(?:discount|diskonto).*?(\d+(?:\.\d+)?)%', re.IGNORECASE),
            'mortality_rate': re.compile(r'(?:mortality|mortalitas).*?(\d+(?:\.\d+)?)', re.IGNORECASE),
            'salary_increase': re.compile(r'(?:salary.*?increase|kenaikan.*?gaji).*?(\d+(?:\.\d+)?)%', re.IGNORECASE),
            'year': re.compile(r'\b(20\d{2})\b'),
            'age': re.compile(r'(?:umur|age).*?(\d+)', re.IGNORECASE),
            
            # Regulatory patterns
            'psak_reference': re.compile(r'PSAK\s*\d+|PSAK\s*No\.?\s*\d+', re.IGNORECASE),
            'regulation': re.compile(r'Peraturan|Undang-Undang|UU\s*No\.?\s*\d+|PP\s*No\.?\s*\d+', re.IGNORECASE),
            'compliance': re.compile(r'wajib|harus|diharuskan|kewajiban|compliance|mandatory', re.IGNORECASE),
            
            # Technical patterns
            'acronym': re.compile(r'\b[A-Z]{2,}\b'),
            'technical_term': re.compile(r'aktuaria|actuarial|liabilitas|asuransi|premi|klaim|reserv|diskonto', re.IGNORECASE),
            'calculation': re.compile(r'hitung|kalkulasi|rumus|formula|perhitungan|compute|calculate', re.IGNORECASE),
            
            # Structural patterns
            'cross_reference': re.compile(r'lihat\s+(?:bagian|bab|pasal|ayat)\s*\d+|see\s+(?:section|chapter)\s*\d+', re.IGNORECASE),
            'list_item': re.compile(r'^\s*[-*+]\s+|^\s*\d+\.\s+', re.MULTILINE),
            'table_marker': re.compile(r'^\|.*\|$', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]*?```|`[^`\n]+`'),
            
            # Content quality patterns
            'incomplete': re.compile(r'TODO|FIXME|TBD|to be determined|\.\.\.|…', re.IGNORECASE),
            'emphasis': re.compile(r'\*\*.*?\*\*|__.*?__|_.*?_|\*.*?\*'),
            'question': re.compile(r'\?'),
            'exclamation': re.compile(r'!')
        }
    
    def _initialize_domain_knowledge(self):
        """Initialize domain-specific knowledge bases."""
        # Actuarial and insurance terms
        self.actuarial_terms = {
            'basic': [
                'premi', 'klaim', 'polis', 'asuransi', 'tertanggung', 'penanggung',
                'manfaat', 'risiko', 'coverage', 'deductible', 'copayment'
            ],
            'intermediate': [
                'liabilitas', 'reserv', 'diskonto', 'mortalitas', 'morbiditas',
                'lapse', 'surrender', 'underwriting', 'reinsurance', 'cedent'
            ],
            'advanced': [
                'aktuaria', 'stokastik', 'monte carlo', 'var', 'cvar', 'solvency',
                'capital adequacy', 'risk based capital', 'embedded value', 'mcev'
            ]
        }
        
        # PSAK 219 specific terms
        self.psak219_terms = [
            'kontrak asuransi', 'kontrak investasi', 'komponen deposit',
            'liability adequacy test', 'lat', 'unearned premium reserve',
            'outstanding claims reserve', 'ibnr', 'incurred but not reported',
            'premium deficiency reserve', 'pdr', 'catastrophe reserve'
        ]
        
        # Regulatory frameworks
        self.regulatory_frameworks = [
            'PSAK 219', 'PSAK 62', 'SEOJK', 'POJK', 'Solvency II',
            'IFRS 17', 'IFRS 4', 'Basel III', 'Risk Based Capital'
        ]
        
        # Technical complexity indicators
        self.complexity_indicators = {
            'high': ['stokastik', 'monte carlo', 'bootstrap', 'var', 'cvar', 'copula'],
            'medium': ['diskonto', 'mortalitas', 'morbiditas', 'reserv', 'liabilitas'],
            'low': ['premi', 'klaim', 'polis', 'manfaat', 'tertanggung']
        }
    
    def extract_metadata(
        self, 
        content: str, 
        chunk_id: str,
        doc_name: str,
        section_info: Dict[str, Any],
        front_matter: Dict[str, Any]
    ) -> ChunkMetadata:
        """Extract simplified, structured metadata from a chunk.
        
        Args:
            content: Chunk content
            chunk_id: Unique chunk identifier
            doc_name: Document name
            section_info: Section information
            front_matter: YAML front-matter
            
        Returns:
            ChunkMetadata object with simplified structure
        """
        try:
            self.logger.debug(f"Extracting simplified metadata for chunk {chunk_id}")
            
            # Extract numerical data from content
            numbers = self._extract_numerical_data(content)
            
            # Extract keywords (combine frontmatter + content-based)
            keywords = self._extract_keywords(content, front_matter)
            
            # Extract regulations
            regulations = self._extract_regulations(content, front_matter)
            
            # Determine section order
            section_order = self._calculate_section_order(section_info)
            
            # Determine difficulty
            difficulty = self._determine_difficulty_simple(content, front_matter)
            
            # Create simplified metadata
            metadata = ChunkMetadata(
                # Core fields from frontmatter and analysis
                doc_name=doc_name,
                doc_type=front_matter.get('document_type', front_matter.get('doc_type', 'teknis')),
                domain=front_matter.get('domain', 'aktuaria'),
                scope=front_matter.get('scope', 'sains_aktuaria_imbalan_kerja'),
                section_heading=section_info.get('heading', 'Unknown Section'),
                section_order=section_order,
                keywords=keywords,
                difficulty=difficulty,
                last_updated=front_matter.get('last_updated', datetime.now().strftime('%Y-%m-%d')),
                version=front_matter.get('version', '1.0'),
                related_regulations=regulations,
                
                # Optional numerical data
                numbers=numbers,
                
                # Internal fields
                chunk_id=chunk_id,
                confidence_score=0.95  # High confidence for structured extraction
            )
            
            self.logger.debug(f"Successfully extracted simplified metadata for chunk {chunk_id}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting simplified metadata for chunk {chunk_id}: {e}")
            
            # Return minimal metadata on error
            return ChunkMetadata(
                doc_name=doc_name,
                chunk_id=chunk_id,
                section_heading=section_info.get('heading', 'Unknown Section'),
                confidence_score=0.1
            )

    def _extract_numerical_data(self, content: str) -> Dict[str, Union[float, int]]:
        """Extract structured numerical data from content.
        
        Args:
            content: Chunk content
            
        Returns:
            Dictionary of numerical data with semantic keys
        """
        numbers = {}
        
        try:
            # Extract discount rate
            discount_match = self.patterns['discount_rate'].search(content)
            if discount_match:
                numbers['discount_rate'] = float(discount_match.group(1)) / 100
            
            # Extract mortality rate
            mortality_match = self.patterns['mortality_rate'].search(content)
            if mortality_match:
                numbers['mortality_rate'] = float(mortality_match.group(1))
            
            # Extract salary increase rate
            salary_match = self.patterns['salary_increase'].search(content)
            if salary_match:
                numbers['salary_increase_rate'] = float(salary_match.group(1)) / 100
            
            # Extract years
            year_matches = self.patterns['year'].findall(content)
            if year_matches:
                # Take the most recent year or valuation year
                years = [int(y) for y in year_matches]
                numbers['valuation_year'] = max(years)
            
            # Extract ages
            age_matches = self.patterns['age'].findall(content)
            if age_matches:
                ages = [int(a) for a in age_matches]
                numbers['retirement_age'] = max(ages)
            
            # Extract percentages (general)
            percentage_matches = self.patterns['percentage'].findall(content)
            if percentage_matches:
                percentages = [float(p.replace('%', '')) for p in percentage_matches]
                if percentages:
                    numbers['percentage_values'] = percentages
            
        except Exception as e:
            self.logger.error(f"Error extracting numerical data: {e}")
            
        return numbers

    def _extract_keywords(self, content: str, front_matter: Dict[str, Any]) -> List[str]:
        """Extract keywords from frontmatter and content.
        
        Args:
            content: Chunk content
            front_matter: YAML front-matter
            
        Returns:
            List of relevant keywords
        """
        keywords = []
        
        # Get keywords from frontmatter
        fm_keywords = front_matter.get('keywords', [])
        if isinstance(fm_keywords, list):
            keywords.extend(fm_keywords)
        elif isinstance(fm_keywords, str):
            keywords.append(fm_keywords)
        
        # Extract key actuarial terms from content
        actuarial_terms = [
            'PUC', 'projected unit credit', 'PSAK 219', 'PSAK 24',
            'aktuaria', 'valuasi', 'imbalan kerja', 'defined benefit',
            'mortality', 'discount rate', 'salary increase'
        ]
        
        content_lower = content.lower()
        for term in actuarial_terms:
            if term.lower() in content_lower and term not in keywords:
                keywords.append(term)
        
        # Limit to most relevant keywords
        return keywords[:10]

    def _extract_regulations(self, content: str, front_matter: Dict[str, Any]) -> List[str]:
        """Extract regulation references from content and frontmatter.
        
        Args:
            content: Chunk content
            front_matter: YAML front-matter
            
        Returns:
            List of regulation references
        """
        regulations = []
        
        try:
            # Get from frontmatter
            fm_regs = front_matter.get('related_regulations', [])
            if isinstance(fm_regs, list):
                regulations.extend(fm_regs)
            elif isinstance(fm_regs, str):
                regulations.append(fm_regs)
            
            # Extract PSAK references
            psak_matches = self.patterns['psak_reference'].findall(content)
            for match in psak_matches:
                # Clean up the match
                psak_ref = match.replace(' ', '_').upper()
                if psak_ref not in regulations:
                    regulations.append(psak_ref)
            
            # Extract other regulation references
            if 'regulation' in self.patterns:
                reg_matches = self.patterns['regulation'].findall(content)
                for match in reg_matches:
                    if match not in regulations:
                        regulations.append(match)
        
        except Exception as e:
            self.logger.error(f"Error extracting regulations: {e}")
        
        return regulations

    def _calculate_section_order(self, section_info: Dict[str, Any]) -> int:
        """Calculate section order as integer.
        
        Args:
            section_info: Section information
            
        Returns:
            Section order as integer
        """
        order = section_info.get('order', 0)
        if isinstance(order, (int, float)):
            return int(order)
        return 0

    def _determine_difficulty_simple(self, content: str, front_matter: Dict[str, Any]) -> str:
        """Determine difficulty level using simplified logic.
        
        Args:
            content: Chunk content
            front_matter: YAML front-matter
            
        Returns:
            Difficulty level string
        """
        # Check frontmatter first
        if 'difficulty' in front_matter:
            return front_matter['difficulty']
        
        # Simple heuristics based on content
        content_lower = content.lower()
        
        # Advanced indicators
        advanced_terms = ['projected unit credit', 'mortality table', 'discount rate', 'actuarial valuation']
        advanced_count = sum(1 for term in advanced_terms if term in content_lower)
        
        # Technical indicators
        technical_terms = ['formula', 'calculation', 'method', 'assumption']
        technical_count = sum(1 for term in technical_terms if term in content_lower)
        
        if advanced_count >= 2 or technical_count >= 3:
            return "advanced"
        elif advanced_count >= 1 or technical_count >= 2:
            return "intermediate"
        elif technical_count >= 1:
            return "moderate"
        else:
            return "basic"

    def batch_extract_metadata(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> List[ChunkMetadata]:
        """Extract metadata for multiple chunks in batch.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of ChunkMetadata objects
        """
        try:
            metadata_list = []
            
            for chunk_data in chunks:
                metadata = self.extract_metadata(
                    content=chunk_data['content'],
                    chunk_id=chunk_data['chunk_id'],
                    doc_name=chunk_data['doc_name'],
                    section_info=chunk_data.get('section_info', {}),
                    front_matter=chunk_data.get('front_matter', {})
                )
                metadata_list.append(metadata)
            
            self.logger.info(f"Batch metadata extraction completed for {len(chunks)} chunks")
            return metadata_list
            
        except Exception as e:
            error_msg = f"Error in batch metadata extraction: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_extraction_stats(self, metadata_list: List[ChunkMetadata]) -> Dict[str, Any]:
        """Get statistics about metadata extraction results.
        
        Args:
            metadata_list: List of ChunkMetadata objects
            
        Returns:
            Statistics dictionary
        """
        try:
            if not metadata_list:
                return {'error': 'No metadata provided'}
            
            # Basic stats
            total_chunks = len(metadata_list)
            avg_confidence = sum(m.confidence_score for m in metadata_list) / total_chunks
            
            # Complexity distribution
            complexity_dist = Counter(m.complexity_level.value for m in metadata_list)
            
            # Content type distribution
            content_type_dist = Counter(m.content_type for m in metadata_list)
            
            # Quality metrics
            avg_completeness = sum(m.completeness_score for m in metadata_list) / total_chunks
            avg_info_density = sum(m.information_density for m in metadata_list) / total_chunks
            avg_clarity = sum(m.clarity_score for m in metadata_list) / total_chunks
            
            # Concept extraction
            total_concepts = sum(len(m.key_concepts) for m in metadata_list)
            avg_concepts_per_chunk = total_concepts / total_chunks
            
            # Technical content
            chunks_with_formulas = sum(1 for m in metadata_list if m.formula_count > 0)
            chunks_with_tables = sum(1 for m in metadata_list if m.table_count > 0)
            chunks_with_psak = sum(1 for m in metadata_list if m.psak_references)
            
            return {
                'total_chunks': total_chunks,
                'average_confidence': round(avg_confidence, 3),
                'complexity_distribution': dict(complexity_dist),
                'content_type_distribution': dict(content_type_dist),
                'average_completeness': round(avg_completeness, 3),
                'average_information_density': round(avg_info_density, 3),
                'average_clarity': round(avg_clarity, 3),
                'average_concepts_per_chunk': round(avg_concepts_per_chunk, 2),
                'chunks_with_formulas': chunks_with_formulas,
                'chunks_with_tables': chunks_with_tables,
                'chunks_with_psak_references': chunks_with_psak,
                'total_concepts_extracted': total_concepts
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating extraction stats: {e}")
            return {'error': f'Failed to calculate stats: {str(e)}'}