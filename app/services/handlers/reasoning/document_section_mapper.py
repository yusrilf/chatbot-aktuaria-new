"""Document Section Mapper untuk Tree of Thought Document Reader.

Mapper yang memetakan pertanyaan ke bagian dokumen yang spesifik
dan mengidentifikasi section yang paling relevan untuk retrieval.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class SectionType(Enum):
    """Enum untuk jenis section dokumen."""
    DEFINITION = "definition"  # Bagian definisi/konsep
    CALCULATION = "calculation"  # Bagian perhitungan/rumus
    EXAMPLE = "example"  # Bagian contoh/ilustrasi
    PROCEDURE = "procedure"  # Bagian prosedur/langkah
    TABLE = "table"  # Bagian tabel/data
    REGULATION = "regulation"  # Bagian peraturan/compliance
    ANALYSIS = "analysis"  # Bagian analisis/interpretasi
    REFERENCE = "reference"  # Bagian referensi/lampiran

class RelevanceReason(Enum):
    """Enum untuk alasan relevansi."""
    KEYWORD_MATCH = "keyword_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    SECTION_TYPE_MATCH = "section_type_match"
    DOMAIN_RELEVANCE = "domain_relevance"
    CONTEXT_ALIGNMENT = "context_alignment"
    HISTORICAL_USAGE = "historical_usage"

@dataclass
class SectionMapping:
    """Mapping antara pertanyaan dan section dokumen."""
    section_id: str
    section_title: str
    section_type: SectionType
    document_name: str
    relevance_score: float
    relevance_reasons: List[RelevanceReason]
    matched_keywords: List[str]
    content_preview: str
    metadata: Dict[str, Any]

@dataclass
class DocumentStructure:
    """Struktur dokumen dengan section information."""
    document_name: str
    sections: List[Dict[str, Any]]
    document_type: str
    domain: str
    last_updated: Optional[str]
    metadata: Dict[str, Any]

class DocumentSectionMapper:
    """Mapper untuk memetakan pertanyaan ke section dokumen yang relevan."""
    
    def __init__(self, vector_store_manager=None, llm=None):
        """
        Initialize Document Section Mapper.
        
        Args:
            vector_store_manager: Vector store manager untuk document retrieval
            llm: Language model untuk semantic analysis
        """
        self.vector_store_manager = vector_store_manager
        self.llm = llm
        
        # Section type indicators
        self.section_indicators = {
            SectionType.DEFINITION: [
                'definisi', 'pengertian', 'konsep', 'arti', 'makna',
                'definition', 'concept', 'meaning', 'terminology'
            ],
            SectionType.CALCULATION: [
                'perhitungan', 'rumus', 'formula', 'kalkulasi', 'hitung',
                'calculation', 'formula', 'compute', 'equation'
            ],
            SectionType.EXAMPLE: [
                'contoh', 'ilustrasi', 'kasus', 'studi kasus', 'sampel',
                'example', 'illustration', 'case study', 'sample'
            ],
            SectionType.PROCEDURE: [
                'prosedur', 'langkah', 'tahap', 'cara', 'metode',
                'procedure', 'steps', 'method', 'process', 'workflow'
            ],
            SectionType.TABLE: [
                'tabel', 'data', 'daftar', 'list', 'angka',
                'table', 'data', 'list', 'figures', 'statistics'
            ],
            SectionType.REGULATION: [
                'peraturan', 'regulasi', 'ketentuan', 'aturan', 'compliance',
                'regulation', 'rule', 'compliance', 'requirement', 'standard'
            ],
            SectionType.ANALYSIS: [
                'analisis', 'evaluasi', 'assessment', 'interpretasi', 'review',
                'analysis', 'evaluation', 'assessment', 'interpretation'
            ],
            SectionType.REFERENCE: [
                'referensi', 'lampiran', 'appendix', 'sumber', 'daftar pustaka',
                'reference', 'appendix', 'bibliography', 'source'
            ]
        }
        
        # Priority weights untuk different section types berdasarkan question type
        self.section_priority_weights = {
            'theory': {
                SectionType.DEFINITION: 1.0,
                SectionType.ANALYSIS: 0.8,
                SectionType.EXAMPLE: 0.6,
                SectionType.REFERENCE: 0.4
            },
            'calculation': {
                SectionType.CALCULATION: 1.0,
                SectionType.EXAMPLE: 0.9,
                SectionType.PROCEDURE: 0.7,
                SectionType.TABLE: 0.6
            },
            'data_lookup': {
                SectionType.TABLE: 1.0,
                SectionType.REFERENCE: 0.8,
                SectionType.EXAMPLE: 0.6,
                SectionType.CALCULATION: 0.4
            },
            'comparison': {
                SectionType.ANALYSIS: 1.0,
                SectionType.EXAMPLE: 0.8,
                SectionType.DEFINITION: 0.6,
                SectionType.TABLE: 0.5
            },
            'procedure': {
                SectionType.PROCEDURE: 1.0,
                SectionType.EXAMPLE: 0.8,
                SectionType.CALCULATION: 0.6,
                SectionType.REFERENCE: 0.4
            },
            'interpretation': {
                SectionType.ANALYSIS: 1.0,
                SectionType.DEFINITION: 0.8,
                SectionType.EXAMPLE: 0.7,
                SectionType.CALCULATION: 0.5
            }
        }
        
        # Cache untuk document structures
        self.document_cache = {}
        
        # Configuration
        self.config = {
            "max_sections_per_document": 5,
            "min_relevance_threshold": 0.3,
            "enable_semantic_analysis": True,
            "enable_section_ranking": True,
            "cache_document_structures": True
        }
        
        logger.info("Document Section Mapper initialized")
    
    def map_question_to_sections(self, 
                               question_analysis: Dict[str, Any],
                               retrieved_documents: List[Dict[str, Any]],
                               session_id: str) -> List[SectionMapping]:
        """
        Map pertanyaan ke section dokumen yang relevan.
        
        Args:
            question_analysis: Hasil analisis pertanyaan
            retrieved_documents: Dokumen yang sudah diretrieve
            session_id: Session ID
            
        Returns:
            List of SectionMapping objects
        """
        try:
            logger.info(f"Mapping question to document sections for {len(retrieved_documents)} documents")
            
            all_mappings = []
            
            # Process setiap dokumen
            for doc in retrieved_documents:
                try:
                    # Extract document structure
                    doc_structure = self._extract_document_structure(doc)
                    
                    # Map sections untuk dokumen ini
                    doc_mappings = self._map_document_sections(
                        question_analysis, doc_structure, doc
                    )
                    
                    all_mappings.extend(doc_mappings)
                    
                except Exception as e:
                    logger.warning(f"Error mapping sections for document {doc.get('source', 'unknown')}: {e}")
                    continue
            
            # Rank dan filter mappings
            ranked_mappings = self._rank_and_filter_mappings(
                all_mappings, question_analysis
            )
            
            # Enhance dengan semantic analysis jika enabled
            if self.config["enable_semantic_analysis"] and self.llm:
                enhanced_mappings = self._enhance_with_semantic_analysis(
                    ranked_mappings, question_analysis
                )
            else:
                enhanced_mappings = ranked_mappings
            
            logger.info(f"Mapped to {len(enhanced_mappings)} relevant sections")
            return enhanced_mappings
            
        except Exception as e:
            logger.error(f"Error in question to section mapping: {e}")
            return self._get_fallback_mappings(retrieved_documents)
    
    def _extract_document_structure(self, document: Dict[str, Any]) -> DocumentStructure:
        """
        Extract struktur dokumen dengan section information.
        
        Args:
            document: Document data
            
        Returns:
            DocumentStructure object
        """
        try:
            doc_name = document.get('source', 'unknown')
            
            # Check cache first
            if self.config["cache_document_structures"] and doc_name in self.document_cache:
                return self.document_cache[doc_name]
            
            content = document.get('content', '')
            metadata = document.get('metadata', {})
            
            # Extract sections dari content
            sections = self._parse_document_sections(content, metadata)
            
            # Create document structure
            doc_structure = DocumentStructure(
                document_name=doc_name,
                sections=sections,
                document_type=metadata.get('type', 'unknown'),
                domain=metadata.get('domain', 'general'),
                last_updated=metadata.get('last_updated'),
                metadata=metadata
            )
            
            # Cache the structure
            if self.config["cache_document_structures"]:
                self.document_cache[doc_name] = doc_structure
            
            return doc_structure
            
        except Exception as e:
            logger.error(f"Error extracting document structure: {e}")
            return DocumentStructure(
                document_name=document.get('source', 'unknown'),
                sections=[],
                document_type='unknown',
                domain='general',
                last_updated=None,
                metadata={}
            )
    
    def _parse_document_sections(self, 
                               content: str,
                               metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse content untuk mengidentifikasi sections.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of section dictionaries
        """
        try:
            sections = []
            
            # Split content berdasarkan headers/sections
            # Pattern untuk headers (markdown style, numbered sections, etc.)
            header_patterns = [
                r'^#{1,6}\s+(.+)$',  # Markdown headers
                r'^\d+\.\s+(.+)$',   # Numbered sections
                r'^[A-Z][A-Z\s]+:',  # ALL CAPS headers
                r'^\*\*(.+)\*\*',    # Bold headers
                r'^(.+)\n=+$',       # Underlined headers
                r'^(.+)\n-+$'        # Dashed underlined headers
            ]
            
            lines = content.split('\n')
            current_section = {
                'title': 'Introduction',
                'content': '',
                'start_line': 0,
                'type': SectionType.DEFINITION
            }
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check if line is a header
                is_header = False
                header_title = None
                
                for pattern in header_patterns:
                    match = re.match(pattern, line, re.MULTILINE)
                    if match:
                        header_title = match.group(1).strip()
                        is_header = True
                        break
                
                if is_header and header_title:
                    # Save previous section jika ada content
                    if current_section['content'].strip():
                        current_section['end_line'] = i - 1
                        current_section['section_id'] = f"{metadata.get('source', 'doc')}_{len(sections)}"
                        sections.append(current_section.copy())
                    
                    # Start new section
                    section_type = self._detect_section_type(header_title, '')
                    current_section = {
                        'title': header_title,
                        'content': '',
                        'start_line': i,
                        'type': section_type
                    }
                else:
                    # Add line to current section
                    current_section['content'] += line + '\n'
            
            # Add last section
            if current_section['content'].strip():
                current_section['end_line'] = len(lines) - 1
                current_section['section_id'] = f"{metadata.get('source', 'doc')}_{len(sections)}"
                sections.append(current_section)
            
            # Jika tidak ada sections terdeteksi, treat whole content as one section
            if not sections:
                sections.append({
                    'section_id': f"{metadata.get('source', 'doc')}_0",
                    'title': metadata.get('title', 'Document Content'),
                    'content': content,
                    'start_line': 0,
                    'end_line': len(lines) - 1,
                    'type': self._detect_section_type('', content)
                })
            
            # Enhance section types berdasarkan content
            for section in sections:
                enhanced_type = self._detect_section_type(
                    section['title'], section['content']
                )
                section['type'] = enhanced_type
            
            return sections
            
        except Exception as e:
            logger.error(f"Error parsing document sections: {e}")
            return [{
                'section_id': f"{metadata.get('source', 'doc')}_0",
                'title': 'Full Document',
                'content': content,
                'start_line': 0,
                'end_line': 0,
                'type': SectionType.DEFINITION
            }]
    
    def _detect_section_type(self, title: str, content: str) -> SectionType:
        """
        Detect section type berdasarkan title dan content.
        
        Args:
            title: Section title
            content: Section content
            
        Returns:
            SectionType enum
        """
        try:
            text_to_analyze = (title + ' ' + content[:200]).lower()
            
            type_scores = defaultdict(int)
            
            # Score berdasarkan indicators
            for section_type, indicators in self.section_indicators.items():
                for indicator in indicators:
                    if indicator in text_to_analyze:
                        type_scores[section_type] += 1
            
            # Additional heuristics
            # Tables/data sections
            if re.search(r'\|.*\|.*\|', content) or 'tabel' in text_to_analyze:
                type_scores[SectionType.TABLE] += 2
            
            # Calculation sections
            if re.search(r'\d+\s*[+\-*/=]\s*\d+', content) or '=' in content:
                type_scores[SectionType.CALCULATION] += 2
            
            # Procedure sections
            if re.search(r'\d+\.\s', content) or 'langkah' in text_to_analyze:
                type_scores[SectionType.PROCEDURE] += 2
            
            # Return type dengan score tertinggi
            if type_scores:
                return max(type_scores, key=type_scores.get)
            else:
                return SectionType.DEFINITION  # Default
                
        except Exception as e:
            logger.error(f"Error detecting section type: {e}")
            return SectionType.DEFINITION
    
    def _map_document_sections(self, 
                             question_analysis: Dict[str, Any],
                             doc_structure: DocumentStructure,
                             document: Dict[str, Any]) -> List[SectionMapping]:
        """
        Map sections dari satu dokumen ke pertanyaan.
        
        Args:
            question_analysis: Question analysis results
            doc_structure: Document structure
            document: Original document data
            
        Returns:
            List of SectionMapping objects
        """
        try:
            mappings = []
            question_type = question_analysis.get('question_type', 'theory')
            keywords = question_analysis.get('keywords', [])
            
            # Get priority weights untuk question type
            priority_weights = self.section_priority_weights.get(
                question_type, self.section_priority_weights['theory']
            )
            
            for section in doc_structure.sections:
                try:
                    # Calculate relevance score
                    relevance_score, reasons, matched_keywords = self._calculate_section_relevance(
                        section, question_analysis, priority_weights
                    )
                    
                    # Skip sections dengan relevance terlalu rendah
                    if relevance_score < self.config["min_relevance_threshold"]:
                        continue
                    
                    # Create section mapping
                    mapping = SectionMapping(
                        section_id=section['section_id'],
                        section_title=section['title'],
                        section_type=section['type'],
                        document_name=doc_structure.document_name,
                        relevance_score=relevance_score,
                        relevance_reasons=reasons,
                        matched_keywords=matched_keywords,
                        content_preview=section['content'][:300] + '...' if len(section['content']) > 300 else section['content'],
                        metadata={
                            'start_line': section.get('start_line', 0),
                            'end_line': section.get('end_line', 0),
                            'document_type': doc_structure.document_type,
                            'domain': doc_structure.domain,
                            'full_content_length': len(section['content'])
                        }
                    )
                    
                    mappings.append(mapping)
                    
                except Exception as e:
                    logger.warning(f"Error mapping section {section.get('title', 'unknown')}: {e}")
                    continue
            
            # Limit sections per document
            max_sections = self.config["max_sections_per_document"]
            if len(mappings) > max_sections:
                mappings.sort(key=lambda x: x.relevance_score, reverse=True)
                mappings = mappings[:max_sections]
            
            return mappings
            
        except Exception as e:
            logger.error(f"Error mapping document sections: {e}")
            return []
    
    def _calculate_section_relevance(self, 
                                   section: Dict[str, Any],
                                   question_analysis: Dict[str, Any],
                                   priority_weights: Dict[SectionType, float]) -> Tuple[float, List[RelevanceReason], List[str]]:
        """
        Calculate relevance score untuk section.
        
        Args:
            section: Section data
            question_analysis: Question analysis
            priority_weights: Priority weights untuk section types
            
        Returns:
            Tuple of (relevance_score, reasons, matched_keywords)
        """
        try:
            score = 0.0
            reasons = []
            matched_keywords = []
            
            section_text = (section['title'] + ' ' + section['content']).lower()
            keywords = question_analysis.get('keywords', [])
            
            # 1. Section type priority
            section_type = section['type']
            if section_type in priority_weights:
                type_weight = priority_weights[section_type]
                score += type_weight * 0.3  # 30% weight untuk type matching
                if type_weight > 0.7:
                    reasons.append(RelevanceReason.SECTION_TYPE_MATCH)
            
            # 2. Keyword matching
            keyword_score = 0
            for keyword in keywords:
                if keyword.lower() in section_text:
                    keyword_score += 1
                    matched_keywords.append(keyword)
            
            if keywords:
                keyword_ratio = keyword_score / len(keywords)
                score += keyword_ratio * 0.4  # 40% weight untuk keyword matching
                if keyword_ratio > 0.3:
                    reasons.append(RelevanceReason.KEYWORD_MATCH)
            
            # 3. Domain relevance
            primary_domain = question_analysis.get('primary_domain', 'general')
            if primary_domain in section_text or primary_domain == 'general':
                score += 0.1  # 10% weight untuk domain matching
                reasons.append(RelevanceReason.DOMAIN_RELEVANCE)
            
            # 4. Context alignment
            context = question_analysis.get('context', {})
            context_score = 0
            
            if context.get('has_numbers') and re.search(r'\d+', section['content']):
                context_score += 0.1
            if context.get('has_formulas') and ('=' in section['content'] or 'rumus' in section_text):
                context_score += 0.1
            if context.get('has_procedures') and re.search(r'\d+\.\s', section['content']):
                context_score += 0.1
            
            score += context_score
            if context_score > 0.1:
                reasons.append(RelevanceReason.CONTEXT_ALIGNMENT)
            
            # 5. Content length bonus (longer sections might be more comprehensive)
            content_length = len(section['content'])
            if content_length > 500:
                score += 0.05
            elif content_length < 100:
                score -= 0.05  # Penalty untuk sections yang terlalu pendek
            
            # Normalize score to 0-1 range
            score = min(max(score, 0.0), 1.0)
            
            return score, reasons, matched_keywords
            
        except Exception as e:
            logger.error(f"Error calculating section relevance: {e}")
            return 0.3, [RelevanceReason.KEYWORD_MATCH], []
    
    def _rank_and_filter_mappings(self, 
                                mappings: List[SectionMapping],
                                question_analysis: Dict[str, Any]) -> List[SectionMapping]:
        """
        Rank dan filter section mappings.
        
        Args:
            mappings: List of section mappings
            question_analysis: Question analysis
            
        Returns:
            Ranked and filtered mappings
        """
        try:
            if not mappings:
                return []
            
            # Sort by relevance score
            mappings.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply additional ranking factors
            if self.config["enable_section_ranking"]:
                mappings = self._apply_advanced_ranking(mappings, question_analysis)
            
            # Filter by threshold
            filtered_mappings = [
                mapping for mapping in mappings 
                if mapping.relevance_score >= self.config["min_relevance_threshold"]
            ]
            
            # Limit total number of mappings
            max_total_sections = 15  # Reasonable limit
            if len(filtered_mappings) > max_total_sections:
                filtered_mappings = filtered_mappings[:max_total_sections]
            
            return filtered_mappings
            
        except Exception as e:
            logger.error(f"Error ranking and filtering mappings: {e}")
            return mappings[:10]  # Fallback limit
    
    def _apply_advanced_ranking(self, 
                              mappings: List[SectionMapping],
                              question_analysis: Dict[str, Any]) -> List[SectionMapping]:
        """
        Apply advanced ranking algorithms.
        
        Args:
            mappings: Section mappings
            question_analysis: Question analysis
            
        Returns:
            Re-ranked mappings
        """
        try:
            # Diversity bonus - prefer sections dari different documents
            document_counts = defaultdict(int)
            for mapping in mappings:
                document_counts[mapping.document_name] += 1
            
            # Apply diversity bonus
            for mapping in mappings:
                doc_count = document_counts[mapping.document_name]
                if doc_count == 1:
                    mapping.relevance_score += 0.05  # Bonus untuk unique document
                elif doc_count > 3:
                    mapping.relevance_score -= 0.02  # Penalty untuk over-representation
            
            # Section type diversity
            type_counts = defaultdict(int)
            for mapping in mappings:
                type_counts[mapping.section_type] += 1
            
            # Prefer diverse section types
            for mapping in mappings:
                type_count = type_counts[mapping.section_type]
                if type_count == 1:
                    mapping.relevance_score += 0.03
                elif type_count > 2:
                    mapping.relevance_score -= 0.01
            
            # Re-sort after adjustments
            mappings.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return mappings
            
        except Exception as e:
            logger.error(f"Error in advanced ranking: {e}")
            return mappings
    
    def _enhance_with_semantic_analysis(self, 
                                      mappings: List[SectionMapping],
                                      question_analysis: Dict[str, Any]) -> List[SectionMapping]:
        """
        Enhance mappings dengan semantic analysis menggunakan LLM.
        
        Args:
            mappings: Current mappings
            question_analysis: Question analysis
            
        Returns:
            Enhanced mappings
        """
        try:
            if not self.llm or not mappings:
                return mappings
            
            # Prepare context untuk LLM analysis
            question_type = question_analysis.get('question_type', 'theory')
            keywords = question_analysis.get('keywords', [])
            
            # Analyze top mappings dengan LLM
            top_mappings = mappings[:5]  # Analyze top 5 only untuk efficiency
            
            for mapping in top_mappings:
                try:
                    semantic_score = self._get_semantic_relevance_score(
                        mapping, question_analysis
                    )
                    
                    # Blend semantic score dengan existing score
                    original_score = mapping.relevance_score
                    blended_score = (original_score * 0.7) + (semantic_score * 0.3)
                    mapping.relevance_score = min(blended_score, 1.0)
                    
                    # Add semantic similarity reason jika score tinggi
                    if semantic_score > 0.7:
                        mapping.relevance_reasons.append(RelevanceReason.SEMANTIC_SIMILARITY)
                        
                except Exception as e:
                    logger.warning(f"Error in semantic analysis for mapping {mapping.section_id}: {e}")
                    continue
            
            # Re-sort after semantic enhancement
            mappings.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return mappings
            
        except Exception as e:
            logger.error(f"Error in semantic enhancement: {e}")
            return mappings
    
    def _get_semantic_relevance_score(self, 
                                    mapping: SectionMapping,
                                    question_analysis: Dict[str, Any]) -> float:
        """
        Get semantic relevance score menggunakan LLM.
        
        Args:
            mapping: Section mapping
            question_analysis: Question analysis
            
        Returns:
            Semantic relevance score (0.0 - 1.0)
        """
        try:
            # Create prompt untuk semantic analysis
            analysis_prompt = f"""
            Evaluasi relevansi section dokumen berikut untuk pertanyaan aktuaria:
            
            Jenis Pertanyaan: {question_analysis.get('question_type', 'theory')}
            Keywords: {', '.join(question_analysis.get('keywords', []))}
            Domain: {question_analysis.get('primary_domain', 'general')}
            
            Section Title: {mapping.section_title}
            Section Type: {mapping.section_type.value}
            Content Preview: {mapping.content_preview}
            
            Berikan skor relevansi 0.0-1.0 dalam format JSON:
            {{
                "relevance_score": 0.85,
                "reasoning": "alasan singkat"
            }}
            """
            
            response = self.llm.invoke(analysis_prompt)
            
            try:
                result = json.loads(response.content)
                return float(result.get('relevance_score', 0.5))
            except (json.JSONDecodeError, ValueError):
                # Fallback: extract score dari response text
                score_match = re.search(r'\b0\.[0-9]+\b|\b1\.0\b', response.content)
                if score_match:
                    return float(score_match.group())
                return 0.5
                
        except Exception as e:
            logger.error(f"Error getting semantic relevance score: {e}")
            return 0.5
    
    def _get_fallback_mappings(self, documents: List[Dict[str, Any]]) -> List[SectionMapping]:
        """
        Generate fallback mappings jika terjadi error.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Fallback section mappings
        """
        try:
            fallback_mappings = []
            
            for i, doc in enumerate(documents[:3]):  # Limit to 3 documents
                mapping = SectionMapping(
                    section_id=f"fallback_{i}",
                    section_title=doc.get('metadata', {}).get('title', 'Document Section'),
                    section_type=SectionType.DEFINITION,
                    document_name=doc.get('source', f'document_{i}'),
                    relevance_score=0.5,
                    relevance_reasons=[RelevanceReason.KEYWORD_MATCH],
                    matched_keywords=["aktuaria"],
                    content_preview=doc.get('content', '')[:300],
                    metadata={
                        'fallback_used': True,
                        'document_type': 'unknown',
                        'domain': 'general'
                    }
                )
                fallback_mappings.append(mapping)
            
            return fallback_mappings
            
        except Exception as e:
            logger.error(f"Error creating fallback mappings: {e}")
            return []
    
    def get_section_statistics(self, mappings: List[SectionMapping]) -> Dict[str, Any]:
        """
        Get statistics tentang section mappings.
        
        Args:
            mappings: Section mappings
            
        Returns:
            Statistics dictionary
        """
        try:
            if not mappings:
                return {"total_sections": 0}
            
            # Count by section type
            type_counts = defaultdict(int)
            for mapping in mappings:
                type_counts[mapping.section_type.value] += 1
            
            # Count by document
            doc_counts = defaultdict(int)
            for mapping in mappings:
                doc_counts[mapping.document_name] += 1
            
            # Relevance statistics
            scores = [mapping.relevance_score for mapping in mappings]
            
            return {
                "total_sections": len(mappings),
                "section_types": dict(type_counts),
                "documents_covered": len(doc_counts),
                "document_distribution": dict(doc_counts),
                "relevance_stats": {
                    "average_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "high_relevance_count": len([s for s in scores if s > 0.8])
                },
                "top_reasons": self._get_top_relevance_reasons(mappings)
            }
            
        except Exception as e:
            logger.error(f"Error getting section statistics: {e}")
            return {"error": str(e)}
    
    def _get_top_relevance_reasons(self, mappings: List[SectionMapping]) -> List[str]:
        """
        Get top relevance reasons dari mappings.
        
        Args:
            mappings: Section mappings
            
        Returns:
            List of top reasons
        """
        try:
            reason_counts = defaultdict(int)
            for mapping in mappings:
                for reason in mapping.relevance_reasons:
                    reason_counts[reason.value] += 1
            
            # Sort by frequency
            sorted_reasons = sorted(
                reason_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return [reason for reason, count in sorted_reasons[:5]]
            
        except Exception as e:
            logger.error(f"Error getting top relevance reasons: {e}")
            return []
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration.
        
        Args:
            new_config: New configuration values
        """
        try:
            self.config.update(new_config)
            logger.info(f"Document Section Mapper config updated: {new_config}")
        except Exception as e:
            logger.error(f"Error updating config: {e}")
    
    def clear_cache(self) -> None:
        """
        Clear document structure cache.
        """
        try:
            self.document_cache.clear()
            logger.info("Document structure cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")