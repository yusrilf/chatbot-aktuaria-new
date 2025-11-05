"""Query Refinement Service for Multihop Retrieval.

This module provides advanced query refinement capabilities for
multihop retrieval scenarios, including context-aware query expansion
and iterative query improvement.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel

logger = logging.getLogger(__name__)

class QueryRefinementService:
    """Service for refining queries in multihop retrieval scenarios.
    
    This service provides various query refinement strategies:
    1. Context-aware expansion
    2. Semantic decomposition
    3. Gap-filling queries
    4. Specificity adjustment
    """
    
    def __init__(self, 
                 llm: BaseLanguageModel,
                 query_expansion_service=None,
                 max_query_length: int = 100):
        """Initialize QueryRefinementService.
        
        Args:
            llm: Language model for query refinement
            query_expansion_service: Existing query expansion service
            max_query_length: Maximum length for refined queries
        """
        self.llm = llm
        self.query_expansion_service = query_expansion_service
        self.max_query_length = max_query_length
        
        # Aktuaria-specific terms and concepts
        self.actuarial_terms = {
            'mortality': ['mortalitas', 'kematian', 'tingkat kematian', 'tabel mortalitas'],
            'morbidity': ['morbiditas', 'kesakitan', 'tingkat kesakitan'],
            'reserves': ['cadangan', 'reserve', 'cadangan teknis', 'mathematical reserve'],
            'premium': ['premi', 'premium', 'tarif premi', 'perhitungan premi'],
            'valuation': ['valuasi', 'penilaian', 'actuarial valuation'],
            'liability': ['kewajiban', 'liabilitas', 'actuarial liability'],
            'risk': ['risiko', 'risk assessment', 'manajemen risiko'],
            'insurance': ['asuransi', 'pertanggungan', 'polis asuransi'],
            'pension': ['pensiun', 'dana pensiun', 'pension fund'],
            'annuity': ['anuitas', 'annuity', 'pembayaran berkala']
        }
        
        self._init_refinement_chains()
        logger.info("QueryRefinementService initialized")
        
    def _init_refinement_chains(self) -> None:
        """Initialize query refinement chains."""
        try:
            # Context-aware refinement chain
            context_refinement_prompt = PromptTemplate(
                input_variables=["original_query", "context_summary", "missing_aspects"],
                template="""Anda adalah ahli aktuaria yang membantu memperbaiki query pencarian.

Query Asli: {original_query}

Konteks yang sudah ditemukan:
{context_summary}

Aspek yang masih kurang:
{missing_aspects}

Buat query pencarian baru yang:
1. Fokus pada aspek yang masih kurang
2. Menggunakan istilah teknis aktuaria yang tepat
3. Lebih spesifik dari query asli
4. Maksimal 15 kata
5. Dalam bahasa Indonesia

Query yang diperbaiki:"""
            )
            
            self.context_refinement_chain = LLMChain(
                llm=self.llm,
                prompt=context_refinement_prompt,
                verbose=False
            )
            
            # Gap analysis chain
            gap_analysis_prompt = PromptTemplate(
                input_variables=["query", "retrieved_content"],
                template="""Analisis gap dalam informasi yang ditemukan untuk query aktuaria.

Query: {query}

Informasi yang ditemukan:
{retrieved_content}

Identifikasi aspek yang masih kurang:
1. Konsep aktuaria yang belum tercakup
2. Detail teknis yang masih diperlukan
3. Perhitungan atau formula yang hilang
4. Regulasi atau standar yang relevan

Daftar aspek yang kurang (maksimal 5 poin, masing-masing 1 kalimat):"""
            )
            
            self.gap_analysis_chain = LLMChain(
                llm=self.llm,
                prompt=gap_analysis_prompt,
                verbose=False
            )
            
            # Query decomposition chain
            decomposition_prompt = PromptTemplate(
                input_variables=["complex_query"],
                template="""Pecah query aktuaria kompleks menjadi sub-query yang lebih sederhana.

Query Kompleks: {complex_query}

Pecah menjadi 2-4 sub-query yang:
1. Masing-masing fokus pada satu aspek
2. Dapat dicari secara independen
3. Menggunakan istilah aktuaria yang tepat
4. Berurutan dari umum ke spesifik

Sub-queries:
1.
2.
3.
4."""
            )
            
            self.decomposition_chain = LLMChain(
                llm=self.llm,
                prompt=decomposition_prompt,
                verbose=False
            )
            
            logger.info("Query refinement chains initialized")
            
        except Exception as e:
            logger.error(f"Error initializing refinement chains: {str(e)}")
            raise
    
    def refine_query_with_context(self, 
                                 original_query: str,
                                 retrieved_documents: List[Tuple[Document, float]],
                                 hop_number: int = 1) -> Dict[str, Any]:
        """Refine query based on retrieved context.
        
        Args:
            original_query: Original user query
            retrieved_documents: Documents retrieved so far
            hop_number: Current hop number
            
        Returns:
            Dictionary containing refined query and metadata
        """
        try:
            logger.info(f"Refining query for hop {hop_number}: {original_query[:50]}...")
            
            # Analyze current context
            context_analysis = self._analyze_retrieved_context(retrieved_documents)
            
            # Identify gaps
            gap_analysis = self._identify_information_gaps(
                original_query, 
                context_analysis['summary']
            )
            
            # Generate refined query
            refined_query = self._generate_context_aware_query(
                original_query=original_query,
                context_summary=context_analysis['summary'],
                missing_aspects=gap_analysis['gaps']
            )
            
            # Apply actuarial term enhancement
            enhanced_query = self._enhance_with_actuarial_terms(refined_query)
            
            # Validate and adjust query length
            final_query = self._validate_and_adjust_query(enhanced_query)
            
            result = {
                'success': True,
                'original_query': original_query,
                'refined_query': final_query,
                'context_analysis': context_analysis,
                'gap_analysis': gap_analysis,
                'hop_number': hop_number,
                'refinement_strategy': 'context_aware',
                'confidence': self._calculate_refinement_confidence(context_analysis, gap_analysis)
            }
            
            logger.info(f"Query refined: '{final_query[:50]}...' (confidence: {result['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in query refinement: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_query': original_query,
                'refined_query': original_query,  # Fallback to original
                'hop_number': hop_number
            }
    
    def decompose_complex_query(self, complex_query: str) -> Dict[str, Any]:
        """Decompose complex query into simpler sub-queries.
        
        Args:
            complex_query: Complex query to decompose
            
        Returns:
            Dictionary containing sub-queries and metadata
        """
        try:
            logger.info(f"Decomposing complex query: {complex_query[:50]}...")
            
            # Use LLM to decompose query
            try:
                decomposition_result = self.decomposition_chain.run(
                    complex_query=complex_query
                )
                
                # Parse sub-queries from result
                sub_queries = self._parse_sub_queries(decomposition_result)
                
            except Exception as e:
                logger.warning(f"LLM decomposition failed: {str(e)}")
                # Fallback to rule-based decomposition
                sub_queries = self._rule_based_decomposition(complex_query)
            
            # Enhance sub-queries with actuarial terms
            enhanced_sub_queries = []
            for sq in sub_queries:
                enhanced = self._enhance_with_actuarial_terms(sq)
                enhanced_sub_queries.append(enhanced)
            
            result = {
                'success': True,
                'original_query': complex_query,
                'sub_queries': enhanced_sub_queries,
                'decomposition_method': 'llm_with_fallback',
                'query_count': len(enhanced_sub_queries)
            }
            
            logger.info(f"Query decomposed into {len(enhanced_sub_queries)} sub-queries")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in query decomposition: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_query': complex_query,
                'sub_queries': [complex_query]  # Fallback
            }
    
    def generate_follow_up_queries(self, 
                                  original_query: str,
                                  current_results: List[Tuple[Document, float]],
                                  num_queries: int = 3) -> List[str]:
        """Generate follow-up queries based on current results.
        
        Args:
            original_query: Original query
            current_results: Current retrieval results
            num_queries: Number of follow-up queries to generate
            
        Returns:
            List of follow-up queries
        """
        try:
            # Analyze what's been found
            covered_topics = self._extract_covered_topics(current_results)
            
            # Generate complementary queries
            follow_up_queries = []
            
            # Strategy 1: Detail expansion
            if covered_topics:
                detail_query = f"{original_query} detail {' '.join(list(covered_topics)[:2])}"
                follow_up_queries.append(detail_query)
            
            # Strategy 2: Related concepts
            related_terms = self._find_related_actuarial_terms(original_query)
            if related_terms:
                related_query = f"{related_terms[0]} {original_query.split()[-1] if original_query.split() else ''}"
                follow_up_queries.append(related_query)
            
            # Strategy 3: Practical application
            practical_query = f"contoh penerapan {original_query}"
            follow_up_queries.append(practical_query)
            
            # Strategy 4: Regulatory/standard aspects
            regulatory_query = f"regulasi standar {original_query}"
            follow_up_queries.append(regulatory_query)
            
            # Take only requested number and enhance
            selected_queries = follow_up_queries[:num_queries]
            enhanced_queries = []
            
            for query in selected_queries:
                enhanced = self._enhance_with_actuarial_terms(query)
                validated = self._validate_and_adjust_query(enhanced)
                enhanced_queries.append(validated)
            
            logger.info(f"Generated {len(enhanced_queries)} follow-up queries")
            
            return enhanced_queries
            
        except Exception as e:
            logger.error(f"Error generating follow-up queries: {str(e)}")
            return [original_query]  # Fallback
    
    def _analyze_retrieved_context(self, 
                                  documents: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """Analyze retrieved documents to understand current context.
        
        Args:
            documents: Retrieved documents with scores
            
        Returns:
            Context analysis dictionary
        """
        if not documents:
            return {
                'summary': 'Tidak ada dokumen ditemukan',
                'topics': set(),
                'confidence': 0.0,
                'document_count': 0
            }
        
        # Extract topics and create summary
        topics = set()
        content_snippets = []
        total_confidence = 0.0
        
        for doc, score in documents[:5]:  # Analyze top 5 documents
            # Extract key terms from content
            content = doc.page_content[:300]  # First 300 chars
            doc_topics = self._extract_topics_from_text(content)
            topics.update(doc_topics)
            
            # Create snippet for summary
            filename = doc.metadata.get('filename', 'Unknown')
            snippet = f"[{filename}] {content[:100]}..."
            content_snippets.append(snippet)
            
            total_confidence += score
        
        avg_confidence = total_confidence / len(documents)
        summary = "\n".join(content_snippets)
        
        return {
            'summary': summary,
            'topics': topics,
            'confidence': avg_confidence,
            'document_count': len(documents)
        }
    
    def _identify_information_gaps(self, 
                                  original_query: str,
                                  context_summary: str) -> Dict[str, Any]:
        """Identify gaps in retrieved information.
        
        Args:
            original_query: Original query
            context_summary: Summary of retrieved context
            
        Returns:
            Gap analysis dictionary
        """
        try:
            # Use LLM for gap analysis
            gap_result = self.gap_analysis_chain.run(
                query=original_query,
                retrieved_content=context_summary
            )
            
            # Parse gaps from result
            gaps = self._parse_gaps_from_result(gap_result)
            
        except Exception as e:
            logger.warning(f"LLM gap analysis failed: {str(e)}")
            # Fallback to rule-based gap identification
            gaps = self._rule_based_gap_identification(original_query, context_summary)
        
        return {
            'gaps': gaps,
            'gap_count': len(gaps),
            'analysis_method': 'llm_with_fallback'
        }
    
    def _generate_context_aware_query(self, 
                                     original_query: str,
                                     context_summary: str,
                                     missing_aspects: List[str]) -> str:
        """Generate context-aware refined query.
        
        Args:
            original_query: Original query
            context_summary: Summary of current context
            missing_aspects: List of missing aspects
            
        Returns:
            Refined query string
        """
        try:
            missing_text = "\n".join([f"- {aspect}" for aspect in missing_aspects[:3]])
            
            refined_query = self.context_refinement_chain.run(
                original_query=original_query,
                context_summary=context_summary[:500],  # Limit context length
                missing_aspects=missing_text
            )
            
            return refined_query.strip()
            
        except Exception as e:
            logger.warning(f"Context-aware query generation failed: {str(e)}")
            # Fallback to simple refinement
            if missing_aspects:
                return f"{original_query} {missing_aspects[0]}"
            else:
                return f"{original_query} detail"
    
    def _enhance_with_actuarial_terms(self, query: str) -> str:
        """Enhance query with relevant actuarial terms.
        
        Args:
            query: Query to enhance
            
        Returns:
            Enhanced query with actuarial terms
        """
        query_lower = query.lower()
        enhanced_terms = []
        
        # Find matching actuarial concepts
        for concept, terms in self.actuarial_terms.items():
            for term in terms:
                if term.lower() in query_lower:
                    # Add related terms
                    related = [t for t in terms if t.lower() != term.lower()]
                    if related:
                        enhanced_terms.append(related[0])
                    break
        
        # Add enhanced terms to query
        if enhanced_terms:
            enhanced_query = f"{query} {' '.join(enhanced_terms[:2])}"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def _validate_and_adjust_query(self, query: str) -> str:
        """Validate and adjust query length and format.
        
        Args:
            query: Query to validate
            
        Returns:
            Validated and adjusted query
        """
        # Clean up query
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Adjust length if necessary
        if len(cleaned) > self.max_query_length:
            words = cleaned.split()
            # Keep most important words (first and last parts)
            if len(words) > 10:
                cleaned = ' '.join(words[:5] + words[-5:])
            else:
                cleaned = cleaned[:self.max_query_length]
        
        return cleaned
    
    def _extract_topics_from_text(self, text: str) -> set:
        """Extract topics from text using actuarial term matching.
        
        Args:
            text: Text to extract topics from
            
        Returns:
            Set of identified topics
        """
        text_lower = text.lower()
        topics = set()
        
        for concept, terms in self.actuarial_terms.items():
            for term in terms:
                if term.lower() in text_lower:
                    topics.add(concept)
                    break
        
        return topics
    
    def _parse_sub_queries(self, decomposition_result: str) -> List[str]:
        """Parse sub-queries from LLM decomposition result.
        
        Args:
            decomposition_result: Result from decomposition chain
            
        Returns:
            List of sub-queries
        """
        lines = decomposition_result.strip().split('\n')
        sub_queries = []
        
        for line in lines:
            # Look for numbered items
            match = re.match(r'^\d+\.\s*(.+)$', line.strip())
            if match:
                sub_query = match.group(1).strip()
                if sub_query and len(sub_query) > 5:  # Minimum length check
                    sub_queries.append(sub_query)
        
        return sub_queries[:4]  # Maximum 4 sub-queries
    
    def _rule_based_decomposition(self, complex_query: str) -> List[str]:
        """Fallback rule-based query decomposition.
        
        Args:
            complex_query: Complex query to decompose
            
        Returns:
            List of sub-queries
        """
        # Simple rule-based decomposition
        base_terms = complex_query.split()
        
        if len(base_terms) <= 2:
            return [complex_query]
        
        sub_queries = [
            ' '.join(base_terms[:2]),  # First part
            ' '.join(base_terms[-2:]),  # Last part
            f"contoh {complex_query}",  # Example query
        ]
        
        return [sq for sq in sub_queries if len(sq) > 3]
    
    def _parse_gaps_from_result(self, gap_result: str) -> List[str]:
        """Parse gaps from LLM gap analysis result.
        
        Args:
            gap_result: Result from gap analysis chain
            
        Returns:
            List of identified gaps
        """
        lines = gap_result.strip().split('\n')
        gaps = []
        
        for line in lines:
            # Look for numbered or bulleted items
            match = re.match(r'^[\d\-\*]\s*(.+)$', line.strip())
            if match:
                gap = match.group(1).strip()
                if gap and len(gap) > 10:  # Minimum length check
                    gaps.append(gap)
        
        return gaps[:5]  # Maximum 5 gaps
    
    def _rule_based_gap_identification(self, 
                                      original_query: str,
                                      context_summary: str) -> List[str]:
        """Fallback rule-based gap identification.
        
        Args:
            original_query: Original query
            context_summary: Context summary
            
        Returns:
            List of identified gaps
        """
        gaps = []
        
        # Check for missing actuarial concepts
        query_topics = self._extract_topics_from_text(original_query)
        context_topics = self._extract_topics_from_text(context_summary)
        
        missing_topics = query_topics - context_topics
        
        for topic in missing_topics:
            gaps.append(f"Detail tentang {topic}")
        
        # Add common gaps
        if 'perhitungan' not in context_summary.lower():
            gaps.append("Formula dan perhitungan")
        
        if 'contoh' not in context_summary.lower():
            gaps.append("Contoh penerapan praktis")
        
        return gaps[:3]
    
    def _find_related_actuarial_terms(self, query: str) -> List[str]:
        """Find related actuarial terms for query expansion.
        
        Args:
            query: Query to find related terms for
            
        Returns:
            List of related terms
        """
        query_lower = query.lower()
        related_terms = []
        
        for concept, terms in self.actuarial_terms.items():
            for term in terms:
                if term.lower() in query_lower:
                    # Add other terms from same concept
                    other_terms = [t for t in terms if t.lower() != term.lower()]
                    related_terms.extend(other_terms[:2])
                    break
        
        return related_terms[:3]
    
    def _extract_covered_topics(self, 
                               documents: List[Tuple[Document, float]]) -> set:
        """Extract topics covered in retrieved documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Set of covered topics
        """
        covered_topics = set()
        
        for doc, _ in documents[:3]:  # Check top 3 documents
            doc_topics = self._extract_topics_from_text(doc.page_content)
            covered_topics.update(doc_topics)
        
        return covered_topics
    
    def _calculate_refinement_confidence(self, 
                                        context_analysis: Dict[str, Any],
                                        gap_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for query refinement.
        
        Args:
            context_analysis: Context analysis results
            gap_analysis: Gap analysis results
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from context quality
        base_confidence = context_analysis.get('confidence', 0.0)
        
        # Adjust based on gap identification
        gap_count = gap_analysis.get('gap_count', 0)
        gap_penalty = min(gap_count * 0.1, 0.3)  # Max 30% penalty
        
        # Adjust based on document count
        doc_count = context_analysis.get('document_count', 0)
        doc_bonus = min(doc_count * 0.05, 0.2)  # Max 20% bonus
        
        final_confidence = max(0.0, min(1.0, base_confidence - gap_penalty + doc_bonus))
        
        return final_confidence
    
    def get_refinement_statistics(self) -> Dict[str, Any]:
        """Get statistics about query refinement operations.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            'max_query_length': self.max_query_length,
            'actuarial_terms_count': sum(len(terms) for terms in self.actuarial_terms.values()),
            'llm_available': self.llm is not None,
            'query_expansion_service_available': self.query_expansion_service is not None,
            'refinement_chains_available': hasattr(self, 'context_refinement_chain')
        }