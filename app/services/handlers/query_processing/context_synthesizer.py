"""Context Synthesizer for Multihop Retrieval Results.

This module synthesizes and combines context from multiple retrieval hops
to create coherent and comprehensive responses.
"""

import logging
import time
from typing import Dict, List, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel

logger = logging.getLogger(__name__)

class ContextSynthesizer:
    """Synthesizes context from multihop retrieval results.
    
    This class combines documents from multiple retrieval hops into
    a coherent context that can be used for response generation.
    """
    
    def __init__(self, 
                 llm: BaseLanguageModel,
                 max_context_length: int = 4000,
                 overlap_threshold: float = 0.8):
        """Initialize ContextSynthesizer.
        
        Args:
            llm: Language model for synthesis
            max_context_length: Maximum length of synthesized context
            overlap_threshold: Threshold for detecting content overlap
        """
        self.llm = llm
        self.max_context_length = max_context_length
        self.overlap_threshold = overlap_threshold
        
        self._init_synthesis_chains()
        logger.info(f"ContextSynthesizer initialized with max_length={max_context_length}")
        
    def _init_synthesis_chains(self) -> None:
        """Initialize synthesis chains."""
        try:
            # Chain for synthesizing context
            synthesis_prompt = PromptTemplate(
                input_variables=["query", "hop_contexts", "document_count"],
                template="""Anda adalah ahli aktuaria yang menggabungkan informasi dari berbagai sumber.

Pertanyaan: {query}
Jumlah dokumen: {document_count}

Informasi dari berbagai sumber:
{hop_contexts}

Tugas Anda:
1. Gabungkan informasi dari semua sumber menjadi konteks yang koheren
2. Hilangkan duplikasi dan informasi yang bertentangan
3. Prioritaskan informasi yang paling relevan dengan pertanyaan
4. Organisir informasi secara logis
5. Maksimal {max_length} karakter

Konteks yang disintesis:"""
            )
            
            self.synthesis_chain = LLMChain(
                llm=self.llm,
                prompt=synthesis_prompt,
                verbose=False
            )
            
            # Chain for summarizing individual hops
            hop_summary_prompt = PromptTemplate(
                input_variables=["hop_number", "documents", "query"],
                template="""Ringkas informasi dari hop {hop_number} yang relevan dengan pertanyaan: {query}

Dokumen dari hop ini:
{documents}

Buat ringkasan yang:
1. Fokus pada informasi yang relevan dengan pertanyaan
2. Maksimal 300 karakter
3. Menggunakan bahasa yang jelas dan teknis

Ringkasan hop {hop_number}:"""
            )
            
            self.hop_summary_chain = LLMChain(
                llm=self.llm,
                prompt=hop_summary_prompt,
                verbose=False
            )
            
            logger.info("Synthesis chains initialized")
            
        except Exception as e:
            logger.error(f"Error initializing synthesis chains: {str(e)}")
            raise
    
    def synthesize_multihop_context(self, 
                                   multihop_result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize context from multihop retrieval results.
        
        Args:
            multihop_result: Results from MultihopRetrievalManager
            
        Returns:
            Dictionary containing synthesized context
        """
        start_time = time.time()
        
        try:
            if not multihop_result.get('success', False):
                logger.warning("Cannot synthesize context from failed multihop retrieval")
                return {
                    'success': False,
                    'error': 'Multihop retrieval failed',
                    'synthesis_time': time.time() - start_time
                }
            
            query = multihop_result['original_query']
            hop_results = multihop_result['hop_results']
            final_documents = multihop_result['final_documents']
            
            logger.info(f"Synthesizing context from {len(hop_results)} hops, "
                       f"{len(final_documents)} final documents")
            
            # Create hop summaries
            hop_summaries = self._create_hop_summaries(hop_results, query)
            
            # Detect and resolve conflicts
            resolved_context = self._resolve_context_conflicts(hop_summaries, final_documents)
            
            # Synthesize final context
            synthesized_context = self._synthesize_final_context(
                query=query,
                hop_summaries=resolved_context,
                document_count=len(final_documents)
            )
            
            # Create context metadata
            context_metadata = self._create_context_metadata(
                hop_results, final_documents, synthesized_context
            )
            
            synthesis_time = time.time() - start_time
            
            result = {
                'success': True,
                'synthesized_context': synthesized_context,
                'context_metadata': context_metadata,
                'hop_summaries': hop_summaries,
                'source_documents': final_documents,
                'synthesis_time': synthesis_time,
                'context_length': len(synthesized_context),
                'original_query': query
            }
            
            logger.info(f"Context synthesis completed: {len(synthesized_context)} chars, "
                       f"{synthesis_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in context synthesis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'synthesis_time': time.time() - start_time,
                'original_query': multihop_result.get('original_query', '')
            }
    
    def _create_hop_summaries(self, 
                             hop_results: List[Dict[str, Any]],
                             query: str) -> List[Dict[str, Any]]:
        """Create summaries for each hop.
        
        Args:
            hop_results: Results from each hop
            query: Original query
            
        Returns:
            List of hop summaries
        """
        hop_summaries = []
        
        for hop_result in hop_results:
            try:
                hop_number = hop_result['hop_number']
                documents = hop_result.get('documents', [])
                
                if not documents:
                    hop_summaries.append({
                        'hop_number': hop_number,
                        'summary': f"Hop {hop_number}: Tidak ada dokumen ditemukan",
                        'document_count': 0,
                        'confidence': 0.0
                    })
                    continue
                
                # Create document text for summarization
                doc_texts = []
                for i, (doc, score) in enumerate(documents[:3], 1):  # Top 3 docs per hop
                    filename = doc.metadata.get('filename', 'Unknown')
                    content_preview = doc.page_content[:300]
                    doc_texts.append(f"{i}. [{filename}] {content_preview}")
                
                documents_text = "\n".join(doc_texts)
                
                # Generate summary using LLM
                try:
                    summary = self.hop_summary_chain.run(
                        hop_number=hop_number,
                        documents=documents_text,
                        query=query
                    )
                except Exception as e:
                    logger.warning(f"LLM summarization failed for hop {hop_number}: {str(e)}")
                    # Fallback to simple summary
                    summary = f"Hop {hop_number}: Ditemukan {len(documents)} dokumen terkait {query[:50]}..."
                
                avg_confidence = sum(score for _, score in documents) / len(documents)
                
                hop_summaries.append({
                    'hop_number': hop_number,
                    'summary': summary.strip(),
                    'document_count': len(documents),
                    'confidence': avg_confidence,
                    'query_used': hop_result.get('query', query)
                })
                
            except Exception as e:
                logger.error(f"Error creating summary for hop {hop_result.get('hop_number', '?')}: {str(e)}")
                hop_summaries.append({
                    'hop_number': hop_result.get('hop_number', 0),
                    'summary': f"Error dalam hop: {str(e)}",
                    'document_count': 0,
                    'confidence': 0.0
                })
        
        return hop_summaries
    
    def _resolve_context_conflicts(self, 
                                  hop_summaries: List[Dict[str, Any]],
                                  final_documents: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """Resolve conflicts between different hop contexts.
        
        Args:
            hop_summaries: Summaries from each hop
            final_documents: Final ranked documents
            
        Returns:
            Resolved context summaries
        """
        # For now, implement simple conflict resolution
        # In the future, this could use more sophisticated NLP techniques
        
        resolved_summaries = []
        seen_concepts = set()
        
        # Sort hops by confidence
        sorted_hops = sorted(hop_summaries, key=lambda x: x['confidence'], reverse=True)
        
        for hop_summary in sorted_hops:
            summary_text = hop_summary['summary'].lower()
            
            # Simple duplicate detection based on key terms
            key_terms = self._extract_key_terms(summary_text)
            
            # Check for significant overlap with existing summaries
            overlap_ratio = len(key_terms.intersection(seen_concepts)) / len(key_terms) if key_terms else 0
            
            if overlap_ratio < self.overlap_threshold:
                resolved_summaries.append(hop_summary)
                seen_concepts.update(key_terms)
                logger.debug(f"Added hop {hop_summary['hop_number']} summary (overlap: {overlap_ratio:.2f})")
            else:
                logger.debug(f"Skipped hop {hop_summary['hop_number']} summary due to high overlap ({overlap_ratio:.2f})")
        
        return resolved_summaries
    
    def _extract_key_terms(self, text: str) -> set:
        """Extract key terms from text for conflict detection.
        
        Args:
            text: Text to extract terms from
            
        Returns:
            Set of key terms
        """
        # Simple keyword extraction
        # In production, this could use more sophisticated NLP
        
        import re
        
        # Remove common words and extract meaningful terms
        common_words = {'dan', 'atau', 'yang', 'dari', 'untuk', 'dengan', 'pada', 'dalam', 'adalah', 'akan', 'dapat', 'harus', 'jika', 'maka', 'ini', 'itu', 'tersebut'}
        
        # Extract words (3+ characters)
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Filter out common words
        key_terms = {word for word in words if word not in common_words}
        
        return key_terms
    
    def _create_context_summary(self, documents: List[Tuple[Document, float]]) -> str:
        """Create a summary of the context from documents.
        
        Args:
            documents: List of (document, score) tuples
            
        Returns:
            Context summary string
        """
        try:
            if not documents:
                return "Tidak ada konteks yang ditemukan."
            
            summary_parts = []
            
            for doc, score in documents[:5]:  # Top 5 documents
                filename = doc.metadata.get('filename', 'Unknown')
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                
                summary_parts.append(f"Dokumen: {filename}\nKonten: {content_preview}\nSkor: {score:.2f}\n")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating context summary: {e}")
            return "Error dalam membuat ringkasan konteks."
    
    def _synthesize_final_context(self, 
                                 query: str,
                                 hop_summaries: List[Dict[str, Any]],
                                 document_count: int) -> str:
        """Synthesize final context from hop summaries.
        
        Args:
            query: Original query
            hop_summaries: Resolved hop summaries
            document_count: Total number of documents
            
        Returns:
            Synthesized context string
        """
        try:
            # Combine hop summaries
            hop_contexts = []
            for hop_summary in hop_summaries:
                hop_contexts.append(
                    f"Hop {hop_summary['hop_number']} (confidence: {hop_summary['confidence']:.2f}): "
                    f"{hop_summary['summary']}"
                )
            
            hop_contexts_text = "\n\n".join(hop_contexts)
            
            # Generate synthesized context using LLM
            try:
                synthesized = self.synthesis_chain.run(
                    query=query,
                    hop_contexts=hop_contexts_text,
                    document_count=document_count,
                    max_length=self.max_context_length
                )
                
                # Ensure context doesn't exceed max length
                if len(synthesized) > self.max_context_length:
                    synthesized = synthesized[:self.max_context_length - 3] + "..."
                
                return synthesized.strip()
                
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {str(e)}")
                # Fallback to simple concatenation
                fallback_context = f"Berdasarkan {document_count} dokumen dari {len(hop_summaries)} tahap pencarian:\n\n"
                fallback_context += "\n\n".join([hs['summary'] for hs in hop_summaries])
                
                if len(fallback_context) > self.max_context_length:
                    fallback_context = fallback_context[:self.max_context_length - 3] + "..."
                
                return fallback_context
                
        except Exception as e:
            logger.error(f"Error in final synthesis: {str(e)}")
            return f"Error dalam sintesis konteks: {str(e)}"
    
    def _create_context_metadata(self, 
                                hop_results: List[Dict[str, Any]],
                                final_documents: List[Tuple[Document, float]],
                                synthesized_context: str) -> Dict[str, Any]:
        """Create metadata for synthesized context.
        
        Args:
            hop_results: Results from each hop
            final_documents: Final ranked documents
            synthesized_context: Synthesized context text
            
        Returns:
            Context metadata dictionary
        """
        # Extract source information
        source_files = set()
        total_confidence = 0.0
        
        for doc, score in final_documents:
            filename = doc.metadata.get('filename', 'Unknown')
            source_files.add(filename)
            total_confidence += score
        
        avg_confidence = total_confidence / len(final_documents) if final_documents else 0.0
        
        # Calculate hop statistics
        hop_stats = {
            'total_hops': len(hop_results),
            'successful_hops': len([hr for hr in hop_results if hr.get('document_count', 0) > 0]),
            'total_documents_retrieved': sum(hr.get('document_count', 0) for hr in hop_results),
            'final_documents_count': len(final_documents)
        }
        
        return {
            'source_files': list(source_files),
            'source_count': len(source_files),
            'average_confidence': avg_confidence,
            'context_length': len(synthesized_context),
            'hop_statistics': hop_stats,
            'synthesis_method': 'llm_with_fallback'
        }
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get statistics about synthesis operations.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            'max_context_length': self.max_context_length,
            'overlap_threshold': self.overlap_threshold,
            'llm_available': self.llm is not None,
            'synthesis_chain_available': hasattr(self, 'synthesis_chain'),
            'hop_summary_chain_available': hasattr(self, 'hop_summary_chain')
        }