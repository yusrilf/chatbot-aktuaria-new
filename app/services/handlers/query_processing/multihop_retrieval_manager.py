"""Multihop Retrieval Manager for Advanced Document Search.

This module implements multihop retrieval functionality that performs
iterative document retrieval with query refinement between hops.
"""

import logging
import time
from typing import Dict, List, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel

logger = logging.getLogger(__name__)

class MultihopRetrievalManager:
    """Manager untuk multihop retrieval dengan iterative refinement.
    
    This class implements a multihop retrieval strategy where:
    1. Initial query retrieves documents
    2. Query is refined based on retrieved context
    3. Refined query retrieves additional documents
    4. Process repeats until stopping criteria are met
    """
    
    def __init__(self, 
                 retrieval_manager,
                 query_expansion_service,
                 llm: BaseLanguageModel,
                 max_hops: int = 3,
                 confidence_threshold: float = 0.75,
                 min_docs_per_hop: int = 3,
                 max_docs_per_hop: int = 8):
        """Initialize MultihopRetrievalManager.
        
        Args:
            retrieval_manager: Existing retrieval manager
            query_expansion_service: Query expansion service
            llm: Language model for query refinement
            max_hops: Maximum number of retrieval hops
            confidence_threshold: Minimum confidence to continue hopping
            min_docs_per_hop: Minimum documents required per hop
            max_docs_per_hop: Maximum documents to retrieve per hop
        """
        self.retrieval_manager = retrieval_manager
        self.query_expansion_service = query_expansion_service
        self.llm = llm
        self.max_hops = max_hops
        self.confidence_threshold = confidence_threshold
        self.min_docs_per_hop = min_docs_per_hop
        self.max_docs_per_hop = max_docs_per_hop
        
        self._init_refinement_chain()
        logger.info(f"MultihopRetrievalManager initialized with max_hops={max_hops}")
        
    def _init_refinement_chain(self) -> None:
        """Initialize query refinement chain."""
        try:
            refinement_prompt = PromptTemplate(
                input_variables=["original_query", "retrieved_context", "hop_number"],
                template="""Anda adalah ahli aktuaria yang membantu memperbaiki query pencarian.

Query Asli: {original_query}
Hop ke-: {hop_number}

Konteks yang sudah ditemukan:
{retrieved_context}

Berdasarkan konteks di atas, buat query pencarian yang lebih spesifik untuk menemukan informasi tambahan yang masih kurang. Query baru harus:
1. Fokus pada aspek yang belum tercakup dalam konteks
2. Menggunakan istilah teknis aktuaria yang relevan
3. Mencari informasi pelengkap atau detail yang lebih mendalam
4. Maksimal 20 kata

Query yang diperbaiki:"""
            )
            
            self.refinement_chain = LLMChain(
                llm=self.llm,
                prompt=refinement_prompt,
                verbose=False
            )
            
            logger.info("Query refinement chain initialized")
            
        except Exception as e:
            logger.error(f"Error initializing refinement chain: {str(e)}")
            raise
    
    def multihop_retrieve(self, 
                         original_query: str,
                         session_id: str,
                         context_type: str = "aktuaria") -> Dict[str, Any]:
        """Perform multihop retrieval.
        
        Args:
            original_query: Original user query
            session_id: Session identifier
            context_type: Context type for query expansion
            
        Returns:
            Dictionary containing multihop retrieval results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting multihop retrieval for query: {original_query[:50]}...")
            
            hop_results = []
            current_query = original_query
            accumulated_docs = []
            seen_content_hashes = set()
            
            for hop in range(self.max_hops):
                logger.info(f"Performing hop {hop + 1}/{self.max_hops}")
                
                # Perform single hop retrieval
                hop_result = self._perform_single_hop(
                    query=current_query,
                    session_id=session_id,
                    hop_number=hop + 1,
                    previous_context=accumulated_docs
                )
                
                hop_results.append(hop_result)
                
                # Add new documents to accumulated results (with deduplication)
                new_docs = self._deduplicate_documents(
                    hop_result['documents'], 
                    seen_content_hashes
                )
                accumulated_docs.extend(new_docs)
                
                logger.info(f"Hop {hop + 1} retrieved {len(new_docs)} new documents")
                
                # Check stopping criteria
                if self._should_stop_hopping(hop_result, accumulated_docs, hop + 1):
                    logger.info(f"Stopping criteria met after hop {hop + 1}")
                    break
                    
                # Refine query for next hop (if not last hop)
                if hop < self.max_hops - 1:
                    current_query = self._refine_query_for_next_hop(
                        original_query=original_query,
                        current_results=accumulated_docs,
                        hop_number=hop + 1
                    )
                    logger.info(f"Refined query for hop {hop + 2}: {current_query[:50]}...")
            
            # Final ranking and filtering
            final_documents = self._rank_and_filter_final_documents(
                accumulated_docs, 
                original_query
            )
            
            total_time = time.time() - start_time
            
            result = {
                'success': True,
                'original_query': original_query,
                'total_hops': len(hop_results),
                'hop_results': hop_results,
                'final_documents': final_documents,
                'total_documents': len(final_documents),
                'confidence_score': self._calculate_overall_confidence(hop_results),
                'retrieval_time': total_time,
                'session_id': session_id
            }
            
            logger.info(f"Multihop retrieval completed: {len(hop_results)} hops, "
                       f"{len(final_documents)} final documents, {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multihop retrieval: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_query': original_query,
                'session_id': session_id,
                'retrieval_time': time.time() - start_time
            }
    
    def _perform_single_hop(self, 
                           query: str,
                           session_id: str,
                           hop_number: int,
                           previous_context: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """Perform a single hop of retrieval.
        
        Args:
            query: Query for this hop
            session_id: Session identifier
            hop_number: Current hop number
            previous_context: Documents from previous hops
            
        Returns:
            Dictionary containing hop results
        """
        try:
            # Generate retrieval plan
            plan = self.retrieval_manager.generate_retrieval_plan(
                question=query,
                session_id=session_id,
                include_global=True
            )
            
            # Adjust k based on hop number (fewer docs in later hops)
            k = max(self.min_docs_per_hop, self.max_docs_per_hop - (hop_number - 1) * 2)
            
            # Update plan with adjusted k
            for strategy in plan.get('search_strategies', []):
                strategy['k'] = k
            
            # Retrieve documents
            documents = self.retrieval_manager.retrieve_documents(
                plan=plan,
                session_id=session_id
            )
            
            # Calculate hop-specific metrics
            avg_confidence = sum(score for _, score in documents) / len(documents) if documents else 0.0
            
            return {
                'hop_number': hop_number,
                'query': query,
                'documents': documents,
                'document_count': len(documents),
                'average_confidence': avg_confidence,
                'retrieval_plan': plan
            }
            
        except Exception as e:
            logger.error(f"Error in hop {hop_number}: {str(e)}")
            return {
                'hop_number': hop_number,
                'query': query,
                'documents': [],
                'document_count': 0,
                'average_confidence': 0.0,
                'error': str(e)
            }
    
    def _should_stop_hopping(self, 
                            hop_result: Dict[str, Any],
                            accumulated_docs: List[Tuple[Document, float]],
                            current_hop: int) -> bool:
        """Determine if we should stop hopping.
        
        Args:
            hop_result: Result from current hop
            accumulated_docs: All documents accumulated so far
            current_hop: Current hop number
            
        Returns:
            True if should stop hopping
        """
        # Stop if we've reached max hops
        if current_hop >= self.max_hops:
            return True
            
        # Stop if current hop found no documents
        if hop_result['document_count'] == 0:
            logger.info("Stopping: No documents found in current hop")
            return True
            
        # Stop if we have enough high-confidence documents
        high_conf_docs = [doc for doc, score in accumulated_docs 
                         if score >= self.confidence_threshold]
        if len(high_conf_docs) >= 8:  # Sufficient high-quality documents
            logger.info(f"Stopping: Found {len(high_conf_docs)} high-confidence documents")
            return True
            
        # Stop if average confidence in current hop is too low
        if hop_result['average_confidence'] < 0.5:
            logger.info(f"Stopping: Low confidence in hop {current_hop}")
            return True
            
        return False
    
    def _refine_query_for_next_hop(self,
                                  original_query: str,
                                  current_results: List[Tuple[Document, float]],
                                  hop_number: int) -> str:
        """Refine query for next hop based on current results.
        
        Args:
            original_query: Original user query
            current_results: Documents retrieved so far
            hop_number: Current hop number
            
        Returns:
            Refined query for next hop
        """
        try:
            # Create context summary from current results
            context_summary = self._create_context_summary(current_results[:5])  # Top 5 docs
            
            # Generate refined query
            refined_query = self.refinement_chain.run(
                original_query=original_query,
                retrieved_context=context_summary,
                hop_number=hop_number + 1
            )
            
            return refined_query.strip()
            
        except Exception as e:
            logger.error(f"Error refining query: {str(e)}")
            # Fallback to original query with slight modification
            return f"{original_query} detail tambahan"
    
    def _create_context_summary(self, documents: List[Tuple[Document, float]]) -> str:
        """Create a summary of retrieved documents for context.
        
        Args:
            documents: List of (document, score) tuples
            
        Returns:
            Context summary string
        """
        if not documents:
            return "Tidak ada konteks yang ditemukan."
            
        summaries = []
        for i, (doc, score) in enumerate(documents[:3], 1):  # Top 3 docs
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            filename = doc.metadata.get('filename', 'Unknown')
            summaries.append(f"{i}. [{filename}] {content_preview}")
            
        return "\n".join(summaries)
    
    def _deduplicate_documents(self, 
                              new_docs: List[Tuple[Document, float]],
                              seen_hashes: set) -> List[Tuple[Document, float]]:
        """Remove duplicate documents based on content hash.
        
        Args:
            new_docs: New documents to check
            seen_hashes: Set of already seen content hashes
            
        Returns:
            List of unique documents
        """
        unique_docs = []
        
        for doc, score in new_docs:
            # Create hash from first 200 characters
            content_hash = hash(doc.page_content[:200])
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append((doc, score))
                
        return unique_docs
    
    def _rank_and_filter_final_documents(self,
                                        documents: List[Tuple[Document, float]],
                                        original_query: str) -> List[Tuple[Document, float]]:
        """Rank and filter final documents.
        
        Args:
            documents: All accumulated documents
            original_query: Original query for relevance scoring
            
        Returns:
            Ranked and filtered documents
        """
        # Sort by score (descending)
        sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
        
        # Take top 10 documents
        final_docs = sorted_docs[:10]
        
        logger.info(f"Final ranking: {len(final_docs)} documents selected from {len(documents)} total")
        
        return final_docs
    
    def _calculate_overall_confidence(self, hop_results: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score from all hops.
        
        Args:
            hop_results: Results from all hops
            
        Returns:
            Overall confidence score (0-1)
        """
        if not hop_results:
            return 0.0
            
        total_confidence = 0.0
        total_docs = 0
        
        for hop_result in hop_results:
            if hop_result.get('documents'):
                hop_confidence = sum(score for _, score in hop_result['documents'])
                total_confidence += hop_confidence
                total_docs += len(hop_result['documents'])
                
        return total_confidence / total_docs if total_docs > 0 else 0.0
    
    def get_multihop_statistics(self) -> Dict[str, Any]:
        """Get statistics about multihop retrieval operations.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            'max_hops': self.max_hops,
            'confidence_threshold': self.confidence_threshold,
            'min_docs_per_hop': self.min_docs_per_hop,
            'max_docs_per_hop': self.max_docs_per_hop,
            'retrieval_manager_available': self.retrieval_manager is not None,
            'query_expansion_available': self.query_expansion_service is not None
        }