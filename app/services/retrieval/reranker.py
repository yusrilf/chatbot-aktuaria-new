"""Reranking System for CoT Retrieval.

This module implements cross-encoder and LLM-based reranking to improve
chunk relevance scoring for better context assembly.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import logging
import json
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain.llms.base import LLM

from app.services.retrieval.semantic_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RerankingScore:
    """Container for reranking score and explanation."""
    score: float
    explanation: str
    confidence: float
    chunk_index: int


@dataclass
class RerankingResult:
    """Result from reranking process."""
    reranked_chunks: List[RetrievedChunk]
    scores: List[RerankingScore]
    reranking_method: str
    original_count: int
    final_count: int
    metadata: Dict[str, Any]


class ChunkReranker:
    """Reranks retrieved chunks using cross-encoder or LLM scoring."""
    
    def __init__(self, llm: Optional[LLM] = None, use_cross_encoder: bool = False):
        """
        Initialize the reranker.
        
        Args:
            llm: Language model for LLM-based reranking
            use_cross_encoder: Whether to use cross-encoder (requires additional setup)
        """
        self.llm = llm
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder = None
        
        # Initialize cross-encoder if requested
        if use_cross_encoder:
            self._initialize_cross_encoder()
        
        logger.info(f"ChunkReranker initialized with method: {'cross_encoder' if use_cross_encoder else 'llm'}")
    
    def _initialize_cross_encoder(self):
        """Initialize cross-encoder model (optional dependency)."""
        try:
            # This would require sentence-transformers library
            # from sentence_transformers import CrossEncoder
            # self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.warning("Cross-encoder not implemented - falling back to LLM reranking")
            self.use_cross_encoder = False
        except ImportError:
            logger.warning("sentence-transformers not available - using LLM reranking")
            self.use_cross_encoder = False
    
    def rerank_chunks(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
        min_score_threshold: float = 0.01  # Very low threshold for debugging
    ) -> RerankingResult:
        """
        Rerank chunks based on relevance to query.
        
        Args:
            query: User query
            chunks: List of retrieved chunks to rerank
            top_k: Number of top chunks to return
            min_score_threshold: Minimum score threshold for inclusion
            
        Returns:
            RerankingResult with reranked chunks
        """
        try:
            print(f"RERANKER CALLED: Query={query[:50]}..., Chunks={len(chunks)}")
            logger.error(f"=== RERANKING DEBUG START ===")
            logger.error(f"Query: {query}")
            logger.error(f"Input chunks count: {len(chunks)}")
            logger.error(f"Top K: {top_k}, Min threshold: {min_score_threshold}")
            
            # Debug chunk types and structure
            for i, chunk in enumerate(chunks[:3]):  # Only show first 3 to avoid spam
                logger.error(f"Chunk {i} type: {type(chunk)}")
                if hasattr(chunk, 'content'):
                    logger.error(f"Chunk {i} content preview: {chunk.content[:100]}...")
                elif hasattr(chunk, 'page_content'):
                    logger.error(f"Chunk {i} page_content preview: {chunk.page_content[:100]}...")
                else:
                    logger.error(f"Chunk {i} value: {str(chunk)[:100]}...")
            
            if not chunks:
                logger.error("No chunks to rerank - returning empty result")
                return self._create_empty_result("No chunks to rerank")
            
            # Choose reranking method
            if self.use_cross_encoder and self.cross_encoder:
                logger.error("Using cross-encoder reranking")
                result = self._rerank_with_cross_encoder(query, chunks, top_k, min_score_threshold)
            elif self.llm:
                logger.error("Using LLM reranking")
                result = self._rerank_with_llm(query, chunks, top_k, min_score_threshold)
            else:
                logger.info("Using simple scoring reranking")
                result = self._rerank_with_simple_scoring(query, chunks, top_k, min_score_threshold)
            
            logger.info(f"Reranking method used: {result.reranking_method}")
            logger.info(f"Final result - chunks: {len(result.reranked_chunks)}, scores: {len(result.scores)}")
            logger.info(f"=== RERANKING DEBUG END ===")
            return result
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            return self._create_error_result(chunks, str(e))
    
    def _rerank_with_cross_encoder(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int,
        min_score_threshold: float
    ) -> RerankingResult:
        """Rerank using cross-encoder model."""
        try:
            logger.info("Using cross-encoder for reranking")
            
            # Prepare query-chunk pairs with safe content extraction
            pairs = []
            for chunk in chunks:
                try:
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                    elif hasattr(chunk, 'page_content'):
                        content = chunk.page_content
                    elif isinstance(chunk, str):
                        content = chunk
                    elif isinstance(chunk, dict):
                        content = chunk.get('content', chunk.get('page_content', str(chunk)))
                    else:
                        content = str(chunk)
                    pairs.append((query, content))
                except Exception as e:
                    logger.warning(f"Error extracting content from chunk: {e}")
                    pairs.append((query, str(chunk)))
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Create reranking scores
            reranking_scores = []
            for i, score in enumerate(scores):
                reranking_scores.append(RerankingScore(
                    score=float(score),
                    explanation=f"Cross-encoder relevance score: {score:.3f}",
                    confidence=min(float(score), 1.0),
                    chunk_index=i
                ))
            
            # Sort by score and apply thresholds
            scored_chunks = list(zip(chunks, reranking_scores))
            scored_chunks.sort(key=lambda x: x[1].score, reverse=True)
            
            # Filter by threshold and top_k
            filtered_chunks = [
                (chunk, score) for chunk, score in scored_chunks
                if score.score >= min_score_threshold
            ][:top_k]
            
            logger.info(f"Simple scoring: {len(filtered_chunks)}/{len(chunks)} chunks passed threshold {min_score_threshold}")
            
            reranked_chunks = [chunk for chunk, _ in filtered_chunks]
            final_scores = [score for _, score in filtered_chunks]
            
            return RerankingResult(
                reranked_chunks=reranked_chunks,
                scores=final_scores,
                reranking_method="cross_encoder",
                original_count=len(chunks),
                final_count=len(reranked_chunks),
                metadata={
                    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "threshold": min_score_threshold,
                    "top_k": top_k
                }
            )
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            return self._rerank_with_llm(query, chunks, top_k, min_score_threshold)
    
    def _rerank_with_llm(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int,
        min_score_threshold: float
    ) -> RerankingResult:
        """Rerank using LLM scoring."""
        try:
            logger.info("Using LLM for reranking")
            
            # Prepare chunks for LLM evaluation
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                # Truncate content for LLM processing
                # Safe content extraction for preview
                try:
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                    elif hasattr(chunk, 'page_content'):
                        content = chunk.page_content
                    elif isinstance(chunk, str):
                        content = chunk
                    elif isinstance(chunk, dict):
                        content = chunk.get('content', chunk.get('page_content', str(chunk)))
                    else:
                        content = str(chunk)
                    
                    content_preview = content[:300] + "..." if len(content) > 300 else content
                except Exception as e:
                    logger.warning(f"Error extracting content for preview: {e}")
                    content_preview = str(chunk)[:300]
                chunk_summaries.append({
                    "index": i,
                    "document": chunk.document_name,
                    "section": chunk.section_heading or "Unknown",
                    "content_preview": content_preview,
                    "original_score": chunk.score
                })
            
            # Create LLM prompt for reranking
            prompt = self._create_reranking_prompt(query, chunk_summaries)
            
            # Get LLM response
            llm_response = self.llm(prompt)
            
            # Handle different response types
            if hasattr(llm_response, 'content'):
                response_text = llm_response.content
            elif isinstance(llm_response, str):
                response_text = llm_response
            else:
                response_text = str(llm_response)
            
            # Parse LLM response
            reranking_scores = self._parse_llm_reranking_response(response_text, len(chunks))
            
            # Apply reranking
            scored_chunks = list(zip(chunks, reranking_scores))
            scored_chunks.sort(key=lambda x: x[1].score, reverse=True)
            
            # Filter by threshold and top_k
            filtered_chunks = [
                (chunk, score) for chunk, score in scored_chunks
                if score.score >= min_score_threshold
            ][:top_k]
            
            reranked_chunks = [chunk for chunk, _ in filtered_chunks]
            final_scores = [score for _, score in filtered_chunks]
            
            return RerankingResult(
                reranked_chunks=reranked_chunks,
                scores=final_scores,
                reranking_method="llm",
                original_count=len(chunks),
                final_count=len(reranked_chunks),
                metadata={
                    "llm_model": str(type(self.llm).__name__),
                    "threshold": min_score_threshold,
                    "top_k": top_k,
                    "prompt_length": len(prompt)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in LLM reranking: {str(e)}")
            logger.info(f"Falling back to simple scoring for {len(chunks)} chunks")
            return self._rerank_with_simple_scoring(query, chunks, top_k, min_score_threshold)
    
    def _rerank_with_simple_scoring(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int,
        min_score_threshold: float
    ) -> RerankingResult:
        """Fallback reranking using simple text matching."""
        try:
            logger.info(f"Using simple scoring for reranking {len(chunks)} chunks")
            query_terms = set(query.lower().split())
            logger.info(f"Query terms: {query_terms}")
            
            reranking_scores = []
            
            for i, chunk in enumerate(chunks):
                # Simple relevance scoring
                # Safe content extraction for term analysis
                try:
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                    elif hasattr(chunk, 'page_content'):
                        content = chunk.page_content
                    elif isinstance(chunk, str):
                        content = chunk
                    elif isinstance(chunk, dict):
                        content = chunk.get('content', chunk.get('page_content', str(chunk)))
                    else:
                        content = str(chunk)
                    
                    content_terms = set(content.lower().split())
                    logger.info(f"Chunk {i} content preview: {content[:100]}...")
                    logger.info(f"Chunk {i} full content: {content}")  # Add full content logging
                except Exception as e:
                    logger.warning(f"Error extracting content for analysis: {e}")
                    content_terms = set()
                    logger.info(f"Chunk {i} content preview: {str(chunk)[:100]}...")
                logger.info(f"Chunk {i} content terms (first 10): {list(content_terms)[:10]}")
                overlap = len(query_terms.intersection(content_terms))
                total_terms = len(query_terms)
                
                # Calculate relevance score
                relevance_score = overlap / total_terms if total_terms > 0 else 0
                
                # Combine with original score
                combined_score = (chunk.score * 0.7) + (relevance_score * 0.3)
                
                logger.info(f"Chunk {i}: original={chunk.score:.3f}, relevance={relevance_score:.3f}, combined={combined_score:.3f}")
                logger.info(f"Chunk {i}: overlap={overlap}, total_terms={total_terms}")
                
                reranking_scores.append(RerankingScore(
                    score=combined_score,
                    explanation=f"Term overlap: {overlap}/{total_terms}, combined score: {combined_score:.3f}",
                    confidence=0.6,  # Lower confidence for simple method
                    chunk_index=i
                ))
            
            # Sort and filter
            scored_chunks = list(zip(chunks, reranking_scores))
            scored_chunks.sort(key=lambda x: x[1].score, reverse=True)
            
            logger.info(f"Before filtering: {len(scored_chunks)} chunks, threshold: {min_score_threshold}")
            
            filtered_chunks = [
                (chunk, score) for chunk, score in scored_chunks
                if score.score >= min_score_threshold
            ][:top_k]
            
            logger.info(f"After filtering: {len(filtered_chunks)} chunks passed threshold")
            
            reranked_chunks = [chunk for chunk, _ in filtered_chunks]
            final_scores = [score for _, score in filtered_chunks]
            
            logger.info(f"Reranking completed: {len(reranked_chunks)} chunks selected")
            
            return RerankingResult(
                reranked_chunks=reranked_chunks,
                scores=final_scores,
                reranking_method="simple_scoring",
                original_count=len(chunks),
                final_count=len(reranked_chunks),
                metadata={
                    "method": "term_overlap",
                    "threshold": min_score_threshold,
                    "top_k": top_k
                }
            )
            
        except Exception as e:
            logger.error(f"Error in simple scoring: {str(e)}")
            return self._create_error_result(chunks, str(e))
    
    def _create_reranking_prompt(self, query: str, chunk_summaries: List[Dict]) -> str:
        """Create prompt for LLM-based reranking."""
        prompt = f"""You are a relevance scoring assistant for actuarial documents. 
Given a user query and document chunks, score each chunk's relevance from 0.0 to 1.0.

User Query: "{query}"

Document Chunks:
"""
        
        for chunk in chunk_summaries:
            prompt += f"""
Chunk {chunk['index']}:
- Document: {chunk['document']}
- Section: {chunk['section']}
- Content Preview: {chunk['content_preview']}
- Original Score: {chunk['original_score']:.3f}
"""
        
        prompt += """
Return JSON with format:
{
  "scores": [
    {"index": 0, "score": 0.85, "explanation": "Highly relevant because..."},
    {"index": 1, "score": 0.42, "explanation": "Partially relevant because..."}
  ]
}

Consider:
1. Direct relevance to query terms
2. Actuarial context and domain specificity
3. Technical accuracy and completeness
4. Document type and authority

Output only valid JSON."""
        
        return prompt
    
    def _parse_llm_reranking_response(
        self, 
        response: str, 
        num_chunks: int
    ) -> List[RerankingScore]:
        """Parse LLM response for reranking scores."""
        try:
            # Try to parse JSON response
            response_data = json.loads(response.strip())
            scores_data = response_data.get("scores", [])
            
            # Create reranking scores
            reranking_scores = [None] * num_chunks
            
            for score_data in scores_data:
                index = score_data.get("index", -1)
                if 0 <= index < num_chunks:
                    reranking_scores[index] = RerankingScore(
                        score=float(score_data.get("score", 0.0)),
                        explanation=score_data.get("explanation", "No explanation"),
                        confidence=0.8,  # High confidence for LLM scoring
                        chunk_index=index
                    )
            
            # Fill missing scores with default
            for i in range(num_chunks):
                if reranking_scores[i] is None:
                    reranking_scores[i] = RerankingScore(
                        score=0.3,  # Default low score
                        explanation="No LLM score provided",
                        confidence=0.2,
                        chunk_index=i
                    )
            
            return reranking_scores
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing LLM reranking response: {str(e)}")
            
            # Fallback: create default scores
            return [
                RerankingScore(
                    score=0.5,
                    explanation="LLM parsing failed - using default score",
                    confidence=0.1,
                    chunk_index=i
                )
                for i in range(num_chunks)
            ]
    
    def _create_empty_result(self, reason: str) -> RerankingResult:
        """Create empty reranking result."""
        return RerankingResult(
            reranked_chunks=[],
            scores=[],
            reranking_method="none",
            original_count=0,
            final_count=0,
            metadata={"reason": reason}
        )
    
    def _create_error_result(self, chunks: List[RetrievedChunk], error: str) -> RerankingResult:
        """Create error result with original chunks."""
        return RerankingResult(
            reranked_chunks=chunks,  # Return original chunks on error
            scores=[
                RerankingScore(
                    score=chunk.score,
                    explanation=f"Error in reranking: {error}",
                    confidence=0.1,
                    chunk_index=i
                )
                for i, chunk in enumerate(chunks)
            ],
            reranking_method="error_fallback",
            original_count=len(chunks),
            final_count=len(chunks),
            metadata={"error": error}
        )
    
    def get_reranking_statistics(self, result: RerankingResult) -> Dict[str, Any]:
        """Get statistics about reranking performance."""
        try:
            if not result.scores:
                return {"error": "No scores available"}
            
            scores = [score.score for score in result.scores]
            
            stats = {
                "method": result.reranking_method,
                "original_count": result.original_count,
                "final_count": result.final_count,
                "reduction_ratio": (result.original_count - result.final_count) / result.original_count if result.original_count > 0 else 0,
                "average_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "score_distribution": {
                    "high (>0.7)": len([s for s in scores if s > 0.7]),
                    "medium (0.4-0.7)": len([s for s in scores if 0.4 <= s <= 0.7]),
                    "low (<0.4)": len([s for s in scores if s < 0.4])
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating reranking statistics: {str(e)}")
            return {"error": str(e)}