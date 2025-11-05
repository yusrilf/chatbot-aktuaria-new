"""Context Assembly System for CoT Retrieval.

This module assembles retrieved and reranked chunks into coherent context,
ordering by section and relevance while prepending README.md guidelines.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.services.retrieval.semantic_retriever import RetrievedChunk
from app.services.retrieval.reranker import RerankingResult

logger = logging.getLogger(__name__)


@dataclass
class ContextSection:
    """Container for a context section."""
    title: str
    content: str
    source_document: str
    section_heading: Optional[str]
    relevance_score: float
    section_order: int
    chunk_count: int


@dataclass
class AssembledContext:
    """Result from context assembly."""
    full_context: str
    sections: List[ContextSection]
    total_length: int
    chunk_count: int
    document_count: int
    has_readme: bool
    assembly_metadata: Dict[str, Any]


class ContextAssembler:
    """Assembles retrieved chunks into coherent context."""
    
    def __init__(self, vector_store_manager=None, search_manager=None):
        """
        Initialize the context assembler.
        
        Args:
            vector_store_manager: Vector store manager for README retrieval
            search_manager: Search manager for README retrieval
        """
        self.vector_store_manager = vector_store_manager
        self.search_manager = search_manager
        self.max_context_length = 8000  # Maximum context length in characters
        self.readme_priority = True
        logger.info("ContextAssembler initialized")
    
    def assemble_context(
        self,
        reranking_result: RerankingResult,
        session_id: Optional[str] = None,
        include_readme: bool = True,
        max_length: Optional[int] = None,
        context_metadata: Optional[Dict[str, Any]] = None
    ) -> AssembledContext:
        """
        Assemble context from reranked chunks with optional metadata optimization.
        
        Args:
            reranking_result: Result from chunk reranking
            session_id: Optional session ID
            include_readme: Whether to include README.md
            max_length: Maximum context length (overrides default)
            context_metadata: Optional metadata for assembly optimization
            
        Returns:
            AssembledContext with organized content
        """
        try:
            logger.info(f"Assembling context from {len(reranking_result.reranked_chunks)} chunks")
            
            max_len = max_length or self.max_context_length
            
            # Use metadata optimization hints if available
            optimization_hints = {}
            if context_metadata and 'optimization_hints' in context_metadata:
                optimization_hints = context_metadata['optimization_hints']
                logger.debug(f"Using optimization hints: {optimization_hints}")
            
            # Get README content if requested
            readme_content = ""
            if include_readme:
                readme_content = self._get_readme_content(session_id)
            
            # Group chunks by document and section
            grouped_chunks = self._group_chunks_by_section(reranking_result.reranked_chunks)
            
            # Create context sections with metadata optimization
            sections = self._create_context_sections(
                grouped_chunks, 
                reranking_result.scores,
                optimization_hints
            )
            
            # Order sections by priority and relevance
            ordered_sections = self._order_sections(sections)
            
            # Assemble final context within length limits
            final_context, included_sections = self._assemble_final_context(
                readme_content, ordered_sections, max_len
            )
            
            # Create result with enhanced metadata
            assembly_metadata = {
                "max_length": max_len,
                "readme_length": len(readme_content),
                "sections_created": len(sections),
                "sections_included": len(included_sections),
                "reranking_method": reranking_result.reranking_method
            }
            
            # Include context metadata if provided
            if context_metadata:
                assembly_metadata.update({
                    "has_optimization_metadata": True,
                    "sequential_context_used": context_metadata.get('has_sequential_context', False),
                    "sequential_themes": context_metadata.get('sequential_themes', [])
                })
            
            result = AssembledContext(
                full_context=final_context,
                sections=included_sections,
                total_length=len(final_context),
                chunk_count=len(reranking_result.reranked_chunks),
                document_count=len(set(chunk.document_name for chunk in reranking_result.reranked_chunks)),
                has_readme=bool(readme_content),
                assembly_metadata=assembly_metadata
            )
            
            logger.info(f"Context assembled: {result.total_length} chars, {len(included_sections)} sections")
            return result
            
        except Exception as e:
            logger.error(f"Error assembling context: {str(e)}")
            return self._create_error_context(str(e))
    
    def _get_readme_content(self, session_id: Optional[str]) -> str:
        """Retrieve README.md content for global guidelines."""
        try:
            logger.debug("Retrieving README.md content")
            
            # Try search manager first
            if self.search_manager:
                try:
                    results = self.search_manager.search_documents(
                        query="README guidelines introduction",
                        session_id=session_id,
                        top_k=3,
                        metadata_filter={"filename": {"$regex": "README"}}
                    )
                    
                    if results:
                        # Combine README chunks
                        readme_chunks = [result.get('content', '') for result in results]
                        return "\n\n".join(readme_chunks)
                        
                except Exception as e:
                    logger.debug(f"Search manager README retrieval failed: {str(e)}")
            
            # Try vector store manager
            if self.vector_store_manager:
                try:
                    results = self.vector_store_manager.similarity_search(
                        query="README introduction guidelines",
                        k=3,
                        filter={"filename": {"$regex": "README"}},
                        session_id=session_id
                    )
                    
                    if results:
                        readme_chunks = [doc.page_content for doc in results]
                        return "\n\n".join(readme_chunks)
                        
                except Exception as e:
                    logger.debug(f"Vector store README retrieval failed: {str(e)}")
            
            # Fallback README content
            logger.debug("Using fallback README content")
            return self._get_fallback_readme()
            
        except Exception as e:
            logger.error(f"Error retrieving README: {str(e)}")
            return ""
    
    def _get_fallback_readme(self) -> str:
        """Provide fallback README content."""
        return """# Sistem Konsultasi Aktuaria

## Panduan Penggunaan
Sistem ini menyediakan konsultasi aktuaria berdasarkan dokumen PSAK 219, PUC, dan panduan teknis.

## Prinsip Jawaban
1. Semua jawaban harus berdasarkan dokumen yang tersedia
2. Cantumkan sumber referensi untuk setiap klaim
3. Jika informasi tidak tersedia, nyatakan dengan jelas
4. Prioritaskan akurasi teknis dan kepatuhan regulasi

## Cakupan Dokumen
- PSAK 219: Standar akuntansi untuk program imbalan kerja
- PUC (Projected Unit Credit): Metode aktuaria
- FAQ dan panduan implementasi
- Dokumen teknis dan regulasi terkait
"""
    
    def _group_chunks_by_section(
        self, 
        chunks: List[RetrievedChunk]
    ) -> Dict[Tuple[str, str], List[RetrievedChunk]]:
        """Group chunks by document and section."""
        try:
            grouped = {}
            
            for chunk in chunks:
                # Create grouping key (document, section)
                doc_name = chunk.document_name
                section = chunk.section_heading or "General"
                key = (doc_name, section)
                
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(chunk)
            
            logger.debug(f"Grouped chunks into {len(grouped)} sections")
            return grouped
            
        except Exception as e:
            logger.error(f"Error grouping chunks: {str(e)}")
            return {}
    
    def _create_context_sections(
        self,
        grouped_chunks: Dict[Tuple[str, str], List[RetrievedChunk]],
        scores: List[Any],
        optimization_hints: Optional[Dict[str, Any]] = None
    ) -> List[ContextSection]:
        """
        Create context sections from grouped chunks with optional optimization hints.
        
        Args:
            grouped_chunks: Chunks grouped by document and section
            scores: Relevance scores for chunks
            optimization_hints: Optional hints for section optimization
            
        Returns:
            List of context sections
        """
        try:
            sections = []
            score_map = {score.chunk_index: score for score in scores}
            
            # Apply optimization hints if available
            priority_themes = optimization_hints.get('priority_themes', []) if optimization_hints else []
            boost_factor = optimization_hints.get('theme_boost_factor', 1.1) if optimization_hints else 1.0
            
            for (doc_name, section_heading), chunk_list in grouped_chunks.items():
                # Calculate section relevance (average of chunk scores)
                chunk_scores = []
                for chunk in chunk_list:
                    # Find corresponding score (this is approximate)
                    chunk_scores.append(chunk.score)
                
                avg_relevance = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
                
                # Apply theme-based boosting if optimization hints are available
                if priority_themes:
                    section_text = f"{doc_name} {section_heading}".lower()
                    for theme in priority_themes:
                        if theme.lower() in section_text:
                            avg_relevance *= boost_factor
                            logger.debug(f"Boosted relevance for section '{section_heading}' due to theme '{theme}'")
                            break
                
                # Combine chunk contents with safe extraction
                section_content_parts = []
                for chunk in chunk_list:
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
                        section_content_parts.append(content)
                    except Exception as e:
                        logger.warning(f"Error extracting content from chunk: {e}")
                        section_content_parts.append(str(chunk))
                
                section_content = "\n\n".join(section_content_parts)
                
                # Determine section order (README first, then by document type)
                section_order = self._get_section_order(doc_name, section_heading)
                
                # Create section title
                section_title = f"## {doc_name} - {section_heading}"
                
                section = ContextSection(
                    title=section_title,
                    content=section_content,
                    source_document=doc_name,
                    section_heading=section_heading,
                    relevance_score=avg_relevance,
                    section_order=section_order,
                    chunk_count=len(chunk_list)
                )
                
                sections.append(section)
            
            logger.debug(f"Created {len(sections)} context sections with optimization hints: {bool(optimization_hints)}")
            return sections
            
        except Exception as e:
            logger.error(f"Error creating context sections: {str(e)}")
            return []
    
    def _get_section_order(self, doc_name: str, section_heading: str) -> int:
        """Determine section ordering priority."""
        try:
            # README gets highest priority
            if "README" in doc_name.upper():
                return 0
            
            # FAQ sections get high priority
            if "FAQ" in doc_name.upper() or "faq" in doc_name.lower():
                return 1
            
            # PSAK documents get medium-high priority
            if "PSAK" in doc_name.upper() or "psak" in doc_name.lower():
                return 2
            
            # PUC documents get medium priority
            if "PUC" in doc_name.upper() or "puc" in doc_name.lower():
                return 3
            
            # Technical documents get medium-low priority
            if any(term in doc_name.lower() for term in ["teknis", "technical", "metode", "method"]):
                return 4
            
            # Other documents get lowest priority
            return 5
            
        except Exception as e:
            logger.error(f"Error determining section order: {str(e)}")
            return 10  # Default low priority
    
    def _order_sections(self, sections: List[ContextSection]) -> List[ContextSection]:
        """Order sections by priority and relevance."""
        try:
            # Sort by section order (priority) first, then by relevance score
            ordered = sorted(
                sections,
                key=lambda s: (s.section_order, -s.relevance_score)
            )
            
            logger.debug(f"Ordered {len(ordered)} sections")
            return ordered
            
        except Exception as e:
            logger.error(f"Error ordering sections: {str(e)}")
            return sections
    
    def _assemble_final_context(
        self,
        readme_content: str,
        ordered_sections: List[ContextSection],
        max_length: int
    ) -> Tuple[str, List[ContextSection]]:
        """Assemble final context within length limits."""
        try:
            context_parts = []
            included_sections = []
            current_length = 0
            
            # Add README first if available
            if readme_content:
                readme_section = f"# Panduan Sistem\n\n{readme_content}\n\n"
                if current_length + len(readme_section) <= max_length:
                    context_parts.append(readme_section)
                    current_length += len(readme_section)
                    logger.debug("Added README to context")
            
            # Add sections in order until length limit
            for section in ordered_sections:
                section_text = f"{section.title}\n\n{section.content}\n\n"
                
                # Check if adding this section would exceed limit
                if current_length + len(section_text) <= max_length:
                    context_parts.append(section_text)
                    included_sections.append(section)
                    current_length += len(section_text)
                    logger.debug(f"Added section: {section.source_document} - {section.section_heading}")
                else:
                    # Try to add partial content if space allows
                    remaining_space = max_length - current_length - len(section.title) - 10
                    if remaining_space > 100:  # Minimum useful content
                        partial_content = section.content[:remaining_space] + "..."
                        partial_section_text = f"{section.title}\n\n{partial_content}\n\n"
                        context_parts.append(partial_section_text)
                        
                        # Create partial section
                        partial_section = ContextSection(
                            title=section.title,
                            content=partial_content,
                            source_document=section.source_document,
                            section_heading=section.section_heading,
                            relevance_score=section.relevance_score,
                            section_order=section.section_order,
                            chunk_count=section.chunk_count
                        )
                        included_sections.append(partial_section)
                        logger.debug(f"Added partial section: {section.source_document}")
                    
                    break  # Stop adding sections
            
            final_context = "".join(context_parts)
            
            logger.info(f"Final context: {len(final_context)} chars, {len(included_sections)} sections")
            return final_context, included_sections
            
        except Exception as e:
            logger.error(f"Error assembling final context: {str(e)}")
            return "", []
    
    def _create_error_context(self, error: str) -> AssembledContext:
        """Create error context result."""
        error_content = f"Error assembling context: {error}"
        
        return AssembledContext(
            full_context=error_content,
            sections=[],
            total_length=len(error_content),
            chunk_count=0,
            document_count=0,
            has_readme=False,
            assembly_metadata={"error": error}
        )
    
    def get_context_statistics(self, context: AssembledContext) -> Dict[str, Any]:
        """Get statistics about assembled context."""
        try:
            stats = {
                "total_length": context.total_length,
                "section_count": len(context.sections),
                "document_count": context.document_count,
                "chunk_count": context.chunk_count,
                "has_readme": context.has_readme,
                "average_section_length": context.total_length / len(context.sections) if context.sections else 0,
                "sections_by_document": {},
                "relevance_distribution": {
                    "high (>0.7)": 0,
                    "medium (0.4-0.7)": 0,
                    "low (<0.4)": 0
                }
            }
            
            # Analyze sections
            for section in context.sections:
                doc_name = section.source_document
                if doc_name not in stats["sections_by_document"]:
                    stats["sections_by_document"][doc_name] = 0
                stats["sections_by_document"][doc_name] += 1
                
                # Relevance distribution
                if section.relevance_score > 0.7:
                    stats["relevance_distribution"]["high (>0.7)"] += 1
                elif section.relevance_score >= 0.4:
                    stats["relevance_distribution"]["medium (0.4-0.7)"] += 1
                else:
                    stats["relevance_distribution"]["low (<0.4)"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating context statistics: {str(e)}")
            return {"error": str(e)}
    
    def format_context_for_llm(self, context: AssembledContext) -> str:
        """Format context specifically for LLM consumption."""
        try:
            # Add metadata header
            header = f"""# Context Information
Total Documents: {context.document_count}
Total Sections: {len(context.sections)}
Context Length: {context.total_length} characters

---

"""
            
            # Add the assembled context
            formatted_context = header + context.full_context
            
            # Add footer with source summary
            footer = "\n\n---\n\n# Source Documents\n"
            doc_sources = set(section.source_document for section in context.sections)
            for doc in sorted(doc_sources):
                footer += f"- {doc}\n"
            
            return formatted_context + footer
            
        except Exception as e:
            logger.error(f"Error formatting context for LLM: {str(e)}")
            return context.full_context