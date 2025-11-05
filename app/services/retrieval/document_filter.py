"""Document Filter for CoT Retrieval System.

This module implements document filtering based on orchestrator recommendations
and applies fallback rules when needed.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

from app.services.retrieval.document_orchestrator import OrchestrationResult, DocumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result from document filtering."""
    filtered_files: List[str]
    filter_criteria: Dict[str, Any]
    total_available: int
    total_filtered: int
    fallback_applied: bool
    reasoning: str


class DocumentFilter:
    """Filters documents based on orchestration results and fallback rules."""
    
    def __init__(self):
        """Initialize the document filter."""
        self.fallback_files = [
            'README.md',
            'readme.md', 
            'README.MD',
            'faq.md',
            'FAQ.md',
            'faq_general.md',
            'panduan.md',
            'guide.md'
        ]
        logger.info("DocumentFilter initialized")
    
    def apply_filtering(
        self,
        orchestration_result: OrchestrationResult,
        available_documents: List[DocumentMetadata],
        session_id: Optional[str] = None,
        complexity_level: Optional[str] = None
    ) -> FilterResult:
        """
        Apply document filtering with complexity-based early filtering
        
        Args:
            orchestration_result: Results from document orchestration
            available_documents: All available documents
            session_id: Session identifier for session-specific filtering
            complexity_level: Query complexity level for early filtering
        """
        try:
            logger.info(f"Starting document filtering with complexity: {complexity_level}")
            
            # Apply complexity-based early filtering
            if complexity_level:
                available_documents = self._apply_complexity_filtering(
                    available_documents, complexity_level
                )
                logger.info(f"After complexity filtering: {len(available_documents)} documents")
            
            # Get available filenames
            available_filenames = [doc.filename for doc in available_documents]
            
            # Start with orchestrator selection
            selected_files = orchestration_result.selected_files.copy()
            
            # Validate selected files exist
            valid_selected = [f for f in selected_files if f in available_filenames]
            
            if len(valid_selected) != len(selected_files):
                missing_files = set(selected_files) - set(valid_selected)
                logger.warning(f"Some orchestrator-selected files not found: {missing_files}")
            
            # Apply fallback if no valid files or orchestrator failed
            fallback_applied = False
            if not valid_selected or orchestration_result.fallback_used:
                logger.info("Applying fallback filtering rules")
                fallback_files = self._apply_fallback_rules(
                    available_documents, 
                    orchestration_result
                )
                
                # Combine with existing selection
                combined_files = valid_selected + fallback_files
                
                # Remove duplicates while preserving order
                unique_files = []
                for file in combined_files:
                    if file not in unique_files:
                        unique_files.append(file)
                
                valid_selected = unique_files
                fallback_applied = True
            
            # Apply session-specific filtering if needed
            if session_id:
                valid_selected = self._apply_session_filtering(
                    valid_selected, 
                    session_id, 
                    available_documents
                )
            
            # Ensure we have at least some documents
            if not valid_selected:
                logger.warning("No documents after filtering, using emergency fallback")
                valid_selected = self._emergency_fallback(available_documents)
                fallback_applied = True
            
            # Create filter criteria summary
            filter_criteria = self._create_filter_criteria(
                orchestration_result, 
                fallback_applied
            )
            
            # Add complexity filtering info to criteria
            if complexity_level:
                filter_criteria["complexity_filtering"] = True
                filter_criteria["complexity_level"] = complexity_level
            
            # Create reasoning
            reasoning = self._create_filtering_reasoning(
                orchestration_result,
                valid_selected,
                fallback_applied
            )
            
            result = FilterResult(
                filtered_files=valid_selected,
                filter_criteria=filter_criteria,
                total_available=len(available_documents),
                total_filtered=len(valid_selected),
                fallback_applied=fallback_applied,
                reasoning=reasoning
            )
            
            logger.info(f"Filtering completed: {len(valid_selected)} documents selected")
            return result
            
        except Exception as e:
            logger.error(f"Error in document filtering: {str(e)}")
            
            # Emergency fallback
            emergency_files = self._emergency_fallback(available_documents)
            return FilterResult(
                filtered_files=emergency_files,
                filter_criteria={"error": str(e)},
                total_available=len(available_documents),
                total_filtered=len(emergency_files),
                fallback_applied=True,
                reasoning=f"Emergency fallback due to filtering error: {str(e)}"
            )
    
    def _apply_fallback_rules(
        self,
        available_documents: List[DocumentMetadata],
        orchestration_result: OrchestrationResult
    ) -> List[str]:
        """Apply fallback rules for document selection."""
        try:
            fallback_files = []
            available_filenames = [doc.filename for doc in available_documents]
            
            # Rule 1: Always include README if available
            for readme_name in self.fallback_files:
                if readme_name in available_filenames:
                    fallback_files.append(readme_name)
                    break
            
            # Rule 2: Include FAQ documents
            faq_docs = [
                doc.filename for doc in available_documents 
                if 'faq' in doc.filename.lower() or doc.doc_type == 'faq'
            ]
            fallback_files.extend(faq_docs[:2])  # Limit to 2 FAQ docs
            
            # Rule 3: Include general guidance documents
            guide_docs = [
                doc.filename for doc in available_documents 
                if any(term in doc.filename.lower() for term in ['panduan', 'guide', 'petunjuk'])
            ]
            fallback_files.extend(guide_docs[:1])
            
            # Rule 4: Include high-priority technical documents
            tech_docs = [
                doc.filename for doc in available_documents 
                if doc.doc_type == 'teknis' and doc.domain == 'aktuaria'
            ]
            fallback_files.extend(tech_docs[:3])
            
            # Remove duplicates
            unique_fallback = []
            for file in fallback_files:
                if file not in unique_fallback:
                    unique_fallback.append(file)
            
            return unique_fallback
            
        except Exception as e:
            logger.error(f"Error applying fallback rules: {str(e)}")
            return []
    
    def _apply_session_filtering(
        self,
        selected_files: List[str],
        session_id: str,
        available_documents: List[DocumentMetadata]
    ) -> List[str]:
        """Apply session-specific filtering if needed."""
        try:
            # For now, just return the selected files
            # This can be extended to include session-specific logic
            logger.debug(f"Session filtering for {session_id}: no changes applied")
            return selected_files
            
        except Exception as e:
            logger.error(f"Error in session filtering: {str(e)}")
            return selected_files
    
    def _emergency_fallback(self, available_documents: List[DocumentMetadata]) -> List[str]:
        """Emergency fallback when no documents are selected."""
        try:
            # Return first 5 documents or all if less than 5
            emergency_files = [doc.filename for doc in available_documents[:5]]
            logger.warning(f"Emergency fallback: selected {len(emergency_files)} documents")
            return emergency_files
            
        except Exception as e:
            logger.error(f"Error in emergency fallback: {str(e)}")
            return []
    
    def _create_filter_criteria(
        self,
        orchestration_result: OrchestrationResult,
        fallback_applied: bool
    ) -> Dict[str, Any]:
        """Create filter criteria summary."""
        try:
            criteria = {
                "orchestrator_confidence": orchestration_result.confidence_score,
                "orchestrator_fallback": orchestration_result.fallback_used,
                "filter_fallback": fallback_applied,
                "selected_count": len(orchestration_result.selected_files),
                "metadata_count": len(orchestration_result.metadata_used)
            }
            
            return criteria
            
        except Exception as e:
            logger.error(f"Error creating filter criteria: {str(e)}")
            return {"error": str(e)}
    
    def _create_filtering_reasoning(
        self,
        orchestration_result: OrchestrationResult,
        final_files: List[str],
        fallback_applied: bool
    ) -> str:
        """Create reasoning for filtering decisions."""
        try:
            reasoning_parts = []
            
            # Orchestrator reasoning
            if orchestration_result.reasoning:
                reasoning_parts.append(f"Orchestrator: {orchestration_result.reasoning}")
            
            # Fallback reasoning
            if fallback_applied:
                reasoning_parts.append("Applied fallback rules to ensure document coverage")
            
            # Final selection summary
            reasoning_parts.append(f"Final selection: {len(final_files)} documents")
            
            return " | ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Error creating filtering reasoning: {str(e)}")
            return f"Filtering completed with {len(final_files)} documents"
    
    def _apply_complexity_filtering(
        self,
        documents: List[DocumentMetadata],
        complexity_level: str
    ) -> List[DocumentMetadata]:
        """
        Apply early document filtering based on query complexity.
        
        Args:
            documents: Available documents to filter
            complexity_level: Query complexity ('simple', 'moderate', 'complex')
            
        Returns:
            Filtered list of documents based on complexity
        """
        try:
            if complexity_level == 'simple':
                # For simple queries, prioritize basic documentation
                priority_patterns = [
                    'readme', 'faq', 'panduan', 'guide', 'basic', 'intro',
                    'getting_started', 'overview', 'summary'
                ]
                max_docs = 15
                
            elif complexity_level == 'moderate':
                # For moderate queries, include technical docs but limit scope
                priority_patterns = [
                    'readme', 'faq', 'panduan', 'guide', 'api', 'reference',
                    'tutorial', 'example', 'config', 'setup'
                ]
                max_docs = 30
                
            else:  # complex
                # For complex queries, allow broader document access
                priority_patterns = [
                    'readme', 'faq', 'panduan', 'guide', 'api', 'reference',
                    'technical', 'advanced', 'architecture', 'implementation',
                    'specification', 'detailed'
                ]
                max_docs = 50
            
            # Score documents based on priority patterns
            scored_docs = []
            for doc in documents:
                score = 0
                filename_lower = doc.filename.lower()
                
                # Check for priority patterns
                for pattern in priority_patterns:
                    if pattern in filename_lower:
                        score += 10
                
                # Boost score for session documents
                if hasattr(doc, 'is_session_doc') and doc.is_session_doc:
                    score += 20
                
                # Boost score for recently modified documents
                if hasattr(doc, 'last_modified') and doc.last_modified:
                    score += 5
                
                scored_docs.append((doc, score))
            
            # Sort by score (descending) and take top documents
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            filtered_docs = [doc for doc, _ in scored_docs[:max_docs]]
            
            logger.info(
                f"Complexity filtering ({complexity_level}): "
                f"{len(documents)} -> {len(filtered_docs)} documents"
            )
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error in complexity filtering: {str(e)}")
            # Return original documents if filtering fails
            return documents[:50]  # Safe fallback limit
    
    def get_filter_metadata(
        self,
        filtered_files: List[str],
        available_documents: List[DocumentMetadata]
    ) -> List[DocumentMetadata]:
        """Get metadata for filtered documents."""
        try:
            filtered_metadata = [
                doc for doc in available_documents 
                if doc.filename in filtered_files
            ]
            
            return filtered_metadata
            
        except Exception as e:
            logger.error(f"Error getting filter metadata: {str(e)}")
            return []
    
    def validate_filter_result(self, filter_result: FilterResult) -> bool:
        """Validate filter result."""
        try:
            # Check if we have any files
            if not filter_result.filtered_files:
                logger.error("Filter result has no files")
                return False
            
            # Check if counts make sense
            if filter_result.total_filtered > filter_result.total_available:
                logger.error("Filtered count exceeds available count")
                return False
            
            # Check if files are unique
            if len(filter_result.filtered_files) != len(set(filter_result.filtered_files)):
                logger.warning("Filter result contains duplicate files")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating filter result: {str(e)}")
            return False