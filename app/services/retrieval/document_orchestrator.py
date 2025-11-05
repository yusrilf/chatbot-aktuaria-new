"""Document Orchestrator for CoT Retrieval System.

This module implements the LLM-based document orchestrator that selects
relevant documents based on metadata and provides reasoning for selection.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.services.retrieval.query_preprocessor import ExtractedEntities

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Container for document metadata."""
    filename: str
    doc_type: str
    keywords: List[str]
    domain: str
    difficulty: str
    version: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


@dataclass
class OrchestrationResult:
    """Result from document orchestration."""
    selected_files: List[str]
    reasoning: str
    confidence_score: float
    fallback_used: bool
    metadata_used: List[DocumentMetadata]


class DocumentOrchestrator:
    """LLM-based document orchestrator for retrieval."""
    
    def __init__(self, llm: LLM):
        """
        Initialize the document orchestrator.
        
        Args:
            llm: Language model for orchestration
        """
        self.llm = llm
        self._setup_prompts()
        logger.info("DocumentOrchestrator initialized")
    
    def _setup_prompts(self):
        """Setup prompt templates for orchestration."""
        self.orchestration_prompt_template = """You are an orchestration assistant for an actuarial document retrieval system.

Given a user query and document metadata, select the most relevant documents and provide reasoning.

User Query: "{query}"

Extracted Entities:
- PSAK References: {psak_refs}
- PUC References: {puc_refs}  
- Technical Terms: {tech_terms}
- Numbers: {numbers}
- Years: {years}

Available Documents:
{documents_metadata}

Instructions:
1. Analyze the query and extracted entities
2. Select 5-10 most relevant documents based on:
   - PRIORITY DOCUMENTS (always include if available):
     * README.md, INDEX.md (navigation and overview)
     * step01_*.md, step02_*.md, step03_*.md, step04_*.md, step05_*.md (calculation guides)
     * 05_faq.md, 05c_faq_teknis.md (FAQ documents)
     * troubleshooting_guide.md (problem solving)
   - Keyword matching with query entities
   - Document type relevance (teknis for technical questions, faq for general questions)
   - Domain alignment (aktuaria for actuarial topics)
   - Difficulty level appropriateness
3. For calculation questions, prioritize step-by-step guides and technical documents
4. For general questions, prioritize FAQ and README documents
5. Always include at least one FAQ document if available
6. Provide clear reasoning for your selection
7. Return ONLY valid JSON in this exact format:

{{"selected_files": ["README.md", "INDEX.md", "step01_employee_data.md", "05_faq.md", "file1.md", "file2.md"], "reasoning": "Clear explanation of why these documents were selected based on query analysis and metadata matching. Prioritized core navigation documents, step-by-step guides for calculations, and FAQ for general guidance.", "confidence": 0.85}}

Output must be parseable JSON only. No additional text."""

        self.orchestration_prompt = PromptTemplate(
            input_variables=[
                "query", "psak_refs", "puc_refs", "tech_terms", 
                "numbers", "years", "documents_metadata"
            ],
            template=self.orchestration_prompt_template
        )
        
        self.orchestration_chain = LLMChain(
            llm=self.llm,
            prompt=self.orchestration_prompt,
            verbose=False
        )
    
    def orchestrate_documents(
        self, 
        query: str,
        entities: ExtractedEntities,
        available_documents: List[DocumentMetadata]
    ) -> OrchestrationResult:
        """
        Orchestrate document selection based on query and entities.
        
        Args:
            query: User query
            entities: Extracted entities from query
            available_documents: List of available document metadata
            
        Returns:
            OrchestrationResult with selected documents and reasoning
        """
        try:
            logger.info(f"Orchestrating documents for query: {query[:100]}...")
            
            # Format documents metadata for prompt
            docs_metadata_str = self._format_documents_metadata(available_documents)
            
            # Prepare prompt variables
            prompt_vars = {
                "query": query,
                "psak_refs": ", ".join(entities.psak_references) if entities.psak_references else "None",
                "puc_refs": ", ".join(entities.puc_references) if entities.puc_references else "None",
                "tech_terms": ", ".join(entities.technical_terms) if entities.technical_terms else "None",
                "numbers": ", ".join(map(str, entities.numbers)) if entities.numbers else "None",
                "years": ", ".join(map(str, entities.years)) if entities.years else "None",
                "documents_metadata": docs_metadata_str
            }
            
            # Run orchestration
            response = self.orchestration_chain.run(prompt_vars)
            
            # Parse response
            result = self._parse_orchestration_response(response, available_documents)
            
            # Apply fallback if needed
            if not result.selected_files:
                logger.warning("No documents selected by orchestrator, applying fallback")
                result = self._apply_fallback_selection(entities, available_documents)
                result.fallback_used = True
            
            logger.info(f"Orchestration completed: {len(result.selected_files)} documents selected")
            return result
            
        except Exception as e:
            logger.error(f"Error in document orchestration: {str(e)}")
            # Return fallback result
            return self._apply_fallback_selection(entities, available_documents)
    
    def _format_documents_metadata(self, documents: List[DocumentMetadata]) -> str:
        """Format document metadata for prompt."""
        try:
            formatted_docs = []
            
            for doc in documents:
                keywords_str = ", ".join(doc.keywords) if doc.keywords else "None"
                doc_line = (f"- {doc.filename}: "
                           f"keywords: [{keywords_str}], "
                           f"doc_type: {doc.doc_type}, "
                           f"domain: {doc.domain}, "
                           f"difficulty: {doc.difficulty}")
                
                if doc.title:
                    doc_line += f", title: {doc.title}"
                
                formatted_docs.append(doc_line)
            
            return "\n".join(formatted_docs)
            
        except Exception as e:
            logger.error(f"Error formatting documents metadata: {str(e)}")
            return "Error formatting metadata"
    
    def _parse_orchestration_response(
        self, 
        response: str, 
        available_documents: List[DocumentMetadata]
    ) -> OrchestrationResult:
        """Parse LLM orchestration response."""
        try:
            # Clean response
            response = response.strip()
            
            # Try to extract JSON from response
            json_match = None
            if response.startswith('{') and response.endswith('}'):
                json_match = response
            else:
                # Try to find JSON in response
                import re
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, response, re.DOTALL)
                if matches:
                    json_match = matches[-1]  # Take the last/most complete match
            
            if not json_match:
                raise ValueError("No valid JSON found in response")
            
            # Parse JSON
            parsed = json.loads(json_match)
            
            # Extract fields
            selected_files = parsed.get("selected_files", [])
            reasoning = parsed.get("reasoning", "No reasoning provided")
            confidence = float(parsed.get("confidence", 0.5))
            
            # Validate selected files exist
            available_filenames = [doc.filename for doc in available_documents]
            valid_files = [f for f in selected_files if f in available_filenames]
            
            if len(valid_files) != len(selected_files):
                logger.warning(f"Some selected files not found: {set(selected_files) - set(valid_files)}")
            
            # Get metadata for selected files
            selected_metadata = [
                doc for doc in available_documents 
                if doc.filename in valid_files
            ]
            
            return OrchestrationResult(
                selected_files=valid_files,
                reasoning=reasoning,
                confidence_score=confidence,
                fallback_used=False,
                metadata_used=selected_metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing orchestration response: {str(e)}")
            logger.error(f"Response was: {response}")
            
            # Return empty result for fallback handling
            return OrchestrationResult(
                selected_files=[],
                reasoning=f"Failed to parse orchestration response: {str(e)}",
                confidence_score=0.0,
                fallback_used=False,
                metadata_used=[]
            )
    
    def _apply_fallback_selection(
        self, 
        entities: ExtractedEntities, 
        available_documents: List[DocumentMetadata]
    ) -> OrchestrationResult:
        """Apply enhanced fallback document selection rules with priority system."""
        try:
            logger.info("Applying enhanced fallback document selection")
            
            selected_files = []
            reasoning_parts = []
            priority_docs = []
            
            # Priority 1: Always include core navigation documents
            core_docs = [doc for doc in available_documents 
                        if any(core_name in doc.filename.lower() 
                              for core_name in ['readme.md', 'index.md'])]
            if core_docs:
                priority_docs.extend(core_docs)
                reasoning_parts.append("Included core navigation documents (README, INDEX)")
            
            # Priority 2: Include step-by-step calculation guides
            step_docs = [doc for doc in available_documents 
                        if any(step_pattern in doc.filename.lower() 
                              for step_pattern in ['step01', 'step02', 'step03', 'step04', 'step05',
                                                 'step_', 'langkah', 'panduan_langkah'])]
            if step_docs:
                # Sort step documents by number for logical order
                step_docs.sort(key=lambda x: x.filename.lower())
                priority_docs.extend(step_docs[:3])  # Include first 3 step documents
                reasoning_parts.append("Included step-by-step calculation guides")
            
            # Priority 3: Include FAQ documents (main and sub-categories)
            faq_docs = [doc for doc in available_documents 
                       if any(faq_pattern in doc.filename.lower() 
                             for faq_pattern in ['faq', '05_faq', '05a_faq', '05b_faq', 
                                               '05c_faq', '05d_faq'])]
            if faq_docs:
                # Prioritize main FAQ first, then technical FAQ
                main_faq = [doc for doc in faq_docs if doc.filename.lower() in ['05_faq.md', 'faq.md']]
                tech_faq = [doc for doc in faq_docs if '05c_faq' in doc.filename.lower()]
                other_faq = [doc for doc in faq_docs if doc not in main_faq and doc not in tech_faq]
                
                priority_docs.extend(main_faq[:1])  # Main FAQ
                priority_docs.extend(tech_faq[:1])  # Technical FAQ
                priority_docs.extend(other_faq[:2])  # Other FAQ categories
                reasoning_parts.append("Included FAQ documents (main, technical, and specialized)")
            
            # Priority 4: Include troubleshooting and reference documents
            support_docs = [doc for doc in available_documents 
                           if any(support_pattern in doc.filename.lower() 
                                 for support_pattern in ['troubleshooting', 'assumptions_reference',
                                                       'benefit_factors', 'yield_curve', 'mortality'])]
            if support_docs:
                priority_docs.extend(support_docs[:2])
                reasoning_parts.append("Included troubleshooting and reference documents")
            
            # Priority 5: Query-specific document selection
            if entities.psak_references:
                psak_docs = [
                    doc for doc in available_documents 
                    if any('psak' in keyword.lower() for keyword in (doc.keywords or []))
                    or any(psak_pattern in doc.filename.lower() 
                          for psak_pattern in ['psak', '01_dasar', '02_teknis'])
                ]
                priority_docs.extend(psak_docs[:2])
                reasoning_parts.append("Included PSAK-related documents based on query entities")
            
            if entities.puc_references:
                puc_docs = [
                    doc for doc in available_documents 
                    if any(puc_keyword in keyword.lower() 
                          for keyword in (doc.keywords or [])
                          for puc_keyword in ['puc', 'projected unit credit'])
                    or '02c_metode_puc' in doc.filename.lower()
                ]
                priority_docs.extend(puc_docs[:2])
                reasoning_parts.append("Included PUC method documents based on query entities")
            
            # Priority 6: Technical documents for complex queries
            if entities.technical_terms:
                tech_docs = [
                    doc for doc in available_documents 
                    if (hasattr(doc, 'doc_type') and doc.doc_type == 'teknis' and 
                        hasattr(doc, 'domain') and doc.domain == 'aktuaria')
                    or any(tech_pattern in doc.filename.lower() 
                          for tech_pattern in ['02_teknis', '02a_konsultan', '02b_dana',
                                             '02d_asumsi', '02e_laporan', '02f_proses'])
                ]
                priority_docs.extend(tech_docs[:3])
                reasoning_parts.append("Included technical actuarial documents")
            
            # Priority 7: Implementation and reporting documents
            impl_docs = [doc for doc in available_documents 
                        if any(impl_pattern in doc.filename.lower() 
                              for impl_pattern in ['03_implementasi', '04_penyajian', 
                                                 'implementasi', 'penyajian', 'laporan'])]
            if impl_docs:
                priority_docs.extend(impl_docs[:2])
                reasoning_parts.append("Included implementation and reporting documents")
            
            # Remove duplicates while preserving priority order
            unique_files = []
            seen_files = set()
            for doc in priority_docs:
                if doc.filename not in seen_files:
                    unique_files.append(doc.filename)
                    seen_files.add(doc.filename)
            
            # Ensure we have a reasonable number of documents (5-10)
            if len(unique_files) < 5:
                # Add more documents if we don't have enough
                additional_docs = [doc for doc in available_documents 
                                 if doc.filename not in seen_files][:5-len(unique_files)]
                unique_files.extend([doc.filename for doc in additional_docs])
                if additional_docs:
                    reasoning_parts.append("Added additional documents to meet minimum selection")
            
            # Limit to maximum of 10 documents for performance
            unique_files = unique_files[:10]
            
            # Get metadata for selected files
            selected_metadata = [
                doc for doc in available_documents 
                if doc.filename in unique_files
            ]
            
            reasoning = "Enhanced fallback selection applied: " + "; ".join(reasoning_parts)
            
            logger.info(f"Fallback selection completed: {len(unique_files)} documents selected")
            logger.info(f"Selected documents: {unique_files}")
            
            return OrchestrationResult(
                selected_files=unique_files,
                reasoning=reasoning,
                confidence_score=0.7,  # Higher confidence for enhanced fallback
                fallback_used=True,
                metadata_used=selected_metadata
            )
            
        except Exception as e:
            logger.error(f"Error in fallback selection: {str(e)}")
            
            # Ultimate fallback - return first few documents
            fallback_files = [doc.filename for doc in available_documents[:5]]
            return OrchestrationResult(
                selected_files=fallback_files,
                reasoning=f"Emergency fallback due to error: {str(e)}",
                confidence_score=0.3,
                fallback_used=True,
                metadata_used=available_documents[:5]
            )