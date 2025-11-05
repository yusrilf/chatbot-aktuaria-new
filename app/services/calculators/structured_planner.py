"""Structured JSON Planner for Actuarial Calculations."""

import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from app.config import config

logger = logging.getLogger(__name__)

class StructuredPlanner:
    """Generates structured JSON plans for actuarial calculations."""
    
    def __init__(self, llm=None, prompt_manager=None, schema_validator=None, vector_store_manager=None):
        """Initialize the structured planner.
        
        Args:
            llm: Language model for plan generation
            prompt_manager: Prompt template manager
            schema_validator: JSON schema validator
            vector_store_manager: Vector store for document retrieval
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.schema_validator = schema_validator
        self.vector_store_manager = vector_store_manager
    
    def generate_structured_plan(self, question: str, session_id: str, 
                               context_docs: Optional[List[Document]] = None) -> Dict[str, Any]:
        """Generate structured JSON plan for calculation.
        
        Args:
            question: User's calculation question
            session_id: Session identifier
            context_docs: Optional context documents
            
        Returns:
            Dict with plan generation result
        """
        try:
            # Prepare context
            context = self._prepare_context(context_docs, session_id)
            available_files = self._get_available_files(session_id)
            
            # Get structured planner template
            if self.prompt_manager:
                template = self.prompt_manager.get_template('structured_planner')
            else:
                template = self._get_default_template()
            
            # Create prompt
            prompt = PromptTemplate(
                input_variables=["context", "available_files", "question"],
                template=template
            )
            
            # Generate plan
            llm_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=False)
            raw_response = llm_chain.run({
                "context": context,
                "available_files": available_files,
                "question": question
            })
            
            # Clean and validate JSON response
            cleaned_json = self._clean_json_response(raw_response)
            
            if self.schema_validator:
                validation_result = self.schema_validator.validate_plan(cleaned_json)
            else:
                validation_result = self._basic_json_validation(cleaned_json)
            
            if validation_result["valid"]:
                return {
                    "success": True,
                    "plan": validation_result["parsed_plan"],
                    "raw_response": raw_response,
                    "cleaned_json": cleaned_json,
                    "validation_errors": [],
                    "session_id": session_id
                }
            else:
                logger.error(f"Plan validation failed: {validation_result['errors']}")
                return {
                    "success": False,
                    "plan": None,
                    "raw_response": raw_response,
                    "cleaned_json": cleaned_json,
                    "validation_errors": validation_result["errors"],
                    "session_id": session_id
                }
                
        except Exception as e:
            logger.error(f"Structured plan generation failed: {e}\n{traceback.format_exc()}")
            return {
                "success": False,
                "plan": None,
                "raw_response": "",
                "cleaned_json": "",
                "validation_errors": [f"Plan generation error: {str(e)}"],
                "session_id": session_id
            }
    
    def _prepare_context(self, context_docs: Optional[List[Document]], session_id: str) -> str:
        """Prepare context from documents.
        
        Args:
            context_docs: List of context documents
            session_id: Session identifier
            
        Returns:
            Formatted context string
        """
        if not context_docs:
            # Try to get INDEX.md and key documents
            context_docs = self._get_key_documents(session_id)
        
        if not context_docs:
            return "No context documents available."
        
        context_parts = []
        for doc in context_docs[:5]:  # Limit to 5 docs to avoid token limits
            metadata = getattr(doc, 'metadata', {})
            filename = metadata.get('filename', 'Unknown')
            doc_type = metadata.get('doc_type', 'Unknown')
            
            context_parts.append(f"**{filename}** ({doc_type}):\n{doc.page_content[:1000]}...\n")
        
        return "\n".join(context_parts)
    
    def _get_available_files(self, session_id: str) -> str:
        """Get list of available files for the session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted file list string
        """
        try:
            if self.vector_store_manager:
                # Get unique filenames from vector store using hybrid search
                docs = self.vector_store_manager.hybrid_similarity_search_with_score(
                    query="*", 
                    session_id=session_id,
                    k=config.PLANNER_SEARCH_K,  # Use configurable K for planner search
                    use_hybrid=True
                )
                
                filenames = set()
                for doc, _ in docs:
                    metadata = getattr(doc, 'metadata', {})
                    filename = metadata.get('filename')
                    if filename:
                        filenames.add(filename)
                
                if filenames:
                    return "Available files: " + ", ".join(sorted(filenames))
            
            return "Available files: INDEX.md, step01_employee_data.md, step02_multiple_decrement.md, step03_benefit_calculation.md, step04_pvfb_pvdbo.md, step05_sensitivity_analysis.md"
            
        except Exception as e:
            logger.error(f"Error getting available files: {e}")
            return "Available files: INDEX.md, calculation steps, assumptions documents"
    
    def _get_key_documents(self, session_id: str) -> List[Document]:
        """Get key documents for planning context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of key documents
        """
        try:
            if not self.vector_store_manager:
                return []
            
            # Search for INDEX.md and key calculation documents
            key_queries = [
                "INDEX calculation steps",
                "employee data foundation",
                "benefit calculation",
                "assumptions mortality"
            ]
            
            docs = []
            for query in key_queries:
                results = self.vector_store_manager.similarity_search_with_score(
                    query=query, k=config.PLANNER_KEY_DOCS_K, session_id=session_id  # Use configurable K for key docs
                )
                docs.extend([doc for doc, _ in results])
            
            # Remove duplicates
            unique_docs = []
            seen_content = set()
            for doc in docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(content_hash)
            
            return unique_docs[:5]
            
        except Exception as e:
            logger.error(f"Error getting key documents: {e}")
            return []
    
    def _clean_json_response(self, raw_response: str) -> str:
        """Clean LLM response to extract valid JSON.
        
        Args:
            raw_response: Raw response from LLM
            
        Returns:
            Cleaned JSON string
        """
        try:
            # Remove markdown code blocks
            cleaned = raw_response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:]
            
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            
            # Find JSON object boundaries
            start_idx = cleaned.find('{')
            if start_idx == -1:
                return cleaned
            
            # Find matching closing brace
            brace_count = 0
            end_idx = -1
            
            for i, char in enumerate(cleaned[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx != -1:
                return cleaned[start_idx:end_idx]
            
            return cleaned.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            return raw_response
    
    def _basic_json_validation(self, json_str: str) -> Dict[str, Any]:
        """Basic JSON validation without schema.
        
        Args:
            json_str: JSON string to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            parsed = json.loads(json_str)
            
            # Check required keys
            required_keys = ["variables", "retrieval_plan", "calculation_steps", "assumptions"]
            missing_keys = [key for key in required_keys if key not in parsed]
            
            if missing_keys:
                return {
                    "valid": False,
                    "errors": [f"Missing required keys: {', '.join(missing_keys)}"],
                    "parsed_plan": parsed
                }
            
            return {
                "valid": True,
                "errors": [],
                "parsed_plan": parsed
            }
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "errors": [f"Invalid JSON: {str(e)}"],
                "parsed_plan": None
            }
    
    def _get_default_template(self) -> str:
        """Get default structured planner template."""
        return """
You are a planning assistant for actuarial calculations. Output ONLY a JSON object.

QUESTION: {question}
CONTEXT: {context}
FILES: {available_files}

Return JSON with: variables[], retrieval_plan[], calculation_steps[], assumptions{}
"""