"""Query processing and retrieval logic for actuarial chat service."""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Handles query processing and document retrieval."""
    
    def __init__(self, llm, vector_store_manager):
        """Initialize query processor.
        
        Args:
            llm: Language model instance
            vector_store_manager: Vector store manager instance
        """
        self.llm = llm
        self.vector_store_manager = vector_store_manager
    
    def refine_query_with_readme(self, readme_text: str, user_query: str) -> str:
        """LLM membaca README untuk menyusun refined query/keywords yang lebih tajam.
        
        Args:
            readme_text: README content for guidance
            user_query: Original user query
            
        Returns:
            Refined query string
        """
        try:
            prompt = PromptTemplate(
                input_variables=["readme", "q"],
                template=(
                    "Gunakan panduan navigasi di bawah ini untuk menyusun query pencarian yang spesifik:\n\n"
                    "=== NAVIGATION GUIDE ===\n{readme}\n\n"
                    "=== USER QUESTION ===\n{q}\n\n"
                    "TULISKAN refined query (maks 25 kata) yang memuat istilah teknis, step, atau tabel yang relevan."
                )
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            refined = chain.run(readme=readme_text, q=user_query)
            return refined.strip()
            
        except Exception as e:
            logger.error(f"Error refining query: {e}")
            return user_query
    
    def generate_retrieval_plan(self, question: str, session_id: str, 
                              refined_query: Optional[str] = None, 
                              include_global: bool = True) -> Dict[str, Any]:
        """Generate retrieval plan for document search.
        
        Args:
            question: User question
            session_id: Session identifier
            refined_query: Optional refined query
            include_global: Whether to include global documents
            
        Returns:
            Retrieval plan dictionary
        """
        try:
            # Basic retrieval plan structure
            plan = {
                "query_type": "general",
                "keywords": self._extract_keywords(question),
                "documents_needed": ["general"],
                "search_strategy": "hybrid",
                "confidence_threshold": 0.7
            }
            
            # Enhance with refined query if available
            if refined_query:
                plan["refined_keywords"] = self._extract_keywords(refined_query)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating retrieval plan: {e}")
            return {
                "query_type": "general",
                "keywords": [],
                "documents_needed": ["general"],
                "search_strategy": "basic",
                "confidence_threshold": 0.5
            }
    
    def retrieve_for_document(self, doc_name: str, keywords: str, 
                            session_id: str, k: int = 2) -> List[Tuple[Document, float]]:
        """Retrieve documents for specific document type.
        
        Args:
            doc_name: Document name/type
            keywords: Search keywords
            session_id: Session identifier
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Use vector store manager for retrieval
            results = self.vector_store_manager.similarity_search_with_score(
                query=keywords,
                k=k,
                filter_dict={"doc_type": doc_name} if doc_name != "general" else None
            )
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents for {doc_name}: {e}")
            return []
    
    def clean_json_response(self, response: str) -> str:
        """Clean and validate JSON response.
        
        Args:
            response: Raw response string
            
        Returns:
            Cleaned JSON string
        """
        try:
            # Remove markdown code blocks
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*$', '', response)
            
            # Remove extra whitespace
            response = response.strip()
            
            # Validate JSON
            json.loads(response)
            return response
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return '{"error": "Invalid JSON format"}'
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            return '{"error": "Processing error"}'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted keywords
        """
        try:
            # Simple keyword extraction
            words = re.findall(r'\b\w+\b', text.lower())
            # Filter out common words
            stop_words = {'dan', 'atau', 'yang', 'adalah', 'untuk', 'dengan', 'pada', 'di', 'ke', 'dari'}
            keywords = [word for word in words if len(word) > 2 and word not in stop_words]
            return keywords[:10]  # Limit to 10 keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []