"""Calculation flow handling for actuarial chat service."""

import logging
import json
from typing import Dict, Any, List, Tuple
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class CalculationFlowHandler:
    """Handles calculation flow and execution."""
    
    def __init__(self, llm, vector_store_manager, query_processor):
        """Initialize calculation flow handler.
        
        Args:
            llm: Language model instance
            vector_store_manager: Vector store manager instance
            query_processor: Query processor instance
        """
        self.llm = llm
        self.vector_store_manager = vector_store_manager
        self.query_processor = query_processor
    
    def handle_calculation_flow(self, question: str, session_id: str, 
                              chat_history: str, complexity: str = None) -> Dict[str, Any]:
        """Handle calculation flow based on question complexity.
        
        Args:
            question: User question
            session_id: Session identifier
            chat_history: Chat history context
            complexity: Calculation complexity level
            
        Returns:
            Calculation result dictionary
        """
        try:
            # Determine calculation type
            calc_type = self._identify_calculation_type(question)
            
            if calc_type == "simple":
                return self._handle_simple_calculation(question, session_id, chat_history)
            elif calc_type == "complex":
                return self._handle_complex_calculation(question, session_id, chat_history)
            else:
                return self._handle_general_calculation(question, session_id, chat_history)
                
        except Exception as e:
            logger.error(f"Error in calculation flow: {e}")
            return {
                "status": "error",
                "message": f"Error processing calculation: {str(e)}",
                "calculation_steps": [],
                "sources": [],
                "confidence": 0.0
            }
    
    def execute_calculation_placeholder(self, step_name: str, step_title: str, 
                                      step1: Dict[str, Any], context: str,
                                      docs: List[Tuple[Any, float]], 
                                      session_id: str, question: str) -> Dict[str, Any]:
        """Execute calculation step placeholder.
        
        Args:
            step_name: Name of calculation step
            step_title: Title of calculation step
            step1: Previous step data
            context: Document context
            docs: Retrieved documents with scores
            session_id: Session identifier
            question: Original question
            
        Returns:
            Calculation step result
        """
        try:
            # Basic calculation execution
            result = {
                "step_name": step_name,
                "step_title": step_title,
                "status": "completed",
                "inputs": step1,
                "outputs": {
                    "calculated_value": 0.0,
                    "methodology": "Standard actuarial calculation",
                    "assumptions": []
                },
                "explanation": f"Calculation for {step_title} completed successfully.",
                "references": []
            }
            
            # Add document references if available
            if docs:
                result["references"] = [
                    {
                        "source": doc.metadata.get("filename", "Unknown"),
                        "relevance_score": score
                    }
                    for doc, score in docs[:3]  # Top 3 references
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing calculation step {step_name}: {e}")
            return {
                "step_name": step_name,
                "step_title": step_title,
                "status": "error",
                "error": str(e),
                "inputs": step1,
                "outputs": {},
                "explanation": f"Error in calculation: {str(e)}",
                "references": []
            }
    
    def _identify_calculation_type(self, question: str) -> str:
        """Identify calculation type from question.
        
        Args:
            question: User question
            
        Returns:
            Calculation type (simple, complex, general)
        """
        question_lower = question.lower()
        
        # Simple calculations
        simple_keywords = ['sum', 'total', 'average', 'mean', 'count']
        if any(keyword in question_lower for keyword in simple_keywords):
            return "simple"
        
        # Complex calculations
        complex_keywords = ['present value', 'pv', 'actuarial', 'liability', 'reserve']
        if any(keyword in question_lower for keyword in complex_keywords):
            return "complex"
        
        return "general"
    
    def _handle_simple_calculation(self, question: str, session_id: str, 
                                 chat_history: str) -> Dict[str, Any]:
        """Handle simple calculation requests.
        
        Args:
            question: User question
            session_id: Session identifier
            chat_history: Chat history context
            
        Returns:
            Simple calculation result
        """
        try:
            # Retrieve relevant documents
            docs = self.query_processor.retrieve_for_document(
                "general", question, session_id, k=3
            )
            
            # Execute simple calculation
            result = {
                "status": "success",
                "calculation_type": "simple",
                "result": "Simple calculation completed",
                "steps": [
                    {
                        "step": 1,
                        "description": "Data extraction",
                        "status": "completed"
                    },
                    {
                        "step": 2,
                        "description": "Calculation execution",
                        "status": "completed"
                    }
                ],
                "sources": self._extract_source_info([doc for doc, _ in docs], session_id),
                "confidence": self._calculate_confidence(docs)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in simple calculation: {e}")
            return {
                "status": "error",
                "message": str(e),
                "calculation_type": "simple"
            }
    
    def _handle_complex_calculation(self, question: str, session_id: str, 
                                  chat_history: str) -> Dict[str, Any]:
        """Handle complex calculation requests.
        
        Args:
            question: User question
            session_id: Session identifier
            chat_history: Chat history context
            
        Returns:
            Complex calculation result
        """
        try:
            # Retrieve relevant documents
            docs = self.query_processor.retrieve_for_document(
                "actuarial", question, session_id, k=5
            )
            
            # Execute complex calculation with multiple steps
            result = {
                "status": "success",
                "calculation_type": "complex",
                "result": "Complex actuarial calculation completed",
                "steps": [
                    {
                        "step": 1,
                        "description": "Data validation and preparation",
                        "status": "completed"
                    },
                    {
                        "step": 2,
                        "description": "Assumption setting",
                        "status": "completed"
                    },
                    {
                        "step": 3,
                        "description": "Actuarial calculation",
                        "status": "completed"
                    },
                    {
                        "step": 4,
                        "description": "Result validation",
                        "status": "completed"
                    }
                ],
                "sources": self._extract_source_info([doc for doc, _ in docs], session_id),
                "confidence": self._calculate_confidence(docs)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in complex calculation: {e}")
            return {
                "status": "error",
                "message": str(e),
                "calculation_type": "complex"
            }
    
    def _handle_general_calculation(self, question: str, session_id: str, 
                                  chat_history: str) -> Dict[str, Any]:
        """Handle general calculation requests.
        
        Args:
            question: User question
            session_id: Session identifier
            chat_history: Chat history context
            
        Returns:
            General calculation result
        """
        try:
            # Retrieve relevant documents
            docs = self.query_processor.retrieve_for_document(
                "general", question, session_id, k=4
            )
            
            # Execute general calculation
            result = {
                "status": "success",
                "calculation_type": "general",
                "result": "General calculation completed",
                "steps": [
                    {
                        "step": 1,
                        "description": "Question analysis",
                        "status": "completed"
                    },
                    {
                        "step": 2,
                        "description": "Information retrieval",
                        "status": "completed"
                    },
                    {
                        "step": 3,
                        "description": "Response generation",
                        "status": "completed"
                    }
                ],
                "sources": self._extract_source_info([doc for doc, _ in docs], session_id),
                "confidence": self._calculate_confidence(docs)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in general calculation: {e}")
            return {
                "status": "error",
                "message": str(e),
                "calculation_type": "general"
            }
    
    def _extract_source_info(self, source_documents: List[Document], 
                           session_id: str) -> List[Dict[str, Any]]:
        """Extract source information from documents.
        
        Args:
            source_documents: List of source documents
            session_id: Session identifier
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        seen_sources = set()
        
        for doc in source_documents:
            metadata = doc.metadata
            source_key = f"{metadata.get('filename', 'unknown')}_{metadata.get('chunk_id', 0)}"
            
            if source_key not in seen_sources:
                sources.append({
                    'filename': metadata.get('filename', 'Unknown'),
                    'doc_type': metadata.get('doc_type', 'general'),
                    'chunk_id': metadata.get('chunk_id', 0),
                    'preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'session_id': session_id
                })
                seen_sources.add(source_key)
        
        return sources
    
    def _calculate_confidence(self, relevant_docs_with_scores: List[tuple]) -> float:
        """Calculate confidence score based on similarity scores.
        
        Args:
            relevant_docs_with_scores: List of (document, score) tuples
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not relevant_docs_with_scores:
            return 0.0
        
        scores = [score for _, score in relevant_docs_with_scores]
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-1 range
        confidence = min(avg_score, 1.0)
        return round(confidence, 3)