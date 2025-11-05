"""Response generation and formatting for actuarial chat service."""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.utils.preview_extractor import extract_meaningful_preview

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Handles response generation and formatting."""
    
    def __init__(self, llm, prompt_manager):
        """Initialize response generator.
        
        Args:
            llm: Language model instance
            prompt_manager: Prompt manager instance
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
    
    def generate_response(self, question: str, context: str, 
                         calculation_result: Dict[str, Any] = None,
                         response_type: str = "general") -> Dict[str, Any]:
        """Generate response based on question and context.
        
        Args:
            question: User question
            context: Document context
            calculation_result: Calculation result if applicable
            response_type: Type of response (general, calculation, external)
            
        Returns:
            Generated response dictionary
        """
        try:
            if response_type == "calculation" and calculation_result:
                return self._generate_calculation_response(
                    question, context, calculation_result
                )
            elif response_type == "external":
                return self._generate_external_response(question, context)
            else:
                return self._generate_general_response(question, context)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_error_response(str(e))
    
    def format_response_with_sources(self, response: str, sources: List[Dict[str, Any]], 
                                   confidence: float = None) -> Dict[str, Any]:
        """Format response with source information.
        
        Args:
            response: Generated response text
            sources: List of source documents
            confidence: Confidence score
            
        Returns:
            Formatted response dictionary
        """
        try:
            formatted_response = {
                "response": response,
                "sources": self._format_sources(sources),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "confidence": confidence or 0.0,
                    "source_count": len(sources)
                }
            }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return {
                "response": response,
                "sources": [],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.0,
                    "source_count": 0,
                    "error": str(e)
                }
            }
    
    def generate_calculation_summary(self, calculation_result: Dict[str, Any], 
                                   extracted_numbers: List[str],
                                   calculation_type: str) -> str:
        """Generate calculation summary in Indonesian.
        
        Args:
            calculation_result: Calculation result dictionary
            extracted_numbers: List of extracted numbers
            calculation_type: Type of calculation
            
        Returns:
            Calculation summary text
        """
        try:
            summary_parts = []
            
            # Add calculation type
            if calculation_type == "present_value":
                summary_parts.append("Perhitungan nilai sekarang (present value) telah diselesaikan.")
            elif calculation_type == "psak219":
                summary_parts.append("Perhitungan berdasarkan PSAK 219 telah diselesaikan.")
            else:
                summary_parts.append("Perhitungan aktuaria telah diselesaikan.")
            
            # Add extracted numbers if available
            if extracted_numbers:
                numbers_text = ", ".join(extracted_numbers[:3])  # Limit to first 3
                summary_parts.append(f"Angka yang diidentifikasi: {numbers_text}")
            
            # Add calculation steps if available
            if "steps" in calculation_result and calculation_result["steps"]:
                step_count = len(calculation_result["steps"])
                summary_parts.append(f"Perhitungan dilakukan dalam {step_count} langkah.")
            
            # Add confidence if available
            if "confidence" in calculation_result:
                confidence = calculation_result["confidence"]
                if confidence > 0.8:
                    summary_parts.append("Tingkat kepercayaan hasil: Tinggi")
                elif confidence > 0.6:
                    summary_parts.append("Tingkat kepercayaan hasil: Sedang")
                else:
                    summary_parts.append("Tingkat kepercayaan hasil: Rendah")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating calculation summary: {e}")
            return "Perhitungan telah diselesaikan dengan hasil yang tersedia."
    
    def _generate_calculation_response(self, question: str, context: str, 
                                     calculation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for calculation requests.
        
        Args:
            question: User question
            context: Document context
            calculation_result: Calculation result
            
        Returns:
            Calculation response dictionary
        """
        try:
            # Create calculation-specific prompt
            calc_prompt = f"""
            Berdasarkan pertanyaan: {question}
            
            Konteks dokumen: {context[:1000]}...
            
            Hasil perhitungan: {json.dumps(calculation_result, indent=2)}
            
            Berikan penjelasan yang komprehensif tentang hasil perhitungan ini dalam bahasa Indonesia.
            Sertakan metodologi yang digunakan dan interpretasi hasil.
            """
            
            # Generate response using LLM
            response = self.llm.invoke(calc_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "type": "calculation",
                "response": response_text,
                "calculation_result": calculation_result,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating calculation response: {e}")
            return {
                "type": "calculation",
                "response": "Maaf, terjadi kesalahan dalam menghasilkan penjelasan perhitungan.",
                "calculation_result": calculation_result,
                "status": "error",
                "error": str(e)
            }
    
    def _generate_external_response(self, question: str, context: str) -> Dict[str, Any]:
        """Generate response for external document queries.
        
        Args:
            question: User question
            context: Document context
            
        Returns:
            External response dictionary
        """
        try:
            # Get external prompt template
            external_prompt = self.prompt_manager.get_external_prompt()
            
            # Format prompt with question and context
            formatted_prompt = external_prompt.format(
                question=question,
                context=context[:2000]  # Limit context length
            )
            
            # Generate response
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "type": "external",
                "response": response_text,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating external response: {e}")
            return {
                "type": "external",
                "response": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
                "status": "error",
                "error": str(e)
            }
    
    def _generate_general_response(self, question: str, context: str) -> Dict[str, Any]:
        """Generate general response.
        
        Args:
            question: User question
            context: Document context
            
        Returns:
            General response dictionary
        """
        try:
            # Get general prompt template
            general_prompt = self.prompt_manager.get_chain_of_thought_prompt()
            
            # Format prompt
            formatted_prompt = general_prompt.format(
                question=question,
                context=context[:2000]
            )
            
            # Generate response
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "type": "general",
                "response": response_text,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            return {
                "type": "general",
                "response": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
                "status": "error",
                "error": str(e)
            }
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response.
        
        Args:
            error_message: Error message
            
        Returns:
            Error response dictionary
        """
        return {
            "type": "error",
            "response": "Maaf, terjadi kesalahan dalam memproses permintaan Anda. Silakan coba lagi.",
            "status": "error",
            "error": error_message,
            "metadata": {
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for response with full content for AI processing.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Formatted source list with full content for AI processing
        """
        formatted_sources = []
        
        for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
            try:
                # Use full content for AI processing instead of preview
                full_content = source.get("full_content", source.get("content", ""))
                ai_ready_format = source.get("ai_ready_format", "")
                
                # Fallback to old format if new fields not available
                if not full_content and not ai_ready_format:
                    raw_preview = source.get("preview", "")
                    meaningful_preview = extract_meaningful_preview(raw_preview, max_length=50) if raw_preview else "[Tidak ada preview]"
                    display_content = meaningful_preview
                else:
                    # Use AI-ready format or full content
                    display_content = ai_ready_format if ai_ready_format else full_content
                
                formatted_source = {
                    "id": i,
                    "filename": source.get("document_name", source.get("filename", "Unknown")),
                    "section": source.get("section", "General"),
                    "type": source.get("doc_type", "general"),
                    "full_content": full_content,  # Full content for AI processing
                    "ai_ready_format": ai_ready_format,  # Ready-to-use format
                    "display_content": display_content,  # What to show in response
                    "relevance_score": source.get("relevance_score", 0.0)
                }
                
                # Add chunk info if available
                if "chunk_id" in source:
                    formatted_source["chunk_id"] = source["chunk_id"]
                
                formatted_sources.append(formatted_source)
                
            except Exception as e:
                logger.warning(f"Error formatting source {i}: {e}")
                continue
        
        return formatted_sources
    
    def clean_json_response(self, response_text: str) -> Dict[str, Any]:
        """Clean and parse JSON response from LLM.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON dictionary or error dict
        """
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            # Parse JSON
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {
                "error": "Invalid JSON response",
                "raw_response": response_text[:500],
                "parse_error": str(e)
            }
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            return {
                "error": "Error processing response",
                "raw_response": response_text[:500],
                "processing_error": str(e)
            }