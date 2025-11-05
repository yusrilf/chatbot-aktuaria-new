"""Query Expansion Service for Smart Query Enhancement.

This module provides query expansion functionality using LLM to generate
synonymous questions with same context but different phrasing for better
RAG retrieval performance.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel

from app.config import config
# Import performance monitoring
from app.utils.performance_monitor import get_global_monitor

logger = logging.getLogger(__name__)

class QueryExpansionService:
    """Service for expanding user queries into multiple synonymous variations."""
    
    def __init__(self, llm: BaseLanguageModel):
        """Initialize QueryExpansionService.
        
        Args:
            llm: Language model for generating query expansions
        """
        self.llm = llm
        
        # Initialize performance monitor
        self.performance_monitor = get_global_monitor()
        
        self._init_expansion_chain()
        
        logger.info("QueryExpansionService initialized")
    
    def _init_expansion_chain(self) -> None:
        """Initialize the query expansion chain with prompt template."""
        try:
            expansion_prompt = PromptTemplate(
                input_variables=["original_query", "context_type"],
                template="""Anda adalah ahli aktuaria yang membantu mengoptimalkan pencarian dokumen.

Tugas: Buat HANYA 2 variasi pertanyaan yang memiliki makna dan konteks yang sama dengan pertanyaan asli, tetapi menggunakan kata-kata dan struktur kalimat yang berbeda.

Pertanyaan Asli: {original_query}
Konteks: {context_type}

Pedoman:
1. Pertahankan makna dan tujuan pertanyaan yang sama
2. Gunakan sinonim dan variasi kata yang relevan dengan aktuaria
3. Variasikan struktur kalimat (aktif/pasif, formal/informal)
4. Sertakan istilah teknis aktuaria yang relevan
5. HANYA 2 variasi untuk efisiensi dan kecepatan

Format output dalam JSON:
{{
    "expanded_queries": [
        "variasi pertanyaan 1",
        "variasi pertanyaan 2"
    ],
    "original_query": "{original_query}",
    "expansion_reasoning": "penjelasan singkat mengapa 2 variasi ini dipilih"
}}

JSON Output:"""
            )
            
            self.expansion_chain = LLMChain(
                llm=self.llm,
                prompt=expansion_prompt,
                verbose=False
            )
            
            logger.info("Query expansion chain initialized successfully (2 synonyms mode)")
            
        except Exception as e:
            logger.error(f"Error initializing expansion chain: {str(e)}")
            raise
    
    def expand_query(self, 
                    original_query: str, 
                    context_type: str = "aktuaria",
                    max_retries: int = 2) -> Dict[str, Any]:
        """Expand a single query into 2 synonymous variations for efficiency.
        
        Args:
            original_query: The original user query to expand
            context_type: Context type for better expansion (default: "aktuaria")
            max_retries: Maximum number of retries if parsing fails
            
        Returns:
            Dictionary containing expanded queries and metadata
        """
        # Start timing for query expansion
        expansion_timer = self.performance_monitor.start_timer("query_expansion_total")
        llm_timer = None
        parsing_timer = None
        
        try:
            logger.info(f"Expanding query to 2 synonyms: {original_query[:50]}...")
            
            for attempt in range(max_retries + 1):
                try:
                    # Start timing for LLM inference
                    llm_timer = self.performance_monitor.start_timer("query_expansion_llm_inference")
                    
                    # Generate expansion using LLM
                    response = self.expansion_chain.run(
                        original_query=original_query,
                        context_type=context_type
                    )
                    
                    # Stop LLM inference timer
                    if llm_timer:
                        self.performance_monitor.stop_timer(llm_timer)
                        llm_timer = None
                    
                    # Start timing for response parsing
                    parsing_timer = self.performance_monitor.start_timer("query_expansion_parsing")
                    
                    # Parse JSON response
                    expansion_result = self._parse_expansion_response(response)
                    
                    # Stop parsing timer
                    if parsing_timer:
                        self.performance_monitor.stop_timer(parsing_timer)
                        parsing_timer = None
                    
                    if expansion_result:
                        # Get exactly 2 synonym queries
                        expanded_queries = expansion_result.get("expanded_queries", [])
                        
                        # Ensure we have exactly 2 synonyms (not including original)
                        if len(expanded_queries) > 2:
                            expanded_queries = expanded_queries[:2]
                        elif len(expanded_queries) < 2:
                            # Fill with simple variations if needed
                            while len(expanded_queries) < 2:
                                expanded_queries.append(f"{original_query} aktuaria")
                        
                        # Final query list: original + 2 synonyms = 3 total
                        final_queries = [original_query] + expanded_queries
                        
                        # Stop expansion timer
                        if expansion_timer:
                            self.performance_monitor.stop_timer(expansion_timer)
                        
                        result = {
                            "success": True,
                            "original_query": original_query,
                            "expanded_queries": final_queries,
                            "synonym_queries": expanded_queries,  # Only the 2 synonyms
                            "total_queries": len(final_queries),
                            "expansion_reasoning": expansion_result.get("expansion_reasoning", ""),
                            "context_type": context_type,
                            "attempt": attempt + 1
                        }
                        
                        # Add performance metrics only if enabled in config
                        if config.PERFORMANCE_METRICS_IN_RESPONSE:
                            result["performance_metrics"] = self.performance_monitor.get_metrics()
                        
                        logger.info(f"Query expansion successful: 1 original + 2 synonyms = 3 total queries")
                        return result
                    
                except json.JSONDecodeError as e:
                    # Stop timers on error
                    if llm_timer:
                        self.performance_monitor.stop_timer(llm_timer)
                        llm_timer = None
                    if parsing_timer:
                        self.performance_monitor.stop_timer(parsing_timer)
                        parsing_timer = None
                    
                    logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {str(e)}")
                    
                    if attempt == max_retries:
                        # Use fallback on final attempt
                        fallback_timer = self.performance_monitor.start_timer("query_expansion_fallback")
                        fallback_result = self._create_fallback_expansion(original_query, context_type)
                        self.performance_monitor.stop_timer(fallback_timer)
                        
                        if expansion_timer:
                            self.performance_monitor.stop_timer(expansion_timer)
                        
                        # Add performance metrics only if enabled in config
                        if config.PERFORMANCE_METRICS_IN_RESPONSE:
                            fallback_result["performance_metrics"] = self.performance_monitor.get_metrics()
                        return fallback_result
                        
                except Exception as e:
                    # Stop timers on error
                    if llm_timer:
                        self.performance_monitor.stop_timer(llm_timer)
                        llm_timer = None
                    if parsing_timer:
                        self.performance_monitor.stop_timer(parsing_timer)
                        parsing_timer = None
                    
                    logger.error(f"Error in query expansion attempt {attempt + 1}: {str(e)}")
                    
                    if attempt == max_retries:
                        # Use fallback on final attempt
                        fallback_timer = self.performance_monitor.start_timer("query_expansion_fallback")
                        fallback_result = self._create_fallback_expansion(original_query, context_type)
                        self.performance_monitor.stop_timer(fallback_timer)
                        
                        if expansion_timer:
                            self.performance_monitor.stop_timer(expansion_timer)
                        
                        # Add performance metrics only if enabled in config
                        if config.PERFORMANCE_METRICS_IN_RESPONSE:
                            fallback_result["performance_metrics"] = self.performance_monitor.get_metrics()
                        return fallback_result
            
            # If all attempts failed, use fallback
            fallback_timer = self.performance_monitor.start_timer("query_expansion_fallback")
            fallback_result = self._create_fallback_expansion(original_query, context_type)
            self.performance_monitor.stop_timer(fallback_timer)
            
            if expansion_timer:
                self.performance_monitor.stop_timer(expansion_timer)
            
            # Add performance metrics only if enabled in config
            if config.PERFORMANCE_METRICS_IN_RESPONSE:
                fallback_result["performance_metrics"] = self.performance_monitor.get_metrics()
            return fallback_result
            
        except Exception as e:
            # Stop any active timers on critical error
            if llm_timer:
                self.performance_monitor.stop_timer(llm_timer)
            if parsing_timer:
                self.performance_monitor.stop_timer(parsing_timer)
            if expansion_timer:
                self.performance_monitor.stop_timer(expansion_timer)
                
            logger.error(f"Critical error in query expansion: {str(e)}")
            
            # Return fallback result with error info
            fallback_result = self._create_fallback_expansion(original_query, context_type)
            fallback_result["error"] = str(e)
            
            # Add performance metrics only if enabled in config
            if config.PERFORMANCE_METRICS_IN_RESPONSE:
                fallback_result["performance_metrics"] = self.performance_monitor.get_metrics()
            return fallback_result
    
    def _parse_expansion_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response to extract exactly 2 expanded queries.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed expansion result or None if parsing fails
        """
        try:
            # Clean response - remove markdown formatting if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            # Parse JSON
            parsed_result = json.loads(cleaned_response.strip())
            
            # Validate required fields
            if "expanded_queries" not in parsed_result:
                logger.error("Missing 'expanded_queries' field in response")
                return None
            
            if not isinstance(parsed_result["expanded_queries"], list):
                logger.error("'expanded_queries' must be a list")
                return None
            
            # Ensure exactly 2 synonyms
            expanded_queries = parsed_result["expanded_queries"]
            if len(expanded_queries) < 1:
                logger.error("Need at least 1 expanded query")
                return None
            
            # Limit to 2 synonyms maximum
            if len(expanded_queries) > 2:
                parsed_result["expanded_queries"] = expanded_queries[:2]
                logger.info(f"Limited to 2 synonyms from {len(expanded_queries)} generated")
            
            return parsed_result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Response content: {response[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Error parsing expansion response: {str(e)}")
            return None
    
    def _create_fallback_expansion(self, original_query: str, context_type: str) -> Dict[str, Any]:
        """Create fallback expansion when LLM expansion fails.
        
        Args:
            original_query: Original query
            context_type: Context type
            
        Returns:
            Fallback expansion result
        """
        try:
            # Simple rule-based variations
            variations = [
                original_query,
                f"Bagaimana cara {original_query.lower()}",
                f"Jelaskan tentang {original_query.lower()}",
                f"Apa yang dimaksud dengan {original_query.lower()}"
            ]
            
            # Remove duplicates while preserving order
            unique_variations = []
            for var in variations:
                if var not in unique_variations:
                    unique_variations.append(var)
            
            # Ensure we have exactly 4 queries
            while len(unique_variations) < 4:
                unique_variations.append(original_query)
            
            result = {
                "success": True,
                "original_query": original_query,
                "expanded_queries": unique_variations[:4],
                "total_queries": 4,
                "expansion_reasoning": "Fallback expansion using rule-based variations",
                "context_type": context_type,
                "fallback_used": True
            }
            
            logger.info("Fallback expansion created successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error creating fallback expansion: {str(e)}")
            return {
                "success": False,
                "original_query": original_query,
                "expanded_queries": [original_query],
                "total_queries": 1,
                "expansion_reasoning": "Error in expansion, using original query only",
                "context_type": context_type,
                "error": str(e)
            }
    
    def batch_expand_queries(self, 
                           queries: List[str], 
                           context_type: str = "aktuaria") -> List[Dict[str, Any]]:
        """Expand multiple queries in batch.
        
        Args:
            queries: List of queries to expand
            context_type: Context type for expansion
            
        Returns:
            List of expansion results
        """
        try:
            logger.info(f"Batch expanding {len(queries)} queries")
            
            results = []
            for i, query in enumerate(queries):
                logger.info(f"Expanding query {i+1}/{len(queries)}")
                result = self.expand_query(query, context_type)
                results.append(result)
            
            logger.info(f"Batch expansion completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch query expansion: {str(e)}")
            return []
    
    def get_expansion_stats(self, expansion_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about query expansion result.
        
        Args:
            expansion_result: Result from expand_query method
            
        Returns:
            Statistics dictionary
        """
        try:
            if not expansion_result.get("success", False):
                return {"error": "Expansion was not successful"}
            
            expanded_queries = expansion_result.get("expanded_queries", [])
            original_query = expansion_result.get("original_query", "")
            
            stats = {
                "total_queries": len(expanded_queries),
                "original_length": len(original_query),
                "avg_query_length": sum(len(q) for q in expanded_queries) / len(expanded_queries) if expanded_queries else 0,
                "unique_queries": len(set(expanded_queries)),
                "fallback_used": expansion_result.get("fallback_used", False),
                "context_type": expansion_result.get("context_type", ""),
                "expansion_reasoning": expansion_result.get("expansion_reasoning", "")
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating expansion stats: {str(e)}")
            return {"error": str(e)}