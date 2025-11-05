"""Response parsing utilities for actuarial chatbot."""

import json
import re
import logging
from typing import Dict, Any, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from app.config import config

logger = logging.getLogger(__name__)

class ResponseParser:
    """Handles parsing and cleaning of LLM responses."""
    
    def __init__(self, llm=None):
        """Initialize the response parser."""
        self.llm = llm
    
    def parse_cot_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Robust parser for Chain-of-Thought responses.
        - Try local regex/json parsing first.
        - If fails, ask the LLM (deterministic) to repair/extract a valid JSON object with keys:
        'thought', 'answer', 'sources'.
        - Retry once with a stricter fallback prompt.
        - If still fails, use heuristic extraction and return fallback dict (never raise).
        """
        try:
            # --- 1) try current direct extraction patterns first (fast path) ---
            patterns = [
                r"```json\s*({.*?})\s*```",
                r"```.*?({.*?})\s*```",
                r"(\{[\s\S]*\})",
                r"\{\{(.*?)\}\}",
            ]
            for pattern in patterns:
                match = re.search(pattern, raw_response, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    if pattern == r"\{\{(.*?)\}\}":
                        json_str = "{" + json_str + "}"
                    # quick fixes
                    json_str = json_str.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
                    try:
                        parsed = json.loads(json_str)
                        if not all(k in parsed for k in ("thought", "answer", "sources")):
                            raise ValueError("JSON tidak lengkap")
                        return parsed
                    except Exception:
                        # continue to repair path
                        break

            # --- 2) fallback: ask the LLM to repair/extract a valid JSON ---
            if self.llm:
                try:
                    # keep repair prompt deterministic
                    repair_prompt = PromptTemplate(
                        input_variables=["raw"],
                        template=(
                            "Dari teks berikut, keluarkan satu objek JSON valid dengan tiga kunci saja: \"thought\", \"answer\", dan \"sources\".\n\n"
                            "Ketentuan:\n"
                            "- Jangan keluarkan teks tambahan selain JSON.\n"
                            "- Jika sumber tidak ada, isi \"sources\": {}.\n"
                            "- Semua string harus pakai tanda kutip ganda (\").\n"
                            f"- Panjang 'answer' maksimal {config.MAX_CONTEXT_LENGTH} karakter.\n\n"
                            "Contoh:\n"
                            "{\"thought\":\"...\",\"answer\":\"...\",\"sources\":{}}\n\n"
                            "Teks:\n"
                            "{raw}"
                        )
                    )

                    llm_chain = LLMChain(llm=self.llm, prompt=repair_prompt, verbose=False)
                    # send truncated raw_response to avoid huge prompts
                    safe_raw = (raw_response or "")[:3200]
                    repaired = llm_chain.run({"raw": safe_raw})
                    # attempt to find JSON in LLM reply
                    m2 = re.search(r"```json\s*({[\s\S]*})\s*```", repaired, re.DOTALL) or re.search(r"({[\s\S]*})", repaired, re.DOTALL)
                    if m2:
                        candidate = m2.group(1)
                        candidate = candidate.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
                        try:
                            parsed = json.loads(candidate)
                            if not all(k in parsed for k in ("thought", "answer", "sources")):
                                # fall through to retry
                                raise ValueError("Repaired JSON missing keys")
                            return parsed
                        except Exception:
                            # continue to next attempt
                            pass
                except Exception as e_repair:
                    logger.debug(f"parse_cot_response: LLM repair attempt failed: {e_repair}")

                # --- 3) second LLM attempt (stricter, return fields separately) ---
                try:
                    fallback_prompt = PromptTemplate(
                        input_variables=["raw"],
                        template=(
                            "Jika Anda tidak bisa mengeluarkan JSON yang kompleks, tolong EXTRACT tiga bagian "
                            "dari input RAW berikut dan keluarkan HANYA dalam format yang mudah diparse:\n\n"
                            "THOUGHT:\n<short thought>\n\nANSWER:\n<long answer (max " + str(config.MAX_CONTEXT_LENGTH) + " chars)>\n\nSOURCES:\n<format: filename: isi dokumen lengkap...>\n\n"
                            "RAW INPUT:\n{raw}\n\n"
                            "Important: output exactly the labels THOUGHT, ANSWER, SOURCES as shown. "
                            "For SOURCES, use format 'filename: full document content...' instead of JSON format."
                        )
                    )
                    llm_chain2 = LLMChain(llm=self.llm, prompt=fallback_prompt, verbose=False)
                    safe_raw = (raw_response or "")[:3200]
                    alt = llm_chain2.run({"raw": safe_raw})
                    # extract blocks
                    thought_m = re.search(r"THOUGHT:\s*(.*?)\n\s*\n", alt, re.DOTALL)
                    answer_m = re.search(r"ANSWER:\s*(.*?)\n\s*\nSOURCES:", alt, re.DOTALL)
                    sources_m = re.search(r"SOURCES:\s*(\{[\s\S]*\})", alt, re.DOTALL)
                    thought = thought_m.group(1).strip() if thought_m else ""
                    answer = answer_m.group(1).strip() if answer_m else (alt.strip()[:config.MAX_CONTEXT_LENGTH])
                    sources = {}
                    if sources_m:
                        try:
                            src = sources_m.group(1).replace("'", '"')
                            src = re.sub(r",\s*([\]\}])", r"\1", src)
                            sources = json.loads(src)
                        except Exception:
                            sources = {}
                    return {"thought": thought, "answer": answer, "sources": sources}
                except Exception as e_alt:
                    logger.debug(f"parse_cot_response: LLM fallback extraction failed: {e_alt}")

            # --- 4) final heuristic fallback: best-effort regex & raw return (never raise) ---
            try:
                # try to extract simple quoted fields from the raw_response
                thought_m = re.search(r'"?thought"?\s*:\s*"(.*?)"\s*(?:,|\})', raw_response, re.DOTALL)
                answer_m = re.search(r'"?answer"?\s*:\s*"(.*?)"\s*(?:,|\})', raw_response, re.DOTALL)
                sources_m = re.search(r'"?sources"?\s*:\s*(\{[\s\S]*?\})\s*(?:,|\})', raw_response, re.DOTALL)

                thought = thought_m.group(1).strip() if thought_m else ""
                answer = answer_m.group(1).strip() if answer_m else (raw_response or "")[:config.MAX_CONTEXT_LENGTH]
                sources = {}
                if sources_m:
                    try:
                        src = sources_m.group(1).replace("'", '"')
                        src = re.sub(r",\s*([\]\}])", r"\1", src)
                        sources = json.loads(src)
                    except Exception:
                        sources = {}

                return {"thought": thought, "answer": answer, "sources": sources}
            except Exception as e_final:
                logger.error(f"parse_cot_response final fallback failed: {e_final}")
                # absolute fallback
                return {"thought": "", "answer": (raw_response or "")[:config.MAX_CONTEXT_LENGTH], "sources": {}}

        except Exception as e:
            logger.error(f"parse_cot_response unexpected fatal: {e}")
            return {"thought": "", "answer": (raw_response or "")[:config.MAX_CONTEXT_LENGTH], "sources": {}}
    
    def clean_json_response(self, response: str) -> str:
        """Clean and normalize JSON response string."""
        try:
            # Remove common formatting issues
            cleaned = response.strip()
            
            # Fix common JSON issues
            cleaned = re.sub(r'\\n', '\n', cleaned)
            cleaned = re.sub(r'\\"', '"', cleaned)
            cleaned = re.sub(r'"\s*:\s*"([^"]*?)"\s*,?\s*}', r'":\"\1"}', cleaned)
            
            # Remove trailing commas
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            return response
    
    def clean_number_token(self, s: str) -> Optional[float]:
        """Normalize numeric token like '(682,186,554)' or '682.186.554' or '6.98%' -> float."""
        try:
            if s is None:
                return None
            s = str(s).strip()
            # detect percent
            is_pct = s.endswith('%')
            s = s.replace('%', '')
            # detect negative via parentheses or leading minus
            is_negative = False
            if s.startswith('(') and s.endswith(')'):
                is_negative = True
                s = s[1:-1]
            s = s.strip()
            # remove currency markers
            s = re.sub(r'Rp\.?|IDR', '', s, flags=re.IGNORECASE).strip()
            # remove spaces
            s = s.replace(' ', '')
            
            # Handle different decimal separators
            if ',' in s and '.' in s:
                # Assume comma is thousands separator
                s = s.replace(',', '')
            elif ',' in s:
                # Could be decimal separator (European style)
                if s.count(',') == 1 and len(s.split(',')[1]) <= 2:
                    s = s.replace(',', '.')
                else:
                    # Thousands separator
                    s = s.replace(',', '')
            
            try:
                result = float(s)
                if is_negative:
                    result = -result
                if is_pct:
                    result = result / 100
                return result
            except ValueError:
                return None
                
        except Exception as e:
            logger.debug(f"clean_number_token error for '{s}': {e}")
            return None
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with robust fallback handling.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            Dict containing parsed JSON or fallback structure
        """
        if not response or not response.strip():
            logger.warning("Empty response received")
            return {
                "intent": "theory",
                "complexity": "moderate", 
                "reasoning": "Empty response received"
            }
            
        # Multiple parsing attempts with increasing robustness
        parsing_attempts = [
            self._attempt_direct_json_parse,
            self._attempt_regex_json_extract,
            self._attempt_json_repair,
            self._attempt_llm_json_repair
        ]
        
        for attempt_func in parsing_attempts:
            try:
                result = attempt_func(response)
                if result and self._validate_intent_response(result):
                    return result
            except Exception as e:
                logger.debug(f"Parsing attempt {attempt_func.__name__} failed: {e}")
                continue
                
        # Final fallback with keyword detection
        logger.warning(f"All JSON parsing attempts failed for response: {response[:200]}...")
        return self._keyword_based_fallback(response)
    
    def _attempt_direct_json_parse(self, response: str) -> Optional[Dict[str, Any]]:
        """Attempt direct JSON parsing."""
        cleaned = response.strip()
        return json.loads(cleaned)
    
    def _attempt_regex_json_extract(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON using regex patterns."""
        patterns = [
            r'```json\s*({.*?})\s*```',
            r'```\s*({.*?})\s*```', 
            r'({\s*"[^"]+"\s*:[^}]+})',
            r'({.*?})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                json_str = self._clean_json_string(json_str)
                return json.loads(json_str)
        return None
    
    def _attempt_json_repair(self, response: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair common JSON issues."""
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            return None
            
        json_str = json_match.group(0)
        json_str = self._clean_json_string(json_str)
        
        # Additional repairs for delimiter issues
        json_str = self._fix_delimiter_issues(json_str)
        
        return json.loads(json_str)
    
    def _attempt_llm_json_repair(self, response: str) -> Optional[Dict[str, Any]]:
        """Use LLM to repair malformed JSON if available."""
        if not self.llm:
            return None
            
        try:
            repair_prompt = PromptTemplate(
                input_variables=["broken_json"],
                template="""Fix this broken JSON and return ONLY valid JSON:
                
{broken_json}
                
Return format: {{"intent": "theory/calculation", "complexity": "simple/moderate/complex", "reasoning": "explanation"}}"""
            )
            
            llm_chain = LLMChain(llm=self.llm, prompt=repair_prompt, verbose=False)
            repaired = llm_chain.run({"broken_json": response[:500]})  # Limit input size
            
            # Try to extract JSON from LLM response
            return self._attempt_regex_json_extract(repaired)
            
        except Exception as e:
            logger.debug(f"LLM repair failed: {e}")
            return None
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues.
        
        Args:
            json_str: Raw JSON string
            
        Returns:
            Cleaned JSON string
        """
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        
        # Fix unquoted keys
        json_str = re.sub(r'(\w+)\s*:', r'"\1":', json_str)
        
        # Fix Python boolean/null values
        json_str = json_str.replace('True', 'true')
        json_str = json_str.replace('False', 'false') 
        json_str = json_str.replace('None', 'null')
        
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str.strip()
    
    def _fix_delimiter_issues(self, json_str: str) -> str:
        """Fix common delimiter issues in JSON.
        
        Args:
            json_str: JSON string with potential delimiter issues
            
        Returns:
            JSON string with fixed delimiters
        """
        # Fix missing commas between key-value pairs
        json_str = re.sub(r'"\s*"\s*([a-zA-Z_])', r'", "\1', json_str)
        
        # Fix missing commas after values
        json_str = re.sub(r'"\s*"\s*([a-zA-Z_])', r'", "\1', json_str)
        json_str = re.sub(r'(["\d])\s*"([a-zA-Z_])', r'\1, "\2', json_str)
        
        # Fix double commas
        json_str = re.sub(r',,+', ',', json_str)
        
        # Fix spaces around colons and commas
        json_str = re.sub(r'\s*:\s*', ': ', json_str)
        json_str = re.sub(r'\s*,\s*', ', ', json_str)
        
        return json_str
    
    def _validate_intent_response(self, response: Dict[str, Any]) -> bool:
        """Validate that response has required fields for intent classification.
        
        Args:
            response: Parsed response dictionary
            
        Returns:
            True if response is valid for intent classification
        """
        required_fields = ['intent', 'complexity']
        return all(field in response for field in required_fields)
    
    def parse_custom_response(self, raw_response: str, relevant_docs: list, session_id: str = None) -> Dict[str, Any]:
        """Parse custom response from LLM and format with sources.
        
        Args:
            raw_response: Raw LLM response
            relevant_docs: List of relevant documents with scores
            
        Returns:
            Formatted response dictionary
        """
        try:
            # Extract sources from relevant documents
            sources = []
            for doc, score in relevant_docs:
                metadata = doc.metadata or {}
                source = metadata.get('source', 'Unknown')
                
                # Create meaningful preview from document content
                preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                
                # Format score untuk readability
                formatted_score = float(f"{score:.6f}") if score < 0.001 else round(score, 4)
                
                source_info = {
                    'source': source,
                    'score': formatted_score,
                    'preview': preview.strip(),
                    'relevance': 'high' if score > 0.8 else 'medium' if score > 0.5 else 'low',
                    'session_id': session_id
                }
                sources.append(source_info)
            
            # Calculate confidence based on document scores
            if relevant_docs:
                avg_score = sum(score for _, score in relevant_docs) / len(relevant_docs)
                confidence = min(95, max(60, int(avg_score * 100)))
            else:
                confidence = 50
            
            return {
                'answer': raw_response.strip(),
                'confidence': confidence,
                'sources': sources,
                'type': 'custom_analysis',
                'metadata': {
                    'processing_type': 'custom',
                    'documents_used': len(relevant_docs),
                    'response_length': len(raw_response)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing custom response: {str(e)}")
            # Fallback response
            return {
                'answer': raw_response.strip() if raw_response else "Maaf, terjadi kesalahan dalam memproses respons.",
                'confidence': 50,
                'sources': [],
                'type': 'custom_analysis',
                'metadata': {
                    'processing_type': 'custom_fallback',
                    'documents_used': len(relevant_docs) if relevant_docs else 0,
                    'error': str(e)
                }
            }
    
    def _keyword_based_fallback(self, response: str) -> Dict[str, Any]:
        """Generate fallback response based on keyword detection.
        
        Args:
            response: Original response string
            
        Returns:
            Fallback intent classification
        """
        response_lower = response.lower()
        
        # Check for calculation keywords
        calc_keywords = ['hitung', 'perhitungan', 'nilai', 'berapa', 'pvfb', 'pvdbo', 'diskonto', 'sensitiv', 'analisis']
        if any(keyword in response_lower for keyword in calc_keywords):
            intent = "calculation"
        else:
            intent = "theory"  # Default to theory instead of 'other'
            
        return {
            "intent": intent,
            "complexity": "moderate",
            "reasoning": f"Keyword-based fallback classification due to JSON parsing failure"
        }