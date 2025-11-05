"""Intent classification module for chat service.

This module handles the classification of user intents into theory or calculation flows.
"""

from typing import Dict, Any, Optional
import json
import re
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# Safe optional import to avoid ModuleNotFoundError during tests
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  # type: ignore

from app.services.parsers.response_parser import ResponseParser

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Handles intent classification for user questions."""
    
    def __init__(self, llm: Optional[ChatOpenAI], response_parser: ResponseParser):
        """Initialize the intent classifier.
        
        Args:
            llm: The language model for classification (optional)
            response_parser: Parser for LLM responses
        """
        self.llm = llm
        self.response_parser = response_parser
        self.session_state = {}
        
    def is_psak219_data_question(self, question: str, session_id: Optional[str] = None) -> bool:
        """Check if question is asking for data that already exists in parsed PSAK219 documents.
        
        Args:
            question: User question to analyze
            session_id: Optional session identifier
            
        Returns:
            True if question is asking for existing PSAK219 data
        """
        try:
            question_lower = question.lower()
            logger.info(f"Checking if PSAK219 data question: '{question_lower}'")
            
            # Keywords that indicate data retrieval from PSAK219
            psak219_data_keywords = [
                'current service cost', 'biaya jasa kini', 'csc',
                'present value obligation', 'pvdbo', 'kewajiban nilai sekarang',
                'present value benefit', 'pvfb', 'manfaat nilai sekarang',
                'interest cost', 'biaya bunga',
                'past service cost', 'biaya jasa lalu',
                'actuarial gain', 'actuarial loss', 'keuntungan aktuaria', 'kerugian aktuaria',
                'discount rate', 'tingkat diskonto',
                'salary increase', 'kenaikan gaji',
                'mortality rate', 'tingkat kematian',
                'withdrawal rate', 'tingkat keluar',
                'nama perusahaan', 'company name',
                'tanggal laporan', 'report date',
                'periode laporan', 'reporting period',
                'asumsi', 'assumptions', 'asumsi aktuaria', 'actuarial assumptions',
                'jumlah karyawan', 'total employees', 'karyawan tetap', 'permanent employee',
                'karyawan kontrak', 'contract employee', 'employee benefit', 'imbalan karyawan'
            ]
            
            # Question patterns that indicate data lookup (not calculation)
            lookup_patterns = [
                r'berapa\s+.*?(current service cost|csc|biaya jasa kini)',
                r'berapa\s+.*?(present value|pvdbo|pvfb)',
                r'berapa\s+.*?(discount rate|tingkat diskonto)',
                r'berapa\s+.*?(salary increase|kenaikan gaji)',
                r'berapa\s+.*?(jumlah karyawan|total employees)',
                r'apa\s+.*?(nama perusahaan|company name)',
                r'apa\s+.*?(asumsi|assumptions)',
                r'kapan\s+.*?(tanggal|periode)',
                r'siapa\s+.*?(perusahaan|company)'
            ]
            
            # Check if question contains PSAK219 data keywords
            has_psak219_keywords = any(keyword in question_lower for keyword in psak219_data_keywords)
            logger.info(f"Has PSAK219 keywords: {has_psak219_keywords}")
            
            # Check if question matches lookup patterns
            has_lookup_pattern = any(re.search(pattern, question_lower) for pattern in lookup_patterns)
            logger.info(f"Has lookup pattern: {has_lookup_pattern}")
            
            # Check if question is asking for existing data (not new calculations)
            calculation_indicators = [
                'hitung', 'kalkulasi', 'simulasi', 'proyeksi', 'analisis sensitivitas',
                'dengan asumsi', 'jika', 'bagaimana jika', 'bandingkan'
            ]
            
            has_calculation_indicators = any(indicator in question_lower for indicator in calculation_indicators)
            logger.info(f"Has calculation indicators: {has_calculation_indicators}")
            
            # Return True if it's a PSAK219 data question without calculation indicators
            result = (has_psak219_keywords or has_lookup_pattern) and not has_calculation_indicators
            logger.info(f"PSAK219 data question result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error checking PSAK219 data question: {str(e)}")
            return False
    
    def classify_intent(self, question: str, session_id: Optional[str] = None, 
                      chat_history: str = "") -> Dict[str, Any]:
        """Classify user intent using LLM to determine theory vs calculation flow.
        
        Args:
            question: User question to classify
            session_id: Optional session identifier
            chat_history: Previous conversation context
            
        Returns:
            Dictionary with intent, complexity, and reasoning
        """
        try:
            # Heuristic fast path if LLM is unavailable
            if not self.llm:
                q_lower = question.lower()
                if self.is_psak219_data_question(question, session_id):
                    return {
                        "intent": "theory",
                        "complexity": "simple",
                        "reasoning": "PSAK219 data lookup detected without calculation indicators (LLM unavailable)",
                    }
                # Simple keyword-based detection
                calc_keywords = ['hitung', 'perhitungan', 'simulasi', 'proyeksi', 'sensitiv', 'analisis', 'pvfb', 'pvdbo', 'diskonto']
                if any(kw in q_lower for kw in calc_keywords):
                    complexity = 'complex' if any(kw in q_lower for kw in ['simulasi', 'proyeksi', 'sensitiv']) else 'moderate'
                    return {
                        "intent": "calculation",
                        "complexity": complexity,
                        "reasoning": "Heuristic classification due to missing LLM",
                    }
                return {
                    "intent": "theory",
                    "complexity": "moderate",
                    "reasoning": "Default to theory using heuristics (LLM unavailable)",
                }
            
            # Check for PSAK219 data retrieval questions first
            if self.is_psak219_data_question(question, session_id):
                return {
                    "intent": "theory",
                    "complexity": "simple",
                    "reasoning": "PSAK219 data retrieval question - can be answered directly from parsed data"
                }
            
            # Get previous classification for context
            prev_classification = None
            if session_id and session_id in self.session_state:
                state = self.session_state[session_id]
                prev_classification = state.get("intent_classification", {}).get("parsed")
            
            # Create classification prompt - ONLY 2 ROUTES: theory or calculation
            classification_template = """
Anda adalah asisten aktuaria expert. Klasifikasikan pertanyaan pengguna dengan SANGAT TELITI:

1) intent: HANYA salah satu dari [theory, calculation] - TIDAK ADA 'other'
2) complexity: salah satu dari [simple, moderate, complex, troubleshoot]

PENJELASAN DETAIL:
- theory: konsep, definisi, metode, penjelasan teori aktuaria, pertanyaan umum tentang dokumen
  * Kata kunci theory: apa itu, definisi, penjelasan, bagaimana, metode, konsep, teori
  * SEMUA pertanyaan tentang dokumen yang tidak meminta perhitungan = theory
  * PERTANYAAN TENTANG DATA PSAK219 YANG SUDAH ADA = theory (bukan calculation)
- calculation: HANYA pertanyaan yang meminta perhitungan BARU, simulasi, analisis kuantitatif
  * Kata kunci calculation: hitung, perhitungan, simulasi, proyeksi, sensitivitas, analisis
  * BUKAN untuk pertanyaan yang hanya meminta nilai yang sudah ada di dokumen

COMPLEXITY LEVELS:
- simple: lookup langsung, satu nilai, tabel sederhana, pertanyaan definisi
- moderate: perhitungan satu langkah atau teori dengan beberapa konsep  
- complex: perhitungan multi-step, analisis lengkap, sensitivitas, PVFB/PVDBO
- troubleshoot: diagnosis error, validasi, investigasi masalah

CONTOH CLASSIFICATION:
- "Berapa current service cost karyawan tetap?" → theory, simple (data lookup)
- "Berapa present value obligation?" → theory, simple (data lookup)
- "Hitung PVFB untuk karyawan usia 30 dengan asumsi baru" → calculation, complex
- "Simulasi sensitivitas diskonto 5% vs 7%" → calculation, complex
- "Apa itu PVFB?" → theory, simple
- "Jelaskan tentang dokumen PSAK 219" → theory, moderate

RIWAYAT PERCAKAPAN:
{history}

KLASIFIKASI SEBELUMNYA:
{prev_classification}

PERTANYAAN: {question}

Output HANYA JSON valid:
{{"intent": "theory/calculation", "complexity": "simple/moderate/complex/troubleshoot", "reasoning": "alasan spesifik dengan kata kunci yang ditemukan"}}
"""
            
            prompt = PromptTemplate(
                input_variables=["question", "history", "prev_classification"],
                template=classification_template
            )
            
            llm_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=False)
            raw_response = llm_chain.run({
                "question": question,
                "history": chat_history or "",
                "prev_classification": json.dumps(prev_classification) if prev_classification else ""
            })
            
            # Parse response
            parsed = self.response_parser.parse_json_response(raw_response)
            
            # Validate and normalize - ONLY 2 ROUTES
            intent = parsed.get("intent", "").strip().lower()
            complexity = parsed.get("complexity", "moderate").strip().lower()
            reasoning = parsed.get("reasoning", "")
            
            # Fallback validation - ONLY theory or calculation
            if intent not in ["theory", "calculation"]:
                # Keyword-based fallback
                q_lower = question.lower()
                if any(kw in q_lower for kw in ['hitung', 'perhitungan', 'nilai', 'berapa', 'pvfb', 'pvdbo', 'diskonto', 'sensitiv', 'analisis']):
                    intent = "calculation"
                else:
                    # Default to theory for all other questions including documents
                    intent = "theory"
            
            if complexity not in ["simple", "moderate", "complex", "troubleshoot"]:
                complexity = "moderate"
            
            result = {
                "intent": intent,
                "complexity": complexity,
                "reasoning": reasoning,
                "raw_response": raw_response
            }
            
            # Store in session state
            if session_id:
                if session_id not in self.session_state:
                    self.session_state[session_id] = {}
                state = self.session_state[session_id]
                state["intent_classification"] = {"parsed": result}
            
            return result
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Fallback classification - default to theory
            return {
                "intent": "theory",
                "complexity": "moderate",
                "reasoning": f"Fallback due to error: {str(e)}",
                "raw_response": ""
            }