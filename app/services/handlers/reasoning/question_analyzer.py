"""Question Analyzer untuk Tree of Thought Document Reader.

Analyzer yang mengidentifikasi jenis pertanyaan, domain keywords,
dan kebutuhan dokumen spesifik untuk pertanyaan aktuaria.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json

logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """Enum untuk jenis pertanyaan."""
    THEORY = "theory"  # Pertanyaan konseptual/teoritis
    CALCULATION = "calculation"  # Pertanyaan perhitungan
    DATA_LOOKUP = "data_lookup"  # Pencarian data/nilai spesifik
    COMPARISON = "comparison"  # Perbandingan konsep/metode
    PROCEDURE = "procedure"  # Langkah-langkah prosedur
    INTERPRETATION = "interpretation"  # Interpretasi hasil/data

class ActuarialDomain(Enum):
    """Enum untuk domain aktuaria."""
    LIFE_INSURANCE = "life_insurance"
    GENERAL_INSURANCE = "general_insurance"
    PENSION = "pension"
    INVESTMENT = "investment"
    RISK_MANAGEMENT = "risk_management"
    STATISTICS = "statistics"
    FINANCIAL_MATH = "financial_math"
    REGULATION = "regulation"
    GENERAL = "general"

@dataclass
class QuestionContext:
    """Context information dari pertanyaan."""
    has_numbers: bool
    has_formulas: bool
    has_comparisons: bool
    has_procedures: bool
    complexity_indicators: List[str]
    temporal_context: Optional[str]  # past, present, future

@dataclass
class DocumentRequirement:
    """Kebutuhan dokumen untuk pertanyaan."""
    document_types: List[str]  # regulation, manual, example, etc.
    priority_keywords: List[str]
    section_hints: List[str]  # hints untuk section yang relevan
    scope_preference: str  # global, session, both

class QuestionAnalyzer:
    """Analyzer untuk menganalisis pertanyaan aktuaria secara mendalam."""
    
    def __init__(self, llm=None):
        """
        Initialize Question Analyzer.
        
        Args:
            llm: Language model untuk analisis lanjutan (optional)
        """
        self.llm = llm
        
        # Pattern untuk deteksi jenis pertanyaan
        self.question_patterns = {
            QuestionType.CALCULATION: [
                r'hitung|kalkulasi|perhitungan|rumus|formula',
                r'berapa|nilai|jumlah|total|hasil',
                r'\d+.*%|\d+.*tahun|\d+.*bulan',
                r'premi|cadangan|nilai tunai|surrender'
            ],
            QuestionType.THEORY: [
                r'apa itu|definisi|pengertian|konsep',
                r'jelaskan|terangkan|uraikan',
                r'mengapa|kenapa|alasan',
                r'bagaimana.*bekerja|cara.*kerja'
            ],
            QuestionType.DATA_LOOKUP: [
                r'tabel|data|daftar|list',
                r'cari|temukan|lihat',
                r'nilai.*untuk|angka.*untuk',
                r'berapa.*tarif|berapa.*rate'
            ],
            QuestionType.COMPARISON: [
                r'bandingkan|versus|vs|dibanding',
                r'perbedaan|beda|berbeda',
                r'mana yang.*baik|lebih.*dari',
                r'kelebihan.*kekurangan'
            ],
            QuestionType.PROCEDURE: [
                r'langkah|tahap|prosedur|cara',
                r'bagaimana.*melakukan|how to',
                r'proses.*untuk|metode.*untuk',
                r'urutan|sequence'
            ],
            QuestionType.INTERPRETATION: [
                r'arti|makna|interpretasi',
                r'maksud|berarti|menunjukkan',
                r'analisis|evaluasi|assessment',
                r'kesimpulan|implikasi'
            ]
        }
        
        # Keywords untuk domain aktuaria
        self.domain_keywords = {
            ActuarialDomain.LIFE_INSURANCE: [
                'asuransi jiwa', 'life insurance', 'premi', 'polis',
                'nilai tunai', 'surrender value', 'endowment',
                'whole life', 'term life', 'unit link'
            ],
            ActuarialDomain.GENERAL_INSURANCE: [
                'asuransi umum', 'general insurance', 'kendaraan',
                'properti', 'kebakaran', 'gempa', 'banjir',
                'liability', 'marine', 'aviation'
            ],
            ActuarialDomain.PENSION: [
                'pensiun', 'pension', 'dana pensiun', 'iuran',
                'manfaat pensiun', 'anuitas', 'vesting',
                'defined benefit', 'defined contribution'
            ],
            ActuarialDomain.INVESTMENT: [
                'investasi', 'investment', 'portofolio', 'return',
                'yield', 'bond', 'saham', 'obligasi',
                'asset allocation', 'diversifikasi'
            ],
            ActuarialDomain.RISK_MANAGEMENT: [
                'manajemen risiko', 'risk management', 'var',
                'stress test', 'scenario', 'mitigasi',
                'exposure', 'hedging', 'reinsurance'
            ],
            ActuarialDomain.STATISTICS: [
                'statistik', 'probability', 'distribusi',
                'regression', 'correlation', 'variance',
                'standard deviation', 'confidence interval'
            ],
            ActuarialDomain.FINANCIAL_MATH: [
                'matematika keuangan', 'present value', 'future value',
                'annuity', 'perpetuity', 'discount rate',
                'compound interest', 'effective rate'
            ],
            ActuarialDomain.REGULATION: [
                'regulasi', 'peraturan', 'ojk', 'seojk',
                'solvabilitas', 'rbc', 'psak', 'ifrs',
                'compliance', 'governance'
            ]
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'simple': [
                'sederhana', 'dasar', 'basic', 'simple',
                'pengenalan', 'introduction'
            ],
            'moderate': [
                'menengah', 'intermediate', 'standard',
                'umum', 'typical', 'normal'
            ],
            'complex': [
                'kompleks', 'advanced', 'sophisticated',
                'detail', 'comprehensive', 'in-depth',
                'multi', 'integrated', 'kombinasi'
            ]
        }
        
        logger.info("Question Analyzer initialized")
    
    def analyze_question(self, 
                        question: str,
                        chat_history: str = "",
                        session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze pertanyaan secara komprehensif.
        
        Args:
            question: Pertanyaan pengguna
            chat_history: Riwayat percakapan
            session_context: Context dari session
            
        Returns:
            Dict berisi hasil analisis lengkap
        """
        try:
            logger.info(f"Analyzing question: {question[:100]}...")
            
            # Step 1: Basic question analysis
            question_type = self._detect_question_type(question)
            domains = self._detect_actuarial_domains(question)
            context = self._extract_question_context(question)
            
            # Step 2: Extract keywords dan entities
            keywords = self._extract_domain_keywords(question, domains)
            entities = self._extract_entities(question)
            
            # Step 3: Determine document requirements
            doc_requirements = self._determine_document_requirements(
                question_type, domains, context, keywords
            )
            
            # Step 4: Analyze complexity
            complexity = self._analyze_complexity(question, context)
            
            # Step 5: Generate search strategies
            search_strategies = self._generate_search_strategies(
                question, question_type, domains, keywords
            )
            
            # Step 6: LLM-based enhancement (jika tersedia)
            llm_insights = self._get_llm_insights(question, chat_history) if self.llm else {}
            
            # Compile hasil analisis
            analysis_result = {
                "question_type": question_type.value,
                "primary_domain": domains[0].value if domains else ActuarialDomain.GENERAL.value,
                "all_domains": [d.value for d in domains],
                "keywords": keywords,
                "entities": entities,
                "context": context.__dict__,
                "document_requirements": doc_requirements.__dict__,
                "complexity_level": complexity,
                "search_strategies": search_strategies,
                "confidence_score": self._calculate_confidence_score(
                    question_type, domains, keywords, context
                ),
                "llm_insights": llm_insights,
                "metadata": {
                    "question_length": len(question),
                    "has_chat_history": bool(chat_history),
                    "analysis_timestamp": self._get_timestamp()
                }
            }
            
            logger.info(f"Question analysis completed: {question_type.value}, {complexity} complexity")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in question analysis: {e}")
            return self._get_fallback_analysis(question)
    
    def _detect_question_type(self, question: str) -> QuestionType:
        """
        Detect jenis pertanyaan berdasarkan pattern matching.
        
        Args:
            question: Pertanyaan pengguna
            
        Returns:
            QuestionType enum
        """
        try:
            question_lower = question.lower()
            type_scores = {}
            
            # Score setiap question type berdasarkan pattern matching
            for q_type, patterns in self.question_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, question_lower))
                    score += matches
                type_scores[q_type] = score
            
            # Return type dengan score tertinggi
            if type_scores:
                best_type = max(type_scores, key=type_scores.get)
                if type_scores[best_type] > 0:
                    return best_type
            
            # Default fallback
            return QuestionType.THEORY
            
        except Exception as e:
            logger.error(f"Error detecting question type: {e}")
            return QuestionType.THEORY
    
    def _detect_actuarial_domains(self, question: str) -> List[ActuarialDomain]:
        """
        Detect domain aktuaria yang relevan.
        
        Args:
            question: Pertanyaan pengguna
            
        Returns:
            List of ActuarialDomain enums (sorted by relevance)
        """
        try:
            question_lower = question.lower()
            domain_scores = {}
            
            # Score setiap domain berdasarkan keyword matching
            for domain, keywords in self.domain_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword.lower() in question_lower:
                        # Exact match gets higher score
                        score += 2 if keyword.lower() == question_lower else 1
                domain_scores[domain] = score
            
            # Sort domains by score
            sorted_domains = sorted(
                domain_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Return domains dengan score > 0
            relevant_domains = [domain for domain, score in sorted_domains if score > 0]
            
            # Jika tidak ada domain spesifik, return GENERAL
            return relevant_domains if relevant_domains else [ActuarialDomain.GENERAL]
            
        except Exception as e:
            logger.error(f"Error detecting actuarial domains: {e}")
            return [ActuarialDomain.GENERAL]
    
    def _extract_question_context(self, question: str) -> QuestionContext:
        """
        Extract context information dari pertanyaan.
        
        Args:
            question: Pertanyaan pengguna
            
        Returns:
            QuestionContext object
        """
        try:
            question_lower = question.lower()
            
            # Detect various context indicators
            has_numbers = bool(re.search(r'\d+', question))
            has_formulas = bool(re.search(r'rumus|formula|persamaan|equation', question_lower))
            has_comparisons = bool(re.search(r'bandingkan|versus|vs|dibanding|lebih', question_lower))
            has_procedures = bool(re.search(r'langkah|tahap|cara|bagaimana|how', question_lower))
            
            # Extract complexity indicators
            complexity_indicators = []
            for level, indicators in self.complexity_indicators.items():
                for indicator in indicators:
                    if indicator in question_lower:
                        complexity_indicators.append(f"{level}:{indicator}")
            
            # Detect temporal context
            temporal_context = None
            if re.search(r'dulu|sebelum|masa lalu|historical', question_lower):
                temporal_context = "past"
            elif re.search(r'sekarang|saat ini|current|present', question_lower):
                temporal_context = "present"
            elif re.search(r'akan|future|proyeksi|forecast', question_lower):
                temporal_context = "future"
            
            return QuestionContext(
                has_numbers=has_numbers,
                has_formulas=has_formulas,
                has_comparisons=has_comparisons,
                has_procedures=has_procedures,
                complexity_indicators=complexity_indicators,
                temporal_context=temporal_context
            )
            
        except Exception as e:
            logger.error(f"Error extracting question context: {e}")
            return QuestionContext(
                has_numbers=False,
                has_formulas=False,
                has_comparisons=False,
                has_procedures=False,
                complexity_indicators=[],
                temporal_context=None
            )
    
    def _extract_domain_keywords(self, 
                               question: str,
                               domains: List[ActuarialDomain]) -> List[str]:
        """
        Extract keywords yang relevan dengan domain.
        
        Args:
            question: Pertanyaan pengguna
            domains: Detected domains
            
        Returns:
            List of relevant keywords
        """
        try:
            question_lower = question.lower()
            keywords = set()
            
            # Extract keywords dari detected domains
            for domain in domains:
                if domain in self.domain_keywords:
                    for keyword in self.domain_keywords[domain]:
                        if keyword.lower() in question_lower:
                            keywords.add(keyword)
            
            # Extract additional keywords menggunakan simple NLP
            # Remove common words dan extract meaningful terms
            words = re.findall(r'\b\w{3,}\b', question_lower)
            common_words = {
                'yang', 'adalah', 'untuk', 'dari', 'dengan', 'pada',
                'dalam', 'akan', 'dapat', 'atau', 'dan', 'ini', 'itu',
                'the', 'and', 'or', 'for', 'with', 'from', 'this', 'that'
            }
            
            meaningful_words = [w for w in words if w not in common_words and len(w) > 3]
            keywords.update(meaningful_words[:5])  # Top 5 meaningful words
            
            return list(keywords)[:10]  # Limit to 10 keywords
            
        except Exception as e:
            logger.error(f"Error extracting domain keywords: {e}")
            return ["aktuaria"]
    
    def _extract_entities(self, question: str) -> Dict[str, List[str]]:
        """
        Extract entities dari pertanyaan (numbers, dates, names, etc.).
        
        Args:
            question: Pertanyaan pengguna
            
        Returns:
            Dict berisi berbagai jenis entities
        """
        try:
            entities = {
                "numbers": [],
                "percentages": [],
                "dates": [],
                "currencies": [],
                "regulations": []
            }
            
            # Extract numbers
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)
            entities["numbers"] = numbers
            
            # Extract percentages
            percentages = re.findall(r'\b\d+(?:\.\d+)?\s*%', question)
            entities["percentages"] = percentages
            
            # Extract years/dates
            dates = re.findall(r'\b(?:19|20)\d{2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', question)
            entities["dates"] = dates
            
            # Extract currency mentions
            currencies = re.findall(r'\b(?:rupiah|dollar|usd|idr|rp)\b', question.lower())
            entities["currencies"] = currencies
            
            # Extract regulation mentions
            regulations = re.findall(r'\b(?:psak|seojk|ojk|ifrs|sak|peraturan)\s*\d*\b', question.lower())
            entities["regulations"] = regulations
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"numbers": [], "percentages": [], "dates": [], "currencies": [], "regulations": []}
    
    def _determine_document_requirements(self, 
                                       question_type: QuestionType,
                                       domains: List[ActuarialDomain],
                                       context: QuestionContext,
                                       keywords: List[str]) -> DocumentRequirement:
        """
        Determine kebutuhan dokumen berdasarkan analisis.
        
        Args:
            question_type: Jenis pertanyaan
            domains: Detected domains
            context: Question context
            keywords: Extracted keywords
            
        Returns:
            DocumentRequirement object
        """
        try:
            # Document types berdasarkan question type
            type_to_docs = {
                QuestionType.THEORY: ["manual", "guide", "explanation"],
                QuestionType.CALCULATION: ["example", "formula", "calculation"],
                QuestionType.DATA_LOOKUP: ["table", "data", "reference"],
                QuestionType.COMPARISON: ["analysis", "comparison", "guide"],
                QuestionType.PROCEDURE: ["manual", "procedure", "guide"],
                QuestionType.INTERPRETATION: ["analysis", "interpretation", "guide"]
            }
            
            document_types = type_to_docs.get(question_type, ["general"])
            
            # Priority keywords dari domains dan extracted keywords
            priority_keywords = []
            for domain in domains[:2]:  # Top 2 domains
                if domain in self.domain_keywords:
                    priority_keywords.extend(self.domain_keywords[domain][:3])
            priority_keywords.extend(keywords[:5])
            
            # Section hints berdasarkan question type dan context
            section_hints = []
            if question_type == QuestionType.CALCULATION:
                section_hints.extend(["perhitungan", "rumus", "contoh"])
            elif question_type == QuestionType.THEORY:
                section_hints.extend(["definisi", "konsep", "penjelasan"])
            elif context.has_procedures:
                section_hints.extend(["langkah", "prosedur", "cara"])
            
            # Scope preference
            scope_preference = "both"  # Default
            if question_type in [QuestionType.CALCULATION, QuestionType.DATA_LOOKUP]:
                scope_preference = "session"  # Prefer session docs for specific data
            elif question_type == QuestionType.THEORY:
                scope_preference = "global"  # Prefer global docs for theory
            
            return DocumentRequirement(
                document_types=document_types,
                priority_keywords=list(set(priority_keywords))[:10],
                section_hints=section_hints,
                scope_preference=scope_preference
            )
            
        except Exception as e:
            logger.error(f"Error determining document requirements: {e}")
            return DocumentRequirement(
                document_types=["general"],
                priority_keywords=["aktuaria"],
                section_hints=[],
                scope_preference="both"
            )
    
    def _analyze_complexity(self, question: str, context: QuestionContext) -> str:
        """
        Analyze complexity level pertanyaan.
        
        Args:
            question: Pertanyaan pengguna
            context: Question context
            
        Returns:
            Complexity level (simple, moderate, complex)
        """
        try:
            complexity_score = 0
            
            # Length-based scoring
            if len(question) > 200:
                complexity_score += 2
            elif len(question) > 100:
                complexity_score += 1
            
            # Context-based scoring
            if context.has_numbers:
                complexity_score += 1
            if context.has_formulas:
                complexity_score += 2
            if context.has_comparisons:
                complexity_score += 1
            if context.has_procedures:
                complexity_score += 1
            
            # Complexity indicators
            for indicator in context.complexity_indicators:
                if indicator.startswith("complex"):
                    complexity_score += 3
                elif indicator.startswith("moderate"):
                    complexity_score += 1
            
            # Multiple question indicators
            question_marks = question.count('?')
            if question_marks > 1:
                complexity_score += 1
            
            # Determine final complexity
            if complexity_score >= 5:
                return "complex"
            elif complexity_score >= 2:
                return "moderate"
            else:
                return "simple"
                
        except Exception as e:
            logger.error(f"Error analyzing complexity: {e}")
            return "moderate"
    
    def _generate_search_strategies(self, 
                                  question: str,
                                  question_type: QuestionType,
                                  domains: List[ActuarialDomain],
                                  keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Generate multiple search strategies.
        
        Args:
            question: Original question
            question_type: Question type
            domains: Detected domains
            keywords: Extracted keywords
            
        Returns:
            List of search strategies
        """
        try:
            strategies = []
            
            # Strategy 1: Direct question search
            strategies.append({
                "name": "direct_search",
                "query": question[:100],  # Limit length
                "weight": 1.0,
                "scope": "both"
            })
            
            # Strategy 2: Keyword-based search
            if keywords:
                keyword_query = " ".join(keywords[:3])
                strategies.append({
                    "name": "keyword_search",
                    "query": keyword_query,
                    "weight": 0.8,
                    "scope": "both"
                })
            
            # Strategy 3: Domain-specific search
            if domains:
                primary_domain = domains[0]
                domain_keywords = self.domain_keywords.get(primary_domain, [])
                if domain_keywords:
                    domain_query = " ".join(domain_keywords[:2])
                    strategies.append({
                        "name": "domain_search",
                        "query": domain_query,
                        "weight": 0.7,
                        "scope": "global" if question_type == QuestionType.THEORY else "both"
                    })
            
            # Strategy 4: Type-specific search
            type_terms = {
                QuestionType.CALCULATION: "perhitungan rumus",
                QuestionType.THEORY: "konsep definisi",
                QuestionType.DATA_LOOKUP: "data tabel",
                QuestionType.COMPARISON: "perbandingan analisis",
                QuestionType.PROCEDURE: "langkah prosedur",
                QuestionType.INTERPRETATION: "interpretasi analisis"
            }
            
            if question_type in type_terms:
                type_query = type_terms[question_type]
                if keywords:
                    type_query += f" {keywords[0]}"
                strategies.append({
                    "name": "type_specific_search",
                    "query": type_query,
                    "weight": 0.6,
                    "scope": "both"
                })
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error generating search strategies: {e}")
            return [{
                "name": "fallback_search",
                "query": question[:50],
                "weight": 1.0,
                "scope": "both"
            }]
    
    def _get_llm_insights(self, question: str, chat_history: str) -> Dict[str, Any]:
        """
        Get additional insights dari LLM jika tersedia.
        
        Args:
            question: Pertanyaan pengguna
            chat_history: Chat history
            
        Returns:
            Dict berisi LLM insights
        """
        try:
            if not self.llm:
                return {}
            
            insight_prompt = f"""
            Analisis pertanyaan aktuaria berikut dan berikan insights tambahan:
            
            Pertanyaan: {question}
            Riwayat: {chat_history[-300:] if chat_history else "Tidak ada"}
            
            Berikan insights dalam format JSON:
            {{
                "intent_clarification": "klarifikasi maksud pertanyaan",
                "missing_context": ["konteks yang mungkin kurang"],
                "related_topics": ["topik terkait"],
                "difficulty_assessment": "mudah|sedang|sulit",
                "recommended_approach": "pendekatan yang disarankan"
            }}
            """
            
            response = self.llm.invoke(insight_prompt)
            
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {"raw_insight": response.content[:200]}
                
        except Exception as e:
            logger.error(f"Error getting LLM insights: {e}")
            return {}
    
    def _calculate_confidence_score(self, 
                                  question_type: QuestionType,
                                  domains: List[ActuarialDomain],
                                  keywords: List[str],
                                  context: QuestionContext) -> float:
        """
        Calculate confidence score untuk analisis.
        
        Args:
            question_type: Detected question type
            domains: Detected domains
            keywords: Extracted keywords
            context: Question context
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        try:
            confidence = 0.5  # Base confidence
            
            # Domain detection confidence
            if domains and domains[0] != ActuarialDomain.GENERAL:
                confidence += 0.2
            
            # Keyword extraction confidence
            if len(keywords) >= 3:
                confidence += 0.1
            elif len(keywords) >= 1:
                confidence += 0.05
            
            # Context richness
            context_indicators = sum([
                context.has_numbers,
                context.has_formulas,
                context.has_comparisons,
                context.has_procedures
            ])
            confidence += context_indicators * 0.05
            
            # Complexity indicators
            if context.complexity_indicators:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _get_fallback_analysis(self, question: str) -> Dict[str, Any]:
        """
        Fallback analysis jika terjadi error.
        
        Args:
            question: Original question
            
        Returns:
            Fallback analysis result
        """
        return {
            "question_type": QuestionType.THEORY.value,
            "primary_domain": ActuarialDomain.GENERAL.value,
            "all_domains": [ActuarialDomain.GENERAL.value],
            "keywords": ["aktuaria"],
            "entities": {"numbers": [], "percentages": [], "dates": [], "currencies": [], "regulations": []},
            "context": QuestionContext(
                has_numbers=False,
                has_formulas=False,
                has_comparisons=False,
                has_procedures=False,
                complexity_indicators=[],
                temporal_context=None
            ).__dict__,
            "document_requirements": DocumentRequirement(
                document_types=["general"],
                priority_keywords=["aktuaria"],
                section_hints=[],
                scope_preference="both"
            ).__dict__,
            "complexity_level": "moderate",
            "search_strategies": [{
                "name": "fallback_search",
                "query": question[:50],
                "weight": 1.0,
                "scope": "both"
            }],
            "confidence_score": 0.3,
            "llm_insights": {},
            "metadata": {
                "question_length": len(question),
                "has_chat_history": False,
                "analysis_timestamp": self._get_timestamp(),
                "fallback_used": True
            }
        }
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp.
        
        Returns:
            ISO format timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_supported_question_types(self) -> List[str]:
        """
        Get list of supported question types.
        
        Returns:
            List of question type names
        """
        return [q_type.value for q_type in QuestionType]
    
    def get_supported_domains(self) -> List[str]:
        """
        Get list of supported actuarial domains.
        
        Returns:
            List of domain names
        """
        return [domain.value for domain in ActuarialDomain]