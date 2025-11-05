"""
Complexity Detector Service

This service analyzes question complexity to determine if LLM verification is needed.
Simple questions (definitions, basic info) skip verification for performance.
Complex questions (calculations, analysis) require verification for accuracy.

Author: Assistant
Date: 2024
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Question complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"


@dataclass
class ComplexityAnalysis:
    """Result of complexity analysis"""
    level: ComplexityLevel
    confidence: float
    reasoning: List[str]
    requires_verification: bool
    keywords_found: List[str]


class ComplexityDetector:
    """
    Detects question complexity to optimize verification usage
    """
    
    def __init__(self):
        """Initialize complexity detector with keyword patterns"""
        self._setup_patterns()
        logger.info("ComplexityDetector initialized")
    
    def _setup_patterns(self) -> None:
        """Setup keyword patterns for complexity detection"""
        
        # Simple question patterns (no verification needed)
        self.simple_patterns = {
            'definition': [
                'apa itu', 'definisi', 'pengertian', 'arti dari',
                'what is', 'define', 'definition of', 'meaning of'
            ],
            'basic_info': [
                'siapa', 'dimana', 'kapan', 'berapa lama',
                'who', 'where', 'when', 'how long'
            ],
            'explanation': [
                'jelaskan', 'bagaimana cara', 'mengapa',
                'explain', 'how to', 'why'
            ],
            'comparison_simple': [
                'perbedaan', 'beda', 'persamaan',
                'difference', 'similar', 'compare'
            ]
        }
        
        # Complex question patterns (verification required)
        self.complex_patterns = {
            'calculation': [
                'hitung', 'perhitungan', 'kalkulasi', 'compute',
                'calculate', 'calculation', 'nilai sekarang',
                'present value', 'npv'
            ],
            'analysis': [
                'analisis', 'evaluasi', 'bandingkan', 'compare',
                'analyze', 'evaluation', 'assessment', 'sensitivity'
            ],
            'simulation': [
                'simulasi', 'skenario', 'proyeksi', 'forecast',
                'simulation', 'scenario', 'projection', 'modeling'
            ],
            'validation': [
                'validasi', 'verifikasi', 'periksa', 'cek',
                'validate', 'verify', 'check', 'audit'
            ]
        }
        
        # Technical actuarial terms that indicate complexity
        self.technical_complex_terms = [
            'pvdbo', 'pvfbo', 'discount rate', 'mortality rate',
            'morbidity', 'reserving', 'underwriting'
        ]
        
        # Moderate complexity patterns
        self.moderate_patterns = {
            'process': [
                'proses', 'tahapan', 'langkah', 'metode',
                'process', 'steps', 'method', 'procedure'
            ],
            'comparison': [
                'perbedaan', 'persamaan', 'vs', 'versus',
                'difference', 'similarity', 'comparison'
            ]
        }
    
    def analyze_complexity(
        self, 
        question: str, 
        context: Optional[str] = None
    ) -> ComplexityAnalysis:
        """
        Analyze question complexity
        
        Args:
            question: User question to analyze
            context: Optional context for better analysis
            
        Returns:
            ComplexityAnalysis with complexity level and verification requirement
        """
        try:
            question_lower = question.lower().strip()
            
            # Track found keywords and reasoning
            keywords_found = []
            reasoning = []
            
            # Check for complex patterns first (highest priority)
            complex_score = 0
            for category, patterns in self.complex_patterns.items():
                found_patterns = [p for p in patterns if p in question_lower]
                if found_patterns:
                    complex_score += len(found_patterns) * 3  # Increased weight
                    keywords_found.extend(found_patterns)
                    reasoning.append(f"Complex {category} keywords: {found_patterns}")
            
            # Check for technical actuarial terms (moderate complexity unless with calculation)
            technical_score = 0
            found_technical = [term for term in self.technical_complex_terms if term in question_lower]
            if found_technical:
                keywords_found.extend(found_technical)
                
                # If technical terms are with simple question patterns, treat as moderate
                has_simple_patterns = any(
                    any(p in question_lower for p in patterns) 
                    for patterns in self.simple_patterns.values()
                )
                
                if has_simple_patterns:
                    technical_score += len(found_technical)  # Moderate weight
                    reasoning.append(f"Technical terms with simple question: {found_technical}")
                else:
                    complex_score += len(found_technical) * 2  # Complex weight
                    reasoning.append(f"Technical terms: {found_technical}")
            
            # Check for moderate patterns
            moderate_score = 0
            for category, patterns in self.moderate_patterns.items():
                found_patterns = [p for p in patterns if p in question_lower]
                if found_patterns:
                    moderate_score += len(found_patterns)
                    keywords_found.extend(found_patterns)
                    reasoning.append(f"Moderate {category} keywords: {found_patterns}")
            
            # Add technical score to moderate if applicable
            moderate_score += technical_score
            
            # Check for simple patterns
            simple_score = 0
            for category, patterns in self.simple_patterns.items():
                found_patterns = [p for p in patterns if p in question_lower]
                if found_patterns:
                    simple_score += len(found_patterns)
                    keywords_found.extend(found_patterns)
                    reasoning.append(f"Simple {category} keywords: {found_patterns}")
            
            # Additional complexity indicators
            additional_complexity = self._check_additional_complexity(question_lower)
            if additional_complexity['score'] > 0:
                complex_score += additional_complexity['score']
                reasoning.extend(additional_complexity['reasons'])
            
            # Determine complexity level
            level, confidence = self._determine_complexity_level(
                complex_score, moderate_score, simple_score, question_lower
            )
            
            # Determine if verification is required
            requires_verification = self._requires_verification(level, complex_score)
            
            if not reasoning:
                reasoning.append("No specific complexity indicators found")
            
            result = ComplexityAnalysis(
                level=level,
                confidence=confidence,
                reasoning=reasoning,
                requires_verification=requires_verification,
                keywords_found=list(set(keywords_found))
            )
            
            logger.debug(f"Complexity analysis: {level.value} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in complexity analysis: {str(e)}")
            # Default to complex for safety
            return ComplexityAnalysis(
                level=ComplexityLevel.COMPLEX,
                confidence=0.5,
                reasoning=[f"Analysis error: {str(e)}"],
                requires_verification=True,
                keywords_found=[]
            )
    
    def _check_additional_complexity(self, question: str) -> Dict[str, Any]:
        """Check for additional complexity indicators"""
        score = 0
        reasons = []
        
        # Numbers and mathematical expressions
        if re.search(r'\d+[.,]\d+|\d+%|\d+\s*(juta|miliar|ribu|million|billion)', question):
            score += 2
            reasons.append("Contains specific numbers/percentages")
        
        # Question length (longer questions tend to be more complex)
        if len(question.split()) > 15:
            score += 1
            reasons.append("Long question (>15 words)")
        
        # Multiple questions in one
        if question.count('?') > 1 or ' dan ' in question or ' or ' in question:
            score += 1
            reasons.append("Multiple questions/conditions")
        
        # Technical actuarial terms
        technical_terms = [
            'aktuaria', 'actuarial', 'mortality', 'morbidity', 'reserving',
            'underwriting', 'premium', 'premi', 'klaim', 'claim'
        ]
        found_technical = [term for term in technical_terms if term in question]
        if found_technical:
            score += len(found_technical)
            reasons.append(f"Technical terms: {found_technical}")
        
        return {'score': score, 'reasons': reasons}
    
    def _determine_complexity_level(
        self, 
        complex_score: int, 
        moderate_score: int, 
        simple_score: int,
        question: str
    ) -> tuple[ComplexityLevel, float]:
        """Determine complexity level and confidence"""
        
        # Determine final complexity level with improved logic
        if complex_score > 0:
            # If there are complex patterns, it's complex regardless of other scores
            complexity = ComplexityLevel.COMPLEX
            confidence = min(0.95, 0.6 + (complex_score * 0.1))
        elif moderate_score > simple_score and moderate_score > 1:
            # Moderate if moderate score is higher and significant
            complexity = ComplexityLevel.MODERATE
            confidence = min(0.85, 0.5 + (moderate_score * 0.1))
        elif simple_score > 0:
            # Simple if simple patterns found
            complexity = ComplexityLevel.SIMPLE
            confidence = min(0.90, 0.6 + (simple_score * 0.1))
        else:
            # Default to moderate for unknown patterns
            complexity = ComplexityLevel.MODERATE
            confidence = 0.5
            
        return complexity, confidence
    
    def _requires_verification(self, level: ComplexityLevel, complex_score: int) -> bool:
        """Determine if verification is required based on complexity"""
        
        # Always verify complex questions
        if level == ComplexityLevel.COMPLEX:
            return True
        
        # Never verify simple questions
        if level == ComplexityLevel.SIMPLE:
            return False
        
        # For moderate questions, verify if there are any complex indicators
        if level == ComplexityLevel.MODERATE:
            return complex_score > 0
        
        return True  # Default to verification for safety
    
    def get_verification_recommendation(self, analysis: ComplexityAnalysis) -> Dict[str, Any]:
        """Get detailed verification recommendation"""
        
        return {
            'should_verify': analysis.requires_verification,
            'complexity_level': analysis.level.value,
            'confidence': analysis.confidence,
            'reasoning': analysis.reasoning,
            'performance_impact': {
                'estimated_time_saved': 0 if analysis.requires_verification else 60,  # seconds
                'verification_necessity': 'high' if analysis.level == ComplexityLevel.COMPLEX else 'low'
            },
            'keywords_found': analysis.keywords_found
        }