#!/usr/bin/env python3
"""
Tree of Thought Document Reader for enhanced document retrieval.

This module implements a multi-step reasoning approach to document retrieval,
analyzing questions and determining optimal retrieval strategies.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class DocumentScope(Enum):
    """Document scope enumeration"""
    GLOBAL = "global"
    SESSION = "session"
    BOTH = "both"


@dataclass
class QuestionAnalysis:
    """Question analysis result"""
    question_type: str
    complexity: str
    entities: List[str]
    requires_calculation: bool
    requires_definition: bool
    confidence: float = 0.8


@dataclass
class DocumentSection:
    """Document section with metadata"""
    content: str
    source: str
    section_type: str
    relevance_score: float
    metadata: Dict[str, Any]


class QuestionAnalyzer:
    """
    Analyzes questions to determine optimal retrieval strategies.
    
    This class uses natural language processing techniques to understand
    question intent, complexity, and required information types.
    """
    
    def __init__(self):
        """Initialize Question Analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze question to determine retrieval strategy.
        
        Args:
            question: User question to analyze
            
        Returns:
            Analysis results with question type, complexity, and entities
        """
        try:
            # Determine question type
            question_type = self._determine_question_type(question)
            
            # Extract key entities and concepts
            entities = self._extract_key_entities(question)
            
            # Determine complexity level
            complexity = self._determine_complexity(question)
            
            return {
                "question_type": question_type,
                "entities": entities,
                "complexity": complexity,
                "requires_calculation": "hitung" in question.lower() or "kalkulasi" in question.lower(),
                "requires_definition": "apa" in question.lower() or "definisi" in question.lower()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing question: {e}")
            return {
                "question_type": "general",
                "entities": [],
                "complexity": "medium",
                "requires_calculation": False,
                "requires_definition": False
            }
    
    def determine_complexity(self, question: str) -> str:
        """Public method for determining question complexity"""
        return self._determine_complexity(question)
    
    def extract_entities(self, question: str) -> List[str]:
        """Public method for extracting entities"""
        return self._extract_key_entities(question)
    
    def _determine_question_type(self, question: str) -> str:
        """Determine the type of question being asked"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["hitung", "kalkulasi", "rumus"]):
            return "calculation"
        elif any(word in question_lower for word in ["apa", "definisi", "pengertian"]):
            return "definition"
        elif any(word in question_lower for word in ["bandingkan", "perbedaan", "versus"]):
            return "comparison"
        elif any(word in question_lower for word in ["data", "tabel", "nilai"]):
            return "data_lookup"
        else:
            return "general"
    
    def _extract_key_entities(self, question: str) -> List[str]:
        """Extract key entities from the question"""
        # Simple keyword extraction for actuarial terms
        actuarial_terms = [
            "asuransi", "premi", "klaim", "mortalitas", "morbiditas",
            "reserv", "aktuaria", "probabilitas", "risiko", "underwriting",
            "anuitas", "bunga", "diskonto", "nilai tunai", "benefit"
        ]
        
        question_lower = question.lower()
        found_entities = [term for term in actuarial_terms if term in question_lower]
        
        return found_entities
    
    def _determine_complexity(self, question: str) -> str:
        """Determine the complexity level of the question"""
        question_lower = question.lower()
        
        # High complexity indicators
        high_complexity_words = ["analisis", "evaluasi", "optimasi", "model", "simulasi"]
        
        # Low complexity indicators  
        low_complexity_words = ["apa", "siapa", "kapan", "dimana"]
        
        if any(word in question_lower for word in high_complexity_words):
            return "high"
        elif any(word in question_lower for word in low_complexity_words):
            return "low"
        else:
            return "medium"


class DocumentSectionMapper:
    """
    Maps questions to relevant document sections using ToT reasoning.
    
    This class analyzes questions and determines which document sections
    are most likely to contain relevant information.
    """
    
    def __init__(self):
        """Initialize Document Section Mapper"""
        self.logger = logging.getLogger(__name__)
        
        # Define section mappings for actuarial documents
        self.section_mappings = {
            "calculation": ["teknis_perhitungan", "valuasi", "rumus", "metode"],
            "definition": ["pendahuluan", "dasar", "pengertian", "konsep"],
            "implementation": ["implementasi", "teknologi", "sistem", "proses"],
            "reporting": ["laporan", "penyajian", "disclosure", "catatan"],
            "assumptions": ["asumsi", "parameter", "faktor", "tingkat"]
        }
    
    def calculate_section_relevance(self, question: str, section: str) -> float:
        """Calculate relevance score between question and section"""
        try:
            question_lower = question.lower()
            section_lower = section.lower()
            
            # Simple keyword matching score
            score = 0.0
            
            # Check direct keyword matches
            for category, keywords in self.section_mappings.items():
                for keyword in keywords:
                    if keyword in question_lower and keyword in section_lower:
                        score += 0.3
            
            # Check section name in question
            if section_lower in question_lower:
                score += 0.5
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating section relevance: {e}")
            return 0.0
    
    def map_question_to_sections(self, question: str, retrieved_documents: List[Any], session_id: str) -> Dict[str, Any]:
        """Map question to relevant document sections"""
        try:
            # Analyze question to determine relevant sections
            question_analysis = self._analyze_question_for_sections(question)
            
            # Score documents based on relevance
            scored_sections = []
            for doc in retrieved_documents:
                section_name = getattr(doc, 'metadata', {}).get('section', 'unknown')
                relevance_score = self.calculate_section_relevance(question, section_name)
                
                scored_sections.append({
                    'section': section_name,
                    'document': doc,
                    'relevance_score': relevance_score
                })
            
            # Sort by relevance score
            scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                'mapped_sections': scored_sections[:5],  # Top 5 most relevant
                'question_analysis': question_analysis,
                'total_sections': len(scored_sections)
            }
            
        except Exception as e:
            self.logger.error(f"Error mapping question to sections: {e}")
            return {
                'mapped_sections': [],
                'question_analysis': {},
                'total_sections': 0
            }
    
    def _analyze_question_for_sections(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine relevant section types"""
        question_lower = question.lower()
        
        relevant_categories = []
        for category, keywords in self.section_mappings.items():
            if any(keyword in question_lower for keyword in keywords):
                relevant_categories.append(category)
        
        return {
            'relevant_categories': relevant_categories,
            'question_type': self._determine_question_type(question),
            'complexity': 'medium'  # Default complexity
        }
    
    def _determine_question_type(self, question: str) -> str:
        """Determine the type of question for section mapping"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["hitung", "kalkulasi", "rumus"]):
            return "calculation"
        elif any(word in question_lower for word in ["apa", "definisi", "pengertian"]):
            return "definition"
        elif any(word in question_lower for word in ["implementasi", "cara", "proses"]):
            return "implementation"
        else:
            return "general"


class ToTRetrievalStrategy:
    """
    Tree of Thought Retrieval Strategy for enhanced document retrieval.
    
    This class implements multiple retrieval strategies and selects
    the optimal approach based on question analysis.
    """
    
    def __init__(self):
        """Initialize ToT Retrieval Strategy"""
        self.logger = logging.getLogger(__name__)
        
        # Define retrieval strategies
        self.strategies = {
            "broad_search": {
                "description": "Wide search across all documents",
                "scope": ["global", "session"],
                "query_expansion": True,
                "max_results": 20
            },
            "focused_search": {
                "description": "Targeted search in specific sections",
                "scope": ["session"],
                "query_expansion": False,
                "max_results": 10
            },
            "hybrid_search": {
                "description": "Combination of broad and focused approaches",
                "scope": ["global", "session"],
                "query_expansion": True,
                "max_results": 15
            }
        }
    
    def _determine_optimal_strategy(self, question_analysis: Dict[str, Any], session_id: str) -> str:
        """Internal method for determining optimal strategy (for backward compatibility)"""
        return self.determine_optimal_strategy(question_analysis, session_id)
    
    def determine_optimal_strategy(self, question_analysis: Dict[str, Any], session_id: str) -> str:
        """Determine optimal retrieval strategy based on question analysis"""
        try:
            complexity = question_analysis.get('complexity', 'medium')
            question_type = question_analysis.get('question_type', 'general')
            
            # Strategy selection logic
            if complexity == 'high' or question_type == 'calculation':
                return 'hybrid_search'
            elif complexity == 'low' or question_type == 'definition':
                return 'focused_search'
            else:
                return 'broad_search'
                
        except Exception as e:
            self.logger.error(f"Error determining optimal strategy: {e}")
            return 'broad_search'  # Default fallback
    
    async def execute_retrieval(self, strategy_name: str, question: str, session_id: str) -> Dict[str, Any]:
        """Execute retrieval using specified strategy"""
        try:
            strategy = self.strategies.get(strategy_name, self.strategies['broad_search'])
            
            # Mock retrieval execution for now
            # In real implementation, this would call actual retrieval services
            results = {
                'documents': [],
                'strategy_used': strategy_name,
                'total_results': 0,
                'execution_time': 0.1
            }
            
            self.logger.info(f"Executed {strategy_name} retrieval for session {session_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing retrieval: {e}")
            return {
                'documents': [],
                'strategy_used': 'fallback',
                'total_results': 0,
                'execution_time': 0.0,
                'error': str(e)
            }


class ToTDocumentReader:
    """
    Tree of Thought Document Reader for enhanced document retrieval.
    
    This class implements a multi-step reasoning approach to document retrieval,
    analyzing questions and determining optimal retrieval strategies.
    """
    
    def __init__(self, llm=None, vector_store_manager=None):
        """Initialize ToT Document Reader with required components"""
        self.llm = llm
        self.vector_store_manager = vector_store_manager
        self.question_analyzer = QuestionAnalyzer()
        self.section_mapper = DocumentSectionMapper()
        self.retrieval_strategy = ToTRetrievalStrategy()
        self.logger = logging.getLogger(__name__)
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question using the question analyzer"""
        return self.question_analyzer.analyze_question(question)
    
    def execute_multipath_retrieval(self, question: str, session_id: str) -> Dict[str, Any]:
        """Execute multipath retrieval (synchronous wrapper)"""
        try:
            # Run async method synchronously
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.analyze_and_retrieve(question, session_id))
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error in multipath retrieval: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_retrieval_strategies(self, question_analysis: Dict[str, Any], session_id: str) -> List[str]:
        """Generate list of retrieval strategies based on question analysis"""
        try:
            optimal_strategy = self.retrieval_strategy.determine_optimal_strategy(question_analysis, session_id)
            
            # Return list of strategies in order of preference
            if optimal_strategy == "hybrid_search":
                return ["hybrid_search", "broad_search", "focused_search"]
            elif optimal_strategy == "focused_search":
                return ["focused_search", "hybrid_search", "broad_search"]
            else:
                return ["broad_search", "hybrid_search", "focused_search"]
                
        except Exception as e:
            self.logger.error(f"Error generating retrieval strategies: {e}")
            return ["broad_search"]  # Default fallback
    
    async def analyze_and_retrieve(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Main method for ToT-based document retrieval.
        
        Args:
            question: User question
            session_id: Session identifier
            
        Returns:
            Enhanced retrieval results with ToT analysis
        """
        try:
            # Step 1: Analyze question
            question_analysis = self.question_analyzer.analyze_question(question)
            
            # Step 2: Determine optimal retrieval strategy
            strategy_name = self.retrieval_strategy.determine_optimal_strategy(
                question_analysis, session_id
            )
            
            # Step 3: Execute multi-path retrieval
            retrieval_results = await self.retrieval_strategy.execute_retrieval(
                strategy_name, question, session_id
            )
            
            # Step 4: Map to document sections
            section_mapping = self.section_mapper.map_question_to_sections(
                question, retrieval_results.get('documents', []), session_id
            )
            
            return {
                'success': True,
                'question_analysis': question_analysis,
                'strategy_used': strategy_name,
                'retrieval_results': retrieval_results,
                'section_mapping': section_mapping,
                'enhanced_context': self._prepare_enhanced_context(
                    question, section_mapping.get('mapped_sections', [])
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error in ToT document retrieval: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_used': True
            }
    
    def _prepare_enhanced_context(self, question: str, mapped_sections: List[Dict[str, Any]]) -> str:
        """
        Prepare enhanced context from mapped sections.
        
        Args:
            question: Original question
            mapped_sections: Sections mapped by ToT analysis
            
        Returns:
            Enhanced context string
        """
        try:
            if not mapped_sections:
                return "No relevant sections found."
            
            context_parts = []
            context_parts.append(f"Enhanced context for: {question}\n")
            
            for i, section in enumerate(mapped_sections[:3], 1):
                section_name = section.get('section', 'Unknown')
                relevance = section.get('relevance_score', 0.0)
                doc = section.get('document')
                
                if doc and hasattr(doc, 'page_content'):
                    content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    context_parts.append(
                        f"Section {i} ({section_name}, relevance: {relevance:.2f}):\n{content}\n"
                    )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error preparing enhanced context: {e}")
            return "Error preparing context."