"""Path Evaluation Service untuk Tree of Thought Implementation.

Service ini mengevaluasi dan me-rerank multiple reasoning paths
untuk memilih jalur terbaik berdasarkan kriteria akurasi, kelengkapan, dan relevansi.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EvaluationCriteria(Enum):
    """Kriteria evaluasi untuk reasoning paths."""
    ACCURACY = "accuracy"  # Akurasi informasi
    COMPLETENESS = "completeness"  # Kelengkapan jawaban
    RELEVANCE = "relevance"  # Relevansi dengan pertanyaan
    CLARITY = "clarity"  # Kejelasan penjelasan
    TECHNICAL_DEPTH = "technical_depth"  # Kedalaman teknis

@dataclass
class PathScore:
    """Score untuk setiap reasoning path."""
    path_index: int
    path_type: str
    accuracy_score: float
    completeness_score: float
    relevance_score: float
    clarity_score: float
    technical_depth_score: float
    weighted_total: float
    confidence_adjustment: float
    final_score: float

class PathEvaluationService:
    """Service untuk evaluasi dan ranking reasoning paths."""
    
    def __init__(self, llm: LLM):
        """
        Initialize Path Evaluation Service.
        
        Args:
            llm: Language model untuk evaluasi paths
        """
        self.llm = llm
        self.evaluation_weights = {
            "accuracy": 0.25,
            "completeness": 0.20,
            "relevance": 0.25,
            "clarity": 0.15,
            "technical_depth": 0.15
        }
        self._init_evaluation_chain()
        
    def _init_evaluation_chain(self) -> None:
        """Initialize LLM chain untuk evaluasi paths."""
        try:
            evaluation_template = """
Anda adalah evaluator ahli untuk sistem Tree of Thought dalam domain aktuaria.

TUGAS:
- Evaluasi setiap reasoning path berdasarkan 5 kriteria
- Berikan score 0-100 untuk setiap kriteria
- Berikan reasoning untuk setiap score
- Tentukan path mana yang terbaik

Pertanyaan Asli: {original_question}

Reasoning Paths untuk dievaluasi:
{reasoning_paths}

Kriteria Evaluasi:
1. ACCURACY (0-100): Seberapa akurat informasi yang diberikan?
2. COMPLETENESS (0-100): Seberapa lengkap jawaban menjawab pertanyaan?
3. RELEVANCE (0-100): Seberapa relevan dengan pertanyaan asli?
4. CLARITY (0-100): Seberapa jelas dan mudah dipahami?
5. TECHNICAL_DEPTH (0-100): Seberapa mendalam secara teknis?

Format output JSON:
{{
    "path_evaluations": [
        {{
            "path_index": 0,
            "path_type": "definition",
            "scores": {{
                "accuracy": 85,
                "completeness": 90,
                "relevance": 95,
                "clarity": 80,
                "technical_depth": 75
            }},
            "score_reasoning": {{
                "accuracy": "informasi akurat berdasarkan standar",
                "completeness": "menjawab semua aspek pertanyaan",
                "relevance": "sangat relevan dengan pertanyaan",
                "clarity": "penjelasan cukup jelas",
                "technical_depth": "cukup mendalam secara teknis"
            }},
            "strengths": ["kekuatan 1", "kekuatan 2"],
            "weaknesses": ["kelemahan 1", "kelemahan 2"]
        }}
    ],
    "best_path_index": 0,
    "evaluation_reasoning": "penjelasan mengapa path ini terbaik",
    "combined_answer_suggestion": "saran untuk menggabungkan kekuatan dari berbagai path"
}}

Output JSON:"""
            
            self.evaluation_prompt = PromptTemplate(
                input_variables=["original_question", "reasoning_paths"],
                template=evaluation_template
            )
            
            self.evaluation_chain = LLMChain(
                llm=self.llm,
                prompt=self.evaluation_prompt,
                verbose=False
            )
            
            logger.info("Path evaluation chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing evaluation chain: {e}")
            raise
    
    def evaluate_reasoning_paths(self, 
                               original_question: str,
                               reasoning_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate dan rank multiple reasoning paths.
        
        Args:
            original_question: Pertanyaan asli dari user
            reasoning_paths: List reasoning paths untuk dievaluasi
            
        Returns:
            Dict berisi evaluasi dan ranking paths
        """
        try:
            logger.info(f"Evaluating {len(reasoning_paths)} reasoning paths")
            
            # Format paths untuk evaluasi
            formatted_paths = self._format_paths_for_evaluation(reasoning_paths)
            
            # Generate evaluation menggunakan LLM
            response = self.evaluation_chain.run(
                original_question=original_question,
                reasoning_paths=formatted_paths
            )
            
            # Parse evaluation response
            evaluation_result = self._parse_evaluation_response(response)
            
            # Calculate weighted scores
            scored_paths = self._calculate_weighted_scores(evaluation_result, reasoning_paths)
            
            # Rank paths berdasarkan final score
            ranked_paths = self._rank_paths(scored_paths)
            
            # Prepare final result
            final_result = {
                "original_question": original_question,
                "total_paths_evaluated": len(reasoning_paths),
                "path_scores": [self._path_score_to_dict(score) for score in ranked_paths],
                "best_path": self._get_best_path(ranked_paths, reasoning_paths),
                "evaluation_summary": evaluation_result.get("evaluation_reasoning", ""),
                "combined_answer_suggestion": evaluation_result.get("combined_answer_suggestion", ""),
                "evaluation_metadata": {
                    "weights_used": self.evaluation_weights,
                    "evaluation_timestamp": self._get_timestamp()
                }
            }
            
            logger.info(f"Path evaluation completed. Best path: {final_result['best_path']['path_type']}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error evaluating reasoning paths: {e}")
            return self._get_fallback_evaluation(original_question, reasoning_paths)
    
    def _format_paths_for_evaluation(self, reasoning_paths: List[Dict[str, Any]]) -> str:
        """Format reasoning paths untuk input ke LLM."""
        formatted = []
        
        for i, path in enumerate(reasoning_paths):
            path_text = f"""
Path {i} ({path.get('path_type', 'unknown')}):
Reasoning Steps: {path.get('reasoning_steps', [])}
Answer: {path.get('answer', '')}
Confidence: {path.get('confidence_score', 0)}
Key Points: {path.get('key_points', [])}
"""
            formatted.append(path_text)
        
        return "\n".join(formatted)
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response dari LLM evaluator."""
        try:
            # Clean response
            cleaned_response = self._clean_json_response(response)
            
            # Parse JSON
            result = json.loads(cleaned_response)
            
            # Validate structure
            if "path_evaluations" not in result:
                raise ValueError("Missing path_evaluations in response")
                
            # Validate each evaluation
            for evaluation in result["path_evaluations"]:
                if "scores" not in evaluation:
                    evaluation["scores"] = self._get_default_scores()
                    
                # Ensure all criteria have scores
                for criteria in self.evaluation_weights.keys():
                    if criteria not in evaluation["scores"]:
                        evaluation["scores"][criteria] = 70  # Default score
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
            raise
    
    def _clean_json_response(self, response: str) -> str:
        """Clean dan extract JSON dari response."""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # Find JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return response.strip()
    
    def _get_default_scores(self) -> Dict[str, int]:
        """Get default scores untuk missing evaluations."""
        return {
            "accuracy": 70,
            "completeness": 70,
            "relevance": 70,
            "clarity": 70,
            "technical_depth": 70
        }
    
    def _calculate_weighted_scores(self, 
                                 evaluation_result: Dict[str, Any],
                                 original_paths: List[Dict[str, Any]]) -> List[PathScore]:
        """Calculate weighted scores untuk setiap path."""
        scored_paths = []
        
        path_evaluations = evaluation_result.get("path_evaluations", [])
        
        for i, evaluation in enumerate(path_evaluations):
            if i >= len(original_paths):
                break
                
            scores = evaluation.get("scores", {})
            original_path = original_paths[i]
            
            # Calculate weighted total
            weighted_total = sum(
                scores.get(criteria, 70) * weight 
                for criteria, weight in self.evaluation_weights.items()
            )
            
            # Adjust berdasarkan original confidence
            original_confidence = original_path.get("confidence_score", 70)
            confidence_adjustment = (original_confidence - 70) * 0.1  # Small adjustment
            
            final_score = weighted_total + confidence_adjustment
            
            path_score = PathScore(
                path_index=i,
                path_type=original_path.get("path_type", "unknown"),
                accuracy_score=scores.get("accuracy", 70),
                completeness_score=scores.get("completeness", 70),
                relevance_score=scores.get("relevance", 70),
                clarity_score=scores.get("clarity", 70),
                technical_depth_score=scores.get("technical_depth", 70),
                weighted_total=weighted_total,
                confidence_adjustment=confidence_adjustment,
                final_score=final_score
            )
            
            scored_paths.append(path_score)
        
        return scored_paths
    
    def _rank_paths(self, scored_paths: List[PathScore]) -> List[PathScore]:
        """Rank paths berdasarkan final score."""
        return sorted(scored_paths, key=lambda x: x.final_score, reverse=True)
    
    def _path_score_to_dict(self, path_score: PathScore) -> Dict[str, Any]:
        """Convert PathScore ke dictionary."""
        return {
            "path_index": path_score.path_index,
            "path_type": path_score.path_type,
            "scores": {
                "accuracy": path_score.accuracy_score,
                "completeness": path_score.completeness_score,
                "relevance": path_score.relevance_score,
                "clarity": path_score.clarity_score,
                "technical_depth": path_score.technical_depth_score
            },
            "weighted_total": path_score.weighted_total,
            "confidence_adjustment": path_score.confidence_adjustment,
            "final_score": path_score.final_score
        }
    
    def _get_best_path(self, 
                      ranked_paths: List[PathScore], 
                      original_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get path terbaik dengan detail lengkap."""
        if not ranked_paths:
            return {}
            
        best_score = ranked_paths[0]
        best_path = original_paths[best_score.path_index].copy()
        
        # Add evaluation metadata
        best_path["evaluation_score"] = best_score.final_score
        best_path["rank"] = 1
        best_path["selected_as_best"] = True
        
        return best_path
    
    def _get_fallback_evaluation(self, 
                               original_question: str,
                               reasoning_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback evaluation jika LLM gagal."""
        logger.info("Using fallback evaluation method")
        
        # Simple rule-based evaluation
        scored_paths = []
        
        for i, path in enumerate(reasoning_paths):
            # Simple scoring berdasarkan confidence dan length
            confidence = path.get("confidence_score", 70)
            answer_length = len(path.get("answer", ""))
            
            # Basic scoring
            base_score = confidence
            length_bonus = min(answer_length / 10, 10)  # Max 10 bonus points
            final_score = base_score + length_bonus
            
            path_score = PathScore(
                path_index=i,
                path_type=path.get("path_type", "unknown"),
                accuracy_score=confidence,
                completeness_score=confidence,
                relevance_score=confidence,
                clarity_score=confidence,
                technical_depth_score=confidence,
                weighted_total=confidence,
                confidence_adjustment=0,
                final_score=final_score
            )
            
            scored_paths.append(path_score)
        
        # Rank by final score
        ranked_paths = sorted(scored_paths, key=lambda x: x.final_score, reverse=True)
        
        return {
            "original_question": original_question,
            "total_paths_evaluated": len(reasoning_paths),
            "path_scores": [self._path_score_to_dict(score) for score in ranked_paths],
            "best_path": self._get_best_path(ranked_paths, reasoning_paths),
            "evaluation_summary": "Fallback evaluation based on confidence and completeness",
            "combined_answer_suggestion": "Consider combining insights from top-ranked paths",
            "fallback_used": True,
            "evaluation_metadata": {
                "weights_used": self.evaluation_weights,
                "evaluation_timestamp": self._get_timestamp()
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def update_evaluation_weights(self, new_weights: Dict[str, float]) -> None:
        """Update evaluation weights."""
        try:
            # Validate weights sum to 1.0
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Weights sum to {total_weight}, normalizing...")
                new_weights = {k: v/total_weight for k, v in new_weights.items()}
            
            self.evaluation_weights.update(new_weights)
            logger.info(f"Evaluation weights updated: {self.evaluation_weights}")
            
        except Exception as e:
            logger.error(f"Error updating evaluation weights: {e}")