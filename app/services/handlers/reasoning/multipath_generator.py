"""Multi-Path Reasoning Generator untuk Tree of Thought Implementation.

Service ini menghasilkan multiple jalur reasoning untuk setiap pertanyaan,
memungkinkan evaluasi dan pemilihan jalur terbaik dalam ToT pipeline.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import re
from enum import Enum

logger = logging.getLogger(__name__)

class ReasoningPathType(Enum):
    """Tipe jalur reasoning yang berbeda."""
    DEFINITION_FOCUSED = "definition"  # Fokus pada definisi dan konsep
    CALCULATION_FOCUSED = "calculation"  # Fokus pada perhitungan dan angka
    IMPLEMENTATION_FOCUSED = "implementation"  # Fokus pada implementasi praktis
    CONTEXT_FOCUSED = "context"  # Fokus pada konteks dan latar belakang

class MultipathReasoningGenerator:
    """Generator untuk multiple jalur reasoning dalam ToT pipeline."""
    
    def __init__(self, llm: LLM):
        """
        Initialize Multi-Path Reasoning Generator.
        
        Args:
            llm: Language model untuk generate reasoning paths
        """
        self.llm = llm
        self._init_reasoning_chains()
        
    def _init_reasoning_chains(self) -> None:
        """Initialize LLM chains untuk berbagai tipe reasoning."""
        try:
            # Template untuk generate multiple reasoning paths
            multipath_template = """
Anda adalah asisten ahli aktuaria yang menggunakan Tree of Thought reasoning.

TUGAS:
- Generate 3-4 jalur reasoning berbeda untuk menjawab pertanyaan
- Setiap jalur harus fokus pada aspek berbeda
- Berikan jawaban lengkap untuk setiap jalur
- Sertakan confidence score (0-100) untuk setiap jalur

Pertanyaan: {question}
Konteks Dokumen: {context}
Riwayat Chat: {chat_history}

Jalur Reasoning yang harus dibuat:
1. DEFINITION PATH: Fokus pada definisi, konsep, dan teori
2. CALCULATION PATH: Fokus pada perhitungan, angka, dan formula
3. IMPLEMENTATION PATH: Fokus pada implementasi praktis dan contoh
4. CONTEXT PATH: Fokus pada konteks, latar belakang, dan relevansi

Format output JSON:
{{
    "reasoning_paths": [
        {{
            "path_type": "definition",
            "reasoning_steps": ["langkah 1", "langkah 2", "langkah 3"],
            "answer": "jawaban lengkap untuk jalur ini",
            "confidence_score": 85,
            "key_points": ["poin kunci 1", "poin kunci 2"],
            "sources_used": ["dokumen1.md", "dokumen2.md"]
        }},
        {{
            "path_type": "calculation",
            "reasoning_steps": ["langkah 1", "langkah 2"],
            "answer": "jawaban dengan fokus perhitungan",
            "confidence_score": 90,
            "key_points": ["poin kunci"],
            "sources_used": ["dokumen3.md"]
        }}
    ],
    "meta_reasoning": "penjelasan mengapa jalur-jalur ini dipilih"
}}

Output JSON:"""
            
            self.multipath_prompt = PromptTemplate(
                input_variables=["question", "context", "chat_history"],
                template=multipath_template
            )
            
            self.multipath_chain = LLMChain(
                llm=self.llm,
                prompt=self.multipath_prompt,
                verbose=False
            )
            
            logger.info("Multi-path reasoning chains initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing reasoning chains: {e}")
            raise
    
    def generate_reasoning_paths(self, 
                               question: str,
                               context: str,
                               chat_history: str = "") -> Dict[str, Any]:
        """
        Generate multiple reasoning paths untuk pertanyaan.
        
        Args:
            question: Pertanyaan dari user
            context: Konteks dokumen yang relevan
            chat_history: Riwayat percakapan
            
        Returns:
            Dict berisi multiple reasoning paths dan metadata
        """
        try:
            logger.info(f"Generating reasoning paths for: {question[:100]}...")
            
            # Generate multiple reasoning paths
            response = self.multipath_chain.run(
                question=question,
                context=context,
                chat_history=chat_history or "Tidak ada riwayat sebelumnya"
            )
            
            # Parse response
            reasoning_result = self._parse_reasoning_response(response)
            
            # Validate dan enhance paths
            enhanced_result = self._enhance_reasoning_paths(reasoning_result, question)
            
            logger.info(f"Generated {len(enhanced_result.get('reasoning_paths', []))} reasoning paths")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error generating reasoning paths: {e}")
            return self._get_fallback_reasoning(question, context)
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response dari LLM."""
        try:
            # Clean response
            cleaned_response = self._clean_json_response(response)
            
            # Parse JSON
            result = json.loads(cleaned_response)
            
            # Validate structure
            if "reasoning_paths" not in result:
                raise ValueError("Missing reasoning_paths in response")
                
            if not isinstance(result["reasoning_paths"], list):
                raise ValueError("reasoning_paths must be a list")
                
            # Validate each path
            for i, path in enumerate(result["reasoning_paths"]):
                required_fields = ["path_type", "reasoning_steps", "answer", "confidence_score"]
                for field in required_fields:
                    if field not in path:
                        logger.warning(f"Missing {field} in path {i}, adding default")
                        path[field] = self._get_default_field_value(field)
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse reasoning response: {e}")
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
    
    def _get_default_field_value(self, field: str) -> Any:
        """Get default value untuk missing fields."""
        defaults = {
            "path_type": "general",
            "reasoning_steps": ["Analisis pertanyaan", "Cari informasi relevan", "Berikan jawaban"],
            "answer": "Jawaban tidak tersedia",
            "confidence_score": 50,
            "key_points": [],
            "sources_used": []
        }
        return defaults.get(field, "")
    
    def _enhance_reasoning_paths(self, reasoning_result: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Enhance reasoning paths dengan metadata tambahan."""
        try:
            paths = reasoning_result.get("reasoning_paths", [])
            
            # Ensure we have 3-4 paths
            if len(paths) < 3:
                # Add missing paths
                missing_types = self._get_missing_path_types(paths)
                for path_type in missing_types[:4-len(paths)]:
                    fallback_path = self._create_fallback_path(path_type, question)
                    paths.append(fallback_path)
            elif len(paths) > 4:
                # Keep top 4 by confidence
                paths = sorted(paths, key=lambda x: x.get("confidence_score", 0), reverse=True)[:4]
            
            # Add metadata
            enhanced_result = {
                "reasoning_paths": paths,
                "total_paths": len(paths),
                "avg_confidence": sum(p.get("confidence_score", 0) for p in paths) / len(paths) if paths else 0,
                "path_types": [p.get("path_type", "unknown") for p in paths],
                "meta_reasoning": reasoning_result.get("meta_reasoning", "Multiple reasoning paths generated"),
                "question": question,
                "generation_timestamp": self._get_timestamp()
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error enhancing reasoning paths: {e}")
            return reasoning_result
    
    def _get_missing_path_types(self, existing_paths: List[Dict[str, Any]]) -> List[str]:
        """Get path types yang belum ada."""
        existing_types = {p.get("path_type", "") for p in existing_paths}
        all_types = {"definition", "calculation", "implementation", "context"}
        return list(all_types - existing_types)
    
    def _create_fallback_path(self, path_type: str, question: str) -> Dict[str, Any]:
        """Create fallback reasoning path."""
        fallback_answers = {
            "definition": f"Definisi dan konsep terkait: {question}",
            "calculation": f"Aspek perhitungan dari: {question}", 
            "implementation": f"Implementasi praktis untuk: {question}",
            "context": f"Konteks dan latar belakang: {question}"
        }
        
        return {
            "path_type": path_type,
            "reasoning_steps": ["Analisis pertanyaan", "Identifikasi aspek kunci", "Berikan jawaban"],
            "answer": fallback_answers.get(path_type, f"Jawaban untuk {question}"),
            "confidence_score": 60,
            "key_points": [f"Aspek {path_type}"],
            "sources_used": [],
            "fallback_generated": True
        }
    
    def _get_fallback_reasoning(self, question: str, context: str) -> Dict[str, Any]:
        """Fallback reasoning jika LLM gagal."""
        logger.info("Using fallback reasoning generation")
        
        fallback_paths = [
            {
                "path_type": "definition",
                "reasoning_steps": ["Analisis definisi", "Cari konsep kunci", "Berikan penjelasan"],
                "answer": f"Berdasarkan konteks yang tersedia: {context[:200]}...",
                "confidence_score": 70,
                "key_points": ["Definisi dasar"],
                "sources_used": ["konteks dokumen"]
            },
            {
                "path_type": "implementation",
                "reasoning_steps": ["Identifikasi kebutuhan", "Cari solusi praktis", "Berikan panduan"],
                "answer": f"Untuk implementasi praktis terkait: {question}",
                "confidence_score": 65,
                "key_points": ["Aspek praktis"],
                "sources_used": ["konteks dokumen"]
            }
        ]
        
        return {
            "reasoning_paths": fallback_paths,
            "total_paths": len(fallback_paths),
            "avg_confidence": 67.5,
            "path_types": ["definition", "implementation"],
            "meta_reasoning": "Fallback reasoning paths generated",
            "question": question,
            "fallback_used": True,
            "generation_timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()