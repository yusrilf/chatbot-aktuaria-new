#!/usr/bin/env python3
"""Calculation Explanation Generator for Actuarial Calculations.

This module provides functionality to generate human-readable explanations
for complex actuarial calculation steps and results.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class CalculationExplanationGenerator:
    """Generate step-by-step explanations for calculation results.
    
    This class provides methods to generate human-readable explanations
    for actuarial calculation steps and final summaries in Indonesian.
    """
    
    def __init__(self, llm: Optional[Any] = None) -> None:
        """Initialize the calculation explanation generator.
        
        Args:
            llm: Language model instance for generating explanations
        """
        self.llm = llm
        logger.info("CalculationExplanationGenerator initialized")
    
    def generate_step_explanation(self, step_name: str, step_title: str, 
                                step_result: Dict[str, Any], 
                                question: str) -> str:
        """Generate explanation for a single calculation step.
        
        Args:
            step_name: Internal name of the calculation step
            step_title: Human-readable title of the step
            step_result: Dictionary containing calculation results
            question: Original question being answered
            
        Returns:
            Human-readable explanation of the calculation step in Indonesian
            
        Note:
            Uses LLM to generate contextual explanations that explain
            what was calculated, why it's important, and how it relates
            to the overall actuarial analysis.
        """
        
        explanation_template = """
Anda adalah asisten aktuaria yang menjelaskan perhitungan step-by-step.

TUGAS: Jelaskan dengan kata-kata yang mudah dipahami hasil perhitungan berikut:

STEP: {step_title}
HASIL PERHITUNGAN: {step_result}
PERTANYAAN ASLI: {question}

Buat penjelasan yang:
1. Menjelaskan apa yang dihitung pada step ini
2. Mengapa step ini penting dalam konteks pertanyaan
3. Bagaimana hasil ini akan digunakan untuk step selanjutnya
4. Jika ada angka, jelaskan artinya dalam konteks aktuaria

Format: paragraf naratif yang mudah dipahami (bukan bullet points).
"""
        
        try:
            prompt = PromptTemplate(
                input_variables=["step_title", "step_result", "question"],
                template=explanation_template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            explanation = chain.run({
                "step_title": step_title,
                "step_result": str(step_result),
                "question": question
            })
            
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Error generating step explanation for {step_name}: {e}")
            return f"Penjelasan untuk {step_title}: {str(step_result)}"
    
    def generate_final_summary(self, all_steps: Dict[str, Any], 
                             question: str, 
                             final_answer: str) -> str:
        """Generate final summary explanation of all calculation steps.
        
        Args:
            all_steps: Dictionary containing all calculation steps and results
            question: Original question being answered
            final_answer: Final calculated answer
            
        Returns:
            Comprehensive summary explanation in Indonesian
            
        Note:
            Creates a narrative explanation that connects all calculation
            steps and provides actuarial insights about the final result.
        """
        
        summary_template = """
Anda adalah asisten aktuaria yang membuat ringkasan final perhitungan.

TUGAS: Buat penjelasan final yang menggabungkan semua step perhitungan:

PERTANYAAN: {question}
SEMUA STEP PERHITUNGAN: {all_steps}
JAWaban FINAL: {final_answer}

Buat ringkasan yang:
1. Menjelaskan alur perhitungan dari awal sampai akhir
2. Menghubungkan setiap step dengan step berikutnya
3. Menjelaskan makna hasil akhir dalam konteks aktuaria
4. Memberikan insight atau interpretasi hasil

Format: penjelasan naratif yang komprehensif dan mudah dipahami.
"""
        
        try:
            prompt = PromptTemplate(
                input_variables=["question", "all_steps", "final_answer"],
                template=summary_template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            summary = chain.run({
                "question": question,
                "all_steps": str(all_steps),
                "final_answer": final_answer
            })
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            return f"Ringkasan perhitungan: {final_answer}"