#!/usr/bin/env python3
"""Prompt Management System for Actuarial Chatbot.

This module provides centralized management of all prompt templates
used throughout the actuarial chatbot system.

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

from typing import Dict, Any, Optional, List
import logging

from app.config import config

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages all prompt templates for the actuarial chatbot.
    
    This class provides centralized access to all prompt templates
    including datastory, external, custom, chain-of-thought, and
    structured planner templates.
    """
    
    def __init__(self) -> None:
        """Initialize the prompt manager.
        
        Loads all available prompt templates into memory for
        efficient access during runtime.
        """
        self._templates: Dict[str, str] = {}
        self._load_templates()
        logger.info("PromptManager initialized with templates loaded")
    
    def _load_templates(self) -> None:
        """Load all prompt templates into memory.
        
        Initializes all available prompt templates including:
        - datastory: For data analysis and discussion
        - external: For external knowledge queries
        - custom: For custom actuarial calculations
        - cot: For chain-of-thought reasoning
        - structured_planner: For structured calculation planning
        """
        self._templates = {
            'datastory': self._get_datastory_template(),
            'external': self._get_external_template(),
            'custom': self._get_custom_template(),
            'cot': self._get_cot_template(),
            'structured_planner': self._get_structured_planner_template()
        }
    
    def get_template(self, template_name: str) -> str:
        """Get a specific template by name.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            The requested template string, or default template if not found
            
        Note:
            Available templates: datastory, external, custom, cot, structured_planner
        """
        if template_name not in self._templates:
            logger.warning(f"Template '{template_name}' not found")
            return self._get_default_template()
        return self._templates[template_name]
    
    def _get_datastory_template(self) -> str:
        """Get datastory prompt template."""
        return """
Anda adalah asisten ahli aktuaria yang membantu dengan diskusi dan analisis data aktuaria.

KONTEKS DOKUMEN:
{context}

RIWAYAT PERCAKAPAN:
{chat_history}

PERTANYAAN: {question}

PETUNJUK JAWABAN:
1. Berikan analisis yang mendalam dan terstruktur
2. Gunakan data dan referensi dari dokumen yang tersedia
3. Jika ada keterbatasan data, jelaskan dengan transparan
4. Berikan rekomendasi praktis jika memungkinkan
5. Gunakan bahasa Indonesia yang profesional

Jawaban Anda:
"""
    
    def _get_external_template(self) -> str:
        """Get external prompt template."""
        return f"""
Anda adalah konsultan aktuaria berpengalaman yang membantu dengan pertanyaan umum tentang aktuaria.

KONTEKS DOKUMEN:
{{context}}

RIWAYAT PERCAKAPAN:
{{chat_history}}

PERTANYAAN: {{question}}

PETUNJUK:
- Jawab berdasarkan pengetahuan aktuaria yang solid
- Gunakan referensi dari dokumen jika tersedia
- Berikan penjelasan yang mudah dipahami
- Jika tidak yakin, katakan dengan jujur
- Fokus pada aspek praktis dan implementasi
- PENTING: Batasi jawaban maksimal {config.MAX_CONTEXT_LENGTH} karakter untuk menjaga respons tetap ringkas

Jawaban:
"""
    
    def _get_custom_template(self) -> str:
        """Get custom prompt template."""
        return f"""
Sebagai ahli aktuaria, jawab pertanyaan berikut dengan menggunakan konteks dokumen yang diberikan.

KONTEKS RELEVAN:
{{context}}

RIWAYAT PERCAKAPAN:
{{chat_history}}

PERTANYAAN PENGGUNA: {{question}}

INSTRUKSI:
1. Analisis pertanyaan dengan cermat
2. Gunakan informasi dari konteks dokumen
3. Berikan jawaban yang akurat dan komprehensif
4. Sertakan referensi ke dokumen sumber jika memungkinkan
5. Gunakan format yang mudah dibaca
6. PENTING: Batasi jawaban maksimal {config.MAX_CONTEXT_LENGTH} karakter untuk menjaga respons tetap ringkas

JAWABAN:
"""
    
    def _get_cot_template(self) -> str:
        """Get Chain of Thought prompt template."""
        return """
Anda adalah asisten ahli aktuaria. Berikan jawaban yang ringkas dan langsung pada intinya.

KONTEKS DOKUMEN:
{context}

RIWAYAT PERCAKAPAN:
{chat_history}

PERTANYAAN: {question}

INSTRUKSI:
- Jawab langsung dan ringkas (maksimal 3-4 paragraf)
- Fokus pada informasi penting saja
- Hindari penjelasan berlebihan
- Gunakan poin-poin jika perlu untuk kejelasan
- Sertakan referensi dokumen jika relevan

Jawaban:
"""
    
    def _get_default_template(self) -> str:
        """Get default fallback template."""
        return """
Sebagai asisten aktuaria, jawab pertanyaan berikut:

Konteks: {context}
Riwayat: {chat_history}
Pertanyaan: {question}

Jawaban:
"""
    
    def _get_structured_planner_template(self) -> str:
        """Get structured JSON planner template."""
        return """
You are a planning assistant for actuarial calculations. Output ONLY a JSON object following the exact schema provided.

CONTEXT DOCUMENTS:
{context}

AVAILABLE FILES:
{available_files}

QUESTION: {question}

You MUST return ONLY a valid JSON object with this exact structure:

{{
  "variables": [
    {{
      "name": "string (original name from question)",
      "canonical_name": "string (standardized variable name)",
      "required": true|false,
      "keywords": ["keyword1", "keyword2", "..."],
      "metadata_filters": {{
        "doc_type": ["financial_report", "calculation_step", "assumptions"],
        "period": "YYYY-MM-DD (if applicable)",
        "company_id": "string (if applicable)"
      }},
      "extraction_hint": "string (how to find this variable in documents)"
    }}
  ],
  "retrieval_plan": [
    {{
      "doc_type": "string (document type to search)",
      "priority": 1,
      "reason": "string (why this document type is needed)"
    }}
  ],
  "calculation_steps": [
    "Step 1: Employee Data Foundation",
    "Step 2: Multiple Decrement Analysis",
    "Step 3: Benefit Calculation",
    "Step 4: PVFB and PVDBO Calculation",
    "Step 5: Sensitivity Analysis"
  ],
  "assumptions": {{
    "mortality_table": "TMI IV",
    "discount_rate": "7%",
    "salary_increase": "assumption from documents"
  }}
}}

IMPORTANT RULES:
1. Output ONLY the JSON object - no explanatory text before or after
2. All required actuarial variables must be included: discount_rate, mortality_table, retirement_age, benefit_amount, salary, service_years
3. Use canonical names: discount_rate, mortality_table, retirement_age, benefit_amount, current_salary, service_years, current_age
4. Keywords should include Indonesian and English actuarial terms
5. Calculation steps must follow the 5-step actuarial process
6. Assumptions should reference specific values from context documents
"""

    def get_available_templates(self) -> List[str]:
        """Get list of available template names.
        
        Returns:
            List of available template names that can be used with get_template()
        """
        return list(self._templates.keys())
    
    def format_template(self, template_name: str, **kwargs) -> str:
        """Format a template with provided variables.
        
        Args:
            template_name: Name of the template to format
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted template string with variables substituted
            
        Note:
            If a required variable is missing, returns the unformatted template
            and logs an error.
        """
        template = self.get_template(template_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return template
        except Exception as e:
            logger.error(f"Error formatting template: {e}")
            return template