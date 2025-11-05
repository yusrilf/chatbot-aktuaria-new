# ... existing code ...

def _format_final_response(self, calculation_results: dict) -> dict:
    """Format final response with comprehensive summary."""
    
    # Generate final summary
    summary_generator = CalculationSummaryGenerator()
    final_summary = summary_generator.generate_final_summary(
        calculation_results['calculation_steps'],
        calculation_results['extracted_fields']
    )
    
    return {
        "answer": self._create_narrative_answer(calculation_results, final_summary),
        "calculation_steps": calculation_results['calculation_steps'],
        "final_summary": final_summary,  # NEW: Comprehensive summary
        "confidence": calculation_results['confidence'],
        "extracted_fields": calculation_results['extracted_fields'],
        "sources": calculation_results['sources']
    }

def _create_narrative_answer(self, results: dict, summary: dict) -> str:
    """Create narrative answer in Indonesian."""
    return f"""Perhitungan aktuaria telah selesai dilakukan dengan hasil sebagai berikut:
    
    {summary['executive_summary']}
    
    **Hasil Utama:**
    - PVFB: {self._extract_pvfb_value(results)}
    - PVDBO: {self._extract_pvdbo_value(results)}
    - Tingkat Diskonto: {self._extract_discount_rate(results)}
    
    **Metodologi:** {summary['methodology_used']}
    
    **Rekomendasi:** {summary['recommendations']}
    
    Detail lengkap tersedia pada field `calculation_steps` dan `final_summary`."""