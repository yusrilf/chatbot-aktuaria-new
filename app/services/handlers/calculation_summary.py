class CalculationSummaryGenerator:
    def generate_final_summary(self, calculation_steps: dict, extracted_fields: dict) -> dict:
        """Generate comprehensive final summary from all calculation steps."""
        
        summary = {
            "executive_summary": self._create_executive_summary(calculation_steps),
            "key_results": self._extract_key_results(calculation_steps),
            "methodology_used": self._summarize_methodology(calculation_steps),
            "assumptions_applied": self._list_assumptions(calculation_steps),
            "recommendations": self._generate_recommendations(calculation_steps)
        }
        
        return summary
    
    def _create_executive_summary(self, steps: dict) -> str:
        """Create executive summary in Indonesian."""
        return f"""Berdasarkan perhitungan aktuaria yang telah dilakukan melalui {len(steps)} tahapan, 
        diperoleh hasil valuasi dengan tingkat kepercayaan tinggi. Perhitungan mencakup 
        analisis multiple decrement, perhitungan manfaat, dan analisis sensitivitas."""