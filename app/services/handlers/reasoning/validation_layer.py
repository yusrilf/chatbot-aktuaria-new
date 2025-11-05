from typing import Dict, List, Any, Optional, Set
import re
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """
    Represents a validation rule for Tree of Thought responses.
    """
    name: str
    description: str
    keywords: List[str]
    required: bool = True
    weight: float = 1.0

class ValidationLayer:
    """
    Validation Layer untuk Tree of Thought responses.
    Mengecek keyword wajib, format output, dan kualitas jawaban.
    """
    
    def __init__(self):
        self.actuarial_keywords = self._setup_actuarial_keywords()
        self.calculation_keywords = self._setup_calculation_keywords()
        self.format_validators = self._setup_format_validators()
        
    def _setup_actuarial_keywords(self) -> Dict[str, ValidationRule]:
        """
        Setup keyword validation rules untuk topik aktuaria.
        
        Returns:
            Dict mapping rule names to ValidationRule objects
        """
        return {
            "pvdbo": ValidationRule(
                name="pvdbo",
                description="Present Value of Defined Benefit Obligation",
                keywords=["pvdbo", "present value", "defined benefit", "obligation"],
                required=True,
                weight=2.0
            ),
            "service_cost": ValidationRule(
                name="service_cost",
                description="Service Cost calculation",
                keywords=["service cost", "current service", "biaya jasa"],
                required=True,
                weight=1.5
            ),
            "discount_rate": ValidationRule(
                name="discount_rate",
                description="Discount rate or interest rate",
                keywords=["discount rate", "tingkat diskonto", "suku bunga", "interest rate"],
                required=True,
                weight=1.5
            ),
            "mortality_table": ValidationRule(
                name="mortality_table",
                description="Mortality table reference",
                keywords=["mortality", "tabel mortalitas", "kematian", "tmii"],
                required=False,
                weight=1.0
            ),
            "spot_rate": ValidationRule(
                name="spot_rate",
                description="Spot rate from yield curve",
                keywords=["spot rate", "yield curve", "kurva hasil", "tenor"],
                required=True,
                weight=1.5
            )
        }
    
    def _setup_calculation_keywords(self) -> Dict[str, ValidationRule]:
        """
        Setup keyword validation rules untuk perhitungan.
        
        Returns:
            Dict mapping rule names to ValidationRule objects
        """
        return {
            "numerical_result": ValidationRule(
                name="numerical_result",
                description="Must contain numerical results",
                keywords=[r"\d+[.,]?\d*", r"\d+%", "hasil", "nilai"],
                required=True,
                weight=2.0
            ),
            "formula": ValidationRule(
                name="formula",
                description="Should contain calculation formula",
                keywords=["rumus", "formula", "perhitungan", "=" , "+", "-", "*", "/"],
                required=False,
                weight=1.0
            ),
            "units": ValidationRule(
                name="units",
                description="Should specify units or currency",
                keywords=["rupiah", "rp", "idr", "persen", "%", "tahun", "bulan"],
                required=False,
                weight=0.5
            )
        }
    
    def _setup_format_validators(self) -> Dict[str, callable]:
        """
        Setup format validation functions.
        
        Returns:
            Dict mapping validator names to functions
        """
        return {
            "json_structure": self._validate_json_structure,
            "response_length": self._validate_response_length,
            "language_consistency": self._validate_language_consistency,
            "completeness": self._validate_completeness
        }
    
    def validate_response(
        self, 
        response: str, 
        task_type: str = "general",
        required_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate Tree of Thought response comprehensively.
        
        Args:
            response: The response text to validate
            task_type: Type of task ("theory", "calculation", "general")
            required_keywords: Additional required keywords
            
        Returns:
            Dict containing validation results and scores
        """
        try:
            validation_result = {
                "is_valid": True,
                "overall_score": 0.0,
                "keyword_validation": {},
                "format_validation": {},
                "issues": [],
                "suggestions": []
            }
            
            # 1. Keyword validation
            keyword_score = self._validate_keywords(
                response, task_type, required_keywords
            )
            validation_result["keyword_validation"] = keyword_score
            
            # 2. Format validation
            format_score = self._validate_formats(response, task_type)
            validation_result["format_validation"] = format_score
            
            # 3. Calculate overall score
            overall_score = (
                keyword_score.get("score", 0.0) * 0.6 +
                format_score.get("score", 0.0) * 0.4
            )
            validation_result["overall_score"] = overall_score
            
            # 4. Determine if valid (threshold: 0.6)
            validation_result["is_valid"] = overall_score >= 0.6
            
            # 5. Collect issues and suggestions
            validation_result["issues"].extend(
                keyword_score.get("issues", []) + format_score.get("issues", [])
            )
            validation_result["suggestions"].extend(
                keyword_score.get("suggestions", []) + format_score.get("suggestions", [])
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in response validation: {e}")
            return {
                "is_valid": False,
                "overall_score": 0.0,
                "error": str(e)
            }
    
    def _validate_keywords(
        self, 
        response: str, 
        task_type: str,
        additional_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate presence of required keywords.
        
        Args:
            response: Response text
            task_type: Task type
            additional_keywords: Additional required keywords
            
        Returns:
            Dict containing keyword validation results
        """
        try:
            response_lower = response.lower()
            result = {
                "score": 0.0,
                "matched_rules": [],
                "missing_rules": [],
                "issues": [],
                "suggestions": []
            }
            
            # Select appropriate keyword rules
            if task_type == "calculation":
                rules = {**self.actuarial_keywords, **self.calculation_keywords}
            else:
                rules = self.actuarial_keywords
            
            total_weight = sum(rule.weight for rule in rules.values() if rule.required)
            matched_weight = 0.0
            
            # Check each rule
            for rule_name, rule in rules.items():
                matched = False
                for keyword in rule.keywords:
                    if re.search(keyword.lower(), response_lower):
                        matched = True
                        break
                
                if matched:
                    result["matched_rules"].append(rule_name)
                    if rule.required:
                        matched_weight += rule.weight
                else:
                    if rule.required:
                        result["missing_rules"].append(rule_name)
                        result["issues"].append(
                            f"Missing required keywords for: {rule.description}"
                        )
                        result["suggestions"].append(
                            f"Include keywords: {', '.join(rule.keywords[:3])}"
                        )
            
            # Check additional keywords
            if additional_keywords:
                for keyword in additional_keywords:
                    if keyword.lower() not in response_lower:
                        result["issues"].append(f"Missing required keyword: {keyword}")
                        result["suggestions"].append(f"Include keyword: {keyword}")
            
            # Calculate score
            if total_weight > 0:
                result["score"] = min(matched_weight / total_weight, 1.0)
            else:
                result["score"] = 1.0
            
            return result
            
        except Exception as e:
            logger.error(f"Error in keyword validation: {e}")
            return {"score": 0.0, "error": str(e)}
    
    def _validate_formats(self, response: str, task_type: str) -> Dict[str, Any]:
        """
        Validate response format and structure.
        
        Args:
            response: Response text
            task_type: Task type
            
        Returns:
            Dict containing format validation results
        """
        try:
            result = {
                "score": 0.0,
                "passed_checks": [],
                "failed_checks": [],
                "issues": [],
                "suggestions": []
            }
            
            total_checks = len(self.format_validators)
            passed_checks = 0
            
            # Run each format validator
            for validator_name, validator_func in self.format_validators.items():
                try:
                    is_valid, message = validator_func(response, task_type)
                    if is_valid:
                        result["passed_checks"].append(validator_name)
                        passed_checks += 1
                    else:
                        result["failed_checks"].append(validator_name)
                        result["issues"].append(message)
                        result["suggestions"].append(
                            self._get_format_suggestion(validator_name)
                        )
                except Exception as e:
                    logger.warning(f"Format validator {validator_name} failed: {e}")
                    result["failed_checks"].append(validator_name)
            
            # Calculate score
            result["score"] = passed_checks / total_checks if total_checks > 0 else 0.0
            
            return result
            
        except Exception as e:
            logger.error(f"Error in format validation: {e}")
            return {"score": 0.0, "error": str(e)}
    
    def _validate_json_structure(self, response: str, task_type: str) -> tuple:
        """
        Validate if response contains valid JSON structure when expected.
        
        Args:
            response: Response text
            task_type: Task type
            
        Returns:
            Tuple of (is_valid, message)
        """
        if task_type != "calculation":
            return True, "JSON validation not required for this task type"
        
        # Look for JSON-like structures
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        
        if not json_matches:
            return False, "No JSON structure found in calculation response"
        
        # Try to parse found JSON
        for json_str in json_matches:
            try:
                json.loads(json_str)
                return True, "Valid JSON structure found"
            except json.JSONDecodeError:
                continue
        
        return False, "Found JSON-like structure but invalid format"
    
    def _validate_response_length(self, response: str, task_type: str) -> tuple:
        """
        Validate response length appropriateness.
        
        Args:
            response: Response text
            task_type: Task type
            
        Returns:
            Tuple of (is_valid, message)
        """
        length = len(response.strip())
        
        if length < 50:
            return False, "Response too short, may lack detail"
        elif length > 2000:
            return False, "Response too long, may be verbose"
        else:
            return True, "Response length appropriate"
    
    def _validate_language_consistency(self, response: str, task_type: str) -> tuple:
        """
        Validate language consistency (Indonesian/English mix).
        
        Args:
            response: Response text
            task_type: Task type
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Simple heuristic: check for common Indonesian words
        indonesian_indicators = [
            "adalah", "dengan", "untuk", "dari", "yang", "ini", "itu",
            "dapat", "akan", "pada", "dalam", "sebagai", "atau", "dan"
        ]
        
        response_lower = response.lower()
        indonesian_count = sum(
            1 for word in indonesian_indicators 
            if word in response_lower
        )
        
        # If response has Indonesian indicators, it should be primarily Indonesian
        if indonesian_count > 3:
            english_indicators = ["the", "and", "or", "is", "are", "was", "were", "have", "has"]
            english_count = sum(
                1 for word in english_indicators 
                if word in response_lower
            )
            
            if english_count > indonesian_count:
                return False, "Mixed language usage detected"
        
        return True, "Language consistency maintained"
    
    def _validate_completeness(self, response: str, task_type: str) -> tuple:
        """
        Validate response completeness.
        
        Args:
            response: Response text
            task_type: Task type
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check for incomplete sentences or abrupt endings
        if response.strip().endswith(('...', '..', 'dll', 'etc')):
            return False, "Response appears incomplete"
        
        # Check for proper conclusion
        conclusion_indicators = [
            "kesimpulan", "hasil", "jadi", "sehingga", "oleh karena itu",
            "therefore", "thus", "in conclusion", "result"
        ]
        
        response_lower = response.lower()
        has_conclusion = any(
            indicator in response_lower 
            for indicator in conclusion_indicators
        )
        
        if task_type == "calculation" and not has_conclusion:
            return False, "Calculation response should include conclusion"
        
        return True, "Response appears complete"
    
    def _get_format_suggestion(self, validator_name: str) -> str:
        """
        Get suggestion for failed format validation.
        
        Args:
            validator_name: Name of failed validator
            
        Returns:
            Suggestion string
        """
        suggestions = {
            "json_structure": "Include structured data in JSON format for calculations",
            "response_length": "Adjust response length to be more appropriate",
            "language_consistency": "Maintain consistent language throughout response",
            "completeness": "Ensure response has proper conclusion and completeness"
        }
        
        return suggestions.get(validator_name, "Review and improve response format")
    
    def get_validation_summary(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of multiple validation results.
        
        Args:
            validation_results: List of validation result dicts
            
        Returns:
            Summary dict with aggregated metrics
        """
        try:
            if not validation_results:
                return {"error": "No validation results provided"}
            
            valid_count = sum(1 for result in validation_results if result.get("is_valid", False))
            total_count = len(validation_results)
            avg_score = sum(result.get("overall_score", 0.0) for result in validation_results) / total_count
            
            # Collect all issues
            all_issues = []
            for result in validation_results:
                all_issues.extend(result.get("issues", []))
            
            # Count issue types
            issue_types = {}
            for issue in all_issues:
                issue_type = issue.split(":")[0] if ":" in issue else "general"
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            return {
                "total_responses": total_count,
                "valid_responses": valid_count,
                "validation_rate": valid_count / total_count,
                "average_score": avg_score,
                "common_issues": issue_types,
                "overall_quality": "high" if avg_score >= 0.8 else "medium" if avg_score >= 0.6 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error creating validation summary: {e}")
            return {"error": str(e)}