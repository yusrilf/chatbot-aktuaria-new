"""LLM Verifier for Citation and Claim Validation.

This module implements an LLM-based verifier that checks if claims in answers
are properly cited from the provided context and flags unsourced claims.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain.llms.base import LLM

logger = logging.getLogger(__name__)


@dataclass
class CitationIssue:
    """Container for citation validation issue."""
    claim: str
    issue_type: str  # 'missing_citation', 'invalid_source', 'unsupported_claim'
    severity: str  # 'high', 'medium', 'low'
    suggestion: str
    line_number: Optional[int] = None


@dataclass
class VerificationResult:
    """Result from LLM verification."""
    is_valid: bool
    confidence_score: float
    issues: List[CitationIssue]
    verified_claims: List[str]
    total_claims: int
    citation_coverage: float  # Percentage of claims with proper citations
    verification_metadata: Dict[str, Any]


class LLMVerifier:
    """Verifies answer quality and citation accuracy using LLM."""
    
    def __init__(self, llm: LLM):
        """
        Initialize the LLM verifier.
        
        Args:
            llm: Language model for verification
        """
        self.llm = llm
        self.min_confidence_threshold = 0.7
        self.max_unsourced_claims = 2
        logger.info("LLMVerifier initialized")
    
    def verify_answer(
        self,
        answer: str,
        context: str,
        original_query: str,
        strict_mode: bool = True
    ) -> VerificationResult:
        """
        Verify answer against context for citation accuracy.
        
        Args:
            answer: Generated answer to verify
            context: Source context used for answer generation
            original_query: Original user query
            strict_mode: Whether to use strict verification rules
            
        Returns:
            VerificationResult with validation details
        """
        try:
            logger.info("Starting LLM verification of answer")
            
            # Extract claims from answer
            claims = self._extract_claims_from_answer(answer)
            
            # Verify each claim against context
            verification_prompt = self._create_verification_prompt(
                answer, context, original_query, claims, strict_mode
            )
            
            # Get LLM verification response
            try:
                # Use proper LangChain invocation method
                llm_response = self.llm.invoke(verification_prompt)
                
                # Handle different response types
                if hasattr(llm_response, 'content'):
                    response_text = llm_response.content
                elif isinstance(llm_response, str):
                    response_text = llm_response
                else:
                    response_text = str(llm_response)
            except Exception as llm_error:
                logger.error(f"LLM invocation failed: {str(llm_error)}")
                # Fallback to direct call if invoke fails
                try:
                    llm_response = self.llm(verification_prompt)
                    response_text = str(llm_response)
                except Exception as fallback_error:
                    logger.error(f"LLM fallback call also failed: {str(fallback_error)}")
                    raise fallback_error
            
            # Parse verification result
            result = self._parse_verification_response(
                response_text, answer, claims, strict_mode
            )
            
            logger.info(f"Verification completed: {result.citation_coverage:.1%} citation coverage")
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM verification: {str(e)}")
            return self._create_error_result(answer, str(e))
    
    def _extract_claims_from_answer(self, answer: str) -> List[str]:
        """Extract factual claims from the answer."""
        try:
            # Split answer into sentences
            sentences = re.split(r'[.!?]+', answer)
            
            claims = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                
                # Skip questions and greetings
                if sentence.startswith(('Apakah', 'Bagaimana', 'Mengapa', 'Kapan', 'Dimana')):
                    continue
                if any(greeting in sentence.lower() for greeting in ['halo', 'selamat', 'terima kasih']):
                    continue
                
                # Consider as claim if it contains factual indicators
                factual_indicators = [
                    'adalah', 'merupakan', 'sebesar', 'mencapai', 'berdasarkan',
                    'sesuai', 'menurut', 'dalam', 'pada', 'dengan', 'harus',
                    'wajib', 'dapat', 'tidak dapat', 'diatur', 'ditetapkan'
                ]
                
                if any(indicator in sentence.lower() for indicator in factual_indicators):
                    claims.append(sentence)
            
            logger.debug(f"Extracted {len(claims)} claims from answer")
            return claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {str(e)}")
            return [answer]  # Fallback: treat entire answer as one claim
    
    def _create_verification_prompt(
        self,
        answer: str,
        context: str,
        query: str,
        claims: List[str],
        strict_mode: bool
    ) -> str:
        """Create prompt for LLM verification."""
        
        strictness = "VERY STRICT" if strict_mode else "MODERATE"
        
        prompt = f"""You are a citation verification expert for actuarial documents. 
Your task is to verify if claims in an answer are properly supported by the provided context.

VERIFICATION MODE: {strictness}

Original Query: "{query}"

Context Documents:
{context}

Generated Answer:
{answer}

Extracted Claims to Verify:
"""
        
        for i, claim in enumerate(claims, 1):
            prompt += f"{i}. {claim}\n"
        
        prompt += f"""
VERIFICATION INSTRUCTIONS:
1. For each claim, check if it is supported by the context
2. Verify that citations reference actual content from context
3. Flag claims that make assertions not found in context
4. Check for proper attribution to source documents

{"STRICT MODE: Flag ANY claim without explicit context support" if strict_mode else "MODERATE MODE: Allow reasonable inferences from context"}

Return JSON with format:
{{
  "overall_valid": true/false,
  "confidence": 0.85,
  "claim_verification": [
    {{
      "claim_number": 1,
      "claim_text": "...",
      "is_supported": true/false,
      "supporting_evidence": "Quote from context or 'None found'",
      "issue_type": "missing_citation|invalid_source|unsupported_claim|none",
      "severity": "high|medium|low",
      "suggestion": "How to fix this issue"
    }}
  ],
  "summary": {{
    "total_claims": 5,
    "verified_claims": 4,
    "citation_coverage": 0.8,
    "major_issues": ["List of serious problems"],
    "recommendations": ["List of improvements"]
  }}
}}

Output only valid JSON."""
        
        return prompt
    
    def _parse_verification_response(
        self,
        response: str,
        original_answer: str,
        claims: List[str],
        strict_mode: bool
    ) -> VerificationResult:
        """Parse LLM verification response."""
        try:
            # Parse JSON response
            response_data = json.loads(response.strip())
            
            # Extract verification data
            overall_valid = response_data.get("overall_valid", False)
            confidence = float(response_data.get("confidence", 0.0))
            claim_verifications = response_data.get("claim_verification", [])
            summary = response_data.get("summary", {})
            
            # Create citation issues
            issues = []
            verified_claims = []
            
            for claim_data in claim_verifications:
                claim_text = claim_data.get("claim_text", "")
                is_supported = claim_data.get("is_supported", False)
                
                if is_supported:
                    verified_claims.append(claim_text)
                else:
                    # Create citation issue
                    issue = CitationIssue(
                        claim=claim_text,
                        issue_type=claim_data.get("issue_type", "unsupported_claim"),
                        severity=claim_data.get("severity", "medium"),
                        suggestion=claim_data.get("suggestion", "Add proper citation"),
                        line_number=None  # Could be enhanced to find line numbers
                    )
                    issues.append(issue)
            
            # Calculate citation coverage
            total_claims = len(claims)
            verified_count = len(verified_claims)
            citation_coverage = verified_count / total_claims if total_claims > 0 else 0.0
            
            # Determine overall validity
            is_valid = self._determine_validity(
                overall_valid, confidence, citation_coverage, issues, strict_mode
            )
            
            return VerificationResult(
                is_valid=is_valid,
                confidence_score=confidence,
                issues=issues,
                verified_claims=verified_claims,
                total_claims=total_claims,
                citation_coverage=citation_coverage,
                verification_metadata={
                    "strict_mode": strict_mode,
                    "llm_overall_valid": overall_valid,
                    "major_issues": summary.get("major_issues", []),
                    "recommendations": summary.get("recommendations", []),
                    "response_length": len(response)
                }
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing verification response: {str(e)}")
            return self._create_fallback_verification(original_answer, claims, str(e))
    
    def _determine_validity(
        self,
        llm_valid: bool,
        confidence: float,
        citation_coverage: float,
        issues: List[CitationIssue],
        strict_mode: bool
    ) -> bool:
        """Determine overall validity based on multiple factors."""
        try:
            # Count high-severity issues
            high_severity_issues = len([issue for issue in issues if issue.severity == "high"])
            
            # Strict mode criteria
            if strict_mode:
                return (
                    llm_valid and
                    confidence >= 0.8 and
                    citation_coverage >= 0.9 and
                    high_severity_issues == 0
                )
            
            # Moderate mode criteria
            else:
                return (
                    llm_valid and
                    confidence >= self.min_confidence_threshold and
                    citation_coverage >= 0.7 and
                    high_severity_issues <= 1
                )
                
        except Exception as e:
            logger.error(f"Error determining validity: {str(e)}")
            return False
    
    def _create_fallback_verification(
        self,
        answer: str,
        claims: List[str],
        error: str
    ) -> VerificationResult:
        """Create fallback verification when LLM parsing fails."""
        
        # Simple heuristic verification
        has_citations = bool(re.search(r'\([^)]*\)|【[^】]*】|\[[^\]]*\]', answer))
        estimated_coverage = 0.6 if has_citations else 0.2
        
        issues = [
            CitationIssue(
                claim="LLM verification failed",
                issue_type="verification_error",
                severity="medium",
                suggestion=f"Manual review required due to parsing error: {error}"
            )
        ]
        
        return VerificationResult(
            is_valid=has_citations,
            confidence_score=0.3,  # Low confidence due to fallback
            issues=issues,
            verified_claims=[],
            total_claims=len(claims),
            citation_coverage=estimated_coverage,
            verification_metadata={
                "fallback_reason": error,
                "heuristic_has_citations": has_citations
            }
        )
    
    def _create_error_result(self, answer: str, error: str) -> VerificationResult:
        """Create error verification result."""
        return VerificationResult(
            is_valid=False,
            confidence_score=0.0,
            issues=[
                CitationIssue(
                    claim="Verification system error",
                    issue_type="system_error",
                    severity="high",
                    suggestion=f"System error occurred: {error}"
                )
            ],
            verified_claims=[],
            total_claims=0,
            citation_coverage=0.0,
            verification_metadata={"error": error}
        )
    
    def create_verification_report(self, result: VerificationResult) -> str:
        """Create human-readable verification report."""
        try:
            report = f"""# Verification Report

## Overall Assessment
- **Valid**: {'✅ Yes' if result.is_valid else '❌ No'}
- **Confidence**: {result.confidence_score:.1%}
- **Citation Coverage**: {result.citation_coverage:.1%} ({len(result.verified_claims)}/{result.total_claims} claims)

## Issues Found ({len(result.issues)})
"""
            
            if not result.issues:
                report += "✅ No issues found\n\n"
            else:
                for i, issue in enumerate(result.issues, 1):
                    severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue.severity, "⚪")
                    report += f"""
{i}. {severity_icon} **{issue.issue_type.replace('_', ' ').title()}**
   - Claim: "{issue.claim[:100]}{'...' if len(issue.claim) > 100 else ''}"
   - Severity: {issue.severity}
   - Suggestion: {issue.suggestion}
"""
            
            # Add recommendations if available
            recommendations = result.verification_metadata.get("recommendations", [])
            if recommendations:
                report += "\n## Recommendations\n"
                for i, rec in enumerate(recommendations, 1):
                    report += f"{i}. {rec}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error creating verification report: {str(e)}")
            return f"Error creating report: {str(e)}"
    
    def suggest_improvements(self, result: VerificationResult) -> List[str]:
        """Suggest improvements based on verification result."""
        try:
            suggestions = []
            
            # Citation coverage improvements
            if result.citation_coverage < 0.8:
                suggestions.append(
                    f"Improve citation coverage from {result.citation_coverage:.1%} to at least 80%"
                )
            
            # Issue-specific suggestions
            issue_types = set(issue.issue_type for issue in result.issues)
            
            if "missing_citation" in issue_types:
                suggestions.append("Add proper citations for factual claims")
            
            if "unsupported_claim" in issue_types:
                suggestions.append("Remove or modify claims not supported by context")
            
            if "invalid_source" in issue_types:
                suggestions.append("Verify that cited sources exist in the provided context")
            
            # Confidence improvements
            if result.confidence_score < 0.7:
                suggestions.append("Review answer for accuracy and clarity")
            
            # High-severity issue handling
            high_severity_count = len([i for i in result.issues if i.severity == "high"])
            if high_severity_count > 0:
                suggestions.append(f"Address {high_severity_count} high-severity citation issues")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error creating suggestions: {str(e)}")
            return ["Manual review recommended due to system error"]
    
    def get_verification_statistics(self, result: VerificationResult) -> Dict[str, Any]:
        """Get statistics about verification result."""
        try:
            issue_counts = {}
            severity_counts = {"high": 0, "medium": 0, "low": 0}
            
            for issue in result.issues:
                # Count by type
                issue_type = issue.issue_type
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                
                # Count by severity
                severity_counts[issue.severity] += 1
            
            stats = {
                "overall_valid": result.is_valid,
                "confidence_score": result.confidence_score,
                "citation_coverage": result.citation_coverage,
                "total_claims": result.total_claims,
                "verified_claims": len(result.verified_claims),
                "total_issues": len(result.issues),
                "issue_breakdown": issue_counts,
                "severity_breakdown": severity_counts,
                "pass_rate": len(result.verified_claims) / result.total_claims if result.total_claims > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating verification statistics: {str(e)}")
            return {"error": str(e)}