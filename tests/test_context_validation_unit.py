"""
Unit tests for context information helper functions.

This module contains unit tests for the helper functions used to generate
context information in API responses, ensuring they work correctly in isolation.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.chat_service_original_helpers import (
    get_confidence_level,
    assess_tot_response_quality,
    classify_error_type,
    count_sources_by_type,
    extract_relevance_scores,
    create_enhanced_context_info
)


class TestContextHelperFunctions(unittest.TestCase):
    """Test cases for context information helper functions."""
    
    def test_get_confidence_level(self) -> None:
        """Test confidence level categorization."""
        # Test very_low confidence
        self.assertEqual(get_confidence_level(0.0), "very_low")
        self.assertEqual(get_confidence_level(0.3), "very_low")
        
        # Test low confidence
        self.assertEqual(get_confidence_level(0.4), "low")
        self.assertEqual(get_confidence_level(0.5), "low")
        
        # Test medium confidence
        self.assertEqual(get_confidence_level(0.6), "medium")
        self.assertEqual(get_confidence_level(0.7), "medium")
        
        # Test high confidence
        self.assertEqual(get_confidence_level(0.8), "high")
        self.assertEqual(get_confidence_level(1.0), "high")
        
        # Test edge cases
        self.assertEqual(get_confidence_level(-0.1), "very_low")  # Below 0
        self.assertEqual(get_confidence_level(1.1), "high")  # Above 1
    
    def test_assess_tot_response_quality(self) -> None:
        """Test ToT response quality assessment."""
        # Test with empty reasoning steps
        self.assertEqual(assess_tot_response_quality(0.2, []), "poor")
        
        # Test with low confidence and few steps
        self.assertEqual(assess_tot_response_quality(0.2, ["step1"]), "poor")
        
        # Test with medium confidence and some steps
        self.assertEqual(assess_tot_response_quality(0.5, ["step1", "step2"]), "good")
        
        # Test with high confidence and many steps
        self.assertEqual(assess_tot_response_quality(0.8, ["step1", "step2", "step3"]), "excellent")
        
        # Test with medium confidence and few steps
        self.assertEqual(assess_tot_response_quality(0.4, ["step1"]), "fair")
    
    def test_classify_error_type(self) -> None:
        """Test error type classification."""
        # Test with None error
        self.assertEqual(classify_error_type(None), "unknown")
        
        # Test with empty string
        self.assertEqual(classify_error_type(""), "unknown")
        
        # Test timeout errors
        self.assertEqual(classify_error_type("Connection timeout occurred"), "timeout")
        self.assertEqual(classify_error_type("Request timed out"), "timeout")
        
        # Test network errors
        self.assertEqual(classify_error_type("Network connection failed"), "network")
        self.assertEqual(classify_error_type("Connection error"), "network")
        
        # Test API errors
        self.assertEqual(classify_error_type("OpenAI API error"), "api_error")
        self.assertEqual(classify_error_type("API rate limit exceeded"), "api_error")
        
        # Test processing errors
        self.assertEqual(classify_error_type("ToT processing failed"), "processing_error")
        self.assertEqual(classify_error_type("Processing error occurred"), "processing_error")
        
        # Test retrieval errors
        self.assertEqual(classify_error_type("Document retrieval failed"), "retrieval_error")
        self.assertEqual(classify_error_type("Retrieval system error"), "retrieval_error")
        
        # Test unknown error
        self.assertEqual(classify_error_type("Some random error"), "general_error")
    
    def test_count_sources_by_type(self) -> None:
        """Test source counting by type."""
        # Test with empty sources
        self.assertEqual(count_sources_by_type([]), {})
        
        # Test with None sources
        self.assertEqual(count_sources_by_type(None), {})
        
        # Test with mixed sources
        sources = [
            {"source": "document1.pdf", "type": "pdf"},
            {"source": "document2.pdf", "type": "pdf"},
            {"source": "webpage1.html", "type": "web"},
            {"source": "database_query", "type": "database"},
            {"source": "document3.pdf", "type": "pdf"}
        ]
        
        expected = {
            "pdf": 3,
            "web": 1,
            "database": 1
        }
        
        self.assertEqual(count_sources_by_type(sources), expected)
        
        # Test with sources without type field
        sources_no_type = [
            {"source": "document1.pdf"},
            {"source": "document2.pdf"}
        ]
        
        expected_no_type = {"unknown": 2}
        self.assertEqual(count_sources_by_type(sources_no_type), expected_no_type)
    
    def test_extract_relevance_scores(self) -> None:
        """Test relevance score extraction."""
        # Test with empty sources
        self.assertEqual(extract_relevance_scores([]), [])
        
        # Test with None sources
        self.assertEqual(extract_relevance_scores(None), [])
        
        # Test with sources containing scores (using 'score' field)
        sources_with_scores = [
            {"source": "doc1.pdf", "score": 0.8},
            {"source": "doc2.pdf", "score": 0.6},
            {"source": "doc3.pdf", "score": 0.9}
        ]
        
        expected_scores = [0.8, 0.6, 0.9]
        self.assertEqual(extract_relevance_scores(sources_with_scores), expected_scores)
        
        # Test with sources containing relevance scores
        sources_with_relevance = [
            {"source": "doc1.pdf", "relevance": 0.7},
            {"source": "doc2.pdf", "relevance": 0.5}
        ]
        
        expected_relevance = [0.7, 0.5]
        self.assertEqual(extract_relevance_scores(sources_with_relevance), expected_relevance)
        
        # Test with sources without scores
        sources_no_scores = [
            {"source": "doc1.pdf"},
            {"source": "doc2.pdf"}
        ]
        
        self.assertEqual(extract_relevance_scores(sources_no_scores), [])
        
        # Test with mixed sources (some with scores, some without)
        mixed_sources = [
            {"source": "doc1.pdf", "score": 0.7},
            {"source": "doc2.pdf"},
            {"source": "doc3.pdf", "relevance": 0.5}
        ]
        
        expected_mixed = [0.7, 0.5]
        self.assertEqual(extract_relevance_scores(mixed_sources), expected_mixed)
    
    def test_create_enhanced_context_info(self) -> None:
        """Test enhanced context info creation."""
        # Test basic context info creation
        confidence = 0.8
        reasoning_steps = ["Step 1", "Step 2"]
        doc_retrieval_info = {
            "relevant_sections": ["section1", "section2", "section3", "section4", "section5"],
            "total_paths_generated": 10,
            "query_expansion_enabled": True,
            "strategies_used": ["semantic", "keyword"],
            "context_enhancement_ratio": 1.5,
            "question_type": "factual"
        }
        sources = [
            {"source": "doc1.pdf", "score": 0.8},
            {"source": "doc2.pdf", "score": 0.6}
        ]
        mode = "theory"
        has_error = False
        
        result = create_enhanced_context_info(
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            doc_retrieval_info=doc_retrieval_info,
            sources=sources,
            mode=mode,
            has_error=has_error
        )
        
        # Verify structure
        self.assertIn("retrieval_metadata", result)
        self.assertIn("reasoning_context", result)
        self.assertIn("document_context", result)
        self.assertIn("error_context", result)
        
        # Verify content
        self.assertEqual(result["retrieval_metadata"]["confidence_level"], "high")
        self.assertEqual(result["reasoning_context"]["reasoning_steps_count"], 2)
        self.assertEqual(result["retrieval_metadata"]["documents_retrieved"], 5)  # len of relevant_sections
        self.assertEqual(result["error_context"]["has_error"], False)
        
        # Test with error
        result_with_error = create_enhanced_context_info(
            confidence=0.3,
            reasoning_steps=[],
            doc_retrieval_info={"relevant_sections": []},
            sources=[],
            mode="calculation",
            has_error=True,
            error_msg="Test error"
        )
        
        self.assertEqual(result_with_error["error_context"]["has_error"], True)
        self.assertEqual(result_with_error["error_context"]["error_type"], "general_error")
        self.assertEqual(result_with_error["retrieval_metadata"]["confidence_level"], "very_low")


class TestContextInfoIntegration(unittest.TestCase):
    """Integration tests for context info helper functions."""
    
    def test_context_info_consistency(self) -> None:
        """Test that context info maintains consistency across different scenarios."""
        # Test scenario 1: High confidence with good sources
        scenario1 = create_enhanced_context_info(
            confidence=0.9,
            reasoning_steps=["Analysis step"],
            doc_retrieval_info={"relevant_sections": ["section1", "section2"]},
            sources=[{"source": "doc1.pdf", "type": "pdf", "score": 0.9}],
            mode="high_quality_mode",
            has_error=False
        )
        
        self.assertEqual(scenario1["retrieval_metadata"]["confidence_level"], "high")
        
        # Test scenario 2: Low confidence with errors
        scenario2 = create_enhanced_context_info(
            confidence=0.2,
            reasoning_steps=[],
            doc_retrieval_info={"relevant_sections": []},
            sources=[],
            mode="error_mode",
            has_error=True,
            error_msg="Processing failed"
        )
        
        self.assertEqual(scenario2["retrieval_metadata"]["confidence_level"], "very_low")
        self.assertEqual(scenario2["error_context"]["has_error"], True)
        self.assertEqual(scenario2["error_context"]["error_type"], "processing_error")
    
    def test_context_info_data_types(self) -> None:
        """Test that all context info fields have correct data types."""
        context = create_enhanced_context_info(
            confidence=0.7,
            reasoning_steps=["step 1"],
            doc_retrieval_info={"relevant_sections": ["section1", "section2", "section3", "section4", "section5"]},
            sources=[{"source": "test.pdf"}],
            mode="test",
            has_error=False
        )
        
        # Check data types
        self.assertIsInstance(context["retrieval_metadata"]["confidence_level"], str)
        self.assertIsInstance(context["retrieval_metadata"]["documents_retrieved"], int)
        self.assertIsInstance(context["retrieval_metadata"]["processing_mode"], str)
        self.assertIsInstance(context["retrieval_metadata"]["response_quality"], str)
        
        self.assertIsInstance(context["reasoning_context"]["reasoning_steps_count"], int)
        self.assertIsInstance(context["reasoning_context"]["paths_evaluated"], int)
        self.assertIsInstance(context["reasoning_context"]["query_expansion_used"], bool)
        
        self.assertIsInstance(context["error_context"]["has_error"], bool)
        self.assertIsInstance(context["error_context"]["fallback_used"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)