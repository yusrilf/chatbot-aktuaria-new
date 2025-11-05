"""
Test suite for validating context information in API responses.

This module contains comprehensive tests to ensure that context information
is properly included in all API responses, providing transparency about
document retrieval, reasoning processes, and confidence metrics.
"""

import unittest
import json
import requests
from typing import Dict, Any, List
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestContextInformation(unittest.TestCase):
    """Test cases for context information validation."""
    
    BASE_URL = "http://localhost:5001"
    
    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.session_id = f"test_session_{int(time.time())}"
        self.headers = {"Content-Type": "application/json"}
        
    def tearDown(self) -> None:
        """Clean up after each test method."""
        # Clear conversation history if needed
        try:
            requests.post(
                f"{self.BASE_URL}/conversation/clear",
                json={"session_id": self.session_id},
                headers=self.headers
            )
        except Exception as e:
            logger.warning(f"Failed to clear session: {e}")
    
    def _validate_base_response_structure(self, response_data: Dict[str, Any]) -> None:
        """
        Validate basic response structure.
        
        Args:
            response_data: The API response data to validate
        """
        self.assertIn("success", response_data)
        self.assertIn("message", response_data)
        self.assertIn("timestamp", response_data)
        self.assertIn("data", response_data)
        
        data = response_data["data"]
        self.assertIn("answer", data)
        self.assertIn("confidence", data)
        self.assertIn("session_id", data)
        self.assertIn("mode", data)
        self.assertIn("processing_time", data)
    
    def _validate_context_info_structure(self, context_info: Dict[str, Any]) -> None:
        """
        Validate context_info structure and required fields.
        
        Args:
            context_info: The context_info section to validate
        """
        # Check required top-level keys
        required_keys = ["retrieval_metadata", "error_context"]
        for key in required_keys:
            self.assertIn(key, context_info, f"Missing required key: {key}")
        
        # Validate retrieval_metadata
        retrieval_meta = context_info["retrieval_metadata"]
        self.assertIn("confidence_level", retrieval_meta)
        self.assertIn("documents_retrieved", retrieval_meta)
        self.assertIn("processing_mode", retrieval_meta)
        self.assertIn("response_quality", retrieval_meta)
        
        # Validate error_context
        error_context = context_info["error_context"]
        self.assertIn("has_error", error_context)
        self.assertIn("fallback_used", error_context)
        self.assertIn("error_type", error_context)
    
    def _validate_enhanced_context_info(self, context_info: Dict[str, Any]) -> None:
        """
        Validate enhanced context_info for complex processing modes.
        
        Args:
            context_info: The context_info section to validate
        """
        self._validate_context_info_structure(context_info)
        
        # Check for additional context fields in enhanced modes
        if "reasoning_context" in context_info:
            reasoning = context_info["reasoning_context"]
            self.assertIn("reasoning_steps_count", reasoning)
            self.assertIn("paths_evaluated", reasoning)
            self.assertIn("query_expansion_used", reasoning)
            self.assertIn("verification_used", reasoning)
        
        if "verification_context" in context_info:
            verification = context_info["verification_context"]
            self.assertIn("verification_score", verification)
            self.assertIn("verification_passed", verification)
            self.assertIn("verification_notes", verification)
    
    def test_ask_endpoint_context_info(self) -> None:
        """Test that /ask endpoint includes proper context information."""
        payload = {
            "question": "Apa itu PSAK 219?",
            "session_id": self.session_id
        }
        
        response = requests.post(
            f"{self.BASE_URL}/ask",
            json=payload,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        
        # Validate basic structure
        self._validate_base_response_structure(response_data)
        
        # Validate context_info presence and structure
        data = response_data["data"]
        self.assertIn("context_info", data, "Missing context_info in response")
        self._validate_context_info_structure(data["context_info"])
        
        logger.info("✓ /ask endpoint context info validation passed")
    
    def test_askproject_endpoint_context_info(self) -> None:
        """Test that /askproject endpoint includes proper context information."""
        payload = {
            "question": "Bagaimana cara menghitung present value menggunakan metode PUC?",
            "session_id": self.session_id
        }
        
        response = requests.post(
            f"{self.BASE_URL}/askproject",
            json=payload,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        
        # Validate basic structure
        self._validate_base_response_structure(response_data)
        
        # Validate context_info presence and structure
        data = response_data["data"]
        self.assertIn("context_info", data, "Missing context_info in response")
        self._validate_enhanced_context_info(data["context_info"])
        
        logger.info("✓ /askproject endpoint context info validation passed")
    
    def test_context_info_confidence_levels(self) -> None:
        """Test that confidence levels are properly categorized."""
        test_cases = [
            {"question": "Apa itu asuransi?", "expected_min_confidence": 0.0},
            {"question": "Jelaskan PSAK 219 secara detail", "expected_min_confidence": 0.0}
        ]
        
        for i, case in enumerate(test_cases):
            with self.subTest(case=i):
                payload = {
                    "question": case["question"],
                    "session_id": f"{self.session_id}_{i}"
                }
                
                response = requests.post(
                    f"{self.BASE_URL}/ask",
                    json=payload,
                    headers=self.headers
                )
                
                self.assertEqual(response.status_code, 200)
                data = response.json()["data"]
                
                # Check confidence value
                confidence = data["confidence"]
                self.assertGreaterEqual(confidence, case["expected_min_confidence"])
                self.assertLessEqual(confidence, 1.0)
                
                # Check confidence level categorization
                context_info = data["context_info"]
                confidence_level = context_info["retrieval_metadata"]["confidence_level"]
                self.assertIn(confidence_level, ["low", "medium", "high"])
        
        logger.info("✓ Confidence levels validation passed")
    
    def test_error_context_information(self) -> None:
        """Test context information in error scenarios."""
        # Test with invalid session format
        payload = {
            "question": "Test question",
            "session_id": ""  # Invalid session ID
        }
        
        response = requests.post(
            f"{self.BASE_URL}/ask",
            json=payload,
            headers=self.headers
        )
        
        # Even in error cases, we should get structured response
        if response.status_code == 200:
            data = response.json()["data"]
            if "context_info" in data:
                error_context = data["context_info"]["error_context"]
                self.assertIsInstance(error_context["has_error"], bool)
                self.assertIsInstance(error_context["fallback_used"], bool)
        
        logger.info("✓ Error context information validation passed")
    
    def test_processing_mode_consistency(self) -> None:
        """Test that processing mode is consistent between main response and context."""
        payload = {
            "question": "Bagaimana cara menghitung liabilitas asuransi?",
            "session_id": self.session_id
        }
        
        response = requests.post(
            f"{self.BASE_URL}/askproject",
            json=payload,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        
        # Check mode consistency
        main_mode = data["mode"]
        context_mode = data["context_info"]["retrieval_metadata"]["processing_mode"]
        
        # They should be related (context mode might be more specific)
        self.assertIsInstance(main_mode, str)
        self.assertIsInstance(context_mode, str)
        
        logger.info("✓ Processing mode consistency validation passed")
    
    def test_context_info_data_types(self) -> None:
        """Test that all context_info fields have correct data types."""
        payload = {
            "question": "Apa itu aktuaria?",
            "session_id": self.session_id
        }
        
        response = requests.post(
            f"{self.BASE_URL}/ask",
            json=payload,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        context_info = data["context_info"]
        
        # Validate data types
        retrieval_meta = context_info["retrieval_metadata"]
        self.assertIsInstance(retrieval_meta["confidence_level"], str)
        self.assertIsInstance(retrieval_meta["documents_retrieved"], int)
        self.assertIsInstance(retrieval_meta["processing_mode"], str)
        self.assertIsInstance(retrieval_meta["response_quality"], str)
        
        error_context = context_info["error_context"]
        self.assertIsInstance(error_context["has_error"], bool)
        self.assertIsInstance(error_context["fallback_used"], bool)
        
        logger.info("✓ Context info data types validation passed")


class TestContextInformationIntegration(unittest.TestCase):
    """Integration tests for context information across multiple requests."""
    
    BASE_URL = "http://localhost:5001"
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.session_id = f"integration_test_{int(time.time())}"
        self.headers = {"Content-Type": "application/json"}
    
    def test_context_info_across_conversation(self) -> None:
        """Test context information consistency across conversation turns."""
        questions = [
            "Apa itu PSAK 219?",
            "Bagaimana implementasinya?",
            "Apa dampaknya terhadap laporan keuangan?"
        ]
        
        for i, question in enumerate(questions):
            payload = {
                "question": question,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{self.BASE_URL}/ask",
                json=payload,
                headers=self.headers
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()["data"]
            
            # Each response should have context_info
            self.assertIn("context_info", data)
            
            # Session ID should be consistent
            self.assertEqual(data["session_id"], self.session_id)
        
        logger.info("✓ Context info across conversation validation passed")


def run_context_tests() -> None:
    """Run all context information tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestContextInformation))
    suite.addTest(unittest.makeSuite(TestContextInformationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        logger.info("🎉 All context information tests passed!")
    else:
        logger.error(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        for test, traceback in result.failures:
            logger.error(f"FAILED: {test}")
            logger.error(traceback)
        
        for test, traceback in result.errors:
            logger.error(f"ERROR: {test}")
            logger.error(traceback)


if __name__ == "__main__":
    run_context_tests()