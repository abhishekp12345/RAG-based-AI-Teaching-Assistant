"""
Unit Tests for LLMClient Module in LearnMate AI.
Uses Python standard library unittest.
Tests:
- Client configuration and lazy loading
- Safe API key presence validation without secret leakage
- Structured JSON response parsing and schema compliance
- Offline mock mode for English, Hindi, and Hinglish queries
- Handling of unanswerable topics in mock mode
- Defensive handling of malformed or invalid JSON
- Rejection of fabricated chunk IDs via validate_grounded_response
- API error and exception handling
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
from llm import LLMClient, LLMGenerationResult
from prompts import validate_grounded_response, format_context_for_llm


class TestLLMClientConfiguration(unittest.TestCase):
    """Test LLM client configuration, parameters, and credential validation for Gemini and OpenAI."""

    def test_client_defaults(self):
        """Verify client initializes with config defaults for active provider."""
        client = LLMClient(mock_mode=True)
        self.assertEqual(client.provider, config.LLM_PROVIDER)
        expected_model = config.GEMINI_MODEL if config.LLM_PROVIDER == "gemini" else config.OPENAI_MODEL
        self.assertEqual(client.model, expected_model)
        self.assertEqual(client.timeout, config.LLM_TIMEOUT)
        self.assertEqual(client.max_output_tokens, config.LLM_MAX_OUTPUT_TOKENS)
        self.assertTrue(client.mock_mode)

    def test_gemini_defaults_explicit(self):
        """Verify client initializes with explicit Gemini provider."""
        client = LLMClient(provider="gemini", mock_mode=True)
        self.assertEqual(client.provider, "gemini")
        self.assertEqual(client.model, config.GEMINI_MODEL)
        self.assertEqual(client.get_usage_summary()["estimated_cost_usd"], 0.0)
        self.assertEqual(client.get_usage_summary()["estimated_cost_inr"], 0.0)

    def test_openai_defaults_explicit(self):
        """Verify client initializes with explicit OpenAI provider for backward compatibility."""
        client = LLMClient(provider="openai", mock_mode=True)
        self.assertEqual(client.provider, "openai")
        self.assertEqual(client.model, config.OPENAI_MODEL)

    def test_api_key_validation_detects_placeholders(self):
        """Verify is_api_key_valid flags missing or placeholder keys."""
        client_empty = LLMClient(api_key="", mock_mode=True)
        self.assertFalse(client_empty.is_api_key_valid())

        client_placeholder = LLMClient(api_key="your_gemini_api_key_here", mock_mode=True)
        self.assertFalse(client_placeholder.is_api_key_valid())

        client_short = LLMClient(api_key="AQ12345", mock_mode=True)
        self.assertFalse(client_short.is_api_key_valid())

        client_valid = LLMClient(api_key="AQvalid53charactersmockgeminikey1234567890abcdef12345", mock_mode=True)
        self.assertTrue(client_valid.is_api_key_valid())

    def test_get_client_raises_on_invalid_key_outside_mock(self):
        """Verify get_client raises ValueError if API key is invalid in live mode."""
        client_gemini = LLMClient(provider="gemini", api_key="your_gemini_api_key_here", mock_mode=False)
        with self.assertRaises(ValueError) as ctx_gemini:
            client_gemini.get_client()
        self.assertIn("GEMINI_API_KEY", str(ctx_gemini.exception))

        client_openai = LLMClient(provider="openai", api_key="your_openai_api_key_here", mock_mode=False)
        with self.assertRaises(ValueError) as ctx_openai:
            client_openai.get_client()
        self.assertIn("OPENAI_API_KEY", str(ctx_openai.exception))


class TestLLMMockGeneration(unittest.TestCase):
    """Test offline mock generation across multilingual questions and edge cases."""

    @classmethod
    def setUpClass(cls):
        cls.client = LLMClient(mock_mode=True)
        cls.sample_chunks = [
            {
                "chunk_id": "LEC_02_CHK_0008",
                "lecture_id": "LEC_02",
                "lecture_title": "Video, Audio & Media in HTML",
                "start_timestamp": "00:05:28",
                "end_timestamp": "00:06:57",
                "text": "video controls attribute src is equal to video.mp4"
            }
        ]

    def test_mock_answerable_english(self):
        """Verify mock response for answerable English query."""
        res = self.client.generate_grounded_answer(
            question="How to embed video player in HTML with controls?",
            retrieved_chunks=self.sample_chunks
        )
        self.assertTrue(res.is_mock)
        self.assertIsNotNone(res.parsed_json)
        self.assertEqual(res.parsed_json["answerability_status"], "ANSWERABLE")
        self.assertEqual(res.parsed_json["response_language"], "English")
        self.assertIn("LEC_02_CHK_0008", res.parsed_json["supporting_chunk_ids"])
        self.assertGreater(len(res.parsed_json["answer"]), 10)

    def test_mock_answerable_hindi(self):
        """Verify mock response for answerable Hindi query in Devanagari."""
        res = self.client.generate_grounded_answer(
            question="एचटीएमएल में वीडियो कैसे जोड़ें?",
            retrieved_chunks=self.sample_chunks
        )
        self.assertEqual(res.parsed_json["response_language"], "Hindi")
        self.assertEqual(res.parsed_json["answerability_status"], "ANSWERABLE")
        # Check Devanagari characters in answer
        self.assertTrue(any('\u0900' <= c <= '\u097f' for c in res.parsed_json["answer"]))

    def test_mock_answerable_hinglish(self):
        """Verify mock response for answerable Hinglish query."""
        res = self.client.generate_grounded_answer(
            question="HTML me video play karne ke liye kaun sa tag use karte hai?",
            retrieved_chunks=self.sample_chunks
        )
        self.assertEqual(res.parsed_json["response_language"], "Hinglish")
        self.assertEqual(res.parsed_json["answerability_status"], "ANSWERABLE")

    def test_mock_unanswerable_query(self):
        """Verify mock response for out-of-domain unanswerable question."""
        res = self.client.generate_grounded_answer(
            question="What is photosynthesis and how does chlorophyll work?",
            retrieved_chunks=self.sample_chunks
        )
        self.assertEqual(res.parsed_json["answerability_status"], "NOT_ANSWERABLE")
        self.assertEqual(res.parsed_json["supporting_chunk_ids"], [])
        self.assertIn("HTML", res.parsed_json["answer"])
        self.assertIsNotNone(res.parsed_json["missing_information"])

    def test_mock_override_injection(self):
        """Verify mock_override injects custom payload for testing."""
        custom_payload = {
            "answerability_status": "PARTIALLY_ANSWERABLE",
            "answer": "This is a partially supported answer.",
            "supporting_chunk_ids": ["LEC_02_CHK_0008"],
            "citations": [],
            "missing_information": "Part 2 is not in lecture.",
            "response_language": "English"
        }
        res = self.client.generate_grounded_answer(
            question="Some question",
            retrieved_chunks=self.sample_chunks,
            mock_override=custom_payload
        )
        self.assertEqual(res.parsed_json["answerability_status"], "PARTIALLY_ANSWERABLE")
        self.assertEqual(res.parsed_json["supporting_chunk_ids"], ["LEC_02_CHK_0008"])


class TestCitationValidationDefense(unittest.TestCase):
    """Test citation validation defenses against hallucinated chunk IDs."""

    def test_filter_fabricated_chunk_ids(self):
        """Verify validate_grounded_response removes hallucinated chunk IDs."""
        valid_chunks = [{"chunk_id": "REAL_01", "start_timestamp": "00:01:00", "end_timestamp": "00:02:00"}]
        hallucinated_output = {
            "answerability_status": "ANSWERABLE",
            "answer": "Answer based on real and fake.",
            "supporting_chunk_ids": ["REAL_01", "FABRICATED_CHUNK_999"],
            "citations": [],
            "missing_information": None,
            "response_language": "English"
        }
        validated = validate_grounded_response(hallucinated_output, valid_chunks)
        self.assertIn("REAL_01", validated["supporting_chunk_ids"])
        self.assertNotIn("FABRICATED_CHUNK_999", validated["supporting_chunk_ids"])
        self.assertEqual(len(validated["verified_citations"]), 1)
        self.assertEqual(validated["verified_citations"][0]["chunk_id"], "REAL_01")

    def test_unanswerable_strips_all_citations(self):
        """Verify NOT_ANSWERABLE status forcibly clears all citations."""
        valid_chunks = [{"chunk_id": "REAL_01", "start_timestamp": "00:01:00", "end_timestamp": "00:02:00"}]
        erroneous_output = {
            "answerability_status": "NOT_ANSWERABLE",
            "answer": "Cannot answer this.",
            "supporting_chunk_ids": ["REAL_01"],
            "citations": [],
            "missing_information": "Not in lecture.",
            "response_language": "English"
        }
        validated = validate_grounded_response(erroneous_output, valid_chunks)
        self.assertEqual(validated["supporting_chunk_ids"], [])
        self.assertEqual(validated["verified_citations"], [])

    def test_attempt_citation_correction_mock(self):
        """Verify attempt_citation_correction in mock mode retrieves relevant chunk IDs."""
        client = LLMClient(mock_mode=True)
        chunks = [
            {"chunk_id": "CHK_01", "text": "video controls attribute src is equal to video.mp4"}
        ]
        cids = client.attempt_citation_correction(
            question="How to embed video?",
            answer="Use video tag with controls.",
            retrieved_chunks=chunks
        )
        self.assertIsInstance(cids, list)
        self.assertIn("CHK_01", cids)

    def test_live_mode_failure_does_not_silently_fallback_to_mock(self):
        """Verify that live mode failures return explicit error message and do NOT silently return mock answer."""
        client = LLMClient(provider="openai", api_key="sk-proj-invalidmockkey1234567890abcdef1234567890", mock_mode=False)
        # Mock the client's internal call to simulate an API error
        with patch.object(client, "get_client") as mock_get_client:
            mock_openai = MagicMock()
            mock_openai.chat.completions.create.side_effect = RuntimeError("OpenAI API rate limit exceeded")
            mock_get_client.return_value = mock_openai

            chunks = [{"chunk_id": "CHK_01", "text": "some text"}]
            res = client.generate_grounded_answer("What is HTML?", chunks)

            # Must NOT silently be mock
            self.assertFalse(res.is_mock)
            self.assertIsNotNone(res.error_message)
            self.assertIn("OpenAI API rate limit exceeded", res.error_message)
            self.assertEqual(res.parsed_json["answerability_status"], "NOT_ANSWERABLE")
            self.assertIn("API Error: OpenAI API rate limit exceeded", res.parsed_json["missing_information"])
            self.assertIn("OpenAI API rate limit exceeded", res.parsed_json["answer"])

    def test_gemini_live_rate_limit_error_handling(self):
        """Verify Gemini rate limit (429 / RESOURCE_EXHAUSTED) returns explicit refusal without mock fallback."""
        client = LLMClient(provider="gemini", api_key="AQvalidmockkey1234567890abcdef12345678901234567890123", mock_mode=False)
        with patch.object(client, "get_client") as mock_get_client:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED: Rate limit exceeded")
            mock_get_client.return_value = mock_genai_client

            chunks = [{"chunk_id": "CHK_01", "text": "some text"}]
            res = client.generate_grounded_answer("What is HTML?", chunks)

            self.assertFalse(res.is_mock)
            self.assertEqual(res.provider, "gemini")
            self.assertIsNotNone(res.error_message)
            self.assertIn("429", res.error_message)
            self.assertEqual(res.parsed_json["answerability_status"], "NOT_ANSWERABLE")
            self.assertIn("rate limit or quota exceeded", res.parsed_json["answer"].lower())

    def test_gemini_live_timeout_error_handling(self):
        """Verify Gemini request timeout returns explicit error message."""
        client = LLMClient(provider="gemini", api_key="AQvalidmockkey1234567890abcdef12345678901234567890123", mock_mode=False)
        with patch.object(client, "get_client") as mock_get_client:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.side_effect = TimeoutError("Request timed out after 30 seconds")
            mock_get_client.return_value = mock_genai_client

            chunks = [{"chunk_id": "CHK_01", "text": "some text"}]
            res = client.generate_grounded_answer("What is HTML?", chunks)

            self.assertFalse(res.is_mock)
            self.assertEqual(res.parsed_json["answerability_status"], "NOT_ANSWERABLE")
            self.assertIn("timed out", res.parsed_json["answer"].lower())

    def test_gemini_live_model_not_found_handling(self):
        """Verify Gemini 404 model not found error is explicitly reported."""
        client = LLMClient(provider="gemini", api_key="AQvalidmockkey1234567890abcdef12345678901234567890123", mock_mode=False)
        with patch.object(client, "get_client") as mock_get_client:
            mock_genai_client = MagicMock()
            mock_genai_client.models.generate_content.side_effect = RuntimeError("404 NOT_FOUND: Model is no longer available")
            mock_get_client.return_value = mock_genai_client

            chunks = [{"chunk_id": "CHK_01", "text": "some text"}]
            res = client.generate_grounded_answer("What is HTML?", chunks)

            self.assertFalse(res.is_mock)
            self.assertEqual(res.parsed_json["answerability_status"], "NOT_ANSWERABLE")
            self.assertIn("unavailable", res.parsed_json["answer"].lower())

    def test_gemini_mocked_success_generation(self):
        """Verify mocked live Gemini generate_content cleanly parses JSON schema and tracking metrics."""
        client = LLMClient(provider="gemini", api_key="AQvalidmockkey1234567890abcdef12345678901234567890123", mock_mode=False)
        with patch.object(client, "get_client") as mock_get_client:
            mock_genai_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "answerability_status": "ANSWERABLE",
                "answer": "HTML is HyperText Markup Language used for web development.",
                "supporting_chunk_ids": ["CHK_01"],
                "citations": [],
                "missing_information": None,
                "response_language": "English"
            })
            mock_usage = MagicMock()
            mock_usage.prompt_token_count = 50
            mock_usage.candidates_token_count = 25
            mock_usage.total_token_count = 75
            mock_response.usage_metadata = mock_usage
            mock_genai_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_genai_client

            chunks = [{"chunk_id": "CHK_01", "start_timestamp": "00:01:00", "end_timestamp": "00:02:00", "text": "HTML markup"}]
            res = client.generate_grounded_answer("What is HTML?", chunks)

            self.assertFalse(res.is_mock)
            self.assertEqual(res.provider, "gemini")
            self.assertEqual(res.parsed_json["answerability_status"], "ANSWERABLE")
            self.assertIn("CHK_01", res.parsed_json["supporting_chunk_ids"])
            self.assertEqual(res.prompt_tokens, 50)
            self.assertEqual(res.completion_tokens, 25)
            self.assertEqual(res.total_tokens, 75)
            self.assertGreater(client.request_count, 0)
            self.assertEqual(client.get_usage_summary()["estimated_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()

