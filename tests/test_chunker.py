"""
Comprehensive Unit Test Suite for Transcript Preprocessing and Chunking.
Uses standard library unittest for zero extra dependencies.
Tests edge cases including empty transcripts, oversized segments, multilingual content,
short transcripts, timestamp gaps, overlap verification, and idempotence.
"""

import sys
import json
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from chunker import TranscriptChunker, TextPreprocessor, MultilingualTokenizer, seconds_to_timestamp


class TestTextPreprocessor(unittest.TestCase):
    def test_clean_text_normalizes_spaces_and_invisible_chars(self):
        raw = "Hello   \t\n  world!\u200b\ufeff  This is   a test.  "
        cleaned = TextPreprocessor.clean_text(raw)
        self.assertEqual(cleaned, "Hello world! This is a test.")

    def test_clean_text_preserves_html_tags_and_code(self):
        html_code = 'In this video we use <audio controls autoplay> and <video width="640"> tags.'
        cleaned = TextPreprocessor.clean_text(html_code)
        self.assertIn('<audio controls autoplay>', cleaned)
        self.assertIn('<video width="640">', cleaned)

    def test_clean_text_preserves_hindi_and_hinglish(self):
        hindi_text = "आज हम लोग सीखेंगे HTML tags और web development."
        cleaned = TextPreprocessor.clean_text(hindi_text)
        self.assertEqual(cleaned, hindi_text)

    def test_preprocess_segments_filters_empty_and_invalid(self):
        tokenizer = MultilingualTokenizer()
        raw_segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Valid speech segment."},
            {"id": 1, "start": 2.0, "end": 3.0, "text": "   \n\t  "},  # Empty
            {"id": 2, "start": 5.0, "end": 4.0, "text": "Invalid negative duration"},  # End < start
            {"id": 3, "start": 6.0, "end": 8.0, "text": "Another valid segment."},
        ]
        preprocessed = TextPreprocessor.preprocess_segments(raw_segments, tokenizer)
        self.assertEqual(len(preprocessed), 2)
        self.assertEqual(preprocessed[0].id, 0)
        self.assertEqual(preprocessed[1].id, 3)


class TestTranscriptChunkerEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MultilingualTokenizer()
        self.chunker = TranscriptChunker(
            chunk_size_tokens=300,
            max_chunk_size_tokens=400,
            chunk_overlap_tokens=50,
            gap_threshold_seconds=15.0,
            tokenizer=self.tokenizer
        )

    def test_empty_transcript(self):
        chunks = self.chunker.chunk_lecture({"segments": []}, {"lecture_id": "LEC_TEST"})
        self.assertEqual(chunks, [])

    def test_short_transcript(self):
        short_data = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 4.0, "text": "Welcome to HTML tutorial."},
                {"id": 1, "start": 4.5, "end": 8.0, "text": "HTML stands for HyperText Markup Language."}
            ]
        }
        lec_info = {
            "lecture_id": "LEC_TEST",
            "lecture_title": "Test Lecture",
            "tutorial_number": 1,
            "video_filename": "test.webm",
            "transcript_path": "data/transcripts/test.json"
        }
        chunks = self.chunker.chunk_lecture(short_data, lec_info)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_time, 0.0)
        self.assertEqual(chunks[0].end_time, 8.0)
        self.assertEqual(chunks[0].source_segment_indices, [0, 1])

    def test_oversized_segment_is_split_safely(self):
        long_sentence = "This is an important HTML concept explaining elements attributes and structure. " * 35
        raw_data = {
            "segments": [
                {"id": 0, "start": 10.0, "end": 120.0, "text": long_sentence}
            ]
        }
        lec_info = {
            "lecture_id": "LEC_TEST",
            "lecture_title": "Oversized Test",
            "tutorial_number": 99,
            "video_filename": "oversized.webm",
            "transcript_path": "data/transcripts/oversized.json"
        }
        chunks = self.chunker.chunk_lecture(raw_data, lec_info)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.token_count, self.chunker.max_chunk_size_tokens)
            self.assertGreaterEqual(c.start_time, 10.0)
            self.assertLessEqual(c.end_time, 120.0)

    def test_timestamp_gap_prevents_overlap_across_silence(self):
        raw_data = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 15.0, "text": "Part one of the lecture before coding."},
                {"id": 1, "start": 15.0, "end": 30.0, "text": "Let us roll the intro and open VS code."},
                # 40-second gap (30.0 to 70.0s)
                {"id": 2, "start": 70.0, "end": 85.0, "text": "Now we are back after the intro on the screen."},
                {"id": 3, "start": 85.0, "end": 100.0, "text": "Here is the code in our index.html file."}
            ]
        }
        lec_info = {
            "lecture_id": "LEC_GAP",
            "lecture_title": "Gap Test",
            "tutorial_number": 5,
            "video_filename": "gap.webm",
            "transcript_path": "data/transcripts/gap.json"
        }
        chunks = self.chunker.chunk_lecture(raw_data, lec_info)
        for c in chunks:
            self.assertFalse(c.start_time < 30.0 and c.end_time > 70.0, "Chunk crossed 40s silence gap!")

    def test_idempotence_and_deterministic_ids(self):
        raw_data = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 5.0, "text": "First segment."},
                {"id": 1, "start": 5.0, "end": 10.0, "text": "Second segment."},
                {"id": 2, "start": 10.0, "end": 15.0, "text": "Third segment."}
            ]
        }
        lec_info = {
            "lecture_id": "LEC_IDEM",
            "lecture_title": "Idempotence Test",
            "tutorial_number": 3,
            "video_filename": "idem.webm",
            "transcript_path": "data/transcripts/idem.json"
        }
        run1 = self.chunker.chunk_lecture(raw_data, lec_info)
        run2 = self.chunker.chunk_lecture(raw_data, lec_info)

        self.assertEqual(len(run1), len(run2))
        for c1, c2 in zip(run1, run2):
            self.assertEqual(c1.chunk_id, c2.chunk_id)
            self.assertEqual(c1.text, c2.text)
            self.assertEqual(c1.start_time, c2.start_time)
            self.assertEqual(c1.end_time, c2.end_time)


class TestDatasetChunkIntegrity(unittest.TestCase):
    """Validate the actual generated dataset in data/chunks/."""

    def test_dataset_files_exist(self):
        expected_files = [
            config.CHUNKS_DIR / "LEC_01_chunks.json",
            config.CHUNKS_DIR / "LEC_02_chunks.json",
            config.CHUNKS_DIR / "LEC_03_chunks.json",
            config.CHUNKS_DIR / "all_chunks.json",
            config.CHUNKS_DIR / "chunk_manifest.json",
            config.CHUNKS_DIR / "chunk_statistics.json"
        ]
        for f in expected_files:
            self.assertTrue(f.exists(), f"Generated dataset file missing: {f.name}")

    def test_all_chunks_unique_ids_and_non_empty(self):
        all_chunks_path = config.CHUNKS_DIR / "all_chunks.json"
        self.assertTrue(all_chunks_path.exists(), "all_chunks.json does not exist")

        with open(all_chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        self.assertGreater(len(chunks), 0, "No chunks in all_chunks.json")
        chunk_ids = [c["chunk_id"] for c in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)), "Duplicate chunk IDs found!")

        for c in chunks:
            self.assertGreater(len(c["text"].strip()), 0, f"Empty text in chunk {c['chunk_id']}")
            self.assertLessEqual(c["start_time"], c["end_time"], f"Invalid timing in chunk {c['chunk_id']}")
            self.assertLessEqual(c["token_count"], config.MAX_CHUNK_SIZE_TOKENS, f"Max token exceeded in chunk {c['chunk_id']}")
            self.assertIn(c["lecture_id"], ["LEC_01", "LEC_02", "LEC_03"])
            self.assertIn(c["tutorial_number"], [6, 10, 2])
            self.assertIn(":", c["start_timestamp"])
            self.assertIn(":", c["end_timestamp"])

    def test_no_phase2a_transcript_content(self):
        all_chunks_path = config.CHUNKS_DIR / "all_chunks.json"
        with open(all_chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        lec01_chunks = [c for c in chunks if c["lecture_id"] == "LEC_01"]
        for c in lec01_chunks:
            self.assertNotIn("The order has been cancelled", c["text"])
            self.assertNotIn("into central Asia", c["text"])
            self.assertTrue(
                "_cleaned.json" in c["source_transcript_path"] or "_hi_en_corrected.json" in c["source_transcript_path"]
            )

    def test_hindi_english_language_support_in_chunks(self):
        """Verify that chunks contain valid Devanagari Hindi and English without Urdu/Korean artifacts."""
        all_chunks_path = config.CHUNKS_DIR / "all_chunks.json"
        with open(all_chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for c in chunks:
            # Must not contain foreign Korean, Chinese, Greek artifacts
            import re
            korean_chars = re.findall(r"[\uAC00-\uD7AF\u1100-\u11FF]", c["text"])
            self.assertEqual(len(korean_chars), 0, f"Found Korean in chunk {c['chunk_id']}")
            # Must not contain Urdu script in final clean chunks
            urdu_chars = re.findall(r"[\u0600-\u06FF]", c["text"])
            self.assertEqual(len(urdu_chars), 0, f"Found Urdu script in chunk {c['chunk_id']}")
            # Must contain either Latin or Devanagari
            has_valid = bool(re.search(r"[a-zA-Z\u0900-\u097F]", c["text"]))
            self.assertTrue(has_valid, f"Chunk {c['chunk_id']} has no valid Latin or Devanagari text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
