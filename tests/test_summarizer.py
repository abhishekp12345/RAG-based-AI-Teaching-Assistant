"""
Unit Tests for Smart Notes Summarizer (Phase 8 & 9).
Tests:
- LectureSummarizer initialization and directory resolution.
- Chunk loading and filtering per lecture (LEC_01, LEC_02, LEC_03).
- Structured notes generation across all 3 styles (quick_revision, detailed_study, key_concepts).
- Multi-language support (English, Hindi, Hinglish).
- Strict schema compliance and required fields.
- Verified chunk citations and timestamp integrity.
- Caching behavior, cache hits, and force_refresh invalidation.
- Markdown export formatting.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from summarizer import LectureSummarizer
from llm import LLMClient


class TestLectureSummarizer(unittest.TestCase):
    """Unit test suite for LectureSummarizer."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(tempfile.mkdtemp(prefix="test_notes_"))
        cls.llm = LLMClient(mock_mode=True)
        cls.summarizer = LectureSummarizer(
            llm_client=cls.llm,
            notes_dir=cls.test_dir,
            chunks_dir=config.CHUNKS_DIR
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Verify summarizer initializes directories properly."""
        self.assertTrue(self.test_dir.exists())
        self.assertTrue(self.summarizer.chunks_dir.exists())
        self.assertTrue(self.summarizer.llm.mock_mode)

    def test_load_lecture_chunks(self):
        """Verify loading chunks for all three curriculum lectures."""
        for lid in ["LEC_01", "LEC_02", "LEC_03"]:
            chunks = self.summarizer.load_lecture_chunks(lid)
            self.assertGreater(len(chunks), 0, f"No chunks loaded for {lid}")
            for c in chunks:
                self.assertEqual(c["lecture_id"], lid)
                self.assertIn("chunk_id", c)
                self.assertIn("start_timestamp", c)

    def test_generate_notes_detailed_study_schema(self):
        """Verify detailed study notes adhere strictly to required JSON schema."""
        notes = self.summarizer.generate_smart_notes(
            lecture_id="LEC_01",
            notes_style="detailed_study",
            language="English"
        )

        required_keys = [
            "lecture_id", "lecture_title", "notes_style", "language",
            "executive_summary", "learning_objectives", "sections",
            "definitions_and_terms", "common_pitfalls", "revision_checklist"
        ]
        for rk in required_keys:
            self.assertIn(rk, notes)

        self.assertEqual(notes["lecture_id"], "LEC_01")
        self.assertEqual(notes["notes_style"], "detailed_study")
        self.assertEqual(notes["language"], "English")
        self.assertGreaterEqual(len(notes["learning_objectives"]), 3)
        self.assertGreaterEqual(len(notes["sections"]), 3)
        self.assertGreaterEqual(len(notes["definitions_and_terms"]), 3)
        self.assertGreaterEqual(len(notes["common_pitfalls"]), 3)
        self.assertGreaterEqual(len(notes["revision_checklist"]), 4)

    def test_notes_styles_variations(self):
        """Verify quick_revision, detailed_study, and key_concepts styles."""
        styles = ["quick_revision", "detailed_study", "key_concepts"]
        for st in styles:
            notes = self.summarizer.generate_smart_notes(
                lecture_id="LEC_02",
                notes_style=st,
                language="English"
            )
            self.assertEqual(notes["notes_style"], st)
            self.assertIsInstance(notes["executive_summary"], str)
            self.assertGreater(len(notes["executive_summary"]), 50)

    def test_notes_languages_variations(self):
        """Verify English, Hindi, and Hinglish language generations."""
        langs = ["English", "Hindi", "Hinglish"]
        for lg in langs:
            notes = self.summarizer.generate_smart_notes(
                lecture_id="LEC_03",
                notes_style="detailed_study",
                language=lg
            )
            self.assertEqual(notes["language"], lg)
            self.assertIsInstance(notes["executive_summary"], str)

    def test_citation_integrity(self):
        """Verify supporting chunk IDs exist and have valid timestamps."""
        notes = self.summarizer.generate_smart_notes(
            lecture_id="LEC_02",
            notes_style="detailed_study",
            language="English"
        )
        for sec in notes["sections"]:
            cids = sec.get("supporting_chunk_ids", [])
            citations = sec.get("citations", [])
            self.assertGreater(len(cids), 0)
            self.assertGreater(len(citations), 0)
            for cite in citations:
                self.assertIn("chunk_id", cite)
                self.assertIn("timestamp", cite)
                self.assertTrue(cite["chunk_id"].startswith("LEC_02"))

    def test_caching_and_force_refresh(self):
        """Verify notes are saved to cache and loaded from cache unless force_refresh is True."""
        notes1 = self.summarizer.generate_smart_notes(
            lecture_id="LEC_01",
            notes_style="quick_revision",
            language="English",
            force_refresh=True
        )
        cached_files = list(self.test_dir.glob("LEC_01_quick_revision_english_*.json"))
        self.assertEqual(len(cached_files), 1)

        # Second call should load from cache
        notes2 = self.summarizer.generate_smart_notes(
            lecture_id="LEC_01",
            notes_style="quick_revision",
            language="English",
            force_refresh=False
        )
        self.assertEqual(notes1["executive_summary"], notes2["executive_summary"])

    def test_export_notes_to_markdown(self):
        """Verify markdown export generates rich, well-structured document."""
        notes = self.summarizer.generate_smart_notes(
            lecture_id="LEC_01",
            notes_style="detailed_study",
            language="English"
        )
        md = self.summarizer.export_notes_to_markdown(notes)

        self.assertIn("# 📖 SEO and Core Web Vitals", md)
        self.assertIn("## 📌 Executive Summary", md)
        self.assertIn("## 🎯 Learning Objectives", md)
        self.assertIn("## 📖 Core Topics & Lecture Breakdown", md)
        self.assertIn("## 📚 Key Definitions & Terminology", md)
        self.assertIn("## ⚠️ Common Pitfalls & Exam Traps", md)
        self.assertIn("## ✅ Quick Revision Checklist", md)
        self.assertIn("```html", md)


if __name__ == "__main__":
    unittest.main()

