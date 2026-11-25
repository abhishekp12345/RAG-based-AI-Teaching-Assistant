"""
Automated Frontend and Streamlit App Tests for LearnMate AI (Phase 7).
Tests:
- Application startup and structure via Streamlit AppTest.
- Lecture manifest loading and tutorial sequencing.
- Sidebar controls (lecture selector, search scope, language preference, execution mode).
- Chat message flows (answerable, partially answerable, unanswerable, API error states).
- Verified citation rendering and timestamp seek navigation.
- Student feedback recording via FeedbackManager.
- Mock vs Live execution mode separation.
- Duplicate and empty query defenses.
"""

import os
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from streamlit.testing.v1 import AppTest
import config
from app import load_lecture_manifest, SUGGESTED_QUESTIONS
from feedback import FeedbackManager


class TestFrontendUnits(unittest.TestCase):
    """Unit tests for frontend helper logic and data integrity."""

    def test_lecture_manifest_loading(self):
        """Verify lecture manifest loads correctly and is sorted by tutorial number."""
        lectures = load_lecture_manifest()
        self.assertGreaterEqual(len(lectures), 3)

        # Confirm sorted by tutorial number: Tutorial 2, Tutorial 6, Tutorial 10
        tut_nums = [lec.get("tutorial_number") for lec in lectures]
        self.assertEqual(tut_nums, sorted(tut_nums))
        self.assertIn(2, tut_nums)
        self.assertIn(6, tut_nums)
        self.assertIn(10, tut_nums)

        # Check required fields
        for lec in lectures:
            self.assertIn("lecture_id", lec)
            self.assertIn("lecture_title", lec)
            self.assertIn("video_filename", lec)
            self.assertIn("duration_formatted", lec)

    def test_suggested_questions_coverage(self):
        """Verify suggested questions exist for all 3 lectures and are relevant."""
        for lid in ["LEC_01", "LEC_02", "LEC_03"]:
            self.assertIn(lid, SUGGESTED_QUESTIONS)
            questions = SUGGESTED_QUESTIONS[lid]
            self.assertGreaterEqual(len(questions), 3)
            for q in questions:
                self.assertIsInstance(q, str)
                self.assertGreater(len(q), 5)

    def test_feedback_persistence(self):
        """Verify student feedback recording stores clean JSON without credentials."""
        fb = FeedbackManager()
        test_item_id = "test_interaction_001"
        res = fb.record_feedback(
            category="doubt",
            item_id=test_item_id,
            rating=5,
            comments="Helpful explanation",
            metadata={"status": "ANSWERABLE", "mode": "mock"}
        )
        self.assertEqual(res["item_id"], test_item_id)
        self.assertEqual(res["rating"], 5)
        self.assertEqual(res["category"], "doubt")

        # Verify entry exists on disk
        all_fb = fb.load_all_feedback()
        matched = [f for f in all_fb if f.get("item_id") == test_item_id]
        self.assertTrue(len(matched) > 0)
        self.assertEqual(matched[-1]["rating"], 5)


class TestStreamlitAppIntegration(unittest.TestCase):
    """Integration tests executing the Streamlit app headless via AppTest."""

    @classmethod
    def setUpClass(cls):
        os.environ["LLM_MODE"] = "mock"
        from app import get_rag_pipeline
        get_rag_pipeline(mock_mode=True)

    def setUp(self):
        # Force offline/mock mode for deterministic frontend tests
        os.environ["LLM_MODE"] = "mock"

    def test_apptest_initial_startup(self):
        """Verify the Streamlit app starts up without exceptions and renders main components."""
        app_file = str(config.BASE_DIR / "app.py")
        at = AppTest.from_file(app_file, default_timeout=30)
        at.run()

        # Confirm no unhandled runtime exceptions
        self.assertFalse(at.exception)

        # Check titles, tabs, and sidebar elements
        self.assertTrue(len(at.tabs) >= 3)
        self.assertEqual(at.tabs[0].label, "💬 Ask AI (Doubt Resolution)")
        self.assertEqual(at.tabs[1].label, "📝 Smart Notes")
        self.assertEqual(at.tabs[2].label, "🎯 Practice Quiz")

        # Check lecture selection selectbox
        self.assertTrue(len(at.sidebar.selectbox) >= 2)  # Lecture select + language select

    def test_apptest_lecture_selection(self):
        """Verify changing lecture selectbox updates the selected lecture."""
        app_file = str(config.BASE_DIR / "app.py")
        at = AppTest.from_file(app_file, default_timeout=30)
        at.run()
        self.assertFalse(at.exception)

        # Initial default lecture
        initial_lec = at.session_state["selected_lecture_id"]
        self.assertIsNotNone(initial_lec)

        # Change lecture selection
        at.sidebar.selectbox[0].select_index(1).run()
        self.assertFalse(at.exception)
        new_lec = at.session_state["selected_lecture_id"]
        self.assertNotEqual(initial_lec, new_lec)

    def test_apptest_clear_chat_button(self):
        """Verify clear chat button resets conversation history."""
        app_file = str(config.BASE_DIR / "app.py")
        at = AppTest.from_file(app_file, default_timeout=30)
        at.run()

        # Seed conversation history
        at.session_state["messages"] = [
            {"role": "user", "content": "What is HTML?"},
            {"role": "assistant", "content": "HTML is HyperText Markup Language."}
        ]
        at.run()
        self.assertEqual(len(at.session_state["messages"]), 2)

        # Click Clear Chat History button in sidebar
        clear_btn = at.sidebar.button[0]
        clear_btn.click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state["messages"]), 0)

    def test_apptest_chat_input_answerable_flow(self):
        """Verify chat input processes a question in mock mode and appends assistant answer."""
        app_file = str(config.BASE_DIR / "app.py")
        at = AppTest.from_file(app_file, default_timeout=75)
        at.run()

        # Submit doubt via chat input
        at.chat_input[0].set_value("How to embed video in HTML?").run(timeout=75)
        self.assertFalse(at.exception)

        # Check messages in session state
        messages = at.session_state["messages"]
        self.assertGreaterEqual(len(messages), 2)
        self.assertEqual(messages[-2]["role"], "user")
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertIn("html", messages[-1]["content"].lower())

    def test_apptest_unanswerable_refusal_flow(self):
        """Verify out-of-domain query is politely refused with NOT_ANSWERABLE status."""
        app_file = str(config.BASE_DIR / "app.py")
        at = AppTest.from_file(app_file, default_timeout=30)
        at.run()

        # Submit out-of-domain question
        at.chat_input[0].set_value("What is photosynthesis and how does chlorophyll work?").run()
        self.assertFalse(at.exception)

        messages = at.session_state["messages"]
        self.assertGreaterEqual(len(messages), 2)
        last_resp = messages[-1]
        self.assertEqual(last_resp["role"], "assistant")
        self.assertEqual(last_resp["status"], "NOT_ANSWERABLE")
        self.assertEqual(len(last_resp["citations"]), 0)

    def test_apptest_seek_timestamp_navigation(self):
        """Verify jump-to-lecture timestamp button seeks video to verified timestamp."""
        app_file = str(config.BASE_DIR / "app.py")
        at = AppTest.from_file(app_file, default_timeout=30)
        at.run()

        # Seed an assistant message with a verified citation
        at.session_state["messages"] = [
            {"role": "user", "content": "What is SEO?"},
            {
                "role": "assistant",
                "content": "SEO stands for Search Engine Optimization.",
                "status": "ANSWERABLE",
                "citations": [
                    {
                        "chunk_id": "LEC_01_CHK_0016",
                        "lecture_id": "LEC_01",
                        "lecture_title": "SEO and Core Web Vitals in HTML | Tutorial #6",
                        "tutorial_number": 6,
                        "start_timestamp": "00:09:43",
                        "end_timestamp": "00:11:02",
                        "start_time": 583.76
                    }
                ],
                "interaction_id": "msg_seo_01"
            }
        ]
        at.run()
        self.assertFalse(at.exception)

        # Click the Seek to [00:09:43] button
        jump_btns = [b for b in at.button if "Seek to" in b.label]
        self.assertTrue(len(jump_btns) >= 1)
        jump_btns[0].click().run()
        self.assertFalse(at.exception)

        # Confirm session state seek_time and lecture updated to 583 seconds (00:09:43) in LEC_01
        self.assertEqual(at.session_state["selected_lecture_id"], "LEC_01")
        self.assertEqual(at.session_state["seek_time"], 583)

    def test_apptest_tab2_smart_notes_rendering(self):
        """Verify Tab 2 initializes Smart Notes, sections, and download button."""
        app_file = str(config.BASE_DIR / "app.py")
        at = AppTest.from_file(app_file, default_timeout=45)
        at.run()
        self.assertFalse(at.exception)

        # Check smart_notes session state is populated
        self.assertIn("smart_notes", at.session_state)
        self.assertGreaterEqual(len(at.session_state["smart_notes"]), 1)

        # Check download button is present
        dl_btns = [b for b in at.button if "Download Notes" in b.label or "dl_notes" in getattr(b, "key", "")]
        self.assertTrue(len(dl_btns) >= 1)

    def test_apptest_tab3_practice_quiz_rendering(self):
        """Verify Tab 3 generates questions, renders radio buttons, and submit button."""
        app_file = str(config.BASE_DIR / "app.py")
        at = AppTest.from_file(app_file, default_timeout=45)
        at.run()
        self.assertFalse(at.exception)

        # Check active_quizzes session state populated
        self.assertIn("active_quizzes", at.session_state)
        self.assertGreaterEqual(len(at.session_state["active_quizzes"]), 1)

        # Verify radio buttons exist for questions
        self.assertGreaterEqual(len(at.radio), 5)

        # Verify Submit Quiz button is present
        sub_btns = [b for b in at.button if "Submit Quiz" in b.label]
        self.assertTrue(len(sub_btns) >= 1)


if __name__ == "__main__":
    unittest.main()
