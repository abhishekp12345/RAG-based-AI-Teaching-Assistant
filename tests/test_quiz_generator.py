"""
Unit Tests for Practice Quiz Generator (Phase 8 & 9).
Tests:
- QuizGenerator initialization and directory resolution.
- Chunk loading and filtering per lecture.
- Quiz generation for 5 and 10 questions.
- Quiz generation across all 3 difficulty tiers (Easy, Medium, Hard).
- Multi-language support (English, Hindi, Hinglish).
- Question schema integrity: exactly 4 options, valid correct_index (0..3), non-empty explanations.
- Client-side evaluation: 100% perfect score, 0% score, partial score.
- Topic performance analytics and weak topic timestamp recommendations.
- Caching behavior and cache hit persistence.
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
from quiz_generator import QuizGenerator
from llm import LLMClient


class TestQuizGenerator(unittest.TestCase):
    """Unit test suite for QuizGenerator."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(tempfile.mkdtemp(prefix="test_quizzes_"))
        cls.llm = LLMClient(mock_mode=True)
        cls.quiz_gen = QuizGenerator(
            llm_client=cls.llm,
            quizzes_dir=cls.test_dir,
            chunks_dir=config.CHUNKS_DIR
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Verify quiz generator initializes properly."""
        self.assertTrue(self.test_dir.exists())
        self.assertTrue(self.quiz_gen.chunks_dir.exists())
        self.assertTrue(self.quiz_gen.llm.mock_mode)

    def test_generate_quiz_5_and_10_questions(self):
        """Verify generation of both 5 and 10 question quizzes."""
        for count in [5, 10]:
            quiz = self.quiz_gen.generate_quiz(
                lecture_id="LEC_01",
                difficulty="Medium",
                num_questions=count,
                language="English"
            )
            self.assertEqual(len(quiz["questions"]), count)
            self.assertEqual(quiz["num_questions"], count)

    def test_generate_quiz_all_difficulties(self):
        """Verify generation across Easy, Medium, and Hard tiers."""
        for diff in ["Easy", "Medium", "Hard"]:
            quiz = self.quiz_gen.generate_quiz(
                lecture_id="LEC_02",
                difficulty=diff,
                num_questions=5,
                language="English"
            )
            self.assertEqual(quiz["difficulty"], diff)
            self.assertEqual(len(quiz["questions"]), 5)

    def test_generate_quiz_all_languages(self):
        """Verify generation across English, Hindi, and Hinglish."""
        for lang in ["English", "Hindi", "Hinglish"]:
            quiz = self.quiz_gen.generate_quiz(
                lecture_id="LEC_03",
                difficulty="Medium",
                num_questions=5,
                language=lang
            )
            self.assertEqual(quiz["language"], lang)
            self.assertEqual(len(quiz["questions"]), 5)

    def test_question_schema_integrity(self):
        """Verify every question contains exactly 4 options and valid correct_index."""
        for lid in ["LEC_01", "LEC_02", "LEC_03"]:
            quiz = self.quiz_gen.generate_quiz(
                lecture_id=lid,
                difficulty="Medium",
                num_questions=10,
                language="English"
            )
            for idx, q in enumerate(quiz["questions"]):
                self.assertIn("id", q)
                self.assertEqual(q["id"], idx + 1)
                self.assertIn("question", q)
                self.assertGreater(len(q["question"]), 10)

                # Exactly 4 options
                opts = q.get("options", [])
                self.assertEqual(len(opts), 4, f"Question {q['id']} does not have exactly 4 options in {lid}")

                # Valid correct index in 0..3
                corr = q.get("correct_index")
                self.assertIn(corr, [0, 1, 2, 3], f"Invalid correct_index {corr} in {lid}")

                # Explanation, topic, and timestamp
                self.assertGreater(len(q.get("explanation", "")), 10)
                self.assertGreater(len(q.get("topic", "")), 2)
                self.assertIn("timestamp_ref", q)
                self.assertGreater(len(q.get("supporting_chunk_ids", [])), 0)

    def test_evaluate_quiz_perfect_score(self):
        """Verify 100% correct answers evaluate to perfect score and grade."""
        quiz = self.quiz_gen.generate_quiz(
            lecture_id="LEC_01",
            difficulty="Medium",
            num_questions=5,
            language="English"
        )
        user_answers = {q["id"]: q["correct_index"] for q in quiz["questions"]}
        eval_res = self.quiz_gen.evaluate_quiz_submission(quiz, user_answers)

        self.assertEqual(eval_res["score"], 5)
        self.assertEqual(eval_res["total"], 5)
        self.assertEqual(eval_res["percentage"], 100.0)
        self.assertTrue(eval_res["passed"])
        self.assertIn("Outstanding", eval_res["grade"])
        self.assertEqual(len(eval_res["weak_topics"]), 0)
        self.assertGreater(len(eval_res["strong_topics"]), 0)

    def test_evaluate_quiz_zero_score(self):
        """Verify 0% score flags weak topics and provides timestamp recommendations."""
        quiz = self.quiz_gen.generate_quiz(
            lecture_id="LEC_01",
            difficulty="Medium",
            num_questions=5,
            language="English"
        )
        # Choose wrong index for all
        user_answers = {q["id"]: (q["correct_index"] + 1) % 4 for q in quiz["questions"]}
        eval_res = self.quiz_gen.evaluate_quiz_submission(quiz, user_answers)

        self.assertEqual(eval_res["score"], 0)
        self.assertEqual(eval_res["percentage"], 0.0)
        self.assertFalse(eval_res["passed"])
        self.assertGreater(len(eval_res["weak_topics"]), 0)
        for wt in eval_res["weak_topics"]:
            self.assertIn("topic", wt)
            self.assertIn("recommended_timestamp", wt)
            self.assertIn("lecture_id", wt)

    def test_evaluate_quiz_partial_score(self):
        """Verify partial score calculation."""
        quiz = self.quiz_gen.generate_quiz(
            lecture_id="LEC_02",
            difficulty="Medium",
            num_questions=5,
            language="English"
        )
        # Answer first 3 correctly, last 2 wrongly
        user_answers = {}
        for i, q in enumerate(quiz["questions"]):
            if i < 3:
                user_answers[q["id"]] = q["correct_index"]
            else:
                user_answers[q["id"]] = (q["correct_index"] + 1) % 4

        eval_res = self.quiz_gen.evaluate_quiz_submission(quiz, user_answers)
        self.assertEqual(eval_res["score"], 3)
        self.assertEqual(eval_res["total"], 5)
        self.assertEqual(eval_res["percentage"], 60.0)
        self.assertTrue(eval_res["passed"])

    def test_quiz_caching(self):
        """Verify quiz persistence in caching directory."""
        quiz1 = self.quiz_gen.generate_quiz(
            lecture_id="LEC_03",
            difficulty="Hard",
            num_questions=5,
            language="English",
            force_refresh=True
        )
        cached_files = list(self.test_dir.glob("LEC_03_hard_5q_english_*.json"))
        self.assertEqual(len(cached_files), 1)

        quiz2 = self.quiz_gen.generate_quiz(
            lecture_id="LEC_03",
            difficulty="Hard",
            num_questions=5,
            language="English",
            force_refresh=False
        )
        self.assertEqual(quiz1["quiz_id"], quiz2["quiz_id"])


if __name__ == "__main__":
    unittest.main()

