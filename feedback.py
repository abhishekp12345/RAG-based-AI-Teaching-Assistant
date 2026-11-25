"""
Student Feedback Module for LearnMate AI.
Collects and persists user ratings and suggestions for answers, notes, and quizzes.
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, List
import config


class FeedbackManager:
    """Manages recording and retrieving student feedback."""

    def __init__(self, feedback_dir: Path = config.FEEDBACK_DIR):
        self.feedback_dir = feedback_dir
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback_log.json"

    def record_feedback(
        self,
        category: str,  # 'doubt', 'notes', 'quiz'
        item_id: str,
        rating: int,  # 1 to 5
        comments: str = "",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Record student feedback entry to JSON store."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "category": category,
            "item_id": item_id,
            "rating": rating,
            "comments": comments,
            "metadata": metadata or {}
        }

        records = self.load_all_feedback()
        records.append(entry)

        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        return entry

    def load_all_feedback(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded feedback."""
        if not self.feedback_file.exists():
            return []
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

