"""
Evaluation Module for LearnMate AI.
Computes RAG evaluation metrics: Retrieval Relevance, Faithfulness, and Answer Quality.
"""

from typing import List, Dict, Any
import numpy as np


class RAGEvaluator:
    """Evaluates RAG retrieval precision and answer quality."""

    @staticmethod
    def compute_retrieval_metrics(
        retrieved_chunks: List[Dict[str, Any]],
        ground_truth_keywords: List[str]
    ) -> Dict[str, float]:
        """Compute keyword coverage and mean similarity score."""
        if not retrieved_chunks:
            return {"hit_rate": 0.0, "mean_similarity": 0.0}

        text_corpus = " ".join([c.get("text", "").lower() for c in retrieved_chunks])
        hits = sum(1 for kw in ground_truth_keywords if kw.lower() in text_corpus)
        hit_rate = hits / max(1, len(ground_truth_keywords))

        scores = [c.get("similarity_score", 0.0) for c in retrieved_chunks]
        mean_sim = float(np.mean(scores)) if scores else 0.0

        return {
            "hit_rate": round(hit_rate, 4),
            "mean_similarity": round(mean_sim, 4),
            "num_chunks_retrieved": len(retrieved_chunks)
        }

    @staticmethod
    def evaluate_quiz_performance(
        user_answers: Dict[int, int],
        quiz_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute quiz score, percentage, and detailed question breakdown."""
        total = len(quiz_data)
        if total == 0:
            return {"score": 0, "total": 0, "percentage": 0.0}

        correct = 0
        details = []

        for q in quiz_data:
            qid = q.get("id")
            selected = user_answers.get(qid)
            is_correct = selected == q.get("correct_index")
            if is_correct:
                correct += 1
            details.append({
                "id": qid,
                "selected": selected,
                "correct_index": q.get("correct_index"),
                "is_correct": is_correct,
                "explanation": q.get("explanation", "")
            })

        return {
            "score": correct,
            "total": total,
            "percentage": round((correct / total) * 100, 2),
            "details": details
        }

