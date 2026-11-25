"""
Comprehensive Evaluation Script for LearnMate AI RAG Pipeline.
Evaluates end-to-end RAG pipeline across 32+ labeled evaluation questions.
Computes:
- Answerability classification accuracy (True Positive, True Negative)
- False rejection rate on answerable queries
- Unsupported answer rate on unanswerable queries
- Citation correctness & groundedness metrics
- Language matching fidelity (English, Hindi, Hinglish)
- End-to-end latency breakdown (retrieval vs generation)
- Token consumption and estimated API pricing
Saves output to data/vector_store/rag_pipeline_evaluation_report.json.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

import config
from rag_pipeline import RAGPipeline
from llm import LLMClient

# Pricing constants per 1M tokens (USD)
# Google Gemini Free Tier is ₹0 / $0
MODEL_PRICING = {
    "gemini-3.5-flash-lite": {"prompt": 0.0, "completion": 0.0},
    "gemini-3.8-flash": {"prompt": 0.0, "completion": 0.0},
    "gemini": {"prompt": 0.0, "completion": 0.0},
    "gpt-4o-mini": {"prompt": 0.150, "completion": 0.600},
    "gpt-5-mini": {"prompt": 0.250, "completion": 1.000},
    "gpt-4o": {"prompt": 2.500, "completion": 10.000},
    "default": {"prompt": 0.0, "completion": 0.0}
}


def evaluate_pipeline(live_mode: bool = False, max_eval_queries: Optional[int] = None) -> Dict[str, Any]:
    dataset_file = Path("tests/retrieval_evaluation_dataset.json")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_file}")

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if max_eval_queries:
        dataset = dataset[:max_eval_queries]

    print("=" * 80)
    print(f"EVALUATING RAG PIPELINE ({len(dataset)} Queries | Mode: {'LIVE API' if live_mode else 'MOCK MODE'})")
    print("=" * 80)

    # Initialize RAG Pipeline
    llm = LLMClient(mock_mode=(not live_mode))
    pipeline = RAGPipeline(llm_client=llm)

    # Metrics accumulators
    total_queries = len(dataset)
    answerable_count = 0
    unanswerable_count = 0

    true_positives = 0  # Expected Answerable, classified Answerable or Partially Answerable
    false_rejections = 0  # Expected Answerable, classified Not Answerable
    true_negatives = 0  # Expected Not Answerable, classified Not Answerable
    unsupported_answers = 0  # Expected Not Answerable, classified Answerable (hallucination hazard)

    correct_language_matches = 0
    citation_valid_count = 0

    retrieval_latencies = []
    llm_latencies = []
    total_latencies = []

    total_prompt_tokens = 0
    total_comp_tokens = 0

    eval_items = []

    for item in dataset:
        qid = item["question_id"]
        qtext = item["question"]
        expected_lang = item["language"]
        expected_ans = item["expected_answerability"]
        expected_lec = item.get("expected_lecture")
        relevant_cids = set(item.get("relevant_chunk_ids", []))

        t0 = time.time()
        res = pipeline.answer_doubt(
            question=qtext,
            lecture_id=expected_lec if expected_lec else None
        )
        wall_time = (time.time() - t0) * 1000

        pred_status = res["answerability_status"]
        pred_lang = res["response_language"]
        citations = res.get("citations", [])
        cids = res.get("supporting_chunk_ids", [])
        metrics = res.get("metrics", {})

        # Accumulate latencies
        retrieval_latencies.append(metrics.get("retrieval_latency_ms", 0.0))
        llm_latencies.append(metrics.get("llm_latency_ms", 0.0))
        total_latencies.append(metrics.get("total_latency_ms", wall_time))

        # Accumulate token usage
        toks = metrics.get("token_usage", {})
        p_tok = toks.get("prompt_tokens", 0)
        c_tok = toks.get("completion_tokens", 0)
        total_prompt_tokens += p_tok
        total_comp_tokens += c_tok

        # Answerability classification evaluation
        if expected_ans == "ANSWERABLE":
            answerable_count += 1
            if pred_status in ("ANSWERABLE", "PARTIALLY_ANSWERABLE"):
                true_positives += 1
            else:
                false_rejections += 1
        else:
            unanswerable_count += 1
            if pred_status == "NOT_ANSWERABLE":
                true_negatives += 1
            else:
                unsupported_answers += 1

        # Language match check
        lang_match = (pred_lang.lower() == expected_lang.lower())
        if lang_match:
            correct_language_matches += 1

        # Citation validity: All returned chunk IDs must exist in retrieved chunks
        retrieved_ids = {c["chunk_id"] for c in res.get("retrieved_chunks", [])}
        all_cids_valid = all(cid in retrieved_ids for cid in cids)
        if pred_status == "ANSWERABLE" and all_cids_valid and len(cids) > 0:
            citation_valid_count += 1
        elif pred_status == "NOT_ANSWERABLE" and len(cids) == 0:
            citation_valid_count += 1

        eval_items.append({
            "question_id": qid,
            "question": qtext,
            "expected_language": expected_lang,
            "predicted_language": pred_lang,
            "language_matched": lang_match,
            "expected_answerability": expected_ans,
            "predicted_answerability": pred_status,
            "is_correctly_classified": (expected_ans == "ANSWERABLE" and pred_status in ("ANSWERABLE", "PARTIALLY_ANSWERABLE")) or (expected_ans == "NOT_ANSWERABLE" and pred_status == "NOT_ANSWERABLE"),
            "citations_count": len(citations),
            "citations_valid": all_cids_valid,
            "retrieval_latency_ms": metrics.get("retrieval_latency_ms", 0.0),
            "llm_latency_ms": metrics.get("llm_latency_ms", 0.0),
            "total_latency_ms": round(wall_time, 2),
            "answer_snippet": res["answer"][:100].replace("\n", " ") + "..."
        })

    # Pricing calculation
    model_name = config.OPENAI_MODEL
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
    estimated_cost_usd = (total_prompt_tokens / 1_000_000 * pricing["prompt"]) + \
                         (total_comp_tokens / 1_000_000 * pricing["completion"])

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "evaluation_mode": "LIVE_API" if live_mode else "MOCK_OFFLINE",
        "model_name": model_name,
        "sample_size": {
            "total_queries": total_queries,
            "answerable_queries": answerable_count,
            "unanswerable_queries": unanswerable_count
        },
        "answerability_metrics": {
            "true_positive_rate_pct": round(true_positives / answerable_count * 100, 2) if answerable_count else 0.0,
            "true_negative_rate_pct": round(true_negatives / unanswerable_count * 100, 2) if unanswerable_count else 0.0,
            "overall_accuracy_pct": round((true_positives + true_negatives) / total_queries * 100, 2),
            "false_rejection_rate_pct": round(false_rejections / answerable_count * 100, 2) if answerable_count else 0.0,
            "unsupported_answer_rate_pct": round(unsupported_answers / unanswerable_count * 100, 2) if unanswerable_count else 0.0
        },
        "grounding_and_language_metrics": {
            "language_matching_rate_pct": round(correct_language_matches / total_queries * 100, 2),
            "citation_validity_rate_pct": round(citation_valid_count / total_queries * 100, 2)
        },
        "latency_metrics_ms": {
            "average_retrieval_latency": round(float(np.mean(retrieval_latencies)), 2),
            "average_llm_latency": round(float(np.mean(llm_latencies)), 2),
            "average_total_latency": round(float(np.mean(total_latencies)), 2),
            "median_total_latency": round(float(np.median(total_latencies)), 2),
            "p95_total_latency": round(float(np.percentile(total_latencies, 95)), 2)
        },
        "token_usage_and_cost": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_comp_tokens,
            "total_tokens": total_prompt_tokens + total_comp_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
            "pricing_rates_per_million": pricing
        },
        "query_evaluations": eval_items
    }

    out_file = config.DATA_DIR / "vector_store" / "rag_pipeline_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RAG PIPELINE EVALUATION SUMMARY")
    print("=" * 80)
    ans_m = report["answerability_metrics"]
    lat_m = report["latency_metrics_ms"]
    tok_m = report["token_usage_and_cost"]
    print(f"Total Evaluated Questions:        {total_queries}")
    print(f"Answerability Overall Accuracy:   {ans_m['overall_accuracy_pct']}%")
    print(f"True Positive Rate (Answerable):  {ans_m['true_positive_rate_pct']}% ({true_positives}/{answerable_count})")
    print(f"True Negative Rate (Refusals):    {ans_m['true_negative_rate_pct']}% ({true_negatives}/{unanswerable_count})")
    print(f"False Rejection Rate:             {ans_m['false_rejection_rate_pct']}% ({false_rejections}/{answerable_count})")
    print(f"Unsupported Answer Rate:          {ans_m['unsupported_answer_rate_pct']}% ({unsupported_answers}/{unanswerable_count})")
    print(f"Language Matching Rate:           {report['grounding_and_language_metrics']['language_matching_rate_pct']}%")
    print(f"Citation Validity Rate:           {report['grounding_and_language_metrics']['citation_validity_rate_pct']}%")
    print(f"Median Total Latency:             {lat_m['median_total_latency']} ms (Avg: {lat_m['average_total_latency']} ms, P95: {lat_m['p95_total_latency']} ms)")
    print(f"Total Token Usage:                {tok_m['total_tokens']} tokens")
    print(f"Estimated Cost for this Run:      ${tok_m['estimated_cost_usd']:.6f} USD (Model: {model_name})")
    print("=" * 80)
    print(f"Saved evaluation report to: {out_file}\n")

    return report


if __name__ == "__main__":
    is_live = "--live" in sys.argv
    evaluate_pipeline(live_mode=is_live)

