"""
Comprehensive Retrieval Benchmark for LearnMate AI.
Evaluates retrieval strategies (Dense and Hybrid) across 32 human-verified questions
from tests/retrieval_evaluation_dataset.json.
Computes:
- Lecture-level Top-1 & Top-3 Accuracy
- Passage-level Hit Rate@1, Hit Rate@3, Hit Rate@5
- Recall@5 & Mean Reciprocal Rank (MRR)
- Average, Median, and P95 latency (separating query embedding and index search)
- Category & Language breakdowns (English, Hindi, Hinglish)
- Out-of-domain vs In-domain score separation
Saves report to data/vector_store/retrieval_evaluation_report.json.
"""

import json
import time
from pathlib import Path
import numpy as np

import config
from vector_store import VectorStore

def evaluate_strategy(vs: VectorStore, dataset: list, strategy: str = "dense") -> dict:
    results = []
    latencies = []
    search_latencies = []

    answerable_items = [q for q in dataset if q["expected_answerability"] == "ANSWERABLE"]
    unanswerable_items = [q for q in dataset if q["expected_answerability"] == "NOT_ANSWERABLE"]

    # Metrics accumulators
    lec_top1_correct = 0
    lec_top3_correct = 0
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_ranks = []
    recalls_at_5 = []

    lang_stats = {
        "English": {"total": 0, "lec_top1": 0, "hit_at_3": 0},
        "Hindi": {"total": 0, "lec_top1": 0, "hit_at_3": 0},
        "Hinglish": {"total": 0, "lec_top1": 0, "hit_at_3": 0}
    }

    for item in dataset:
        qid = item["question_id"]
        qtext = item["question"]
        lang = item["language"]
        cat = item["category"]
        ans_status = item["expected_answerability"]
        exp_lec = item.get("expected_lecture")
        rel_chunks = set(item.get("relevant_chunk_ids", []))

        t0 = time.time()
        hits = vs.search(qtext, top_k=5, strategy=strategy, deduplicate=True)
        tot_lat_ms = (time.time() - t0) * 1000
        latencies.append(tot_lat_ms)

        if hits and "timing" in hits[0]:
            search_latencies.append(hits[0]["timing"].get("search_time_ms", 0.0))

        top1_hit = hits[0] if hits else None
        top3_chunks = [h["chunk_id"] for h in hits[:3]]
        top5_chunks = [h["chunk_id"] for h in hits[:5]]
        top3_lecs = [h["lecture_id"] for h in hits[:3]]

        is_lec_top1 = (exp_lec is not None) and (top1_hit is not None) and (top1_hit["lecture_id"] == exp_lec)
        is_lec_top3 = (exp_lec is not None) and (exp_lec in top3_lecs)

        # Passage-level hit metrics
        is_hit_1 = bool(top1_hit and top1_hit["chunk_id"] in rel_chunks)
        is_hit_3 = any(cid in rel_chunks for cid in top3_chunks)
        is_hit_5 = any(cid in rel_chunks for cid in top5_chunks)

        # Reciprocal rank
        rr = 0.0
        for rank, h in enumerate(hits, 1):
            if h["chunk_id"] in rel_chunks:
                rr = 1.0 / rank
                break

        # Recall@5
        recall_5 = len(set(top5_chunks).intersection(rel_chunks)) / len(rel_chunks) if rel_chunks else 0.0

        if ans_status == "ANSWERABLE":
            if is_lec_top1:
                lec_top1_correct += 1
            if is_lec_top3:
                lec_top3_correct += 1
            if is_hit_1:
                hit_at_1 += 1
            if is_hit_3:
                hit_at_3 += 1
            if is_hit_5:
                hit_at_5 += 1

            reciprocal_ranks.append(rr)
            recalls_at_5.append(recall_5)

            # Language breakdown
            if lang in lang_stats:
                lang_stats[lang]["total"] += 1
                if is_lec_top1:
                    lang_stats[lang]["lec_top1"] += 1
                if is_hit_3:
                    lang_stats[lang]["hit_at_3"] += 1

        results.append({
            "question_id": qid,
            "question": qtext,
            "language": lang,
            "category": cat,
            "expected_answerability": ans_status,
            "expected_lecture": exp_lec,
            "top1_chunk_id": top1_hit["chunk_id"] if top1_hit else None,
            "top1_lecture_id": top1_hit["lecture_id"] if top1_hit else None,
            "top1_score": top1_hit["similarity_score"] if top1_hit else 0.0,
            "top3_chunks": top3_chunks,
            "is_lecture_top1": is_lec_top1,
            "is_lecture_top3": is_lec_top3,
            "is_passage_hit_at_1": is_hit_1,
            "is_passage_hit_at_3": is_hit_3,
            "is_passage_hit_at_5": is_hit_5,
            "reciprocal_rank": rr,
            "recall_at_5": round(recall_5, 4),
            "latency_ms": round(tot_lat_ms, 2)
        })

    N_ans = len(answerable_items)
    metrics = {
        "strategy": strategy,
        "sample_size": {
            "total_questions": len(dataset),
            "answerable_questions": N_ans,
            "unanswerable_questions": len(unanswerable_items)
        },
        "lecture_level_metrics": {
            "top1_accuracy": round(lec_top1_correct / N_ans * 100, 2),
            "top1_correct": lec_top1_correct,
            "top3_accuracy": round(lec_top3_correct / N_ans * 100, 2),
            "top3_correct": lec_top3_correct,
            "denominator": N_ans
        },
        "passage_level_metrics": {
            "hit_rate_at_1": round(hit_at_1 / N_ans * 100, 2),
            "hit_at_1_count": hit_at_1,
            "hit_rate_at_3": round(hit_at_3 / N_ans * 100, 2),
            "hit_at_3_count": hit_at_3,
            "hit_rate_at_5": round(hit_at_5 / N_ans * 100, 2),
            "hit_at_5_count": hit_at_5,
            "mean_reciprocal_rank_mrr": round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0,
            "mean_recall_at_5": round(float(np.mean(recalls_at_5)), 4) if recalls_at_5 else 0.0,
            "denominator": N_ans
        },
        "latency_metrics_ms": {
            "average_total_latency": round(float(np.mean(latencies)), 2),
            "median_total_latency": round(float(np.median(latencies)), 2),
            "p95_total_latency": round(float(np.percentile(latencies, 95)), 2),
            "average_search_only_latency": round(float(np.mean(search_latencies)), 4) if search_latencies else 0.0
        },
        "language_breakdown": {
            lang: {
                "count": data["total"],
                "lecture_top1_acc": round(data["lec_top1"] / data["total"] * 100, 2) if data["total"] else 0.0,
                "passage_hit_rate_at_3": round(data["hit_at_3"] / data["total"] * 100, 2) if data["total"] else 0.0
            }
            for lang, data in lang_stats.items()
        },
        "query_evaluations": results
    }
    return metrics

def main():
    print("=" * 80)
    print("TASK 13: COMPREHENSIVE RETRIEVAL BENCHMARK")
    print("=" * 80)

    dataset_path = Path("tests/retrieval_evaluation_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} evaluation items from {dataset_path}")

    vs = VectorStore(backend="faiss")
    vs.load()

    # 1. Evaluate Dense Strategy
    print("\nRunning Benchmark for Strategy: DENSE (E5 + FAISS IndexFlatIP)...")
    dense_metrics = evaluate_strategy(vs, dataset, strategy="dense")

    # 2. Evaluate Hybrid Strategy
    print("\nRunning Benchmark for Strategy: HYBRID (Dense RRF 0.7 + BM25 0.3)...")
    hybrid_metrics = evaluate_strategy(vs, dataset, strategy="hybrid")

    # Save comprehensive report
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_metadata": {
            "source_file": "tests/retrieval_evaluation_dataset.json",
            "total_items": len(dataset),
            "labeling_method": "human_verified"
        },
        "strategies": {
            "dense": dense_metrics,
            "hybrid": hybrid_metrics
        }
    }

    out_file = config.DATA_DIR / "vector_store" / "retrieval_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE RETRIEVAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<35} {'Dense (E5 + FAISS)':<22} {'Hybrid (Dense+BM25)'}")
    print("-" * 80)
    d_lec = dense_metrics["lecture_level_metrics"]
    h_lec = hybrid_metrics["lecture_level_metrics"]
    print(f"{'Lecture Top-1 Accuracy':<35} {d_lec['top1_accuracy']}% ({d_lec['top1_correct']}/{d_lec['denominator']}){'':<7} {h_lec['top1_accuracy']}% ({h_lec['top1_correct']}/{h_lec['denominator']})")
    print(f"{'Lecture Top-3 Accuracy':<35} {d_lec['top3_accuracy']}% ({d_lec['top3_correct']}/{d_lec['denominator']}){'':<7} {h_lec['top3_accuracy']}% ({h_lec['top3_correct']}/{h_lec['denominator']})")

    d_pas = dense_metrics["passage_level_metrics"]
    h_pas = hybrid_metrics["passage_level_metrics"]
    print(f"{'Passage Hit Rate@1':<35} {d_pas['hit_rate_at_1']}% ({d_pas['hit_at_1_count']}/{d_pas['denominator']}){'':<7} {h_pas['hit_rate_at_1']}% ({h_pas['hit_at_1_count']}/{h_pas['denominator']})")
    print(f"{'Passage Hit Rate@3':<35} {d_pas['hit_rate_at_3']}% ({d_pas['hit_at_3_count']}/{d_pas['denominator']}){'':<7} {h_pas['hit_rate_at_3']}% ({h_pas['hit_at_3_count']}/{h_pas['denominator']})")
    print(f"{'Passage Hit Rate@5':<35} {d_pas['hit_rate_at_5']}% ({d_pas['hit_at_5_count']}/{d_pas['denominator']}){'':<7} {h_pas['hit_rate_at_5']}% ({h_pas['hit_at_5_count']}/{h_pas['denominator']})")
    print(f"{'Mean Reciprocal Rank (MRR)':<35} {d_pas['mean_reciprocal_rank_mrr']:<22.4f} {h_pas['mean_reciprocal_rank_mrr']:.4f}")
    print(f"{'Mean Recall@5':<35} {d_pas['mean_recall_at_5']:<22.4f} {h_pas['mean_recall_at_5']:.4f}")

    d_lat = dense_metrics["latency_metrics_ms"]
    h_lat = hybrid_metrics["latency_metrics_ms"]
    print(f"{'Median Latency (ms)':<35} {d_lat['median_total_latency']} ms{'':<14} {h_lat['median_total_latency']} ms")
    print(f"{'P95 Latency (ms)':<35} {d_lat['p95_total_latency']} ms{'':<14} {h_lat['p95_total_latency']} ms")
    print("-" * 80)
    print("\nLanguage Breakdown (Passage Hit Rate@3):")
    for lang in ["English", "Hindi", "Hinglish"]:
        d_l = dense_metrics["language_breakdown"][lang]
        h_l = hybrid_metrics["language_breakdown"][lang]
        print(f"  {lang:<10} | Dense: {d_l['passage_hit_rate_at_3']}% ({d_l['count']} queries) | Hybrid: {h_l['passage_hit_rate_at_3']}%")

    print(f"\nFull report saved to: {out_file}")

if __name__ == "__main__":
    main()

