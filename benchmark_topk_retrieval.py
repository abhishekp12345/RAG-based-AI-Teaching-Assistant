"""
Benchmark Top-K Retrieval Configurations for LearnMate AI.
Evaluates K in {3, 5, 8, 10} across:
- Passage Hit Rate
- Recall on relevant chunks
- Average context tokens / words retrieved
- Redundancy rate before and after context deduplication
- Dev split vs Held-out Test split
Outputs data/vector_store/topk_benchmark_results.json.
"""

import json
import time
from pathlib import Path
import numpy as np

import config
from vector_store import VectorStore

K_CANDIDATES = [3, 5, 8, 10]

def run_benchmark():
    print("=" * 80)
    print("TASK 5: BENCHMARKING TOP-K RETRIEVAL CONFIGURATIONS")
    print("=" * 80)

    dataset_path = Path("tests/retrieval_evaluation_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    answerable = [q for q in dataset if q["expected_answerability"] == "ANSWERABLE"]
    dev_split = [q for q in answerable if q.get("is_dev_split", False)]
    test_split = [q for q in answerable if not q.get("is_dev_split", False)]

    print(f"Total Answerable Queries: {len(answerable)} (Dev: {len(dev_split)}, Test: {len(test_split)})")

    vs = VectorStore(backend="faiss")
    vs.load()

    benchmark_results = {}

    for k in K_CANDIDATES:
        print(f"\nEvaluating Top-K = {k}...")
        raw_hits_count = 0
        dedup_hits_count = 0
        recalls = []
        context_word_counts = []
        redundancy_dropped = 0

        for q in answerable:
            qtext = q["question"]
            rel_cids = set(q.get("relevant_chunk_ids", []))

            # Raw search without deduplication
            raw_results = vs.search(qtext, top_k=k, strategy="dense", deduplicate=False)
            raw_cids = [h["chunk_id"] for h in raw_results]

            # With context deduplication
            dedup_results = vs.search(qtext, top_k=k, strategy="dense", deduplicate=True)
            dedup_cids = [h["chunk_id"] for h in dedup_results]

            redundancy_dropped += (len(raw_results) - len(dedup_results))

            # Hit checks
            if any(cid in rel_cids for cid in raw_cids):
                raw_hits_count += 1
            if any(cid in rel_cids for cid in dedup_cids):
                dedup_hits_count += 1

            # Recall
            inter = len(set(dedup_cids).intersection(rel_cids))
            recalls.append(inter / len(rel_cids) if rel_cids else 0.0)

            # Context size
            total_words = sum(len(h["text"].split()) for h in dedup_results)
            context_word_counts.append(total_words)

        # Dev vs Test split metrics
        dev_hits = sum(1 for q in dev_split if any(cid in set(q["relevant_chunk_ids"]) for cid in [h["chunk_id"] for h in vs.search(q["question"], top_k=k, strategy="dense", deduplicate=True)]))
        test_hits = sum(1 for q in test_split if any(cid in set(q["relevant_chunk_ids"]) for cid in [h["chunk_id"] for h in vs.search(q["question"], top_k=k, strategy="dense", deduplicate=True)]))

        N = len(answerable)
        benchmark_results[f"top_{k}"] = {
            "k": k,
            "raw_hit_rate_pct": round(raw_hits_count / N * 100, 2),
            "dedup_hit_rate_pct": round(dedup_hits_count / N * 100, 2),
            "hit_count": dedup_hits_count,
            "total_answerable": N,
            "mean_recall": round(float(np.mean(recalls)), 4),
            "mean_context_words": round(float(np.mean(context_word_counts)), 1),
            "mean_context_tokens_approx": round(float(np.mean(context_word_counts)) * 1.35, 1),
            "total_redundant_chunks_pruned": redundancy_dropped,
            "dev_split_hit_rate_pct": round(dev_hits / len(dev_split) * 100, 2) if dev_split else 0.0,
            "test_split_hit_rate_pct": round(test_hits / len(test_split) * 100, 2) if test_split else 0.0
        }

    out_file = config.DATA_DIR / "vector_store" / "topk_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n" + "=" * 80)
    print("TOP-K BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Config':<10} {'Hit Rate':<14} {'Mean Recall':<14} {'Dev Hit%':<12} {'Test Hit%':<12} {'Avg Tokens':<12} {'Pruned Chunks'}")
    print("-" * 80)
    for k_key, res in benchmark_results.items():
        print(f"K = {res['k']:<6} {res['dedup_hit_rate_pct']}% ({res['hit_count']}/{res['total_answerable']}){'':<4} {res['mean_recall']:<14.4f} {res['dev_split_hit_rate_pct']}%{'':<5} {res['test_split_hit_rate_pct']}%{'':<5} ~{res['mean_context_tokens_approx']:<11} {res['total_redundant_chunks_pruned']}")
    print("-" * 80)
    print(f"Saved results to: {out_file}")

if __name__ == "__main__":
    run_benchmark()

