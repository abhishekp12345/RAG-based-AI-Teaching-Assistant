"""
Evaluation of Final Context Selection Configurations for LearnMate AI.
Evaluates candidate retrieval pool of Top-K = 10 with:
Configuration A: Max 4 context chunks, 1200 token budget
Configuration B: Max 6 context chunks, 1600 token budget
Configuration C: Max 8 context chunks, 2000 token budget
Measures:
- Passage Hit Rate after final context selection
- Relevant chunk recall after final context selection
- Context word & token count overhead
- Redundancy pruned
Outputs data/vector_store/final_context_selection_results.json.
"""

import json
from pathlib import Path
import numpy as np

import config
from vector_store import VectorStore

CONFIGS = [
    {"name": "Config_4_Chunks_1200T", "max_chunks": 4, "max_tokens": 1200},
    {"name": "Config_6_Chunks_1600T", "max_chunks": 6, "max_tokens": 1600},
    {"name": "Config_8_Chunks_2000T", "max_chunks": 8, "max_tokens": 2000}
]

def select_context(candidate_chunks, max_chunks, max_tokens):
    """Simulates pipeline context selection and budgeting."""
    selected = []
    accum_words = 0
    max_words = int(max_tokens / 1.35)

    for chunk in candidate_chunks[:max_chunks]:
        words = len(chunk.get("text", "").split())
        if accum_words + words <= max_words or not selected:
            selected.append(chunk)
            accum_words += words
        else:
            break
    return selected

def main():
    print("=" * 80)
    print("TASK 2: EVALUATING FINAL CONTEXT SELECTION CONFIGURATIONS")
    print("=" * 80)

    dataset_path = Path("tests/retrieval_evaluation_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    answerable = [q for q in dataset if q["expected_answerability"] == "ANSWERABLE"]
    print(f"Total Answerable Queries: {len(answerable)}")

    vs = VectorStore(backend="faiss")
    vs.load()

    results = {}

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        max_c = cfg["max_chunks"]
        max_t = cfg["max_tokens"]

        hits_count = 0
        recalls = []
        context_words_list = []
        selected_chunk_counts = []

        for q in answerable:
            qtext = q["question"]
            rel_cids = set(q.get("relevant_chunk_ids", []))

            # Retrieve candidate pool of K=10 with deduplication
            candidates = vs.search(qtext, top_k=10, strategy="dense", deduplicate=True)

            # Final context selection
            final_context = select_context(candidates, max_chunks=max_c, max_tokens=max_t)
            final_cids = set(c["chunk_id"] for c in final_context)

            # Check if any relevant chunk is in final context
            is_hit = bool(final_cids.intersection(rel_cids))
            if is_hit:
                hits_count += 1

            # Recall
            inter = len(final_cids.intersection(rel_cids))
            recalls.append(inter / len(rel_cids) if rel_cids else 0.0)

            tot_words = sum(len(c["text"].split()) for c in final_context)
            context_words_list.append(tot_words)
            selected_chunk_counts.append(len(final_context))

        N = len(answerable)
        hit_rate = round(hits_count / N * 100, 2)
        mean_recall = round(float(np.mean(recalls)), 4)
        mean_words = round(float(np.mean(context_words_list)), 1)
        mean_tokens = round(mean_words * 1.35, 1)
        mean_chunks = round(float(np.mean(selected_chunk_counts)), 2)

        results[cfg_name] = {
            "max_chunks": max_c,
            "max_tokens": max_t,
            "candidate_k": 10,
            "final_context_hit_rate_pct": hit_rate,
            "final_context_hit_count": hits_count,
            "total_queries": N,
            "final_context_mean_recall": mean_recall,
            "mean_selected_chunks": mean_chunks,
            "mean_context_words": mean_words,
            "mean_context_tokens_approx": mean_tokens
        }

        print(f"\n{cfg_name}:")
        print(f"  Final Context Passage Hit Rate: {hit_rate}% ({hits_count}/{N})")
        print(f"  Final Context Mean Recall:     {mean_recall:.4f}")
        print(f"  Mean Chunks Selected:          {mean_chunks} / {max_c}")
        print(f"  Mean Context Tokens (approx):  ~{mean_tokens} tokens")

    out_file = config.DATA_DIR / "vector_store" / "final_context_selection_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL CONTEXT SELECTION SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Hit Rate':<16} {'Mean Recall':<14} {'Mean Chunks':<14} {'Avg Tokens'}")
    print("-" * 80)
    for cname, res in results.items():
        print(f"{cname:<25} {res['final_context_hit_rate_pct']}% ({res['final_context_hit_count']}/{res['total_queries']}){'':<4} {res['final_context_mean_recall']:<14.4f} {res['mean_selected_chunks']:<14} ~{res['mean_context_tokens_approx']}")
    print("-" * 80)
    print(f"Saved results to: {out_file}")

if __name__ == "__main__":
    main()

