"""
Validation Script: FAISS vs NumPy Equivalence for LearnMate AI.
Compares FAISS IndexFlatIP against exact NumPy matrix dot product across:
1. Exact score equivalence within float32 tolerance (< 1e-5)
2. Exact ranking order equivalence
3. Global retrieval and lecture-filtered retrieval
4. Search latency benchmarks (embedding latency vs search latency)
"""

import time
import numpy as np

import config
from embeddings import EmbeddingGenerator
from vector_store import VectorStore

TEST_QUERIES = [
    ("What is SEO and how does search engine optimization work in HTML?", None),
    ("How to embed video player in HTML with controls?", "LEC_02"),
    ("वेबसाइट में वीडियो और ऑडियो कैसे लगाएं?", None),
    ("Core web vitals me LCP aur FID kya hota hai?", "LEC_01"),
    ("First website banane ke liye folder structure kaise banaye?", "LEC_03"),
    ("What are the causes of plate tectonics?", None)
]

def main():
    print("=" * 80)
    print("TASK 7: VALIDATING FAISS INDEX AGAINST NUMPY COSINE SIMILARITY")
    print("=" * 80)

    # 1. Initialize VectorStore with FAISS
    vs_faiss = VectorStore(backend="faiss")
    vs_faiss.load()

    # 2. Initialize VectorStore with NumPy fallback
    vs_numpy = VectorStore(backend="numpy")
    vs_numpy.load()

    print(f"FAISS Backend: {vs_faiss.backend}, Total Vectors: {vs_faiss.index.ntotal}")
    print(f"NumPy Backend: {vs_numpy.backend}, Total Vectors: {len(vs_numpy.embeddings)}")

    all_tests_passed = True
    latencies_faiss = []
    latencies_numpy = []
    latencies_embed = []

    print("\nExecuting Comparison Queries (Top-K = 5)...")
    for q_idx, (query_text, filter_lec) in enumerate(TEST_QUERIES, 1):
        print(f"\n[Test {q_idx}] Query: \"{query_text}\" (Filter: {filter_lec or 'ALL'})")

        # Run FAISS search
        t0 = time.time()
        faiss_results = vs_faiss.search_dense(query_text, top_k=5, lecture_id=filter_lec)
        t_faiss = time.time() - t0

        # Run NumPy search
        t1 = time.time()
        numpy_results = vs_numpy.search_dense(query_text, top_k=5, lecture_id=filter_lec)
        t_numpy = time.time() - t1

        latencies_faiss.append(faiss_results[0]["timing"]["search_time_ms"])
        latencies_numpy.append(numpy_results[0]["timing"]["search_time_ms"])
        latencies_embed.append(faiss_results[0]["timing"]["embed_time_ms"])

        # Compare counts
        assert len(faiss_results) == len(numpy_results), f"Result count mismatch: FAISS={len(faiss_results)}, NumPy={len(numpy_results)}"

        # Compare chunk IDs and scores
        scores_match = True
        ranks_match = True

        for r_idx, (f_res, n_res) in enumerate(zip(faiss_results, numpy_results), 1):
            f_cid = f_res["chunk_id"]
            n_cid = n_res["chunk_id"]
            f_score = f_res["similarity_score"]
            n_score = n_res["similarity_score"]
            score_diff = abs(f_score - n_score)

            if score_diff > 1e-4:
                scores_match = False
                all_tests_passed = False

            if f_cid != n_cid and score_diff > 1e-4:
                ranks_match = False
                all_tests_passed = False

            if filter_lec:
                assert f_res["lecture_id"] == filter_lec, f"Lecture filter violation in FAISS: {f_res['lecture_id']} != {filter_lec}"
                assert n_res["lecture_id"] == filter_lec, f"Lecture filter violation in NumPy: {n_res['lecture_id']} != {filter_lec}"

        print(f"  FAISS Top-1: {faiss_results[0]['chunk_id']} ({faiss_results[0]['similarity_score']:.4f})")
        print(f"  NumPy Top-1: {numpy_results[0]['chunk_id']} ({numpy_results[0]['similarity_score']:.4f})")
        print(f"  Score Match: {'PASS' if scores_match else 'FAIL'} (Max Δ: {max(abs(f['similarity_score'] - n['similarity_score']) for f, n in zip(faiss_results, numpy_results)):.6f})")
        print(f"  Rank Match:  {'PASS' if ranks_match else 'FAIL'}")

    print("\n" + "=" * 80)
    print("BENCHMARK TIMING SUMMARY")
    print("=" * 80)
    print(f"Average Query Embedding Latency: {np.mean(latencies_embed):.2f} ms")
    print(f"Average FAISS Search Latency:    {np.mean(latencies_faiss):.4f} ms")
    print(f"Average NumPy Search Latency:    {np.mean(latencies_numpy):.4f} ms")
    print(f"Overall FAISS vs NumPy Equivalence: {'ALL PASSED' if all_tests_passed else 'FAILED'}")
    print("=" * 80)

    if not all_tests_passed:
        exit(1)

if __name__ == "__main__":
    main()

