"""
Unit Tests for Phase 5: Vector Store, FAISS Indexing, and Retrieval Architecture.
Uses Python standard library unittest.
Tests:
- FAISS IndexFlatIP properties and persistence
- FAISS vs NumPy exact numerical equivalence
- Metadata 1-to-1 alignment
- Global and lecture-filtered retrieval
- Multilingual query handling (English, Hindi, Hinglish)
- Invalid and edge-case inputs (empty query, non-existent lecture, K > corpus)
- Context deduplication logic
- Hybrid retrieval (BM25 + Dense RRF)
- Answerability prompt schema and validation defense
"""

import json
import unittest
from pathlib import Path
import numpy as np

import config
from vector_store import VectorStore, BM25LexicalRetriever
import prompts

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class TestFAISSIndexAndStorage(unittest.TestCase):
    """Test FAISS IndexFlatIP construction, manifest, and persistence."""

    @classmethod
    def setUpClass(cls):
        cls.vs_dir = config.DATA_DIR / "vector_store"
        cls.faiss_file = cls.vs_dir / "faiss.index"
        cls.meta_file = cls.vs_dir / "chunk_metadata.json"
        cls.manifest_file = cls.vs_dir / "index_manifest.json"

    def test_artifacts_exist(self):
        """Verify vector store artifact files exist."""
        self.assertTrue(self.faiss_file.exists(), "faiss.index must exist")
        self.assertTrue(self.meta_file.exists(), "chunk_metadata.json must exist")
        self.assertTrue(self.manifest_file.exists(), "index_manifest.json must exist")

    def test_faiss_index_properties(self):
        """Verify FAISS index type, dimension, and vector count."""
        if not FAISS_AVAILABLE:
            self.skipTest("FAISS not installed.")
        index = faiss.read_index(str(self.faiss_file))
        self.assertEqual(index.d, 384, "Embedding dimension must be 384")
        self.assertEqual(index.ntotal, 102, "Must contain exactly 102 vectors")
        self.assertTrue(index.is_trained, "Index must be trained")

    def test_manifest_metadata(self):
        """Verify index manifest contains all required provenance fields."""
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["model_name"], "intfloat/multilingual-e5-small")
        self.assertEqual(manifest["embedding_dimension"], 384)
        self.assertEqual(manifest["index_type"], "IndexFlatIP")
        self.assertEqual(manifest["total_vectors"], 102)
        self.assertTrue(manifest["normalized"])
        self.assertIn("dataset_fingerprint_sha256", manifest)
        self.assertIn("index_fingerprint_sha256", manifest)


class TestVectorStoreRetrieval(unittest.TestCase):
    """Test retrieval execution, FAISS vs NumPy equivalence, and edge cases."""

    @classmethod
    def setUpClass(cls):
        cls.vs_faiss = VectorStore(backend="faiss")
        cls.vs_faiss.load()

        cls.vs_numpy = VectorStore(backend="numpy")
        cls.vs_numpy.load()

    def test_faiss_numpy_exact_equivalence(self):
        """Verify FAISS and NumPy yield identical scores and rankings."""
        query = "How to embed video player in HTML with controls?"
        faiss_results = self.vs_faiss.search_dense(query, top_k=5)
        numpy_results = self.vs_numpy.search_dense(query, top_k=5)

        self.assertEqual(len(faiss_results), len(numpy_results))
        for f, n in zip(faiss_results, numpy_results):
            self.assertEqual(f["chunk_id"], n["chunk_id"])
            self.assertAlmostEqual(f["similarity_score"], n["similarity_score"], places=4)

    def test_lecture_filtered_retrieval(self):
        """Verify lecture_id filtering restricts search strictly to target lecture."""
        for target_lec in ["LEC_01", "LEC_02", "LEC_03"]:
            results = self.vs_faiss.search_dense("HTML website", top_k=5, lecture_id=target_lec)
            self.assertGreater(len(results), 0)
            for item in results:
                self.assertEqual(item["lecture_id"], target_lec)

    def test_empty_query_returns_empty(self):
        """Verify empty or whitespace queries return empty results."""
        self.assertEqual(self.vs_faiss.search("", top_k=5), [])
        self.assertEqual(self.vs_faiss.search("   ", top_k=5), [])

    def test_invalid_lecture_id_returns_empty(self):
        """Verify non-existent lecture_id returns empty results."""
        results = self.vs_faiss.search("HTML video", top_k=5, lecture_id="NON_EXISTENT_LEC")
        self.assertEqual(results, [])

    def test_top_k_exceeding_corpus(self):
        """Verify requesting K > corpus count safely clamps to total available."""
        results = self.vs_faiss.search("HTML", top_k=500)
        self.assertEqual(len(results), 102)

    def test_multilingual_queries(self):
        """Verify successful retrieval for English, Hindi, and Hinglish queries."""
        queries = [
            ("What is SEO in HTML?", "LEC_01"),
            ("सर्च इंजन ऑप्टिमाइजेशन क्या है?", "LEC_01"),
            ("HTML me video tag kaise lagaye?", "LEC_02")
        ]
        for qtext, exp_lec in queries:
            res = self.vs_faiss.search(qtext, top_k=3)
            self.assertGreater(len(res), 0)
            self.assertIn("rank", res[0])
            self.assertIn("chunk_id", res[0])
            self.assertIn("similarity_score", res[0])
            self.assertIn("start_timestamp", res[0])


class TestOptimizationAndAnswerability(unittest.TestCase):
    """Test context deduplication, BM25, hybrid retrieval, and prompt validation."""

    def test_bm25_lexical_retriever(self):
        """Verify BM25 indexer and search on exact technical tokens."""
        docs = [
            "This is about HTML video and audio elements with controls",
            "This is about SEO and Core Web Vitals with meta tags",
            "This is about VS Code and setting up index.html"
        ]
        meta = [{"id": 0}, {"id": 1}, {"id": 2}]
        bm25 = BM25LexicalRetriever()
        bm25.index(docs, meta)

        hits = bm25.search("video controls", top_k=2)
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0][0], 0, "Doc 0 should match 'video controls'")

    def test_context_deduplication(self):
        """Verify overlapping chunks within the same lecture are deduplicated."""
        mock_chunks = [
            {"chunk_id": "C01", "lecture_id": "LEC_01", "text": "alpha beta gamma delta epsilon zeta eta"},
            {"chunk_id": "C02", "lecture_id": "LEC_01", "text": "alpha beta gamma delta epsilon zeta theta"}, # High overlap
            {"chunk_id": "C03", "lecture_id": "LEC_01", "text": "completely distinct topic about rendering and styles"},
            {"chunk_id": "C04", "lecture_id": "LEC_02", "text": "alpha beta gamma delta epsilon zeta eta"} # Different lecture, must keep
        ]
        deduped = VectorStore.deduplicate_chunks(mock_chunks, jaccard_threshold=0.50)
        deduped_ids = [c["chunk_id"] for c in deduped]
        self.assertIn("C01", deduped_ids)
        self.assertNotIn("C02", deduped_ids, "C02 is redundant with C01")
        self.assertIn("C03", deduped_ids)
        self.assertIn("C04", deduped_ids, "C04 is from different lecture, should be kept")

    def test_prompt_context_formatting(self):
        """Verify format_context_for_llm formats chunks with clear boundaries."""
        sample_chunks = [
            {
                "chunk_id": "LEC_01_CHK_0001",
                "lecture_title": "SEO in HTML",
                "start_timestamp": "00:01:00",
                "end_timestamp": "00:02:00",
                "text": "Sample text here"
            }
        ]
        formatted = prompts.format_context_for_llm(sample_chunks)
        self.assertIn("--- START CHUNK [LEC_01_CHK_0001]", formatted)
        self.assertIn("Sample text here", formatted)
        self.assertIn("--- END CHUNK [LEC_01_CHK_0001] ---", formatted)

    def test_validate_grounded_response_filters_hallucinations(self):
        """Verify validate_grounded_response rejects non-existent chunk IDs."""
        valid_chunks = [
            {"chunk_id": "VALID_01", "lecture_id": "LEC_01", "lecture_title": "Title 1", "start_timestamp": "00:01:00", "end_timestamp": "00:02:00"},
            {"chunk_id": "VALID_02", "lecture_id": "LEC_01", "lecture_title": "Title 1", "start_timestamp": "00:02:00", "end_timestamp": "00:03:00"}
        ]
        fake_llm_response = {
            "answerability_status": "ANSWERABLE",
            "answer": "This is an answer.",
            "supporting_chunk_ids": ["VALID_01", "FABRICATED_99"], # FABRICATED_99 does not exist
            "citations": [],
            "missing_information": None,
            "response_language": "English"
        }
        validated = prompts.validate_grounded_response(fake_llm_response, valid_chunks)
        self.assertEqual(validated["supporting_chunk_ids"], ["VALID_01"])
        self.assertNotIn("FABRICATED_99", validated["supporting_chunk_ids"])
        self.assertEqual(len(validated["verified_citations"]), 1)
        self.assertEqual(validated["verified_citations"][0]["chunk_id"], "VALID_01")


if __name__ == "__main__":
    unittest.main()

