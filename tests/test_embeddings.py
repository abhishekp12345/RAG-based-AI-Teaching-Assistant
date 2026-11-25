"""
Unit Tests for Phase 4: Semantic Embedding Generation
Uses Python standard library unittest.
Tests embedding model configuration, prefix handling, dimensions, normalization,
numerical integrity, persistence, and metadata alignment.
"""

import os
import json
import unittest
from pathlib import Path
import numpy as np

import config
from embeddings import EmbeddingGenerator


class TestEmbeddingConfiguration(unittest.TestCase):
    """Test embedding model configuration and default parameters."""

    def test_config_constants(self):
        """Verify configuration constants in config.py."""
        self.assertEqual(config.EMBEDDING_MODEL_NAME, "intfloat/multilingual-e5-small")
        self.assertEqual(config.EMBEDDING_DIMENSION, 384)
        self.assertEqual(config.EMBEDDING_MAX_LENGTH, 512)
        self.assertEqual(config.EMBEDDING_PASSAGE_PREFIX, "passage: ")
        self.assertEqual(config.EMBEDDING_QUERY_PREFIX, "query: ")
        self.assertTrue(config.EMBEDDING_NORMALIZE)
        self.assertEqual(config.DEVICE, "cpu")

    def test_generator_initialization(self):
        """Verify EmbeddingGenerator initialization defaults."""
        generator = EmbeddingGenerator()
        self.assertEqual(generator.model_name, "intfloat/multilingual-e5-small")
        self.assertEqual(generator.device, "cpu")
        self.assertEqual(generator.max_length, 512)
        self.assertEqual(generator.passage_prefix, "passage: ")
        self.assertEqual(generator.query_prefix, "query: ")
        self.assertTrue(str(generator.cache_folder).startswith("D:"))


class TestEmbeddingPersistenceAndIntegrity(unittest.TestCase):
    """Test saved vector store artifacts and data integrity."""

    @classmethod
    def setUpClass(cls):
        cls.vector_store_dir = config.DATA_DIR / "vector_store"
        cls.embeddings_file = cls.vector_store_dir / "embeddings.npy"
        cls.metadata_file = cls.vector_store_dir / "chunk_metadata.json"
        cls.manifest_file = cls.vector_store_dir / "embedding_manifest.json"
        cls.stats_file = cls.vector_store_dir / "embedding_statistics.json"

    def test_artifacts_exist(self):
        """Verify all expected artifact files exist in vector_store directory."""
        if not self.embeddings_file.exists():
            self.skipTest("embeddings.npy not generated yet.")
        self.assertTrue(self.embeddings_file.exists(), "embeddings.npy must exist")
        self.assertTrue(self.metadata_file.exists(), "chunk_metadata.json must exist")
        self.assertTrue(self.manifest_file.exists(), "embedding_manifest.json must exist")
        self.assertTrue(self.stats_file.exists(), "embedding_statistics.json must exist")

    def test_embeddings_matrix_shape_and_dtype(self):
        """Verify embeddings.npy shape, dtype, and lack of NaNs/Infs."""
        if not self.embeddings_file.exists():
            self.skipTest("embeddings.npy not generated yet.")

        embeddings = np.load(self.embeddings_file)
        self.assertEqual(embeddings.ndim, 2)
        self.assertEqual(embeddings.shape[1], 384, "Embedding dimension must be 384")
        self.assertGreaterEqual(embeddings.shape[0], 102, "Must contain at least 102 chunks")
        self.assertEqual(embeddings.dtype, np.float32, "dtype must be float32")
        self.assertFalse(np.isnan(embeddings).any(), "Matrix must not contain NaN")
        self.assertFalse(np.isinf(embeddings).any(), "Matrix must not contain Inf")

    def test_embeddings_l2_normalization(self):
        """Verify all stored embeddings have unit L2 norm."""
        if not self.embeddings_file.exists():
            self.skipTest("embeddings.npy not generated yet.")

        embeddings = np.load(self.embeddings_file)
        norms = np.linalg.norm(embeddings, axis=1)
        self.assertTrue(
            np.allclose(norms, 1.0, atol=1e-4),
            f"Embeddings must be unit-normalized. Norm range: [{np.min(norms)}, {np.max(norms)}]"
        )

    def test_metadata_alignment(self):
        """Verify 1-to-1 alignment between chunk_metadata.json and embeddings.npy."""
        if not self.embeddings_file.exists() or not self.metadata_file.exists():
            self.skipTest("Vector store artifacts not generated yet.")

        embeddings = np.load(self.embeddings_file)
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertEqual(len(metadata), embeddings.shape[0], "Metadata count must match embedding matrix rows")

        # Verify indices and required fields
        for idx, item in enumerate(metadata):
            self.assertEqual(item["index"], idx)
            self.assertIn("chunk_id", item)
            self.assertIn("lecture_id", item)
            self.assertIn("start_time", item)
            self.assertIn("end_time", item)
            self.assertIn("text", item)
            self.assertGreater(len(item["text"].strip()), 0)

    def test_manifest_metadata(self):
        """Verify embedding manifest contains required provenance metadata."""
        if not self.manifest_file.exists():
            self.skipTest("embedding_manifest.json not generated yet.")

        with open(self.manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["model_name"], "intfloat/multilingual-e5-small")
        self.assertEqual(manifest["embedding_dimension"], 384)
        self.assertTrue(manifest["normalized"])
        self.assertEqual(manifest["passage_prefix"], "passage: ")
        self.assertEqual(manifest["query_prefix"], "query: ")
        self.assertIn("dataset_fingerprint_sha256", manifest["dataset"])
        self.assertIn("artifacts", manifest)


class TestEmbeddingLogic(unittest.TestCase):
    """Test embedding generator logic, prefixes, and fingerprinting."""

    def test_dataset_fingerprint_determinism(self):
        """Verify dataset fingerprint calculation is deterministic."""
        sample_chunks = [
            {"chunk_id": "LEC_01_C001", "start_time": 0.0, "end_time": 10.0, "text": "Sample text A"},
            {"chunk_id": "LEC_01_C002", "start_time": 10.0, "end_time": 20.0, "text": "Sample text B"}
        ]
        fp1 = EmbeddingGenerator.compute_dataset_fingerprint(sample_chunks)
        fp2 = EmbeddingGenerator.compute_dataset_fingerprint(sample_chunks)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

    def test_empty_query_raises_value_error(self):
        """Verify empty query string raises ValueError."""
        generator = EmbeddingGenerator()
        with self.assertRaises(ValueError):
            generator.embed_query("")
        with self.assertRaises(ValueError):
            generator.embed_query("   ")

    def test_empty_passages_returns_empty_array(self):
        """Verify empty passages list returns array with shape (0, 384)."""
        generator = EmbeddingGenerator()
        result = generator.embed_passages([])
        self.assertEqual(result.shape, (0, 384))
        self.assertEqual(result.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

