"""
Vector Store and Retrieval Module for LearnMate AI.
Supports FAISS IndexFlatIP with automatic NumPy fallback.
Features:
- Global and lecture-filtered semantic retrieval
- Asymmetric query embedding with 'query: ' prefix
- Exact cosine similarity scoring over unit-normalized vectors
- Pure-Python BM25 lexical retrieval for exact technical syntax
- Reciprocal Rank Fusion (RRF) hybrid retrieval
- Semantic context deduplication for overlapping chunk windows
- Full provenance metadata and citation resolution
"""

import os
import re
import math
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

import config
from embeddings import EmbeddingGenerator

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class BM25LexicalRetriever:
    """
    Lightweight, self-contained Okapi BM25 implementation.
    Requires no external dependencies.
    Preserves HTML/code tokens (<video>, <iframe>, controls, etc.) and Devanagari words.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.metadata: List[Dict[str, Any]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize preserving programming syntax, tags (<video>, <iframe>),
        English technical terms, and Devanagari words.
        """
        if not text:
            return []
        # Pattern captures HTML tags, alphanumeric words, and hyphenated terms
        tokens = re.findall(r"</?[a-zA-Z0-9_\-]+>|[a-zA-Z0-9_\-]+|[\u0900-\u097F]+", text.lower())
        return tokens

    def index(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Index documents and compute collection term frequencies and IDF."""
        self.metadata = metadata
        self.doc_count = len(documents)
        self.corpus = [self.tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avg_doc_len = float(np.mean(self.doc_lengths)) if self.doc_lengths else 0.0

        # Document frequencies
        self.df = {}
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1

        # Lucene-style IDF
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        eligible_indices: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Search corpus for query using BM25 scoring.
        Returns list of (doc_index, score) tuples.
        """
        q_tokens = self.tokenize(query)
        if not q_tokens or not self.corpus:
            return []

        target_indices = eligible_indices if eligible_indices is not None else list(range(self.doc_count))
        scores = []

        for idx in target_indices:
            doc = self.corpus[idx]
            doc_len = self.doc_lengths[idx]
            score = 0.0

            # Count term frequencies in this document
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1

            for q_term in q_tokens:
                if q_term in tf:
                    term_freq = tf[q_term]
                    idf_val = self.idf.get(q_term, 0.0)
                    numerator = term_freq * (self.k1 + 1.0)
                    denominator = term_freq + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                    score += idf_val * (numerator / denominator)

            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class VectorStore:
    """
    Unified Vector Store supporting FAISS IndexFlatIP with automatic NumPy fallback,
    hybrid retrieval (dense + BM25 via RRF), lecture filtering, and context deduplication.
    """

    def __init__(
        self,
        backend: str = config.VECTOR_STORE_BACKEND,
        directory: Optional[Path] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        self.backend = "faiss" if (backend == "auto" and FAISS_AVAILABLE) else backend
        self.directory = Path(directory or config.VECTOR_STORE_DIR)
        self.generator = embedding_generator
        self.index: Optional[Any] = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension: int = config.EMBEDDING_DIMENSION
        self.manifest: Dict[str, Any] = {}
        self.bm25: Optional[BM25LexicalRetriever] = None

    def get_embedding_generator(self) -> EmbeddingGenerator:
        """Lazy loader for query embedding generator."""
        if self.generator is None:
            self.generator = EmbeddingGenerator()
        return self.generator

    def load(self, directory: Optional[Path] = None):
        """
        Load vector index, metadata, and manifest from disk.
        Validates model and dataset compatibility.
        """
        load_dir = Path(directory or self.directory)
        meta_file = load_dir / "chunk_metadata.json"
        npy_file = load_dir / "embeddings.npy"
        faiss_file = load_dir / "faiss.index"
        manifest_file = load_dir / "index_manifest.json"

        if not meta_file.exists() or not npy_file.exists():
            # Resilient fresh-clone auto-initialization from available chunks
            chunks_dir = Path(config.CHUNKS_DIR)
            found_chunks: List[Dict[str, Any]] = []
            if chunks_dir.exists():
                for chunk_file in sorted(chunks_dir.glob("*.json")):
                    try:
                        with open(chunk_file, "r", encoding="utf-8") as cf:
                            loaded = json.load(cf)
                            if isinstance(loaded, list):
                                found_chunks.extend(loaded)
                    except Exception:
                        pass

            if found_chunks:
                logger.info(f"Auto-initializing vector store from {len(found_chunks)} available chunks...")
                load_dir.mkdir(parents=True, exist_ok=True)
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(found_chunks, f, indent=2)
                self.metadata = found_chunks
                gen = self.get_embedding_generator()
                texts = [c.get("text", "") for c in found_chunks]
                vecs = gen.embed_passages(texts)
                np.save(str(npy_file), vecs.astype(np.float32))
                self.embeddings = vecs.astype(np.float32)
                self.dimension = self.embeddings.shape[1]
                if self.backend == "faiss" and FAISS_AVAILABLE:
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.index.add(self.embeddings)
                    try:
                        faiss.write_index(self.index, str(faiss_file))
                    except Exception:
                        pass
                else:
                    self.backend = "numpy"
                self._init_bm25()
                return
            else:
                raise FileNotFoundError(
                    f"Vector store index not found in {load_dir}. "
                    f"Please run 'python generate_embeddings.py && python build_faiss_index.py' to generate embeddings."
                )

        # Load metadata
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load embeddings (kept in memory for NumPy fallback & lecture filtering)
        self.embeddings = np.load(npy_file).astype(np.float32)
        self.dimension = self.embeddings.shape[1]

        # Load manifest if available
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
                expected_dim = self.manifest.get("embedding_dimension", self.dimension)
                if expected_dim != self.dimension:
                    raise ValueError(f"Index dimension mismatch: manifest={expected_dim}, matrix={self.dimension}")

        # Load or initialize FAISS index
        if self.backend == "faiss" and FAISS_AVAILABLE:
            if faiss_file.exists():
                self.index = faiss.read_index(str(faiss_file))
                if self.index.ntotal != len(self.metadata):
                    raise ValueError(f"FAISS vector count {self.index.ntotal} != metadata count {len(self.metadata)}")
            else:
                # Build index dynamically from embeddings.npy
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(self.embeddings)
        else:
            self.backend = "numpy"

        # Initialize lightweight BM25 index over chunk texts
        self._init_bm25()

    def _init_bm25(self):
        """Initialize BM25 index over chunk text."""
        texts = [c.get("text", "") for c in self.metadata]
        self.bm25 = BM25LexicalRetriever()
        self.bm25.index(texts, self.metadata)

    def search_dense(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        lecture_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute dense semantic search using E5 query embedding and FAISS / NumPy.
        Supports lecture-specific filtering across the exact candidate subset.
        """
        if not query or not query.strip():
            return []
        if not self.metadata or self.embeddings is None:
            return []

        generator = self.get_embedding_generator()
        t_embed_start = time.time()
        q_emb = generator.embed_query(query)
        embed_time = time.time() - t_embed_start

        # Determine eligible indices based on lecture_id
        if lecture_id:
            eligible_indices = [
                idx for idx, c in enumerate(self.metadata)
                if c.get("lecture_id") == lecture_id
            ]
            if not eligible_indices:
                return []
        else:
            eligible_indices = list(range(len(self.metadata)))

        eff_k = min(top_k, len(eligible_indices))
        if eff_k <= 0:
            return []

        t_search_start = time.time()

        # If filtering is applied or FAISS not used, perform exact inner-product over eligible subset
        if lecture_id is not None or self.backend != "faiss" or self.index is None:
            sub_embeddings = self.embeddings[eligible_indices]
            # Exact inner product (vectors are normalized, so inner product == cosine similarity)
            sims = np.dot(sub_embeddings, q_emb)
            top_sub_idx = np.argsort(sims)[::-1][:eff_k]

            results = []
            for rank, sub_i in enumerate(top_sub_idx, 1):
                orig_idx = eligible_indices[sub_i]
                item = self._format_result_item(self.metadata[orig_idx], rank, float(sims[sub_i]))
                results.append(item)
        else:
            # Global FAISS search
            query_mat = q_emb.reshape(1, -1).astype(np.float32)
            scores, indices = self.index.search(query_mat, eff_k)

            results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                if idx != -1 and idx < len(self.metadata):
                    item = self._format_result_item(self.metadata[idx], rank, float(score))
                    results.append(item)

        search_time = time.time() - t_search_start

        # Attach timing metrics to results
        for r in results:
            r["timing"] = {
                "embed_time_ms": round(embed_time * 1000, 2),
                "search_time_ms": round(search_time * 1000, 2),
                "backend": self.backend
            }

        return results

    def search_lexical(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        lecture_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute BM25 lexical keyword search.
        Useful for exact programming tag matching (<video>, <iframe>, controls, etc.).
        """
        if not self.bm25 or not query or not query.strip():
            return []

        eligible_indices = None
        if lecture_id:
            eligible_indices = [
                idx for idx, c in enumerate(self.metadata)
                if c.get("lecture_id") == lecture_id
            ]
            if not eligible_indices:
                return []

        scored_docs = self.bm25.search(query, top_k=top_k, eligible_indices=eligible_indices)
        results = []
        for rank, (idx, score) in enumerate(scored_docs, 1):
            item = self._format_result_item(self.metadata[idx], rank, float(score))
            results.append(item)
        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        lecture_id: Optional[str] = None,
        dense_weight: float = 0.7,
        lexical_weight: float = 0.3,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining Dense Semantic Retrieval (E5 + FAISS)
        and Lexical BM25 using Reciprocal Rank Fusion (RRF).
        RRF Score = (w_dense / (rrf_k + rank_dense)) + (w_lexical / (rrf_k + rank_lexical))
        """
        # Fetch candidate pools from both retrievers (e.g. 2x top_k)
        pool_size = max(top_k * 2, 10)
        dense_results = self.search_dense(query, top_k=pool_size, lecture_id=lecture_id)
        lexical_results = self.search_lexical(query, top_k=pool_size, lecture_id=lecture_id)

        # Map chunk_id to ranks and objects
        dense_ranks = {r["chunk_id"]: (idx + 1, r) for idx, r in enumerate(dense_results)}
        lexical_ranks = {r["chunk_id"]: (idx + 1, r) for idx, r in enumerate(lexical_results)}

        all_chunk_ids = set(dense_ranks.keys()).union(set(lexical_ranks.keys()))
        if not all_chunk_ids:
            return []

        fused_scores: List[Tuple[str, float, Dict[str, Any]]] = []

        for cid in all_chunk_ids:
            rrf_score = 0.0
            chunk_item = None

            if cid in dense_ranks:
                d_rank, chunk_item = dense_ranks[cid]
                rrf_score += dense_weight * (1.0 / (rrf_k + d_rank))

            if cid in lexical_ranks:
                l_rank, item = lexical_ranks[cid]
                if chunk_item is None:
                    chunk_item = item
                rrf_score += lexical_weight * (1.0 / (rrf_k + l_rank))

            if chunk_item is not None:
                fused_scores.append((cid, rrf_score, chunk_item))

        # Sort descending by RRF score
        fused_scores.sort(key=lambda x: x[1], reverse=True)

        hybrid_results = []
        for rank, (cid, rrf_val, item) in enumerate(fused_scores[:top_k], 1):
            res_item = dict(item)
            res_item["rank"] = rank
            res_item["rrf_score"] = round(rrf_val, 6)
            # If dense score was present, preserve it as dense_similarity
            if cid in dense_ranks:
                res_item["dense_score"] = dense_ranks[cid][1].get("similarity_score", 0.0)
            if cid in lexical_ranks:
                res_item["bm25_score"] = lexical_ranks[cid][1].get("similarity_score", 0.0)
            hybrid_results.append(res_item)

        return hybrid_results

    def search(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        lecture_id: Optional[str] = None,
        strategy: str = "dense",
        deduplicate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Unified search endpoint supporting 'dense', 'lexical', and 'hybrid' strategies,
        with optional context deduplication.
        """
        if strategy == "hybrid":
            results = self.search_hybrid(query, top_k=top_k, lecture_id=lecture_id)
        elif strategy == "lexical":
            results = self.search_lexical(query, top_k=top_k, lecture_id=lecture_id)
        else:
            results = self.search_dense(query, top_k=top_k, lecture_id=lecture_id)

        if deduplicate and len(results) > 1:
            results = self.deduplicate_chunks(results)

        return results

    @staticmethod
    def deduplicate_chunks(
        chunks: List[Dict[str, Any]],
        jaccard_threshold: float = 0.55
    ) -> List[Dict[str, Any]]:
        """
        Lightweight context deduplication for overlapping lecture chunks.
        Discards a lower-ranked chunk if its word-level Jaccard overlap with a
        higher-ranked chunk from the same lecture exceeds the threshold,
        while preserving distinct semantic passages and timestamps.
        """
        if not chunks:
            return []

        def get_words(text: str) -> set:
            return set(text.lower().split())

        selected: List[Dict[str, Any]] = []

        for candidate in chunks:
            cand_words = get_words(candidate.get("text", ""))
            cand_lec = candidate.get("lecture_id")
            is_redundant = False

            for kept in selected:
                # Deduplication applies only within the same lecture
                if kept.get("lecture_id") == cand_lec:
                    kept_words = get_words(kept.get("text", ""))
                    intersection = len(cand_words.intersection(kept_words))
                    union = len(cand_words.union(kept_words))
                    jaccard = intersection / union if union > 0 else 0.0

                    if jaccard >= jaccard_threshold:
                        is_redundant = True
                        break

            if not is_redundant:
                selected.append(candidate)

        # Re-number ranks after deduplication
        for rank, item in enumerate(selected, 1):
            item["rank"] = rank

        return selected

    def _format_result_item(self, chunk: Dict[str, Any], rank: int, score: float) -> Dict[str, Any]:
        """Format chunk record into enriched retrieval result schema."""
        start_sec = chunk.get("start_time", 0.0)
        end_sec = chunk.get("end_time", 0.0)
        dur = round(end_sec - start_sec, 2)

        def sec_to_ts(seconds: float) -> str:
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        return {
            "rank": rank,
            "chunk_id": chunk.get("chunk_id", ""),
            "lecture_id": chunk.get("lecture_id", ""),
            "lecture_title": chunk.get("lecture_title", ""),
            "tutorial_number": chunk.get("tutorial_number", None),
            "similarity_score": round(score, 4),
            "text": chunk.get("text", ""),
            "start_time": start_sec,
            "end_time": end_sec,
            "duration": dur,
            "start_timestamp": chunk.get("start_timestamp", sec_to_ts(start_sec)),
            "end_timestamp": chunk.get("end_timestamp", sec_to_ts(end_sec)),
            "source_video": chunk.get("source_video", ""),
            "source_transcript_path": chunk.get("source_transcript_path", "")
        }
