"""
Embeddings Module for LearnMate AI.
Generates normalized dense vector embeddings using intfloat/multilingual-e5-small on CPU.
Supports Hindi, English, and Hinglish semantic retrieval with passage/query prefixes,
lazy model loading, and strict memory management on D: drive.
"""

import os
import gc
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch

import config


class EmbeddingGenerator:
    """
    Manages dense embedding extraction for text chunks and queries using
    multilingual E5 models with memory-efficient CPU inference.
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        device: str = config.DEVICE,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        max_length: int = config.EMBEDDING_MAX_LENGTH,
        passage_prefix: str = config.EMBEDDING_PASSAGE_PREFIX,
        query_prefix: str = config.EMBEDDING_QUERY_PREFIX,
        cache_folder: Optional[Path] = None
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.passage_prefix = passage_prefix
        self.query_prefix = query_prefix
        self.cache_folder = Path(cache_folder or config.SENTENCE_TRANSFORMERS_HOME)
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self._model = None

    def load_model(self):
        """
        Lazy loader for SentenceTransformer E5 model.
        Forces cache directory to D: drive and configures maximum sequence length.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # Set HF cache environment variables for safety
            os.environ["HF_HOME"] = str(config.HF_HOME)
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(config.SENTENCE_TRANSFORMERS_HOME)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_folder)
            )
            # Set maximum sequence length to 512
            self._model.max_seq_length = self.max_length
        return self._model

    def release_model(self):
        """Release model weights and garbage collect to preserve system memory."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()

    def count_tokens(self, text: str, is_passage: bool = True) -> int:
        """
        Count tokens using the E5 model's actual tokenizer,
        including passage/query prefixes and special tokens.
        """
        model = self.load_model()
        prefix = self.passage_prefix if is_passage else self.query_prefix
        full_text = f"{prefix}{text}"
        tokenizer = model.tokenizer
        tokens = tokenizer.encode(full_text, truncation=False, add_special_tokens=True)
        return len(tokens)

    def embed_passages(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate normalized dense embeddings for document passages.
        Automatically prepends the required 'passage: ' prefix.
        """
        if not texts:
            return np.empty((0, config.EMBEDDING_DIMENSION), dtype=np.float32)

        # Prepend passage prefix
        prefixed_texts = [
            f"{self.passage_prefix}{t}" if not t.startswith(self.passage_prefix) else t
            for t in texts
        ]

        model = self.load_model()
        eff_batch_size = batch_size or self.batch_size

        with torch.no_grad():
            embeddings = model.encode(
                prefixed_texts,
                batch_size=eff_batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

        embeddings_f32 = np.asarray(embeddings, dtype=np.float32)

        # Validate numerical integrity
        if np.isnan(embeddings_f32).any() or np.isinf(embeddings_f32).any():
            raise ValueError("Generated passage embeddings contain NaN or infinite values!")

        return embeddings_f32

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate normalized dense embedding for a single user query.
        Automatically prepends the required 'query: ' prefix.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        prefixed_query = (
            f"{self.query_prefix}{query.strip()}"
            if not query.strip().startswith(self.query_prefix)
            else query.strip()
        )

        model = self.load_model()

        with torch.no_grad():
            embedding = model.encode(
                prefixed_query,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

        embedding_f32 = np.asarray(embedding, dtype=np.float32)

        if np.isnan(embedding_f32).any() or np.isinf(embedding_f32).any():
            raise ValueError("Generated query embedding contains NaN or infinite values!")

        return embedding_f32

    @staticmethod
    def compute_dataset_fingerprint(chunks: List[Dict[str, Any]]) -> str:
        """Compute deterministic SHA-256 hash of the chunk dataset."""
        hasher = hashlib.sha256()
        for c in chunks:
            row_str = f"{c.get('chunk_id')}:{c.get('start_time')}:{c.get('end_time')}:{c.get('text')}"
            hasher.update(row_str.encode("utf-8"))
        return hasher.hexdigest()

    def process_and_validate_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate all chunks against the E5 tokenizer's 512 token limit.
        If any chunk exceeds 512 tokens with prefix, safely split it on sentence boundaries.
        """
        validated_chunks: List[Dict[str, Any]] = []

        for c in chunks:
            raw_text = c.get("text", "")
            e5_tokens = self.count_tokens(raw_text, is_passage=True)

            if e5_tokens <= self.max_length:
                new_c = dict(c)
                new_c["e5_token_count"] = e5_tokens
                validated_chunks.append(new_c)
            else:
                # Safely split oversized chunk
                print(f"Warning: Chunk {c.get('chunk_id')} has {e5_tokens} E5 tokens (> {self.max_length}). Splitting...")
                import re
                parts = [p.strip() for p in re.split(r"(?<=[.!?।])\s+", raw_text) if p.strip()]
                if len(parts) <= 1:
                    words = raw_text.split()
                    mid = len(words) // 2
                    parts = [" ".join(words[:mid]), " ".join(words[mid:])]

                sub_chunks = []
                current_part = []
                for p in parts:
                    candidate = " ".join(current_part + [p])
                    if self.count_tokens(candidate, is_passage=True) <= self.max_length:
                        current_part.append(p)
                    else:
                        if current_part:
                            sub_chunks.append(" ".join(current_part))
                        current_part = [p]
                if current_part:
                    sub_chunks.append(" ".join(current_part))

                dur = c.get("end_time", 0.0) - c.get("start_time", 0.0)
                tot_chars = len(raw_text)
                cur_char = 0

                for s_idx, s_text in enumerate(sub_chunks):
                    sub_c = dict(c)
                    sub_c["chunk_id"] = f"{c.get('chunk_id')}_{chr(65 + s_idx)}"
                    sub_c["text"] = s_text
                    sub_c["token_count"] = self.count_tokens(s_text, is_passage=False)
                    sub_c["e5_token_count"] = self.count_tokens(s_text, is_passage=True)
                    sub_c["character_count"] = len(s_text)
                    p_len = len(s_text)
                    sub_c["start_time"] = round(c.get("start_time", 0.0) + (cur_char / tot_chars) * dur, 2)
                    sub_c["end_time"] = round(c.get("start_time", 0.0) + ((cur_char + p_len) / tot_chars) * dur, 2)
                    cur_char += p_len + 1
                    validated_chunks.append(sub_c)

        return validated_chunks
