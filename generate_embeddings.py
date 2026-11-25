"""
Batch Embedding Generation for LearnMate AI.
Generates 384-dimensional normalized dense embeddings for all validated chunks
using intfloat/multilingual-e5-small on CPU with batch_size=1 and memory management.
Saves:
- data/vector_store/embeddings.npy
- data/vector_store/chunk_metadata.json
- data/vector_store/embedding_manifest.json
- data/vector_store/embedding_statistics.json
"""

import os
import sys
import time
import json
import ctypes
from ctypes import wintypes
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np

import config
from embeddings import EmbeddingGenerator

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]

def get_process_memory_mb() -> float:
    try:
        p = ctypes.windll.kernel32.GetCurrentProcess()
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if ctypes.windll.psapi.GetProcessMemoryInfo(p, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0

def main():
    print("=" * 70)
    print("LearnMate AI - Phase 4: Batch Semantic Embedding Generation")
    print("=" * 70)

    start_time = time.time()
    initial_mem = get_process_memory_mb()
    print(f"Start Time: {datetime.now().isoformat()}")
    print(f"Initial Memory: {initial_mem:.2f} MB")

    # 1. Output directory setup
    vector_store_dir = config.DATA_DIR / "vector_store"
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load authoritative chunks
    chunks_path = config.DATA_DIR / "chunks" / "all_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    print(f"Loaded {len(raw_chunks)} chunks from {chunks_path}")

    # 3. Instantiate generator and load model
    generator = EmbeddingGenerator()
    print(f"Model: {generator.model_name}")
    print(f"Target Device: {generator.device}")
    print(f"Batch Size: {generator.batch_size}")
    print(f"Max Sequence Length: {generator.max_length}")
    print(f"Passage Prefix: '{generator.passage_prefix}'")
    print(f"Query Prefix: '{generator.query_prefix}'")

    print("\n[Step 1/4] Validating token counts against E5 512-token limit...")
    validated_chunks = generator.process_and_validate_chunks(raw_chunks)
    print(f"Chunks after token length validation: {len(validated_chunks)} (Raw was: {len(raw_chunks)})")

    # 4. Generate embeddings
    texts = [c["text"] for c in validated_chunks]
    print(f"\n[Step 2/4] Generating dense embeddings for {len(texts)} chunks...")
    t_gen_start = time.time()
    embeddings = generator.embed_passages(texts, batch_size=generator.batch_size, show_progress_bar=True)
    t_gen_end = time.time()
    total_gen_time = t_gen_end - t_gen_start
    peak_mem = get_process_memory_mb()

    print(f"\nEmbedding generation complete in {total_gen_time:.2f}s ({total_gen_time / len(texts):.3f}s / chunk)")
    print(f"Memory during inference: {peak_mem:.2f} MB")

    # 5. Numerical integrity verification
    print("\n[Step 3/4] Verifying matrix integrity...")
    N, D = embeddings.shape
    print(f"  Matrix shape: {N} x {D}")
    print(f"  Data type: {embeddings.dtype}")
    assert N == len(validated_chunks), f"Row count {N} does not match chunk count {len(validated_chunks)}"
    assert D == config.EMBEDDING_DIMENSION, f"Dimension {D} != {config.EMBEDDING_DIMENSION}"
    assert embeddings.dtype == np.float32, f"dtype {embeddings.dtype} != float32"
    assert not np.isnan(embeddings).any(), "Found NaN values in embeddings!"
    assert not np.isinf(embeddings).any(), "Found Inf values in embeddings!"

    norms = np.linalg.norm(embeddings, axis=1)
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))
    mean_norm = float(np.mean(norms))
    print(f"  L2 Norm range: [{min_norm:.6f}, {max_norm:.6f}], mean: {mean_norm:.6f}")
    assert np.allclose(norms, 1.0, atol=1e-4), "Not all embeddings are unit-normalized!"
    print("  Integrity checks: ALL PASSED")

    # 6. Save artifacts
    print("\n[Step 4/4] Saving vector store artifacts...")
    # 6a. embeddings.npy
    embeddings_file = vector_store_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"  Saved embeddings matrix: {embeddings_file} ({embeddings_file.stat().st_size / (1024*1024):.2f} MB)")

    # 6b. chunk_metadata.json (1-to-1 indexed metadata)
    metadata_file = vector_store_dir / "chunk_metadata.json"
    chunk_metadata = []
    for idx, c in enumerate(validated_chunks):
        item = {
            "index": idx,
            "chunk_id": c["chunk_id"],
            "lecture_id": c["lecture_id"],
            "lecture_title": c["lecture_title"],
            "start_time": c["start_time"],
            "end_time": c["end_time"],
            "duration": round(c["end_time"] - c["start_time"], 2),
            "token_count": c.get("token_count", 0),
            "e5_token_count": c.get("e5_token_count", 0),
            "character_count": c.get("character_count", len(c["text"])),
            "text": c["text"]
        }
        chunk_metadata.append(item)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved metadata: {metadata_file} ({len(chunk_metadata)} items)")

    # 6c. Dataset fingerprint and manifest
    dataset_fingerprint = generator.compute_dataset_fingerprint(validated_chunks)

    # Compute SHA256 of embeddings.npy
    with open(embeddings_file, "rb") as f:
        npy_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "model_name": generator.model_name,
        "embedding_dimension": config.EMBEDDING_DIMENSION,
        "normalized": True,
        "max_sequence_length": generator.max_length,
        "batch_size": generator.batch_size,
        "passage_prefix": generator.passage_prefix,
        "query_prefix": generator.query_prefix,
        "device": generator.device,
        "dataset": {
            "source_chunks_file": str(chunks_path.relative_to(config.BASE_DIR)),
            "raw_chunk_count": len(raw_chunks),
            "embedded_chunk_count": len(validated_chunks),
            "dataset_fingerprint_sha256": dataset_fingerprint,
        },
        "artifacts": {
            "embeddings_npy": {
                "file": "embeddings.npy",
                "shape": [N, D],
                "dtype": "float32",
                "sha256": npy_hash,
                "size_bytes": embeddings_file.stat().st_size
            },
            "chunk_metadata_json": {
                "file": "chunk_metadata.json",
                "total_records": len(chunk_metadata)
            }
        }
    }

    manifest_file = vector_store_dir / "embedding_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest: {manifest_file}")

    # 6d. Embedding statistics
    per_lecture_counts = {}
    for c in validated_chunks:
        lec = c["lecture_id"]
        per_lecture_counts[lec] = per_lecture_counts.get(lec, 0) + 1

    stats = {
        "generated_at": datetime.now().isoformat(),
        "total_chunks_embedded": len(validated_chunks),
        "embedding_dimension": D,
        "per_lecture_breakdown": per_lecture_counts,
        "performance": {
            "total_time_seconds": round(time.time() - start_time, 2),
            "embedding_time_seconds": round(total_gen_time, 2),
            "average_time_per_chunk_seconds": round(total_gen_time / len(texts), 4),
            "peak_memory_mb": round(peak_mem, 2),
            "initial_memory_mb": round(initial_mem, 2)
        },
        "embedding_norms": {
            "min": round(min_norm, 6),
            "max": round(max_norm, 6),
            "mean": round(mean_norm, 6)
        }
    }

    stats_file = vector_store_dir / "embedding_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved statistics: {stats_file}")

    # 7. Release model
    generator.release_model()
    final_mem = get_process_memory_mb()
    print(f"\nFinal Memory after release: {final_mem:.2f} MB")
    print(f"Total Phase 4 Batch Generation Runtime: {time.time() - start_time:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
