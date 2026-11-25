"""
Build FAISS Index for LearnMate AI.
Constructs exact inner-product IndexFlatIP on unit-normalized 384-dim embeddings.
Saves:
- data/vector_store/faiss.index
- data/vector_store/index_manifest.json
- data/vector_store/index_statistics.json
Validates compatibility, row-to-chunk mapping, and persistence.
"""

import os
import sys
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import faiss

import config

def main():
    print("=" * 80)
    print("PHASE 5: BUILDING FAISS VECTOR INDEX")
    print("=" * 80)

    t_start = time.time()
    vector_store_dir = config.DATA_DIR / "vector_store"

    # 1. Verify embedding artifacts
    embeddings_path = vector_store_dir / "embeddings.npy"
    metadata_path = vector_store_dir / "chunk_metadata.json"
    manifest_path = vector_store_dir / "embedding_manifest.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file missing: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file missing: {metadata_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Embedding manifest missing: {manifest_path}")

    # Load embeddings
    embeddings = np.load(embeddings_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        emb_manifest = json.load(f)

    print(f"Loaded embeddings matrix: {embeddings.shape}, dtype={embeddings.dtype}")
    print(f"Loaded chunk metadata records: {len(metadata)}")

    # 2. Strict artifact verification
    N, D = embeddings.shape
    assert N == len(metadata), f"Row count {N} != metadata records {len(metadata)}"
    assert D == config.EMBEDDING_DIMENSION, f"Embedding dim {D} != {config.EMBEDDING_DIMENSION}"
    assert embeddings.dtype == np.float32, f"dtype {embeddings.dtype} != float32"
    assert not np.isnan(embeddings).any(), "Found NaN in embeddings"
    assert not np.isinf(embeddings).any(), "Found Inf in embeddings"

    # L2 normalization verification
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "Embeddings are not unit normalized!"
    print("Vector validations passed: Shape, Dtype, Finite, Unit L2 normalized.")

    # Verify unique chunk IDs and 1-to-1 indexing
    chunk_ids = [c["chunk_id"] for c in metadata]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in metadata!"
    for idx, c in enumerate(metadata):
        assert c["index"] == idx, f"Index mismatch at row {idx}"
    print("Metadata 1-to-1 indexing verified.")

    # 3. Construct FAISS IndexFlatIP
    t0 = time.time()
    index = faiss.IndexFlatIP(D)
    index.add(embeddings)
    construction_time = time.time() - t0

    assert index.ntotal == N, f"Index count {index.ntotal} != {N}"
    assert index.is_trained, "IndexFlatIP must be trained"
    print(f"FAISS IndexFlatIP constructed with {index.ntotal} vectors in {construction_time:.4f}s")

    # 4. Save FAISS index
    index_file = vector_store_dir / "faiss.index"
    faiss.write_index(index, str(index_file))
    index_file_size = index_file.stat().st_size
    print(f"Saved FAISS index to: {index_file} ({index_file_size} bytes)")

    # 5. Compute SHA256 of faiss.index
    with open(index_file, "rb") as f:
        index_hash = hashlib.sha256(f.read()).hexdigest()

    # 6. Generate Index Manifest
    index_manifest = {
        "manifest_version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "model_name": config.EMBEDDING_MODEL_NAME,
        "embedding_dimension": D,
        "index_type": "IndexFlatIP",
        "metric_type": "METRIC_INNER_PRODUCT",
        "similarity_measure": "Cosine Similarity (via unit L2-normalized vectors)",
        "total_vectors": index.ntotal,
        "normalized": True,
        "device": "cpu",
        "dataset_fingerprint_sha256": emb_manifest.get("dataset", {}).get("dataset_fingerprint_sha256", ""),
        "embedding_fingerprint_sha256": emb_manifest.get("artifacts", {}).get("embeddings_npy", {}).get("sha256", ""),
        "index_fingerprint_sha256": index_hash,
        "index_file": "faiss.index",
        "metadata_file": "chunk_metadata.json",
        "embeddings_file": "embeddings.npy"
    }

    index_manifest_file = vector_store_dir / "index_manifest.json"
    with open(index_manifest_file, "w", encoding="utf-8") as f:
        json.dump(index_manifest, f, indent=2)
    print(f"Saved index manifest: {index_manifest_file}")

    # 7. Generate Index Statistics
    per_lecture = {}
    for c in metadata:
        lec = c["lecture_id"]
        per_lecture[lec] = per_lecture.get(lec, 0) + 1

    index_statistics = {
        "created_at": datetime.now().isoformat(),
        "total_vectors": index.ntotal,
        "embedding_dimension": D,
        "index_type": "IndexFlatIP",
        "per_lecture_distribution": per_lecture,
        "performance": {
            "construction_time_seconds": round(construction_time, 6),
            "total_elapsed_seconds": round(time.time() - t_start, 4)
        },
        "storage": {
            "index_file_bytes": index_file_size,
            "embeddings_npy_bytes": embeddings_path.stat().st_size,
            "metadata_json_bytes": metadata_path.stat().st_size
        }
    }

    index_stats_file = vector_store_dir / "index_statistics.json"
    with open(index_stats_file, "w", encoding="utf-8") as f:
        json.dump(index_statistics, f, indent=2)
    print(f"Saved index statistics: {index_stats_file}")

    # 8. Test reload
    loaded_index = faiss.read_index(str(index_file))
    assert loaded_index.ntotal == N
    print(f"Index reload verification successful ({loaded_index.ntotal} vectors).")

    print("\n" + "=" * 80)
    print("FAISS VECTOR INDEX CREATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()

