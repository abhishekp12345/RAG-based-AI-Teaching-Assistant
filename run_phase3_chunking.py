"""
Phase 3 Execution Script:
1. Loads authoritative lecture_manifest.json.
2. Performs text preprocessing & timestamp-aware overlapping chunking for all 3 lectures.
3. Generates per-lecture chunks, all_chunks.json, chunk_manifest.json, and chunk_statistics.json.
4. Performs comprehensive quality validation and generates statistics.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from chunker import TranscriptChunker, MultilingualTokenizer, seconds_to_timestamp


def main():
    print("=" * 70)
    print("LEARNMATE AI - PHASE 3: TIMESTAMP-AWARE TEXT CHUNKING")
    print("=" * 70)

    # 1. Load authoritative manifest
    manifest_path = config.DATA_DIR / "lecture_manifest.json"
    assert manifest_path.exists(), f"Authoritative manifest missing: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    lectures = manifest.get("lectures", [])
    print(f"Loaded Master Manifest: {len(lectures)} lectures found.")
    for lec in lectures:
        print(f"  - [{lec['lecture_id']}] {lec['lecture_title']}")
        print(f"      Transcript Path: {lec['transcript_path']}")
        print(f"      Duration: {lec['duration_formatted']} ({lec['duration_seconds']}s)")

    # Ensure LEC_01 uses validated cleaned/corrected transcript
    lec_01 = next(l for l in lectures if l["lecture_id"] == "LEC_01")
    assert ("_cleaned.json" in lec_01["transcript_path"] or "_hi_en_corrected.json" in lec_01["transcript_path"]), (
        f"LEC_01 must use the cleaned/corrected transcript! Found: {lec_01['transcript_path']}"
    )

    # 2. Initialize Chunker
    print("\n--- Initializing Segment-Aware Chunker ---")
    print(f"Target Chunk Size: {config.CHUNK_SIZE_TOKENS} tokens")
    print(f"Max Chunk Size: {config.MAX_CHUNK_SIZE_TOKENS} tokens")
    print(f"Target Overlap: {config.CHUNK_OVERLAP_TOKENS} tokens")
    print(f"Gap Threshold: {config.CHUNK_GAP_THRESHOLD_SECONDS} seconds")

    tokenizer = MultilingualTokenizer()
    chunker = TranscriptChunker(
        chunk_size_tokens=config.CHUNK_SIZE_TOKENS,
        max_chunk_size_tokens=config.MAX_CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        gap_threshold_seconds=config.CHUNK_GAP_THRESHOLD_SECONDS,
        tokenizer=tokenizer
    )

    t0_all = time.time()
    all_chunks = []
    lecture_chunk_counts = {}
    lecture_stats = {}

    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    for lec in lectures:
        lec_id = lec["lecture_id"]
        t_path = BASE_DIR / lec["transcript_path"]
        assert t_path.exists(), f"Transcript file not found: {t_path}"

        with open(t_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)

        raw_segments = t_data.get("segments", [])
        raw_segment_ids = set(s.get("id") for s in raw_segments)

        print(f"\nProcessing {lec_id}: {lec['lecture_title']} ({len(raw_segments)} segments)...")
        t0_lec = time.time()
        chunks = chunker.chunk_lecture(t_data, lec)
        t_lec = round(time.time() - t0_lec, 4)

        chunk_dicts = [c.to_dict() for c in chunks]
        lecture_chunk_counts[lec_id] = len(chunks)
        all_chunks.extend(chunk_dicts)

        # Save individual lecture chunks
        lec_chunk_path = config.CHUNKS_DIR / f"{lec_id}_chunks.json"
        with open(lec_chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(chunks)} chunks to {lec_chunk_path.name} in {t_lec}s")

        # Compute per-lecture stats
        tokens = [c.token_count for c in chunks]
        durations = [c.end_time - c.start_time for c in chunks]
        covered_segments = set()
        for c in chunks:
            covered_segments.update(c.source_segment_indices)

        # Compute overlaps between consecutive chunks
        overlaps = []
        for i in range(len(chunks) - 1):
            if chunks[i+1].start_time < chunks[i].end_time:
                overlap_sec = round(chunks[i].end_time - chunks[i+1].start_time, 2)
                shared_segs = set(chunks[i].source_segment_indices).intersection(set(chunks[i+1].source_segment_indices))
                overlaps.append({"overlap_sec": overlap_sec, "shared_segments": len(shared_segs)})

        avg_tok = round(sum(tokens) / len(tokens), 1) if tokens else 0
        min_tok = min(tokens) if tokens else 0
        max_tok = max(tokens) if tokens else 0
        avg_dur = round(sum(durations) / len(durations), 1) if durations else 0
        cov_pct = round((len(covered_segments) / len(raw_segment_ids)) * 100, 2) if raw_segment_ids else 0

        lecture_stats[lec_id] = {
            "lecture_title": lec["lecture_title"],
            "total_source_segments": len(raw_segments),
            "covered_segments": len(covered_segments),
            "coverage_percentage": cov_pct,
            "total_chunks": len(chunks),
            "min_tokens": min_tok,
            "max_tokens": max_tok,
            "avg_tokens": avg_tok,
            "avg_duration_seconds": avg_dur,
            "total_overlapping_pairs": len(overlaps),
            "output_file": str(lec_chunk_path.name)
        }

    total_processing_time = round(time.time() - t0_all, 4)

    # 3. Save all_chunks.json
    all_chunks_path = config.CHUNKS_DIR / "all_chunks.json"
    with open(all_chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved master all_chunks.json with {len(all_chunks)} chunks total.")

    # 4. Generate chunk_manifest.json
    chunk_manifest = {
        "dataset_name": "LearnMate AI Multilingual RAG Chunks",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_lectures": len(lectures),
        "total_chunks": len(all_chunks),
        "chunk_configuration": {
            "target_chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
            "max_chunk_size_tokens": config.MAX_CHUNK_SIZE_TOKENS,
            "target_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
            "gap_threshold_seconds": config.CHUNK_GAP_THRESHOLD_SECONDS,
            "tokenizer": "Whisper Multilingual BPE (Offline)"
        },
        "lectures": []
    }

    for lec in lectures:
        lec_id = lec["lecture_id"]
        stats = lecture_stats[lec_id]
        chunk_manifest["lectures"].append({
            "lecture_id": lec_id,
            "lecture_title": lec["lecture_title"],
            "tutorial_number": lec["tutorial_number"],
            "source_video": lec["video_filename"],
            "chunk_count": stats["total_chunks"],
            "chunks_file": f"data/chunks/{lec_id}_chunks.json",
            "coverage_percentage": stats["coverage_percentage"],
            "avg_tokens_per_chunk": stats["avg_tokens"]
        })

    chunk_manifest_path = config.CHUNKS_DIR / "chunk_manifest.json"
    with open(chunk_manifest_path, "w", encoding="utf-8") as f:
        json.dump(chunk_manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved {chunk_manifest_path.name}")

    # 5. Generate comprehensive chunk_statistics.json
    all_tokens = [c["token_count"] for c in all_chunks]
    all_durations = [c["end_time"] - c["start_time"] for c in all_chunks]
    all_chars = [c["character_count"] for c in all_chunks]

    # Validate duplicates and timestamps
    chunk_ids = [c["chunk_id"] for c in all_chunks]
    duplicate_chunks = len(chunk_ids) - len(set(chunk_ids))
    invalid_chunks = 0
    for c in all_chunks:
        if c["start_time"] > c["end_time"] or not c["text"].strip() or c["token_count"] <= 0:
            invalid_chunks += 1

    statistics = {
        "summary": {
            "total_lectures": len(lectures),
            "total_chunks": len(all_chunks),
            "total_processing_time_seconds": total_processing_time,
            "duplicate_chunk_count": duplicate_chunks,
            "invalid_chunk_count": invalid_chunks
        },
        "token_statistics": {
            "min_tokens_per_chunk": min(all_tokens) if all_tokens else 0,
            "max_tokens_per_chunk": max(all_tokens) if all_tokens else 0,
            "avg_tokens_per_chunk": round(sum(all_tokens) / len(all_tokens), 2) if all_tokens else 0,
            "total_tokens": sum(all_tokens)
        },
        "duration_statistics": {
            "min_duration_seconds": round(min(all_durations), 2) if all_durations else 0,
            "max_duration_seconds": round(max(all_durations), 2) if all_durations else 0,
            "avg_duration_seconds": round(sum(all_durations) / len(all_durations), 2) if all_durations else 0
        },
        "character_statistics": {
            "avg_characters_per_chunk": round(sum(all_chars) / len(all_chars), 2) if all_chars else 0,
            "total_characters": sum(all_chars)
        },
        "per_lecture_breakdown": lecture_stats
    }

    stats_path = config.CHUNKS_DIR / "chunk_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"Saved {stats_path.name}")

    # 6. Quality Validation Assertions
    print("\n--- [Step 5] Quality Validation Assertions ---")
    assert duplicate_chunks == 0, f"Found {duplicate_chunks} duplicate chunk IDs!"
    print(f"  ✓ 0 Duplicate chunk IDs (Total unique IDs: {len(set(chunk_ids))})")

    assert invalid_chunks == 0, f"Found {invalid_chunks} invalid chunks!"
    print(f"  ✓ 0 Invalid chunks (All chunks have non-empty text and valid monotonic timestamps)")

    for c in all_chunks:
        assert c["token_count"] <= config.MAX_CHUNK_SIZE_TOKENS, (
            f"Chunk {c['chunk_id']} exceeds max tokens ({c['token_count']} > {config.MAX_CHUNK_SIZE_TOKENS})!"
        )
    print(f"  ✓ All chunks strictly respect max token limit ({config.MAX_CHUNK_SIZE_TOKENS} tokens)")

    for lec_id, s in lecture_stats.items():
        assert s["coverage_percentage"] >= 99.0, (
            f"{lec_id} segment coverage too low: {s['coverage_percentage']}%"
        )
        print(f"  ✓ {lec_id} segment coverage: {s['coverage_percentage']}% ({s['covered_segments']}/{s['total_source_segments']})")

    # 7. Print Sample Chunks from each Lecture
    print("\n" + "=" * 70)
    print("REPRESENTATIVE CHUNKS FROM EACH LECTURE")
    print("=" * 70)

    for lec_id in ["LEC_01", "LEC_02", "LEC_03"]:
        lec_c = [c for c in all_chunks if c["lecture_id"] == lec_id]
        print(f"\n>>> {lec_id}: {lec_c[0]['lecture_title']} (Total: {len(lec_c)} chunks) <<<")
        for sample_idx in [0, min(1, len(lec_c)-1)]:
            c = lec_c[sample_idx]
            print(f"\n  Chunk ID: {c['chunk_id']} | Index: {c['chunk_index']}")
            print(f"  Timestamps: {c['start_timestamp']} -> {c['end_timestamp']} ({c['start_time']}s - {c['end_time']}s)")
            print(f"  Tokens: {c['token_count']} | Characters: {c['character_count']} | Segments: {c['source_segment_indices']}")
            print(f"  Text: {c['text'][:180]}...")


if __name__ == "__main__":
    main()

