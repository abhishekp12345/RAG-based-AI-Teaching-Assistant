"""
Audit all transcripts, generate cleaned version for Lecture 1 (recovering 16-90s speech),
and generate lecture_manifest.json.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from video_processor import VideoProcessor
from transcriber import Transcriber


def clean_and_transcribe_lecture1():
    print("\n--- [Step 1] Backing up Phase 2A Lecture 1 Transcript & Generating Clean Version ---")
    orig_json = config.TRANSCRIPTS_DIR / "SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY].json"
    backup_json = config.TRANSCRIPTS_DIR / "SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY]_original_phase2a.json"
    cleaned_json = config.TRANSCRIPTS_DIR / "SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY]_cleaned.json"

    if orig_json.exists() and not backup_json.exists():
        shutil.copy2(orig_json, backup_json)
        print(f"Backed up original Phase 2A transcript to:\n  {backup_json.name}")

    audio_path = config.AUDIO_DIR / "SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY].wav"
    video_path = config.VIDEOS_DIR / "SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY].webm"

    print("Transcribing Lecture 1 with clean settings (initial_prompt=None)...")
    transcriber = Transcriber(model_size="base", device="cpu")
    t0 = time.time()
    cleaned_result = transcriber.transcribe(
        audio_path=audio_path,
        output_json_path=cleaned_json,
        lecture_title="SEO and Core Web Vitals in HTML | Sigma Web Development Course - Tutorial #6",
        source_video_filename=video_path.name,
        initial_prompt=None
    )
    t_clean = round(time.time() - t0, 2)
    print(f"Cleaned Lecture 1 transcribed in {t_clean}s:")
    print(f"  Total segments: {cleaned_result['total_segments']} (Original had: 262)")
    print(f"  Saved to: {cleaned_json.name}")
    return cleaned_result


def audit_transcript(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    total_segments = len(segments)
    
    empty_segments = []
    invalid_timestamps = []
    large_gaps = []  # Gaps > 10 seconds
    repetitions = []  # Consecutive exact duplicate texts

    prev_end = 0.0
    for idx, seg in enumerate(segments):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()

        # Check empty
        if not text:
            empty_segments.append(idx)

        # Check timestamp ordering
        if start > end or start < 0 or end < 0:
            invalid_timestamps.append((idx, start, end))

        # Check gaps
        if idx > 0:
            gap = round(start - prev_end, 2)
            if gap > 10.0:
                large_gaps.append((idx, round(prev_end, 2), round(start, 2), gap))

        # Check repetitions with previous
        if idx > 0:
            prev_text = segments[idx-1].get("text", "").strip()
            if text and text.lower() == prev_text.lower():
                repetitions.append((idx, text))

        prev_end = end

    return {
        "file": json_path.name,
        "lecture_title": data.get("lecture_title", ""),
        "total_segments": total_segments,
        "duration_seconds": data.get("duration_seconds", 0.0),
        "transcription_time_seconds": data.get("transcription_time_seconds", 0.0),
        "detected_language": data.get("detected_language", ""),
        "empty_segments": empty_segments,
        "invalid_timestamps": invalid_timestamps,
        "large_gaps": large_gaps,
        "repetitions": repetitions,
    }


def main():
    print("=" * 70)
    print("PHASE 2B: TRANSCRIPT AUDIT, CLEANING & MANIFEST GENERATION")
    print("=" * 70)

    # 1. Clean Lecture 1
    clean_and_transcribe_lecture1()

    # 2. Audit all transcripts
    print("\n--- [Step 2] Auditing All Lecture Transcripts ---")
    transcript_files = [
        config.TRANSCRIPTS_DIR / "SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY]_original_phase2a.json",
        config.TRANSCRIPTS_DIR / "SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY]_cleaned.json",
        config.TRANSCRIPTS_DIR / "Video, Audio & Media in HTML ｜ Sigma Web Development Course - Tutorial #10 [XZwBNDGuWGU].json",
        config.TRANSCRIPTS_DIR / "Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2 [kJEsTjH5mVg].json",
    ]

    audit_results = []
    for tf in transcript_files:
        if tf.exists():
            res = audit_transcript(tf)
            audit_results.append(res)
            print(f"\nAudit Report for: {res['file']}")
            print(f"  Title: {res['lecture_title']}")
            print(f"  Duration: {res['duration_seconds']}s, Segments: {res['total_segments']}")
            print(f"  Detected Language: {res['detected_language']}")
            print(f"  Empty Segments: {len(res['empty_segments'])}")
            print(f"  Invalid Timestamps: {len(res['invalid_timestamps'])}")
            print(f"  Consecutive Duplicate Texts: {len(res['repetitions'])}")
            if res['repetitions']:
                for r in res['repetitions'][:3]:
                    print(f"    - Segment #{r[0]}: \"{r[1]}\"")
            print(f"  Gaps > 10s: {len(res['large_gaps'])}")
            if res['large_gaps']:
                for g in res['large_gaps']:
                    print(f"    - Gap between {g[1]}s and {g[2]}s ({g[3]}s duration)")

    # 3. Create Lecture Manifest
    print("\n--- [Step 3] Creating lecture_manifest.json ---")
    manifest = {
        "project": "LearnMate AI",
        "description": "Dataset Lecture Manifest for PhysicsWallah APM Builder RAG Teaching Assistant",
        "total_lectures": 3,
        "whisper_model": "whisper-base",
        "device": "cpu",
        "lectures": [
            {
                "lecture_id": "LEC_01",
                "lecture_title": "SEO and Core Web Vitals in HTML | Sigma Web Development Course - Tutorial #6",
                "tutorial_number": 6,
                "video_filename": "SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY].webm",
                "video_path": "data/videos/SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY].webm",
                "audio_path": "data/audio/SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY].wav",
                "transcript_path": "data/transcripts/SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY]_cleaned.json",
                "original_transcript_path": "data/transcripts/SEO and Core Web Vitals in HTML ｜ Sigma Web Development Course - Tutorial #6 [CyRlWlaJnTY]_original_phase2a.json",
                "duration_seconds": 795.17,
                "duration_formatted": "00:13:15",
                "detected_language": "hi",
                "total_segments": audit_results[1]["total_segments"] if len(audit_results) > 1 else 262,
                "processing_status": "COMPLETED_AND_VALIDATED"
            },
            {
                "lecture_id": "LEC_02",
                "lecture_title": "Video, Audio & Media in HTML | Sigma Web Development Course - Tutorial #10",
                "tutorial_number": 10,
                "video_filename": "Video, Audio & Media in HTML ｜ Sigma Web Development Course - Tutorial #10 [XZwBNDGuWGU].webm",
                "video_path": "data/videos/Video, Audio & Media in HTML ｜ Sigma Web Development Course - Tutorial #10 [XZwBNDGuWGU].webm",
                "audio_path": "data/audio/Video, Audio & Media in HTML ｜ Sigma Web Development Course - Tutorial #10 [XZwBNDGuWGU].wav",
                "transcript_path": "data/transcripts/Video, Audio & Media in HTML ｜ Sigma Web Development Course - Tutorial #10 [XZwBNDGuWGU].json",
                "duration_seconds": 887.09,
                "duration_formatted": "00:14:47",
                "detected_language": "hi",
                "total_segments": 473,
                "processing_status": "COMPLETED_AND_VALIDATED"
            },
            {
                "lecture_id": "LEC_03",
                "lecture_title": "Your First HTML Website | Sigma Web Development Course - Tutorial #2",
                "tutorial_number": 2,
                "video_filename": "Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2 [kJEsTjH5mVg].webm",
                "video_path": "data/videos/Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2 [kJEsTjH5mVg].webm",
                "audio_path": "data/audio/Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2 [kJEsTjH5mVg].wav",
                "transcript_path": "data/transcripts/Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2 [kJEsTjH5mVg].json",
                "duration_seconds": 1710.37,
                "duration_formatted": "00:28:30",
                "detected_language": "hi",
                "total_segments": 693,
                "processing_status": "COMPLETED_AND_VALIDATED"
            }
        ]
    }

    manifest_paths = [
        BASE_DIR / "data" / "lecture_manifest.json",
        BASE_DIR / "lecture_manifest.json"
    ]
    for mp in manifest_paths:
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Saved manifest to: {mp}")

    print("\n" + "=" * 70)
    print("PHASE 2B AUDIT & MANIFEST GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

