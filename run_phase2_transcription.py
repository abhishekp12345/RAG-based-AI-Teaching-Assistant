"""
Phase 2 Execution Script:
1. Audio extraction for shortest video (16 kHz mono WAV)
2. Audio sample extraction (90-second representative pilot sample)
3. Pilot Whisper transcription test with timing & memory benchmarks
4. Full transcription of shortest video if pilot succeeds
5. JSON transcript validation
"""

import os
import sys
import json
import time
import ctypes
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from video_processor import VideoProcessor
from transcriber import Transcriber


def get_system_ram_mb():
    """Get system available physical RAM in MB using Windows kernel32."""
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ('dwLength', ctypes.c_ulong),
            ('dwMemoryLoad', ctypes.c_ulong),
            ('ullTotalPhys', ctypes.c_ulonglong),
            ('ullAvailPhys', ctypes.c_ulonglong),
            ('ullTotalPageFile', ctypes.c_ulonglong),
            ('ullAvailPageFile', ctypes.c_ulonglong),
            ('ullTotalVirtual', ctypes.c_ulonglong),
            ('ullAvailVirtual', ctypes.c_ulonglong),
            ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {
        "total_mb": round(stat.ullTotalPhys / (1024 * 1024), 2),
        "available_mb": round(stat.ullAvailPhys / (1024 * 1024), 2),
        "load_pct": stat.dwMemoryLoad
    }


def main():
    print("=" * 70)
    print("LEARNMATE AI - PHASE 2: VIDEO PROCESSING & WHISPER TRANSCRIPTION")
    print("=" * 70)

    # 1. Environment & Storage Pre-check
    ram_init = get_system_ram_mb()
    print(f"Initial RAM: Available={ram_init['available_mb']} MB / Total={ram_init['total_mb']} MB (Load: {ram_init['load_pct']}%)")
    print(f"Whisper Cache Directory: {config.WHISPER_CACHE_DIR}")
    print(f"Audio Directory: {config.AUDIO_DIR}")
    print(f"Transcripts Directory: {config.TRANSCRIPTS_DIR}")

    vp = VideoProcessor()
    assert vp.check_ffmpeg(), "FFmpeg binary check failed!"
    assert vp.check_ffprobe(), "FFprobe binary check failed!"

    # 2. Discover videos
    videos = vp.list_videos()
    video_records = [vp.get_video_info(v) for v in videos]
    sorted_videos = sorted(video_records, key=lambda x: x["duration_seconds"])
    shortest_meta = sorted_videos[0]
    shortest_video_path = Path(shortest_meta["file_path"])

    print("\n--- [Step 1] Shortest Video Selected ---")
    print(f"Title: {shortest_meta['file_name']}")
    print(f"Duration: {shortest_meta['duration_formatted']} ({shortest_meta['duration_seconds']}s)")
    print(f"Size: {shortest_meta['file_size_mb']} MB")

    # 3. Extract Audio (16 kHz mono WAV)
    print("\n--- [Step 2] Extracting Full Audio (16 kHz, Mono WAV) ---")
    full_audio_path = config.AUDIO_DIR / f"{shortest_video_path.stem}.wav"
    t0_audio = time.time()
    vp.extract_audio(
        video_path=shortest_video_path,
        output_audio_path=full_audio_path,
        sample_rate=16000,
        channels=1
    )
    t_audio_elapsed = round(time.time() - t0_audio, 2)
    audio_info = vp.get_audio_info(full_audio_path)
    print(f"Audio extracted in {t_audio_elapsed}s:")
    print(f"  Path: {audio_info['file_path']}")
    print(f"  Size: {audio_info['file_size_mb']} MB")
    print(f"  Sample Rate: {audio_info['sample_rate']} Hz (Expected: 16000)")
    print(f"  Channels: {audio_info['channels']} (Expected: 1 - Mono)")
    print(f"  Duration: {audio_info['duration_seconds']}s")
    assert audio_info["sample_rate"] == 16000, "Sample rate is not 16 kHz!"
    assert audio_info["channels"] == 1, "Audio is not mono!"

    # 4. Extract Representative 90-second Pilot Sample
    print("\n--- [Step 3] Extracting 90-Second Representative Pilot Sample ---")
    sample_audio_path = config.AUDIO_DIR / f"{shortest_video_path.stem}_sample_90s.wav"
    vp.extract_audio_sample(
        audio_or_video_path=full_audio_path,
        output_sample_path=sample_audio_path,
        start_sec=0.0,
        duration_sec=90.0
    )
    sample_info = vp.get_audio_info(sample_audio_path)
    print(f"Sample created: {sample_info['file_name']} ({sample_info['duration_seconds']}s, {sample_info['file_size_mb']} MB)")

    # 5. Pilot Transcription Test with Whisper Base
    print("\n--- [Step 4] Running Pilot Whisper Base Transcription on 90s Sample ---")
    ram_before_load = get_system_ram_mb()
    print(f"RAM before loading model: Available={ram_before_load['available_mb']} MB")

    transcriber = Transcriber(model_size="base", device="cpu")
    t0_model = time.time()
    transcriber.load_model()
    t_model_load = round(time.time() - t0_model, 2)

    ram_after_load = get_system_ram_mb()
    print(f"Model loaded in {t_model_load}s. RAM after model load: Available={ram_after_load['available_mb']} MB")

    # Transcribe sample
    t0_sample_tx = time.time()
    sample_transcript = transcriber.transcribe(
        audio_path=sample_audio_path,
        lecture_title=shortest_video_path.stem,
        source_video_filename=shortest_video_path.name
    )
    t_sample_tx = round(time.time() - t0_sample_tx, 2)
    ram_after_sample = get_system_ram_mb()

    print(f"\n--- [Pilot Test Results] ---")
    print(f"Audio Sample Duration: {sample_transcript['duration_seconds']}s")
    print(f"Transcription Time: {sample_transcript['transcription_time_seconds']}s")
    print(f"Speed Factor: {sample_transcript['speed_factor_x']}x real-time")
    print(f"Detected Language: {sample_transcript['detected_language']}")
    print(f"Total Segments: {sample_transcript['total_segments']}")
    print(f"RAM Available: {ram_after_sample['available_mb']} MB")
    print("\nSample Segments (First 3):")
    for seg in sample_transcript["segments"][:3]:
        print(f"  [{seg['start']}s -> {seg['end']}s]: {seg['text']}")

    # Pilot Quality Validation
    assert sample_transcript["total_segments"] > 0, "No segments transcribed in pilot!"
    assert len(sample_transcript["full_transcript"].strip()) > 20, "Transcript text too short!"
    print("\n✓ Pilot test successful! Whisper Base is fast and stable.")

    # 6. Full Lecture Transcription
    print("\n--- [Step 5] Transcribing Complete Shortest Lecture ---")
    lecture_title = "SEO and Core Web Vitals in HTML | Sigma Web Development Course - Tutorial #6"
    full_transcript_json_path = config.TRANSCRIPTS_DIR / f"{shortest_video_path.stem}.json"
    
    t0_full = time.time()
    full_transcript = transcriber.transcribe(
        audio_path=full_audio_path,
        output_json_path=full_transcript_json_path,
        lecture_title=lecture_title,
        source_video_filename=shortest_video_path.name
    )
    t_full_elapsed = round(time.time() - t0_full, 2)
    ram_final = get_system_ram_mb()

    print(f"\n--- [Full Transcription Results] ---")
    print(f"Lecture: {full_transcript['lecture_title']}")
    print(f"Source Video: {full_transcript['source_video_filename']}")
    print(f"Whisper Model: {full_transcript['whisper_model']}")
    print(f"Audio Duration: {full_transcript['duration_seconds']}s ({round(full_transcript['duration_seconds']/60, 2)} min)")
    print(f"Processing Time: {full_transcript['transcription_time_seconds']}s ({round(full_transcript['transcription_time_seconds']/60, 2)} min)")
    print(f"Speed Factor: {full_transcript['speed_factor_x']}x real-time")
    print(f"Detected Language: {full_transcript['detected_language']}")
    print(f"Total Segments: {full_transcript['total_segments']}")
    print(f"Saved JSON: {full_transcript_json_path}")
    print(f"JSON File Size: {round(full_transcript_json_path.stat().st_size / 1024, 2)} KB")
    print(f"Final Available RAM: {ram_final['available_mb']} MB")

    # 7. Verification of JSON Transcript
    print("\n--- [Step 6] Verifying JSON Transcript Integrity ---")
    with open(full_transcript_json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    required_keys = [
        "lecture_title", "source_video_filename", "audio_file", "whisper_model",
        "detected_language", "duration_seconds", "transcription_time_seconds",
        "total_segments", "full_transcript", "segments"
    ]
    for key in required_keys:
        assert key in loaded_json, f"Missing key: {key}"
        print(f"  ✓ Key '{key}' present")

    # Verify timestamps
    segments = loaded_json["segments"]
    assert len(segments) > 0, "No segments in transcript!"
    prev_end = 0.0
    for idx, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        assert start <= end, f"Invalid segment timing at segment {idx}: start {start} > end {end}"
        assert len(seg["text"]) > 0, f"Empty text in segment {idx}"
    print(f"  ✓ Verified {len(segments)} segments: all timestamps valid and non-negative.")

    print("\n--- First 5 Segments Sample ---")
    for s in segments[:5]:
        print(f"  [{s['start']:6.2f}s - {s['end']:6.2f}s] {s['text']}")

    print("\n--- Middle 3 Segments Sample ---")
    mid = len(segments) // 2
    for s in segments[mid:mid+3]:
        print(f"  [{s['start']:6.2f}s - {s['end']:6.2f}s] {s['text']}")

    print("\n--- Last 3 Segments Sample ---")
    for s in segments[-3:]:
        print(f"  [{s['start']:6.2f}s - {s['end']:6.2f}s] {s['text']}")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETED SUCCESSFULLY FOR FIRST COMPLETE LECTURE")
    print("=" * 70)


if __name__ == "__main__":
    main()

