"""
Phase 1 Setup and Verification Test Suite.
Verifies hardware, virtual environment, FFmpeg, and Python package imports.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_python_version():
    """Verify Python version is 3.10+"""
    version = sys.version_info
    print(f"[TEST 1] Python Version: {version.major}.{version.minor}.{version.micro}")
    assert version.major == 3 and version.minor >= 10, "Python version must be >= 3.10"
    return True

def test_directories():
    """Verify all project directories exist."""
    print("[TEST 2] Verifying Project Directories...")
    import config
    required_dirs = [
        config.DATA_DIR,
        config.VIDEOS_DIR,
        config.AUDIO_DIR,
        config.TRANSCRIPTS_DIR,
        config.CHUNKS_DIR,
        config.VECTOR_STORE_DIR,
        config.FEEDBACK_DIR
    ]
    for d in required_dirs:
        assert d.exists(), f"Directory missing: {d}"
        print(f"  ✓ {d.relative_to(config.BASE_DIR)}")
    return True

def test_videos_extracted():
    """Verify videos exist in data/videos/."""
    print("[TEST 3] Verifying Extracted Educational Videos...")
    import config
    videos = list(config.VIDEOS_DIR.glob("*.*"))
    assert len(videos) >= 3, f"Expected at least 3 videos, found {len(videos)}"
    for v in videos:
        print(f"  ✓ {v.name} ({v.stat().st_size / (1024**2):.2f} MB)")
    return True

def test_ffmpeg():
    """Verify FFmpeg binary."""
    print("[TEST 4] Verifying FFmpeg Installation...")
    from video_processor import VideoProcessor
    vp = VideoProcessor()
    assert vp.check_ffmpeg(), "FFmpeg binary is not accessible on PATH"
    print("  ✓ FFmpeg is accessible and working.")
    return True

def test_core_imports():
    """Verify core imports."""
    print("[TEST 5] Verifying Core Library Imports...")
    
    import numpy as np
    print(f"  ✓ numpy {np.__version__}")
    
    import pandas as pd
    print(f"  ✓ pandas {pd.__version__}")
    
    import streamlit as st
    print(f"  ✓ streamlit {st.__version__}")
    
    import dotenv
    print("  ✓ python-dotenv loaded")
    
    return True

def test_ai_imports():
    """Verify AI/ML library imports."""
    print("[TEST 6] Verifying AI/ML Library Imports...")
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False

    try:
        import whisper
        print("  ✓ openai-whisper loaded")
    except ImportError as e:
        print(f"  ✗ whisper: {e}")
        return False

    try:
        import sentence_transformers
        print(f"  ✓ sentence-transformers {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ sentence-transformers: {e}")
        return False

    try:
        import faiss
        print(f"  ✓ faiss {faiss.__version__}")
    except ImportError as e:
        print(f"  ⚠️ faiss not installed or incompatible ({e}). Using NumPy fallback as designed.")

    try:
        import openai
        print(f"  ✓ openai {openai.__version__}")
    except ImportError as e:
        print(f"  ✗ openai: {e}")
        return False

    return True

def test_project_modules():
    """Verify all internal modules can be imported."""
    print("[TEST 7] Verifying Internal Project Modules...")
    import config
    import prompts
    import video_processor
    import transcriber
    import chunker
    import embeddings
    import vector_store
    import llm
    import rag_pipeline
    import summarizer
    import quiz_generator
    import feedback
    import evaluate
    print("  ✓ All 13 internal project modules successfully imported.")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("LearnMate AI - Phase 1 Verification Suite")
    print("=" * 60)
    
    success = True
    for test_fn in [
        test_python_version,
        test_directories,
        test_videos_extracted,
        test_ffmpeg,
        test_project_modules
    ]:
        try:
            test_fn()
        except Exception as ex:
            print(f"  ✗ FAILED: {ex}")
            success = False
            
    print("\nAttempting dependency checks (may require pip install if not run yet)...")
    try:
        test_core_imports()
    except Exception as ex:
        print(f"  Core imports not ready: {ex}")
        
    try:
        test_ai_imports()
    except Exception as ex:
        print(f"  AI imports not ready: {ex}")
        
    print("=" * 60)

