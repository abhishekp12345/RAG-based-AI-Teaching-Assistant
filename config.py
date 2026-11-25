"""
Configuration Module for LearnMate AI.
Loads settings from environment variables and sets global project paths on D: drive.
Ensures all large caches (HuggingFace, PyTorch, Whisper, pip) reside on D: drive.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Workspace Directory on D: Drive
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

# Cache & Temp Directories on D: Drive
CACHE_DIR = BASE_DIR / ".cache"
HF_HOME = Path(os.getenv("HF_HOME", str(CACHE_DIR / "huggingface")))
TORCH_HOME = Path(os.getenv("TORCH_HOME", str(CACHE_DIR / "torch")))
WHISPER_CACHE_DIR = Path(os.getenv("WHISPER_CACHE_DIR", str(CACHE_DIR / "whisper")))
SENTENCE_TRANSFORMERS_HOME = Path(os.getenv("SENTENCE_TRANSFORMERS_HOME", str(CACHE_DIR / "sentence_transformers")))
TMP_DIR = Path(os.getenv("TMPDIR", str(CACHE_DIR / "tmp")))

# Set environment variables for external ML libraries and temp storage
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["TORCH_HOME"] = str(TORCH_HOME)
os.environ["WHISPER_CACHE_DIR"] = str(WHISPER_CACHE_DIR)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(SENTENCE_TRANSFORMERS_HOME)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "0")
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TMPDIR"] = str(TMP_DIR)

# Set python tempfile directory
import tempfile
tempfile.tempdir = str(TMP_DIR)

# Ensure cache and temp directories exist on D:
for cdir in [CACHE_DIR, HF_HOME, TORCH_HOME, WHISPER_CACHE_DIR, SENTENCE_TRANSFORMERS_HOME, TMP_DIR]:
    cdir.mkdir(parents=True, exist_ok=True)

# Data Directories
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
VIDEOS_DIR = Path(os.getenv("VIDEOS_DIR", str(DATA_DIR / "videos")))
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", str(DATA_DIR / "audio")))
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", str(DATA_DIR / "transcripts")))
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", str(DATA_DIR / "chunks")))
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", str(DATA_DIR / "vector_store")))
FEEDBACK_DIR = Path(os.getenv("FEEDBACK_DIR", str(DATA_DIR / "feedback")))
NOTES_DIR = Path(os.getenv("NOTES_DIR", str(DATA_DIR / "notes")))
QUIZZES_DIR = Path(os.getenv("QUIZZES_DIR", str(DATA_DIR / "quizzes")))

# Ensure critical data directories exist
for directory in [DATA_DIR, VIDEOS_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, CHUNKS_DIR, VECTOR_STORE_DIR, FEEDBACK_DIR, NOTES_DIR, QUIZZES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Google Gemini Configuration (Primary / Free Tier)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash")
ENABLE_MODEL_FALLBACK = os.getenv("ENABLE_MODEL_FALLBACK", "true").lower() in ("true", "1", "yes")

# OpenAI API Configuration (Optional Inactive Provider)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Shared LLM Parameters
LLM_MODE = os.getenv("LLM_MODE", "mock").lower()
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", os.getenv("OPENAI_TIMEOUT", 30.0)))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", os.getenv("OPENAI_MAX_OUTPUT_TOKENS", 1500)))

# Backward compatibility aliases
OPENAI_TIMEOUT = LLM_TIMEOUT
OPENAI_MAX_OUTPUT_TOKENS = LLM_MAX_OUTPUT_TOKENS

# Speech-to-Text (Whisper) Configuration
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 1))
EMBEDDING_MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", 512))
EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
EMBEDDING_NORMALIZE = True

# Vector Store Configuration
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "auto")

# RAG & Chunking Parameters
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", 300))
MAX_CHUNK_SIZE_TOKENS = int(os.getenv("MAX_CHUNK_SIZE_TOKENS", 400))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", 50))
CHUNK_GAP_THRESHOLD_SECONDS = float(os.getenv("CHUNK_GAP_THRESHOLD_SECONDS", 15.0))
RETRIEVAL_TOP_K = int(os.getenv("RAG_TOP_K", os.getenv("RETRIEVAL_TOP_K", 10)))
RAG_CANDIDATE_TOP_K = int(os.getenv("RAG_CANDIDATE_TOP_K", 10))
RAG_MAX_CONTEXT_CHUNKS = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", 6))
RAG_MAX_CONTEXT_TOKENS = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", 1600))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))

# System Hardware / Execution Settings
DEVICE = "cpu"  # Target CPU architecture
