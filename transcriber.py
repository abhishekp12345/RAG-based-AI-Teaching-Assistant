"""
Transcription Module for LearnMate AI.
Transcribes audio files into timestamped text segments using OpenAI Whisper.
Ensures FP16 is disabled on CPU, models and temporary caches are stored strictly on D: drive,
and output JSON preserves lecture titles, source filenames, detected language, and timestamped segments.
Supports Hindi, English, and mixed-language (Hinglish) educational lectures.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import config


class Transcriber:
    """Handles audio transcription using Whisper with segment-level timestamps and metadata."""

    def __init__(
        self,
        model_size: str = config.WHISPER_MODEL_SIZE,
        device: str = config.DEVICE,
        download_root: Optional[Path] = None
    ):
        self.model_size = model_size
        self.device = device
        self.download_root = Path(download_root or config.WHISPER_CACHE_DIR)
        self.download_root.mkdir(parents=True, exist_ok=True)
        self._model = None

    def load_model(self):
        """Lazy loader for Whisper model to save memory, strictly caching on D: drive."""
        if self._model is None:
            import whisper
            # Explicitly enforce download_root on D: drive so C: is untouched
            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=str(self.download_root)
            )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        output_json_path: Optional[Path] = None,
        lecture_title: Optional[str] = None,
        source_video_filename: Optional[str] = None,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio and return full transcript with timestamped segments and metadata.
        Uses FP16=False on CPU for numerical stability and zero warnings.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Derive title and filename if not provided
        derived_title = lecture_title or audio_path.stem
        derived_video_name = source_video_filename or f"{audio_path.stem}.webm"

        # Load model
        model = self.load_model()

        # Transcription options
        transcribe_options: Dict[str, Any] = {
            "fp16": False,  # CPU FP16 disabled
            "verbose": False,
        }

        if language:
            transcribe_options["language"] = language
        if initial_prompt:
            transcribe_options["initial_prompt"] = initial_prompt

        start_time = time.time()
        raw_result = model.transcribe(str(audio_path), **transcribe_options)
        elapsed_time = round(time.time() - start_time, 2)

        # Extract segments
        segments = []
        for seg in raw_result.get("segments", []):
            segments.append({
                "id": seg.get("id"),
                "seek": seg.get("seek"),
                "start": round(float(seg.get("start", 0.0)), 2),
                "end": round(float(seg.get("end", 0.0)), 2),
                "text": seg.get("text", "").strip(),
                "temperature": seg.get("temperature"),
                "avg_logprob": round(float(seg.get("avg_logprob", 0.0)), 4) if seg.get("avg_logprob") is not None else None,
                "compression_ratio": round(float(seg.get("compression_ratio", 0.0)), 4) if seg.get("compression_ratio") is not None else None,
                "no_speech_prob": round(float(seg.get("no_speech_prob", 0.0)), 4) if seg.get("no_speech_prob") is not None else None,
            })

        full_text = raw_result.get("text", "").strip()
        detected_language = raw_result.get("language", "unknown")

        # Determine audio duration from last segment or file
        audio_duration = round(segments[-1]["end"], 2) if segments else 0.0
        speed_factor = round(audio_duration / elapsed_time, 2) if elapsed_time > 0 else 0.0

        structured_transcript: Dict[str, Any] = {
            "lecture_title": derived_title,
            "source_video_filename": derived_video_name,
            "audio_file": audio_path.name,
            "whisper_model": f"whisper-{self.model_size}",
            "device": self.device,
            "fp16": False,
            "detected_language": detected_language,
            "duration_seconds": audio_duration,
            "transcription_time_seconds": elapsed_time,
            "speed_factor_x": speed_factor,
            "total_segments": len(segments),
            "full_transcript": full_text,
            "segments": segments
        }

        if output_json_path:
            out_path = Path(output_json_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(structured_transcript, f, ensure_ascii=False, indent=2)

        return structured_transcript
