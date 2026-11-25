"""
Video Processing Module for LearnMate AI.
Handles video inspection, metadata extraction, audio extraction using FFmpeg, and format verification.
All outputs and temporary files are stored on D: drive.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import config


class VideoProcessor:
    """Manages video file operations and audio extraction using FFmpeg."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def check_ffmpeg(self) -> bool:
        """Verify FFmpeg binary is accessible."""
        try:
            res = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                check=False
            )
            return res.returncode == 0
        except Exception:
            return False

    def check_ffprobe(self) -> bool:
        """Verify FFprobe binary is accessible."""
        try:
            res = subprocess.run(
                [self.ffprobe_path, "-version"],
                capture_output=True,
                text=True,
                check=False
            )
            return res.returncode == 0
        except Exception:
            return False

    def list_videos(self, video_dir: Optional[Path] = None) -> List[Path]:
        """List all video files in the specified directory."""
        vdir = video_dir or config.VIDEOS_DIR
        valid_extensions = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
        if not vdir.exists():
            return []
        return sorted([p for p in vdir.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions])

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Inspect video metadata using FFprobe.
        Returns duration, size, format, video/audio stream details.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed for {video_path.name}: {result.stderr}")

        data = json.loads(result.stdout)
        format_info = data.get("format", {})
        streams = data.get("streams", [])

        duration_sec = float(format_info.get("duration", 0.0))
        file_size_bytes = int(format_info.get("size", video_path.stat().st_size))

        # Format duration as HH:MM:SS
        hours = int(duration_sec // 3600)
        minutes = int((duration_sec % 3600) // 60)
        seconds = int(duration_sec % 60)
        duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Analyze audio and video streams
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

        return {
            "file_name": video_path.name,
            "file_path": str(video_path),
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "duration_seconds": round(duration_sec, 2),
            "duration_formatted": duration_formatted,
            "format_name": format_info.get("format_name", ""),
            "has_audio": audio_stream is not None,
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
            "audio_sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else None,
            "audio_channels": int(audio_stream.get("channels", 0)) if audio_stream else None,
            "video_codec": video_stream.get("codec_name") if video_stream else None,
            "resolution": f"{video_stream.get('width')}x{video_stream.get('height')}" if video_stream else None,
        }

    def extract_audio(
        self,
        video_path: Path,
        output_audio_path: Optional[Path] = None,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Path:
        """
        Extract audio track from a video file into WAV format optimized for Whisper (16kHz mono).
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_audio_path is None:
            config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            output_audio_path = config.AUDIO_DIR / f"{video_path.stem}.wav"
        else:
            output_audio_path = Path(output_audio_path)
            output_audio_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output
            "-i", str(video_path),
            "-vn",  # Disable video stream
            "-acodec", "pcm_s16le",  # 16-bit PCM WAV
            "-ar", str(sample_rate),  # Sample rate (default 16000 Hz)
            "-ac", str(channels),  # Channels (default mono 1)
            str(output_audio_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

        return output_audio_path

    def extract_audio_sample(
        self,
        audio_or_video_path: Path,
        output_sample_path: Path,
        start_sec: float = 0.0,
        duration_sec: float = 120.0,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Path:
        """
        Extract a sample segment (e.g. 60-120 seconds) for pilot testing.
        """
        input_path = Path(audio_or_video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_sample_path = Path(output_sample_path)
        output_sample_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-i", str(input_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            str(output_sample_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg sample extraction failed: {result.stderr}")

        return output_sample_path

    def get_audio_info(self, audio_path: Path) -> Dict[str, Any]:
        """
        Inspect extracted WAV audio metadata using FFprobe to verify 16kHz mono.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            str(audio_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed for {audio_path.name}: {result.stderr}")

        data = json.loads(result.stdout)
        format_info = data.get("format", {})
        streams = data.get("streams", [])
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

        duration_sec = float(format_info.get("duration", 0.0))
        file_size_bytes = int(format_info.get("size", audio_path.stat().st_size))

        return {
            "file_name": audio_path.name,
            "file_path": str(audio_path),
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "duration_seconds": round(duration_sec, 2),
            "sample_rate": int(audio_stream.get("sample_rate", 0)),
            "channels": int(audio_stream.get("channels", 0)),
            "codec_name": audio_stream.get("codec_name", ""),
        }
