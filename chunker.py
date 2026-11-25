"""
Chunking and Text Preprocessing Module for LearnMate AI.
Segments timestamped Whisper transcripts into overlapping, retrieval-optimized chunks
preserving lecture metadata, segment indices, and precise start/end timestamps.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import config


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds into standard HH:MM:SS format."""
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class MultilingualTokenizer:
    """
    Lightweight, offline tokenizer for token counting in multilingual educational text.
    Uses Whisper's offline multilingual BPE tokenizer without downloading models or loading neural weights.
    Falls back gracefully to regex-based multilingual subword token estimation if needed.
    """

    def __init__(self):
        self._tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            import whisper.tokenizer
            self._tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=True,
                language="hi",
                task="transcribe"
            )
        except Exception:
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text accurately and efficiently."""
        if not text:
            return 0
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        # Multilingual fallback estimation:
        # English words: ~1.3 tokens/word.
        # Devanagari / Indic script characters: ~0.8-1.0 tokens per syllable/character cluster.
        words = text.split()
        indic_chars = len(re.findall(r"[\u0900-\u097F\u0600-\u06FF]", text))
        latin_words = len([w for w in words if re.search(r"[a-zA-Z0-9]", w)])
        est = int(latin_words * 1.3) + int(indic_chars * 0.75)
        return max(len(words), est)


@dataclass
class PreprocessedSegment:
    """A clean, preprocessed transcript segment with valid timing."""
    id: int
    start: float
    end: float
    text: str
    token_count: int
    has_repetition_flag: bool = False


@dataclass
class TranscriptChunk:
    """
    A self-contained, timestamp-aware chunk ready for RAG embedding and retrieval.
    Preserves comprehensive lecture provenance and source segment indices.
    """
    chunk_id: str
    lecture_id: str
    lecture_title: str
    tutorial_number: int
    source_video: str
    start_time: float
    end_time: float
    start_timestamp: str
    end_timestamp: str
    text: str
    token_count: int
    character_count: int
    detected_language: str
    source_transcript_path: str
    source_segment_indices: List[int]
    chunk_index: int
    has_gap_flag: bool = False
    has_repetition_flag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextPreprocessor:
    """
    Conservative preprocessor for multilingual educational transcripts (Hindi, English, Hinglish).
    Preserves programming syntax, HTML tags, and technical terms while normalizing whitespace.
    """

    # Detect excessive repeating phrases (e.g. repeated 4+ times)
    REPETITION_PATTERN = re.compile(r"(\b.+?\b)(?:\s+\1){3,}", re.IGNORECASE)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalize whitespace and invisible characters while preserving
        case, punctuation, HTML tags, and technical terms.
        """
        if not text:
            return ""

        # Remove zero-width characters and BOM
        cleaned = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)

        # Normalize line breaks and tabs to spaces
        cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)

        # Collapse multiple spaces into a single space
        cleaned = re.sub(r" +", " ", cleaned)

        return cleaned.strip()

    @classmethod
    def check_suspicious_repetition(cls, text: str) -> bool:
        """Check if text contains suspicious repetitive loops."""
        if not text:
            return False
        return bool(cls.REPETITION_PATTERN.search(text))

    @classmethod
    def preprocess_segments(
        cls,
        raw_segments: List[Dict[str, Any]],
        tokenizer: MultilingualTokenizer
    ) -> List[PreprocessedSegment]:
        """
        Clean, validate, and tokenize raw transcript segments.
        Filters out empty or invalid segments.
        """
        valid_segments: List[PreprocessedSegment] = []

        for seg in raw_segments:
            raw_text = seg.get("text", "")
            cleaned = cls.clean_text(raw_text)

            if not cleaned:
                continue

            start = round(float(seg.get("start", 0.0)), 2)
            end = round(float(seg.get("end", 0.0)), 2)

            # Ensure valid non-negative monotonic timestamps
            if start < 0 or end < start:
                continue

            has_rep = cls.check_suspicious_repetition(cleaned)
            token_count = tokenizer.count_tokens(cleaned)

            valid_segments.append(
                PreprocessedSegment(
                    id=int(seg.get("id", len(valid_segments))),
                    start=start,
                    end=end,
                    text=cleaned,
                    token_count=token_count,
                    has_repetition_flag=has_rep
                )
            )

        return valid_segments


class TranscriptChunker:
    """
    Timestamp-aware, segment-level overlapping chunker for educational transcripts.
    Guarantees strict token bounds, preserves natural boundaries and timestamps,
    and prevents spanning across large unexplained silence gaps.
    """

    def __init__(
        self,
        chunk_size_tokens: int = config.CHUNK_SIZE_TOKENS,
        max_chunk_size_tokens: int = config.MAX_CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
        gap_threshold_seconds: float = config.CHUNK_GAP_THRESHOLD_SECONDS,
        tokenizer: Optional[MultilingualTokenizer] = None
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.max_chunk_size_tokens = max_chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.gap_threshold_seconds = gap_threshold_seconds
        self.tokenizer = tokenizer or MultilingualTokenizer()
        self.preprocessor = TextPreprocessor()

    def _split_oversized_segment(
        self,
        seg: PreprocessedSegment
    ) -> List[PreprocessedSegment]:
        """
        Split a single segment that exceeds max_chunk_size_tokens on sentence/clause boundaries.
        Proportionally distributes start and end timestamps.
        """
        text = seg.text
        # Split on sentence terminals or clause markers
        parts = [p.strip() for p in re.split(r"(?<=[.!?।])\s+", text) if p.strip()]
        if len(parts) <= 1:
            # Fallback to comma/clause split
            parts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", text) if p.strip()]
        if len(parts) <= 1:
            # Fallback to word chunks
            words = text.split()
            chunk_word_size = max(1, len(words) // ((seg.token_count // self.chunk_size_tokens) + 1))
            parts = [" ".join(words[i:i+chunk_word_size]) for i in range(0, len(words), chunk_word_size)]

        sub_segments: List[PreprocessedSegment] = []
        total_len = len(text)
        current_char = 0
        seg_duration = seg.end - seg.start

        for idx, part in enumerate(parts):
            p_len = len(part)
            sub_start = round(seg.start + (current_char / total_len) * seg_duration, 2)
            sub_end = round(seg.start + ((current_char + p_len) / total_len) * seg_duration, 2)
            current_char += p_len + 1
            tokens = self.tokenizer.count_tokens(part)

            sub_segments.append(
                PreprocessedSegment(
                    id=seg.id,  # Preserve parent segment ID
                    start=sub_start,
                    end=sub_end,
                    text=part,
                    token_count=tokens,
                    has_repetition_flag=seg.has_repetition_flag
                )
            )

        return sub_segments

    def chunk_lecture(
        self,
        transcript_data: Dict[str, Any],
        lecture_info: Dict[str, Any]
    ) -> List[TranscriptChunk]:
        """
        Chunk an entire lecture transcript into timestamp-aware overlapping chunks.
        """
        raw_segments = transcript_data.get("segments", [])
        if not raw_segments:
            return []

        # 1. Preprocess segments
        preprocessed = self.preprocessor.preprocess_segments(raw_segments, self.tokenizer)
        if not preprocessed:
            return []

        # 2. Handle oversized segments
        normalized_segments: List[PreprocessedSegment] = []
        for seg in preprocessed:
            if seg.token_count > self.max_chunk_size_tokens:
                normalized_segments.extend(self._split_oversized_segment(seg))
            else:
                normalized_segments.append(seg)

        # 3. Segment-aware overlapping chunking
        chunks: List[TranscriptChunk] = []
        current_chunk_segs: List[PreprocessedSegment] = []
        current_tokens = 0
        chunk_idx = 0
        gap_detected = False

        lecture_id = lecture_info.get("lecture_id", "LEC_00")
        lecture_title = lecture_info.get("lecture_title", "")
        tutorial_number = int(lecture_info.get("tutorial_number", 0))
        source_video = lecture_info.get("video_filename", "")
        detected_language = transcript_data.get("detected_language", lecture_info.get("detected_language", "hi"))
        source_transcript_path = lecture_info.get("transcript_path", "")

        def build_chunk(
            segs: List[PreprocessedSegment],
            c_index: int,
            flag_gap: bool
        ) -> TranscriptChunk:
            combined_text = " ".join(s.text for s in segs)
            c_tokens = self.tokenizer.count_tokens(combined_text)
            c_start = segs[0].start
            c_end = segs[-1].end
            seg_indices = sorted(list(set(s.id for s in segs)))
            has_rep = any(s.has_repetition_flag for s in segs)

            return TranscriptChunk(
                chunk_id=f"{lecture_id}_CHK_{c_index:04d}",
                lecture_id=lecture_id,
                lecture_title=lecture_title,
                tutorial_number=tutorial_number,
                source_video=source_video,
                start_time=c_start,
                end_time=c_end,
                start_timestamp=seconds_to_timestamp(c_start),
                end_timestamp=seconds_to_timestamp(c_end),
                text=combined_text,
                token_count=c_tokens,
                character_count=len(combined_text),
                detected_language=detected_language,
                source_transcript_path=source_transcript_path,
                source_segment_indices=seg_indices,
                chunk_index=c_index,
                has_gap_flag=flag_gap,
                has_repetition_flag=has_rep
            )

        i = 0
        while i < len(normalized_segments):
            seg = normalized_segments[i]

            # Check for substantial timestamp gap with previous segment in current chunk
            if current_chunk_segs:
                prev_end = current_chunk_segs[-1].end
                time_gap = seg.start - prev_end
                if time_gap > self.gap_threshold_seconds:
                    # Finalize current chunk without bridging the silence gap
                    chunks.append(build_chunk(current_chunk_segs, chunk_idx, gap_detected))
                    chunk_idx += 1
                    # Start clean after the gap (no overlap across large gaps)
                    current_chunk_segs = [seg]
                    current_tokens = seg.token_count
                    gap_detected = True
                    i += 1
                    continue

            # Check token limits
            if current_tokens + seg.token_count > self.chunk_size_tokens and current_chunk_segs:
                # If adding this segment still fits within max_chunk_size and completes a sentence, include it
                if (current_tokens + seg.token_count <= self.max_chunk_size_tokens and
                    re.search(r"[.!?।]$", seg.text.strip())):
                    current_chunk_segs.append(seg)
                    current_tokens += seg.token_count
                    i += 1

                # Finalize current chunk
                chunks.append(build_chunk(current_chunk_segs, chunk_idx, gap_detected))
                chunk_idx += 1
                gap_detected = False

                # Calculate overlap segments from the end of current_chunk_segs
                overlap_segs: List[PreprocessedSegment] = []
                overlap_tokens = 0
                for r_seg in reversed(current_chunk_segs):
                    if overlap_tokens + r_seg.token_count <= self.chunk_overlap_tokens * 1.5:
                        overlap_segs.insert(0, r_seg)
                        overlap_tokens += r_seg.token_count
                    else:
                        break

                # Ensure we make forward progress (never retain all segments as overlap)
                if len(overlap_segs) >= len(current_chunk_segs):
                    overlap_segs = overlap_segs[1:] if len(overlap_segs) > 1 else []

                current_chunk_segs = list(overlap_segs)
                current_tokens = sum(s.token_count for s in current_chunk_segs)
                # Note: i is NOT incremented if seg was not consumed, so seg will be added in next iteration
            else:
                current_chunk_segs.append(seg)
                current_tokens += seg.token_count
                i += 1

        # Handle remaining segments
        if current_chunk_segs:
            # If the remaining segment collection is tiny and we have a prior chunk in this lecture,
            # consider appending if it stays under max_chunk_size
            if chunks and current_tokens < 35:
                combined_with_last = chunks[-1].text + " " + " ".join(s.text for s in current_chunk_segs)
                combined_tokens = self.tokenizer.count_tokens(combined_with_last)
                if (combined_tokens <= self.max_chunk_size_tokens and
                    current_chunk_segs[0].start - chunks[-1].end_time <= self.gap_threshold_seconds):
                    # Merge into the last chunk
                    last_c = chunks[-1]
                    last_c.text = combined_with_last
                    last_c.token_count = combined_tokens
                    last_c.character_count = len(combined_with_last)
                    last_c.end_time = current_chunk_segs[-1].end
                    last_c.end_timestamp = seconds_to_timestamp(last_c.end_time)
                    last_c.source_segment_indices = sorted(list(set(
                        last_c.source_segment_indices + [s.id for s in current_chunk_segs]
                    )))
                    current_chunk_segs = []

            if current_chunk_segs:
                chunks.append(build_chunk(current_chunk_segs, chunk_idx, gap_detected))

        return chunks
