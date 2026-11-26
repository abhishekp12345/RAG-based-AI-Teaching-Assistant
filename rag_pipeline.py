"""
Complete RAG Pipeline Module for LearnMate AI.
Orchestrates:
1. Input validation & language detection
2. Bounded conversation history & contextual query resolution
3. Candidate retrieval (K=10) & context selection (max 6 chunks / 1600 tokens)
4. Grounded answerability assessment & answer generation via Google Gemini / LLM
5. Multi-tiered citation validation with bounded correction (NO top-chunk fallback)
6. Enriched application response payload for Streamlit UI
"""

import time
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import config
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from llm import LLMClient, LLMGenerationResult
from prompts import format_context_for_llm, validate_grounded_response

VALID_LECTURES = {"LEC_01", "LEC_02", "LEC_03", "DEMO_01", "DEMO_02", "DEMO_03"}


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline for LearnMate AI.
    Guarantees lecture-grounded explanations, verified timestamps,
    multilingual support (Hindi, English, Hinglish), and hallucination defense.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_client: Optional[LLMClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        candidate_k: int = config.RAG_CANDIDATE_TOP_K,
        max_context_chunks: int = config.RAG_MAX_CONTEXT_CHUNKS,
        max_context_tokens: int = config.RAG_MAX_CONTEXT_TOKENS
    ):
        self.vector_store = vector_store or VectorStore()
        if self.vector_store.embeddings is None:
            self.vector_store.load()

        self.llm = llm_client or LLMClient()
        self.embedder = embedding_generator or self.vector_store.get_embedding_generator()
        self.candidate_k = candidate_k
        self.max_context_chunks = max_context_chunks
        self.max_context_tokens = max_context_tokens

    @staticmethod
    def resolve_conversational_query(
        current_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Lightweight conversational query resolution.
        If current query contains follow-up pronouns ('it', 'this', 'yeh', 'iska')
        and conversation history is present, prepends prior user topic.
        Does not require an extra LLM call for standalone queries.
        """
        if not conversation_history:
            return current_query

        # Check for specific anaphoric pronouns or follow-up triggers
        follow_up_triggers = [
            r"\b(it|its|this|that|these|those)\b",
            r"\b(yeh|ye|iska|iski|inke|isme|isse|unka)\b",
            r"\b(why\s+is\s+it|how\s+does\s+it|what\s+about\s+it|what\s+about\s+this|aur\s+yeh|iska\s+kya)\b",
            r"^(and|aur|also|then|phir)\s+"
        ]
        is_follow_up = any(re.search(pat, current_query.lower()) for pat in follow_up_triggers)

        if not is_follow_up:
            return current_query

        # Extract last user query from bounded history
        last_user_query = ""
        for turn in reversed(conversation_history[-4:]):
            if turn.get("role") == "user":
                last_user_query = turn.get("content", "").strip()
                break

        if last_user_query:
            clean_prior = re.sub(r"[?!.,]", "", last_user_query)
            return f"Regarding '{clean_prior}': {current_query}"

        return current_query

    def answer_doubt(
        self,
        question: str,
        lecture_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        candidate_k: Optional[int] = None,
        mock_override: Optional[Dict[str, Any]] = None,
        allow_citation_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full RAG pipeline for a student question.
        Returns complete verified response payload.
        """
        t_pipeline_start = time.time()

        # 1. Input Validation
        if not question or not question.strip():
            return self._build_error_response(
                question="",
                error_msg="Question cannot be empty or whitespace.",
                status="NOT_ANSWERABLE"
            )

        clean_question = question.strip()

        # Validate lecture filter if provided
        if lecture_id and lecture_id not in VALID_LECTURES:
            return self._build_error_response(
                question=clean_question,
                error_msg=f"Invalid lecture ID '{lecture_id}'. Valid IDs are: {sorted(list(VALID_LECTURES))}",
                status="NOT_ANSWERABLE"
            )

        # 2. Contextual Query Rewriting for Follow-ups
        resolved_question = self.resolve_conversational_query(clean_question, conversation_history)

        # 3. Two-Stage Retrieval
        # Stage 3A: Candidate Retrieval (fetch candidate pool of K=10 for high recall)
        t_retrieval_start = time.time()
        cand_k = candidate_k or self.candidate_k
        candidate_chunks = self.vector_store.search(
            query=resolved_question,
            top_k=cand_k,
            lecture_id=lecture_id,
            strategy="dense",
            deduplicate=True
        )
        retrieval_time_ms = (time.time() - t_retrieval_start) * 1000

        # Stage 3B: Context Budgeting (Select top N chunks fitting within token budget)
        selected_context_chunks = []
        accumulated_words = 0
        max_words = int(self.max_context_tokens / 1.35)

        for chunk in candidate_chunks[:self.max_context_chunks]:
            chunk_words = len(chunk.get("text", "").split())
            if accumulated_words + chunk_words <= max_words or not selected_context_chunks:
                selected_context_chunks.append(chunk)
                accumulated_words += chunk_words
            else:
                break

        # 4. LLM Answerability & Grounded Generation
        t_llm_start = time.time()
        llm_result: LLMGenerationResult = self.llm.generate_grounded_answer(
            question=resolved_question,
            retrieved_chunks=selected_context_chunks,
            mock_override=mock_override
        )
        llm_time_ms = (time.time() - t_llm_start) * 1000

        # 5. Strict Citation Validation & Bounded Correction (NO UNSAFE FALLBACK)
        raw_output = llm_result.parsed_json or {}
        verified_data = self._strengthen_citations(
            raw_output=raw_output,
            question=resolved_question,
            context_chunks=selected_context_chunks,
            allow_citation_correction=allow_citation_correction
        )

        total_time_ms = (time.time() - t_pipeline_start) * 1000

        # 6. Assemble Application-Level Enriched Response
        return {
            "question": clean_question,
            "resolved_question": resolved_question,
            "lecture_id_filter": lecture_id,
            "answerability_status": verified_data.get("answerability_status", "NOT_ANSWERABLE"),
            "answer": verified_data.get("answer", ""),
            "response_language": verified_data.get("response_language", "English"),
            "citations": verified_data.get("verified_citations", []),
            "supporting_chunk_ids": verified_data.get("supporting_chunk_ids", []),
            "missing_information": verified_data.get("missing_information", None),
            "retrieved_chunks": selected_context_chunks,
            "metrics": {
                "retrieval_latency_ms": round(retrieval_time_ms, 2),
                "llm_latency_ms": round(llm_time_ms, 2),
                "total_latency_ms": round(total_time_ms, 2),
                "token_usage": {
                    "prompt_tokens": llm_result.prompt_tokens,
                    "completion_tokens": llm_result.completion_tokens,
                    "total_tokens": llm_result.total_tokens
                },
                "model_name": llm_result.model_name,
                "provider": getattr(llm_result, "provider", "gemini"),
                "is_mock": llm_result.is_mock
            },
            "error": llm_result.error_message
        }

    def _strengthen_citations(
        self,
        raw_output: Dict[str, Any],
        question: str,
        context_chunks: List[Dict[str, Any]],
        allow_citation_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Strict Citation Validation & Bounded Correction:
        1. Validates all citations against context_chunks (supplied to LLM).
        2. NO automatic attachment of chunk 0.
        3. If marked ANSWERABLE or PARTIALLY_ANSWERABLE but citations missing/invalid:
           - Attempts one bounded response correction (if allowed).
           - If correction fails: withholds unsupported answer and returns safe refusal.
        4. Clears all citations if NOT_ANSWERABLE.
        """
        valid_cids_map = {c["chunk_id"]: c for c in context_chunks}
        status = raw_output.get("answerability_status", "NOT_ANSWERABLE")
        claimed_cids = raw_output.get("supporting_chunk_ids", [])
        current_answer = raw_output.get("answer", "")
        lang = raw_output.get("response_language", "English")

        # Filter strictly valid chunk IDs from the supplied context
        verified_cids = [cid for cid in claimed_cids if cid in valid_cids_map]

        # Citation validation check for substantive answers
        if status in ("ANSWERABLE", "PARTIALLY_ANSWERABLE") and not verified_cids:
            if allow_citation_correction:
                # Attempt ONE bounded correction
                corrected_cids = self.llm.attempt_citation_correction(
                    question=question,
                    answer=current_answer,
                    retrieved_chunks=context_chunks
                )
                if corrected_cids:
                    verified_cids = [cid for cid in corrected_cids if cid in valid_cids_map]

            # If validation still fails: WITHHOLD UNSUPPORTED ANSWER
            if not verified_cids:
                status = "NOT_ANSWERABLE"
                if lang == "Hindi":
                    current_answer = (
                        "मुझे उपलब्ध लेक्चर्स में इस उत्तर की पुष्टि के लिए पर्याप्त प्रमाणित संदर्भ नहीं मिला। "
                        "वर्तमान में हमारे लेक्चर्स HTML बेसिक्स, मीडिया एलिमेंट्स (ऑडियो/वीडियो), SEO और वेब डेवलपमेंट प्रोजेक्ट सेटअप को कवर करते हैं।"
                    )
                elif lang == "Hinglish":
                    current_answer = (
                        "Mujhe available lectures mein is answer ko verify karne ke liye sufficient evidence nahi mila. "
                        "Currently hamare lectures HTML basics, media elements (audio/video), SEO aur web development project setup cover karte hain."
                    )
                else:
                    current_answer = (
                        "I could not verify sufficient supporting evidence in the lecture context to answer this question. "
                        "The current lectures cover HTML fundamentals, audio/video media elements, SEO, and web development project setup."
                    )
                raw_output["missing_information"] = "Citation validation failed: Answer lacked verifiable citations from the supplied lecture context."

        # Build verified citations from trusted metadata
        verified_citations = []
        for cid in verified_cids:
            chunk = valid_cids_map[cid]
            verified_citations.append({
                "chunk_id": cid,
                "lecture_id": chunk.get("lecture_id", ""),
                "lecture_title": chunk.get("lecture_title", ""),
                "tutorial_number": chunk.get("tutorial_number", None),
                "start_timestamp": chunk.get("start_timestamp", "00:00:00"),
                "end_timestamp": chunk.get("end_timestamp", "00:00:00"),
                "start_time": chunk.get("start_time", 0.0),
                "end_time": chunk.get("end_time", 0.0),
                "source_video": chunk.get("source_video", "")
            })

        # Remove duplicate citations
        unique_citations = []
        seen = set()
        for cit in verified_citations:
            if cit["chunk_id"] not in seen:
                seen.add(cit["chunk_id"])
                unique_citations.append(cit)

        # Unanswerable guard
        if status == "NOT_ANSWERABLE":
            verified_cids = []
            unique_citations = []

        raw_output["answerability_status"] = status
        raw_output["answer"] = current_answer
        raw_output["supporting_chunk_ids"] = verified_cids
        raw_output["verified_citations"] = unique_citations
        return raw_output

    def _build_error_response(self, question: str, error_msg: str, status: str) -> Dict[str, Any]:
        """Generate safe, structured error fallback."""
        return {
            "question": question,
            "resolved_question": question,
            "lecture_id_filter": None,
            "answerability_status": status,
            "answer": error_msg,
            "response_language": "English",
            "citations": [],
            "supporting_chunk_ids": [],
            "missing_information": error_msg,
            "retrieved_chunks": [],
            "metrics": {
                "retrieval_latency_ms": 0.0,
                "llm_latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model_name": self.llm.model,
                "is_mock": self.llm.mock_mode
            },
            "error": error_msg
        }
