"""
LLM Interface Module for LearnMate AI.
Supports Google Gemini (Primary / Free Tier) via official google-genai SDK
and OpenAI (Optional Inactive Provider) with:
- Configurable Gemini Flash / Flash-Lite models (e.g. gemini-3.5-flash-lite)
- Structured JSON outputs with strict schema enforcement
- Strict separation of LIVE and MOCK execution modes (no silent fallbacks)
- Free-tier rate-limit, quota exhaustion, and timeout handling
- Token usage tracking and zero-expenditure cost accounting
- Bounded citation correction interface
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, List, Union, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field as PydanticField

import config
from prompts import (
    ANSWERABILITY_SYSTEM_PROMPT,
    GROUNDED_RAG_USER_PROMPT_TEMPLATE,
    ANSWERABILITY_RESPONSE_SCHEMA,
    format_context_for_llm,
    validate_grounded_response
)

logger = logging.getLogger(__name__)


class GroundedDoubtResponse(BaseModel):
    """Pydantic structured response schema for Gemini native JSON generation."""
    answerability_status: Literal["ANSWERABLE", "PARTIALLY_ANSWERABLE", "NOT_ANSWERABLE"] = PydanticField(
        description="Whether the lecture context contains sufficient evidence to answer the student question."
    )
    answer: str = PydanticField(
        description="Clear, helpful pedagogical explanation grounded strictly in the supplied transcript context."
    )
    supporting_chunk_ids: List[str] = PydanticField(
        default_factory=list,
        description="List of exact chunk IDs (e.g. LEC_01_CHK_0001) that directly support this explanation."
    )
    citations: List[Dict[str, Any]] = PydanticField(
        default_factory=list,
        description="List of verified citation objects."
    )
    missing_information: Optional[str] = PydanticField(
        default=None,
        description="Explanation of missing information if PARTIALLY_ANSWERABLE or NOT_ANSWERABLE."
    )
    response_language: Literal["English", "Hindi", "Hinglish"] = PydanticField(
        default="English",
        description="Language of the student's question and generated response."
    )


GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answerability_status": {
            "type": "STRING",
            "enum": ["ANSWERABLE", "PARTIALLY_ANSWERABLE", "NOT_ANSWERABLE"]
        },
        "answer": {"type": "STRING"},
        "supporting_chunk_ids": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        },
        "citations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "chunk_id": {"type": "STRING"},
                    "timestamp": {"type": "STRING"}
                }
            }
        },
        "missing_information": {"type": "STRING"},
        "response_language": {
            "type": "STRING",
            "enum": ["English", "Hindi", "Hinglish"]
        }
    },
    "required": ["answerability_status", "answer", "supporting_chunk_ids", "response_language"]
}

GEMINI_CORRECTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "supporting_chunk_ids": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        }
    },
    "required": ["supporting_chunk_ids"]
}


@dataclass
class LLMGenerationResult:
    """Structured response container from LLM generation."""
    raw_content: str
    parsed_json: Optional[Dict[str, Any]] = None
    model_name: str = ""
    provider: str = "gemini"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    is_mock: bool = False


class LLMClient:
    """
    Unified LLM Client wrapper supporting Google Gemini (primary free tier)
    and OpenAI (optional inactive provider) with strict credential handling,
    structured JSON validation, free-tier guardrails, and strict mock/live separation.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        enable_model_fallback: Optional[bool] = None,
        timeout: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        mock_mode: Optional[bool] = None,
        max_retries: int = 1
    ):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        if self.provider == "gemini":
            self.api_key = api_key if api_key is not None else config.GEMINI_API_KEY
            self.model = model or config.GEMINI_MODEL
            self.fallback_model = fallback_model or config.GEMINI_FALLBACK_MODEL
            self.enable_model_fallback = enable_model_fallback if enable_model_fallback is not None else config.ENABLE_MODEL_FALLBACK
        else:
            self.api_key = api_key if api_key is not None else config.OPENAI_API_KEY
            self.model = model or config.OPENAI_MODEL
            self.fallback_model = None
            self.enable_model_fallback = False

        self.timeout = timeout or config.LLM_TIMEOUT
        self.max_output_tokens = max_output_tokens or config.LLM_MAX_OUTPUT_TOKENS
        self.mock_mode = mock_mode if mock_mode is not None else (config.LLM_MODE == "mock")
        self.max_retries = max_retries
        self._client = None

        # Usage tracking
        self.request_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    def is_api_key_valid(self) -> bool:
        """Check if an API key is present and non-placeholder without exposing it."""
        if not self.api_key:
            return False
        clean = self.api_key.strip()
        if clean.startswith("your_") or len(clean) < 20:
            return False
        return True

    def get_client(self):
        """Lazy loader for LLM provider client (Google Gemini or OpenAI)."""
        if self._client is None and not self.mock_mode:
            if not self.is_api_key_valid():
                var_name = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"
                raise ValueError(
                    f"{var_name} is not configured or contains placeholder. "
                    f"Configure a valid key in .env or run with mock_mode=True."
                )

            if self.provider == "gemini":
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            else:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=2
                )
        return self._client

    def generate_grounded_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        mock_override: Optional[Dict[str, Any]] = None
    ) -> LLMGenerationResult:
        """
        Generate grounded answer with answerability assessment using structured output.
        Strictly respects mock_mode vs live execution.
        """
        # Format context with explicit chunk boundaries
        formatted_context = format_context_for_llm(retrieved_chunks)
        user_prompt = GROUNDED_RAG_USER_PROMPT_TEMPLATE.format(
            formatted_context=formatted_context,
            question=question
        )
        sys_prompt = system_prompt or ANSWERABILITY_SYSTEM_PROMPT

        # MOCK MODE: Explicit offline execution
        if self.mock_mode:
            return self._generate_mock_response(question, retrieved_chunks, mock_override)

        # LIVE MODE: Verify key presence before executing
        if not self.is_api_key_valid():
            var_name = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"
            raise ValueError(
                f"Cannot execute live {self.provider.upper()} API request: {var_name} is missing or invalid in .env. "
                "Set a valid key or enable mock_mode=True."
            )

        # Execute live API call
        if self.provider == "gemini":
            return self._execute_gemini_generation(sys_prompt, user_prompt, retrieved_chunks)
        else:
            return self._execute_openai_generation(sys_prompt, user_prompt, retrieved_chunks)

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
        max_output_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        is_fallback: bool = False
    ) -> LLMGenerationResult:
        """
        Execute structured generation with an arbitrary OpenAPI response schema.
        Supports both Gemini (native schema enforcement with single bounded 503 fallback)
        and OpenAI (json_object).
        """
        active_model = model_name or self.model

        if self.mock_mode:
            return LLMGenerationResult(
                raw_content="{}",
                parsed_json={},
                model_name=f"{active_model}-mock",
                provider=self.provider,
                is_mock=True
            )

        if not self.is_api_key_valid():
            var_name = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENAI_API_KEY"
            raise ValueError(
                f"Cannot execute live {self.provider.upper()} API request: {var_name} is missing or invalid in .env. "
                "Set a valid key or enable mock_mode=True."
            )

        client = self.get_client()
        t0 = time.time()
        max_toks = max_output_tokens or self.max_output_tokens

        if self.provider == "gemini":
            from google.genai import types

            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.2,
                max_output_tokens=max_toks,
                system_instruction=system_prompt
            )

            max_retries = getattr(self, "max_retries", 1)
            backoff_delays = [3.0, 6.0, 10.0, 15.0]
            response = None
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=active_model,
                        contents=user_prompt,
                        config=gen_config
                    )
                    break
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    is_transient = any(k in err_str.lower() for k in ["503", "unavailable", "high demand", "capacity"])
                    if is_transient and attempt < max_retries:
                        delay = backoff_delays[attempt]
                        logger.warning(f"Transient capacity spike on '{active_model}' (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        break

            elapsed_ms = (time.time() - t0) * 1000
            if response is None:
                err_str = str(last_error)
                logger.error(f"Google Gemini structured generation failed on '{active_model}': {err_str}")

                # Bounded Fallback on 503 UNAVAILABLE (Single attempt to alternative free model)
                is_503 = any(k in err_str.lower() for k in ["503", "unavailable", "high demand", "capacity"])
                if is_503 and not is_fallback and self.enable_model_fallback and self.fallback_model:
                    logger.warning(
                        f"Primary model '{active_model}' returned 503 UNAVAILABLE. "
                        f"Executing single bounded fallback to alternative model '{self.fallback_model}'..."
                    )
                    return self.generate_structured(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_schema=response_schema,
                        max_output_tokens=max_output_tokens,
                        model_name=self.fallback_model,
                        is_fallback=True
                    )

                return LLMGenerationResult(
                    raw_content="",
                    parsed_json=None,
                    model_name=active_model,
                    provider="gemini",
                    latency_ms=round(elapsed_ms, 2),
                    error_message=err_str,
                    is_mock=False
                )

            self.request_count += 1
            raw_text = response.text or "{}"
            usage = response.usage_metadata
            p_toks = usage.prompt_token_count if usage else 0
            c_toks = usage.candidates_token_count if usage else 0
            self.total_prompt_tokens += p_toks
            self.total_completion_tokens += c_toks

            try:
                parsed_data = json.loads(raw_text)
            except Exception as jde:
                logger.error(f"JSON decode error in structured gen: {jde}")
                parsed_data = None

            return LLMGenerationResult(
                raw_content=raw_text,
                parsed_json=parsed_data,
                model_name=active_model,
                provider="gemini",
                prompt_tokens=p_toks,
                completion_tokens=c_toks,
                total_tokens=p_toks + c_toks,
                latency_ms=round(elapsed_ms, 2),
                is_mock=False
            )
        else:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=max_toks
                )
                elapsed_ms = (time.time() - t0) * 1000
                self.request_count += 1
                raw_text = response.choices[0].message.content or "{}"
                usage = response.usage
                p_toks = usage.prompt_tokens if usage else 0
                c_toks = usage.completion_tokens if usage else 0
                self.total_prompt_tokens += p_toks
                self.total_completion_tokens += c_toks

                try:
                    parsed_data = json.loads(raw_text)
                except Exception:
                    parsed_data = None

                return LLMGenerationResult(
                    raw_content=raw_text,
                    parsed_json=parsed_data,
                    model_name=self.model,
                    provider="openai",
                    prompt_tokens=p_toks,
                    completion_tokens=c_toks,
                    total_tokens=p_toks + c_toks,
                    latency_ms=round(elapsed_ms, 2),
                    is_mock=False
                )
            except Exception as e:
                elapsed_ms = (time.time() - t0) * 1000
                return LLMGenerationResult(
                    raw_content="",
                    parsed_json=None,
                    model_name=self.model,
                    provider="openai",
                    latency_ms=round(elapsed_ms, 2),
                    error_message=str(e),
                    is_mock=False
                )

    def _execute_gemini_generation(
        self,
        system_prompt: str,
        user_prompt: str,
        retrieved_chunks: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        is_fallback: bool = False
    ) -> LLMGenerationResult:
        """Execute Google Gemini generate_content with structured JSON output, transient 503 retry, and single bounded fallback."""
        client = self.get_client()
        t0 = time.time()
        active_model = model_name or self.model

        from google.genai import types

        gen_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_RESPONSE_SCHEMA,
            temperature=0.0,
            max_output_tokens=self.max_output_tokens,
            system_instruction=system_prompt
        )

        max_retries = getattr(self, "max_retries", 1)
        backoff_delays = [3.0, 6.0, 10.0, 15.0]
        response = None
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=active_model,
                    contents=user_prompt,
                    config=gen_config
                )
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                # Check for transient demand/capacity spikes on free tier
                is_transient_spike = any(k in err_str.lower() for k in ["503", "unavailable", "high demand", "capacity"])
                if is_transient_spike and attempt < max_retries:
                    delay = backoff_delays[attempt]
                    logger.warning(
                        f"Google Gemini transient capacity spike on '{active_model}' (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    break

        if response is None:
            elapsed_ms = (time.time() - t0) * 1000
            err_str = str(last_error)
            logger.error(f"Google Gemini API error during generation on '{active_model}': {err_str}")

            # Bounded fallback on 503 UNAVAILABLE to verified alternative free-tier model
            is_503 = any(k in err_str.lower() for k in ["503", "unavailable", "high demand", "capacity"])
            if is_503 and not is_fallback and self.enable_model_fallback and self.fallback_model:
                logger.warning(
                    f"Primary model '{active_model}' returned 503 UNAVAILABLE. "
                    f"Executing single bounded fallback to alternative model '{self.fallback_model}'..."
                )
                return self._execute_gemini_generation(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    retrieved_chunks=retrieved_chunks,
                    model_name=self.fallback_model,
                    is_fallback=True
                )

            # Specific student-friendly free-tier error categorization
            if is_503:
                user_msg = (
                    "The AI service is temporarily experiencing high demand. "
                    "Please try again later or switch to Offline Demonstration mode."
                )
            elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                user_msg = (
                    "Google Gemini free-tier rate limit or quota exceeded. "
                    "Please wait a moment before trying again (Free tier rate limit: 15 RPM)."
                )
            elif "404" in err_str or "NOT_FOUND" in err_str:
                user_msg = f"The requested Gemini model '{active_model}' is currently unavailable. Please check the model name."
            elif "timeout" in err_str.lower() or "timed out" in err_str.lower():
                user_msg = f"Google Gemini request timed out after {self.timeout}s. Please check your network connection."
            else:
                user_msg = (
                    "The AI service is temporarily experiencing an issue. "
                    "Please try again later or switch to Offline Demonstration mode."
                )

            # NEVER silently fall back to mock generation when live API fails
            fallback_data = {
                "answerability_status": "NOT_ANSWERABLE",
                "answer": user_msg,
                "supporting_chunk_ids": [],
                "citations": [],
                "missing_information": f"API Error ({active_model}): {err_str[:120]}",
                "response_language": "English"
            }

            return LLMGenerationResult(
                raw_content="",
                parsed_json=fallback_data,
                model_name=active_model,
                provider="gemini",
                latency_ms=round(elapsed_ms, 2),
                error_message=err_str,
                is_mock=False
            )

        elapsed_ms = (time.time() - t0) * 1000
        self.request_count += 1

        raw_text = response.text or "{}"
        usage = response.usage_metadata

        prompt_toks = usage.prompt_token_count if usage else 0
        comp_toks = usage.candidates_token_count if usage else 0
        tot_toks = usage.total_token_count if usage else 0

        self.total_prompt_tokens += prompt_toks
        self.total_completion_tokens += comp_toks

        # Parse and validate JSON schema
        try:
            parsed_data = json.loads(raw_text)
        except json.JSONDecodeError as jde:
            parsed_data = {
                "answerability_status": "NOT_ANSWERABLE",
                "answer": "Error parsing model response. Please try rephrasing your question.",
                "supporting_chunk_ids": [],
                "citations": [],
                "missing_information": f"JSON Decode Error: {str(jde)}",
                "response_language": "English"
            }

        # Enforce provenance and citation validation against retrieved chunks
        validated_data = validate_grounded_response(parsed_data, retrieved_chunks)

        return LLMGenerationResult(
            raw_content=raw_text,
            parsed_json=validated_data,
            model_name=self.model,
            provider="gemini",
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            total_tokens=tot_toks,
            latency_ms=round(elapsed_ms, 2),
            is_mock=False
        )

    def _execute_openai_generation(
        self,
        system_prompt: str,
        user_prompt: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> LLMGenerationResult:
        """Execute OpenAI chat completion fallback with structured JSON output and error handling."""
        client = self.get_client()
        t0 = time.time()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=self.max_output_tokens
            )
            elapsed_ms = (time.time() - t0) * 1000
            self.request_count += 1

            choice = response.choices[0]
            raw_text = choice.message.content or "{}"
            usage = response.usage

            prompt_toks = usage.prompt_tokens if usage else 0
            comp_toks = usage.completion_tokens if usage else 0
            tot_toks = usage.total_tokens if usage else 0

            self.total_prompt_tokens += prompt_toks
            self.total_completion_tokens += comp_toks

            try:
                parsed_data = json.loads(raw_text)
            except json.JSONDecodeError as jde:
                parsed_data = {
                    "answerability_status": "NOT_ANSWERABLE",
                    "answer": "Error parsing model response. Please try rephrasing your question.",
                    "supporting_chunk_ids": [],
                    "citations": [],
                    "missing_information": f"JSON Decode Error: {str(jde)}",
                    "response_language": "English"
                }

            validated_data = validate_grounded_response(parsed_data, retrieved_chunks)

            return LLMGenerationResult(
                raw_content=raw_text,
                parsed_json=validated_data,
                model_name=self.model,
                provider="openai",
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                total_tokens=tot_toks,
                latency_ms=round(elapsed_ms, 2),
                is_mock=False
            )

        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            err_str = str(e)
            logger.error(f"OpenAI API error during generation: {err_str}")

            fallback_data = {
                "answerability_status": "NOT_ANSWERABLE",
                "answer": (
                    f"An error occurred connecting to the OpenAI API: {err_str[:120]}. "
                    "Please check your API key, network connection, or quota."
                ),
                "supporting_chunk_ids": [],
                "citations": [],
                "missing_information": f"API Error: {err_str[:120]}",
                "response_language": "English"
            }

            return LLMGenerationResult(
                raw_content="",
                parsed_json=fallback_data,
                model_name=self.model,
                provider="openai",
                latency_ms=round(elapsed_ms, 2),
                error_message=err_str,
                is_mock=False
            )

    def attempt_citation_correction(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Optional[List[str]]:
        """
        Attempts one bounded correction to find supporting chunk IDs from the supplied context.
        Validates the corrected answer and supporting evidence together.
        Returns list of valid chunk IDs if verifiable, or None if validation fails.
        """
        if not retrieved_chunks:
            return None

        valid_cids = [c["chunk_id"] for c in retrieved_chunks]

        # In mock mode, if answer mentions key concepts from any chunk, identify that chunk
        if self.mock_mode:
            import re
            ans_lower = answer.lower()
            ans_words = set(re.findall(r"\w+", ans_lower))
            for chunk in retrieved_chunks:
                c_text_lower = chunk.get("text", "").lower()
                chunk_words = set(re.findall(r"\w+", c_text_lower))
                shared = chunk_words.intersection(ans_words)
                if len(shared) >= 2:
                    return [chunk["chunk_id"]]
            return None

        # In live mode: execute a bounded correction completion
        try:
            chunks_summary = "\n".join([f"- {c['chunk_id']}: {c['text'][:150]}..." for c in retrieved_chunks])
            correction_prompt = (
                f"You previously generated this answer:\n\"{answer}\"\n\n"
                f"However, valid chunk IDs were missing. From the following context chunks:\n{chunks_summary}\n\n"
                f"Return a JSON object with: {{\"supporting_chunk_ids\": [\"chunk_id\", ...]}} "
                f"using ONLY IDs from {valid_cids} that directly support the answer. If none support it, return []."
            )

            if self.provider == "gemini":
                client = self.get_client()
                from google.genai import types
                gen_config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GEMINI_CORRECTION_SCHEMA,
                    temperature=0.0,
                    max_output_tokens=150
                )
                resp = None
                for c_attempt in range(3):
                    try:
                        resp = client.models.generate_content(
                            model=self.model,
                            contents=correction_prompt,
                            config=gen_config
                        )
                        break
                    except Exception as ce:
                        if any(k in str(ce).lower() for k in ["503", "unavailable", "high demand"]) and c_attempt < 2:
                            time.sleep(3.0 * (c_attempt + 1))
                        else:
                            raise ce
                raw = resp.text or "{}" if resp else "{}"
            else:
                client = self.get_client()
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": correction_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=150
                )
                raw = resp.choices[0].message.content or "{}"

            data = json.loads(raw)
            cids = data.get("supporting_chunk_ids", [])
            verified = [cid for cid in cids if cid in valid_cids]
            return verified if verified else None
        except Exception as e:
            logger.warning(f"Bounded citation correction failed: {e}")
            return None

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return cumulative token and request tracking summary."""
        # Google Gemini Free Tier cost is strictly $0 / ₹0
        cost_usd = 0.0 if self.provider == "gemini" else (
            (self.total_prompt_tokens / 1_000_000 * 0.15) + (self.total_completion_tokens / 1_000_000 * 0.60)
        )
        return {
            "provider": self.provider,
            "model": self.model,
            "request_count": self.request_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "estimated_cost_usd": cost_usd,
            "estimated_cost_inr": 0.0
        }

    def _generate_mock_response(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        mock_override: Optional[Dict[str, Any]] = None
    ) -> LLMGenerationResult:
        """
        Generate realistic mock response for offline testing.
        Calculates realistic prompt token usage based on context length.
        """
        t0 = time.time()

        # Calculate realistic token counts
        context_words = sum(len(c.get("text", "").split()) for c in retrieved_chunks)
        approx_prompt_tokens = int(500 + len(question.split()) * 1.35 + context_words * 1.35 + 150)
        approx_comp_tokens = 180

        if mock_override:
            parsed = validate_grounded_response(dict(mock_override), retrieved_chunks)
            return LLMGenerationResult(
                raw_content=json.dumps(parsed),
                parsed_json=parsed,
                model_name=f"{self.model}-mock",
                provider=self.provider,
                prompt_tokens=approx_prompt_tokens,
                completion_tokens=approx_comp_tokens,
                total_tokens=approx_prompt_tokens + approx_comp_tokens,
                latency_ms=round((time.time() - t0) * 1000, 2),
                is_mock=True
            )

        # Detect language of question
        is_hindi = any('\u0900' <= char <= '\u097f' for char in question)
        has_hinglish = any(term in question.lower() for term in ["kya", "kaise", "hai", "hota", "me", "mein", "tag"])
        lang = "Hindi" if is_hindi else ("Hinglish" if has_hinglish else "English")

        # Unanswerable keywords
        unanswerable_keywords = [
            "photosynthesis", "schrödinger", "quantum", "gravity", "newton",
            "industrial revolution", "biology", "heart", "black hole",
            "react", "docker", "subgrid", "websocket", "next.js", "useeffect", "sql"
        ]
        is_unanswerable = any(kw in question.lower() for kw in unanswerable_keywords) or len(retrieved_chunks) == 0

        if is_unanswerable:
            status = "NOT_ANSWERABLE"
            cids = []
            if lang == "Hindi":
                ans = (
                    "मुझे उपलब्ध लेक्चर्स में इस विषय पर पर्याप्त जानकारी नहीं मिली। "
                    "वर्तमान में हमारे लेक्चर्स HTML बेसिक्स, मीडिया एलिमेंट्स (ऑडियो/वीडियो), SEO और वेब डेवलपमेंट प्रोजेक्ट सेटअप को कवर करते हैं।"
                )
            elif lang == "Hinglish":
                ans = (
                    "Mujhe available lectures mein is topic par sufficient evidence nahi mila. "
                    "Currently hamare lectures HTML basics, media elements (audio/video), SEO aur web development project setup cover karte hain."
                )
            else:
                ans = (
                    "I could not find sufficient information about this topic in the available lectures. "
                    "The current lectures cover HTML fundamentals, audio/video media elements, SEO, and web development project setup."
                )
            missing = f"Topic '{question}' is not covered in Tutorial #2, #6, or #10."
        else:
            status = "ANSWERABLE"
            top_chunk = retrieved_chunks[0]
            cids = [top_chunk["chunk_id"]]
            start_ts = top_chunk.get("start_timestamp", "00:00:00")
            title = top_chunk.get("lecture_title", "HTML Lecture")
            chunk_text = top_chunk.get("text", "")

            if lang == "Hindi":
                ans = (
                    f"लेक्चर '{title}' ({start_ts}) के अनुसार:\n\n"
                    f"{chunk_text}\n\n"
                    f"यह व्याख्या आपके प्रश्न का सीधा और प्रमाणित उत्तर प्रदान करती है।"
                )
            elif lang == "Hinglish":
                ans = (
                    f"Lecture '{title}' ({start_ts}) ke mutabiq:\n\n"
                    f"{chunk_text}\n\n"
                    f"Yeh practical explanation aapke doubt ko clear karti hai."
                )
            else:
                ans = (
                    f"According to the lecture '{title}' (starting at timestamp {start_ts}):\n\n"
                    f"{chunk_text}\n\n"
                    f"This verified concept directly addresses your question."
                )
            missing = None

        mock_data = {
            "answerability_status": status,
            "answer": ans,
            "supporting_chunk_ids": cids,
            "citations": [{"chunk_id": cid, "timestamp": retrieved_chunks[0].get("start_timestamp", "00:00:00")} for cid in cids],
            "missing_information": missing,
            "response_language": lang
        }

        validated = validate_grounded_response(mock_data, retrieved_chunks)

        return LLMGenerationResult(
            raw_content=json.dumps(validated, ensure_ascii=False),
            parsed_json=validated,
            model_name=f"{self.model}-mock",
            provider=self.provider,
            prompt_tokens=approx_prompt_tokens,
            completion_tokens=approx_comp_tokens,
            total_tokens=approx_prompt_tokens + approx_comp_tokens,
            latency_ms=round((time.time() - t0) * 1000, 2),
            is_mock=True
        )
