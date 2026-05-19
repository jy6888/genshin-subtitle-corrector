"""OpenAI-compatible LLM arbitrator for subtitle correction.

Supports any OpenAI-compatible API (DeepSeek, 通义千问, GPT, MiMo, etc.)
by reading OPENAI_API_KEY and OPENAI_BASE_URL from environment variables.
"""

from __future__ import annotations

import json
import re
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from subtitle_corrector.config.settings import ResolverSettings
from subtitle_corrector.resolver.base import LLMArbitrator
from subtitle_corrector.resolver.prompts import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    format_candidates_detail,
    format_context,
)
from subtitle_corrector.schemas import (
    ArbitrationDecision,
    ArbitrationRequest,
    Candidate,
    CorrectionAction,
    CorrectionItem,
)

# Load .env at module import time so OPENAI_API_KEY / OPENAI_BASE_URL are
# available before the OpenAI client is instantiated.
load_dotenv(override=False)


class OpenAIArbitrator(LLMArbitrator):
    """LLM arbitrator powered by an OpenAI-compatible chat completions API."""

    def __init__(self, settings: ResolverSettings) -> None:
        self.settings = settings
        self.client = OpenAI()  # reads OPENAI_API_KEY & OPENAI_BASE_URL from env
        self.model = settings.phase2_model or settings.model or "mimo-v2.5"
        self.discovery_model = settings.discovery_model or settings.model or "mimo-v2.5-pro"
        self.phase2_model = settings.phase2_model or settings.model or "mimo-v2.5"
        logger.info(
            "LLM arbitrator initialized: model={}, discovery_model={}, phase2_model={}, base_url={}",
            self.model, self.discovery_model, self.phase2_model,
            self.client.base_url,
        )

    # ------------------------------------------------------------------
    # Public API (implements LLMArbitrator.arbitrate)
    # ------------------------------------------------------------------

    def arbitrate(self, request: ArbitrationRequest) -> ArbitrationDecision:
        """Call the LLM and parse its JSON response into an ArbitrationDecision.

        On any failure (timeout, malformed JSON, unexpected schema) the
        method gracefully falls back to a safe NEEDS_REVIEW decision so
        the pipeline never crashes.
        """
        user_prompt = self._build_user_prompt(request)

        try:
            raw_json = self._call_llm(user_prompt)
            return self._parse_response(raw_json, request)
        except Exception as exc:
            logger.warning(
                "LLM arbitration failed for cue {}: {} – falling back to NEEDS_REVIEW",
                request.cue.index,
                exc,
            )
            return self._fallback_decision(str(exc))

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(request: ArbitrationRequest) -> str:
        return USER_TEMPLATE.format(
            current_text=request.cue.text,
            context_before=format_context(request.context_before),
            context_after=format_context(request.context_after),
            candidates_detail=format_candidates_detail(request.candidates),
            risk_score=request.risk.risk_score,
            reasons="; ".join(request.risk.reasons) or "（无）",
        )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> str:
        """Send the prompt to the LLM and return the raw content string."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            timeout=self.settings.timeout_seconds,
        )
        content = response.choices[0].message.content or ""
        logger.debug("LLM raw response: {}", content[:500])
        return content

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: str, request: ArbitrationRequest
    ) -> ArbitrationDecision:
        """Parse the LLM JSON response into an ArbitrationDecision."""
        data = self._extract_json(raw)

        action_str = str(data.get("action", "")).lower().strip()
        action = self._map_action(action_str)

        reason = str(data.get("reason", ""))

        # Compute best candidate score once for blending
        best_candidate_score = 0.0
        if request.candidates:
            best_candidate_score = max(c.score for c in request.candidates)

        # --- Parse corrections array (LLM gives word pair only, NO coordinates) ---
        corrections: list[CorrectionItem] = []
        raw_corrections = data.get("corrections", [])
        if isinstance(raw_corrections, list):
            for c in raw_corrections:
                try:
                    original_word = str(c.get("original_word", ""))
                    corrected_word = str(c.get("corrected_word", ""))
                    if not original_word or not corrected_word:
                        continue
                    raw_conf = float(c.get("confidence", 0.5))
                    if best_candidate_score > 0:
                        blended = round(
                            raw_conf * 0.35 + best_candidate_score * 0.65, 3
                        )
                    else:
                        blended = self._clamp(raw_conf, 0.0, 1.0)
                    corrections.append(
                        CorrectionItem(
                            original_word=original_word,
                            corrected_word=corrected_word,
                            confidence=blended,
                        )
                    )
                except (ValueError, KeyError):
                    continue

        # Overall confidence
        if corrections:
            overall_confidence = round(
                sum(c.confidence for c in corrections) / len(corrections), 3
            )
        else:
            overall_confidence = self._clamp(
                float(data.get("confidence", 0.5)), 0.0, 1.0
            )

        # --- Legacy fallback: single-candidate path ---
        selected_candidate: Candidate | None = None
        if not corrections and action == CorrectionAction.REPLACE:
            original_word = str(data.get("original_word", ""))
            corrected_word = str(data.get("corrected_word", ""))
            repaired_text = str(data.get("repaired_text", ""))

            original_text = request.cue.text
            # Guard against word-eating bug
            if repaired_text and len(repaired_text) < len(original_text) * 0.6:
                if original_word and corrected_word and original_word in original_text:
                    repaired_text = original_text.replace(
                        original_word, corrected_word, 1
                    )
                elif corrected_word and original_word:
                    repaired_text = original_text.replace(
                        original_word, corrected_word, 1
                    )
                else:
                    repaired_text = original_text
            if not repaired_text and original_word and corrected_word:
                repaired_text = original_text.replace(
                    original_word, corrected_word, 1
                )
            if repaired_text:
                selected_candidate = Candidate(
                    value=repaired_text,
                    source="llm",
                    score=overall_confidence,
                    explanation=f"LLM: '{original_word}' → '{corrected_word}'",
                    metadata={
                        "original_word": original_word,
                        "corrected_word": corrected_word,
                    },
                )
            if selected_candidate is None:
                action = CorrectionAction.NEEDS_REVIEW
                overall_confidence = 0.0

        # --- Parse activated_parent_entities ---
        activated_parent_entities: list[str] = []
        raw_entities = data.get("activated_parent_entities", [])
        if isinstance(raw_entities, list):
            activated_parent_entities = [
                str(e).strip() for e in raw_entities if str(e).strip()
            ]

        return ArbitrationDecision(
            action=action,
            selected_candidate=selected_candidate,
            corrections=corrections,
            confidence=overall_confidence,
            reasoning=reason,
            activated_parent_entities=activated_parent_entities,
        )

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        """Extract JSON from the LLM response, handling markdown fences."""
        text = raw.strip()
        # Strip markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        return json.loads(text)

    @staticmethod
    def _map_action(action_str: str) -> CorrectionAction:
        mapping = {
            "replace": CorrectionAction.REPLACE,
            "keep": CorrectionAction.KEEP,
            "ignore": CorrectionAction.KEEP,
            "needs_human": CorrectionAction.NEEDS_REVIEW,
            "needs_review": CorrectionAction.NEEDS_REVIEW,
        }
        return mapping.get(action_str, CorrectionAction.NEEDS_REVIEW)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _fallback_decision(reason: str) -> ArbitrationDecision:
        return ArbitrationDecision(
            action=CorrectionAction.NEEDS_REVIEW,
            selected_candidate=None,
            corrections=[],
            confidence=0.0,
            reasoning=f"LLM fallback: {reason}",
            activated_parent_entities=[],
        )
