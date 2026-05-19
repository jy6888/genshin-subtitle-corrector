"""Fault-tolerant JSONL (JSON Lines) stream parser.

Unlike batch JSON-array parsing, JSONL processes one line at a time.  If a
single line is malformed (missing bracket, stray comma, etc.) the parser
logs the error and continues — one broken line never poisons the entire batch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ParseStats:
    """Statistics gathered during a JSONL parse run."""

    total_lines: int = 0
    success_count: int = 0
    failure_count: int = 0
    skipped_empty: int = 0
    error_details: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    """Container returned by RobustJSONLParser.parse()."""

    objects: list[dict[str, Any]]
    stats: ParseStats


class RobustJSONLParser:
    """Parse LLM output that uses the JSONL convention (one JSON object per
    line, **no** enclosing ``[{…}, {…}]`` array).

    The parser is deliberately lenient:

    - Lines containing only whitespace are skipped.
    - Lines wrapped in markdown code-fence back-ticks are auto-stripped.
    - JSON decode errors on a single line are caught, logged as warnings,
      and **do not** abort processing of the remaining lines.
    """

    def __init__(self, strict: bool = False) -> None:
        """*strict* – if ``True``, re-raise the first parse error instead of
        skipping the offending line.  Useful during development / debugging.
        """
        self.strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, raw: str) -> ParseResult:
        """Parse multi-line *raw* text as JSONL.

        Returns a :class:`ParseResult` whose ``.objects`` list contains every
        successfully parsed JSON dict, and ``.stats`` provides a diagnostic
        summary.
        """
        stats = ParseStats()
        objects: list[dict[str, Any]] = []

        # Strip a surrounding markdown code fence block if present.
        text = self._strip_fence(raw)

        lines = text.splitlines()

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()

            # ---- empty / whitespace-only --------------------------------------------------
            if not stripped:
                stats.skipped_empty += 1
                continue

            # ---- back-tick fence lines --------------------------------------------------
            if stripped in ("```", "```json"):
                stats.skipped_empty += 1
                continue

            # ---- attempt parse -----------------------------------------------------------
            stats.total_lines += 1
            try:
                obj = json.loads(stripped)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
                objects.append(obj)
                stats.success_count += 1
            except Exception as exc:
                stats.failure_count += 1
                detail = f"line {line_no}: {exc} | raw={stripped[:120]}"
                stats.error_details.append(detail)
                logger.warning("[JSONL] 解析失败 — {}", detail)
                if self.strict:
                    raise

        return ParseResult(objects=objects, stats=stats)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_fence(raw: str) -> str:
        """If *raw* is wrapped in a markdown code fence, remove the fence."""
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence:
            return fence.group(1).strip()
        return text
