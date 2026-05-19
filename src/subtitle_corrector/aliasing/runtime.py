from __future__ import annotations

from pathlib import Path

from loguru import logger

from subtitle_corrector.aliasing.asr_alias import (
    AsrAliasCandidate,
    read_alias_candidates,
)


class AsrAliasRuntime:
    """Load reviewed ASR aliases and produce exact-surface repair candidates."""

    def __init__(self, aliases: list[AsrAliasCandidate]) -> None:
        self._approved = [a for a in aliases if a.review_status == "approved"]
        self._needs_context = [
            a for a in aliases if a.review_status == "needs_context"
        ]

    @classmethod
    def from_csv(cls, path: str | Path | None) -> "AsrAliasRuntime":
        if path is None:
            return cls([])
        csv_path = Path(path)
        if not csv_path.exists():
            logger.debug("ASR alias file not found, runtime disabled: {}", csv_path)
            return cls([])
        aliases = read_alias_candidates(csv_path)
        logger.info("loaded {} ASR alias candidates from {}", len(aliases), csv_path)
        return cls(aliases)

    def lookup(self, text: str, active_entities: set[str]) -> list[dict]:
        results: list[dict] = []
        for alias in self._approved:
            results.extend(self._match_alias(text, alias))
        for alias in self._needs_context:
            if alias.canonical_term not in active_entities:
                continue
            results.extend(self._match_alias(text, alias))
        return results

    def iter_policy_entries(self):
        yield from self._approved
        yield from self._needs_context

    def _match_alias(self, text: str, alias: AsrAliasCandidate) -> list[dict]:
        matches: list[dict] = []
        pos = text.find(alias.alias_surface)
        while pos >= 0:
            matches.append(self._make_candidate(alias, pos))
            pos = text.find(alias.alias_surface, pos + 1)
        return matches

    @staticmethod
    def _category_for(_canonical: str) -> str:
        return "character"

    def _make_candidate(self, alias: AsrAliasCandidate, start: int) -> dict:
        return {
            "surface": alias.alias_surface,
            "value": alias.canonical_term,
            "score": alias.confidence,
            "source": "asr_alias",
            "category": self._category_for(alias.canonical_term),
            "parent_entity": alias.canonical_term,
            "start_char": start,
            "end_char": start + len(alias.alias_surface),
            "metadata": {
                "intent": "repair_candidate",
                "alias_type": alias.alias_type,
                "review_status": alias.review_status,
                "risk_flags": list(alias.risk_flags),
            },
        }
