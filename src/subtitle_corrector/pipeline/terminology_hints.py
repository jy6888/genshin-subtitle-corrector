"""术语无候选 REQUERY hint builder.

当 cue 没有候选词时，基于活跃实体生成窄候选表面，供 Phase2 LLM 做术语拼音近邻 REQUERY。
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitle_corrector.pipeline.candidate_alignment import build_aligned_candidate

HOT_HINT_CATEGORIES = {
    "character", "weapon", "artifact", "artifact_piece",
    "skill", "constellation", "constellation_group",
}


@dataclass
class TerminologyRequeryHintBuilder:
    matcher: object
    min_alignment: float = 0.72
    max_hints: int = 2

    def find_requery_hints(self, text: str, stable_entities: set[str]) -> list[dict]:
        if not text or not stable_entities:
            return []

        choices = self.matcher._build_choices()
        entries = [
            entry for entry in choices.values()
            if self._entry_allowed(entry, stable_entities)
            and getattr(entry, "term", "") not in text
        ]

        hints: list[tuple[float, dict]] = []
        for entry in entries:
            term = getattr(entry, "term", "")
            if len(term) < 2:
                continue
            for surface, start, end in _iter_cjk_windows(
                text, min_len=max(2, len(term) - 1),
                max_len=min(len(text), len(term) + 1),
            ):
                if surface == term:
                    continue
                aligned = build_aligned_candidate(
                    text, start, end, term,
                    source="terminology_requery_hint",
                    category=getattr(entry, "category", "") or "",
                    parent_entity=getattr(entry, "parent_entity", "") or "",
                    allow_prefix=True,
                    expansion_policy="repair_to_canonical",
                )
                if aligned is None:
                    continue
                score = aligned.get("alignment_score", 0.0)
                if score < self.min_alignment:
                    continue
                hints.append((score, {"suspect_surface": surface, "target_hint": term}))

        deduped: dict[tuple[str, str], tuple[float, dict]] = {}
        for score, hint in hints:
            key = (hint["suspect_surface"], hint["target_hint"])
            existing = deduped.get(key)
            if existing is None or score > existing[0]:
                deduped[key] = (score, hint)

        ordered = sorted(deduped.values(), key=lambda item: item[0], reverse=True)
        return [hint for _, hint in ordered[: self.max_hints]]

    @staticmethod
    def _entry_allowed(entry: object, stable_entities: set[str]) -> bool:
        term = getattr(entry, "term", "") or ""
        parent = getattr(entry, "parent_entity", "") or ""
        category = getattr(entry, "category", "") or ""
        if category not in HOT_HINT_CATEGORIES:
            return False
        return term in stable_entities or parent in stable_entities


def _iter_cjk_windows(text: str, min_len: int, max_len: int):
    for start, run in _iter_cjk_runs(text):
        upper = min(max_len, len(run))
        for length in range(max(min_len, 2), upper + 1):
            for offset in range(0, len(run) - length + 1):
                yield run[offset: offset + length], start + offset, start + offset + length


def _iter_cjk_runs(text: str):
    start = None
    chars: list[str] = []
    for index, char in enumerate(text):
        if "一" <= char <= "鿿":
            if start is None:
                start = index
            chars.append(char)
        elif start is not None:
            yield start, "".join(chars)
            start = None
            chars = []
    if start is not None:
        yield start, "".join(chars)
