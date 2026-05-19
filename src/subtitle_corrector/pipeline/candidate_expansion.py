from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from loguru import logger
from pypinyin import lazy_pinyin
from rapidfuzz import fuzz

from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher, TerminologyEntry
from subtitle_corrector.pipeline.candidate_alignment import build_aligned_candidate
from subtitle_corrector.schemas import ChunkSemanticTimeline, CueObservation


HOT_CATEGORIES = {
    "character",
    "weapon",
    "artifact",
    "artifact_piece",
    "skill",
    "constellation",
    "constellation_group",
    "enemy",
    "domain",
}


class CandidateExpander(Protocol):
    def expand_observation(self, observation: CueObservation, stable_entities: set[str]) -> list[dict]:
        ...


@dataclass
class CandidateExpansionSummary:
    observations: int = 0
    observations_with_stable_entities: int = 0
    input_candidates: int = 0
    raw_proposals: int = 0
    added_candidates: int = 0
    output_candidates: int = 0
    capped_observations: int = 0
    elapsed_seconds: float = 0.0
    proposed_by_source: dict[str, int] = field(default_factory=dict)
    added_by_source: dict[str, int] = field(default_factory=dict)
    retained_by_source: dict[str, int] = field(default_factory=dict)

    def to_log_message(self) -> str:
        return (
            "CandidateExpansion: observations={observations}, stable_obs={stable}, "
            "input_candidates={input_candidates}, raw_proposals={raw}, "
            "added={added}, output_candidates={output}, capped_obs={capped}, "
            "elapsed={elapsed:.2f}s, proposed_by_source={proposed}, "
            "added_by_source={added_by_source}, retained_by_source={retained}"
        ).format(
            observations=self.observations,
            stable=self.observations_with_stable_entities,
            input_candidates=self.input_candidates,
            raw=self.raw_proposals,
            added=self.added_candidates,
            output=self.output_candidates,
            capped=self.capped_observations,
            elapsed=self.elapsed_seconds,
            proposed=self.proposed_by_source,
            added_by_source=self.added_by_source,
            retained=self.retained_by_source,
        )


@dataclass(frozen=True)
class _SurfaceMatch:
    surface: str
    start_char: int
    end_char: int
    score: float
    source: str
    entry: TerminologyEntry
    match_kind: str = ""
    surface_coverage: float = 0.0
    target_coverage: float = 0.0


@dataclass(frozen=True)
class _MergeStats:
    added: int
    output_candidates: int
    capped: bool
    added_by_source: dict[str, int]
    retained_by_source: dict[str, int]


class CandidateExpansionEngine:
    """Add deterministic repair candidates after topic reduction, before Phase 2."""

    def __init__(self, expanders: list[CandidateExpander], max_candidates_per_cue: int = 4) -> None:
        self.expanders = expanders
        self.max_candidates_per_cue = max_candidates_per_cue
        self.last_summary = CandidateExpansionSummary()

    def expand(self, timelines: list[ChunkSemanticTimeline], clusters: list) -> int:
        started = perf_counter()
        summary = CandidateExpansionSummary()
        added = 0
        for timeline in timelines:
            for observation in timeline.observations:
                summary.observations += 1
                summary.input_candidates += len(observation.candidates)
                stable_entities = self._stable_entities_for(observation.cue_index, clusters)
                if stable_entities:
                    summary.observations_with_stable_entities += 1
                new_candidates: list[dict] = []
                for expander in self.expanders:
                    expanded = expander.expand_observation(observation, stable_entities)
                    new_candidates.extend(expanded)
                    summary.raw_proposals += len(expanded)
                    _increment_counts(summary.proposed_by_source, expanded)
                merge_stats = self._merge_candidates(observation, new_candidates)
                added += merge_stats.added
                summary.added_candidates += merge_stats.added
                summary.output_candidates += merge_stats.output_candidates
                if merge_stats.capped:
                    summary.capped_observations += 1
                _merge_counts(summary.added_by_source, merge_stats.added_by_source)
                _merge_counts(summary.retained_by_source, merge_stats.retained_by_source)
        summary.elapsed_seconds = perf_counter() - started
        self.last_summary = summary
        logger.info(summary.to_log_message())
        return added

    @staticmethod
    def _stable_entities_for(cue_index: int, clusters: list) -> set[str]:
        entities: set[str] = set()
        for cluster in clusters:
            start, end = getattr(cluster, "temporal_range", (0, -1))
            if start <= cue_index <= end:
                entities.update(getattr(cluster, "dominant_entities", []) or [])
        return entities

    def _merge_candidates(self, observation: CueObservation, new_candidates: list[dict]) -> _MergeStats:
        by_key: dict[tuple, dict] = {}
        for candidate in observation.candidates:
            by_key[self._candidate_key(candidate)] = candidate

        added = 0
        added_by_source: Counter[str] = Counter()
        for candidate in new_candidates:
            key = self._candidate_key(candidate)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = candidate
                added += 1
                added_by_source[candidate.get("source", "unknown")] += 1
            elif candidate.get("score", 0.0) > existing.get("score", 0.0):
                by_key[key] = candidate

        ordered = sorted(
            by_key.values(),
            key=lambda c: (c.get("score", 0.0), c.get("source") != "exact"),
            reverse=True,
        )
        retained = ordered[: self.max_candidates_per_cue]
        observation.candidates = retained
        retained_by_source = Counter(
            candidate.get("source", "unknown") for candidate in retained
        )
        return _MergeStats(
            added=added,
            output_candidates=len(retained),
            capped=len(ordered) > self.max_candidates_per_cue,
            added_by_source=dict(added_by_source),
            retained_by_source=dict(retained_by_source),
        )

    @staticmethod
    def _candidate_key(candidate: dict) -> tuple:
        return (
            candidate.get("surface", ""),
            candidate.get("value", ""),
            candidate.get("start_char", 0),
            candidate.get("end_char", 0),
        )


class EntityConsistencyCandidateExpander:
    """Use reducer-confirmed stable entities to recover likely ASR variants."""

    def __init__(self, matcher: FuzzyTerminologyMatcher, min_score: float = 0.72) -> None:
        self.matcher = matcher
        self.min_score = min_score

    def expand_observation(self, observation: CueObservation, stable_entities: set[str]) -> list[dict]:
        if not stable_entities:
            return []

        choices = self.matcher._build_choices()
        all_cands: list[dict] = []
        for entity in stable_entities:
            entry = choices.get(entity)
            if entry is None or _category(entry) not in HOT_CATEGORIES or entry.term in observation.text:
                continue
            # Use CJK windows + alignment evidence
            best: dict | None = None
            for surface, start, end in _iter_cjk_windows(
                observation.text,
                min_len=max(2, len(entry.term) - 1),
                max_len=min(len(observation.text), len(entry.term) + 1),
            ):
                aligned = build_aligned_candidate(
                    observation.text, start, end, entry.term,
                    source="entity_consistency",
                    category=entry.category or "",
                    parent_entity=entry.parent_entity or "",
                    allow_prefix=False,
                    expansion_policy="repair_to_canonical",
                )
                if aligned is None or aligned["alignment_score"] < self.min_score:
                    continue
                if best is None or aligned["alignment_score"] > best.get("alignment_score", 0):
                    best = aligned
            if best is not None:
                all_cands.append(best)

        return sorted(all_cands, key=lambda c: c.get("score", 0), reverse=True)[:2]


class LongEntityVariantExpander:
    """Recover long entity ASR errors (3-6 chars) using pinyin over CJK windows.

    Targets the current top missed-entity patterns:
      - 纳瓦雷特/纳维特 → 那维莱特 (prefix/partial pinyin match)
      - 鹿野苑 → 鹿野院平藏 (short surface for long entity)
      - Fu li na / Fu lin na → 芙宁娜 (already handled by JiebaSpanDetector;
        this expander adds a second chance with wider window sizes)

    Only operates on reducer-confirmed stable entities, one best candidate
    per cue per entity.  Does not do full-database scanning.
    """

    def __init__(self, matcher: FuzzyTerminologyMatcher, min_score: float = 0.70) -> None:
        self.matcher = matcher
        self.min_score = min_score

    def expand_observation(
        self, observation: CueObservation, stable_entities: set[str],
    ) -> list[dict]:
        if not stable_entities:
            return []

        choices = self.matcher._build_choices()
        matches: list[_SurfaceMatch] = []

        for entity in stable_entities:
            entry = choices.get(entity)
            if entry is None or _category(entry) not in HOT_CATEGORIES:
                continue
            if entry.term in observation.text:
                continue  # already correct

            # Search only within CJK runs — never cross latin/digit boundaries
            best: _SurfaceMatch | None = None
            for surface, start, end in _iter_cjk_windows(
                observation.text,
                min_len=max(2, len(entry.term) - 2),
                max_len=min(len(observation.text), len(entry.term) + 2),
            ):
                start, end = _trim_operation_tail(observation.text, start, end)
                if end - start < 2:
                    continue
                surface = observation.text[start:end]
                if surface == entry.term:
                    continue
                aligned = _build_long_entity_candidate(
                    observation.text, start, end, entry.term,
                    source="long_entity_variant",
                    category=entry.category or "",
                    parent_entity=entry.parent_entity or "",
                    allow_prefix=True,
                    min_score=self.min_score,
                    relaxed_min_score=0.62,
                )
                if aligned is None:
                    continue
                if best is None or aligned["alignment_score"] > best.score:
                    best = _SurfaceMatch(
                        surface=aligned["surface"],
                        start_char=aligned["start_char"],
                        end_char=aligned["end_char"],
                        score=aligned["alignment_score"],
                        source="long_entity_variant",
                        entry=entry,
                        match_kind=aligned.get("match_kind", ""),
                        surface_coverage=aligned.get("surface_coverage", 0.0),
                        target_coverage=aligned.get("target_coverage", 0.0),
                    )

            if best is not None:
                matches.append(best)

        return [_candidate_dict(m) for m in _best_per_value(matches)]


class LocalTermCandidateExpander:
    """Wide local n-gram scan over hot terminology categories.

    When requires_stable_entity is True (default), only scans against terms
    belonging to stable entities.  When False, scans against all hot-category
    entries (more noise, but higher recall).
    """

    def __init__(
        self, matcher: FuzzyTerminologyMatcher,
        min_score: float = 0.74,
        requires_stable_entity: bool = True,
    ) -> None:
        self.matcher = matcher
        self.min_score = min_score
        self.requires_stable_entity = requires_stable_entity
        self._entries = _unique_hot_entries(matcher)

    def expand_observation(self, observation: CueObservation, stable_entities: set[str]) -> list[dict]:
        if self.requires_stable_entity and not stable_entities:
            return []
        entries = self._entries_for(stable_entities) if self.requires_stable_entity else self._entries
        all_cands: list[dict] = []
        for entry in entries:
            if entry.term in observation.text:
                continue
            best: dict | None = None
            for surface, start, end in _iter_cjk_windows(
                observation.text,
                min_len=2,
                max_len=min(6, len(observation.text)),
            ):
                if surface == entry.term:
                    continue
                aligned = build_aligned_candidate(
                    observation.text, start, end, entry.term,
                    source="local_term",
                    category=entry.category or "",
                    parent_entity=entry.parent_entity or "",
                    allow_prefix=False,
                    expansion_policy="repair_to_canonical",
                )
                if aligned is None or aligned["alignment_score"] < self.min_score:
                    continue
                if best is None or aligned["alignment_score"] > best.get("alignment_score", 0):
                    best = aligned
            if best is not None:
                all_cands.append(best)
        return sorted(all_cands, key=lambda c: c.get("score", 0), reverse=True)[:2]

    def _entries_for(self, stable_entities: set[str]) -> list[TerminologyEntry]:
        if not stable_entities:
            return self._entries
        choices = self.matcher._build_choices()
        entries: list[TerminologyEntry] = []
        seen: set[str] = set()
        for entity in stable_entities:
            entry = choices.get(entity)
            if entry is None or entry.term in seen or _category(entry) not in HOT_CATEGORIES:
                continue
            seen.add(entry.term)
            entries.append(entry)
        return entries


def _unique_hot_entries(matcher: FuzzyTerminologyMatcher) -> list[TerminologyEntry]:
    seen: set[str] = set()
    entries: list[TerminologyEntry] = []
    for entry in matcher._build_choices().values():
        if entry.term in seen or _category(entry) not in HOT_CATEGORIES:
            continue
        seen.add(entry.term)
        entries.append(entry)
    return entries


def _iter_windows(text: str, min_len: int, max_len: int):
    upper = min(max_len, len(text))
    for length in range(max(min_len, 2), upper + 1):
        for start in range(0, len(text) - length + 1):
            surface = text[start : start + length]
            if _contains_cjk(surface):
                yield surface, start, start + length


def _is_term_prefix(surface: str, term: str) -> bool:
    if len(surface) < 3:
        return False
    surface_py = _pinyin(surface)
    term_py = _pinyin(term)
    return bool(surface_py and term_py and term_py.startswith(surface_py))


def _pinyin(text: str) -> str:
    return " ".join(lazy_pinyin(text, errors="ignore"))


def _pinyin_tokens(text: str) -> list[str]:
    return lazy_pinyin(text, errors="ignore")


def _build_long_entity_candidate(
    text: str,
    start: int,
    end: int,
    target: str,
    source: str,
    category: str,
    parent_entity: str,
    allow_prefix: bool,
    min_score: float,
    relaxed_min_score: float,
) -> dict | None:
    aligned = build_aligned_candidate(
        text, start, end, target,
        source=source,
        category=category,
        parent_entity=parent_entity,
        allow_prefix=allow_prefix,
        expansion_policy="repair_to_canonical",
    )
    if aligned is not None and aligned["alignment_score"] >= min_score:
        return aligned

    relaxed = _build_relaxed_long_entity_candidate(
        text, start, end, target,
        source=source,
        category=category,
        parent_entity=parent_entity,
        relaxed_min_score=relaxed_min_score,
    )
    if relaxed is None:
        return None
    if aligned is not None and aligned["alignment_score"] >= relaxed["alignment_score"]:
        return aligned
    return relaxed


def _build_relaxed_long_entity_candidate(
    text: str,
    start: int,
    end: int,
    target: str,
    source: str,
    category: str,
    parent_entity: str,
    relaxed_min_score: float,
) -> dict | None:
    if start >= end or start < 0 or end > len(text):
        return None
    surface = text[start:end]
    if not surface or surface == target:
        return None
    if any(ch.isascii() and (ch.isdigit() or ch.isalpha()) for ch in surface):
        return None

    surface_tokens = _pinyin_tokens(surface)
    target_tokens = _pinyin_tokens(target)
    if not surface_tokens or not target_tokens:
        return None

    repeated_first_syllable = (
        len(target) == 3
        and len(surface) == 2
        and len(surface_tokens) == 2
        and all(_syllable_similar(token, target_tokens[0]) for token in surface_tokens)
    )
    if repeated_first_syllable:
        return {
            "surface": surface,
            "value": target,
            "score": 0.68,
            "start_char": start,
            "end_char": end,
            "source": source,
            "category": category,
            "parent_entity": parent_entity,
            "match_kind": "stable_entity_repeated_syllable",
            "alignment_score": 0.68,
            "surface_coverage": 1.0,
            "target_coverage": round(2 / len(target_tokens), 3),
            "extra_surface_ratio": 0.0,
            "expansion_policy": "repair_to_canonical",
            "evidence_type": source,
        }

    if len(target) < 4 or len(surface) < 3:
        return None

    surface_coverage = _ordered_surface_coverage(surface_tokens, target_tokens)
    extra_surface_ratio = 1.0 - surface_coverage
    target_coverage = min(1.0, surface_coverage * len(surface_tokens) / len(target_tokens))
    if surface_coverage < 0.75 or extra_surface_ratio > 0.25 or target_coverage < 0.50:
        return None

    phonetic_ratio = fuzz.ratio(" ".join(surface_tokens), " ".join(target_tokens)) / 100.0
    char_ratio = fuzz.ratio(surface, target) / 100.0
    score = max(phonetic_ratio, char_ratio * 0.85)
    if score < relaxed_min_score:
        return None

    return {
        "surface": surface,
        "value": target,
        "score": round(score, 3),
        "start_char": start,
        "end_char": end,
        "source": source,
        "category": category,
        "parent_entity": parent_entity,
        "match_kind": "stable_entity_relaxed",
        "alignment_score": round(score, 3),
        "surface_coverage": round(surface_coverage, 3),
        "target_coverage": round(target_coverage, 3),
        "extra_surface_ratio": round(extra_surface_ratio, 3),
        "expansion_policy": "repair_to_canonical",
        "evidence_type": source,
    }


def _ordered_surface_coverage(surface_tokens: list[str], target_tokens: list[str]) -> float:
    target_index = 0
    matched = 0
    for token in surface_tokens:
        for idx in range(target_index, len(target_tokens)):
            if _syllable_similar(token, target_tokens[idx]):
                matched += 1
                target_index = idx + 1
                break
    return matched / len(surface_tokens) if surface_tokens else 0.0


def _syllable_similar(left: str, right: str) -> bool:
    return fuzz.ratio(left, right) >= 65


def _best_per_value(matches: list[_SurfaceMatch]) -> list[_SurfaceMatch]:
    best: dict[str, _SurfaceMatch] = {}
    for match in matches:
        existing = best.get(match.entry.term)
        if existing is None or match.score > existing.score:
            best[match.entry.term] = match
    return sorted(best.values(), key=lambda m: m.score, reverse=True)


def _candidate_dict(match: _SurfaceMatch) -> dict:
    c = {
        "surface": match.surface,
        "value": match.entry.term,
        "score": round(match.score, 3),
        "category": _category(match.entry),
        "parent_entity": match.entry.parent_entity or "",
        "start_char": match.start_char,
        "end_char": match.end_char,
        "source": match.source,
        "expansion_policy": "repair_to_canonical",
        "evidence_type": match.source,
    }
    if match.match_kind:
        c["match_kind"] = match.match_kind
        c["alignment_score"] = round(match.score, 3)
        c["surface_coverage"] = round(match.surface_coverage, 3)
        c["target_coverage"] = round(match.target_coverage, 3)
    return c


def _category(entry: TerminologyEntry) -> str:
    return entry.category or ""


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


# CJK run extraction for span boundary safety
_CJK_RE = __import__("re").compile(r"[\u4e00-\u9fff]+")


def _iter_cjk_runs(text: str):
    """Yield (start_offset, cjk_text) for each continuous CJK segment."""
    for match in _CJK_RE.finditer(text):
        yield match.start(), match.group(0)


def _iter_cjk_windows(text: str, min_len: int, max_len: int):
    """Like _iter_windows but only within CJK runs (no cross-boundary spans)."""
    for base_offset, run_text in _iter_cjk_runs(text):
        upper = min(max_len, len(run_text))
        for length in range(max(min_len, 2), upper + 1):
            for start in range(0, len(run_text) - length + 1):
                surface = run_text[start : start + length]
                yield surface, base_offset + start, base_offset + start + length


def _trim_operation_tail(text: str, start: int, end: int) -> tuple[int, int]:
    surface = text[start:end]
    for suffix in ("大招",):
        if surface.endswith(suffix):
            end -= len(suffix)
            surface = text[start:end]
    if surface.endswith("长") and end < len(text) and text[end : end + 1].lower() == "e":
        end -= 1
    return start, end




def _increment_counts(target: dict[str, int], candidates: list[dict]) -> None:
    for candidate in candidates:
        source = candidate.get("source", "unknown")
        target[source] = target.get(source, 0) + 1


def _merge_counts(target: dict[str, int], counts: dict[str, int]) -> None:
    for key, value in counts.items():
        target[key] = target.get(key, 0) + value
