"""RequeryCandidateEngine: targeted re-candidate generation for REQUERY.

When Phase2 LLM requests REQUERY (suspect_surface → target_hint),
this engine generates a narrow set of candidates using phonetic alignment
against the hinted target only — no full-database scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from subtitle_corrector.pipeline.candidate_alignment import build_aligned_candidate
from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher


@dataclass
class RequeryResult:
    candidates: list[dict] = field(default_factory=list)
    generated: int = 0
    skipped_surface_mismatch: int = 0
    skipped_target_unknown: int = 0
    skipped_too_many: int = 0
    generated_by_type: dict[str, int] = field(default_factory=dict)
    skipped_by_type: dict[str, int] = field(default_factory=dict)


class RequeryCandidateEngine:
    """Generate candidates from Phase2 REQUERY requests."""

    def __init__(
        self,
        matcher: FuzzyTerminologyMatcher,
        max_per_cue: int = 2,
        asr_alias_runtime: object | None = None,
        team_comp_parser: object | None = None,
    ) -> None:
        self.matcher = matcher
        self.max_per_cue = max_per_cue
        self.asr_alias_runtime = asr_alias_runtime
        self.team_comp_parser = team_comp_parser
        self.last_result = RequeryResult()

    def build_candidates(
        self, requests: list, cues: list, clusters: list,
    ) -> list[dict]:
        """Process requery requests and return new candidates."""
        result = RequeryResult()
        all_candidates: list[dict] = []

        choices = self.matcher._build_choices()

        # Per-cue requery count
        cue_counts: dict[int, int] = {}

        for req in requests:
            ci = req.cue_index
            requery_type = getattr(req, "metadata", {}).get(
                "requery_type",
                "terminology_phonetic",
            )
            if ci < 0 or ci >= len(cues):
                self._increment(result.skipped_by_type, requery_type)
                continue

            # Limit per cue
            if cue_counts.get(ci, 0) >= self.max_per_cue:
                result.skipped_too_many += 1
                self._increment(result.skipped_by_type, requery_type)
                continue

            # suspect_surface must exist in the cue text
            text = cues[ci].text
            pos = text.find(req.suspect_surface)
            if pos == -1:
                result.skipped_surface_mismatch += 1
                self._increment(result.skipped_by_type, requery_type)
                continue

            if requery_type == "asr_alias":
                if self.asr_alias_runtime is None:
                    self._increment(result.skipped_by_type, requery_type)
                    continue
                generated = 0
                active_entities = self._active_entities_for(ci, clusters)
                for candidate in self.asr_alias_runtime.lookup(text, active_entities):
                    if candidate.get("surface") != req.suspect_surface:
                        continue
                    candidate.setdefault("expansion_policy", "repair_to_canonical")
                    candidate.setdefault("evidence_type", requery_type)
                    candidate["cue_index"] = ci
                    all_candidates.append(candidate)
                    result.generated += 1
                    generated += 1
                    cue_counts[ci] = cue_counts.get(ci, 0) + 1
                    self._increment(result.generated_by_type, requery_type)
                    if cue_counts[ci] >= self.max_per_cue:
                        break
                if generated == 0:
                    self._increment(result.skipped_by_type, requery_type)
                continue

            if requery_type == "team_comp_alias":
                if self.team_comp_parser is None:
                    self._increment(result.skipped_by_type, requery_type)
                    continue
                candidate = self.team_comp_parser.requery_alias_candidate(
                    text,
                    req.suspect_surface,
                )
                if candidate is None:
                    self._increment(result.skipped_by_type, requery_type)
                    continue
                candidate.setdefault("expansion_policy", "repair_to_canonical")
                candidate.setdefault("evidence_type", requery_type)
                candidate["cue_index"] = ci
                all_candidates.append(candidate)
                result.generated += 1
                cue_counts[ci] = cue_counts.get(ci, 0) + 1
                self._increment(result.generated_by_type, requery_type)
                continue

            # target_hint must exist in terminology
            target = req.target_hint.strip()
            if target not in choices:
                result.skipped_target_unknown += 1
                self._increment(result.skipped_by_type, requery_type)
                continue

            entry = choices[target]
            aligned = build_aligned_candidate(
                text, pos, pos + len(req.suspect_surface),
                target,
                source="requery_phonetic",
                category=entry.category or "",
                parent_entity=entry.parent_entity or "",
                allow_prefix=True,
                expansion_policy="repair_to_canonical",
            )
            if aligned is not None:
                aligned["cue_index"] = ci
                aligned["score"] = aligned.get("alignment_score", 0)
                all_candidates.append(aligned)
                result.generated += 1
                cue_counts[ci] = cue_counts.get(ci, 0) + 1
                self._increment(result.generated_by_type, requery_type)
            else:
                self._increment(result.skipped_by_type, requery_type)

        logger.info(
            "RequeryCandidateEngine: {} requests → {} candidates, "
            "skipped_surface={}, skipped_target={}, skipped_count={}",
            len(requests), result.generated,
            result.skipped_surface_mismatch, result.skipped_target_unknown,
            result.skipped_too_many,
        )
        result.candidates = list(all_candidates)
        self.last_result = result
        return all_candidates

    @staticmethod
    def _active_entities_for(cue_index: int, clusters: list) -> set[str]:
        entities: set[str] = set()
        for cluster in clusters:
            start, end = getattr(cluster, "temporal_range", (0, -1))
            if start <= cue_index <= end:
                entities.update(getattr(cluster, "dominant_entities", []) or [])
        return entities

    @staticmethod
    def _increment(target: dict[str, int], key: str) -> None:
        target[key] = target.get(key, 0) + 1
