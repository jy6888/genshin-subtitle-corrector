"""Per-cue, per-stage FN diagnostic data structures and pure classification helpers.

No LLM client, no file IO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class StageStatus(StrEnum):
    DETECTOR_MISS = "DETECTOR_MISS"
    DISCOVERY_FILTERED_GT = "DISCOVERY_FILTERED_GT"
    EXPANSION_SKIPPED_NO_STABLE_ENTITY = "EXPANSION_SKIPPED_NO_STABLE_ENTITY"
    PHASE2_REJECTED_GT = "PHASE2_REJECTED_GT"
    REQUERY_NOT_REQUESTED = "REQUERY_NOT_REQUESTED"
    REQUERY_FAILED = "REQUERY_FAILED"
    WRONG_CANDIDATE_ONLY = "WRONG_CANDIDATE_ONLY"
    UNKNOWN = "UNKNOWN"


@dataclass
class FNStageTrace:
    cue_index: int
    raw_text: str
    gt_text: str
    gt_terms: list[str] = field(default_factory=list)
    detector_candidates: list[dict] = field(default_factory=list)
    discovery_candidates: list[dict] = field(default_factory=list)
    expanded_candidates: list[dict] = field(default_factory=list)
    phase2_input_candidates: list[dict] = field(default_factory=list)
    phase2_actions: list[str] = field(default_factory=list)
    requery_requested: bool = False
    requery_generated: bool = False
    expansion_skip_reasons: list[str] = field(default_factory=list)

    @property
    def stage_status(self) -> StageStatus:
        return classify_trace_stage(self)


def classify_trace_stage(trace: FNStageTrace) -> StageStatus:
    if not _contains_gt(trace.detector_candidates, trace.gt_terms):
        if trace.detector_candidates:
            return StageStatus.WRONG_CANDIDATE_ONLY
        return StageStatus.DETECTOR_MISS

    if not _contains_gt(trace.discovery_candidates, trace.gt_terms):
        return StageStatus.DISCOVERY_FILTERED_GT

    merged_before_phase2 = [*trace.discovery_candidates, *trace.expanded_candidates]
    if (
        _contains_gt(merged_before_phase2, trace.gt_terms)
        and not _contains_gt(trace.phase2_input_candidates, trace.gt_terms)
        and "missing_stable_entities" in trace.expansion_skip_reasons
    ):
        return StageStatus.EXPANSION_SKIPPED_NO_STABLE_ENTITY

    if _contains_gt(trace.phase2_input_candidates, trace.gt_terms):
        if any(action.upper() == "REPLACE" for action in trace.phase2_actions):
            return StageStatus.UNKNOWN
        return StageStatus.PHASE2_REJECTED_GT

    if trace.requery_requested and not trace.requery_generated:
        return StageStatus.REQUERY_FAILED
    if not trace.requery_requested:
        return StageStatus.REQUERY_NOT_REQUESTED
    return StageStatus.UNKNOWN


def _contains_gt(candidates: list[dict], gt_terms: list[str]) -> bool:
    values = {candidate.get("value", "") for candidate in candidates}
    return any(term in values for term in gt_terms)
