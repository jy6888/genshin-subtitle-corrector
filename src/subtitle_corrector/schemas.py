from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SubtitleFormat(str, Enum):
    SRT = "srt"
    VTT = "vtt"
    ASS = "ass"
    UNKNOWN = "unknown"


class CorrectionAction(str, Enum):
    KEEP = "keep"
    REPLACE = "replace"
    NEEDS_REVIEW = "needs_review"


class SubtitleCue(BaseModel):
    index: int
    start_ms: int
    end_ms: int
    text: str
    style: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubtitleDocument(BaseModel):
    format: SubtitleFormat
    cues: list[SubtitleCue]
    source_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedText(BaseModel):
    original: str
    normalized: str
    operations: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    value: str
    source: str
    score: float = Field(ge=0.0, le=1.0)
    explanation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Terminology(BaseModel):
    term: str
    aliases: list[str] = Field(default_factory=list)
    category: str | None = None
    game_title: str | None = None
    source: str | None = None
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0)
    parent_entity: Optional[str] = Field(default=None)


class DetectionResult(BaseModel):
    detector: str
    cue_index: int
    risk_score: float = Field(ge=0.0, le=1.0)
    reason: str
    candidates: list[Candidate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FusedRisk(BaseModel):
    cue_index: int
    risk_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)


class ArbitrationRequest(BaseModel):
    cue: SubtitleCue
    context_before: list[SubtitleCue] = Field(default_factory=list)
    context_after: list[SubtitleCue] = Field(default_factory=list)
    candidates: list[Candidate]
    risk: FusedRisk


class CorrectionItem(BaseModel):
    """LLM 输出的修正项——只描述"哪个词改成什么"，坐标由执行层反向映射。

    start_char / end_char 由 Executor 从检测器 Span 或正则查找回填，
    LLM 不参与坐标定位。
    """
    original_word: str
    corrected_word: str
    confidence: float = Field(ge=0.0, le=1.0)
    # Backfilled by executor:
    start_char: int = Field(default=0, ge=0)
    end_char: int = Field(default=0, ge=0)


class Span(BaseModel):
    surface_text: str
    start_char: int
    end_char: int
    candidate_value: str
    score: float = Field(ge=0.0, le=1.0)
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArbitrationDecision(BaseModel):
    action: CorrectionAction
    selected_candidate: Candidate | None = None
    corrections: list[CorrectionItem] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    activated_parent_entities: list[str] = Field(default_factory=list)


class RepairResult(BaseModel):
    cue_index: int
    original_text: str
    repaired_text: str
    action: CorrectionAction
    confidence: float
    explanation: str


class ReplaceSpan(BaseModel):
    """A planned character-level replacement with provenance tracking."""

    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    replacement: str
    provenance: list[str] = Field(default_factory=list)
    provenance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class EntityMemorySnapshot(BaseModel):
    """A point-in-time snapshot of the entity memory state for one cue."""

    cue_index: int
    entities_before: dict[str, float] = Field(default_factory=dict)
    entities_activated: list[str] = Field(default_factory=list)
    entities_after: dict[str, float] = Field(default_factory=dict)
    decays_applied: bool = False


# ── Chunk-level semantic state structures ────────────────────────────────────


class CueObservation(BaseModel):
    """Per-cue structured observation from the detection layer.
    This is the INPUT to Phase 1 LLM — not its output.
    """

    cue_index: int
    text: str
    context_before: str = ""
    context_after: str = ""
    candidates: list[dict] = Field(default_factory=list)
    active_entities: dict[str, float] = Field(default_factory=dict)
    active_categories: list[str] = Field(default_factory=list)


class CompressedContextWindow(BaseModel):
    """Discourse memory around a dominant entity within a chunk."""

    entity: str
    start_cue: int
    end_cue: int
    context_sequence: list[dict] = Field(default_factory=list)
    supporting_categories: list[str] = Field(default_factory=list)
    semantic_density: float = Field(default=0.0, ge=0.0, le=1.0)


class ChunkSemanticTimeline(BaseModel):
    """Code-built structured input for Phase 1 LLM.
    Stitches all CueObservations in time order with entity persistence,
    category flow, and compressed discourse windows.
    """

    chunk_index: int
    observations: list[CueObservation] = Field(default_factory=list)
    entity_persistence: dict[str, float] = Field(default_factory=dict)
    category_flow: list[dict] = Field(default_factory=list)
    compressed_context_windows: list[CompressedContextWindow] = Field(default_factory=list)


class SemanticFilterOutput(BaseModel):
    """Phase 1 LLM output — semantic observations, NOT repair decisions."""

    chunk_index: int
    confirmed_entities: list[dict] = Field(default_factory=list)
    dominant_categories: list[str] = Field(default_factory=list)
    semantic_signals: list[str] = Field(default_factory=list)
    possible_transition: bool = False
    transition_region: dict | None = None
    detector_noise: list[dict] = Field(default_factory=list)
