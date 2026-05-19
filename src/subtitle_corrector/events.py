"""Standardised event models and append-only event log for the Discovery Layer.

Architecture: Pass 1 is a PURE OBSERVER.  It outputs structured observation
events only — never final corrections, never memory writes, never subtitle
edits.  The Reducer / Consensus Layer alone converges on truth.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Type, TypeVar

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReasonCode(str, Enum):
    """Structured reason codes — no free-text reasoning allowed."""

    PHONETIC_MATCH = "PHONETIC_MATCH"
    CONTEXT_ALIGNMENT = "CONTEXT_ALIGNMENT"
    CONTEXT_CONFLICT = "CONTEXT_CONFLICT"
    ENTITY_MEMORY_MATCH = "ENTITY_MEMORY_MATCH"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    VOCABULARY_SHIFT = "VOCABULARY_SHIFT"
    ENTITY_CLUSTER_CHANGE = "ENTITY_CLUSTER_CHANGE"
    GRAMMAR_ERROR = "GRAMMAR_ERROR"
    CATEGORY_MISMATCH = "CATEGORY_MISMATCH"
    UNKNOWN_TERM = "UNKNOWN_TERM"


class RepairAction(str, Enum):
    """Action for a repair hypothesis — proposal only, not execution."""

    KEEP = "KEEP"
    PROPOSE_REPLACE = "PROPOSE_REPLACE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ShiftSignalType(str, Enum):
    """Signal categories for topic-transition scoring."""

    ENTITY_DENSITY_SPIKE = "entity_density_spike"
    ENTITY_TURNOVER = "entity_turnover"
    ENTITY_OVERLAP_DROP = "entity_overlap_drop"
    VOCABULARY_DRIFT = "vocabulary_drift"
    SILENCE_GAP = "silence_gap"
    SEMANTIC_SIMILARITY_DROP = "semantic_similarity_drop"
    NEW_ENTITY_CLUSTER = "new_entity_cluster"
    PRONOUN_DENSITY_CHANGE = "pronoun_density_change"


class ShiftSignal(BaseModel):
    """A single structured signal from the Chunk Worker."""

    type: ShiftSignalType
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    detail: str = ""


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

E = TypeVar("E", bound="BaseEvent")


class BaseEvent(BaseModel):
    """Every event is anchored to an absolute range in the original SRT."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: ClassVar[str] = "base"
    chunk_index: int = Field(ge=0)
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Pass 1 observation events (discovery layer ONLY)
# ---------------------------------------------------------------------------


class EntityDetectedEvent(BaseEvent):
    """A potential entity / high-frequency term was spotted.

    This is a raw observation — the Reducer decides whether this entity is
    truly active in the video's world model.
    """

    event_type: ClassVar[str] = "entity_detected"
    surface_text: str
    matched_entity: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_line_indices: list[int] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)


class RepairHypothesisEvent(BaseEvent):
    """A tentative repair proposal — NOT a final decision.

    The Reducer may accept, reject, or refine this hypothesis after
    cross-referencing evidence from other chunks.
    """

    event_type: ClassVar[str] = "repair_hypothesis"
    observed_text: str
    candidate_resolution: str
    confidence: float = Field(ge=0.0, le=1.0)
    action: RepairAction = RepairAction.NEEDS_REVIEW
    source_line_indices: list[int] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    detail: str = ""


class TopicTransitionHypothesisEvent(BaseEvent):
    """Chunk Worker evidence: a topic boundary MAY exist near this region.

    The Worker reports structured signals plus an approximate transition
    region.  The Reducer alone computes the definitive TopicSpan and
    assigns topic labels.

    Reducer scoring formula::

        topic_shift_score = sum(
            signal.strength * weight
            for signal in signals
        )
    """

    event_type: ClassVar[str] = "topic_transition_hypothesis"
    signals: list[ShiftSignal] = Field(default_factory=list)
    previous_semantic_cluster: str = ""
    current_semantic_cluster: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    approximate_transition_region: dict[str, int] = Field(default_factory=dict)
    pre_context_entities: dict[str, float] = Field(default_factory=dict)
    post_context_entities: dict[str, float] = Field(default_factory=dict)

    def get_signal_strength(self, signal_type: ShiftSignalType) -> float:
        for s in self.signals:
            if s.type == signal_type:
                return s.strength
        return 0.0


# ---------------------------------------------------------------------------
# Backward-compatible aliases (existing code may reference old names)
# ---------------------------------------------------------------------------

CorrectionProposedEvent = RepairHypothesisEvent
TopicTransitionEvent = TopicTransitionHypothesisEvent

# ---------------------------------------------------------------------------
# Append-only event log
# ---------------------------------------------------------------------------


class EventLogger:
    """Append-only, ordered event log — the source of truth for the Reducer."""

    def __init__(self) -> None:
        self._events: list[BaseEvent] = []
        self._lock = threading.Lock()

    # -- write (append only, thread-safe) -------------------------------------

    def append_event(self, event: BaseEvent) -> None:
        """Add an event.  Thread-safe.  No update or delete is ever permitted."""
        with self._lock:
            self._events.append(event)

    def append_events(self, events: list[BaseEvent]) -> None:
        with self._lock:
            self._events.extend(events)

    # -- read queries ---------------------------------------------------------

    @property
    def all_events(self) -> list[BaseEvent]:
        return list(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def get_events_by_type(self, event_cls: Type[E]) -> list[E]:
        return [e for e in self._events if type(e) is event_cls]  # noqa: E721

    def get_events_in_range(
        self, start_index: int, end_index: int
    ) -> list[BaseEvent]:
        return [
            e
            for e in self._events
            if e.start_index <= end_index and e.end_index >= start_index
        ]

    def get_repair_hypotheses(self) -> list[RepairHypothesisEvent]:
        return self.get_events_by_type(RepairHypothesisEvent)

    def get_topic_transitions(self) -> list[TopicTransitionHypothesisEvent]:
        return self.get_events_by_type(TopicTransitionHypothesisEvent)

    def get_entity_detections(self) -> list[EntityDetectedEvent]:
        return self.get_events_by_type(EntityDetectedEvent)

    # -- summary --------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "total_events": self.event_count,
            "by_type": {
                cls.event_type: len(self.get_events_by_type(cls))
                for cls in (
                    EntityDetectedEvent,
                    RepairHypothesisEvent,
                    TopicTransitionHypothesisEvent,
                )
            },
        }
