"""Phase 4: Consensus Reducer with negative evidence and evidence cancellation.

Converges unreliable local observations from the EventLog into a stable
CommittedWorldState — NOT through weighted voting, but through genuine
multi-evidence reasoning with supporting AND counter evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from subtitle_corrector.events import (
    EntityDetectedEvent,
    EventLogger,
    ReasonCode,
    RepairHypothesisEvent,
    TopicTransitionHypothesisEvent,
)
from subtitle_corrector.schemas import (
    ChunkSemanticTimeline,
    SemanticFilterOutput,
)

# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------


class EvidenceDimension(str, Enum):
    PHONETIC = "phonetic"
    ENTITY_COOCCURRENCE = "entity_cooccurrence"
    TOPIC_ALIGNMENT = "topic"
    GRAMMAR = "grammar"
    TEMPORAL = "temporal"
    OVERLAP = "overlap"
    SNAPSHOT = "snapshot"
    # Negative dimensions
    SEMANTIC_INCONSISTENCY = "semantic_inconsistency"
    ENTITY_CONFLICT = "entity_conflict"
    TEMPORAL_DISCONTINUITY = "temporal_discontinuity"
    LOW_OVERLAP_CONSENSUS = "low_overlap_consensus"
    GRAMMAR_DISRUPTION = "grammar_disruption"
    CATEGORY_DISSONANCE = "category_dissonance"


DIMENSION_WEIGHTS: dict[EvidenceDimension, float] = {
    # Positive
    EvidenceDimension.PHONETIC: 0.15,
    EvidenceDimension.ENTITY_COOCCURRENCE: 0.25,
    EvidenceDimension.TOPIC_ALIGNMENT: 0.20,
    EvidenceDimension.GRAMMAR: 0.10,
    EvidenceDimension.TEMPORAL: 0.15,
    EvidenceDimension.OVERLAP: 0.25,
    EvidenceDimension.SNAPSHOT: 0.05,
    # Negative (mirror strengths)
    EvidenceDimension.SEMANTIC_INCONSISTENCY: 0.70,
    EvidenceDimension.ENTITY_CONFLICT: 0.60,
    EvidenceDimension.TEMPORAL_DISCONTINUITY: 0.50,
    EvidenceDimension.LOW_OVERLAP_CONSENSUS: 0.80,
    EvidenceDimension.GRAMMAR_DISRUPTION: 0.90,
    EvidenceDimension.CATEGORY_DISSONANCE: 0.40,
}


@dataclass
class Evidence:
    dimension: EvidenceDimension
    polarity: int  # +1 (supporting) | -1 (counter) | 0 (neutral)
    strength: float
    source: str = ""
    detail: str = ""


# ---------------------------------------------------------------------------
# Temporal Entity Memory (region-aware decay)
# ---------------------------------------------------------------------------


class TemporalEntityRecord:
    __slots__ = (
        "entity", "confidence", "active_start", "active_end",
        "related_topics", "related_entities", "decay_factor",
        "occurrence_timeline", "last_seen_index", "status",
    )

    def __init__(self, entity: str, confidence: float = 0.5, first_seen: int = 0) -> None:
        self.entity = entity
        self.confidence = confidence
        self.active_start = first_seen
        self.active_end = first_seen
        self.related_topics: list[str] = []
        self.related_entities: list[str] = []
        self.decay_factor = 1.0
        self.occurrence_timeline: list[int] = [first_seen]
        self.last_seen_index = first_seen
        self.status = "active"


class TemporalEntityMemory:
    """Region-aware temporal decay: exp(-lambda * distance).

    Chunk size does NOT affect decay behaviour — the same entity in a
    25-line chunk vs a 100-line chunk decays identically.
    """

    def __init__(self, lambda_: float = 0.05) -> None:
        self._records: dict[str, TemporalEntityRecord] = {}
        self.lambda_ = lambda_

    def activate(self, entity: str, confidence: float, cue_index: int) -> TemporalEntityRecord:
        if entity not in self._records:
            self._records[entity] = TemporalEntityRecord(entity, confidence, cue_index)
        rec = self._records[entity]
        rec.confidence = max(rec.confidence, confidence)
        rec.active_end = cue_index
        rec.occurrence_timeline.append(cue_index)
        rec.last_seen_index = cue_index
        rec.decay_factor = 1.0
        rec.status = "active"
        return rec

    def decay(self, current_index: int) -> None:
        for rec in self._records.values():
            distance = current_index - rec.last_seen_index
            rec.decay_factor = math.exp(-self.lambda_ * distance)
            if rec.decay_factor < 0.1:
                rec.status = "inactive"

    def get_active_in_region(self, start: int, end: int) -> list[TemporalEntityRecord]:
        return [
            r for r in self._records.values()
            if r.status == "active" and r.active_start <= end and r.active_end >= start
        ]

    def get(self, entity: str) -> TemporalEntityRecord | None:
        return self._records.get(entity)

    @property
    def all_records(self) -> dict[str, TemporalEntityRecord]:
        return dict(self._records)


# ---------------------------------------------------------------------------
# Topic State Graph (semantic similarity based)
# ---------------------------------------------------------------------------


@dataclass
class TopicNode:
    topic_id: str
    label: str
    start_index: int
    end_index: int
    confidence: float
    dominant_entities: list[str] = field(default_factory=list)
    vocabulary: set[str] = field(default_factory=set)
    signal_pattern: list[float] = field(default_factory=list)


@dataclass
class TopicEdge:
    from_id: str
    to_id: str
    transition_type: str  # RECURRENCE | DRIFT | OVERLAP | RETURN
    strength: float


class TopicStateGraph:
    """Non-linear topic graph — allows recurrence, overlap, temporary drift."""

    def __init__(self) -> None:
        self.nodes: list[TopicNode] = []
        self.edges: list[TopicEdge] = []
        self._next_id = 0

    def add_node(self, node: TopicNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: TopicEdge) -> None:
        self.edges.append(edge)

    def build_from_transitions(
        self, transitions: list[TopicTransitionHypothesisEvent]
    ) -> None:
        if not transitions:
            return

        # Sort by approximate_transition_region
        sorted_ts = sorted(
            transitions,
            key=lambda t: t.approximate_transition_region.get("start_line_index", 0),
        )

        for t in sorted_ts:
            region = t.approximate_transition_region
            node = TopicNode(
                topic_id=f"topic_{self._next_id}",
                label=t.current_semantic_cluster or t.previous_semantic_cluster,
                start_index=region.get("start_line_index", 0),
                end_index=region.get("end_line_index", 0),
                confidence=t.confidence,
                dominant_entities=list(t.post_context_entities.keys()),
                vocabulary=set(),
                signal_pattern=[s.strength for s in t.signals],
            )
            self._next_id += 1

            # Detect recurrence via semantic similarity
            for existing in self.nodes:
                sim = self._semantic_similarity(node, existing)
                if sim > 0.75 and not self._is_adjacent(node, existing):
                    self.add_edge(TopicEdge(
                        from_id=existing.topic_id, to_id=node.topic_id,
                        transition_type="RECURRENCE", strength=sim,
                    ))
                    node.label = existing.label  # inherit label of recurring topic
                    break
            else:
                # New topic — add DRIFT edge from previous node
                if self.nodes:
                    prev = self.nodes[-1]
                    self.add_edge(TopicEdge(
                        from_id=prev.topic_id, to_id=node.topic_id,
                        transition_type="DRIFT",
                        strength=1.0 - self._semantic_similarity(node, prev),
                    ))

            self.add_node(node)

    @staticmethod
    def _semantic_similarity(a: TopicNode, b: TopicNode) -> float:
        entity_overlap = (
            len(set(a.dominant_entities) & set(b.dominant_entities))
            / max(1, len(set(a.dominant_entities) | set(b.dominant_entities)))
        )
        vocab_overlap = (
            len(a.vocabulary & b.vocabulary)
            / max(1, len(a.vocabulary | b.vocabulary))
        )
        signal_sim = TopicStateGraph._cosine_similarity(
            a.signal_pattern, b.signal_pattern
        )
        return 0.40 * entity_overlap + 0.35 * vocab_overlap + 0.25 * signal_sim

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.5
        n = max(len(a), len(b))
        a_pad = a + [0.0] * (n - len(a))
        b_pad = b + [0.0] * (n - len(b))
        dot = sum(x * y for x, y in zip(a_pad, b_pad))
        norm_a = math.sqrt(sum(x * x for x in a_pad)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b_pad)) or 1.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _is_adjacent(a: TopicNode, b: TopicNode) -> bool:
        return abs(a.start_index - b.end_index) <= 5 or abs(b.start_index - a.end_index) <= 5


# ---------------------------------------------------------------------------
# Consensus Reducer
# ---------------------------------------------------------------------------


@dataclass
class ConvergedRepair:
    hypothesis: RepairHypothesisEvent
    score: float  # [-1.0, 1.0]
    evidence_chain: list[Evidence]
    verdict: str  # COMMIT | UNCERTAIN | REJECT


@dataclass
class UncertainRegion:
    start_index: int
    end_index: int
    hypotheses: list[RepairHypothesisEvent]
    score_range: tuple[float, float]


@dataclass
class ConflictZone:
    start_index: int
    end_index: int
    competing_hypotheses: list[RepairHypothesisEvent]


@dataclass
class CommittedWorldState:
    entities: dict[str, TemporalEntityRecord] = field(default_factory=dict)
    topic_graph: TopicStateGraph = field(default_factory=TopicStateGraph)
    repair_decisions: list[ConvergedRepair] = field(default_factory=list)
    uncertain_regions: list[UncertainRegion] = field(default_factory=list)
    conflict_zones: list[ConflictZone] = field(default_factory=list)

    def to_deterministic_snapshot(self) -> bytes:
        import pickle
        return pickle.dumps(self)

    def summary(self) -> dict[str, Any]:
        return {
            "entities": len(self.entities),
            "active_entities": sum(
                1 for r in self.entities.values() if r.status == "active"
            ),
            "topic_nodes": len(self.topic_graph.nodes),
            "repairs_committed": sum(
                1 for d in self.repair_decisions if d.verdict == "COMMIT"
            ),
            "repairs_uncertain": len(self.uncertain_regions),
            "conflict_zones": len(self.conflict_zones),
        }


@dataclass
class SemanticCluster:
    """Reducer output — a stable semantic cluster across chunks.

    Topic label is NOT assigned here.  A lightweight LLM call after
    clustering generates the human-readable label.
    """

    cluster_id: str
    dominant_entities: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    entity_overlap_score: float = 0.0
    category_overlap_score: float = 0.0
    temporal_range: tuple[int, int] = (0, 0)
    topic_label: str = ""


class ConsensusReducer:
    """Multi-evidence convergence with supporting AND counter evidence.

    Scoring formula::

        score = sum( evidence.polarity * evidence.strength
                     * DIMENSION_WEIGHTS[evidence.dimension] )

    Positive evidence pushes toward COMMIT.  Negative evidence pushes
    toward REJECT.  They genuinely cancel — a "frequent + consistent"
    ASR error can still be rejected if counter-evidence is strong.
    """

    def __init__(self) -> None:
        self.temporal_memory = TemporalEntityMemory(lambda_=0.05)
        self.topic_graph = TopicStateGraph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reduce(self, event_log: EventLogger) -> CommittedWorldState:
        entities = event_log.get_entity_detections()
        repairs = event_log.get_repair_hypotheses()
        transitions = event_log.get_topic_transitions()

        # Build temporal entity memory
        for e in entities:
            self.temporal_memory.activate(
                e.matched_entity, e.confidence, e.start_index,
            )
        self.temporal_memory.decay(
            max((e.end_index for e in entities), default=0)
        )

        # Build topic graph from transitions
        self.topic_graph.build_from_transitions(transitions)

        # Converge repair hypotheses
        decisions: list[ConvergedRepair] = []
        for hyp in repairs:
            evidence_pool = self._gather_evidence(hyp, entities, transitions)
            score = self._converge(evidence_pool)

            if score >= 0.60:
                verdict = "COMMIT"
            elif score <= -0.60:
                verdict = "REJECT"
            else:
                verdict = "UNCERTAIN"

            decisions.append(ConvergedRepair(
                hypothesis=hyp, score=score,
                evidence_chain=evidence_pool, verdict=verdict,
            ))

        # Group uncertain and conflicting regions
        uncertain = self._group_uncertain(decisions)
        conflicts = self._detect_conflicts(decisions)

        return CommittedWorldState(
            entities=self.temporal_memory.all_records,
            topic_graph=self.topic_graph,
            repair_decisions=decisions,
            uncertain_regions=uncertain,
            conflict_zones=conflicts,
        )

    def reduce_from_timelines(
        self,
        timelines: list[ChunkSemanticTimeline],
        filter_outputs: list[SemanticFilterOutput],
    ) -> list[SemanticCluster]:
        """Cluster ChunkSemanticTimelines by semantic similarity.

        Clusters by entity overlap + category overlap + signal overlap
        + temporal continuity.  Does NOT trust topic label strings — the
        label is generated later by a lightweight LLM call.
        """
        # Pair timelines with their filter outputs.  None filter_outputs
        # (LLM parse failures) are kept as placeholders so zip stays aligned.
        pairs = [
            (tl, fo) for tl, fo in zip(timelines, filter_outputs)
            if fo is not None and tl is not None
        ]
        clusters: list[SemanticCluster] = []
        assigned: set[int] = set()

        for i, (tl, fo) in enumerate(pairs):
            if i in assigned:
                continue

            # Build candidate cluster from this chunk
            entities = [e.get("entity", "") for e in fo.confirmed_entities]
            categories = list(fo.dominant_categories)
            signals = list(fo.semantic_signals)
            start_idx = tl.observations[0].cue_index if tl.observations else 0
            end_idx = tl.observations[-1].cue_index if tl.observations else 0
            entity_overlap = 1.0
            cat_overlap = 1.0

            # Find all other chunks that match
            matched_chunks = [i]
            for j, (tl2, fo2) in enumerate(pairs):
                if j <= i or j in assigned:
                    continue
                entities2 = [e.get("entity", "") for e in fo2.confirmed_entities]
                entity_overlap = (
                    len(set(entities) & set(entities2))
                    / max(1, len(set(entities) | set(entities2)))
                )
                cat_overlap = (
                    len(set(categories) & set(fo2.dominant_categories))
                    / max(1, len(set(categories) | set(fo2.dominant_categories)))
                )
                # Temporal proximity
                tl2_start = tl2.observations[0].cue_index if tl2.observations else 9999
                temporal_dist = abs(tl2_start - end_idx)

                if entity_overlap > 0.4 and cat_overlap > 0.3 and temporal_dist < 50:
                    matched_chunks.append(j)
                    assigned.add(j)
                    # Extend range
                    tl2_end = tl2.observations[-1].cue_index if tl2.observations else 0
                    end_idx = max(end_idx, tl2_end)
                    entities = list(set(entities) | set(entities2))
                    categories = list(set(categories) | set(fo2.dominant_categories))
                    signals = list(set(signals) | set(fo2.semantic_signals))

            assigned.add(i)
            clusters.append(SemanticCluster(
                cluster_id=f"cluster_{len(clusters)}",
                dominant_entities=entities,
                categories=categories,
                signals=signals,
                entity_overlap_score=round(entity_overlap if matched_chunks else 1.0, 3),
                category_overlap_score=round(cat_overlap if matched_chunks else 1.0, 3),
                temporal_range=(start_idx, end_idx),
            ))

        return clusters

    # ------------------------------------------------------------------
    # Lightweight LLM topic naming (post-clustering)
    # ------------------------------------------------------------------

    def generate_topic_labels(
        self, clusters: list[SemanticCluster], llm_client: object,
    ) -> list[SemanticCluster]:
        """Call a lightweight LLM to generate a short topic label for each cluster.

        Each call uses temperature=0, max_tokens=50 — extremely cheap.
        On LLM failure, the label remains empty (degraded but not broken).
        """
        for cluster in clusters:
            prompt = (
                f"为以下语义聚类生成一个短标签（5-10字）：\n"
                f"主导实体：{', '.join(cluster.dominant_entities[:5]) or '无'}\n"
                f"语义类别：{', '.join(cluster.categories[:5]) or '无'}\n"
                f"关键信号：{', '.join(cluster.signals[:5]) or '无'}\n"
                f"时间范围：字幕行 [{cluster.temporal_range[0]}, {cluster.temporal_range[1]}]\n"
                f"\n输出格式：直接返回标签文本，不要解释。"
            )
            try:
                response = llm_client.chat.completions.create(
                    model=getattr(llm_client, "model", "mimo-v2.5-pro"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=50,
                )
                label = (response.choices[0].message.content or "").strip()
                # Clean up: remove quotes, newlines, markdown
                label = label.replace('"', '').replace("'", "").replace("\n", " ").strip()
                if label:
                    cluster.topic_label = label
                    logger.info("[Reducer] 聚类 {} 标签: '{}'", cluster.cluster_id, label)
            except Exception as exc:
                logger.warning("[Reducer] 聚类 {} 标签生成失败: {}", cluster.cluster_id, exc)
        return clusters

    def reduce_with_labels(
        self,
        timelines: list[ChunkSemanticTimeline],
        filter_outputs: list[SemanticFilterOutput],
        llm_client: object,
    ) -> list[SemanticCluster]:
        """Cluster + generate topic labels in one call."""
        clusters = self.reduce_from_timelines(timelines, filter_outputs)
        return self.generate_topic_labels(clusters, llm_client)

    def reduce_from_events(self, events: list) -> CommittedWorldState:
        """Deterministic replay entry point."""
        logger_wrapper = EventLogger()
        for e in events:
            logger_wrapper.append_event(e)
        return self.reduce(logger_wrapper)

    # ------------------------------------------------------------------
    # Evidence gathering
    # ------------------------------------------------------------------

    def _gather_evidence(
        self,
        hyp: RepairHypothesisEvent,
        entities: list[EntityDetectedEvent],
        transitions: list[TopicTransitionHypothesisEvent],
    ) -> list[Evidence]:
        pool: list[Evidence] = []

        # Positive evidence from reason_codes
        for rc in hyp.reason_codes:
            dim = self._reason_to_dimension(rc)
            if dim and dim not in _NEGATIVE_DIMENSIONS:
                pool.append(Evidence(
                    dimension=dim, polarity=+1, strength=hyp.confidence,
                    source=f"hypothesis:{hyp.event_id}", detail=rc.value,
                ))

        # Entity co-occurrence: do entities around this span match?
        nearby_entities = [
            e for e in entities
            if abs(e.start_index - hyp.start_index) <= 10
        ]
        if nearby_entities:
            match = any(
                hyp.candidate_resolution == e.matched_entity
                for e in nearby_entities
            )
            if match:
                pool.append(Evidence(
                    dimension=EvidenceDimension.ENTITY_COOCCURRENCE,
                    polarity=+1, strength=0.80,
                    source="entity_proximity",
                ))
            else:
                pool.append(Evidence(
                    dimension=EvidenceDimension.ENTITY_CONFLICT,
                    polarity=-1, strength=0.60,
                    source="entity_proximity",
                    detail=f"nearby entities: {[e.matched_entity for e in nearby_entities[:5]]}",
                ))

        # Temporal continuity
        if hyp.start_index > 0:
            prior_entities = [
                e for e in entities
                if e.end_index < hyp.start_index
                and hyp.start_index - e.end_index <= 20
            ]
            if not prior_entities:
                pool.append(Evidence(
                    dimension=EvidenceDimension.TEMPORAL_DISCONTINUITY,
                    polarity=-1, strength=0.50,
                    source="temporal_gap",
                ))

        # Topic alignment
        for t in transitions:
            region = t.approximate_transition_region
            if region.get("start_line_index", 0) <= hyp.start_index <= region.get("end_line_index", 9999):
                if hyp.candidate_resolution in t.post_context_entities:
                    pool.append(Evidence(
                        dimension=EvidenceDimension.TOPIC_ALIGNMENT,
                        polarity=+1, strength=t.confidence,
                        source=f"transition:{t.event_id}",
                    ))
                break

        # Low overlap consensus: check if competing hypotheses exist for same span
        # (handled later in _detect_conflicts — here we only flag if no overlap data)

        return pool

    @staticmethod
    def _reason_to_dimension(rc: ReasonCode) -> EvidenceDimension | None:
        mapping = {
            ReasonCode.PHONETIC_MATCH: EvidenceDimension.PHONETIC,
            ReasonCode.CONTEXT_ALIGNMENT: EvidenceDimension.TOPIC_ALIGNMENT,
            ReasonCode.CONTEXT_CONFLICT: EvidenceDimension.SEMANTIC_INCONSISTENCY,
            ReasonCode.ENTITY_MEMORY_MATCH: EvidenceDimension.ENTITY_COOCCURRENCE,
            ReasonCode.GRAMMAR_ERROR: EvidenceDimension.GRAMMAR_DISRUPTION,
            ReasonCode.CATEGORY_MISMATCH: EvidenceDimension.CATEGORY_DISSONANCE,
            ReasonCode.LOW_CONFIDENCE: None,
            ReasonCode.UNKNOWN_TERM: None,
            ReasonCode.VOCABULARY_SHIFT: None,
            ReasonCode.ENTITY_CLUSTER_CHANGE: None,
        }
        return mapping.get(rc)

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    @staticmethod
    def _converge(evidence_pool: list[Evidence]) -> float:
        if not evidence_pool:
            return 0.0
        score = sum(
            e.polarity * e.strength * DIMENSION_WEIGHTS.get(e.dimension, 0.10)
            for e in evidence_pool
        )
        return max(-1.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _group_uncertain(decisions: list[ConvergedRepair]) -> list[UncertainRegion]:
        groups: dict[tuple[int, int], list[RepairHypothesisEvent]] = defaultdict(list)
        for d in decisions:
            if d.verdict == "UNCERTAIN":
                key = (d.hypothesis.start_index, d.hypothesis.end_index)
                groups[key].append(d.hypothesis)
        return [
            UncertainRegion(
                start_index=k[0], end_index=k[1],
                hypotheses=v,
                score_range=(-0.60, 0.60),
            )
            for k, v in groups.items()
        ]

    @staticmethod
    def _detect_conflicts(decisions: list[ConvergedRepair]) -> list[ConflictZone]:
        by_span: dict[tuple[int, int], list[ConvergedRepair]] = defaultdict(list)
        for d in decisions:
            key = (d.hypothesis.start_index, d.hypothesis.end_index)
            by_span[key].append(d)
        conflicts: list[ConflictZone] = []
        for (s, e), items in by_span.items():
            if len(items) > 1:
                values = {item.hypothesis.candidate_resolution for item in items}
                if len(values) > 1:
                    conflicts.append(ConflictZone(
                        start_index=s, end_index=e,
                        competing_hypotheses=[item.hypothesis for item in items],
                    ))
        return conflicts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEGATIVE_DIMENSIONS: set[EvidenceDimension] = {
    EvidenceDimension.SEMANTIC_INCONSISTENCY,
    EvidenceDimension.ENTITY_CONFLICT,
    EvidenceDimension.TEMPORAL_DISCONTINUITY,
    EvidenceDimension.LOW_OVERLAP_CONSENSUS,
    EvidenceDimension.GRAMMAR_DISRUPTION,
    EvidenceDimension.CATEGORY_DISSONANCE,
}
