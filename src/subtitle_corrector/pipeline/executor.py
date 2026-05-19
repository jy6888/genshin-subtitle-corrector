from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from subtitle_corrector.config.settings import EntityMemorySettings
from subtitle_corrector.detector.base import Detector
from subtitle_corrector.memory.entity import EntityMemoryManager
from subtitle_corrector.normalize.text import TextNormalizer
from subtitle_corrector.resolver.base import LLMArbitrator
from subtitle_corrector.schemas import (
    ArbitrationRequest,
    CorrectionAction,
    CorrectionItem,
    EntityMemorySnapshot,
    RepairResult,
    SubtitleDocument,
)
from subtitle_corrector.scoring.fusion import WeightedRiskAggregator

if TYPE_CHECKING:
    from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher
    from subtitle_corrector.memory.category_activation import CategoryActivationManager


@dataclass
class PipelineResult:
    repaired_document: SubtitleDocument
    repairs: list[RepairResult]
    entity_memory_log: list[EntityMemorySnapshot]


class SubtitleCorrectionPipeline:
    def __init__(
        self,
        normalizer: TextNormalizer,
        detectors: list[Detector],
        aggregator: WeightedRiskAggregator,
        arbitrator: LLMArbitrator,
        matcher: FuzzyTerminologyMatcher | None = None,
        entity_memory_settings: EntityMemorySettings | None = None,
        category_activation: CategoryActivationManager | None = None,
        alias_lexicon: object | None = None,
        character_lexicon: object | None = None,
        llm_threshold: float = 0.72,
        context_window: int = 2,
        dry_run: bool = True,
    ) -> None:
        self.normalizer = normalizer
        self.detectors = detectors
        self.aggregator = aggregator
        self.arbitrator = arbitrator
        self.matcher = matcher
        self.entity_memory_settings = entity_memory_settings
        self.category_activation = category_activation
        self.alias_lexicon = alias_lexicon
        self.character_lexicon = character_lexicon
        self.llm_threshold = llm_threshold
        self.context_window = context_window
        self.dry_run = dry_run

    def run(self, document: SubtitleDocument) -> PipelineResult:
        repaired = document.model_copy(deep=True)
        repairs: list[RepairResult] = []
        entity_memory_log: list[EntityMemorySnapshot] = []

        # Initialize entity memory for this document run
        if self.entity_memory_settings is not None:
            entity_mem = EntityMemoryManager(
                decay_rate=self.entity_memory_settings.decay_rate,
                min_weight=self.entity_memory_settings.min_weight,
            )
        else:
            entity_mem = None

        for cue in repaired.cues:
            # Snapshot: entity memory state BEFORE this cue
            entities_before = (
                entity_mem.active_entities if entity_mem is not None else {}
            )

            normalized = self.normalizer.normalize(cue.text)
            cue.text = normalized.normalized

            # Set entity memory on matcher so lookup() applies boost
            if self.matcher is not None and entity_mem is not None:
                self.matcher.entity_memory = entity_mem

            detections = [detector.detect(cue, repaired) for detector in self.detectors]

            # Decay existing entities and cold category activations.
            activated_set: set[str] = set()
            if entity_mem is not None:
                entity_mem.decay()
            if self.category_activation is not None:
                self.category_activation.decay()

            # Activate entities from known spoken aliases detected in cue text.
            # Known aliases (芙芙/龙王/那维) only activate entity context,
            # they are NOT directly replaced in text.
            if self.alias_lexicon is not None and entity_mem is not None:
                alias_matches = self.alias_lexicon.find_surfaces_in_text(cue.text)
                for match in alias_matches:
                    canonical = match["canonical"]
                    if self.matcher is not None:
                        choices = self.matcher._build_choices()
                        if canonical not in choices:
                            continue
                    entity_mem.update_entity(canonical)
                    activated_set.add(canonical)

            # 角色外号激活父节点（芙芙→芙宁娜，不替换原文）
            if self.character_lexicon is not None and entity_mem is not None:
                # 多字外号
                nicknames = self.character_lexicon.find_nicknames_in_text(cue.text)
                for entry in nicknames:
                    if self.matcher is not None:
                        choices = self.matcher._build_choices()
                        if entry.canonical_term not in choices:
                            continue
                    entity_mem.update_entity(entry.canonical_term)
                    activated_set.add(entry.canonical_term)

                # 配队单字简称（仅在有配队语境时解析）
                from subtitle_corrector.character_alias.team_comp import TeamCompParser
                parser = TeamCompParser(self.character_lexicon)
                team_activated = parser.parse(cue.text)
                for _alias, canonical in team_activated:
                    if self.matcher is not None:
                        choices = self.matcher._build_choices()
                        if canonical not in choices:
                            continue
                    entity_mem.update_entity(canonical)
                    activated_set.add(canonical)

            # Snapshot: entity memory state AFTER decay, before LLM
            entities_after = (
                entity_mem.active_entities if entity_mem is not None else {}
            )

            risk = self.aggregator.aggregate(cue.index, detections)

            # 过滤 context_alias 候选：只激活实体，不作为修复候选交给 LLM
            repair_candidates = [
                c for c in risk.candidates
                if c.metadata.get("intent") != "context_alias"
            ]
            risk.candidates = repair_candidates

            if risk.risk_score < self.llm_threshold or not repair_candidates:
                # Record snapshot even when skipping LLM
                entity_memory_log.append(
                    EntityMemorySnapshot(
                        cue_index=cue.index,
                        entities_before=entities_before,
                        entities_activated=sorted(activated_set),
                        entities_after=entities_after,
                        decays_applied=entity_mem is not None,
                    )
                )
                continue
            request = ArbitrationRequest(
                cue=cue,
                context_before=self._context_before(repaired, cue.index),
                context_after=self._context_after(repaired, cue.index),
                candidates=risk.candidates,
                risk=risk,
            )
            decision = self.arbitrator.arbitrate(request)
            repaired_text = cue.text

            # Multi-point replacement with coordinate reverse-mapping.
            # LLM only provides original_word/corrected_word — the executor
            # resolves exact character positions from detector spans (priority)
            # or regex fallback.
            if (
                decision.action == CorrectionAction.REPLACE
                and decision.corrections
                and not self.dry_run
            ):
                self._resolve_correction_coordinates(
                    decision.corrections, detections, cue.text
                )
                sorted_corrections = sorted(
                    decision.corrections,
                    key=lambda c: c.start_char,
                    reverse=True,
                )
                text_chars = list(cue.text)
                for corr in sorted_corrections:
                    if 0 <= corr.start_char < corr.end_char <= len(text_chars):
                        text_chars[corr.start_char : corr.end_char] = list(
                            corr.corrected_word
                        )
                repaired_text = "".join(text_chars)
                cue.text = repaired_text

            # Legacy: single-candidate fallback
            elif (
                decision.action == CorrectionAction.REPLACE
                and decision.selected_candidate is not None
                and not self.dry_run
            ):
                candidate = decision.selected_candidate
                if candidate.source == "llm":
                    repaired_text = candidate.value
                else:
                    repaired_text = self._local_replace(cue.text, candidate.value)
                cue.text = repaired_text

            # Activate entities from LLM-confirmed list + cascade to parents.
            if entity_mem is not None:
                entities_to_activate: list[str] = list(
                    decision.activated_parent_entities
                )

                # Also activate from each correction's corrected_word
                for corr in decision.corrections:
                    if (
                        corr.corrected_word
                        and corr.corrected_word not in entities_to_activate
                    ):
                        entities_to_activate.append(corr.corrected_word)

                # Legacy fallback
                if (
                    not decision.activated_parent_entities
                    and not decision.corrections
                    and decision.selected_candidate is not None
                ):
                    parent = decision.selected_candidate.metadata.get("parent_entity")
                    if parent and parent not in entities_to_activate:
                        entities_to_activate.append(parent)

                for entity_name in entities_to_activate:
                    # ── Terminology whitelist gate ──
                    # Only activate entities that are known terms.
                    # ASR error strings (e.g. "剃草之刀光") won't be
                    # in the terminology and will be silently skipped.
                    if self.matcher is not None:
                        choices = self.matcher._build_choices()
                        if entity_name not in choices:
                            continue
                    # ── End gate ──

                    if entity_name not in activated_set:
                        entity_mem.update_entity(entity_name)
                        activated_set.add(entity_name)

                    # Cascade: look up parent entity
                    if self.matcher is not None:
                        parent = self.matcher.get_parent_for_term(entity_name)
                        if parent and parent not in activated_set:
                            entity_mem.update_entity(parent)
                            activated_set.add(parent)

            # Activate cold-pool categories based on LLM-selected corrections.
            # When the LLM picks a term from a dormant cold category, that
            # category wakes up for subsequent cues (with decay).
            if self.category_activation is not None and self.matcher is not None:
                choices = self.matcher._build_choices()
                for corr in decision.corrections:
                    entry = choices.get(corr.corrected_word)
                    if entry is None:
                        continue
                    cat = getattr(entry, "category", "") or ""
                    if cat:
                        self.category_activation.activate(cat)

            repairs.append(
                RepairResult(
                    cue_index=cue.index,
                    original_text=normalized.original,
                    repaired_text=repaired_text,
                    action=decision.action,
                    confidence=decision.confidence,
                    explanation=decision.reasoning,
                )
            )

            # Record snapshot for this cue (post-activation state)
            entities_after = (
                entity_mem.active_entities if entity_mem is not None else {}
            )
            entity_memory_log.append(
                EntityMemorySnapshot(
                    cue_index=cue.index,
                    entities_before=entities_before,
                    entities_activated=sorted(activated_set),
                    entities_after=entities_after,
                    decays_applied=entity_mem is not None,
                )
            )

        # Clear entity memory reference from matcher after run
        if self.matcher is not None:
            self.matcher.entity_memory = None

        return PipelineResult(
            repaired_document=repaired,
            repairs=repairs,
            entity_memory_log=entity_memory_log,
        )

    @staticmethod
    def _resolve_correction_coordinates(
        corrections: list[CorrectionItem],
        detections: list,
        cue_text: str,
    ) -> None:
        """Backfill start_char/end_char for each CorrectionItem.

        Priority:
        1. Match original_word against detector Span surface_text (exact coordinates).
        2. Fallback: regex scan of cue_text for the literal original_word substring.
        """
        # Build span lookup: surface_text → span dict
        span_by_surface: dict[str, dict] = {}
        for detection in detections:
            for span_dict in detection.metadata.get("spans", []):
                surface = span_dict.get("surface_text", "")
                if surface and surface not in span_by_surface:
                    span_by_surface[surface] = span_dict

        for corr in corrections:
            # --- Priority 1: span lookup ---
            span = span_by_surface.get(corr.original_word)
            if span:
                corr.start_char = span["start_char"]
                corr.end_char = span["end_char"]
                continue

            # --- Priority 2: regex fallback ---
            try:
                pattern = re.escape(corr.original_word)
                matches = list(re.finditer(pattern, cue_text))
                if matches:
                    # If multiple matches, take the first one not yet consumed
                    # (simple heuristic: first occurrence)
                    m = matches[0]
                    corr.start_char = m.start()
                    corr.end_char = m.end()
                else:
                    logger.warning(
                        "Cannot locate '{}' in cue text — skipping correction",
                        corr.original_word,
                    )
                    corr.start_char = 0
                    corr.end_char = 0
            except Exception:
                logger.warning(
                    "Regex failed for '{}' — skipping", corr.original_word
                )
                corr.start_char = 0
                corr.end_char = 0

    def _context_before(self, document: SubtitleDocument, index: int):
        start = max(0, index - self.context_window)
        return document.cues[start:index]

    def _context_after(self, document: SubtitleDocument, index: int):
        return document.cues[index + 1 : index + 1 + self.context_window]

    @staticmethod
    def _local_replace(text: str, candidate: str) -> str:
        # Real replacement strategy must be span-based once detectors emit spans.
        return candidate if len(candidate) <= max(len(text) * 2, len(text) + 8) else text
