"""Pass 1 Discovery Engine — builds ChunkSemanticTimeline from detector data,
sends it to LLM for semantic filtering, and writes confirmed_entities to
EntityMemory.

Architecture: DiscoveryEngine is the bridge between the detection layer
(纯代码) and Phase 1 LLM (Semantic Filter).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from loguru import logger

from subtitle_corrector.events import EventLogger
from subtitle_corrector.resolver.pass1_prompt import (
    PASS1_SYSTEM_PROMPT,
    PASS1_USER_TEMPLATE,
)
from subtitle_corrector.schemas import (
    ChunkSemanticTimeline,
    CompressedContextWindow,
    CueObservation,
    SemanticFilterOutput,
    SubtitleDocument,
)
from subtitle_corrector.utils.chunker import Chunk, SubtitleChunker
from subtitle_corrector.utils.parser import RobustJSONLParser

if TYPE_CHECKING:
    from subtitle_corrector.schemas import SubtitleCue


@dataclass
class DiscoveryResult:
    event_log: EventLogger
    total_chunks: int
    total_events: int
    timelines: list[ChunkSemanticTimeline]  # persisted for Reducer
    filter_outputs: list[SemanticFilterOutput]


class DiscoveryEngine:
    def __init__(
        self,
        llm_client: object,
        matcher: object | None = None,
        entity_memory: object | None = None,
        category_activation: object | None = None,
        hot_categories: set[str] | None = None,
        cold_categories: set[str] | None = None,
        chunk_size: int = 100,
        overlap_size: int = 15,
        max_workers: int = 4,
        max_candidates_per_cue: int = 2,
        model: str | None = None,
        alias_lexicon: object | None = None,
        character_lexicon: object | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.matcher = matcher
        self.entity_memory = entity_memory
        self.category_activation = category_activation
        self.hot_categories = hot_categories or set()
        self.cold_categories = cold_categories or set()
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_workers = max_workers
        self.max_candidates_per_cue = max_candidates_per_cue
        self.model = model
        self.chunker = SubtitleChunker(chunk_size=chunk_size, overlap_size=overlap_size)
        self.jsonl_parser = RobustJSONLParser()

        from subtitle_corrector.detector.jieba_span import JiebaSpanDetector
        self._detector = JiebaSpanDetector(
            matcher,
            hot_categories=self.hot_categories,
            cold_categories=self.cold_categories,
            category_activation=self.category_activation,
            alias_lexicon=alias_lexicon,
            character_lexicon=character_lexicon,
        ) if matcher is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, cues: list[SubtitleCue]) -> DiscoveryResult:
        entity_snapshot = self._create_entity_snapshot()
        category_snapshot = self._create_category_snapshot()

        chunk_result = self.chunker.chunk(cues)
        logger.info(
            "Discovery: {} cues → {} chunks (size={}, overlap={})",
            chunk_result.total_items, len(chunk_result.chunks),
            self.chunk_size, self.overlap_size,
        )

        event_log = EventLogger()
        timelines: list[ChunkSemanticTimeline] = []
        filter_outputs: list[SemanticFilterOutput] = []

        # Pre-warm: inject all terms and build caches in main thread before
        # workers touch jieba's global TRIE or the matcher's lazy caches.
        if self._detector is not None:
            self._detector._ensure_terms_injected()
            self.matcher._build_choices()
            self.matcher._build_pinyin_index()

        results_by_chunk: dict[int, tuple] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_chunk_wrapper,
                    chunk, event_log, entity_snapshot, category_snapshot,
                ): chunk.chunk_id
                for chunk in chunk_result.chunks
            }
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    timeline, sf_output = future.result()
                    results_by_chunk[chunk_id] = (timeline, sf_output)
                except Exception as exc:
                    logger.warning("Chunk {} failed: {}", chunk_id, exc)
                    results_by_chunk[chunk_id] = (None, None)

        # Collect in chunk order (not completion order) to keep timelines
        # and filter_outputs index-aligned for downstream zip()
        for chunk_id in sorted(results_by_chunk):
            timeline, sf_output = results_by_chunk[chunk_id]
            if timeline is not None:
                timelines.append(timeline)
            filter_outputs.append(sf_output)  # always append, may be None
            if sf_output is not None:
                self._update_entity_memory(sf_output)

        logger.info("Discovery complete: {} events, {} timelines",
                     event_log.event_count, len(timelines))
        return DiscoveryResult(
            event_log=event_log,
            total_chunks=len(chunk_result.chunks),
            total_events=event_log.event_count,
            timelines=timelines,
            filter_outputs=filter_outputs,
        )

    # ------------------------------------------------------------------
    # Chunk processing — builds timeline, sends to LLM
    # ------------------------------------------------------------------

    def _process_chunk_wrapper(
        self,
        chunk: Chunk,
        event_log: EventLogger,
        entity_snapshot: MappingProxyType,
        category_snapshot: MappingProxyType,
    ) -> tuple[ChunkSemanticTimeline | None, SemanticFilterOutput | None]:
        try:
            return self._process_chunk(chunk, event_log, entity_snapshot, category_snapshot)
        except Exception as exc:
            logger.warning("Chunk {} error: {}", chunk.chunk_id, exc)
            return None, None

    def _process_chunk(
        self,
        chunk: Chunk,
        event_log: EventLogger,
        entity_snapshot: MappingProxyType,
        category_snapshot: MappingProxyType,
    ) -> tuple[ChunkSemanticTimeline | None, SemanticFilterOutput | None]:
        observations: list[CueObservation] = []
        problem_cues: list[dict] = []

        active_entities = dict(entity_snapshot) if entity_snapshot else {}
        active_categories = list(category_snapshot.keys()) if category_snapshot else []

        # Step 1: Scan all cues, build full observations + identify problem cues
        for cue in chunk.target_lines:
            candidates = self._detect_candidates(cue)
            obs = CueObservation(
                cue_index=cue.index, text=cue.text,
                context_before="", context_after="",
                candidates=candidates,
                active_entities=active_entities,
                active_categories=active_categories,
            )
            observations.append(obs)
            if candidates:
                problem_cues.append({"cue_index": cue.index, "text": cue.text, "candidates": candidates})

        if not observations:
            return None, None

        # Step 2: Dynamic context window based on problem density
        total = len(chunk.target_lines)
        density = len(problem_cues) / total if total else 0
        if density > 0.3:
            ctx_window = 1
        elif density > 0.1:
            ctx_window = 2
        else:
            ctx_window = 3

        # Step 3: Build merged regions (deduplicate overlapping context)
        regions = self._build_regions(chunk, problem_cues, ctx_window)

        # Step 4: Build ChunkSemanticTimeline (full data, for Reducer)
        timeline = ChunkSemanticTimeline(
            chunk_index=chunk.chunk_id,
            observations=observations,
            entity_persistence=self._compute_entity_persistence(observations),
            category_flow=self._compute_category_flow(observations),
            compressed_context_windows=self._build_compressed_windows(observations),
        )

        # Step 5: Send regions (sparse) to LLM
        user_prompt = self._build_semantic_prompt(timeline, regions, density, ctx_window)

        try:
            raw_response = self._call_llm(PASS1_SYSTEM_PROMPT, user_prompt)
            if not raw_response:
                logger.warning("Chunk {} LLM returned empty", chunk.chunk_id)
                return timeline, None

            sf_output = self._parse_semantic_filter_output(raw_response, chunk)
            return timeline, sf_output

        except Exception as exc:
            logger.warning("Chunk {} LLM call failed: {}", chunk.chunk_id, exc)
            return timeline, None

    # ------------------------------------------------------------------
    # Detector bridge
    # ------------------------------------------------------------------

    def _detect_candidates(self, cue: SubtitleCue) -> list[dict]:
        """Run JiebaSpanDetector on a cue and return Top-K candidates with coordinates."""
        if self._detector is None:
            return []
        try:
            choices = self.matcher._build_choices()
            doc = SubtitleDocument(format="srt", cues=[cue])
            detection = self._detector.detect(cue, doc)
            spans = detection.metadata.get("spans", [])
            result = []
            seen = set()
            for s in spans:
                # Cold-pool probes are for category activation only,
                # NOT for repair candidates.  Mixing them in would flag
                # every normal cue as a problem_cue and risk false fixes.
                if s.get("source") == "pinyin_probe":
                    continue
                value = s.get("candidate_value", "")
                if value in seen:
                    continue
                seen.add(value)
                entry = choices.get(value)
                result.append({
                    "surface": s.get("surface_text", ""),
                    "value": value,
                    "score": round(s.get("score", 0), 3),
                    "source": s.get("source", "unknown"),
                    "category": getattr(entry, "category", "") if entry else "",
                    "parent_entity": getattr(entry, "parent_entity", "") if entry else "",
                    "start_char": s.get("start_char", 0),
                    "end_char": s.get("end_char", 0),
                    "metadata": s.get("metadata", {}),
                })
            return result[: self.max_candidates_per_cue]
        except Exception as exc:
            logger.warning("[Discovery] 检测器对 cue {} 失败: {}", cue.index, exc)
            return []

    @staticmethod
    def _context_window_for_cue(
        chunk: Chunk, cue_index: int, direction: int, window: int,
    ) -> list[dict]:
        """Get up to *window* cues before (-1) or after (+1) *cue_index*."""
        all_cues = list(chunk.context_before) + list(chunk.target_lines) + list(chunk.context_after)
        target_pos = next((i for i, c in enumerate(all_cues) if c.index == cue_index), -1)
        if target_pos == -1:
            return []
        result = []
        for step in range(1, window + 1):
            neighbor = target_pos + direction * step
            if 0 <= neighbor < len(all_cues):
                result.append({"cue_index": all_cues[neighbor].index, "text": all_cues[neighbor].text})
        if direction == -1:
            result.reverse()
        return result

    # ------------------------------------------------------------------
    # Timeline builders (pure code, no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entity_persistence(observations: list[CueObservation]) -> dict[str, float]:
        """How frequently does each entity appear across observations?"""
        entity_count: dict[str, int] = {}
        total = len(observations) or 1
        for obs in observations:
            for c in obs.candidates:
                pe = c.get("parent_entity", "")
                if pe:
                    entity_count[pe] = entity_count.get(pe, 0) + 1
        return {e: round(c / total, 3) for e, c in entity_count.items()}

    @staticmethod
    def _compute_category_flow(observations: list[CueObservation]) -> list[dict]:
        """Track category appearance order across cues."""
        seen: set[str] = set()
        flow: list[dict] = []
        for obs in observations:
            for c in obs.candidates:
                cat = c.get("category", "")
                if cat and cat not in seen:
                    seen.add(cat)
                    flow.append({"cue_index": obs.cue_index, "category": cat})
        return flow

    @staticmethod
    def _build_compressed_windows(
        observations: list[CueObservation],
    ) -> list[CompressedContextWindow]:
        """Group cues by dominant parent_entity, preserving context sequence."""
        from collections import defaultdict
        by_entity: dict[str, list[dict]] = defaultdict(list)
        for obs in observations:
            for c in obs.candidates:
                pe = c.get("parent_entity", "")
                if pe:
                    by_entity[pe].append({
                        "cue_index": obs.cue_index,
                        "text": obs.text,
                    })

        windows: list[CompressedContextWindow] = []
        for entity, ctx in by_entity.items():
            if len(ctx) < 2:
                continue
            cues = [item["cue_index"] for item in ctx]
            categories = list(set(
                c.get("category", "")
                for obs in observations
                for c in obs.candidates
                if c.get("parent_entity") == entity and c.get("category")
            ))
            windows.append(CompressedContextWindow(
                entity=entity,
                start_cue=min(cues),
                end_cue=max(cues),
                context_sequence=ctx,
                supporting_categories=categories,
                semantic_density=round(len(ctx) / max(1, len(observations)), 3),
            ))
        return windows

    # ------------------------------------------------------------------
    # LLM prompt + response
    # ------------------------------------------------------------------

    @staticmethod
    def _build_regions(
        chunk: Chunk, problem_cues: list[dict], ctx_window: int,
    ) -> list[dict]:
        """Merge overlapping context windows into deduplicated regions."""
        if not problem_cues:
            return []

        all_cues = list(chunk.context_before) + list(chunk.target_lines) + list(chunk.context_after)
        cue_map = {c.index: c for c in all_cues}
        sorted_cues = sorted(problem_cues, key=lambda p: p["cue_index"])

        regions: list[dict] = []
        current: dict | None = None

        for pc in sorted_cues:
            ci = pc["cue_index"]
            before = []
            for step in range(1, ctx_window + 1):
                neighbor = cue_map.get(ci - step)
                if neighbor is not None:
                    before.insert(0, {"cue_index": neighbor.index, "text": neighbor.text})
            after = []
            for step in range(1, ctx_window + 1):
                neighbor = cue_map.get(ci + step)
                if neighbor is not None:
                    after.append({"cue_index": neighbor.index, "text": neighbor.text})

            if current is None:
                current = {
                    "context_before": before,
                    "problem_cues": [pc],
                    "context_after": after,
                }
            else:
                # Check overlap: does this cue's before overlap with current's after?
                prev_after_max = max(
                    (a["cue_index"] for a in current["context_after"]), default=-1,
                )
                this_before_min = min(
                    (b["cue_index"] for b in before), default=99999,
                )
                if this_before_min <= prev_after_max + 1:
                    # Merge: extend current region
                    existing_indices = {a["cue_index"] for a in current["context_after"]}
                    for a in after:
                        if a["cue_index"] not in existing_indices:
                            current["context_after"].append(a)
                            existing_indices.add(a["cue_index"])
                    current["problem_cues"].append(pc)
                else:
                    regions.append(current)
                    current = {
                        "context_before": before,
                        "problem_cues": [pc],
                        "context_after": after,
                    }

        if current is not None:
            regions.append(current)

        return regions

    @staticmethod
    def _build_semantic_prompt(
        timeline: ChunkSemanticTimeline, regions: list[dict],
        density: float, ctx_window: int,
    ) -> str:
        import json
        compact = {
            "chunk_index": timeline.chunk_index,
            "problem_density": round(density, 3),
            "context_window": ctx_window,
            "entity_persistence": timeline.entity_persistence,
            "category_flow": timeline.category_flow,
            "regions": regions,
        }
        return PASS1_USER_TEMPLATE.format(
            chunk_data=json.dumps(compact, ensure_ascii=False, indent=2),
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        import time
        model = self.model or getattr(
            self.llm_client,
            "discovery_model",
            getattr(self.llm_client, "model", "mimo-v2.5-pro"),
        )
        for attempt in range(3):
            try:
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content
                logger.warning(
                    "Discovery LLM empty (attempt={}, prompt_len={})",
                    attempt + 1, len(user_prompt),
                )
                time.sleep(2.0)
            except Exception as exc:
                logger.warning("LLM call failed (attempt={}): {}", attempt + 1, exc)
                time.sleep(2.0)
        return ""

    @staticmethod
    def _parse_semantic_filter_output(
        raw: str, chunk: Chunk,
    ) -> SemanticFilterOutput | None:
        import json
        import re
        try:
            text = raw.strip()
            fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
            if fence:
                text = fence.group(1).strip()
            data = json.loads(text)
            return SemanticFilterOutput(
                chunk_index=chunk.chunk_id,
                confirmed_entities=data.get("confirmed_entities", []),
                dominant_categories=data.get("dominant_categories", []),
                semantic_signals=data.get("semantic_signals", []),
                possible_transition=data.get("possible_transition", False),
                transition_region=data.get("transition_region"),
                detector_noise=data.get("detector_noise", []),
            )
        except Exception as exc:
            logger.warning("Failed to parse SemanticFilterOutput: {}", exc)
            return None

    # ------------------------------------------------------------------
    # Entity memory bridge
    # ------------------------------------------------------------------

    def _update_entity_memory(self, sf_output: SemanticFilterOutput) -> None:
        """Write confirmed_entities back to EntityMemoryManager."""
        if self.entity_memory is None:
            return
        for ent in sf_output.confirmed_entities:
            name = ent.get("entity", "")
            if name and hasattr(self.entity_memory, "update_entity"):
                self.entity_memory.update_entity(name)

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _create_entity_snapshot(self) -> MappingProxyType:
        if self.entity_memory is not None and hasattr(self.entity_memory, "create_readonly_snapshot"):
            return self.entity_memory.create_readonly_snapshot()
        return MappingProxyType({})

    def _create_category_snapshot(self) -> MappingProxyType:
        if self.category_activation is not None and hasattr(self.category_activation, "create_readonly_snapshot"):
            return self.category_activation.create_readonly_snapshot()
        return MappingProxyType({})
