"""Phase 2: Context-Aware Repair — per-candidate decisions with Span coordinates.

Groups NEEDS_REVIEW cues by SemanticCluster, sends batched LLM calls
with topic context.  LLM selects candidate_index (not raw text), code
resolves precise coordinates from detector spans for safe multi-point
right-to-left replacement.

Each cue can have multiple candidates — the LLM decides independently
per candidate, not per cue.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from subtitle_corrector.pipeline.reducer import SemanticCluster
    from subtitle_corrector.schemas import ChunkSemanticTimeline, SubtitleCue


_GENERIC_LOADOUT_SURFACES = (
    "标准",
    "输出",
    "生成",
    "输出套",
    "输出装",
    "生成装",
    "标准输出",
    "标准生成",
    "装",
    "套",
)
_CHARACTERISH_CATEGORIES = {"character", "team_comp_alias"}
_GENERIC_REPLACEMENT_SAFE_EVIDENCE = {
    "asr_alias",
    "team_comp_reviewed_override",
}


class RepairAction(str, Enum):
    REPLACE = "REPLACE"
    KEEP = "KEEP"
    REVIEW = "REVIEW"
    REQUERY = "REQUERY"


@dataclass
class CandidateDecision:
    cue_index: int
    candidate_index: int
    action: RepairAction
    surface_text: str       # the original error text in the cue
    corrected_text: str     # the candidate value
    start_char: int = 0     # resolved from detector span
    end_char: int = 0
    confidence: float = 0.5
    metadata: dict = field(default_factory=dict)


@dataclass
class Phase2Result:
    decisions: list[CandidateDecision] = field(default_factory=list)
    total_candidates: int = 0
    replaced: int = 0
    kept: int = 0
    reviewed: int = 0
    # Visibility fields
    input_candidates: int = 0
    failed_batches: int = 0
    unparsed_candidates: int = 0
    # Requery
    requery_requested: int = 0
    requery_candidates: int = 0
    requery_replaced: int = 0
    requery_skipped: int = 0
    requery_requests: list[RequeryRequest] = field(default_factory=list)
    requery_generated_by_type: dict[str, int] = field(default_factory=dict)
    requery_skipped_by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class RequeryRequest:
    cue_index: int
    suspect_surface: str
    target_hint: str
    confidence: float = 0.5
    reason: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Phase2PlanSummary:
    input_cues: int = 0
    input_candidates: int = 0
    groups: int = 0
    batches: int = 0
    max_per_batch: int = 10
    max_workers: int = 4
    max_candidates_per_cue: int = 0
    total_prompt_chars: int = 0
    max_prompt_chars: int = 0

    def to_log_message(self) -> str:
        return (
            f"Phase2 plan: cues={self.input_cues}, "
            f"input_candidates={self.input_candidates}, "
            f"groups={self.groups}, batches={self.batches}, "
            f"max_per_batch={self.max_per_batch}, "
            f"max_workers={self.max_workers}, "
            f"max_candidates_per_cue={self.max_candidates_per_cue}, "
            f"prompt_chars_total={self.total_prompt_chars}, "
            f"prompt_chars_max={self.max_prompt_chars}"
        )


class Phase2RefinementEngine:
    """Batched context-aware repair with Span-precise coordinates."""

    def __init__(
        self,
        llm_client: object,
        max_per_batch: int = 10,
        max_workers: int = 4,
        model: str | None = None,
        retry_failed_batch_as_single_cue: bool = True,
        asr_alias_runtime: object | None = None,
        team_comp_parser: object | None = None,
        terminology_hint_builder: object | None = None,
        knowledge_card_builder: object | None = None,
        protected_surface_to_canonical: dict[str, str] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_per_batch = max_per_batch
        self.retry_failed = retry_failed_batch_as_single_cue
        self.max_workers = max_workers
        self.model = model
        self.asr_alias_runtime = asr_alias_runtime
        self.team_comp_parser = team_comp_parser
        self.terminology_hint_builder = terminology_hint_builder
        self.knowledge_card_builder = knowledge_card_builder
        self.protected_surface_to_canonical = protected_surface_to_canonical or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(
        self,
        timelines: list[ChunkSemanticTimeline],
        clusters: list[SemanticCluster],
        original_cues: list[SubtitleCue],
        matcher: object = None,
    ) -> Phase2Result:
        all_cues = self._collect_review_cues(timelines)

        if not all_cues:
            logger.info("Phase2: 0 NEEDS_REVIEW cues")
            return Phase2Result()

        grouped = self._group_by_cluster(all_cues, clusters)

        # Build all batch tasks (pre-compute prompts, no LLM yet)
        tasks = self._build_tasks(grouped)
        plan_summary = self._summarize_plan(all_cues, grouped, tasks)
        logger.info(plan_summary.to_log_message())

        # --- Round 1: concurrent LLM calls ---
        result = Phase2Result(input_candidates=plan_summary.input_candidates)
        results_by_idx, failed_indices = self._run_batches(tasks)
        requery_requests: list[RequeryRequest] = []

        # Collect first-round decisions, separate REQUERY
        for idx in sorted(results_by_idx):
            for d in results_by_idx[idx]:
                if d.action == RepairAction.REQUERY:
                    requery_requests.append(RequeryRequest(
                        cue_index=d.cue_index,
                        suspect_surface=d.surface_text,
                        target_hint=d.corrected_text,
                        confidence=d.confidence,
                        metadata=d.metadata,
                    ))
                    continue
                result.decisions.append(d)
                result.total_candidates += 1
                if d.action == RepairAction.REPLACE:
                    result.replaced += 1
                elif d.action == RepairAction.KEEP:
                    result.kept += 1
                else:
                    result.reviewed += 1

        # --- Requery: generate targeted candidates ---
        if requery_requests:
            result.requery_requested = len(requery_requests)
            result.requery_requests = list(requery_requests)
            if matcher is None:
                logger.warning("Phase2: {} REQUERY requests dropped (no matcher)", len(requery_requests))
                result.requery_skipped = len(requery_requests)
            else:
                from subtitle_corrector.pipeline.requery import RequeryCandidateEngine
                req_engine = RequeryCandidateEngine(
                    matcher,
                    asr_alias_runtime=self.asr_alias_runtime,
                    team_comp_parser=self.team_comp_parser,
                )
                requery_cands = req_engine.build_candidates(
                    requery_requests, original_cues, clusters,
                )
                requery_cands = self._filter_protected_requery_candidates(
                    requery_cands,
                    original_cues,
                )
                result.requery_candidates = len(requery_cands)
                result.requery_generated_by_type = dict(
                    req_engine.last_result.generated_by_type
                )
                result.requery_skipped_by_type = dict(
                    req_engine.last_result.skipped_by_type
                )

                if requery_cands:
                    # Build second-round batches from requery candidates
                    r2_tasks = self._build_requery_tasks(requery_cands, original_cues, clusters)
                    r2_results, r2_failed = self._run_batches(r2_tasks, allow_requery=False)
                    result.failed_batches += len(r2_failed)
                    result.unparsed_candidates += sum(
                        len(r2_tasks[idx]["batch"]) for idx in r2_failed
                    )

                    # Merge second-round results
                    for idx in sorted(r2_results):
                        for d in r2_results[idx]:
                            result.decisions.append(d)
                            result.total_candidates += 1
                            if d.action == RepairAction.REPLACE:
                                result.replaced += 1
                                result.requery_replaced += 1
                            elif d.action == RepairAction.KEEP:
                                result.kept += 1
                            else:
                                result.reviewed += 1
                else:
                    result.requery_skipped = len(requery_requests)

        result.failed_batches += len(failed_indices)
        result.unparsed_candidates += sum(
            len(tasks[idx]["batch"]) for idx in failed_indices
        )

        logger.info(
            "Phase2 complete: parsed={} of input={} → {} REPLACE / {} KEEP / {} REVIEW, "
            "failed_batches={}, unparsed_candidates={}, "
            "requery_req={} requery_cand={} requery_rep={} requery_skip={}",
            result.total_candidates, result.input_candidates,
            result.replaced, result.kept, result.reviewed,
            result.failed_batches, result.unparsed_candidates,
            result.requery_requested, result.requery_candidates, result.requery_replaced,
            result.requery_skipped,
        )
        return result

    def _filter_protected_requery_candidates(
        self,
        candidates: list[dict],
        cues: list,
    ) -> list[dict]:
        if not self.protected_surface_to_canonical:
            return candidates
        filtered: list[dict] = []
        for candidate in candidates:
            cue_index = candidate.get("cue_index", -1)
            if cue_index < 0 or cue_index >= len(cues):
                filtered.append(candidate)
                continue
            if self._candidate_hits_protected_surface(
                candidate,
                cues[cue_index].text,
            ):
                continue
            filtered.append(candidate)
        return filtered

    def _candidate_hits_protected_surface(
        self,
        candidate: dict,
        text: str,
    ) -> bool:
        c_start = candidate.get("start_char", -1)
        c_end = candidate.get("end_char", -1)
        c_value = candidate.get("value", "")
        if c_start < 0 or c_end < 0 or not c_value:
            return False
        for surface, canonical in self.protected_surface_to_canonical.items():
            if len(surface) < 2 or c_value != canonical:
                continue
            pos = 0
            while True:
                pos = text.find(surface, pos)
                if pos == -1:
                    break
                if c_start < pos + len(surface) and c_end > pos:
                    return True
                pos += 1
        return False

    def _collect_review_cues(self, timelines: list[ChunkSemanticTimeline]) -> list[dict]:
        """Collect ordinary candidate cues plus narrow REQUERY-only hints."""
        all_cues: list[dict] = []
        for tl in timelines:
            for obs in tl.observations:
                cue = {
                    "cue_index": obs.cue_index,
                    "text": obs.text,
                    "context_before": obs.context_before,
                    "context_after": obs.context_after,
                    "candidates": obs.candidates,
                }
                if obs.candidates:
                    all_cues.append(cue)

                hints = self._no_candidate_requery_hints(obs)
                if hints:
                    cue["requery_hints"] = hints
                    if not obs.candidates:
                        all_cues.append(cue)
        return all_cues

    def _no_candidate_requery_hints(self, obs) -> list[dict]:
        hints: list[dict] = []
        hints.extend(self._team_comp_requery_hints(obs.text))
        hints.extend(self._terminology_requery_hints(
            obs.text, set(obs.active_entities or {}),
        ))
        return hints

    def _terminology_requery_hints(self, text: str, stable_entities: set[str]) -> list[dict]:
        if self.terminology_hint_builder is None:
            return []
        finder = getattr(self.terminology_hint_builder, "find_requery_hints", None)
        if finder is None:
            return []
        hints = []
        for hint in finder(text, stable_entities):
            suspect = hint.get("suspect_surface", "")
            target = hint.get("target_hint", "")
            if not suspect or not target:
                continue
            hints.append({
                "requery_type": "terminology_phonetic",
                "suspect_surface": suspect,
                "target_hint": target,
            })
        return hints

    def _team_comp_requery_hints(self, text: str) -> list[dict]:
        if self.team_comp_parser is None:
            return []
        finder = getattr(self.team_comp_parser, "find_requery_hints", None)
        if finder is None:
            return []
        return [
            {"requery_type": "team_comp_alias", "suspect_surface": surface}
            for surface in finder(text)
        ]

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def _run_batches(
        self, tasks: list[dict], allow_requery: bool = True,
    ) -> tuple[dict[int, list[CandidateDecision]], set[int]]:
        """Run all batch LLM calls concurrently. Returns (results_by_idx, failed_indices)."""
        results_by_idx: dict[int, list[CandidateDecision]] = {}
        failed_indices: set[int] = set()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._call_llm, task["prompt"]): idx
                for idx, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    raw = future.result()
                    batch_decisions = self._parse_batch_response(
                        raw, tasks[idx]["batch"], allow_requery=allow_requery,
                    )
                except Exception as exc:
                    logger.warning("Phase2 batch {} LLM failed: {}", idx, exc)
                    batch_decisions = []

                if self.retry_failed and not batch_decisions and tasks[idx]["batch"]:
                    logger.info("Phase2 batch {} JSON failed, retrying as single cues", idx)
                    retry = self._retry_split_batch(
                        tasks[idx]["batch"],
                        topic=tasks[idx].get("topic_label", "未知话题"),
                        entities=tasks[idx].get("stable_entities", ""),
                        topic_context=tasks[idx].get("topic_context"),
                    )
                    if not retry:
                        failed_indices.add(idx)
                    batch_decisions = retry

                results_by_idx[idx] = batch_decisions
        return results_by_idx, failed_indices

    def _build_requery_tasks(
        self,
        requery_cands: list[dict],
        cues: list,
        clusters: list[SemanticCluster],
    ) -> list[dict]:
        """Build single-cue batch tasks from requery candidates, with real text."""
        by_cue: dict[int, list[dict]] = {}
        for c in requery_cands:
            ci = c.get("cue_index", -1)
            if ci >= 0:
                by_cue.setdefault(ci, []).append(c)

        tasks: list[dict] = []
        for ci, cands in by_cue.items():
            text = cues[ci].text if ci < len(cues) else ""
            cue_dict = {"cue_index": ci, "text": text, "candidates": cands,
                        "context_before": "", "context_after": ""}
            batch = [cue_dict]
            topic_context = self._cluster_context_for_cue(ci, clusters)
            prompt = self._build_batch_prompt(
                batch,
                topic_context["topic_label"],
                topic_context["stable_entities"],
                set(),
                topic_context=topic_context,
            )
            tasks.append({"batch": batch, "prompt": prompt,
                          "topic_label": topic_context["topic_label"],
                          "stable_entities": topic_context["stable_entities"],
                          "topic_context": topic_context})
        return tasks

    @staticmethod
    def _cluster_context_for_cue(
        cue_index: int,
        clusters: list[SemanticCluster],
    ) -> dict[str, str]:
        for cluster in clusters:
            if cluster.temporal_range[0] <= cue_index <= cluster.temporal_range[1]:
                return Phase2RefinementEngine._topic_context_for_cluster(cluster)
        return Phase2RefinementEngine._topic_context_for_cluster(None)

    @staticmethod
    def _topic_context_for_cluster(cluster: SemanticCluster | None) -> dict[str, str]:
        if cluster is None:
            return {
                "topic_label": "未知话题",
                "stable_entities": "无",
                "semantic_categories": "无",
                "semantic_signals": "无",
                "temporal_range": "未知",
            }
        entities = [e for e in getattr(cluster, "dominant_entities", []) if e]
        categories = [c for c in getattr(cluster, "categories", []) if c]
        signals = [s for s in getattr(cluster, "signals", []) if s]
        label = (getattr(cluster, "topic_label", "") or "").strip()
        if not label:
            label_parts = []
            if entities:
                label_parts.append(", ".join(entities[:2]))
            if categories:
                label_parts.append(", ".join(categories[:2]))
            label = " / ".join(label_parts) or getattr(cluster, "cluster_id", "") or "未知话题"
        start, end = getattr(cluster, "temporal_range", (0, 0))
        return {
            "topic_label": label,
            "stable_entities": ", ".join(entities[:5]) if entities else "无",
            "semantic_categories": ", ".join(categories[:5]) if categories else "无",
            "semantic_signals": ", ".join(signals[:5]) if signals else "无",
            "temporal_range": f"{start}-{end}",
        }

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_cluster(cues: list[dict], clusters: list[SemanticCluster]) -> list[dict]:
        if not clusters:
            return [{"cues": cues, "cluster": None}]
        grouped: list[dict] = []
        unassigned: list[dict] = []
        for cue in cues:
            ci = cue["cue_index"]
            assigned = False
            for c in clusters:
                if c.temporal_range[0] <= ci <= c.temporal_range[1]:
                    for g in grouped:
                        if g["cluster"] is c:
                            g["cues"].append(cue)
                            assigned = True
                            break
                    if not assigned:
                        grouped.append({"cues": [cue], "cluster": c})
                        assigned = True
                    break
            if not assigned:
                unassigned.append(cue)
        if unassigned:
            grouped.append({"cues": unassigned, "cluster": None})
        return grouped

    def _build_tasks(self, grouped: list[dict]) -> list[dict]:
        tasks: list[dict] = []
        for group in grouped:
            cues_in_group = group["cues"]
            cluster = group["cluster"]
            topic_context = self._topic_context_for_cluster(cluster)
            topic_label = topic_context["topic_label"]
            dominant = cluster.dominant_entities if cluster else []
            stable_entities = topic_context["stable_entities"]
            for batch_start in range(0, len(cues_in_group), self.max_per_batch):
                batch = cues_in_group[batch_start : batch_start + self.max_per_batch]
                noise = set()
                for cue in batch:
                    for cand in cue["candidates"]:
                        if cand.get("score", 0) < 0.85:
                            noise.add(cand.get("value", ""))
                knowledge_cards = self._build_knowledge_cards(batch, dominant)
                prompt = self._build_batch_prompt(
                    batch, topic_label, stable_entities, noise, knowledge_cards,
                    topic_context=topic_context,
                )
                tasks.append({
                    "batch": batch, "prompt": prompt,
                    "topic_label": topic_label,
                    "stable_entities": stable_entities,
                    "topic_context": topic_context,
                })
        return tasks

    def _build_knowledge_cards(
        self, batch: list[dict], dominant_entities: list[str],
    ) -> list[dict]:
        if self.knowledge_card_builder is None:
            return []
        entities = self.knowledge_card_builder.entities_for_batch(
            batch, dominant_entities,
        )
        return self.knowledge_card_builder.build_cards(entities)

    def _summarize_plan(
        self,
        cues: list[dict],
        grouped: list[dict],
        tasks: list[dict],
    ) -> Phase2PlanSummary:
        prompt_lengths = [len(task["prompt"]) for task in tasks]
        candidate_counts = [len(cue.get("candidates", [])) for cue in cues]
        return Phase2PlanSummary(
            input_cues=len(cues),
            input_candidates=sum(candidate_counts),
            groups=len(grouped),
            batches=len(tasks),
            max_per_batch=self.max_per_batch,
            max_workers=self.max_workers,
            max_candidates_per_cue=max(candidate_counts, default=0),
            total_prompt_chars=sum(prompt_lengths),
            max_prompt_chars=max(prompt_lengths, default=0),
        )

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    def _retry_split_batch(
        self,
        batch: list[dict],
        topic: str = "未知话题",
        entities: str = "",
        topic_context: dict[str, str] | None = None,
    ) -> list[CandidateDecision]:
        """Split a failed batch into single-cue subtasks and retry each."""
        decisions: list[CandidateDecision] = []
        for cue in batch:
            knowledge_cards = self._build_knowledge_cards([cue], [])
            prompt = self._build_batch_prompt(
                [cue], topic=topic, entities=entities, noise=set(),
                knowledge_cards=knowledge_cards, topic_context=topic_context,
            )
            try:
                raw = self._call_llm(prompt)
                cue_decisions = self._parse_batch_response(raw, [cue])
                decisions.extend(cue_decisions)
            except Exception as exc:
                logger.warning("Phase2 single-cue retry for cue {} failed: {}", cue.get("cue_index"), exc)
        return decisions

    # ------------------------------------------------------------------
    # Prompt + LLM
    # ------------------------------------------------------------------

    @staticmethod
    def _build_batch_prompt(
        cues: list[dict],
        topic: str,
        entities: str,
        noise: set,
        knowledge_cards: list[dict] | None = None,
        topic_context: dict[str, str] | None = None,
    ) -> str:
        import json
        from subtitle_corrector.resolver.pass2_prompt import (
            PASS2_USER_TEMPLATE,
        )
        cues_data = []
        for cue in cues:
            cues_data.append({
                "cue_index": cue["cue_index"],
                "text": cue["text"],
                "context_before": cue.get("context_before", ""),
                "context_after": cue.get("context_after", ""),
                "requery_hints": cue.get("requery_hints", []),
                "candidates": [
                    {
                        "index": i, "value": c.get("value", ""),
                        "score": c.get("score", 0),
                        "category": c.get("category", ""),
                        "surface": c.get("surface", ""),
                        "source": c.get("source", ""),
                        "match_kind": c.get("match_kind", ""),
                        "alignment_score": c.get("alignment_score", 0),
                        "surface_coverage": c.get("surface_coverage", 0),
                        "target_coverage": c.get("target_coverage", 0),
                        "intent": c.get("metadata", {}).get("intent", ""),
                        "expansion_policy": c.get(
                            "expansion_policy",
                            c.get("metadata", {}).get("expansion_policy", ""),
                        ),
                        "evidence_type": c.get(
                            "evidence_type",
                            c.get("metadata", {}).get("evidence_type", ""),
                        ),
                        "canonical_hint": c.get("metadata", {}).get(
                            "canonical_hint",
                            "",
                        ),
                    }
                    for i, c in enumerate(cue.get("candidates", []))
                ],
            })
        cards_json = json.dumps(
            knowledge_cards or [], ensure_ascii=False, indent=2,
        )
        context = topic_context or {
            "topic_label": topic,
            "stable_entities": entities or "无",
            "semantic_categories": "无",
            "semantic_signals": "无",
            "temporal_range": "未知",
        }
        return PASS2_USER_TEMPLATE.format(
            topic_label=context["topic_label"],
            stable_entities=context["stable_entities"],
            semantic_categories=context["semantic_categories"],
            semantic_signals=context["semantic_signals"],
            temporal_range=context["temporal_range"],
            detector_noise=", ".join(noise) if noise else "无",
            knowledge_cards_json=cards_json,
            cues_json=json.dumps(cues_data, ensure_ascii=False, indent=2),
        )

    def _call_llm(self, user_prompt: str) -> str:
        import time
        from subtitle_corrector.resolver.pass2_prompt import PASS2_SYSTEM_PROMPT
        model = self.model or getattr(
            self.llm_client,
            "phase2_model",
            getattr(self.llm_client, "model", "mimo-v2.5"),
        )
        for attempt in range(3):
            try:
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": PASS2_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.05, max_tokens=4096,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content
                logger.warning(
                    "Phase2 LLM empty (attempt={}, prompt_len={})",
                    attempt + 1, len(user_prompt),
                )
                time.sleep(2.0)
            except Exception as exc:
                logger.warning("Phase2 LLM call failed (attempt={}): {}", attempt + 1, exc)
                time.sleep(2.0)
        return ""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_batch_response(
        raw: str, batch: list[dict], allow_requery: bool = True,
    ) -> list[CandidateDecision]:
        import json
        import re
        if not raw or not raw.strip():
            return []
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            data = json.loads(text)
            raw_decisions = data.get("decisions", [])
        except json.JSONDecodeError:
            logger.warning("Phase2 JSON parse failed, raw (前200字): {}", raw[:200])
            return []

        decisions: list[CandidateDecision] = []
        for rd in raw_decisions:
            ci = rd.get("cue_index", -1)
            cand_idx = rd.get("candidate_index", -1)
            action_str = rd.get("action", "KEEP").upper()

            # REQUERY: only in first round; second round forces REVIEW
            if action_str == "REQUERY":
                if allow_requery:
                    decisions.append(CandidateDecision(
                        cue_index=ci,
                        candidate_index=-1,
                        action=RepairAction.REQUERY,
                        surface_text=rd.get("suspect_surface", ""),
                        corrected_text=rd.get("target_hint", ""),
                        confidence=float(rd.get("confidence", 0.5)),
                        metadata={
                            "requery_type": rd.get(
                                "requery_type",
                                "terminology_phonetic",
                            ),
                            "reason": rd.get("reason", ""),
                        },
                    ))
                else:
                    logger.info("Phase2 round-2 REQUERY suppressed for cue {}", ci)
                continue

            cue = next((c for c in batch if c["cue_index"] == ci), None)
            if cue is None or cand_idx < 0:
                continue
            cands = cue.get("candidates", [])
            if cand_idx >= len(cands):
                continue
            candidate = cands[cand_idx]

            action = RepairAction.KEEP
            if action_str == "REPLACE":
                action = RepairAction.REPLACE
            elif action_str == "REVIEW":
                action = RepairAction.REVIEW
            # REJECT from prompt → KEEP

            metadata = Phase2RefinementEngine._decision_metadata_for_candidate(
                candidate,
            )
            if (
                action == RepairAction.REPLACE
                and Phase2RefinementEngine._is_generic_loadout_to_character(
                    candidate,
                )
            ):
                action = RepairAction.KEEP
                metadata["guard_blocked"] = "generic_loadout_to_character"

            decisions.append(CandidateDecision(
                cue_index=ci,
                candidate_index=cand_idx,
                action=action,
                surface_text=candidate.get("surface", ""),
                corrected_text=candidate.get("value", ""),
                start_char=candidate.get("start_char", 0),
                end_char=candidate.get("end_char", 0),
                confidence=float(rd.get("confidence", 0.5)),
                metadata=metadata,
            ))

        return decisions

    @staticmethod
    def _decision_metadata_for_candidate(candidate: dict) -> dict:
        candidate_metadata = Phase2RefinementEngine._candidate_metadata(candidate)
        return {
            "candidate_source": candidate.get("source", ""),
            "candidate_category": candidate.get("category", ""),
            "evidence_type": Phase2RefinementEngine._candidate_field(
                candidate, "evidence_type",
            ),
            "expansion_policy": Phase2RefinementEngine._candidate_field(
                candidate, "expansion_policy",
            ),
            "candidate_metadata": candidate_metadata,
        }

    @staticmethod
    def _candidate_metadata(candidate: dict) -> dict:
        metadata = candidate.get("metadata") or {}
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _candidate_field(candidate: dict, key: str) -> str:
        return str(
            candidate.get(key)
            or Phase2RefinementEngine._candidate_metadata(candidate).get(key)
            or "",
        )

    @staticmethod
    def _is_generic_loadout_to_character(candidate: dict) -> bool:
        surface = str(candidate.get("surface", "") or "")
        if not surface:
            return False
        source = str(candidate.get("source", "") or "")
        evidence_type = Phase2RefinementEngine._candidate_field(
            candidate, "evidence_type",
        )
        match_kind = Phase2RefinementEngine._candidate_field(candidate, "match_kind")
        if (
            source in _GENERIC_REPLACEMENT_SAFE_EVIDENCE
            or evidence_type in _GENERIC_REPLACEMENT_SAFE_EVIDENCE
            or match_kind in _GENERIC_REPLACEMENT_SAFE_EVIDENCE
        ):
            return False

        if not any(term in surface for term in _GENERIC_LOADOUT_SURFACES):
            return False

        category = str(candidate.get("category", "") or "")
        if category in _CHARACTERISH_CATEGORIES:
            return True
        return source in {"requery_phonetic", "team_comp_requery"}
