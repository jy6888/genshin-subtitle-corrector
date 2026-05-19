"""Per-entity knowledge cards for Phase2 LLM context.

Each card summarizes what the system knows about an entity:
its canonical name, category, nicknames to preserve, ASR errors to
repair, and a handful of related terminology terms.

Cards are injected into the Phase2 batch prompt so the LLM can
distinguish real character names from ASR errors without needing
pre-trained Genshin domain knowledge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subtitle_corrector.aliasing.policy import AliasPolicyRegistry
    from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher


class KnowledgeCardBuilder:
    """Build per-entity knowledge cards from alias & terminology sources."""

    def __init__(
        self,
        matcher: FuzzyTerminologyMatcher | None = None,
        policy_registry: AliasPolicyRegistry | None = None,
        character_lexicon: object | None = None,
        spoken_lexicon: object | None = None,
        asr_runtime: object | None = None,
    ) -> None:
        self._matcher = matcher
        self._policy = policy_registry
        self._char_lex = character_lexicon
        self._spoken_lex = spoken_lexicon
        self._asr_runtime = asr_runtime

        # canonical -> {preserve: set[str], repair: dict[str,str], contextual: set[str]}
        self._index: dict[str, dict] = {}
        self._build_index()

    # ── public API ─────────────────────────────────────────────────

    def build_cards(self, entities: set[str], max_cards: int = 5) -> list[dict]:
        """Return knowledge cards for the requested entities (max 5)."""
        cards: list[dict] = []
        for entity in sorted(entities):
            if len(cards) >= max_cards:
                break
            card = self._build_one(entity)
            if card is not None:
                cards.append(card)
        return cards

    def entities_for_batch(
        self, cues: list[dict], dominant_entities: list[str],
    ) -> set[str]:
        """Collect entities relevant to a batch: dominant + candidate parents."""
        entities: set[str] = set()
        for ent in dominant_entities:
            if ent:
                entities.add(ent)
        for cue in cues:
            for cand in cue.get("candidates", []):
                pe = cand.get("parent_entity", "")
                if pe:
                    entities.add(pe)
                val = cand.get("value", "")
                if val and val in self._index:
                    entities.add(val)
        return entities

    # ── index construction ─────────────────────────────────────────

    def _build_index(self) -> None:
        self._index_character_aliases()
        self._index_spoken_aliases()
        self._index_asr_aliases()

    def _index_character_aliases(self) -> None:
        if self._char_lex is None:
            return
        # 单字 team_slot → preserve (需配队语境，不独立进候选)
        for surface in getattr(self._char_lex, "team_slots", {}):
            entry = self._char_lex.team_slots[surface]
            canonical = entry.canonical_term
            slot = self._ensure_entity(canonical)
            slot["preserve"].add(surface)
        # 多字 nickname → contextual (可进候选，需语境支撑)
        for surface, entry in getattr(self._char_lex, "nicknames", {}).items():
            canonical = entry.canonical_term
            slot = self._ensure_entity(canonical)
            slot["contextual"].add(surface)

    def _index_spoken_aliases(self) -> None:
        if self._spoken_lex is None:
            return
        for surface, canonical in getattr(
            self._spoken_lex, "surface_to_canonical", {},
        ).items():
            slot = self._ensure_entity(canonical)
            policy_text = getattr(self._spoken_lex, "usage_policies", {}).get(
                surface, "context_only",
            )
            if policy_text in ("context_only", "exact_context_only"):
                slot["preserve"].add(surface)
            else:
                slot["contextual"].add(surface)

    def _index_asr_aliases(self) -> None:
        if self._asr_runtime is None:
            return
        iterator = getattr(self._asr_runtime, "iter_policy_entries", None)
        if iterator is None:
            return
        for alias in iterator():
            surface = alias.alias_surface
            canonical = alias.canonical_term
            status = getattr(alias, "review_status", "")
            slot = self._ensure_entity(canonical)
            if status == "approved":
                slot["repair"][surface] = canonical
            else:
                slot["contextual"].add(surface)

    def _ensure_entity(self, canonical: str) -> dict:
        if canonical not in self._index:
            self._index[canonical] = {
                "preserve": set(), "repair": {}, "contextual": set(),
            }
        return self._index[canonical]

    # ── single-card construction ───────────────────────────────────

    def _build_one(self, entity: str) -> dict | None:
        idx = self._index.get(entity)

        kind = "unknown"
        if self._matcher is not None:
            choices = self._matcher._build_choices()
            entry = choices.get(entity)
            if entry is not None:
                kind = getattr(entry, "category", "") or "unknown"

        preserve = sorted(idx["preserve"]) if idx else []
        repair = [
            {"surface": s, "canonical": c}
            for s, c in sorted((idx["repair"] if idx else {}).items())
        ]
        contextual = sorted(idx["contextual"]) if idx else []

        if not preserve and not repair and not contextual and kind == "unknown":
            return None

        related = (
            self._collect_related_terms(entity)
            if self._matcher is not None
            else []
        )

        return {
            "canonical": entity,
            "kind": kind,
            "preserve_aliases": preserve,
            "repair_aliases": repair,
            "contextual_aliases": contextual,
            "related_terms": related,
            "policy_notes": (
                "preserve_aliases 中的表面词是标准昵称/简称，原文正确出现时 KEEP，"
                "不展开为标准名。repair_aliases 是已审核 ASR 错听→正确映射，"
                "可以 REPLACE。contextual_aliases 仅在强语境支撑时展开。"
                "related_terms 仅供上下文参考，不能凭此列表自由生成新候选。"
            ),
        }

    def _collect_related_terms(self, entity: str) -> list[str]:
        """Gather up to 8 terms whose parent_entity matches *entity*."""
        if self._matcher is None:
            return []
        choices = self._matcher._build_choices()
        terms: list[str] = []
        seen: set[str] = {entity}
        for _surface, entry in choices.items():
            pe = getattr(entry, "parent_entity", "") or ""
            if pe != entity:
                continue
            term = entry.term
            if term not in seen:
                seen.add(term)
                terms.append(term)
                if len(terms) >= 8:
                    break
        return sorted(terms)
