from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AliasPolicy(StrEnum):
    PRESERVE_SURFACE = "preserve_surface"
    REPAIR_TO_CANONICAL = "repair_to_canonical"
    CONTEXTUAL_EXPAND = "contextual_expand"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SurfacePolicyDecision:
    surface: str
    canonical: str
    policy: AliasPolicy
    source: str
    confidence: float = 0.5
    reason: str = ""


class AliasPolicyRegistry:
    def __init__(self) -> None:
        self._by_surface: dict[str, SurfacePolicyDecision] = {}

    @classmethod
    def from_sources(
        cls,
        spoken_alias_lexicon: object | None = None,
        character_lexicon: object | None = None,
        asr_alias_runtime: object | None = None,
    ) -> AliasPolicyRegistry:
        registry = cls()
        if spoken_alias_lexicon is not None:
            registry.add_spoken_aliases(spoken_alias_lexicon)
        if character_lexicon is not None:
            registry.add_character_aliases(character_lexicon)
        if asr_alias_runtime is not None:
            registry.add_asr_aliases(asr_alias_runtime)
        return registry

    def add_spoken_aliases(self, lexicon: object) -> None:
        for surface, canonical in getattr(lexicon, "surface_to_canonical", {}).items():
            policy_text = getattr(lexicon, "usage_policies", {}).get(
                surface,
                "context_only",
            )
            policy = (
                AliasPolicy.PRESERVE_SURFACE
                if policy_text in {"context_only", "exact_context_only"}
                else AliasPolicy.CONTEXTUAL_EXPAND
            )
            self._put(
                SurfacePolicyDecision(
                    surface=surface,
                    canonical=canonical,
                    policy=policy,
                    source="spoken_alias",
                    confidence=0.85,
                    reason=f"spoken alias policy={policy_text}",
                )
            )

    def add_character_aliases(self, lexicon: object) -> None:
        for surface, entry in getattr(lexicon, "all_aliases", {}).items():
            self._put(
                SurfacePolicyDecision(
                    surface=surface,
                    canonical=entry.canonical_term,
                    policy=AliasPolicy.PRESERVE_SURFACE,
                    source="character_alias",
                    confidence=float(getattr(entry, "confidence", 0.5)),
                    reason="known character nickname should preserve surface",
                )
            )

    def add_asr_aliases(self, runtime: object) -> None:
        iterator = getattr(runtime, "iter_policy_entries", None)
        if iterator is None:
            return
        for alias in iterator():
            status = getattr(alias, "review_status", "")
            policy = (
                AliasPolicy.REPAIR_TO_CANONICAL
                if status == "approved"
                else AliasPolicy.CONTEXTUAL_EXPAND
            )
            self._put(
                SurfacePolicyDecision(
                    surface=alias.alias_surface,
                    canonical=alias.canonical_term,
                    policy=policy,
                    source="asr_alias",
                    confidence=float(getattr(alias, "confidence", 0.5)),
                    reason=f"reviewed ASR alias status={status}",
                )
            )

    def resolve(self, surface: str, target: str = "") -> SurfacePolicyDecision:
        decision = self._by_surface.get(surface)
        if decision is None:
            return SurfacePolicyDecision(
                surface=surface,
                canonical=target,
                policy=AliasPolicy.UNKNOWN,
                source="unknown",
                confidence=0.0,
                reason="surface not found in alias policy registry",
            )
        return decision

    def may_expand(self, surface: str, target: str) -> bool:
        if not target or len(target) <= len(surface):
            return True
        policy = self.resolve(surface, target).policy
        return policy in {
            AliasPolicy.REPAIR_TO_CANONICAL,
            AliasPolicy.CONTEXTUAL_EXPAND,
        }

    def _put(self, decision: SurfacePolicyDecision) -> None:
        existing = self._by_surface.get(decision.surface)
        if existing is None or _priority(decision.policy) >= _priority(existing.policy):
            self._by_surface[decision.surface] = decision


def _priority(policy: AliasPolicy) -> int:
    return {
        AliasPolicy.PRESERVE_SURFACE: 30,
        AliasPolicy.REPAIR_TO_CANONICAL: 20,
        AliasPolicy.CONTEXTUAL_EXPAND: 10,
        AliasPolicy.UNKNOWN: 0,
    }[policy]
