"""Per-video confirmed correction rule memory.

When Phase2 LLM confirms a REPLACE with high confidence (e.g.
弗丽娜 -> 芙宁娜), the rule is learned and reapplied to other
cues in the same video that contain the same surface text.

This is ephemeral — lives only for one pipeline run, never persisted
to SQLite or the global terminology table.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class ConfirmedCorrectionRule:
    surface_text: str
    corrected_text: str
    category: str
    confidence_sum: float = 0.0
    count: int = 0
    conflict: bool = False

    @property
    def avg_confidence(self) -> float:
        return self.confidence_sum / self.count if self.count else 0.0


@dataclass
class CorrectionReuseSummary:
    learned_rules: int = 0
    applied_decisions: int = 0
    conflict_rules: int = 0
    skipped_low_confidence: int = 0
    skipped_short_surface: int = 0

    def to_log_message(self) -> str:
        return (
            f"CorrectionReuse: learned={self.learned_rules}, "
            f"applied={self.applied_decisions}, "
            f"conflict={self.conflict_rules}, "
            f"skipped_low_conf={self.skipped_low_confidence}, "
            f"skipped_short={self.skipped_short_surface}"
        )


class ConfirmedCorrectionMemory:
    """Per-video memory of LLM-confirmed corrections for reuse."""

    def __init__(
        self,
        min_confidence: float = 0.85,
        min_surface_cjk_len: int = 3,
        allowed_categories: set[str] | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_surface_cjk_len = min_surface_cjk_len
        self.allowed_categories = allowed_categories or {"character"}
        self._rules: dict[str, ConfirmedCorrectionRule] = {}
        self.summary = CorrectionReuseSummary()

    def learn(
        self,
        surface_text: str,
        corrected_text: str,
        category: str,
        confidence: float,
    ) -> None:
        """Learn a rule from a Phase2 REPLACE decision."""
        if confidence < self.min_confidence:
            self.summary.skipped_low_confidence += 1
            return

        cjk_len = sum(1 for ch in surface_text if "一" <= ch <= "鿿")
        if cjk_len < self.min_surface_cjk_len:
            self.summary.skipped_short_surface += 1
            return

        if category not in self.allowed_categories:
            return

        rule = self._rules.get(surface_text)
        if rule is None:
            self._rules[surface_text] = ConfirmedCorrectionRule(
                surface_text=surface_text,
                corrected_text=corrected_text,
                category=category,
                confidence_sum=confidence,
                count=1,
            )
            self.summary.learned_rules += 1
        elif rule.corrected_text != corrected_text:
            rule.conflict = True
            self.summary.conflict_rules += 1
        else:
            rule.confidence_sum += confidence
            rule.count += 1

    def get_rule(self, surface_text: str) -> ConfirmedCorrectionRule | None:
        """Return a non-conflict reuse rule, or None."""
        rule = self._rules.get(surface_text)
        if rule is None or rule.conflict:
            return None
        return rule

    def get_confirmed_correction(self, surface_text: str) -> str | None:
        """Return corrected_text if a valid rule exists, else None."""
        rule = self.get_rule(surface_text)
        return rule.corrected_text if rule else None


def learn_confirmed_rules(
    decisions: list,
    matcher: object,
    settings: object,
) -> ConfirmedCorrectionMemory:
    """Learn rules from Phase2 decisions.

    Only REPLACE decisions with confidence >= min_confidence whose
    corrected_text exists in the terminology matcher are learned.
    """
    reuse = ConfirmedCorrectionMemory(
        min_confidence=settings.correction_reuse.min_confidence,
        min_surface_cjk_len=settings.correction_reuse.min_surface_cjk_len,
        allowed_categories=set(settings.correction_reuse.allowed_categories),
    )

    choices = matcher._build_choices()

    for d in decisions:
        if d.action != "REPLACE":
            continue
        surface = getattr(d, "surface_text", "") or ""
        corrected = getattr(d, "corrected_text", "") or ""
        confidence = getattr(d, "confidence", 0.0)
        if not surface or not corrected:
            continue
        # Only learn if corrected_text is a valid terminology term
        if corrected not in choices:
            continue
        entry = choices[corrected]
        category = getattr(entry, "category", "") or ""
        reuse.learn(surface, corrected, category, confidence)

    logger.info(reuse.summary.to_log_message())
    return reuse


def augment_decisions_with_reuse(
    cues: list,
    decisions: list,
    memory: ConfirmedCorrectionMemory,
) -> tuple[list, CorrectionReuseSummary]:
    """Scan all cues for surfaces that have a confirmed rule but no
    existing Phase2 decision covering that span, and append REPLACE
    decisions for them."""
    from subtitle_corrector.pipeline.refinement import CandidateDecision, RepairAction

    summary = memory.summary
    summary.applied_decisions = 0

    # Build index of existing decision spans per cue
    existing_spans: dict[int, set[tuple[int, int]]] = {}
    for d in decisions:
        ci = getattr(d, "cue_index", -1)
        sc = getattr(d, "start_char", 0)
        ec = getattr(d, "end_char", 0)
        if ci >= 0 and sc < ec:
            existing_spans.setdefault(ci, set()).add((sc, ec))

    # Scan all cues for surfaces with learned rules
    new_decisions: list = list(decisions)
    for cue in cues:
        ci = cue.index
        text = cue.text
        covered = existing_spans.get(ci, set())

        for surface, rule in memory._rules.items():
            if rule.conflict:
                continue
            pos = 0
            while True:
                pos = text.find(surface, pos)
                if pos == -1:
                    break
                end = pos + len(surface)
                # Check span overlap
                if not _overlaps(pos, end, covered):
                    new_decisions.append(CandidateDecision(
                        cue_index=ci,
                        candidate_index=-1,  # reuse marker
                        action=RepairAction.REPLACE,
                        surface_text=surface,
                        corrected_text=rule.corrected_text,
                        start_char=pos,
                        end_char=end,
                        confidence=rule.avg_confidence,
                    ))
                    summary.applied_decisions += 1
                    covered.add((pos, end))
                pos = end

    logger.info("CorrectionReuse: applied {} reuse decisions", summary.applied_decisions)
    return new_decisions, summary


def _overlaps(start: int, end: int, covered: set[tuple[int, int]]) -> bool:
    for cs, ce in covered:
        if start < ce and end > cs:
            return True
    return False
