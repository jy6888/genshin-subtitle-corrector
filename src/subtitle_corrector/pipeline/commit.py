"""Phase 6: Final Commit — the only layer allowed to modify subtitles.

Consumes CandidateDecision from Phase 2, resolves precise character
coordinates from original cue text, and applies multi-point right-to-left
replacement via EditPlan.  Each replacement carries a provenance chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ReplaceSpan:
    cue_index: int
    start_char: int
    end_char: int
    replacement: str
    surface_text: str
    provenance: list[str] = field(default_factory=list)


@dataclass
class EditPlan:
    replacements: list[ReplaceSpan] = field(default_factory=list)

    def apply(self, cues: list) -> list:
        """Apply all replacements to *cues*, returning a new list.

        Replacements within the same cue are applied right-to-left
        (already sorted by commit()) to avoid index shifting.
        """
        repaired = [cue.model_copy(deep=True) for cue in cues]
        # Group by cue_index
        by_cue: dict[int, list[ReplaceSpan]] = {}
        for r in self.replacements:
            by_cue.setdefault(r.cue_index, []).append(r)

        for cue_index, replacements in by_cue.items():
            if cue_index >= len(repaired):
                continue
            # Sort right-to-left within this cue
            sorted_r = sorted(replacements, key=lambda r: r.start_char, reverse=True)
            chars = list(repaired[cue_index].text)
            for r in sorted_r:
                if 0 <= r.start_char < r.end_char <= len(chars):
                    actual = "".join(chars[r.start_char : r.end_char])
                    if actual == r.surface_text:
                        chars[r.start_char : r.end_char] = list(r.replacement)
                    else:
                        logger.warning(
                            "Surface mismatch at cue {}: expected '{}', found '{}'",
                            cue_index, r.surface_text, actual,
                        )
            repaired[cue_index].text = "".join(chars)

        return repaired


class CommitPlanner:
    """Build EditPlan from Phase2 CandidateDecision list."""

    @staticmethod
    def plan(decisions: list, original_cues: list) -> EditPlan:
        """Resolve character coordinates for each REPLACE decision.

        Uses the original cue text to locate surface_text — no LLM
        coordinates needed.  Structural safety only: no name lists.
        """
        replacements: list[ReplaceSpan] = []

        for d in decisions:
            if d.action.value != "REPLACE":
                continue
            if d.cue_index >= len(original_cues):
                continue

            # Structural safety: surface must be non-empty
            if not d.surface_text:
                logger.info("Commit skip cue {}: empty surface_text", d.cue_index)
                continue

            cue_text = original_cues[d.cue_index].text
            surface = d.surface_text

            # Use detector span coordinates if available, fallback to find()
            start = getattr(d, "start_char", 0)
            end = getattr(d, "end_char", 0)
            if 0 <= start < end <= len(cue_text) and cue_text[start:end] == surface:
                pos = start
            else:
                pos = cue_text.find(surface)

            if pos == -1:
                logger.warning(
                    "Cannot locate '{}' in cue {} '{}' — skipping",
                    surface, d.cue_index, cue_text[:40],
                )
                continue

            replacements.append(ReplaceSpan(
                cue_index=d.cue_index,
                start_char=pos,
                end_char=pos + len(surface),
                replacement=d.corrected_text,
                surface_text=surface,
                provenance=[f"Phase2:{d.action.value}", f"confidence:{d.confidence:.2f}"],
            ))

        # Sort right-to-left for safe multi-point replacement
        replacements.sort(key=lambda r: (r.cue_index, -r.start_char))
        return EditPlan(replacements=replacements)


def commit(decisions: list, original_cues: list) -> list:
    """One-shot: plan + apply."""
    plan = CommitPlanner.plan(decisions, original_cues)
    logger.info(
        "EditPlan: {} replacements across {} cues",
        len(plan.replacements),
        len(set(r.cue_index for r in plan.replacements)),
    )
    return plan.apply(original_cues)
