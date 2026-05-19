"""FN attribution: classify WHY each false negative (missed correction) happened.

Replaces the coarse "OTHER" bucket with 7 actionable categories so the
next optimization round knows whether to fix candidate generation, Phase2
judgment, REQUERY, or evaluation scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter


class FNBucket:
    WRONG_CANDIDATE_ONLY = "WRONG_CANDIDATE_ONLY"
    NON_TERMINOLOGY_DIFF = "NON_TERMINOLOGY_DIFF"
    PARTIAL_FIX = "PARTIAL_FIX"
    PHASE2_REJECTED = "PHASE2_REJECTED"
    REQUERY_FAILED = "REQUERY_FAILED"
    NO_CANDIDATE = "NO_CANDIDATE"
    UNCLASSIFIED = "UNCLASSIFIED"

    ALL = [
        WRONG_CANDIDATE_ONLY, NON_TERMINOLOGY_DIFF, PARTIAL_FIX,
        PHASE2_REJECTED, REQUERY_FAILED, NO_CANDIDATE, UNCLASSIFIED,
    ]


@dataclass
class FNAttribution:
    cue_index: int
    bucket: str
    gt_term: str = ""
    candidate_values: list[str] = field(default_factory=list)
    decision_action: str = ""
    sample: str = ""
    raw_text: str = ""
    repaired_text: str = ""
    gt_text: str = ""
    trace: object | None = None


@dataclass
class FNAttributionReport:
    attributions: list[FNAttribution] = field(default_factory=list)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    terms_by_bucket: dict[str, Counter] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.bucket_counts.values())

    @property
    def unclassified(self) -> int:
        return self.bucket_counts.get(FNBucket.UNCLASSIFIED, 0)


# CJK range for terminology detection
_CJK_START, _CJK_END = "一", "鿿"


def _extract_gt_terms(raw_text: str, gt_text: str) -> list[str]:
    """Extract terminology targets from GT diff using n-gram comparison.

    Finds CJK substrings (len >= 2) present in GT but absent from raw.
    Uses a sliding window over GT to handle insertions/deletions cleanly.
    """
    if not raw_text or not gt_text:
        return []
    raw_set = {raw_text[i:i+n] for n in (2, 3, 4, 5)
               for i in range(len(raw_text) - n + 1)
               if all(_CJK_START <= ch <= _CJK_END for ch in raw_text[i:i+n])}
    terms: list[str] = []
    for n in (2, 3, 4, 5):
        for i in range(len(gt_text) - n + 1):
            sub = gt_text[i:i+n]
            if not all(_CJK_START <= ch <= _CJK_END for ch in sub):
                continue
            if sub not in raw_set:
                terms.append(sub)
    return terms


def _char_edit_distance(a: str, b: str) -> int:
    """Simple character-level edit distance for short strings."""
    if abs(len(a) - len(b)) > 3:
        return abs(len(a) - len(b)) + 3
    dist = 0
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            dist += 1
    dist += abs(len(a) - len(b))
    return dist


def _is_non_terminology_diff(raw_text: str, gt_text: str) -> bool:
    """Check if the GT diff is NOT a terminology correction target.

    Returns True for: empty diff, no CJK terms extracted.
    Single-char edits are flagged as non-terminology only when no
    multi-char CJK difference is detected (avoids misclassifying
    弗丽娜→芙宁娜 as non-terminology).
    """
    if raw_text == gt_text:
        return True
    terms = _extract_gt_terms(raw_text, gt_text)
    if not terms:
        return True  # no CJK terms detected → not a terminology change
    # Single-char edit with no multi-char term → likely typo, not terminology
    if _char_edit_distance(raw_text, gt_text) <= 1 and not any(len(t) >= 3 for t in terms):
        return True
    return False


def _gt_in_candidates(gt_terms: list[str], candidates: list[dict]) -> bool:
    """Check if any GT term appears as a candidate value."""
    cand_values = {c.get("value", "") for c in candidates}
    return any(t in cand_values for t in gt_terms)


def classify_fn(
    cue_index: int,
    raw_text: str,
    repaired_text: str,
    gt_text: str,
    candidates: list[dict],
    phase2_decisions: list,
    requery_hit: bool = False,
    was_modified: bool = False,
) -> FNAttribution:
    """Classify a FN (false negative) into one of 7 buckets."""
    gt_terms = _extract_gt_terms(raw_text, gt_text)
    gt_str = ", ".join(gt_terms) if gt_terms else ""
    cand_vals = [c.get("value", "") for c in candidates]
    decision_actions = [getattr(d, "action", "?") for d in phase2_decisions]
    action_str = ", ".join(str(a) for a in decision_actions)
    sample = f"raw='{raw_text[:40]}' repaired='{repaired_text[:40]}' gt='{gt_text[:40]}'"

    common = dict(cue_index=cue_index, gt_term=gt_str, candidate_values=cand_vals,
                  decision_action=action_str, sample=sample,
                  raw_text=raw_text, repaired_text=repaired_text, gt_text=gt_text)

    if _is_non_terminology_diff(raw_text, gt_text):
        return FNAttribution(bucket=FNBucket.NON_TERMINOLOGY_DIFF, **common)

    if was_modified:
        return FNAttribution(bucket=FNBucket.PARTIAL_FIX, **common)

    if requery_hit:
        return FNAttribution(bucket=FNBucket.REQUERY_FAILED, **common)

    if candidates and _gt_in_candidates(gt_terms, candidates):
        return FNAttribution(bucket=FNBucket.PHASE2_REJECTED, **common)

    if candidates and not _gt_in_candidates(gt_terms, candidates):
        return FNAttribution(bucket=FNBucket.WRONG_CANDIDATE_ONLY, **common)

    if not candidates:
        return FNAttribution(bucket=FNBucket.NO_CANDIDATE, **common)

    return FNAttribution(bucket=FNBucket.UNCLASSIFIED, **common)


def build_attribution_report(
    fn_cues: list,
    all_candidates_by_cue: dict[int, list[dict]],
    all_decisions_by_cue: dict[int, list],
    requery_cue_indices: set[int],
) -> FNAttributionReport:
    """Build a full FN attribution report from evaluation data."""
    report = FNAttributionReport()
    report.bucket_counts = {b: 0 for b in FNBucket.ALL}
    report.terms_by_bucket = {b: Counter() for b in FNBucket.ALL}

    for cue in fn_cues:
        ci = cue.cue_index
        cands = all_candidates_by_cue.get(ci, [])
        decisions = all_decisions_by_cue.get(ci, [])
        requery = ci in requery_cue_indices

        attr = classify_fn(
            ci, cue.raw_text, cue.repaired_text, cue.gt_text,
            cands, decisions, requery, cue.was_modified,
        )
        report.attributions.append(attr)
        report.bucket_counts[attr.bucket] += 1
        for t in _extract_gt_terms(cue.raw_text, cue.gt_text):
            report.terms_by_bucket[attr.bucket][t] += 1

    return report
