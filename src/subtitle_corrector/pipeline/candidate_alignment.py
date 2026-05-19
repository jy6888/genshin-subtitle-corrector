"""Candidate alignment: evidence-based surface validation replacing name-list filters.

Ordered syllable alignment ensures that a surface can only be accepted as a
candidate when its pinyin syllables can be matched in order against the target
term's syllables.  This replaces set-based coverage (which couldn't distinguish
"那位打一个" from "那维莱特") and removes all name-list filtering.
"""

from __future__ import annotations

from pypinyin import lazy_pinyin
from rapidfuzz import fuzz


def build_aligned_candidate(
    text: str,
    surface_start: int,
    surface_end: int,
    target: str,
    source: str,
    category: str,
    parent_entity: str,
    allow_prefix: bool = False,
    expansion_policy: str = "unknown",
) -> dict | None:
    """Validate a candidate span through ordered syllable alignment.

    Returns a candidate dict with match metadata, or None if the
    alignment evidence is too weak for the span to be safe.
    """
    if surface_start >= surface_end or surface_start < 0:
        return None
    if surface_end > len(text):
        return None

    surface = text[surface_start:surface_end]
    if not surface:
        return None

    # Structural: surface must be CJK-only (no latin/digits)
    if any(ch.isascii() and (ch.isdigit() or ch.isalpha()) for ch in surface):
        return None

    if not target or not target.strip():
        return None

    # 字数方向门：针对 language_model 来源的候选，防止反向扩写。
    # 金珀(2) → 试作金珀(4) 拒绝；金破(2) → 金珀(2) 放行。
    # 不影响实体前缀扩展（long_entity_variant / pinyin 等来源）。
    if len(target) > len(surface):
        if expansion_policy == "preserve_surface":
            return None
        if source == "language_model" and expansion_policy == "unknown":
            return None

    # Phonetic + character similarity
    surface_py_tokens = _pinyin_tokens(surface)
    target_py_tokens = _pinyin_tokens(target)
    phonetic_ratio = fuzz.ratio(" ".join(surface_py_tokens), " ".join(target_py_tokens)) / 100.0
    char_ratio = fuzz.ratio(surface, target) / 100.0
    alignment_score = max(phonetic_ratio, char_ratio * 0.85)
    # Prefix detection before alignment: if surface is a phonetic prefix,
    # boost the alignment score (the ordered alignment will confirm/deny)
    is_prefix = (
        len(surface_py_tokens) <= len(target_py_tokens)
        and all(
            _syllable_similar(surface_py_tokens[i], target_py_tokens[i])
            for i in range(len(surface_py_tokens))
        )
    )
    if is_prefix and allow_prefix:
        # Prefix alignment is high-quality evidence — surface fully
        # consumed by the start of the target
        alignment_score = max(alignment_score, 0.82)

    # Ordered syllable alignment
    aligned = _ordered_syllable_align(surface_py_tokens, target_py_tokens)
    surface_cov = aligned["surface_coverage"]
    target_cov = aligned["target_coverage"]
    extra_surface_ratio = aligned["extra_surface_ratio"]
    prefix_aligned = aligned["prefix_aligned"]

    # Determine match kind
    if surface == target:
        match_kind = "exact"
    elif surface_py_tokens == target_py_tokens and len(surface) == len(target):
        match_kind = "phonetic_exact"
    elif prefix_aligned and allow_prefix:
        match_kind = "phonetic_prefix"
    else:
        match_kind = "phonetic_partial"

    # Rejection rules based on match kind
    if match_kind == "exact":
        pass  # always accept
    elif match_kind == "phonetic_exact":
        if surface_cov < 0.80 or target_cov < 0.75:
            return None
    elif match_kind == "phonetic_prefix":
        if not allow_prefix:
            return None
        cjk_len = sum(1 for ch in surface if "一" <= ch <= "鿿")
        if cjk_len < 3:
            return None
        if surface_cov < 0.90:
            return None
        if target_cov < 0.50:
            return None
    elif match_kind == "phonetic_partial":
        if alignment_score < 0.73:
            return None
        if surface_cov < 0.40 or target_cov < 0.30:
            return None
        if extra_surface_ratio > 0.40:
            return None

    # Short surface guard
    cjk_len = sum(1 for ch in surface if "一" <= ch <= "鿿")
    if cjk_len < 2:
        return None

    return {
        "surface": surface,
        "value": target,
        "score": round(alignment_score, 3),  # unified score for downstream
        "start_char": surface_start,
        "end_char": surface_end,
        "source": source,
        "category": category,
        "parent_entity": parent_entity,
        "match_kind": match_kind,
        "alignment_score": round(alignment_score, 3),
        "surface_coverage": round(surface_cov, 3),
        "target_coverage": round(target_cov, 3),
        "extra_surface_ratio": round(extra_surface_ratio, 3),
        "expansion_policy": expansion_policy,
        "evidence_type": source,
    }


def _pinyin_tokens(text: str) -> list[str]:
    return lazy_pinyin(text, errors="ignore")


def _ordered_syllable_align(surface_tokens: list[str], target_tokens: list[str]) -> dict:
    """Greedy ordered syllable alignment with fuzzy matching.

    Each surface syllable can match a target syllable if their
    individual edit ratio >= 0.65 (catches li→ning, lin→ning).
    Matching is ordered (monotonic) but allows skipping in target.
    """
    if not surface_tokens or not target_tokens:
        return {
            "surface_coverage": 0.0, "target_coverage": 0.0,
            "extra_surface_ratio": 1.0, "prefix_aligned": False,
        }

    # Greedy ordered alignment with fuzzy syllable match
    ti = 0
    matched_surface = 0
    matched_target = 0
    for stok in surface_tokens:
        found = False
        for j in range(ti, len(target_tokens)):
            if _syllable_similar(stok, target_tokens[j]):
                matched_surface += 1
                matched_target += 1
                ti = j + 1
                found = True
                break
        if not found:
            # Try fuzzy best-match in remaining tokens
            best_j = -1
            best_r = 0.0
            for j in range(ti, len(target_tokens)):
                r = fuzz.ratio(stok, target_tokens[j]) / 100.0
                if r > best_r and r >= 0.55:
                    best_r = r
                    best_j = j
            if best_j >= 0:
                matched_surface += 1
                ti = best_j + 1

    surface_cov = matched_surface / len(surface_tokens)
    target_cov = matched_target / len(target_tokens) if target_tokens else 0.0
    extra_surface_ratio = 1.0 - surface_cov

    # Prefix: all surface syllables find a match at the start of target
    prefix_aligned = (
        len(surface_tokens) <= len(target_tokens)
        and all(
            _syllable_similar(surface_tokens[i], target_tokens[i])
            for i in range(len(surface_tokens))
        )
    )

    return {
        "surface_coverage": surface_cov,
        "target_coverage": target_cov,
        "extra_surface_ratio": extra_surface_ratio,
        "prefix_aligned": prefix_aligned,
    }


def _syllable_similar(a: str, b: str) -> bool:
    """Two pinyin syllables are similar enough for alignment."""
    return fuzz.ratio(a, b) >= 65
