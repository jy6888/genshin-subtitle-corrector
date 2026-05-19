"""Evaluation metrics for subtitle correction quality assessment.

Compares pipeline output against ground truth at cue-level and
character-level granularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CueResult:
    """Per-cue evaluation detail."""
    cue_index: int
    raw_text: str
    repaired_text: str
    gt_text: str
    has_error: bool          # GT != raw
    was_modified: bool       # repaired != raw
    is_correct: bool         # repaired == GT
    is_wrong_fix: bool       # modified but wrong
    is_missed: bool          # had error but not fixed


@dataclass
class EvaluationReport:
    """Full evaluation report with metrics and per-cue details."""

    source_file: str
    gt_file: str
    total_cues: int
    error_cues: int          # cues where GT != raw

    # Confusion matrix
    tp: int = 0       # fixed correctly
    fp: int = 0       # incorrectly modified a correct cue
    tn: int = 0       # correctly left alone
    fn: int = 0       # missed correction

    # Per-cue
    wrong_fixes: list[CueResult] = field(default_factory=list)
    missed_errors: list[CueResult] = field(default_factory=list)
    correct_fixes: list[CueResult] = field(default_factory=list)

    # 非术语级差异（代词/语气词/错字），不计入主分数
    chat_diffs: list[CueResult] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total_cues if self.total_cues else 0.0

    @property
    def fix_rate(self) -> float:
        return self.tp / self.error_cues if self.error_cues else 0.0

    @property
    def false_positive_rate(self) -> float:
        """What fraction of modifications were wrong?"""
        denom = self.tp + self.fp
        return self.fp / denom if denom else 0.0


@dataclass
class CharLevelMetrics:
    """Character-level edit distance evaluation."""
    total_raw_chars: int = 0
    total_gt_chars: int = 0
    chars_fixed: int = 0      # chars in wrong cues that were fixed
    chars_missed: int = 0     # chars in wrong cues that were missed
    chars_broken: int = 0     # chars in correct cues that were wrongly changed

    @property
    def char_accuracy(self) -> float:
        correct = self.total_raw_chars - self.chars_broken - self.chars_missed + self.chars_fixed
        return max(0.0, correct / self.total_raw_chars) if self.total_raw_chars else 0.0


def _normalize_for_comparison(text: str) -> str:
    """归一化文本用于比较：去空格、阿拉伯数字转中文数字。"""
    import re
    t = text
    # 去所有空格
    t = re.sub(r"\s+", "", t)
    # 中文数字一对一映射
    digit_map = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
                 "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}
    for d, c in digit_map.items():
        t = t.replace(d, c)
    return t


def evaluate(
    raw_cues: list,
    repaired_cues: list,
    gt_cues: list,
    skip_fn: object | None = None,
) -> EvaluationReport:
    """Compare repaired output against ground truth.

    If skip_fn(raw_text, gt_text) returns True, the cue is treated as a
    chat-level diff (pronouns, particles, etc.) — counted as TN in the
    main score but tracked in report.chat_diffs for manual review.
    """
    report = EvaluationReport(source_file="", gt_file="", total_cues=len(raw_cues), error_cues=0)

    for r, p, g in zip(raw_cues, repaired_cues, gt_cues):
        rn = _normalize_for_comparison(r.text)
        pn = _normalize_for_comparison(p.text)
        gn = _normalize_for_comparison(g.text)
        has_error = rn != gn
        was_modified = pn != rn
        is_correct = pn == gn

        detail = CueResult(
            cue_index=r.index,
            raw_text=r.text,
            repaired_text=p.text,
            gt_text=g.text,
            has_error=has_error,
            was_modified=was_modified,
            is_correct=is_correct,
            is_wrong_fix=was_modified and not is_correct,
            is_missed=has_error and not was_modified,
        )

        if has_error and skip_fn is not None and skip_fn(r.text, g.text):
            report.chat_diffs.append(detail)
            report.tn += 1  # 视为正确，不计入错误
            continue

        if has_error:
            report.error_cues += 1

        if has_error and is_correct:
            report.tp += 1
            report.correct_fixes.append(detail)
        elif has_error and not is_correct and was_modified:
            report.fn += 1
            report.wrong_fixes.append(detail)
        elif has_error and not was_modified:
            report.fn += 1
            report.missed_errors.append(detail)
        elif not has_error and was_modified:
            report.fp += 1
            report.wrong_fixes.append(detail)
        else:
            report.tn += 1

    return report
