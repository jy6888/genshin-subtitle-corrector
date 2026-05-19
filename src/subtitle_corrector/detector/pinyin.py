from __future__ import annotations

from subtitle_corrector.detector.base import Detector
from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher
from subtitle_corrector.pinyin.converter import FuzzyPinyinMatcher
from subtitle_corrector.schemas import DetectionResult, SubtitleCue, SubtitleDocument


class PinyinAnomalyDetector(Detector):
    name = "pinyin"

    def __init__(
        self,
        terminology_matcher: FuzzyTerminologyMatcher,
        pinyin_matcher: FuzzyPinyinMatcher | None = None,
    ) -> None:
        self.terminology_matcher = terminology_matcher
        self.pinyin_matcher = pinyin_matcher or FuzzyPinyinMatcher()

    def detect(self, cue: SubtitleCue, document: SubtitleDocument) -> DetectionResult:
        candidates = self.terminology_matcher.lookup(cue.text)
        if not candidates:
            return DetectionResult(
                detector=self.name,
                cue_index=cue.index,
                risk_score=0.0,
                reason="no pinyin-like terminology candidate",
            )
        best_risk = 0.0
        for candidate in candidates:
            # Compare candidate term against the best-matching substring
            # of similar length in the cue text, not the entire line.
            sim = self._best_substring_similarity(cue.text, candidate.value)
            risk = min(max(sim * candidate.score, 0.0), 1.0)
            if risk > best_risk:
                best_risk = risk
        return DetectionResult(
            detector=self.name,
            cue_index=cue.index,
            risk_score=best_risk,
            reason="subtitle text is phonetically close to known terminology",
            candidates=candidates,
            metadata={"best_pinyin_risk": best_risk},
        )

    def _best_substring_similarity(self, text: str, term: str) -> float:
        """Find the substring of *text* most phonetically similar to *term*."""
        term_len = len(term)
        if term_len == 0:
            return 0.0
        if len(text) <= term_len:
            return self.pinyin_matcher.similarity(text, term)
        best = 0.0
        for i in range(len(text) - term_len + 1):
            window = text[i : i + term_len]
            sim = self.pinyin_matcher.similarity(window, term)
            if sim > best:
                best = sim
                if best >= 1.0:
                    break
        return best

