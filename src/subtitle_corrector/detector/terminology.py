from __future__ import annotations

from subtitle_corrector.detector.base import Detector
from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher
from subtitle_corrector.schemas import DetectionResult, SubtitleCue, SubtitleDocument


class TerminologyAnomalyDetector(Detector):
    name = "terminology"

    def __init__(self, matcher: FuzzyTerminologyMatcher) -> None:
        self.matcher = matcher

    def detect(self, cue: SubtitleCue, document: SubtitleDocument) -> DetectionResult:
        candidates = self.matcher.lookup(cue.text)
        if not candidates:
            return DetectionResult(
                detector=self.name,
                cue_index=cue.index,
                risk_score=0.0,
                reason="no terminology candidate",
            )
        risk = max(candidate.score for candidate in candidates)
        return DetectionResult(
            detector=self.name,
            cue_index=cue.index,
            risk_score=risk,
            reason="text is close to known terminology but not confirmed",
            candidates=candidates,
        )
