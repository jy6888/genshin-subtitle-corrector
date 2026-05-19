from __future__ import annotations

from subtitle_corrector.detector.base import Detector
from subtitle_corrector.schemas import DetectionResult, SubtitleCue, SubtitleDocument


class EntityConsistencyDetector(Detector):
    name = "entity_consistency"

    def detect(self, cue: SubtitleCue, document: SubtitleDocument) -> DetectionResult:
        return DetectionResult(
            detector=self.name,
            cue_index=cue.index,
            risk_score=0.0,
            reason="entity consistency model is not configured",
        )
