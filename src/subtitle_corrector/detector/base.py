from __future__ import annotations

from abc import ABC, abstractmethod

from subtitle_corrector.schemas import DetectionResult, SubtitleCue, SubtitleDocument


class Detector(ABC):
    name: str

    @abstractmethod
    def detect(self, cue: SubtitleCue, document: SubtitleDocument) -> DetectionResult:
        raise NotImplementedError


class DetectorPlugin(Detector):
    """Extension point for project-specific detectors."""
