from __future__ import annotations

from abc import ABC, abstractmethod

from subtitle_corrector.schemas import RepairResult


class CorrectionHistoryStore(ABC):
    @abstractmethod
    def record(self, repair: RepairResult, source_file: str | None = None) -> None:
        raise NotImplementedError
