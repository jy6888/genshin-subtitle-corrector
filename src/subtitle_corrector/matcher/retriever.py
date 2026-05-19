from __future__ import annotations

from abc import ABC, abstractmethod

from subtitle_corrector.schemas import Candidate, SubtitleCue, SubtitleDocument


class CandidateRetriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, cue: SubtitleCue, document: SubtitleDocument) -> list[Candidate]:
        raise NotImplementedError


class CompositeCandidateRetriever(CandidateRetriever):
    name = "composite"

    def __init__(self, retrievers: list[CandidateRetriever], max_candidates: int = 8) -> None:
        self.retrievers = retrievers
        self.max_candidates = max_candidates

    def retrieve(self, cue: SubtitleCue, document: SubtitleDocument) -> list[Candidate]:
        merged: dict[str, Candidate] = {}
        for retriever in self.retrievers:
            for candidate in retriever.retrieve(cue, document):
                existing = merged.get(candidate.value)
                if existing is None or candidate.score > existing.score:
                    merged[candidate.value] = candidate
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[
            : self.max_candidates
        ]


class TerminologyCandidateRetriever(CandidateRetriever):
    name = "terminology"

    def __init__(self, matcher, max_candidates: int = 8) -> None:
        self.matcher = matcher
        self.max_candidates = max_candidates

    def retrieve(self, cue: SubtitleCue, document: SubtitleDocument) -> list[Candidate]:
        return self.matcher.lookup(cue.text, limit=self.max_candidates)
