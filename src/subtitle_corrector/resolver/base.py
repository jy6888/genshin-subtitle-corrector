from __future__ import annotations

from abc import ABC, abstractmethod

from subtitle_corrector.schemas import ArbitrationDecision, ArbitrationRequest


class LLMArbitrator(ABC):
    """High-cost constrained candidate arbiter.

    Implementations must never rewrite freely. They may only choose KEEP,
    NEEDS_REVIEW, or one candidate from ArbitrationRequest.candidates.
    """

    @abstractmethod
    def arbitrate(self, request: ArbitrationRequest) -> ArbitrationDecision:
        raise NotImplementedError
