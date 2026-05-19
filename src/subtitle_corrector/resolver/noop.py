from __future__ import annotations

from subtitle_corrector.resolver.base import LLMArbitrator
from subtitle_corrector.schemas import ArbitrationDecision, ArbitrationRequest, CorrectionAction


class NoopArbitrator(LLMArbitrator):
    def arbitrate(self, request: ArbitrationRequest) -> ArbitrationDecision:
        return ArbitrationDecision(
            action=CorrectionAction.NEEDS_REVIEW,
            selected_candidate=None,
            corrections=[],
            confidence=0.0,
            reasoning="no LLM arbitrator configured",
        )
