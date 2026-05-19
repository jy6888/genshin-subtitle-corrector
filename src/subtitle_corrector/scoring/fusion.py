from __future__ import annotations

from subtitle_corrector.schemas import Candidate, DetectionResult, FusedRisk


class WeightedRiskAggregator:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {}

    def aggregate(self, cue_index: int, results: list[DetectionResult]) -> FusedRisk:
        weighted_sum = 0.0
        total_weight = 0.0
        reasons: list[str] = []
        candidates_by_value: dict[str, Candidate] = {}
        for result in results:
            # Skip unconfigured/stub detectors so they don't dilute the
            # weighted average and make the threshold unreachable.
            if result.risk_score == 0.0 and "not configured" in result.reason:
                continue
            weight = self.weights.get(result.detector, 1.0)
            weighted_sum += result.risk_score * weight
            total_weight += weight
            if result.risk_score > 0:
                reasons.append(f"{result.detector}: {result.reason}")
            for candidate in result.candidates:
                existing = candidates_by_value.get(candidate.value)
                if existing is None or candidate.score > existing.score:
                    candidates_by_value[candidate.value] = candidate
        risk = weighted_sum / total_weight if total_weight else 0.0
        return FusedRisk(
            cue_index=cue_index,
            risk_score=min(max(risk, 0.0), 1.0),
            reasons=reasons,
            candidates=sorted(
                candidates_by_value.values(), key=lambda candidate: candidate.score, reverse=True
            ),
        )
