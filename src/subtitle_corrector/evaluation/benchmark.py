from __future__ import annotations

from pathlib import Path

from subtitle_corrector.evaluation.metrics import CorrectionMetrics


class BenchmarkRunner:
    def run(self, dataset_path: str | Path) -> CorrectionMetrics:
        raise NotImplementedError(
            "Benchmark format is intentionally not fixed yet. Define gold subtitle pairs first."
        )
