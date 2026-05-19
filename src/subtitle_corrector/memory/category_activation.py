from __future__ import annotations

from types import MappingProxyType

from loguru import logger


class CategoryActivationManager:
    """Tracks which cold-pool terminology categories are currently active.

    Cold categories are dormant by default. When the LLM selects a candidate
    from a cold category, that category is activated (weight = 1.0). After
    each cue, all weights decay. Categories whose weight falls below
    ``min_weight`` are deactivated — returning to dormant/probe-only mode.
    """

    def __init__(self, decay_rate: float = 0.7, min_weight: float = 0.3) -> None:
        self._weights: dict[str, float] = {}
        self.decay_rate = decay_rate
        self.min_weight = min_weight

    def activate(self, category: str) -> None:
        """LLM selected a term from *category* — set weight to 1.0."""
        if not category:
            return
        was_active = self.is_active(category)
        self._weights[category] = 1.0
        if not was_active:
            logger.info("[Category] 激活冷池类别 '{}'", category)

    def is_active(self, category: str) -> bool:
        return self._weights.get(category, 0.0) >= self.min_weight

    def decay(self) -> None:
        """Called after each cue — all activation weights decay."""
        for cat in list(self._weights):
            self._weights[cat] *= self.decay_rate
            if self._weights[cat] < self.min_weight:
                logger.debug("[Category] 冷池类别 '{}' 休眠", cat)
                del self._weights[cat]

    def create_readonly_snapshot(self) -> MappingProxyType[str, float]:
        """Return a frozen, read-only view of the current category weights.

        Any attempt to mutate the returned mapping raises ``TypeError`` at
        runtime, safe for concurrent chunk processing.
        """
        return MappingProxyType(dict(self._weights))

    @property
    def active_categories(self) -> set[str]:
        return {c for c, w in self._weights.items() if w >= self.min_weight}
