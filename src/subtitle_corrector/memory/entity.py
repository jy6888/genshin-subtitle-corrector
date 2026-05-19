from __future__ import annotations

from types import MappingProxyType

from loguru import logger


class EntityMemoryManager:
    """Tracks active entities with time-decaying weights.

    Each time an entity is mentioned (activated), its weight resets to 1.0.
    After each cue, all weights decay by ``decay_rate``. Entities whose weight
    falls below ``min_weight`` are pruned.

    Refresh-count-based adaptive decay: entities that are repeatedly mentioned
    (refreshed while already active) get progressively slower decay.  After 5
    refreshes the entity is locked and never decays — it is considered a core
    topic of the current video.
    """

    def __init__(self, decay_rate: float = 0.8, min_weight: float = 0.2) -> None:
        self._entities: dict[str, float] = {}
        self._refresh_counts: dict[str, int] = {}
        self.decay_rate = decay_rate
        self.min_weight = min_weight

    def update_entity(self, entity: str) -> None:
        """Set *entity* weight to 1.0 (fresh activation)."""
        if not entity:
            return
        current = self._entities.get(entity, 0.0)
        if current < 1.0:
            if current == 0.0:
                logger.info("[Memory] 激活母实体 '{}'", entity)
                self._refresh_counts[entity] = 0
            else:
                self._refresh_counts[entity] = self._refresh_counts.get(entity, 0) + 1
                logger.info(
                    "[Memory] 刷新母实体 '{}' (第{}次)", entity, self._refresh_counts[entity]
                )
            self._entities[entity] = 1.0

    def _effective_decay_rate(self, entity: str) -> float:
        """Refresh count越高 → 衰减越慢 → 5次后锁定."""
        n = self._refresh_counts.get(entity, 0)
        if n >= 5:
            return 1.0
        return min(0.98, self.decay_rate + n * 0.04)

    def decay(self) -> None:
        """Multiply all weights by their effective decay rate and prune stale entries."""
        removed: list[str] = []
        for entity, weight in list(self._entities.items()):
            rate = self._effective_decay_rate(entity)
            new_weight = weight * rate
            if new_weight < self.min_weight:
                removed.append(entity)
            else:
                self._entities[entity] = new_weight
        for entity in removed:
            del self._entities[entity]
            self._refresh_counts.pop(entity, None)
            logger.debug("[Memory] 遗忘实体 '{}'", entity)

    def get_weight(self, entity: str) -> float:
        """Return current weight for *entity*, or 0.0 if unknown."""
        return self._entities.get(entity, 0.0)

    def create_readonly_snapshot(self) -> MappingProxyType[str, float]:
        """Return a frozen, read-only view of the current entity memory.

        Any attempt to mutate the returned mapping raises ``TypeError`` at
        runtime, preventing accidental cross-chunk contamination during
        concurrent processing.
        """
        return MappingProxyType(dict(self._entities))

    @property
    def active_entities(self) -> dict[str, float]:
        """Return a snapshot of all tracked entities and their weights."""
        return dict(self._entities)
