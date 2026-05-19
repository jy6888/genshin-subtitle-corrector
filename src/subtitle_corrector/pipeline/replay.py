"""Deterministic replay: rebuild CommittedWorldState from EventLog at any point.

Given the same EventLog, replay() MUST produce byte-identical CommittedWorldState.
This is critical for debugging Reducer drift and for regression testing.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from subtitle_corrector.events import EventLogger
    from subtitle_corrector.pipeline.reducer import CommittedWorldState


class WorldStateReplay:
    """Deterministic replay engine.

    Usage::

        replay = WorldStateReplay()
        state_45 = replay.replay(event_log, target_index=45)
        state_60 = replay.replay(event_log, target_index=60)
        diff = replay.diff(state_45, state_60)
    """

    @staticmethod
    def replay(
        event_log: EventLogger,
        target_index: int,
    ) -> CommittedWorldState:
        """Rebuild CommittedWorldState as it was at *target_index*."""
        from subtitle_corrector.events import EventLogger as _EventLogger
        from subtitle_corrector.pipeline.reducer import ConsensusReducer

        events = event_log.get_events_in_range(0, target_index)
        # Create a temporary EventLogger with only the relevant events
        temp_log = _EventLogger()
        temp_log.append_events(list(events))

        reducer = ConsensusReducer()
        return reducer.reduce(temp_log)

    @staticmethod
    def diff(
        state_a: CommittedWorldState,
        state_b: CommittedWorldState,
    ) -> dict:
        """Compare two WorldStates — identify what changed and why."""
        diff_result: dict = {
            "entities_added": [],
            "entities_removed": [],
            "entities_confidence_delta": [],
            "topics_added": len(state_b.topic_graph.nodes) - len(state_a.topic_graph.nodes),
            "repairs_changed": 0,
            "hash_a": WorldStateReplay._hash_state(state_a),
            "hash_b": WorldStateReplay._hash_state(state_b),
            "identical": False,
        }

        ents_a = set(state_a.entities.keys())
        ents_b = set(state_b.entities.keys())
        diff_result["entities_added"] = list(ents_b - ents_a)
        diff_result["entities_removed"] = list(ents_a - ents_b)

        for e in ents_a & ents_b:
            delta = state_b.entities[e].confidence - state_a.entities[e].confidence
            if abs(delta) > 0.01:
                diff_result["entities_confidence_delta"].append({
                    "entity": e,
                    "from": state_a.entities[e].confidence,
                    "to": state_b.entities[e].confidence,
                })

        # Count repair verdict changes
        decisions_a = {d.hypothesis.event_id: d.verdict for d in state_a.repair_decisions}
        decisions_b = {d.hypothesis.event_id: d.verdict for d in state_b.repair_decisions}
        for eid in set(decisions_a) | set(decisions_b):
            if decisions_a.get(eid) != decisions_b.get(eid):
                diff_result["repairs_changed"] += 1

        diff_result["identical"] = diff_result["hash_a"] == diff_result["hash_b"]
        return diff_result

    @staticmethod
    def verify_determinism(event_log: EventLogger, target_index: int) -> bool:
        """Run replay twice and confirm byte-identical results."""
        state_1 = WorldStateReplay.replay(event_log, target_index)
        state_2 = WorldStateReplay.replay(event_log, target_index)
        h1 = state_1.to_deterministic_snapshot()
        h2 = state_2.to_deterministic_snapshot()
        identical = h1 == h2
        if not identical:
            logger.error("DETERMINISM VIOLATION: replay produced different results!")
        return identical

    @staticmethod
    def _hash_state(state: CommittedWorldState) -> str:
        return hashlib.sha256(state.to_deterministic_snapshot()).hexdigest()[:16]
