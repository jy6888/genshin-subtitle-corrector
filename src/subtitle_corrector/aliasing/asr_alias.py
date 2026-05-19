from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


VALID_ALIAS_TYPES = {"asr_alias", "spoken_alias"}
VALID_REVIEW_STATUSES = {"pending", "approved", "rejected", "needs_context"}


@dataclass(frozen=True)
class AsrAliasCandidate:
    alias_surface: str
    canonical_term: str
    alias_type: str
    review_status: str
    confidence: float
    evidence_count: int = 0
    evidence_sources: str = ""
    evidence_examples: str = ""
    risk_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.alias_surface:
            raise ValueError("alias_surface must not be empty")
        if not self.canonical_term:
            raise ValueError("canonical_term must not be empty")
        if self.alias_surface == self.canonical_term:
            raise ValueError(
                "alias_surface and canonical_term must not be the same: "
                f"{self.alias_surface!r}"
            )
        if self.alias_type not in VALID_ALIAS_TYPES:
            raise ValueError(
                f"alias_type must be one of {sorted(VALID_ALIAS_TYPES)}, "
                f"got {self.alias_type!r}"
            )
        if self.review_status not in VALID_REVIEW_STATUSES:
            raise ValueError(
                "review_status must be one of "
                f"{sorted(VALID_REVIEW_STATUSES)}, got {self.review_status!r}"
            )


def read_alias_candidates(path: str | Path) -> list[AsrAliasCandidate]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []

    rows: list[AsrAliasCandidate] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            surface = (row.get("alias_surface") or "").strip()
            canonical = (row.get("canonical_term") or "").strip()
            if not surface or not canonical:
                continue
            flags = tuple(
                flag.strip()
                for flag in (row.get("risk_flags") or "").split("|")
                if flag.strip()
            )
            rows.append(
                AsrAliasCandidate(
                    alias_surface=surface,
                    canonical_term=canonical,
                    alias_type=(row.get("alias_type") or "asr_alias").strip(),
                    review_status=(row.get("review_status") or "pending").strip(),
                    confidence=float(row.get("confidence") or 0.5),
                    evidence_count=int(row.get("evidence_count") or 0),
                    evidence_sources=(row.get("evidence_sources") or "").strip(),
                    evidence_examples=(row.get("evidence_examples") or "").strip(),
                    risk_flags=flags,
                )
            )
    return rows
