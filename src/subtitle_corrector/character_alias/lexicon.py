"""角色别名词典。

从人工审核的 CSV 加载角色别名，按长度区分：
- len >= 2: character_nickname（多字外号，可作 ASR 修复目标）
- len == 1: team_comp_slot_alias（单字配队简称，仅在配队语境启用）
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class CharacterAliasEntry:
    alias_surface: str
    canonical_term: str
    alias_kind: str
    confidence: float
    risk_flags: list[str] = field(default_factory=list)
    evidence_texts: str = ""
    review_status: str = ""


class CharacterAliasLexicon:
    """角色别名词典。

    Attrs:
        nicknames: {alias_surface -> CharacterAliasEntry}  多字外号 (len>=2)
        team_slots: {alias_surface -> CharacterAliasEntry}  单字配队简称 (len==1)
        all_aliases: {alias_surface -> CharacterAliasEntry}  全部别名
    """

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.nicknames: dict[str, CharacterAliasEntry] = {}
        self.team_slots: dict[str, CharacterAliasEntry] = {}
        self.all_aliases: dict[str, CharacterAliasEntry] = {}
        if csv_path:
            self.load(csv_path)

    def load(self, csv_path: str | Path) -> None:
        """从 CSV 加载角色别名。"""
        count_nick = 0
        count_slot = 0
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                surface = (row.get("alias_surface") or "").strip()
                canonical = (row.get("canonical_term") or "").strip()
                if not surface or not canonical:
                    continue

                entry = CharacterAliasEntry(
                    alias_surface=surface,
                    canonical_term=canonical,
                    alias_kind=(row.get("alias_kind") or "").strip(),
                    confidence=float(row.get("confidence") or 0.5),
                    risk_flags=[f.strip() for f in (row.get("risk_flags") or "").split("|") if f.strip()],
                    evidence_texts=(row.get("evidence_texts") or "").strip(),
                    review_status=(row.get("review_status") or "").strip(),
                )

                self.all_aliases[surface] = entry
                if len(surface) >= 2:
                    self.nicknames[surface] = entry
                    count_nick += 1
                else:
                    self.team_slots[surface] = entry
                    count_slot += 1

        logger.info("CharacterAliasLexicon: {} nicknames, {} team slots (from {})",
                    count_nick, count_slot, csv_path)

    def is_nickname(self, surface: str) -> bool:
        return surface in self.nicknames

    def is_team_slot(self, surface: str) -> bool:
        return surface in self.team_slots

    def get_entry(self, surface: str) -> CharacterAliasEntry | None:
        return self.all_aliases.get(surface)

    def get_canonical(self, surface: str) -> str | None:
        e = self.all_aliases.get(surface)
        return e.canonical_term if e else None

    def find_nicknames_in_text(self, text: str) -> list[CharacterAliasEntry]:
        """查找文本中出现的所有多字外号（按长度降序，避免短别名抢占）。"""
        found: list[CharacterAliasEntry] = []
        # 按长度降序匹配，优先匹配更长的外号
        for alias in sorted(self.nicknames, key=len, reverse=True):
            if alias in text:
                found.append(self.nicknames[alias])
        return found

    def __len__(self) -> int:
        return len(self.all_aliases)

    def __contains__(self, surface: str) -> bool:
        return surface in self.all_aliases
