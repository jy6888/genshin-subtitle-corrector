"""口语简称词典。

运行时加载审核后的口语简称，提供查询接口。
"""

from __future__ import annotations

import csv
from pathlib import Path

from loguru import logger


class SpokenAliasLexicon:
    """口语简称词典。

    从审核后的 CSV 加载已知简称，运行时只做实体识别和上下文激活，
    不直接把简称替换成标准名。

    Attrs:
        surface_to_canonical: 简称 → 标准词
        alias_kinds: 简称 → 类型标记
        usage_policies: 简称 → 使用策略
    """

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.surface_to_canonical: dict[str, str] = {}
        self.alias_kinds: dict[str, str] = {}
        self.usage_policies: dict[str, str] = {}

        if csv_path:
            self.load(csv_path)

    def load(self, csv_path: str | Path) -> None:
        """从审核后 CSV 加载简称。"""
        count = 0
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                surface = (row.get("alias_surface") or "").strip()
                canonical = (row.get("canonical_term") or "").strip()
                policy = (row.get("usage_policy") or "").strip()

                if not surface or not canonical:
                    continue
                if policy == "blocked":
                    continue

                self.surface_to_canonical[surface] = canonical
                self.alias_kinds[surface] = (row.get("alias_kind") or "").strip()
                self.usage_policies[surface] = policy
                count += 1

        logger.info("SpokenAliasLexicon 加载 {} 条简称", count)

    def is_known(self, surface: str) -> bool:
        return surface in self.surface_to_canonical

    def get_canonical(self, surface: str) -> str | None:
        return self.surface_to_canonical.get(surface)

    def get_kind(self, surface: str) -> str:
        return self.alias_kinds.get(surface, "")

    def get_policy(self, surface: str) -> str:
        """获取使用策略: context_only / needs_context"""
        return self.usage_policies.get(surface, "context_only")

    def should_activate_entity(self, surface: str) -> bool:
        """简称出现时是否应激活实体上下文。"""
        return self.is_known(surface) and self.get_policy(surface) != "blocked"

    def find_surfaces_in_text(self, text: str) -> list[dict]:
        """在文本中查找所有已知简称。

        Returns:
            [{surface, canonical, kind, policy, start_char, end_char}]
        """
        results: list[dict] = []
        for surface, canonical in self.surface_to_canonical.items():
            idx = text.find(surface)
            while idx >= 0:
                results.append({
                    "surface": surface,
                    "canonical": canonical,
                    "kind": self.alias_kinds.get(surface, ""),
                    "policy": self.usage_policies.get(surface, "context_only"),
                    "start_char": idx,
                    "end_char": idx + len(surface),
                })
                idx = text.find(surface, idx + 1)
        return results

    def __len__(self) -> int:
        return len(self.surface_to_canonical)

    def __contains__(self, surface: str) -> bool:
        return surface in self.surface_to_canonical
