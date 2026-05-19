"""别名数据三层清洗器。

Tier 1: 规则自动通过 — primary + 非歧义 + 非通用词
Tier 2: 规则自动降级 — secondary 保持 exact_context_only
Tier 3: agent 只审高风险子集 — 自动应用安全决策
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from loguru import logger

# 二字通用中文词
_COMMON_2CHAR = {
    "文字", "黄金", "原木", "森林", "暗影", "黄金", "乐园",
    "深林", "花海", "水仙", "冰风", "辰砂", "余响", "饰金",
    "追忆", "乐团", "角斗", "染血", "奇迹", "守护", "勇士",
    "龙骨", "铁蜂", "冷刃", "弹弓", "银剑", "沙中", "纯水",
    "烈阳", "海渊", "静谧", "公义", "泡沫", "梦想",
    "假面", "面具", "旅人", "行者", "学者", "战士",
    "猎人", "射手", "剑士", "枪兵", "记录", "传说",
    "神话", "历史", "未来", "精华", "碎片",
}

# 通用模式 — 即使 corpus 中出现也不应作为别名
_GENERIC_PATTERNS = {
    "套", "花", "羽", "沙", "杯", "冠",
    "弓", "剑", "刀", "枪", "斧", "锤", "杖",
    "大剑", "长枪", "单手", "双手", "法器",
    "刺", "盾", "矛",
}

# 常见后半段词（artifact_piece 的通用尾缀）—— 不应作为 primary
_WEAK_SUFFIX = {
    "之花", "之羽", "之沙", "之杯", "之冠",
    "之花", "之羽", "之沙", "之杯", "之冠",
    "之花", "之翎", "之沙", "之杯", "之冠",
}


def clean_aliases(
    input_path: str | Path,
    output_path: str | Path,
    review_path: str | Path | None = None,
) -> dict:
    with open(input_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    logger.info("输入 {} 条", len(rows))

    canonical_primaries: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["alias_kind"] == "term_short_primary":
            canonical_primaries[r["canonical_term"]].append(r)

    tier1: list[dict] = []
    tier2: list[dict] = []
    tier3: list[dict] = []  # 真正需要人工看的

    for r in rows:
        alias = r["alias_surface"]
        kind = r["alias_kind"]
        risk = r.get("risk_flags") or ""
        canonical = r["canonical_term"]
        is_ambiguous = "ambiguous" in risk
        is_secondary = "secondary" in kind
        is_common = alias in _COMMON_2CHAR or alias in _GENERIC_PATTERNS
        primaries_for_canonical = canonical_primaries.get(canonical, [])
        multi_primary = len(primaries_for_canonical) > 1

        # ── 需要降级的情况 ──
        should_demote = (
            is_ambiguous          # 歧义简称 → 不能做 primary
            or is_common          # 通用词 → 不能做 primary
            or multi_primary      # 多 primary → 只保留一个
        )

        # ── multi_primary: 保留最短的那个为 primary ──
        if multi_primary and not is_ambiguous and not is_common:
            shortest = min(p["alias_surface"] for p in primaries_for_canonical if p is not None)
            if alias == shortest and kind == "term_short_primary":
                should_demote = False  # 保留最短的 primary

        if is_secondary:
            r["usage_policy"] = "exact_context_only"
            tier2.append(r)
        elif should_demote:
            r["alias_kind"] = "term_short_secondary"
            r["usage_policy"] = "exact_context_only"
            r["risk_flags"] = _append_flag(risk, "needs_review")
            # 只有 ambiguous + 非通用词的才进人工审核
            if is_ambiguous and not is_common:
                tier3.append(r)
            else:
                tier2.append(r)  # 自动降级，无需人工
        else:
            tier1.append(r)

    all_output = tier1 + tier2 + tier3
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_output:
            writer.writerow(r)

    logger.info(
        "清洗完成: tier1={}, tier2={}, tier3_review={}, total={}",
        len(tier1), len(tier2), len(tier3), len(all_output),
    )

    if review_path and tier3:
        _write_review_list(tier3, review_path)

    return {
        "tier1_kept": len(tier1),
        "tier2_demoted": len(tier2),
        "tier3_review": len(tier3),
        "total_output": len(all_output),
    }


def _append_flag(flags: str, new_flag: str) -> str:
    existing = {f for f in flags.split("|") if f}
    existing.add(new_flag)
    return "|".join(sorted(existing))


def _write_review_list(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alias_surface", "canonical_term", "reason", "suggestion"])
        for r in rows:
            writer.writerow([
                r["alias_surface"],
                r["canonical_term"],
                r.get("risk_flags", ""),
                "手动确认: 保留(改primary) / 降级(维持secondary) / 删除",
            ])
