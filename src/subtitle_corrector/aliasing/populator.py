"""术语别名填充器。

从 terminology 表读取术语，按类别规则生成内部简称候选，
分 primary / secondary 两级，写入 terminology_alias 表。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from loguru import logger

from subtitle_corrector.memory.sqlite import SQLiteMemory

_TARGET_CATEGORIES = {"artifact", "artifact_piece", "weapon"}

# 武器系列前缀 / 通用虚词前缀 → 自身不构成简称，对应的后半段优先
_SERIES_PREFIXES = {
    "试作", "破魔", "讨龙", "铁蜂", "西风", "祭礼", "宗室", "暗巷",
    "匣里", "黑岩", "雨裁", "千岩", "贯月", "钢轮", "雪葬", "白影",
    "龙脊", "万国", "昭心", "忍冬", "天目", "桂木", "喜多", "竭泽",
    "原木", "森林", "公义", "静谧", "海渊", "沙中", "纯水", "烈阳",
    "决斗", "黑缨", "沐浴", "旅行", "冷刃", "黎明", "弹弓", "魔导",
    "翡玉", "甲级", "白铁", "银剑", "铁影", "飞天",
}

# 子串过滤
_BLOCKED_PREFIXES = {"之", "的", "与", "和", "及", "或", "一", "圣", "遗"}
_BLOCKED_SUFFIXES = {"之", "的", "与", "和", "及", "或", "套", "物"}

# 基础圣遗物（跳过）
_BASIC_ARTIFACT_SETS = {
    "冒险家", "幸运儿", "游医", "学士", "赌徒", "武人", "教官",
    "战狂", "奇迹", "守护", "勇士",
}
_BASIC_ARTIFACT_TERMS = _BASIC_ARTIFACT_SETS | {
    "冒险家金杯", "冒险家之花", "冒险家尾羽", "冒险家头巾", "冒险家怀表",
    "幸运儿银冠", "幸运儿之杯", "幸运儿绿花", "幸运儿鹰羽", "幸运儿沙漏",
    "学士的时钟", "学士的书签", "学士的羽笔", "学士的眼镜", "学士的墨杯",
    "赌徒的胸花", "赌徒的骰盅", "赌徒的羽饰", "赌徒的耳环", "赌徒的怀表",
    "武人的红花", "武人的水漏", "武人的羽饰", "武人的头巾", "武人的酒杯",
    "教官的胸花", "教官的怀表", "教官的羽饰", "教官的帽子", "教官的茶杯",
    "战狂的蔷薇", "战狂的时计", "战狂的翎羽", "战狂的鬼面", "战狂的骨杯",
    "奇迹之花", "奇迹之杯", "奇迹之羽", "奇迹之沙", "奇迹之冠",
    "守护之花", "守护之杯", "守护之羽", "守护之沙", "守护之冠",
    "勇士之花", "勇士之杯", "勇士之羽", "勇士之沙", "勇士之冠",
}


def populate_aliases(
    db_path: str | Path,
    categories: tuple[str, ...] = ("artifact", "artifact_piece", "weapon"),
    min_term_length: int = 4,
    dry_run: bool = True,
) -> dict:
    memory = SQLiteMemory(db_path)
    stats = {"generated": 0, "skipped_duplicate": 0, "primary": 0, "secondary": 0,
             "by_category": defaultdict(int)}

    with memory.connect() as conn:
        placeholders = ",".join("?" for _ in categories)
        rows = conn.execute(
            f"SELECT id, term, category FROM terminology "
            f"WHERE category IN ({placeholders}) AND LENGTH(term) >= ?",
            (*categories, min_term_length),
        ).fetchall()
        logger.info("加载 {} 个目标术语 (>= {} 字)", len(rows), min_term_length)

        existing = set()
        for row in conn.execute("SELECT terminology_id, alias FROM terminology_alias").fetchall():
            existing.add((row["terminology_id"], row["alias"]))

        inserts: list[tuple[int, str, str]] = []  # (tid, alias, alias_type)

        for row in rows:
            tid = row["id"]
            raw_term = row["term"]
            cat = row["category"]

            if _is_basic_artifact(raw_term, cat):
                continue

            term = _clean_term(raw_term)
            if len(term) < 4:
                continue

            candidates = _generate_candidates(term, cat)
            if not candidates:
                continue

            # artifact 套装加「套」
            if cat == "artifact":
                for c in list(candidates):
                    if (c + "套") not in candidates:
                        candidates.append(c + "套")

            # 每术语保留最优 2 个再分类
            candidates = _filter_best(candidates, term)

            # artifact 套装: 只保留「套」版，去掉裸别名
            if cat == "artifact":
                tao_set = {c for c in candidates if c.endswith("套")}
                if tao_set:
                    bare = {c.replace("套", "") for c in tao_set}
                    candidates = [c for c in candidates if c.endswith("套") or c not in bare]

            # 分类 primary / secondary
            classified = _classify(candidates, term, cat)

            for alias, kind in classified:
                if len(alias) < 2:
                    continue
                if (tid, alias) in existing:
                    stats["skipped_duplicate"] += 1
                    continue

                alias_type = f"term_short_{kind}"  # primary / secondary
                inserts.append((tid, alias, alias_type))
                existing.add((tid, alias))
                stats["generated"] += 1
                stats[kind] += 1
                stats["by_category"][cat] += 1

        if not dry_run and inserts:
            for tid, alias, atype in inserts:
                conn.execute(
                    "INSERT OR IGNORE INTO terminology_alias (terminology_id, alias, alias_type) "
                    "VALUES (?, ?, ?)", (tid, alias, atype),
                )
            conn.commit()
            logger.info("写入 {} 条别名 (primary={}, secondary={})",
                        stats["generated"], stats["primary"], stats["secondary"])
        elif dry_run:
            logger.info("dry-run: 将生成 {} 条 (primary={}, secondary={})",
                        stats["generated"], stats["primary"], stats["secondary"])
            for tid, alias, atype in inserts[:30]:
                logger.info("  [{:>9s}] {} → {}", atype, _term_name(conn, tid), alias)

    return stats


def approve_draft_aliases(db_path: str | Path) -> int:
    memory = SQLiteMemory(db_path)
    with memory.connect() as conn:
        cur = conn.execute(
            "UPDATE terminology_alias SET alias_type = REPLACE(alias_type, '_draft', '') "
            "WHERE alias_type LIKE '%_draft'"
        )
        conn.commit()
        count = cur.rowcount
        logger.info("审批 {} 条 draft", count)
        return count


def _clean_term(term: str) -> str:
    for ch in "「」『』【】《》\"\"''（）()":
        term = term.replace(ch, "")
    return term


def _generate_candidates(term: str, category: str = "") -> list[str]:
    candidates: list[str] = []
    n = len(term)
    max_len = 2 if (category == "weapon" and n == 4) else 3

    # 后缀
    for length in range(max_len, 1, -1):
        if n >= length:
            suffix = term[-length:]
            if _is_valid_alias(suffix):
                candidates.append(suffix)

    # 前缀（不与后缀重复）
    for length in range(2, max_len + 1):
        if n >= length:
            prefix = term[:length]
            if _is_valid_alias(prefix) and prefix not in candidates:
                candidates.append(prefix)

    return candidates


def _is_valid_alias(alias: str) -> bool:
    if alias in _SERIES_PREFIXES:
        return False
    if len(alias) < 2:
        return False
    for ch in alias:
        if ch in ("之", "的", "与", "及", "或"):
            return False
    if alias[0] in _BLOCKED_PREFIXES:
        return False
    if alias[-1] in _BLOCKED_SUFFIXES:
        return False
    if not any("一" <= ch <= "鿿" for ch in alias):
        return False
    return True


def _classify(
    candidates: list[str], term: str, category: str,
) -> list[tuple[str, str]]:
    """分类为 primary / secondary。

    primary: 用于实体激活 + 可作 ASR 错听修复目标
    secondary: 仅精确匹配时激活实体，不参与近音纠错

    规则:
    1. 系列前缀武器 → 后缀=primary, 前缀不保留
    2. artifact 的「套」版 → primary
    3. 「之」前有意义前缀 → primary
    4. 其他 → 最高分=primary, 其余=secondary
    """
    if not candidates:
        return []

    result: list[tuple[str, str]] = []
    n = len(term)

    # 规则 1: 武器系列前缀检测
    if category == "weapon" and n >= 4:
        prefix2 = term[:2]
        if prefix2 in _SERIES_PREFIXES:
            for c in candidates:
                if term.endswith(c.replace("套", "")):
                    result.append((c, "primary"))
                elif c == prefix2:
                    continue  # 系列前缀本身不保留
                else:
                    result.append((c, "secondary"))
            return result

    # artifact_piece 单件最多出 secondary，不参与错听修复
    if category == "artifact_piece":
        for c in candidates:
            result.append((c, "secondary"))
        return result

    # 规则 2 & 3: artifact 套 / 之字术语
    if category == "artifact":
        tao = [c for c in candidates if c.endswith("套")]
        bare = [c for c in candidates if not c.endswith("套")]
        for c in tao:
            result.append((c, "primary"))
        for c in bare:
            result.append((c, "secondary"))
        return result

    # 「之」前的有义前缀 → primary
    zhi_idx = term.find("之")
    if zhi_idx > 0 and zhi_idx < n - 1:
        meaningful = term[:zhi_idx]
        for c in candidates:
            base = c.replace("套", "")
            if base == meaningful or c == meaningful + "套":
                result.append((c, "primary"))
            else:
                result.append((c, "secondary"))
        return result

    # 规则 4: 非系列武器 — 前缀=primary, 后缀=secondary
    # 冬极白星 → 冬极(primary), 白星(secondary)
    if category == "weapon" and len(candidates) == 2:
        # 区分前缀和后缀
        prefixes = [c for c in candidates if term.startswith(c.replace("套", ""))]
        suffixes = [c for c in candidates if term.endswith(c.replace("套", ""))]
        if prefixes and suffixes:
            for c in prefixes:
                result.append((c, "primary"))
            for c in suffixes:
                if c not in [r[0] for r in result]:
                    result.append((c, "secondary"))
            return result

    # 规则 5: 最高分=primary, 其余=secondary
    if len(candidates) == 1:
        result.append((candidates[0], "primary"))
    else:
        scored = [(c, _score_candidate(c, term)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        result.append((scored[0][0], "primary"))
        for c, _ in scored[1:]:
            result.append((c, "secondary"))

    return result


def _score_candidate(alias: str, term: str) -> int:
    score = 0
    base = alias.replace("套", "")

    # 「之」前的有义前缀优先（绝缘之旗印 → 绝缘/绝缘套 优先于 旗印/旗印套）
    zhi_idx = term.find("之")
    if zhi_idx > 0 and zhi_idx < len(term) - 1:
        meaningful = term[:zhi_idx]
        if alias == meaningful or alias == meaningful + "套":
            score += 15
        elif base == meaningful:
            score += 12

    # 后缀优先
    if term.endswith(alias):
        score += 10
    elif term.endswith(base):
        score += 8
    elif term.startswith(alias):
        score += 5
    elif term.startswith(base):
        score += 3

    if alias.endswith("套"):
        score += 5
    if len(alias) == 2:
        score += 3
    elif len(alias) == 3:
        score += 1

    return score


def _filter_best(candidates: list[str], term: str) -> list[str]:
    """每术语只保留最优的 2 个简称候选。"""
    if len(candidates) <= 2:
        return candidates
    scored = [(c, _score_candidate(c, term)) for c in candidates]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:2]]


def _is_basic_artifact(term: str, category: str) -> bool:
    if category not in ("artifact", "artifact_piece"):
        return False
    if term in _BASIC_ARTIFACT_TERMS:
        return True
    for basic in _BASIC_ARTIFACT_SETS:
        if term.startswith(basic) or term.endswith(basic):
            return True
    return False


def _term_name(conn, tid: int) -> str:
    row = conn.execute("SELECT term FROM terminology WHERE id = ?", (tid,)).fetchone()
    return row["term"] if row else f"id={tid}"
