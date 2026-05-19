"""术语内部简称挖掘器 (TermShortAliasMiner)。

从标准术语自身生成简称候选（连续子串），再去攻略语料中验证出现频率。
只做术语内部简称，不挖角色外号/称号。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from loguru import logger

from subtitle_corrector.spoken_alias.schema import AliasCandidate

# 第一版优先类别
_TARGET_CATEGORIES = {"weapon", "artifact", "artifact_piece", "weapon_effect"}

# 子串中不能以这些词开头或结尾（功能虚词/通用后缀）
_BLOCKED_PREFIXES = {"之", "的", "与", "和", "及", "或", "一", "圣", "遗"}
_BLOCKED_SUFFIXES = {"之", "的", "与", "和", "及", "或", "套", "物"}

# 整词不允许作为简称（通用词）
_BLOCKED_FULL = {
    "一套", "武器", "圣遗物", "试作", "破魔", "讨龙", "铁蜂",
    "西风", "祭礼", "宗室", "角斗", "流浪", "教官", "战狂",
    "学士", "赌徒", "奇迹", "冰风", "水仙", "花海", "乐园",
    "深林", "饰金", "辰砂", "来歆", "余响", "追忆",
}

# 有些前缀太通用（属于武器系列前缀），作为独立简称会歧义
_AMBIGUOUS_PREFIXES = {
    "西风", "祭礼", "宗室", "暗巷", "雨裁", "匣里",
}

# 最小语料出现次数
_MIN_EVIDENCE = 2


def mine_aliases(
    sentences_path: str | Path,
    terminology_csv: str | Path | None = None,
    output_path: str | Path | None = None,
) -> list[AliasCandidate]:
    """从术语内部子串挖掘简称候选。

    Args:
        sentences_path: sentences.jsonl 路径
        terminology_csv: genshin_terms.csv 路径
        output_path: 候选 CSV 输出路径
    """
    # 1. 加载术语
    terms: list[dict] = []
    if terminology_csv:
        with open(terminology_csv, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                cat = (row.get("category") or "").strip()
                if cat in _TARGET_CATEGORIES:
                    term = (row.get("term") or "").strip()
                    if term and len(term) >= 4:
                        terms.append({"term": term, "category": cat})
    logger.info("加载 {} 个目标术语 (>=4字)", len(terms))

    # 2. 加载句子语料
    sentences: list[str] = []
    with open(sentences_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                sentences.append(obj.get("sentence", ""))

    # 预计算语料全文（用于快速子串计数）
    corpus_text = "\n".join(sentences)
    logger.info("加载 {} 条句子作为验证语料", len(sentences))

    # 3. 为每个术语生成简称候选并验证
    # alias → list of (term, category, evidence_count)
    alias_map: dict[str, list[dict]] = defaultdict(list)

    for t in terms:
        term = t["term"]
        cat = t["category"]
        candidates = _generate_candidates(term)

        # artifact/artifact_piece 额外生成「子串+套」变体
        if cat in ("artifact", "artifact_piece"):
            for c in list(candidates):
                tao_alias = c + "套"
                if tao_alias not in candidates:
                    candidates.append(tao_alias)

        for alias in candidates:
            count = _count_in_corpus(alias, corpus_text, sentences)
            if count >= _MIN_EVIDENCE:
                alias_map[alias].append({
                    "term": term, "category": cat, "evidence": count,
                })

    # 4. 构建结果
    results: list[AliasCandidate] = []
    for alias, sources in alias_map.items():
        # 取 evidence 最高的 canonical
        sources.sort(key=lambda s: -s["evidence"])
        best = sources[0]
        confidence = _compute_confidence(alias, best["term"], best["evidence"], len(sources))

        risk_flags = _assess_risk(alias, best["term"], len(sources))
        evidence_texts = _collect_evidence(alias, sentences, limit=5)

        results.append(AliasCandidate(
            alias_surface=alias,
            canonical_term=best["term"],
            confidence=round(confidence, 2),
            evidence_count=best["evidence"],
            evidence_examples="|".join(evidence_texts),
            sources="guide_corpus",
            risk_flags=risk_flags,
            review_status="pending",
        ))

    # 按置信度降序
    results.sort(key=lambda c: (-c.confidence, -c.evidence_count))

    logger.info("生成 {} 个简称候选", len(results))

    if output_path:
        _write_candidates_csv(results, output_path)

    return results


def _generate_candidates(term: str) -> list[str]:
    """为术语生成 2-4 字简称候选（优先后缀）。"""
    candidates: list[str] = []
    n = len(term)

    # 后缀优先（最后 2-3 字）
    for length in (3, 2):
        if n >= length:
            suffix = term[-length:]
            if _is_valid_alias(suffix, term):
                candidates.append(suffix)

    # 前缀补充（前 2-3 字），仅在后缀不够时
    for length in (2, 3):
        if n >= length:
            prefix = term[:length]
            if _is_valid_alias(prefix, term) and prefix not in candidates:
                candidates.append(prefix)

    # 不生成中间子串（如 莫斯之 从 阿莫斯之弓）

    return candidates


def _is_valid_alias(alias: str, term: str) -> bool:
    """检查子串是否可作为合法简称。"""
    if alias in _BLOCKED_FULL:
        return False
    if len(alias) < 2:
        return False

    # 虚词不能出现在简称的任何位置
    for ch in alias:
        if ch in ("之", "的", "与", "及", "或"):
            return False

    # 不能以虚字开头或结尾
    if alias[0] in _BLOCKED_PREFIXES:
        return False
    if alias[-1] in _BLOCKED_SUFFIXES:
        return False

    # 纯标点/数字
    if not any("一" <= ch <= "鿿" for ch in alias):
        return False

    return True


def _count_in_corpus(alias: str, corpus_text: str, sentences: list[str]) -> int:
    """统计简称在语料中的出现次数（句级别去重）。"""
    count = 0
    for sent in sentences:
        if alias in sent:
            count += 1
    return count


def _compute_confidence(
    alias: str, term: str, evidence: int, source_count: int,
) -> float:
    """计算置信度。

    因素：
    - evidence 多的加分
    - 单个 canonical (source_count==1) 加分
    - 歧义 (source_count>1) 减分
    """
    # 基础分：evidence 越多越高
    if evidence >= 20:
        base = 0.9
    elif evidence >= 10:
        base = 0.8
    elif evidence >= 5:
        base = 0.7
    elif evidence >= 3:
        base = 0.6
    else:
        base = 0.5

    # 唯一性：只有一个 canonical → 加分
    if source_count == 1:
        base += 0.08

    # 歧义惩罚
    if source_count > 1:
        base -= 0.10 * (source_count - 1)

    # 后缀偏好：后缀简称比前缀更可靠
    if term.endswith(alias):
        base += 0.05

    return max(0.0, min(1.0, base))


def _assess_risk(alias: str, term: str, source_count: int) -> str:
    """评估风险标记。"""
    flags = []

    if term.endswith(alias):
        flags.append("unique_suffix")
    elif term.startswith(alias):
        flags.append("unique_prefix")
    else:
        flags.append("mid_substring")

    if source_count > 1:
        flags.append("ambiguous_alias")

    if alias in _AMBIGUOUS_PREFIXES:
        flags.append("ambiguous_prefix")

    return "|".join(flags)


def _collect_evidence(alias: str, sentences: list[str], limit: int = 5) -> list[str]:
    """收集包含该简称的例句。"""
    examples: list[str] = []
    for sent in sentences:
        if alias in sent:
            # 截取简称周围上下文
            idx = sent.find(alias)
            start = max(0, idx - 6)
            end = min(len(sent), idx + len(alias) + 6)
            snippet = sent[start:end].strip()
            if len(snippet) > 50:
                snippet = snippet[:50]
            examples.append(snippet)
            if len(examples) >= limit:
                break
    return examples


def _write_candidates_csv(candidates: list[AliasCandidate], output_path: str | Path) -> None:
    """写出候选 CSV。"""
    fieldnames = [
        "alias_surface", "canonical_term", "confidence", "evidence_count",
        "evidence_examples", "sources", "risk_flags", "review_status", "notes",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            writer.writerow({
                "alias_surface": c.alias_surface,
                "canonical_term": c.canonical_term,
                "confidence": c.confidence,
                "evidence_count": c.evidence_count,
                "evidence_examples": c.evidence_examples,
                "sources": c.sources,
                "risk_flags": c.risk_flags,
                "review_status": c.review_status,
                "notes": c.notes,
            })
    logger.info("写出 {} 条候选到 {}", len(candidates), output_path)
