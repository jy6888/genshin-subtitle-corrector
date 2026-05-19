"""正常攻略话术模型训练器。

从攻略句子 JSONL 训练轻量统计语言模型。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

from subtitle_corrector.normal_lm.model import NormalGuideLM

_WINDOW_SIZE = 5  # 术语周围窗口大小
_ALIAS_WINDOW = 3  # 简称周围窗口大小


def train_normal_guide_lm(
    sentences_path: str | Path,
    output_path: str | Path | None = None,
    approved_aliases_path: str | Path | None = None,
) -> NormalGuideLM:
    """从攻略句子训练正常攻略话术模型。

    Args:
        sentences_path: sentences.jsonl 路径
        output_path: 模型输出 JSON 路径
        approved_aliases_path: 审核后简称 CSV（用于提取 alias 上下文）

    Returns:
        训练好的 NormalGuideLM
    """
    model = NormalGuideLM()

    # 加载句子
    sentences: list[dict] = []
    with open(sentences_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))
    logger.info("训练集: {} 条句子", len(sentences))

    # 加载已审核简称
    approved_aliases: dict[str, str] = {}  # surface → canonical_term
    if approved_aliases_path:
        import csv
        with open(approved_aliases_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                surface = (row.get("alias_surface") or "").strip()
                canonical = (row.get("canonical_term") or "").strip()
                if surface and canonical:
                    approved_aliases[surface] = canonical

    # 1. 收集字符 bigram 和 trigram
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()

    for sent in sentences:
        text = sent.get("sentence", "")
        # 清理非中文字符
        text = re.sub(r"[^一-鿿]", "", text)
        for i in range(len(text) - 1):
            bigrams[text[i:i + 2]] += 1
        for i in range(len(text) - 2):
            trigrams[text[i:i + 3]] += 1

    model.char_ngrams[2] = bigrams
    model.char_ngrams[3] = trigrams
    model.total_ngrams[2] = sum(bigrams.values())
    model.total_ngrams[3] = sum(trigrams.values())
    logger.info("n-gram 统计: {} bigrams, {} trigrams", len(bigrams), len(trigrams))

    # 2. 实体实体周围上下文模式
    for sent in sentences:
        text = sent.get("sentence", "")
        entities = sent.get("entities", [])
        for entity in entities:
            idx = text.find(entity)
            if idx < 0:
                continue
            start = max(0, idx - _WINDOW_SIZE)
            end = min(len(text), idx + len(entity) + _WINDOW_SIZE)
            context = text[start:end]
            # 提取窗口内的 2-3 字搭配
            for i in range(len(context) - 1):
                bg = context[i:i + 2]
                model.context_patterns[entity][bg] += 1

    # 3. 简称上下文（如果提供）
    if approved_aliases:
        alias_contexts_raw: dict[str, list[str]] = defaultdict(list)
        for sent in sentences:
            text = sent.get("sentence", "")
            for alias_surface in approved_aliases:
                idx = text.find(alias_surface)
                if idx < 0:
                    continue
                start = max(0, idx - _ALIAS_WINDOW)
                end = min(len(text), idx + len(alias_surface) + _ALIAS_WINDOW)
                context = text[start:end]
                # 提取简称前后的词
                before = text[max(0, idx - 2):idx]
                after = text[idx + len(alias_surface):idx + len(alias_surface) + 2]
                if before.strip():
                    alias_contexts_raw[alias_surface].append(before)
                if after.strip():
                    alias_contexts_raw[alias_surface].append(after)

        # 取频率最高的上下文
        for alias_surface, contexts in alias_contexts_raw.items():
            top_contexts = [ctx for ctx, _ in Counter(contexts).most_common(10)]
            model.alias_contexts[alias_surface] = top_contexts

    logger.info(
        "上下文模式: {} 个实体, {} 个简称",
        len(model.context_patterns),
        len(model.alias_contexts),
    )

    # 写出模型
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(model.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("模型已保存到 {}", output_path)

    return model
