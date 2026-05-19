"""中文分句器。

将攻略正文切分为独立的句子，供后续简称挖掘和 LM 训练使用。
"""

from __future__ import annotations

import hashlib
import re

from subtitle_corrector.guide_corpus.schema import CleanedArticle, GuideSentence

# 中英文句子结束标记
_SENTENCE_END_PATTERN = re.compile(
    r"[。！？?!；;．\n](?![」』）\)】』\"\'”])"
)

_MIN_SENTENCE_LEN = 4   # 最短句子长度（字符）
_MAX_SENTENCE_LEN = 200  # 最长句子长度（字符）


def split_sentences(
    articles: list[CleanedArticle],
    known_entities: set[str] | None = None,
) -> list[GuideSentence]:
    """将文章列表切分为句子，输出 GuideSentence 列表。"""
    known_entities = known_entities or set()
    sentences: list[GuideSentence] = []

    for article in articles:
        raw_sentences = _split_text(article.text)
        for sent in raw_sentences:
            sent = sent.strip()
            if len(sent) < _MIN_SENTENCE_LEN or len(sent) > _MAX_SENTENCE_LEN:
                continue

            entities = [e for e in known_entities if e in sent]
            surface_mentions: list[str] = []
            # 简单提取 2-4 字的中文简称（不在已知实体中但出现在含实体的句中）
            if entities:
                short_candidates = re.findall(r"[一-鿿]{2,4}", sent)
                for sc in short_candidates:
                    if sc not in known_entities and sc not in surface_mentions:
                        # 不是已知术语的短中文片段可能是简称
                        surface_mentions.append(sc)

            sentences.append(GuideSentence(
                source=article.source,
                url=article.url,
                title=article.title,
                sentence=sent,
                entities=entities,
                surface_mentions=surface_mentions,
                content_hash=_hash_sentence(sent),
            ))

    return sentences


def _split_text(text: str) -> list[str]:
    """按中英文标点切分文本。"""
    parts = _SENTENCE_END_PATTERN.split(text)
    result: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            result.append(part)
    return result


def _hash_sentence(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
