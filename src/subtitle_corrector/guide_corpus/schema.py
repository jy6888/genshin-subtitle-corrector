"""攻略语料数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CleanedArticle(BaseModel):
    """清洗后的攻略文章（cleaned_articles.jsonl 每行）。"""

    source: str
    url: str
    title: str
    text: str
    fetched_at: str
    content_hash: str


class GuideSentence(BaseModel):
    """分句后的攻略句子（sentences.jsonl 每行）。"""

    source: str
    url: str
    title: str
    sentence: str
    entities: list[str] = Field(default_factory=list)
    surface_mentions: list[str] = Field(default_factory=list)
    content_hash: str
