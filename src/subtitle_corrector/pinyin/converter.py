from __future__ import annotations

from pypinyin import Style, lazy_pinyin
from rapidfuzz.distance import Levenshtein


class PinyinConverter:
    def to_pinyin(self, text: str, tone: bool = False) -> list[str]:
        style = Style.TONE3 if tone else Style.NORMAL
        return lazy_pinyin(text, style=style, errors="ignore")

    def compact(self, text: str, tone: bool = False) -> str:
        return " ".join(self.to_pinyin(text, tone=tone))


class FuzzyPinyinMatcher:
    def __init__(self, converter: PinyinConverter | None = None) -> None:
        self.converter = converter or PinyinConverter()

    def similarity(self, left: str, right: str) -> float:
        left_py = self.converter.compact(left)
        right_py = self.converter.compact(right)
        if not left_py or not right_py:
            return 0.0
        return Levenshtein.normalized_similarity(left_py, right_py)
