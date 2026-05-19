from __future__ import annotations

import re

from subtitle_corrector.config.settings import NormalizationSettings
from subtitle_corrector.schemas import NormalizedText


PUNCT_MAP = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "：": ":",
        "；": ";",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


class TextNormalizer:
    def __init__(self, settings: NormalizationSettings | None = None) -> None:
        self.settings = settings or NormalizationSettings()
        self._opencc = None
        if self.settings.traditional_to_simplified:
            try:
                from opencc import OpenCC

                self._opencc = OpenCC("t2s")
            except ImportError:
                self._opencc = None

    def normalize(self, text: str) -> NormalizedText:
        value = text
        operations: list[str] = []
        if self.settings.traditional_to_simplified and self._opencc is not None:
            value = self._opencc.convert(value)
            operations.append("traditional_to_simplified")
        if self.settings.punctuation:
            value = value.translate(PUNCT_MAP)
            operations.append("punctuation")
        if self.settings.spacing:
            value = re.sub(r"[ \t]+", " ", value).strip()
            value = re.sub(r"\s+([,.!?:;])", r"\1", value)
            operations.append("spacing")
        # ASR 错听修复：Q1/eq 永远是 QE 的错听
        value = re.sub(r"Q1", "QE", value, flags=re.IGNORECASE)
        value = re.sub(r"(?<![a-zA-Z])eq(?![a-zA-Z])", "QE", value, flags=re.IGNORECASE)
        if self.settings.lowercase_latin:
            value = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), value)
            operations.append("lowercase_latin")
        return NormalizedText(original=text, normalized=value, operations=operations)
