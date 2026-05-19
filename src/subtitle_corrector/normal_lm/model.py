"""正常攻略话术模型数据结构。"""

from __future__ import annotations

from collections import Counter, defaultdict


class NormalGuideLM:
    """轻量统计语言模型，基于攻略语料训练。

    Attrs:
        version: 模型版本号
        char_ngrams: 字符 n-gram 频率 (n=2,3)
        context_patterns: 术语周围窗口的搭配频率
        alias_contexts: 简称 → 常见上下文搭配列表
        total_ngrams: 总 n-gram 数（用于归一化）
    """

    def __init__(self) -> None:
        self.version: int = 1
        self.char_ngrams: dict[int, Counter] = defaultdict(Counter)
        self.context_patterns: dict[str, Counter] = defaultdict(Counter)
        self.alias_contexts: dict[str, list[str]] = {}
        self.total_ngrams: dict[int, int] = {}

    def score_text(self, text: str, alias_hint: str | None = None) -> float:
        """对文本片段打分（是否像正常攻略表达）。

        Args:
            text: 待打分文本（2-6 字片段）
            alias_hint: 疑似近音词对应的正常简称（可选）

        Returns:
            0.0-1.0 分数，越高越像正常攻略表达
        """
        text = text.strip()
        if not text:
            return 0.5

        scores: list[float] = []

        # 1. 字符 bigram 频率分
        bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
        if bigrams and 2 in self.char_ngrams and self.total_ngrams.get(2, 0) > 0:
            bg_scores: list[float] = []
            for bg in bigrams:
                count = self.char_ngrams[2].get(bg, 0)
                bg_scores.append(min(count / max(self.total_ngrams[2] * 0.001, 1), 1.0))
            scores.append(sum(bg_scores) / len(bg_scores))
        else:
            scores.append(0.5)  # 无训练数据时中性

        # 2. 字符 trigram 频率分
        trigrams = [text[i:i + 3] for i in range(len(text) - 2)]
        if trigrams and 3 in self.char_ngrams and self.total_ngrams.get(3, 0) > 0:
            tg_scores: list[float] = []
            for tg in trigrams:
                count = self.char_ngrams[3].get(tg, 0)
                tg_scores.append(min(count / max(self.total_ngrams[3] * 0.001, 1), 1.0))
            scores.append(sum(tg_scores) / len(tg_scores))

        # 3. 别名上下文匹配分
        if alias_hint and alias_hint in self.alias_contexts:
            expected_contexts = self.alias_contexts[alias_hint]
            context_match = any(ctx in text for ctx in expected_contexts)
            scores.append(0.9 if context_match else 0.2)
        elif alias_hint:
            scores.append(0.4)  # alias 无上下文数据

        # 4. 整体 n-gram 覆盖率（count>0 才算覆盖）
        all_bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
        if all_bigrams and 2 in self.char_ngrams:
            covered = sum(1 for bg in all_bigrams if self.char_ngrams[2].get(bg, 0) > 0)
            scores.append(covered / len(all_bigrams))

        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    def find_nearest_alias(self, surface: str) -> tuple[str | None, float]:
        """查找与 surface 最相似的已知简称。

        Returns:
            (alias_surface, similarity_score) or (None, 0)
        """
        from difflib import SequenceMatcher
        from pypinyin import lazy_pinyin

        best_match = None
        best_score = 0.0

        surface_py = "".join(lazy_pinyin(surface))

        for alias in self.alias_contexts:
            # 字符相似度
            char_sim = SequenceMatcher(None, surface, alias).ratio()
            # 拼音相似度（作为回退，当字符无重叠时仍有信号）
            alias_py = "".join(lazy_pinyin(alias))
            py_sim = SequenceMatcher(None, surface_py, alias_py).ratio()
            # 取两者最大值（拼音对近音词如 夫妇→芙芙 更有效）
            sim = max(char_sim, py_sim * 0.85)  # 拼音降权，避免跨语言误匹配

            if sim > best_score:
                best_score = sim
                best_match = alias

        if best_score >= 0.5:
            return best_match, best_score
        return None, best_score

    def to_dict(self) -> dict:
        """序列化为 JSON 兼容字典。"""
        return {
            "version": self.version,
            "ngram_counts": {
                str(n): dict(c.most_common(10000))
                for n, c in self.char_ngrams.items()
            },
            "context_patterns": {
                term: dict(ctx.most_common(100))
                for term, ctx in self.context_patterns.items()
            },
            "alias_contexts": dict(self.alias_contexts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NormalGuideLM":
        """从字典反序列化。"""
        model = cls()
        model.version = data.get("version", 1)
        for n_str, counts in data.get("ngram_counts", {}).items():
            n = int(n_str)
            model.char_ngrams[n] = Counter(counts)
            model.total_ngrams[n] = sum(counts.values())
        for term, ctx in data.get("context_patterns", {}).items():
            model.context_patterns[term] = Counter(ctx)
        model.alias_contexts = data.get("alias_contexts", {})
        return model
