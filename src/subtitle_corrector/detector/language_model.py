"""语言模型异常检测器。

基于 NormalGuideLM + 口语简称词典，判断字幕中的中文片段
是否像正常攻略表达。已知简称不触发修复，异常近音词标记为 suspicious。
"""

from __future__ import annotations

import json

from loguru import logger

from subtitle_corrector.detector.base import Detector
from subtitle_corrector.schemas import Candidate, DetectionResult, SubtitleCue, SubtitleDocument


class LanguageModelDetector(Detector):
    name = "language_model"

    def __init__(
        self,
        model_path: str | None = None,
        alias_lexicon: object | None = None,
        character_lexicon: object | None = None,
        anomaly_threshold: float = 0.45,
    ) -> None:
        """初始化语言模型检测器。

        Args:
            model_path: NormalGuideLM JSON 路径，None 则为 stub 模式
            alias_lexicon: SpokenAliasLexicon 实例
            character_lexicon: CharacterAliasLexicon 实例
            anomaly_threshold: 低于此分数的表面形式判定为异常
        """
        from subtitle_corrector.normal_lm.model import NormalGuideLM

        self._model: NormalGuideLM | None = None
        self._lexicon = alias_lexicon
        self._char_lexicon = character_lexicon
        self._anomaly_threshold = anomaly_threshold

        if model_path:
            try:
                with open(model_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._model = NormalGuideLM.from_dict(data)
                logger.info("LanguageModelDetector 加载模型: {} ({} bigrams)",
                            model_path, len(self._model.char_ngrams.get(2, {})))
            except FileNotFoundError:
                logger.info("LanguageModelDetector: 模型文件不存在 ({}), 使用 stub 模式", model_path)
            except Exception as exc:
                logger.warning("LanguageModelDetector: 加载模型失败: {}", exc)

        if self._lexicon is not None and len(self._lexicon) > 0:
            logger.info("LanguageModelDetector 接入简称词典 ({} 条)", len(self._lexicon))

    @property
    def is_active(self) -> bool:
        term_ok = self._model is not None and self._lexicon is not None and len(self._lexicon) > 0
        char_ok = self._char_lexicon is not None and len(self._char_lexicon) > 0
        return term_ok or char_ok

    @property
    def alias_lexicon(self):
        return self._lexicon

    def _extract_surfaces(self, text: str) -> list[str]:
        """滑动窗口提取所有 2-4 字 CJK 片段（去重，保留顺序）。"""
        surfaces: list[str] = []
        seen: set[str] = set()
        for window in (2, 3, 4):
            for i in range(len(text) - window + 1):
                sf = text[i:i + window]
                if sf in seen:
                    continue
                # 必须是纯 CJK
                if not all("一" <= ch <= "鿿" for ch in sf):
                    continue
                seen.add(sf)
                surfaces.append(sf)
        return surfaces

    def detect(self, cue: SubtitleCue, document: SubtitleDocument) -> DetectionResult:
        """检测字幕中是否存在异常的近音词/非正常表达。

        策略:
        1. 提取文本中所有 2-4 字中文片段（滑动窗口）
        2. 若片段在简称词典中 → 正常，不触发
        3. 若片段不在词典中但 LM 分数低 → 标记异常
        4. 若片段有近音已知简称 → 提供 canonical_hint
        """
        if not self.is_active:
            return DetectionResult(
                detector=self.name,
                cue_index=cue.index,
                risk_score=0.0,
                reason="language model not configured",
            )

        text = cue.text
        surfaces = self._extract_surfaces(text)

        # 收集所有已知别名（术语 + 角色），用于排除重叠 surface
        all_known_surfaces: set[str] = set()
        if self._lexicon is not None:
            all_known_surfaces.update(self._lexicon.surface_to_canonical.keys())
        if self._char_lexicon is not None:
            all_known_surfaces.update(self._char_lexicon.all_aliases.keys())

        # 过滤：与已知别名字符位置重叠的 surface 不重复诊断
        # 先找到所有已知别名在原文中的位置
        known_positions: list[tuple[int, int, str]] = []  # (start, end, surface)
        for alias in all_known_surfaces:
            idx = text.find(alias)
            while idx >= 0:
                known_positions.append((idx, idx + len(alias), alias))
                idx = text.find(alias, idx + 1)

        # 如果 surface 的字符位置与已知别名有重叠，跳过
        surfaces_filtered: list[str] = []
        for sf in surfaces:
            if sf in all_known_surfaces:
                surfaces_filtered.append(sf)
                continue
            sf_start = text.find(sf)
            if sf_start < 0:
                surfaces_filtered.append(sf)
                continue
            sf_end = sf_start + len(sf)
            overlaps_known = any(
                not (sf_end <= ks or sf_start >= ke)
                for ks, ke, _ in known_positions
            )
            if not overlaps_known:
                surfaces_filtered.append(sf)
        surfaces = surfaces_filtered
        if not surfaces:
            return DetectionResult(
                detector=self.name,
                cue_index=cue.index,
                risk_score=0.0,
                reason="no CJK surface to analyze",
            )

        from subtitle_corrector.normal_lm.scorer import diagnose_surface

        alias_map = self._lexicon.surface_to_canonical if self._lexicon else {}
        # 合并角色外号到 alias_map（外号→标准角色名），用于近音诊断
        if self._char_lexicon is not None:
            for surface, entry in self._char_lexicon.all_aliases.items():
                if surface not in alias_map:
                    alias_map[surface] = entry.canonical_term

        suspicious: list[dict] = []
        known_aliases: list[dict] = []
        max_risk = 0.0

        for sf in surfaces:
            # 检查是否是已知简称（术语）
            is_term_alias = self._lexicon is not None and sf in self._lexicon
            # 检查是否是角色外号
            is_char_alias = self._char_lexicon is not None and sf in self._char_lexicon

            if is_term_alias:
                known_aliases.append({
                    "surface": sf,
                    "canonical": self._lexicon.get_canonical(sf),
                    "kind": self._lexicon.get_kind(sf),
                })
                continue

            if is_char_alias and len(sf) >= 2:
                # 角色多字外号：记录为已知，不产替换候选（仅激活实体）
                entry = self._char_lexicon.get_entry(sf)
                known_aliases.append({
                    "surface": sf,
                    "canonical": entry.canonical_term,
                    "kind": entry.alias_kind,
                })
                continue

            # 对非已知简称进行 LM 诊断（仅在 model 可用时）
            if self._model is None:
                continue
            diagnosis = diagnose_surface(
                text=sf,
                model=self._model,
                alias_lexicon=alias_map,
                threshold=self._anomaly_threshold,
            )

            # 仅在有近音别名时产出候选（如 夫妇→芙芙），纯噪音忽略
            if diagnosis.is_normal or diagnosis.nearest_alias is None:
                continue
            risk = 1.0 - diagnosis.score
            max_risk = max(max_risk, risk)
            suspicious.append({
                "surface": sf,
                "score": diagnosis.score,
                "nearest_alias": diagnosis.nearest_alias,
                "canonical_hint": diagnosis.canonical_hint,
                "reason": diagnosis.reason,
            })

        # 构建结果
        reason_parts: list[str] = []
        candidates: list[Candidate] = []

        if suspicious:
            reason_parts.append(f"{len(suspicious)} suspicious surfaces")
            for item in suspicious:
                # 修正目标为正常简称（如 金珀），不是 canonical 全称（如 试作金珀）
                nearest = item.get("nearest_alias")
                canonical = item.get("canonical_hint", "")
                candidate_value = nearest or item["surface"]
                candidates.append(Candidate(
                    value=candidate_value,
                    source="language_model",
                    score=round(1.0 - item["score"], 2),
                    explanation=item.get("reason", ""),
                    metadata={
                        "suspicious_surface": item["surface"],
                        "nearest_normal_alias": nearest,
                        "canonical_hint": canonical,
                        "expansion_policy": "contextual_expand",
                        "evidence_type": "lm_near_alias",
                    },
                ))

        if known_aliases:
            reason_parts.append(f"{len(known_aliases)} known aliases detected")
            # 已知简称也产出候选——值用简称本身（如 金珀），全称放 canonical_hint
            # 字数方向门在 candidate_alignment 中拦截反方向扩写
            for item in known_aliases:
                candidates.append(Candidate(
                    value=item["surface"],
                    source="language_model",
                    score=0.95,
                    explanation=f"known {item.get('kind', 'alias')}: {item['surface']} → {item['canonical']}",
                    metadata={
                        "surface": item["surface"],
                        "alias_kind": item.get("kind"),
                        "canonical_hint": item["canonical"],
                        "is_known_alias": True,
                        "expansion_policy": "preserve_surface",
                        "evidence_type": "known_alias",
                    },
                ))

        reason = "; ".join(reason_parts) if reason_parts else "no language model anomalies"

        return DetectionResult(
            detector=self.name,
            cue_index=cue.index,
            risk_score=round(max_risk, 3),
            reason=reason,
            candidates=candidates,
            metadata={
                "suspicious_count": len(suspicious),
                "known_alias_count": len(known_aliases),
                "suspicious_surfaces": suspicious,
                "known_aliases": known_aliases,
            },
        )
