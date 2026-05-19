"""正常攻略话术评分器。

基于训练好的 NormalGuideLM 对字幕片段打分，
判断是否像正常攻略表达。
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitle_corrector.normal_lm.model import NormalGuideLM


@dataclass
class SurfaceDiagnosis:
    """表面形式诊断结果。"""

    surface: str
    is_normal: bool
    score: float
    nearest_alias: str | None
    alias_similarity: float
    canonical_hint: str | None
    reason: str


def diagnose_surface(
    text: str,
    model: NormalGuideLM,
    alias_lexicon: dict[str, str] | None = None,
    threshold: float = 0.45,
) -> SurfaceDiagnosis:
    """诊断一个文本片段是否是正常攻略表达。

    Args:
        text: 待诊断的 2-4 字中文片段
        model: 训练好的 NormalGuideLM
        alias_lexicon: surface → canonical_term 映射（简称词典）
        threshold: 低于此分数判定为异常

    Returns:
        SurfaceDiagnosis
    """
    alias_lexicon = alias_lexicon or {}
    text = text.strip()

    # 如果本身就在简称词典中，直接判为正常
    if text in alias_lexicon:
        return SurfaceDiagnosis(
            surface=text,
            is_normal=True,
            score=1.0,
            nearest_alias=text,
            alias_similarity=1.0,
            canonical_hint=alias_lexicon[text],
            reason="surface is a known spoken alias",
        )

    # 查找最接近的已知简称
    nearest_alias, alias_sim = model.find_nearest_alias(text)

    # 用模型打分
    score = model.score_text(text, alias_hint=nearest_alias)

    # 判定
    is_normal = score >= threshold

    # 构建原因
    if is_normal:
        reason = "surface appears normal in guide corpus"
    elif nearest_alias and alias_sim >= 0.5:
        reason = (
            f"surface is rare in guide corpus but phonetic-near "
            f"to known spoken alias '{nearest_alias}'"
        )
    else:
        reason = "surface is rare in guide corpus with no near alias"

    return SurfaceDiagnosis(
        surface=text,
        is_normal=is_normal,
        score=round(score, 3),
        nearest_alias=nearest_alias,
        alias_similarity=round(alias_sim, 3),
        canonical_hint=alias_lexicon.get(nearest_alias or ""),
        reason=reason,
    )
