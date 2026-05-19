from __future__ import annotations

import os

from loguru import logger

from subtitle_corrector.config.settings import AppSettings
from subtitle_corrector.detector import (
    EntityConsistencyDetector,
    JiebaSpanDetector,
    LanguageModelDetector,
)
from subtitle_corrector.matcher import FuzzyTerminologyMatcher
from subtitle_corrector.memory import SQLiteMemory, SQLiteTerminologyRepository
from subtitle_corrector.memory.category_activation import CategoryActivationManager
from subtitle_corrector.normalize import TextNormalizer
from subtitle_corrector.pipeline.executor import SubtitleCorrectionPipeline
from subtitle_corrector.resolver import NoopArbitrator
from subtitle_corrector.resolver.base import LLMArbitrator
from subtitle_corrector.scoring import WeightedRiskAggregator


def _build_arbitrator(settings: AppSettings) -> LLMArbitrator:
    """Construct the appropriate arbitrator based on configuration.

    If ``resolver.provider`` is set and ``OPENAI_API_KEY`` is available in
    the environment, instantiate the real :class:`OpenAIArbitrator`.
    Otherwise fall back to the no-op stub so the pipeline never crashes.
    """
    if settings.resolver.provider is not None:
        # Check env BEFORE importing — llm.py does load_dotenv() at import
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            from subtitle_corrector.resolver.llm import OpenAIArbitrator  # noqa: WPS433
            return OpenAIArbitrator(settings.resolver)
        else:
            logger.warning(
                "resolver.provider is set to '{}' but OPENAI_API_KEY is not "
                "found in environment – falling back to NoopArbitrator",
                settings.resolver.provider,
            )
    return NoopArbitrator()


def build_pipeline(settings: AppSettings) -> SubtitleCorrectionPipeline:
    memory = SQLiteMemory(settings.memory.sqlite_path)
    terminology_repo = SQLiteTerminologyRepository(memory)
    matcher = FuzzyTerminologyMatcher(
        terminology_repo,
        threshold=settings.matcher.fuzzy_threshold,
        pinyin_threshold=settings.matcher.pinyin_threshold,
        boost_factor=settings.entity_memory.boost_factor,
    )

    hot_categories = set(settings.category_filter.hot_categories)
    cold_categories = set(settings.category_filter.cold_categories)
    category_activation = CategoryActivationManager(
        decay_rate=settings.category_filter.decay_rate,
        min_weight=settings.category_filter.min_weight,
    )

    # 创建口语简称词典（始终加载，用于 JiebaSpanDetector 区分 context_alias）
    alias_lexicon = None
    if settings.language_model.aliases_path:
        from subtitle_corrector.spoken_alias.lexicon import SpokenAliasLexicon
        alias_lexicon = SpokenAliasLexicon(settings.language_model.aliases_path)

    # 创建角色别名词典
    character_lexicon = None
    if settings.language_model.character_aliases_path:
        from subtitle_corrector.character_alias.lexicon import CharacterAliasLexicon
        character_lexicon = CharacterAliasLexicon(settings.language_model.character_aliases_path)

    detectors = [
        JiebaSpanDetector(matcher, hot_categories, cold_categories, category_activation, alias_lexicon=alias_lexicon, character_lexicon=character_lexicon),
        EntityConsistencyDetector(),
        LanguageModelDetector(
            model_path=settings.language_model.model_path,
            alias_lexicon=alias_lexicon,
            character_lexicon=character_lexicon,
            anomaly_threshold=settings.language_model.anomaly_threshold,
        ),
    ]
    return SubtitleCorrectionPipeline(
        normalizer=TextNormalizer(settings.normalization),
        detectors=detectors,
        aggregator=WeightedRiskAggregator(settings.scoring.detector_weights),
        arbitrator=_build_arbitrator(settings),
        matcher=matcher,
        entity_memory_settings=settings.entity_memory,
        category_activation=category_activation,
        alias_lexicon=alias_lexicon,
        character_lexicon=character_lexicon,
        llm_threshold=settings.scoring.llm_threshold,
        context_window=settings.pipeline.context_window,
        dry_run=settings.pipeline.dry_run,
    )
