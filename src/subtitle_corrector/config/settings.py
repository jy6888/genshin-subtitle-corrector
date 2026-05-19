from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


SUPPORTED_MIMO_MODELS = {
    "mimo-v2.5-pro",
    "mimo-v2.5",
    "mimo-v2.5-tts-voiceclone",
    "mimo-v2.5-tts-voicedesign",
    "mimo-v2.5-tts",
    "mimo-v2-pro",
    "mimo-v2-omni",
    "mimo-v2-tts",
}


class MemorySettings(BaseModel):
    sqlite_path: Path = Path("data/subtitle_memory.sqlite3")


class EntityMemorySettings(BaseModel):
    decay_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    min_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    boost_factor: float = Field(default=0.15, ge=0.0, le=1.0)


class NormalizationSettings(BaseModel):
    punctuation: bool = True
    spacing: bool = True
    traditional_to_simplified: bool = False
    lowercase_latin: bool = False


class MatcherSettings(BaseModel):
    fuzzy_threshold: float = 82.0
    pinyin_threshold: float = 76.0
    max_candidates: int = 8


class ScoringSettings(BaseModel):
    detector_weights: dict[str, float] = Field(default_factory=dict)
    llm_threshold: float = 0.72


class ResolverSettings(BaseModel):
    provider: str | None = None
    model: str | None = None
    discovery_model: str | None = None
    phase2_model: str | None = None
    timeout_seconds: int = 30
    temperature: float = 0.1
    max_tokens: int = 4096

    @field_validator("model", "discovery_model", "phase2_model")
    @classmethod
    def validate_mimo_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value != value.lower():
            raise ValueError("MiMo model names must be lowercase")
        if value not in SUPPORTED_MIMO_MODELS:
            raise ValueError(f"Unsupported MiMo model: {value}")
        return value


class PipelineSettings(BaseModel):
    context_window: int = 2
    dry_run: bool = True


class CandidateExpansionSettings(BaseModel):
    discovery_max_candidates_per_cue: int = Field(default=2, ge=1)
    max_candidates_per_cue: int = 4
    enable_entity_consistency: bool = True
    enable_long_entity_variant: bool = True
    enable_local_term: bool = True
    local_term_requires_stable_entity: bool = True
    terminology_requery_hint_enabled: bool = True
    terminology_requery_hint_min_alignment: float = Field(default=0.72, ge=0.0, le=1.0)
    terminology_requery_hint_max_per_cue: int = Field(default=2, ge=1)


class CorrectionReuseSettings(BaseModel):
    enabled: bool = True
    min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    min_surface_cjk_len: int = 3
    allowed_categories: list[str] = Field(default_factory=lambda: ["character"])


class Phase2Settings(BaseModel):
    max_per_batch: int = 10
    retry_failed_batch_as_single_cue: bool = True


class LanguageModelSettings(BaseModel):
    model_path: str | None = None
    aliases_path: str | None = None
    character_aliases_path: str | None = None
    asr_aliases_path: str | None = None
    enabled: bool = False
    anomaly_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


class CategoryFilterSettings(BaseModel):
    hot_categories: list[str] = Field(default_factory=list)
    cold_categories: list[str] = Field(default_factory=list)
    decay_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    min_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class AppSettings(BaseModel):
    app_name: str = "subtitle-corrector"
    log_level: str = "INFO"
    memory: MemorySettings = Field(default_factory=MemorySettings)
    entity_memory: EntityMemorySettings = Field(default_factory=EntityMemorySettings)
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)
    matcher: MatcherSettings = Field(default_factory=MatcherSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    resolver: ResolverSettings = Field(default_factory=ResolverSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    candidate_expansion: CandidateExpansionSettings = Field(default_factory=CandidateExpansionSettings)
    correction_reuse: CorrectionReuseSettings = Field(default_factory=CorrectionReuseSettings)
    phase2: Phase2Settings = Field(default_factory=Phase2Settings)
    category_filter: CategoryFilterSettings = Field(default_factory=CategoryFilterSettings)
    language_model: LanguageModelSettings = Field(default_factory=LanguageModelSettings)


def load_settings(path: str | Path | None = None) -> AppSettings:
    if path is None:
        return AppSettings()
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppSettings.model_validate(data)
