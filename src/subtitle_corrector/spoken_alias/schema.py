"""口语别名数据模型。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AliasKind(str, Enum):
    TERM_SHORT_PRIMARY = "term_short_primary"
    TERM_SHORT_SECONDARY = "term_short_secondary"
    TERM_SHORT = "term_short_alias"
    NORMAL_SHORT = "normal_short_alias"
    NORMAL_TITLE = "normal_title_alias"
    AMBIGUOUS = "ambiguous_alias"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_CONTEXT = "needs_context"
    REJECTED = "rejected"


class UsagePolicy(str, Enum):
    CONTEXT_ONLY = "context_only"             # primary: 激活实体 + ASR错听修复目标
    EXACT_CONTEXT_ONLY = "exact_context_only" # secondary: 仅精确匹配时激活实体
    NEEDS_CONTEXT = "needs_context"
    BLOCKED = "blocked"


class AliasCandidate(BaseModel):
    """口语简称候选。"""

    alias_surface: str
    canonical_term: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    evidence_examples: str = ""
    sources: str = ""
    risk_flags: str = ""
    review_status: str = "pending"
    notes: str = ""


class ApprovedAlias(BaseModel):
    """审核后简称（spoken_aliases_approved.csv 每行）。"""

    alias_surface: str
    canonical_term: str
    alias_kind: str
    usage_policy: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    risk_flags: str = ""
    notes: str = ""
