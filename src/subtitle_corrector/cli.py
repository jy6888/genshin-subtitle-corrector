from __future__ import annotations

import csv
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import typer
from loguru import logger

from subtitle_corrector.config import load_settings
from subtitle_corrector.logging import configure_logging
from subtitle_corrector.memory import SQLiteMemory, SQLiteTerminologyStore
from subtitle_corrector.pipeline import build_pipeline
from subtitle_corrector.schemas import Terminology
from subtitle_corrector.subtitle import SubtitleParser

app = typer.Typer(no_args_is_help=True)


def _empty_pipeline_diag() -> dict:
    return {
        "enabled": False,
        "candidates_by_cue": {},
        "decisions_by_cue": {},
        "requery_by_cue": {},
        "gt_term_choices": set(),
    }


@app.command()
def init_db(config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c")) -> None:
    settings = load_settings(config)
    configure_logging(settings.log_level)
    memory = SQLiteMemory(settings.memory.sqlite_path)
    memory.init_schema()
    logger.info("initialized sqlite memory at {}", settings.memory.sqlite_path)


@app.command()
def import_terms(
    csv_path: Path,
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    settings = load_settings(config)
    configure_logging(settings.log_level)
    memory = SQLiteMemory(settings.memory.sqlite_path)
    memory.init_schema()
    store = SQLiteTerminologyStore(memory)

    imported = 0
    skipped = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            term = (row.get("term") or "").strip()
            if not term:
                skipped += 1
                logger.warning("skip row {}: empty term", row_number)
                continue
            trust_text = (row.get("trust_level") or "0.5").strip()
            try:
                trust_level = float(trust_text)
            except ValueError:
                skipped += 1
                logger.warning("skip row {}: invalid trust_level {}", row_number, trust_text)
                continue
            aliases = [
                alias.strip()
                for alias in (row.get("aliases") or "").split("|")
                if alias.strip()
            ]
            parent_entity = (row.get("parent_entity") or "").strip() or None
            category = (row.get("category") or "").strip() or None

            # --- Fix alias-shadowing for food/recipe entries ---
            # In the game data, character-specific dishes use the character
            # name as an alias (e.g. 德波大蛋糕·绚丽型 has alias "爱可菲").
            # This would shadow the actual character term in the matcher.
            # Instead, extract such aliases as parent_entity and remove them
            # from the alias list so they don't pollute the choices dict.
            if category in ("food", "recipe") and aliases and not parent_entity:
                parent_entity = aliases[0]
                aliases = []

            store.upsert(
                Terminology(
                    term=term,
                    aliases=aliases,
                    category=category,
                    game_title=(row.get("game_title") or "").strip() or None,
                    source=(row.get("source") or "").strip() or None,
                    trust_level=trust_level,
                    parent_entity=parent_entity,
                )
            )
            imported += 1
    logger.info("imported {} terms from {}, skipped {}", imported, csv_path, skipped)


@app.command()
def build_guide_corpus(
    source_urls_path: Path = typer.Argument(..., help="白名单 URL 列表文件路径"),
    output: Path = typer.Option(Path("data/guide_corpus"), "--output", "-o", help="输出目录"),
    terminology_csv: Path | None = typer.Option(None, "--terms", "-t", help="术语 CSV 用于实体识别"),
) -> None:
    """从白名单 URL 抓取攻略文章，清洗后输出 JSONL 语料库。"""
    from subtitle_corrector.guide_corpus.pipeline import build_guide_corpus as build

    known_entities: set[str] = set()
    if terminology_csv and terminology_csv.exists():
        import csv as _csv
        with open(terminology_csv, "r", encoding="utf-8-sig", newline="") as f:
            for row in _csv.DictReader(f):
                term = (row.get("term") or "").strip()
                if term:
                    known_entities.add(term)
        logger.info("从 {} 加载 {} 个已知术语", terminology_csv, len(known_entities))

    result = build(source_urls_path, output, known_entities)
    typer.echo(f"攻略语料已构建: {result['articles_count']} 篇文章, {result['sentences_count']} 个句子")
    typer.echo(f"输出目录: {result['output_dir']}")


@app.command()
def mine_spoken_aliases(
    sentences_path: Path = typer.Argument(..., help="sentences.jsonl 路径"),
    output: Path = typer.Option(..., "--output", "-o", help="候选 CSV 输出路径"),
    terminology_csv: Path | None = typer.Option(None, "--terms", "-t", help="术语 CSV 用于 canonical term 匹配"),
) -> None:
    """从攻略句子中挖掘口语简称候选。"""
    from subtitle_corrector.spoken_alias.miner import mine_aliases

    candidates = mine_aliases(sentences_path, terminology_csv, output)
    approved_count = sum(1 for c in candidates if c.confidence >= 0.85 and "ambiguous" not in c.risk_flags)
    typer.echo(f"挖掘完成: {len(candidates)} 个候选")
    typer.echo(f"  高置信度 (>0.85): {approved_count}")
    typer.echo(f"  需人工审核: {len(candidates) - approved_count}")
    typer.echo(f"输出: {output}")


@app.command()
def export_approved_spoken_aliases(
    candidates_path: Path = typer.Argument(..., help="候选 CSV 路径"),
    output: Path = typer.Option(Path("data/spoken_aliases_approved.csv"), "--output", "-o", help="approved CSV 输出路径"),
    auto_threshold: float = typer.Option(0.85, "--threshold", help="自动审批置信度阈值"),
) -> None:
    """从候选 CSV 导出审核后的 approved CSV。"""
    from subtitle_corrector.spoken_alias.review import export_approved_aliases

    summary = export_approved_aliases(candidates_path, output, auto_threshold)
    typer.echo("导出审核后简称:")
    typer.echo(f"  approved:      {summary['approved']}")
    typer.echo(f"  needs_context: {summary['needs_context']}")
    typer.echo(f"  rejected:      {summary['rejected']}")
    typer.echo(f"  pending:       {summary['pending']}")
    typer.echo(f"输出: {summary['output_path']}")


@app.command()
def train_normal_guide_lm(
    sentences_path: Path = typer.Argument(..., help="sentences.jsonl 路径"),
    output: Path = typer.Option(Path("data/normal_guide_lm.json"), "--output", "-o", help="模型输出 JSON 路径"),
    approved_aliases: Path | None = typer.Option(None, "--aliases", "-a", help="审核后简称 CSV 路径"),
) -> None:
    """从攻略句子训练正常攻略话术模型。"""
    from subtitle_corrector.normal_lm.trainer import train_normal_guide_lm as train

    model = train(sentences_path, output, approved_aliases)
    typer.echo(f"训练完成: {len(model.char_ngrams.get(2, {}))} bigrams, "
               f"{len(model.char_ngrams.get(3, {}))} trigrams")
    typer.echo(f"实体上下文: {len(model.context_patterns)}")
    typer.echo(f"简称上下文: {len(model.alias_contexts)}")
    typer.echo(f"模型已保存: {output}")


@app.command()
def inspect(
    subtitle_path: Path,
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    report: Path | None = typer.Option(None, "--report", "-r", help="输出 Markdown 审查报告的路径"),
    trace: bool = typer.Option(False, "--trace", "-t", help="输出全链路可追溯的 Pipeline Trace 报告"),
) -> None:
    settings = load_settings(config)
    configure_logging(settings.log_level)
    document = SubtitleParser().parse_file(subtitle_path)
    total_cues = len(document.cues)
    logger.info("loaded {} cues from {}", total_cues, subtitle_path)

    _run_trace_mode(subtitle_path, document, settings, report)


def _candidate_surface(c: dict) -> str:
    """候选的原文 surface——优先 metadata.surface_text，回退顶层 surface。"""
    return (
        c.get("metadata", {}).get("surface_text", "")
        or c.get("surface", "")
    )


def _should_protect_surface(
    surface: str,
    value: str,
    context_surfaces: set[str],
    surface_to_canonical: dict[str, str],
) -> bool:
    """检查候选是否试图展开受保护的 surface。

    1. 精确匹配：surface 本身在 context_surfaces 中且 value 不同
    2. 子串匹配：surface 包含某个受保护词，且 value 是该词的 canonical
       （捕获 expansion 生成的"是金珀→试作金珀"）
    """
    if not surface or not value or surface == value:
        return False
    # 精确匹配
    if surface in context_surfaces:
        return True
    # 子串匹配：候选 surface 包含受保护词，且 value 是其 canonical 展开
    for protected, canonical in surface_to_canonical.items():
        if len(protected) >= 2 and protected in surface and value == canonical:
            return True
    return False


def _build_protected_spans(
    text: str,
    surface_to_canonical: dict[str, str],
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for surface, canonical in surface_to_canonical.items():
        if len(surface) < 2:
            continue
        pos = 0
        while True:
            pos = text.find(surface, pos)
            if pos == -1:
                break
            spans.append((pos, pos + len(surface), canonical))
            pos += 1
    return spans


def _candidate_hits_protected_span(
    candidate: dict,
    protected_spans: list[tuple[int, int, str]],
) -> bool:
    c_start = candidate.get("start_char", -1)
    c_end = candidate.get("end_char", -1)
    c_value = candidate.get("value", "")
    if c_start < 0 or c_end < 0 or not c_value:
        return False
    for p_start, p_end, canonical in protected_spans:
        if c_start < p_end and c_end > p_start and c_value == canonical:
            return True
    return False


def _run_trace_pipeline(
    cues: list,
    settings,
    trace_rpt: object | None = None,
) -> tuple[list, object | None, dict | None, dict]:
    """运行 Phase 1-6 全链路管线，返回 (repaired_cues, reuse_summary)."""
    reuse_summary = None
    pipeline_diag = _empty_pipeline_diag()
    pipeline_diag["enabled"] = True
    from subtitle_corrector.memory import SQLiteMemory, SQLiteTerminologyRepository
    from subtitle_corrector.memory.entity import EntityMemoryManager
    from subtitle_corrector.memory.category_activation import CategoryActivationManager
    from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher
    from subtitle_corrector.pipeline.discovery import DiscoveryEngine
    from subtitle_corrector.resolver.llm import OpenAIArbitrator
    from subtitle_corrector.utils.chunker import SubtitleChunker

    chunker = SubtitleChunker(chunk_size=100, overlap_size=15)
    chunk_result = chunker.chunk(cues)

    entity_mem = EntityMemoryManager(
        decay_rate=settings.entity_memory.decay_rate,
        min_weight=settings.entity_memory.min_weight,
    )
    cat_act = CategoryActivationManager(
        decay_rate=settings.category_filter.decay_rate,
        min_weight=settings.category_filter.min_weight,
    )
    repo = SQLiteTerminologyRepository(SQLiteMemory(settings.memory.sqlite_path))
    matcher = FuzzyTerminologyMatcher(
        repo, threshold=settings.matcher.fuzzy_threshold,
        pinyin_threshold=settings.matcher.pinyin_threshold,
        boost_factor=settings.entity_memory.boost_factor,
    )
    matcher.entity_memory = entity_mem

    # ── 创建别名词典（始终加载）──
    alias_lexicon = None
    character_lexicon = None
    if settings.language_model.aliases_path:
        from subtitle_corrector.spoken_alias.lexicon import SpokenAliasLexicon
        alias_lexicon = SpokenAliasLexicon(settings.language_model.aliases_path)
    if settings.language_model.character_aliases_path:
        from subtitle_corrector.character_alias.lexicon import CharacterAliasLexicon
        character_lexicon = CharacterAliasLexicon(settings.language_model.character_aliases_path)
    asr_alias_runtime = None
    if settings.language_model.asr_aliases_path:
        from subtitle_corrector.aliasing.runtime import AsrAliasRuntime
        asr_alias_runtime = AsrAliasRuntime.from_csv(settings.language_model.asr_aliases_path)
    from subtitle_corrector.aliasing.policy import AliasPolicyRegistry
    alias_policy_registry = AliasPolicyRegistry.from_sources(
        spoken_alias_lexicon=alias_lexicon,
        character_lexicon=character_lexicon,
        asr_alias_runtime=asr_alias_runtime,
    )
    team_parser = None
    if character_lexicon is not None:
        from subtitle_corrector.character_alias.team_comp import TeamCompParser
        team_parser = TeamCompParser(character_lexicon)

    # ── ASR 错听硬修复：Q1/eq → QE ──
    import re as _re
    for cue in cues:
        cue.text = _re.sub(r"Q1", "QE", cue.text, flags=_re.IGNORECASE)
        cue.text = _re.sub(r"(?<![a-zA-Z])eq(?![a-zA-Z])", "QE", cue.text, flags=_re.IGNORECASE)

    # ── 逐 cue 实体预激活（术语别名 + 角色外号 + 配队单字）──
    if alias_lexicon is not None or character_lexicon is not None:
        choices = matcher._build_choices()
        for cue in cues:
            entity_mem.decay()
            if alias_lexicon is not None:
                for match in alias_lexicon.find_surfaces_in_text(cue.text):
                    canonical = match["canonical"]
                    if canonical in choices:
                        entity_mem.update_entity(canonical)
            if character_lexicon is not None:
                for entry in character_lexicon.find_nicknames_in_text(cue.text):
                    if entry.canonical_term in choices:
                        entity_mem.update_entity(entry.canonical_term)
            if team_parser is not None:
                for _alias, canonical in team_parser.parse(cue.text):
                    if canonical in choices:
                        entity_mem.update_entity(canonical)

    arbitrator = OpenAIArbitrator(settings.resolver)
    engine = DiscoveryEngine(
        arbitrator.client, matcher, entity_mem, cat_act,
        hot_categories=set(settings.category_filter.hot_categories),
        cold_categories=set(settings.category_filter.cold_categories),
        chunk_size=100, overlap_size=15, max_workers=4,
        max_candidates_per_cue=settings.candidate_expansion.discovery_max_candidates_per_cue,
        model=settings.resolver.discovery_model or settings.resolver.model,
        alias_lexicon=alias_lexicon,
        character_lexicon=character_lexicon,
    )
    result = engine.run(cues)

    # ── context_alias 过滤：已知简称/外号只激活实体，不作修复候选 ──
    # 收集所有 context_only / exact_context_only 的 alias surface（术语+角色）
    context_surfaces: set[str] = set()
    if alias_lexicon is not None:
        for surface in alias_lexicon.surface_to_canonical:
            policy = alias_lexicon.get_policy(surface)
            if policy in ("context_only", "exact_context_only"):
                context_surfaces.add(surface)
    if character_lexicon is not None:
        for surface in character_lexicon.team_slots:
            context_surfaces.add(surface)  # 单字简称需配队语境，不进候选
        for surface in character_lexicon.nicknames:
            context_surfaces.add(surface)  # 多字外号保护原文不展开，但可作为修复目标

    # 构建反向索引：context_only surface → canonical
    # 用于检测 expansion 生成的包含保护词的候选（如"是金珀→试作金珀"）
    context_surface_to_canonical: dict[str, str] = {}
    if alias_lexicon is not None:
        for surface, canonical in alias_lexicon.surface_to_canonical.items():
            policy = alias_lexicon.get_policy(surface)
            if policy in ("context_only", "exact_context_only"):
                context_surface_to_canonical[surface] = canonical

    for tl in result.timelines:
        for obs in tl.observations:
            for candidate in obs.candidates:
                surface = (
                    candidate.get("surface")
                    or candidate.get("metadata", {}).get("surface_text", "")
                )
                target = candidate.get("value", "")
                if surface and "expansion_policy" not in candidate:
                    decision = alias_policy_registry.resolve(surface, target=target)
                    candidate["expansion_policy"] = decision.policy.value
                    candidate["evidence_type"] = decision.source
                    candidate.setdefault("metadata", {})["alias_policy_reason"] = (
                        decision.reason
                    )
            protected_spans = _build_protected_spans(
                obs.text, context_surface_to_canonical,
            )
            obs.candidates = [
                c for c in obs.candidates
                if c.get("metadata", {}).get("intent") != "context_alias"
                and not _should_protect_surface(
                    _candidate_surface(c), c.get("value", ""),
                    context_surfaces, context_surface_to_canonical,
                )
                and not _candidate_hits_protected_span(c, protected_spans)
            ]

    if trace_rpt is not None:
        trace_rpt.record(
            "SubtitleChunker",
            f"{len(cues)} 条字幕",
            f"{len(chunk_result.chunks)} 个 Chunk (size=chunker.chunk_size, overlap={chunker.overlap_size})",
            [f"Chunk {c.chunk_id}: [{c.start_index}:{c.end_index}] ({len(c.target_lines)} cues)" for c in chunk_result.chunks],
        )
        trace_rpt.record(
            "DiscoveryEngine",
            f"{len(chunk_result.chunks)} 个 Chunk → 并发 LLM 调用",
            f"{len(result.timelines)} 个 ChunkSemanticTimeline, {len(result.filter_outputs)} 个 SemanticFilterOutput",
        )
        for tl, fo in zip(result.timelines, result.filter_outputs):
            if fo is None:
                continue
            ents = {e.get("entity",""): e.get("confidence",0) for e in fo.confirmed_entities}
            trace_rpt.record_chunk(
                tl.chunk_index, len(tl.observations),
                ents, fo.dominant_categories,
                fo.possible_transition, len(fo.detector_noise),
            )
        active = entity_mem.active_entities
        trace_rpt.record(
            "EntityMemory",
            "DiscoveryEngine 写入 confirmed_entities",
            f"活跃实体: {active}" if active else "无活跃实体（首次运行或未确认任何实体）",
        )

    if not result.filter_outputs:
        logger.info("DiscoveryEngine 未产出 filter outputs，管线终止（无可修内容）")
        return list(cues), reuse_summary, None, pipeline_diag

    from subtitle_corrector.pipeline.reducer import ConsensusReducer
    reducer = ConsensusReducer()
    clusters = reducer.reduce_from_timelines(result.timelines, result.filter_outputs)

    from subtitle_corrector.pipeline.candidate_expansion import (
        CandidateExpansionEngine,
        EntityConsistencyCandidateExpander,
        LocalTermCandidateExpander,
        LongEntityVariantExpander,
    )
    expanders: list = []
    if settings.candidate_expansion.enable_entity_consistency:
        expanders.append(EntityConsistencyCandidateExpander(matcher))
    if settings.candidate_expansion.enable_long_entity_variant:
        expanders.append(LongEntityVariantExpander(matcher))
    if settings.candidate_expansion.enable_local_term:
        expanders.append(
            LocalTermCandidateExpander(
                matcher,
                requires_stable_entity=settings.candidate_expansion.local_term_requires_stable_entity,
            )
        )
    expansion = CandidateExpansionEngine(
        expanders,
        max_candidates_per_cue=settings.candidate_expansion.max_candidates_per_cue,
    )
    expanded_candidates = expansion.expand(result.timelines, clusters)
    for timeline in result.timelines:
        for observation in timeline.observations:
            pipeline_diag["candidates_by_cue"][observation.cue_index] = list(observation.candidates)

    # ── expansion 后再次过滤 context_surfaces ──
    # expansion 可能为已保护的 surface 生成新候选（如 金珀→试作金珀），必须再次拦截
    # expansion 候选的 surface 在顶层字段，不在 metadata.surface_text
    for tl in result.timelines:
        for obs in tl.observations:
            protected_spans = _build_protected_spans(
                obs.text, context_surface_to_canonical,
            )
            obs.candidates = [
                c for c in obs.candidates
                if not _should_protect_surface(
                    _candidate_surface(c), c.get("value", ""),
                    context_surfaces, context_surface_to_canonical,
                )
                and not _candidate_hits_protected_span(c, protected_spans)
            ]

    from subtitle_corrector.pipeline.refinement import Phase2RefinementEngine
    from subtitle_corrector.pipeline.terminology_hints import TerminologyRequeryHintBuilder
    terminology_hint_builder = TerminologyRequeryHintBuilder(
        matcher,
        min_alignment=settings.candidate_expansion.terminology_requery_hint_min_alignment,
        max_hints=settings.candidate_expansion.terminology_requery_hint_max_per_cue,
    )

    from subtitle_corrector.knowledge.cards import KnowledgeCardBuilder
    knowledge_card_builder = KnowledgeCardBuilder(
        matcher=matcher,
        policy_registry=alias_policy_registry,
        character_lexicon=character_lexicon,
        spoken_lexicon=alias_lexicon,
        asr_runtime=asr_alias_runtime,
    )

    phase2 = Phase2RefinementEngine(
        arbitrator.client,
        max_per_batch=settings.phase2.max_per_batch,
        model=settings.resolver.phase2_model or settings.resolver.model,
        retry_failed_batch_as_single_cue=settings.phase2.retry_failed_batch_as_single_cue,
        asr_alias_runtime=asr_alias_runtime,
        team_comp_parser=team_parser,
        terminology_hint_builder=terminology_hint_builder,
        knowledge_card_builder=knowledge_card_builder,
        protected_surface_to_canonical=context_surface_to_canonical,
    )
    p2_result = phase2.refine(result.timelines, clusters, cues, matcher=matcher)

    # Correction reuse: learn confirmed rules and apply to other cues
    reuse_summary = None
    if settings.correction_reuse.enabled:
        from subtitle_corrector.memory.correction_reuse import (
            augment_decisions_with_reuse,
            learn_confirmed_rules,
        )
        reuse_mem = learn_confirmed_rules(p2_result.decisions, matcher, settings)
        p2_result.decisions, reuse_summary = augment_decisions_with_reuse(
            cues, p2_result.decisions, reuse_mem,
        )
        # Sync Phase2Result counters with added reuse decisions
        p2_result.replaced += reuse_summary.applied_decisions
        p2_result.total_candidates += reuse_summary.applied_decisions

    decisions_by_cue: dict[int, list] = {}
    requery_by_cue: dict[int, list] = {}
    for decision in p2_result.decisions:
        decisions_by_cue.setdefault(decision.cue_index, []).append(decision)
        action_value = getattr(decision.action, "value", decision.action)
        if action_value == "REQUERY":
            requery_by_cue.setdefault(decision.cue_index, []).append(decision)
    for request in getattr(p2_result, "requery_requests", []):
        requery_by_cue.setdefault(request.cue_index, []).append(request)
    pipeline_diag["decisions_by_cue"] = decisions_by_cue
    pipeline_diag["requery_by_cue"] = requery_by_cue

    from subtitle_corrector.pipeline.commit import commit
    repaired_cues = commit(p2_result.decisions, cues)

    if trace_rpt is not None:
        trace_rpt.record(
            "ConsensusReducer",
            f"{len(result.timelines)} timelines + {len(result.filter_outputs)} filter outputs",
            f"{len(clusters)} 个 SemanticCluster",
            [f"cluster_{i}: entities={c.dominant_entities}, categories={c.categories}, range={c.temporal_range}"
             for i, c in enumerate(clusters)],
        )
        trace_rpt.record(
            "CandidateExpansion",
            f"{len(clusters)} 个 SemanticCluster + {len(result.timelines)} timelines",
            f"补充 {expanded_candidates} 个 Phase2 候选",
        )
        trace_rpt.record(
            "Phase2 Refinement",
            f"{p2_result.total_candidates} NEEDS_REVIEW candidates (按 cluster 分组批处理)",
            f"{p2_result.replaced} REPLACE / {p2_result.kept} KEEP / {p2_result.reviewed} REVIEW",
        )
        if p2_result.requery_requested:
            requery_rows: list[list[str]] = []
            for key, value in sorted(p2_result.requery_generated_by_type.items()):
                requery_rows.append([key, str(value), "generated"])
            for key, value in sorted(p2_result.requery_skipped_by_type.items()):
                requery_rows.append([key, str(value), "skipped"])
            if not requery_rows:
                requery_rows.append(["none", "0", "no second-round candidates"])
            trace_rpt.record_table(
                "REQUERY Type Stats",
                ["type", "count", "status"],
                requery_rows,
            )
        if reuse_summary is not None:
            trace_rpt.record(
                "Correction Reuse",
                f"learned={reuse_summary.learned_rules} rules, "
                f"applied={reuse_summary.applied_decisions} decisions",
                f"conflict={reuse_summary.conflict_rules}, "
                f"skipped_low={reuse_summary.skipped_low_confidence}, "
                f"skipped_short={reuse_summary.skipped_short_surface}",
            )
        trace_rpt.record(
            "Final Commit",
            f"{p2_result.replaced} 个修复决策",
            f"{p2_result.replaced} REPLACE 已应用",
        )

    requery_stats = None
    if reuse_summary is not None or p2_result.requery_requested > 0:
        requery_stats = {
            "requested": p2_result.requery_requested,
            "candidates": p2_result.requery_candidates,
            "replaced": p2_result.requery_replaced,
            "skipped": p2_result.requery_skipped,
        }
    return repaired_cues, reuse_summary, requery_stats, pipeline_diag


def _run_trace_mode(
    subtitle_path: Path,
    document,
    settings,
    report: Path | None,
) -> None:
    """全链路可追溯模式 — 记录每个模块的输入/输出并输出 SRT."""
    from datetime import datetime

    from subtitle_corrector.report.trace import TraceReport

    cues = document.cues
    trace_rpt = TraceReport(str(subtitle_path))
    trace_rpt.record(
        "SubtitleParser",
        f"文件 `{subtitle_path.name}`",
        f"{len(cues)} 条字幕",
    )

    repaired_cues, _, _, _ = _run_trace_pipeline(cues, settings, trace_rpt)

    # Write repaired SRT
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_srt = Path(f"reports/{subtitle_path.stem}_repaired_{ts}.srt")
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_srt, "w", encoding="utf-8") as f:
        for cue in repaired_cues:
            s_h = cue.start_ms // 3600000
            s_m = (cue.start_ms % 3600000) // 60000
            s_s = (cue.start_ms % 60000) // 1000
            s_ms = cue.start_ms % 1000
            e_h = cue.end_ms // 3600000
            e_m = (cue.end_ms % 3600000) // 60000
            e_s = (cue.end_ms % 60000) // 1000
            e_ms = cue.end_ms % 1000
            f.write(f"{cue.index+1}\n{s_h:02d}:{s_m:02d}:{s_s:02d},{s_ms:03d} --> {e_h:02d}:{e_m:02d}:{e_s:02d},{e_ms:03d}\n{cue.text}\n\n")
    logger.info("修复后字幕已保存: {}", out_srt)

    if report is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = Path(f"reports/{subtitle_path.stem}_trace_{ts}.md")
    trace_rpt.generate(report)
    logger.info("Pipeline Trace 报告已保存: {}", report)


@app.command()
def evaluate(
    subtitle_path: Path,
    gt_path: Path = typer.Option(..., "--gt", "-g", help="Ground truth SRT 文件"),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    report: Path | None = typer.Option(None, "--report", "-r", help="输出评估报告的路径"),
    trace: bool = typer.Option(False, "--trace", "-t", help="使用 Phase 1-6 新管线（并发分块）"),
) -> None:
    """运行纠错管线并对照 Ground Truth 计算 F1/准确率/召回率."""
    from datetime import datetime

    from subtitle_corrector.evaluation.metrics import evaluate as evaluate_metrics
    from subtitle_corrector.subtitle import SubtitleParser

    settings = load_settings(config)
    configure_logging(settings.log_level)

    raw_doc = SubtitleParser().parse_file(subtitle_path)
    gt_doc = SubtitleParser().parse_file(gt_path)
    assert len(raw_doc.cues) == len(gt_doc.cues), (
        f"字幕条数不一致: raw={len(raw_doc.cues)} vs gt={len(gt_doc.cues)}"
    )

    logger.info("loaded {} cues from {}, {} cues from GT", len(raw_doc.cues), subtitle_path, len(gt_doc.cues))

    repaired_cues, reuse_summary, requery_stats, pipeline_diag = _run_trace_pipeline(raw_doc.cues, settings)
    pipeline_diag["gt_term_choices"] = _build_gt_term_choices(settings)

    # Evaluate against GT (代词/语气词/无术语差异不计入主分数)
    from subtitle_corrector.evaluation.fn_attribution import (
        _is_non_terminology_diff,
        build_attribution_report,
    )
    eval_report = evaluate_metrics(
        raw_doc.cues, repaired_cues, gt_doc.cues,
        skip_fn=_is_non_terminology_diff,
    )

    fn_cues_all = eval_report.missed_errors + eval_report.wrong_fixes
    fn_attr = build_attribution_report(fn_cues_all, {}, {}, set())

    # Terminal summary
    typer.echo(f"\n{'='*50}")
    typer.echo(f"  评估结果: {subtitle_path.name}")
    typer.echo(f"{'='*50}")
    typer.echo(f"  总字幕:    {eval_report.total_cues}")
    typer.echo(f"  含错字幕:  {eval_report.error_cues}")
    if eval_report.chat_diffs:
        typer.echo(f"  非术语差异: {len(eval_report.chat_diffs)} (已从分数排除)")
    typer.echo(f"  TP: {eval_report.tp}  FP: {eval_report.fp}  TN: {eval_report.tn}  FN: {eval_report.fn}")
    typer.echo(f"  Precision: {eval_report.precision:.1%}")
    typer.echo(f"  Recall:    {eval_report.recall:.1%}")
    typer.echo(f"  F1:        {eval_report.f1:.1%}")
    typer.echo(f"  Accuracy:  {eval_report.accuracy:.1%}")
    typer.echo(f"  修复率:    {eval_report.fix_rate:.1%} ({eval_report.tp}/{eval_report.error_cues})")
    typer.echo(f"  误修率:    {eval_report.false_positive_rate:.1%} ({eval_report.fp}/{eval_report.tp+eval_report.fp})")
    typer.echo(f"{'='*50}\n")

    if eval_report.wrong_fixes:
        typer.echo(f"错误修改 ({len(eval_report.wrong_fixes)}):")
        for wf in eval_report.wrong_fixes[:10]:
            typer.echo(f"  #{wf.cue_index}: raw={wf.raw_text[:40]} → repaired={wf.repaired_text[:40]} (GT={wf.gt_text[:40]})")

    if eval_report.missed_errors:
        typer.echo(f"\n漏修 ({len(eval_report.missed_errors)}):")
        for me in eval_report.missed_errors[:10]:
            typer.echo(f"  #{me.cue_index}: raw={me.raw_text[:40]} (GT={me.gt_text[:40]})")

    # Write report
    if report is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = Path(f"reports/{subtitle_path.stem}_eval_{ts}.md")
    _write_eval_report(eval_report, subtitle_path.name, str(gt_path), report,
                       reuse_summary, requery_stats, pipeline_diag, fn_attr)
    logger.info("评估报告已保存: {}", report)


@app.command()
def diagnose_candidates(
    subtitle_path: Path,
    gt_path: Path = typer.Option(..., "--gt", "-g", help="Ground truth SRT 文件"),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    report: Path | None = typer.Option(None, "--report", "-r", help="输出诊断报告的路径"),
) -> None:
    """非 LLM 候选诊断：检测器→Discovery→CandidateExpansion 各阶段候选统计."""
    from collections import Counter
    from datetime import datetime

    from subtitle_corrector.subtitle import SubtitleParser

    settings = load_settings(config)
    configure_logging(settings.log_level)

    raw_doc = SubtitleParser().parse_file(subtitle_path)
    gt_doc = SubtitleParser().parse_file(gt_path)
    assert len(raw_doc.cues) == len(gt_doc.cues), (
        f"字幕条数不一致: raw={len(raw_doc.cues)} vs gt={len(gt_doc.cues)}"
    )
    logger.info("loaded {} cues from {}, {} cues from GT", len(raw_doc.cues), subtitle_path, len(gt_doc.cues))

    cues = raw_doc.cues

    # --- Stage 1: Detector only ---
    from subtitle_corrector.memory import SQLiteMemory, SQLiteTerminologyRepository
    from subtitle_corrector.memory.entity import EntityMemoryManager
    from subtitle_corrector.memory.category_activation import CategoryActivationManager
    from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher
    from subtitle_corrector.pipeline.discovery import DiscoveryEngine
    from subtitle_corrector.resolver.llm import OpenAIArbitrator

    entity_mem = EntityMemoryManager(decay_rate=0.8, min_weight=0.2)
    cat_act = CategoryActivationManager(decay_rate=0.7, min_weight=0.3)
    repo = SQLiteTerminologyRepository(SQLiteMemory(settings.memory.sqlite_path))
    matcher = FuzzyTerminologyMatcher(
        repo, threshold=settings.matcher.fuzzy_threshold,
        pinyin_threshold=settings.matcher.pinyin_threshold,
        boost_factor=settings.entity_memory.boost_factor,
    )
    matcher.entity_memory = entity_mem

    arbitrator = OpenAIArbitrator(settings.resolver)
    engine = DiscoveryEngine(
        arbitrator.client, matcher, entity_mem, cat_act,
        hot_categories=set(settings.category_filter.hot_categories),
        cold_categories=set(settings.category_filter.cold_categories),
        chunk_size=100, overlap_size=15, max_workers=1,
        max_candidates_per_cue=settings.candidate_expansion.discovery_max_candidates_per_cue,
    )

    detector_candidates = 0
    detector_by_source: Counter = Counter()
    for cue in cues:
        cands = engine._detect_candidates(cue)
        detector_candidates += len(cands)
        for c in cands:
            detector_by_source[c.get("source", "unknown")] += 1

    # --- Stage 2: Build timelines from detector only (no LLM) ---
    from subtitle_corrector.utils.chunker import SubtitleChunker
    from subtitle_corrector.schemas import CueObservation, ChunkSemanticTimeline
    chunker = SubtitleChunker(chunk_size=100, overlap_size=15)
    chunk_result = chunker.chunk(cues)
    discovery_candidates = 0
    discovery_by_source: Counter = Counter()
    timelines: list = []
    for chunk in chunk_result.chunks:
        observations: list = []
        for cue in chunk.target_lines:
            cands = engine._detect_candidates(cue)
            discovery_candidates += len(cands)
            for c in cands:
                discovery_by_source[c.get("source", "unknown")] += 1
            observations.append(CueObservation(
                cue_index=cue.index, text=cue.text,
                context_before="", context_after="",
                candidates=cands,
                active_entities={}, active_categories=[],
            ))
        timelines.append(ChunkSemanticTimeline(
            chunk_index=chunk.chunk_id,
            observations=observations,
            entity_persistence={},
            category_flow=[],
            compressed_context_windows=[],
        ))

    # --- Stage 3: Empty clustering (no LLM, entities unknown) ---
    clusters: list = []

    # --- Stage 4: CandidateExpansion ---
    from subtitle_corrector.pipeline.candidate_expansion import (
        CandidateExpansionEngine,
        EntityConsistencyCandidateExpander,
        LocalTermCandidateExpander,
        LongEntityVariantExpander,
    )
    expanders: list = []
    if settings.candidate_expansion.enable_entity_consistency:
        expanders.append(EntityConsistencyCandidateExpander(matcher))
    if settings.candidate_expansion.enable_long_entity_variant:
        expanders.append(LongEntityVariantExpander(matcher))
    if settings.candidate_expansion.enable_local_term:
        expanders.append(
            LocalTermCandidateExpander(
                matcher,
                requires_stable_entity=settings.candidate_expansion.local_term_requires_stable_entity,
            )
        )
    expansion = CandidateExpansionEngine(
        expanders,
        max_candidates_per_cue=settings.candidate_expansion.max_candidates_per_cue,
    )
    expansion.expand(timelines, clusters)
    exp_summary = expansion.last_summary

    # --- Stage 5: GT term coverage ---
    gt_terms: dict[str, int] = Counter()
    for raw_cue, gt_cue in zip(raw_doc.cues, gt_doc.cues):
        if raw_cue.text != gt_cue.text:
            for term_candidate in _gt_term_diffs(raw_cue.text, gt_cue.text):
                gt_terms[term_candidate] += 1

    # Collect all Phase2 candidates per cue
    all_candidates_by_cue: dict[int, list[dict]] = {}
    for tl in timelines:
        for obs in tl.observations:
            if obs.candidates:
                all_candidates_by_cue[obs.cue_index] = obs.candidates

    policy_counts: Counter = Counter()
    for cands in all_candidates_by_cue.values():
        for candidate in cands:
            policy = candidate.get(
                "expansion_policy",
                candidate.get("metadata", {}).get("expansion_policy", "unknown"),
            )
            policy_counts[policy or "unknown"] += 1

    # GT term coverage
    gt_covered: dict[str, int] = {}
    gt_missed: dict[str, int] = {}
    for term, total in gt_terms.most_common():
        covered = 0
        for cue_idx, cands in all_candidates_by_cue.items():
            if any(c.get("value") == term for c in cands):
                covered += 1
        gt_covered[term] = covered
        gt_missed[term] = total - covered

    # --- Output ---
    typer.echo(f"\n{'='*60}")
    typer.echo(f"  候选诊断: {subtitle_path.name}")
    typer.echo(f"{'='*60}")
    typer.echo(f"  总字幕: {len(cues)}")
    typer.echo(f"  GT 差异术语种类: {len(gt_terms)}")
    typer.echo("")
    typer.echo("  各阶段候选数:")
    typer.echo(f"    Detector:      {detector_candidates:>5} candidates")
    typer.echo(f"    Discovery:     {discovery_candidates:>5} candidates (per-cue, no LLM)")
    typer.echo(f"    Expansion raw: {exp_summary.raw_proposals:>5} proposals")
    typer.echo(f"    Expansion add: {exp_summary.added_candidates:>5} added")
    typer.echo(f"    Phase2 input:  {exp_summary.output_candidates:>5} candidates (after cap)")
    typer.echo("")
    typer.echo("  按 source 统计 (Detector):")
    for src, cnt in detector_by_source.most_common():
        typer.echo(f"    {src:>20s}: {cnt:>5}")
    typer.echo("")
    typer.echo("  按 source 统计 (Expansion added):")
    for src, cnt in sorted(exp_summary.added_by_source.items()):
        typer.echo(f"    {src:>20s}: {cnt:>5}")
    typer.echo("")
    typer.echo("  expansion_policy (Phase2 input):")
    for policy, cnt in policy_counts.most_common():
        typer.echo(f"    {policy:>20s}: {cnt:>5}")
    typer.echo("")
    typer.echo("  GT 术语候选覆盖率:")
    total_gt = sum(gt_terms.values())
    total_covered = sum(gt_covered.values())
    typer.echo(f"    总体: {total_covered}/{total_gt} ({total_covered/total_gt*100:.1f}%)" if total_gt else "    总体: N/A")
    typer.echo("")
    typer.echo("  Top 漏检术语:")
    for term, missed in sorted(gt_missed.items(), key=lambda x: -x[1])[:10]:
        total = gt_terms[term]
        typer.echo(f"    {term}: {gt_covered[term]}/{total} (miss={missed})")
    typer.echo(f"{'='*60}\n")

    # --- Write report ---
    if report is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = Path(f"reports/{subtitle_path.stem}_diag_{ts}.md")
    _write_diag_report(
        report, subtitle_path.name, str(gt_path),
        len(cues), detector_candidates, discovery_candidates, exp_summary,
        detector_by_source, gt_terms, gt_covered, gt_missed, policy_counts,
    )
    logger.info("诊断报告已保存: {}", report)


@app.command()
def populate_aliases(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    categories: str = typer.Option("artifact,artifact_piece,weapon", "--categories", help="目标类别（逗号分隔）"),
    min_term_length: int = typer.Option(4, "--min-length", help="最短术语长度"),
    dry_run: bool = typer.Option(True, "--dry-run/--write", help="预览/写入"),
) -> None:
    """从 terminology 表自动生成术语内部简称，写入 terminology_alias。"""
    from subtitle_corrector.aliasing.populator import populate_aliases as populate

    cat_list = tuple(c.strip() for c in categories.split(",") if c.strip())
    settings = load_settings(config)
    result = populate(settings.memory.sqlite_path, cat_list, min_term_length, dry_run)

    typer.echo(f"\n类别: {', '.join(cat_list)}")
    typer.echo(f"生成: {result['generated']} 条别名")
    typer.echo(f"跳过(重复): {result['skipped_duplicate']}")
    typer.echo(f"跳过(过短): {result['skipped_short']}")
    for cat, cnt in sorted(result["by_category"].items()):
        typer.echo(f"  {cat}: {cnt}")
    if dry_run:
        typer.echo("\n[dry-run] 未写入数据库。使用 --write 参数写入。")


@app.command()
def approve_aliases(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    """批量审批所有 draft 别名为正式别名。"""
    from subtitle_corrector.aliasing.populator import approve_draft_aliases

    settings = load_settings(config)
    count = approve_draft_aliases(settings.memory.sqlite_path)
    typer.echo(f"已审批 {count} 条 term_short_draft → term_short")


@app.command()
def list_draft_aliases(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    limit: int = typer.Option(50, "--limit", "-n"),
) -> None:
    """列出所有 draft 别名供人工审核。"""
    from subtitle_corrector.aliasing.populator import list_draft_aliases

    settings = load_settings(config)
    rows = list_draft_aliases(settings.memory.sqlite_path, limit)
    if not rows:
        typer.echo("没有待审核的 draft 别名。")
        return
    typer.echo(f"{'ID':<6} {'类别':<16} {'术语':<24} {'简称'}")
    typer.echo("-" * 60)
    for r in rows:
        typer.echo(f"{r['id']:<6} {r['category']:<16} {r['term']:<24} {r['alias']}")


def _gt_term_diffs(raw_text: str, gt_text: str) -> list[str]:
    """Extract terminology candidates that differ between raw and GT."""
    from subtitle_corrector.memory import SQLiteMemory, SQLiteTerminologyRepository
    from subtitle_corrector.config import load_settings
    from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher
    from pathlib import Path

    # Check if GT has terms from the terminology DB that raw doesn't
    settings = load_settings(Path("configs/default.yaml"))
    repo = SQLiteTerminologyRepository(SQLiteMemory(settings.memory.sqlite_path))
    matcher = FuzzyTerminologyMatcher(repo)
    choices = matcher._build_choices()

    diffs: list[str] = []
    for term in choices:
        if term in gt_text and term not in raw_text:
            diffs.append(term)
    return diffs


def _build_gt_term_choices(settings) -> set[str]:
    from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher
    from subtitle_corrector.memory import SQLiteMemory, SQLiteTerminologyRepository

    repo = SQLiteTerminologyRepository(SQLiteMemory(settings.memory.sqlite_path))
    matcher = FuzzyTerminologyMatcher(repo)
    return set(matcher._build_choices())


def _extract_gt_targets_from_choices(
    raw_text: str,
    gt_text: str,
    gt_term_choices: set[str] | None,
) -> list[str]:
    if not gt_term_choices:
        return []
    return [
        term
        for term in sorted(gt_term_choices, key=lambda item: (gt_text.find(item), item))
        if term and term in gt_text and term not in raw_text
    ]


def _fallback_gt_targets(raw_text: str, gt_text: str) -> list[str]:
    targets: list[str] = []
    matcher = SequenceMatcher(a=raw_text, b=gt_text, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "insert"}:
            target = gt_text[j1:j2].strip()
            if target:
                targets.append(target)
    return targets


def _candidate_value(candidate: object) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("value") or candidate.get("corrected_text") or "")
    return str(getattr(candidate, "value", "") or getattr(candidate, "corrected_text", ""))


def _simple_diff_terms(raw: str, gt: str) -> list[str]:
    """简单地从 raw vs gt 中提取不同的 CJK 片段作为 GT terms."""
    terms: list[str] = []
    for i, ch in enumerate(gt):
        if i < len(raw) and ch != raw[i]:
            # 向前扩展取完整 CJK 词
            start = max(0, i - 1)
            while start > 0 and "一" <= gt[start] <= "鿿":
                start -= 1
            end = min(len(gt), i + 3)
            while end < len(gt) and "一" <= gt[end] <= "鿿":
                end += 1
            fragment = gt[start:end].strip()
            if len(fragment) >= 2 and fragment not in terms:
                terms.append(fragment)
    return terms


def _decision_action(decision: object) -> str:
    action = getattr(decision, "action", "")
    return str(getattr(action, "value", action)).upper()


def _decision_corrected_text(decision: object) -> str:
    if isinstance(decision, dict):
        return str(decision.get("corrected_text") or decision.get("value") or "")
    return str(getattr(decision, "corrected_text", "") or getattr(decision, "value", ""))


def _decision_surface_text(decision: object) -> str:
    if isinstance(decision, dict):
        return str(decision.get("surface_text") or decision.get("surface") or "")
    return str(getattr(decision, "surface_text", "") or getattr(decision, "surface", ""))


def _decision_metadata(decision: object) -> dict:
    if isinstance(decision, dict):
        return dict(decision.get("metadata") or {})
    return dict(getattr(decision, "metadata", {}) or {})


def _format_decision_diagnostics(decisions: list, requery_items: list) -> str:
    parts: list[str] = []
    for decision in decisions:
        action = _decision_action(decision)
        surface = _decision_surface_text(decision)
        corrected = _decision_corrected_text(decision)
        metadata = _decision_metadata(decision)
        candidate_metadata = metadata.get("candidate_metadata") or {}
        if not isinstance(candidate_metadata, dict):
            candidate_metadata = {}
        fields = [
            f"source={metadata.get('candidate_source', '')}",
            f"category={metadata.get('candidate_category', '')}",
            f"evidence={metadata.get('evidence_type', '')}",
            f"policy={metadata.get('expansion_policy', '')}",
            f"match={candidate_metadata.get('match_kind', '')}",
        ]
        if metadata.get("guard_blocked"):
            fields.append(f"guard={metadata['guard_blocked']}")
        compact = "; ".join(field for field in fields if not field.endswith("="))
        parts.append(f"{action}:{surface}->{corrected} ({compact})")

    for item in requery_items:
        if isinstance(item, dict):
            surface = item.get("suspect_surface", "")
            target = item.get("target_hint", "")
            metadata = item.get("metadata") or {}
        else:
            surface = getattr(item, "suspect_surface", "")
            target = getattr(item, "target_hint", "")
            metadata = getattr(item, "metadata", {}) or {}
        requery_type = metadata.get("requery_type", "")
        parts.append(f"REQUERY:{surface}->{target} (type={requery_type})")

    return "<br>".join(parts) if parts else "-"


def _attribute_fn_cue(
    cue_index: int,
    raw_text: str,
    gt_text: str,
    candidates_by_cue: dict,
    decisions_by_cue: dict,
    requery_by_cue: dict,
    diag_enabled: bool = True,
    gt_term_choices: set[str] | None = None,
) -> dict:
    gt_targets = _extract_gt_targets_from_choices(raw_text, gt_text, gt_term_choices)
    if not diag_enabled:
        if not gt_targets:
            gt_targets = _fallback_gt_targets(raw_text, gt_text)
        return {
            "cue_index": cue_index,
            "bucket": "NO_PIPELINE_DIAG",
            "gt_targets": gt_targets,
            "candidate_values": [],
            "raw_text": raw_text,
            "gt_text": gt_text,
        }

    if not gt_targets and gt_term_choices is None:
        try:
            gt_targets = _gt_term_diffs(raw_text, gt_text)
        except Exception as exc:
            logger.debug("GT term diff lookup failed for cue {}: {}", cue_index, exc)
            gt_targets = []
    if not gt_targets:
        gt_targets = _fallback_gt_targets(raw_text, gt_text)

    candidates = list(candidates_by_cue.get(cue_index, []))
    decisions = list(decisions_by_cue.get(cue_index, []))
    requery_items = list(requery_by_cue.get(cue_index, []))
    candidate_values = [_candidate_value(candidate) for candidate in candidates]
    candidate_values = [value for value in candidate_values if value]

    replaced_values = [
        _decision_corrected_text(decision)
        for decision in decisions
        if _decision_action(decision) == "REPLACE"
    ]
    has_replace = any(value in gt_targets for value in replaced_values)
    has_gt_candidate = any(value in gt_targets for value in candidate_values)

    if not candidates:
        bucket = "NO_CANDIDATE"
    elif requery_items and not has_replace:
        bucket = "REQUERY_FAILED"
    elif has_gt_candidate and not has_replace:
        bucket = "PHASE2_REJECTED"
    else:
        bucket = "OTHER"

    return {
        "cue_index": cue_index,
        "bucket": bucket,
        "gt_targets": gt_targets,
        "candidate_values": candidate_values,
        "raw_text": raw_text,
        "gt_text": gt_text,
    }


def _write_diag_report(
    report_path: Path, source_name: str, gt_name: str,
    total_cues: int, detector_candidates: int, discovery_candidates: int,
    exp_summary, detector_by_source,
    gt_terms: dict[str, int], gt_covered: dict[str, int], gt_missed: dict[str, int],
    policy_counts=None,
) -> None:
    lines = [
        "# 候选诊断报告",
        "",
        "| 项目 | 值 |",
        "|------|------|",
        f"| 源文件 | `{source_name}` |",
        f"| GT 文件 | `{gt_name}` |",
        f"| 总字幕数 | {total_cues} |",
        f"| GT 差异术语种类 | {len(gt_terms)} |",
        "",
        "## 各阶段候选数",
        "",
        "| 阶段 | 候选数 |",
        "|------|--------|",
        f"| Detector | {detector_candidates} |",
        f"| Discovery (no LLM) | {discovery_candidates} |",
        f"| Expansion proposals | {exp_summary.raw_proposals} |",
        f"| Expansion added | {exp_summary.added_candidates} |",
        f"| Phase2 input (after cap) | {exp_summary.output_candidates} |",
        "",
        "## Detector 按 source 统计",
        "",
        "| Source | Count |",
        "|--------|-------|",
    ]
    for src, cnt in detector_by_source.most_common():
        lines.append(f"| {src} | {cnt} |")

    lines += [
        "",
        "## Expansion 按 source 统计 (added)",
        "",
        "| Source | Count |",
        "|--------|-------|",
    ]
    for src, cnt in sorted(exp_summary.added_by_source.items()):
        lines.append(f"| {src} | {cnt} |")

    policy_counts = policy_counts or {}
    lines += [
        "",
        "## Phase2 input 按 expansion_policy 统计",
        "",
        "| expansion_policy | Count |",
        "|--------|-------|",
    ]
    policy_items = (
        policy_counts.most_common()
        if hasattr(policy_counts, "most_common")
        else sorted(policy_counts.items())
    )
    for policy, cnt in policy_items:
        lines.append(f"| {policy} | {cnt} |")

    total_gt = sum(gt_terms.values())
    total_covered = sum(gt_covered.values())
    lines += [
        "",
        "## GT 术语候选覆盖率",
        "",
        f"总体: {total_covered}/{total_gt} ({total_covered/total_gt*100:.1f}%)" if total_gt else "总体: N/A",
        "",
        "| 术语 | 命中 | 总数 | 漏检 |",
        "|------|------|------|------|",
    ]
    for term, total in sorted(gt_terms.items(), key=lambda x: -x[1])[:20]:
        covered = gt_covered.get(term, 0)
        missed = gt_missed.get(term, 0)
        lines.append(f"| {term} | {covered} | {total} | {missed} |")

    lines += [
        "",
        "---",
        "*报告由 subtitle-corrector diagnose-candidates 自动生成*",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _render_fn_trace_section(traces: list) -> str:
    if not traces:
        return ""
    from collections import Counter
    counts = Counter(trace.stage_status.value for trace in traces)
    lines = [
        "## 漏修分层诊断",
        "",
        "| 阶段 | 数量 |",
        "|------|------:|",
    ]
    for stage, count in counts.most_common():
        lines.append(f"| `{stage}` | {count} |")
    lines.extend([
        "",
        "### 漏修明细",
        "",
        "| cue | 阶段 | 原文 | GT | GT terms | detector | discovery | expansion | phase2 | requery |",
        "|:---:|------|------|------|------|------:|------:|------:|------:|------|",
    ])
    for trace in traces:
        requery = "requested" if trace.requery_requested else "none"
        if trace.requery_requested and trace.requery_generated:
            requery = "generated"
        elif trace.requery_requested:
            requery = "failed"
        lines.append(
            "| {cue} | `{stage}` | {raw} | {gt} | {terms} | {detector} | {discovery} | {expansion} | {phase2} | {requery} |".format(
                cue=trace.cue_index,
                stage=trace.stage_status.value,
                raw=trace.raw_text[:50],
                gt=trace.gt_text[:50],
                terms=", ".join(trace.gt_terms[:3]),
                detector=len(trace.detector_candidates),
                discovery=len(trace.discovery_candidates),
                expansion=len(trace.expanded_candidates),
                phase2=len(trace.phase2_input_candidates),
                requery=requery,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_eval_report(
    eval_report, source_name: str, gt_name: str, output_path: Path,
    reuse_summary=None, requery_stats=None, pipeline_diag=None, fn_attr=None,
) -> None:
    lines = [
        "# 字幕纠错评估报告",
        "",
        "| 项目 | 值 |",
        "|------|------|",
        f"| 源文件 | `{source_name}` |",
        f"| GT 文件 | `{gt_name}` |",
        f"| 总字幕数 | {eval_report.total_cues} |",
        f"| 含错字幕数 | {eval_report.error_cues} |",
        "",
        "## 核心指标",
        "",
        "| 指标 | 值 |",
        "|------|------|",
        f"| TP (修复正确) | {eval_report.tp} |",
        f"| FP (误修) | {eval_report.fp} |",
        f"| TN (正确保留) | {eval_report.tn} |",
        f"| FN (漏修) | {eval_report.fn} |",
        f"| **Precision** | **{eval_report.precision:.1%}** |",
        f"| **Recall** | **{eval_report.recall:.1%}** |",
        f"| **F1** | **{eval_report.f1:.1%}** |",
        f"| **Accuracy** | **{eval_report.accuracy:.1%}** |",
        f"| 修复率 | {eval_report.fix_rate:.1%} ({eval_report.tp}/{eval_report.error_cues}) |",
        f"| 误修率 | {eval_report.false_positive_rate:.1%} ({eval_report.fp}/{eval_report.tp+eval_report.fp}) |",
        "",
    ]

    if eval_report.chat_diffs:
        lines += [
            "## 非术语差异（已从主分数排除）",
            "",
            "以下差异为代词/语气词/助词/无术语单字改动等非术语级差异，",
            "系统不应修复，但单独列出供人工复核。",
            "",
            "| # | 原文 | GT正确值 |",
            "|:---:|------|------|",
        ]
        for cd in eval_report.chat_diffs[:50]:
            raw = cd.raw_text.replace("|", "\\|")
            gt = cd.gt_text.replace("|", "\\|")
            lines.append(f"| {cd.cue_index} | {raw} | {gt} |")
        lines.append("")

    if reuse_summary is not None and reuse_summary.learned_rules > 0:
        lines += [
            "## 确认规则复用统计",
            "",
            "| 指标 | 值 |",
            "|------|------|",
            f"| 学习规则数 | {reuse_summary.learned_rules} |",
            f"| 复用决策数 | {reuse_summary.applied_decisions} |",
            f"| 冲突规则数 | {reuse_summary.conflict_rules} |",
            f"| 跳过低置信度 | {reuse_summary.skipped_low_confidence} |",
            f"| 跳过短词 | {reuse_summary.skipped_short_surface} |",
            "",
        ]

    if requery_stats is not None and requery_stats.get("requested", 0) > 0:
        lines += [
            "## REQUERY 统计",
            "",
            "| 指标 | 值 |",
            "|------|------|",
            f"| REQUERY 请求 | {requery_stats.get('requested', 0)} |",
            f"| 生成候选 | {requery_stats.get('candidates', 0)} |",
            f"| 二轮 REPLACE | {requery_stats.get('replaced', 0)} |",
            f"| 跳过 | {requery_stats.get('skipped', 0)} |",
            "",
        ]

    # FN attribution: classify each missed/wrong cue
    from subtitle_corrector.evaluation.fn_attribution import build_attribution_report, FNBucket
    fn_cues_all = list(eval_report.missed_errors) + [
        item for item in eval_report.wrong_fixes if item.has_error
    ]
    pipeline_diag = pipeline_diag or _empty_pipeline_diag()
    fn_attr = build_attribution_report(
        fn_cues_all,
        pipeline_diag.get("candidates_by_cue", {}),
        pipeline_diag.get("decisions_by_cue", {}),
        set(pipeline_diag.get("requery_by_cue", {}).keys()),
    )
    lines += [
        "## FN 归因统计",
        "",
        "| Bucket | Count |",
        "|------|------:|",
    ]
    for bucket in FNBucket.ALL:
        cnt = fn_attr.bucket_counts.get(bucket, 0)
        if cnt > 0:
            lines.append(f"| {bucket} | {cnt} |")
    lines += [
        "",
        "| Cue | Bucket | GT目标 |",
        "|:---:|------|------|",
    ]
    for attr in fn_attr.attributions[:40]:
        lines.append(
            f"| {attr.cue_index} | {attr.bucket} | {attr.gt_term} |"
        )
    lines.append("")

    if eval_report.wrong_fixes:
        lines.append("## 错误修改")
        lines.append("")
        lines.append("| # | 原文 | 修正 | GT正确值 | 决策诊断 |")
        lines.append("|:---:|------|------|------|------|")
        for wf in eval_report.wrong_fixes[:30]:
            decisions = pipeline_diag.get("decisions_by_cue", {}).get(wf.cue_index, [])
            requery_items = pipeline_diag.get("requery_by_cue", {}).get(wf.cue_index, [])
            diag = _format_decision_diagnostics(decisions, requery_items)
            lines.append(
                f"| {wf.cue_index} | {wf.raw_text[:50]} | "
                f"{wf.repaired_text[:50]} | {wf.gt_text[:50]} | {diag} |"
            )
        lines.append("")

    if eval_report.missed_errors:
        lines.append("## 漏修")
        lines.append("")
        lines.append("| # | 原文 | GT正确值 |")
        lines.append("|:---:|------|------|")
        for me in eval_report.missed_errors[:30]:
            lines.append(f"| {me.cue_index} | {me.raw_text[:50]} | {me.gt_text[:50]} |")
        lines.append("")

    if eval_report.correct_fixes:
        lines.append("## 正确修复")
        lines.append("")
        lines.append("| # | 原文 | 修正 | GT |")
        lines.append("|:---:|------|------|------|")
        for cf in eval_report.correct_fixes[:20]:
            lines.append(f"| {cf.cue_index} | {cf.raw_text[:50]} | {cf.repaired_text[:50]} | {cf.gt_text[:50]} |")
        lines.append("")

    # ── 漏修分层诊断 ──
    fn_cues = list(eval_report.missed_errors) + [
        item for item in eval_report.wrong_fixes if item.has_error
    ]
    if fn_cues and pipeline_diag and pipeline_diag.get("enabled"):
        from subtitle_corrector.evaluation.fn_trace import FNStageTrace
        traces: list = []
        candidates_by_cue = pipeline_diag.get("candidates_by_cue", {})
        decisions_by_cue = pipeline_diag.get("decisions_by_cue", {})
        requery_by_cue = pipeline_diag.get("requery_by_cue", {})
        for cue in fn_cues[:40]:
            gt_terms = _simple_diff_terms(cue.raw_text, cue.gt_text)
            cands = candidates_by_cue.get(cue.cue_index, [])
            decs = decisions_by_cue.get(cue.cue_index, [])
            trace = FNStageTrace(
                cue_index=cue.cue_index,
                raw_text=cue.raw_text[:80],
                gt_text=cue.gt_text[:80],
                gt_terms=gt_terms[:5],
                detector_candidates=cands,
                discovery_candidates=cands,
                phase2_input_candidates=cands,
                phase2_actions=[_decision_action(d) for d in decs],
                requery_requested=cue.cue_index in requery_by_cue,
            )
            traces.append(trace)
        trace_section = _render_fn_trace_section(traces)
        if trace_section:
            lines.append(trace_section)

    lines.append("---")
    lines.append("*报告由 subtitle-corrector 自动生成*")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _run_normal_mode(
    subtitle_path: Path,
    document,
    settings,
    report: Path | None,
) -> None:
    """正常运行模式 — 现有纠错管线."""
    from datetime import datetime

    from subtitle_corrector.report import generate_markdown_report

    pipeline = build_pipeline(settings)
    result = pipeline.run(document)
    logger.info("pipeline produced {} review/correction candidates", len(result.repairs))

    import json
    for repair in result.repairs[:20]:
        typer.echo(json.dumps(repair.model_dump(), ensure_ascii=False))

    try:
        if report is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report = Path(f"reports/{subtitle_path.stem}_report_{ts}.md")
        generate_markdown_report(
            repairs=result.repairs,
            source_file=str(subtitle_path),
            output_path=report,
            entity_memory_log=result.entity_memory_log,
        )
        logger.info("审查报告已保存: {}", report)
    except Exception as exc:
        logger.error("生成报告失败: {}", exc)


if __name__ == "__main__":
    app()
