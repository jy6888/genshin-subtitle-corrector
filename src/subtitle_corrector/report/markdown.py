"""Markdown diff report generator for subtitle correction results.

Generates a human-readable Markdown file that highlights ASR corrections
with inline diffs, confidence scores, LLM reasoning, and entity memory tracking.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from subtitle_corrector.schemas import CorrectionAction, EntityMemorySnapshot, RepairResult


# ---------------------------------------------------------------------------
# Inline diff helpers
# ---------------------------------------------------------------------------

def _char_diff(original: str, repaired: str) -> tuple[str, str]:
    """Produce inline-annotated strings highlighting character-level changes.

    Returns (marked_original, marked_repaired) where changed spans are
    wrapped with ~~strikethrough~~ and **bold** respectively.
    """
    if original == repaired:
        return original, repaired

    import difflib

    sm = difflib.SequenceMatcher(None, original, repaired, autojunk=False)
    orig_parts: list[str] = []
    repair_parts: list[str] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            orig_parts.append(original[i1:i2])
            repair_parts.append(repaired[j1:j2])
        elif tag == "replace":
            orig_parts.append(f"~~{original[i1:i2]}~~")
            repair_parts.append(f"**{repaired[j1:j2]}**")
        elif tag == "delete":
            orig_parts.append(f"~~{original[i1:i2]}~~")
        elif tag == "insert":
            repair_parts.append(f"**{repaired[j1:j2]}**")

    return "".join(orig_parts), "".join(repair_parts)


def _action_badge(action: CorrectionAction) -> str:
    """Return a human-readable emoji badge for the action."""
    if action == CorrectionAction.REPLACE:
        return "[已修正]"
    if action == CorrectionAction.NEEDS_REVIEW:
        return "[需复核]"
    if action == CorrectionAction.KEEP:
        return "[保持]"
    return f"[{action}]"


def _confidence_bar(confidence: float) -> str:
    """Render a simple text progress bar for confidence."""
    filled = round(confidence * 10)
    return f"{'#' * filled}{'-' * (10 - filled)} {confidence:.0%}"


# ---------------------------------------------------------------------------
# Entity memory timeline helpers
# ---------------------------------------------------------------------------

def _weight_bar(weight: float, width: int = 5) -> str:
    """Render a small inline weight bar for entity memory visualisation."""
    filled = round(weight * width)
    return f"{'*' * filled}{'.' * (width - filled)} {weight:.2f}"


def _build_entity_timeline(snapshots: list[EntityMemorySnapshot]) -> str:
    """Build the entity memory timeline section.

    Shows activation, decay and forgetting events across all cues.
    """
    if not snapshots:
        return ""

    # Collect all entity names that ever appeared
    all_entities: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for entity in snap.entities_before:
            if entity not in seen:
                seen.add(entity)
                all_entities.append(entity)
        for entity in snap.entities_activated:
            if entity not in seen:
                seen.add(entity)
                all_entities.append(entity)

    if not all_entities:
        return ""

    lines: list[str] = []
    lines.append("## 实体记忆时间线")
    lines.append("")
    lines.append("> 展示每个实体在各字幕行的权重变化。")
    lines.append("> `*` = 权重高（活跃）, `.` = 权重低（衰减中）, `-` = 不在记忆中（已遗忘或未激活）")
    lines.append("")

    # Header row
    header_cells = ["字幕行"]
    for entity in all_entities:
        header_cells.append(f"**{entity}**")
    lines.append("| " + " | ".join(header_cells) + " |")
    # Separator
    sep_cells = [":---:"] + [":---:"] * len(all_entities)
    lines.append("| " + " | ".join(sep_cells) + " |")

    # One row per cue
    for snap in snapshots:
        row = [f"#{snap.cue_index}"]
        for entity in all_entities:
            activated_this_cue = entity in snap.entities_activated
            in_before = entity in snap.entities_before
            in_after = entity in snap.entities_after

            if activated_this_cue and in_before:
                cell = f"**刷新 {_weight_bar(1.0)}**"
            elif activated_this_cue and not in_before:
                cell = f"**+激活 {_weight_bar(1.0)}**"
            elif in_after:
                cell = _weight_bar(snap.entities_after[entity])
            elif in_before:
                # Was in memory but decayed out (forgotten)
                before_w = snap.entities_before[entity]
                cell = f"~~遗忘 ({before_w:.2f})~~"
            else:
                cell = "—"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Event log: activation, refresh, and forget events only (decay visible in grid)
    lines.append("### 事件日志")
    lines.append("")
    lines.append("| 字幕行 | 事件 | 实体 |")
    lines.append("|:---:|:---:|------|")

    event_count = 0
    for snap in snapshots:
        for entity in snap.entities_activated:
            was_active = entity in snap.entities_before
            if was_active:
                lines.append(f"| #{snap.cue_index} | 刷新 | `{entity}` (权重恢复至 1.00) |")
            else:
                lines.append(f"| #{snap.cue_index} | **激活** | `{entity}` (首次加入记忆) |")
            event_count += 1

        # Forget events: entity was in before but not in after
        if snap.decays_applied:
            for entity in snap.entities_before:
                if entity not in snap.entities_after and entity not in snap.entities_activated:
                    before_w = snap.entities_before[entity]
                    lines.append(
                        f"| #{snap.cue_index} | **遗忘** | `{entity}` "
                        f"(权重 {before_w:.2f} 跌至阈值以下，已清除) |"
                    )
                    event_count += 1

    if event_count == 0:
        lines.append("| — | — | 本次运行无实体记忆活动 |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(
    repairs: list[RepairResult],
    source_file: str,
    output_path: Path,
    entity_memory_log: list[EntityMemorySnapshot] | None = None,
) -> None:
    """Write a Markdown review report to *output_path*.

    Includes per-line repair details with inline diffs, and an entity memory
    timeline section when entity memory is active.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_name = Path(source_file).name

    actionable = [r for r in repairs if r.action != CorrectionAction.KEEP]
    keep_count = len(repairs) - len(actionable)
    replace_count = sum(1 for r in repairs if r.action == CorrectionAction.REPLACE)
    review_count = sum(1 for r in repairs if r.action == CorrectionAction.NEEDS_REVIEW)

    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────
    lines.append("# 字幕纠错审查报告")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| 源文件 | `{source_name}` |")
    lines.append(f"| 生成时间 | {now} |")
    lines.append(f"| 检测总条数 | {len(repairs)} |")
    lines.append(f"| [已修正] 建议替换 | **{replace_count}** |")
    lines.append(f"| [需复核] 需人工复核 | **{review_count}** |")
    lines.append(f"| [保持] 保持原样 | {keep_count} |")
    lines.append("")

    # ── Entity memory timeline ──────────────────────────────────────────
    if entity_memory_log:
        timeline = _build_entity_timeline(entity_memory_log)
        if timeline:
            lines.append("---")
            lines.append("")
            lines.append(timeline)
            lines.append("---")
            lines.append("")

    # ── Actionable items detail ─────────────────────────────────────────
    if actionable:
        lines.append("## 需要关注的条目")
        lines.append("")

        for i, repair in enumerate(actionable, 1):
            marked_orig, marked_repair = _char_diff(
                repair.original_text, repair.repaired_text
            )
            badge = _action_badge(repair.action)
            conf_bar = _confidence_bar(repair.confidence)

            lines.append(f"### {i}. 字幕行 #{repair.cue_index}")
            lines.append("")
            lines.append(f"**判定**: {badge}    **置信度**: {conf_bar}")
            lines.append("")

            # Diff block
            lines.append("| | 文本 |")
            lines.append("|:---:|------|")
            lines.append(f"| 原文 | {marked_orig} |")
            if repair.action == CorrectionAction.REPLACE:
                lines.append(f"| 修正 | {marked_repair} |")
            lines.append("")

            # Reasoning
            if repair.explanation:
                lines.append(f"> **AI 判决理由**：{repair.explanation}")
                lines.append("")

            lines.append("---")
            lines.append("")
    else:
        lines.append("> 未发现需要关注的 ASR 错误，所有字幕行均正常。")
        lines.append("")

    # ── Full summary table ──────────────────────────────────────────────
    lines.append("## 全部检测结果总览")
    lines.append("")
    lines.append("| # | 原文 | 判定 | 置信度 |")
    lines.append("|:---:|------|:---:|:---:|")
    for repair in repairs:
        action_short = {
            CorrectionAction.REPLACE: "[已修正]",
            CorrectionAction.NEEDS_REVIEW: "[需复核]",
            CorrectionAction.KEEP: "[保持]",
        }.get(repair.action, str(repair.action))
        # Truncate long text for the summary table
        text_preview = repair.original_text[:30]
        if len(repair.original_text) > 30:
            text_preview += "..."
        lines.append(
            f"| {repair.cue_index} | {text_preview} | {action_short} | {repair.confidence:.0%} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("*报告由 subtitle-corrector 自动生成*")

    # ── Write ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
