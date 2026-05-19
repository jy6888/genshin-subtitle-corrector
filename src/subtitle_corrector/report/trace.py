"""全链路可追溯 Pipeline Trace 报告生成器。

记录每个模块的输入/输出边界和模块间通信，不深入模块内部细节。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class StageRecord:
    """一条 pipeline 阶段的输入→输出记录."""
    stage: str
    input_summary: str
    output_summary: str
    details: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TraceReport:
    """累积各阶段记录，最终生成 Markdown 流水线追踪报告."""

    def __init__(self, source_file: str) -> None:
        self.source_file = source_file
        self.stages: list[StageRecord] = []
        self._chunk_records: list[dict] = []

    def record(self, stage: str, input_summary: str, output_summary: str,
               details: list[str] | None = None,
               warnings: list[str] | None = None) -> None:
        self.stages.append(StageRecord(
            stage=stage,
            input_summary=input_summary,
            output_summary=output_summary,
            details=details or [],
            warnings=warnings or [],
        ))

    def record_chunk(self, chunk_id: int, obs_count: int,
                     entities: dict, categories: list[str],
                     transition: bool, noise_count: int) -> None:
        self._chunk_records.append({
            "id": chunk_id, "obs": obs_count,
            "entities": entities, "categories": categories,
            "transition": transition, "noise": noise_count,
        })

    def record_table(self, title: str, headers: list[str],
                     rows: list[list[str]]) -> None:
        """记录一个 Markdown 表格。"""
        self.stages.append(StageRecord(
            stage=title,
            input_summary="",
            output_summary="",
            details=[
                f"TABLE:{'|'.join(headers)}",
                *[f"ROW:{'|'.join(str(c) for c in r)}" for r in rows],
            ],
        ))

    def record_histogram(self, title: str, buckets: list[str],
                         counts: list[int], total: int,
                         threshold: float = 0.0) -> None:
        """记录一个文本直方图。"""
        lines = []
        for b, c in zip(buckets, counts):
            pct = c / total * 100 if total else 0
            bar = "█" * int(pct / 2) if pct >= 1 else ""
            # 检查这个 bucket 是否跨越了 LLM 阈值
            parts = b.split("-")
            bucket_start = float(parts[0].strip()) if parts else 0
            marker = " ← triggered LLM" if threshold > 0 and bucket_start < threshold <= (
                float(parts[1].strip()) if len(parts) > 1 else 2.0
            ) else ""
            lines.append(
                f"  {b:>12s} : {c:>4d} ({pct:5.1f}%) {bar}{marker}"
            )
        self.stages.append(StageRecord(
            stage=title,
            input_summary=f"{total} cues",
            output_summary="",
            details=lines,
        ))

    def generate(self, output_path: Path) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source = Path(self.source_file).name

        lines = [
            "# Pipeline Trace Report",
            "",
            "| 项目 | 值 |",
            "|------|------|",
            f"| 源文件 | `{source}` |",
            f"| 生成时间 | {now} |",
            f"| 流水线阶段 | {len(self.stages)} |",
            "",
            "---",
            "",
            "## 模块间通信追踪",
            "",
        ]

        for i, s in enumerate(self.stages, 1):
            lines.append(f"### {i}. {s.stage}")
            lines.append("")

            # 表格渲染
            if s.details and s.details[0].startswith("TABLE:"):
                headers = s.details[0][6:].split("|")
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join("------" for _ in headers) + "|")
                for d in s.details[1:]:
                    if d.startswith("ROW:"):
                        cells = d[4:].split("|")
                        lines.append("| " + " | ".join(cells) + " |")
            elif s.input_summary or s.output_summary:
                lines.append("| | 内容 |")
                lines.append("|:---|------|")
                if s.input_summary:
                    lines.append(f"| 输入 | {s.input_summary} |")
                if s.output_summary:
                    lines.append(f"| 输出 | {s.output_summary} |")
                if s.details:
                    for d in s.details:
                        lines.append(f"| | {d} |")
            elif s.details:
                for d in s.details:
                    lines.append(d)

            if s.warnings:
                for w in s.warnings:
                    lines.append(f"| ⚠️ | {w} |")
            lines.append("")

        # Chunk summary section
        if self._chunk_records:
            lines.append("---")
            lines.append("")
            lines.append("## DiscoveryEngine — 每个 Chunk 的 LLM 通信")
            lines.append("")
            header = "| Chunk | Cues | confirmed_entities | dominant_categories | transition | noise |"
            sep = "|:---:|:---:|------|------|:---:|:---:|"
            lines.append(header)
            lines.append(sep)
            for ch in self._chunk_records:
                ents = ", ".join(
                    f"{e}({c:.2f})" for e, c in list(ch["entities"].items())[:3]
                ) or "—"
                cats = ", ".join(ch["categories"][:3]) or "—"
                lines.append(
                    f"| {ch['id']} | {ch['obs']} | {ents} | {cats} "
                    f"| {'✅' if ch['transition'] else '—'} | {ch['noise']} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("*报告由 Pipeline Trace 自动生成*")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
