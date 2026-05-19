"""口语别名审核工作流。

从候选 CSV 中筛选 approved/needs_context/rejected，
输出审核后的 approved CSV。
"""

from __future__ import annotations

import csv
from pathlib import Path

from loguru import logger


def export_approved_aliases(
    candidates_path: str | Path,
    output_path: str | Path,
    auto_approve_threshold: float = 0.85,
) -> dict:
    """从候选 CSV 导出审核后的 approved CSV。

    自动审批规则：
    - 置信度 >= auto_approve_threshold 且无 ambiguous 标记 → approved
    - ambiguous_alias 标记 → needs_context（需人工确认消歧）
    - 其余保持 pending

    Returns:
        summary dict with approved, needs_context, rejected, pending counts
    """
    candidates: list[dict] = []
    with open(candidates_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            candidates.append(row)

    approved: list[dict] = []
    needs_context: list[dict] = []
    rejected: list[dict] = []
    pending: list[dict] = []

    for row in candidates:
        status = (row.get("review_status") or "pending").strip()
        confidence = float(row.get("confidence") or 0)
        risk_flags = (row.get("risk_flags") or "").strip()

        if status == "approved":
            approved.append(row)
        elif status == "needs_context":
            needs_context.append(row)
        elif status == "rejected":
            rejected.append(row)
        elif status == "pending":
            if "ambiguous_alias" in risk_flags:
                row["review_status"] = "needs_context"
                needs_context.append(row)
            elif confidence >= auto_approve_threshold:
                row["review_status"] = "approved"
                approved.append(row)
            else:
                pending.append(row)

    # 写 approved CSV
    fieldnames = [
        "alias_surface", "canonical_term", "alias_kind",
        "usage_policy", "confidence", "evidence_count",
        "risk_flags", "notes",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in approved + needs_context:
            risk = (row.get("risk_flags") or "").strip()
            alias_kind = _infer_alias_kind(risk)
            usage_policy = _infer_usage_policy(risk)

            writer.writerow({
                "alias_surface": row.get("alias_surface", ""),
                "canonical_term": row.get("canonical_term", ""),
                "alias_kind": alias_kind,
                "usage_policy": usage_policy,
                "confidence": row.get("confidence", "0"),
                "evidence_count": row.get("evidence_count", "0"),
                "risk_flags": risk,
                "notes": row.get("notes", ""),
            })

    logger.info(
        "导出审核后简称: approved={}, needs_context={}, rejected={}, pending={}",
        len(approved), len(needs_context), len(rejected), len(pending),
    )

    return {
        "approved": len(approved),
        "needs_context": len(needs_context),
        "rejected": len(rejected),
        "pending": len(pending),
        "output_path": str(output_path),
    }


def _infer_alias_kind(risk_flags: str) -> str:
    if "normal_title_alias" in risk_flags:
        return "normal_title_alias"
    if "ambiguous_alias" in risk_flags:
        return "ambiguous_alias"
    return "term_short_alias"


def _infer_usage_policy(risk_flags: str) -> str:
    if "ambiguous_alias" in risk_flags:
        return "needs_context"
    return "context_only"
