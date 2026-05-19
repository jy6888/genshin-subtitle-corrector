"""白名单 URL 来源管理。

从 source_urls.txt 读取攻略 URL 列表，每行一个。
"""

from __future__ import annotations

from pathlib import Path


def load_source_urls(path: str | Path) -> list[str]:
    """读取白名单 URL 列表，跳过空行和注释行。"""
    urls: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            urls.append(stripped)
    return urls
