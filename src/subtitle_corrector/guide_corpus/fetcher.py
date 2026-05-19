"""HTML / API 抓取与缓存。

对米游社 URL 调用 API 获取 JSON，其他 URL 抓取 HTML。
缓存到 raw/ 目录。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import httpx
from loguru import logger


def fetch_articles(
    urls: list[str],
    raw_dir: str | Path,
    timeout: int = 30,
) -> list[Path]:
    """抓取 URL 列表，缓存到 raw_dir，返回已缓存文件路径列表。"""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cached_paths: list[Path] = []
    # Googlebot UA 触发 SSR（非 API 模式回退用）
    html_headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; Googlebot/2.1; "
            "+http://www.google.com/bot.html)"
        )
    }
    api_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.miyoushe.com/",
    }

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for url in urls:
            url_hash = _hash_url(url)
            cache_path = raw_dir / f"{url_hash}.json"  # 优先 JSON

            if cache_path.exists():
                logger.info("缓存命中: {} -> {}", url, cache_path.name)
                cached_paths.append(cache_path)
                continue

            # 米游社 article URL → API
            miyoushe_id = _extract_miyoushe_post_id(url)
            if miyoushe_id:
                try:
                    api_url = (
                        "https://bbs-api.miyoushe.com/post/wapi/"
                        f"getPostFull?post_id={miyoushe_id}"
                    )
                    resp = client.get(api_url, headers=api_headers)
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("retcode") == 0:
                        cache_path.write_text(
                            json.dumps(data, ensure_ascii=False), encoding="utf-8"
                        )
                        logger.info("API 抓取成功: {} -> {}", url, cache_path.name)
                        cached_paths.append(cache_path)
                        continue
                    logger.warning("API 返回错误: {} retcode={}", url, data.get("retcode"))
                except Exception as exc:
                    logger.warning("API 抓取失败 {}: {}", url, exc)

            # 回退：HTML 抓取
            html_path = raw_dir / f"{url_hash}.html"
            if html_path.exists():
                cached_paths.append(html_path)
                continue
            try:
                resp = client.get(url, headers=html_headers)
                resp.raise_for_status()
                html_path.write_text(resp.text, encoding="utf-8")
                logger.info("HTML 抓取成功: {} -> {}", url, html_path.name)
                cached_paths.append(html_path)
            except Exception as exc:
                logger.warning("抓取失败 {}: {}", url, exc)

    return cached_paths


def _extract_miyoushe_post_id(url: str) -> str | None:
    m = re.search(r"miyoushe\.com/[^/]+/article/(\d+)", url)
    if m:
        return m.group(1)
    return None


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]
