"""文章清洗器。

支持两种格式：
1. 米游社 API JSON (bbs-api.miyoushe.com)
2. 通用 HTML (bs4 清洗)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import bs4
from loguru import logger

from subtitle_corrector.guide_corpus.schema import CleanedArticle

_MIN_TEXT_LENGTH = 200


def clean_articles(
    html_paths: list[Path],
    source_label: str = "unknown",
) -> list[CleanedArticle]:
    articles: list[CleanedArticle] = []
    seen_hashes: set[str] = set()

    for path in html_paths:
        try:
            raw = path.read_text(encoding="utf-8")

            if path.suffix == ".json":
                article = _clean_miyoushe_json(raw, source_label)
            else:
                article = _clean_html(raw, source_label, path)

            if article is None:
                continue
            if len(article.text) < _MIN_TEXT_LENGTH:
                logger.info("跳过短文 ({} 字): {}", len(article.text),
                            article.title[:50] if article.title else path.name)
                continue
            if article.content_hash in seen_hashes:
                logger.info("跳过重复文章: {}", article.title[:50])
                continue
            seen_hashes.add(article.content_hash)

            articles.append(article)
        except Exception as exc:
            logger.warning("清洗失败 {}: {}", path.name, exc)

    logger.info("清洗完成: {} 篇有效文章 (来自 {} 个文件)", len(articles), len(html_paths))
    return articles


def _clean_miyoushe_json(raw: str, source_label: str) -> CleanedArticle | None:
    """从米游社 API JSON 提取文章。"""
    data = json.loads(raw)
    post = data.get("data", {}).get("post", {}).get("post", {})
    if not post:
        return None

    subject = (post.get("subject") or "").strip()[:120]

    # 优先从 structured_content (Quill delta) 提取文本
    text = ""
    structured = post.get("structured_content", "")
    if structured and structured != "[]":
        try:
            rows = json.loads(structured) if isinstance(structured, str) else structured
            parts: list[str] = []
            for row in rows:
                if isinstance(row, dict):
                    ins = row.get("insert", "")
                    if isinstance(ins, str):
                        parts.append(ins)
                    elif isinstance(ins, dict):
                        parts.append("\n")
            text = "".join(parts)
        except (json.JSONDecodeError, TypeError):
            pass

    # structured_content 不够时回退到 content (HTML)
    if len(text) < _MIN_TEXT_LENGTH:
        content = post.get("content", "")
        if content:
            text = _html_to_text(content)

    text = _clean_whitespace(text)
    post_id = post.get("post_id", "")

    return CleanedArticle(
        source=source_label,
        url=f"https://www.miyoushe.com/ys/article/{post_id}",
        title=subject,
        text=text,
        fetched_at="",
        content_hash=_hash_text(text),
    )


def _clean_html(raw: str, source_label: str, path: Path) -> CleanedArticle | None:
    """从 HTML 提取文章正文。"""
    soup = bs4.BeautifulSoup(raw, "lxml")

    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    for noise in soup.find_all(class_=re.compile(
        r"comment|reply|sidebar|ad-|advertisement|recommend|related|nav|footer|header|menu",
        re.I,
    )):
        noise.decompose()

    title = _extract_html_title(soup, path)
    text = _extract_html_body(soup)

    return CleanedArticle(
        source=source_label,
        url=f"cached://{path.stem}",
        title=title,
        text=text,
        fetched_at="",
        content_hash=_hash_text(text),
    )


def _html_to_text(html: str) -> str:
    """简单 HTML → 纯文本。"""
    text = re.sub(r"<br\s*/?>", "\n", html)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&amp;", "&").replace("&quot;", '"')
    return text


def _extract_html_title(soup: bs4.BeautifulSoup, path: Path) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    return path.stem


def _extract_html_body(soup: bs4.BeautifulSoup) -> str:
    for selector in ["article", "main", "[class*=content]", "[class*=article]", "[class*=post]"]:
        container = soup.select_one(selector)
        if container:
            text = container.get_text(separator="\n", strip=True)
            if len(text) >= _MIN_TEXT_LENGTH:
                return _clean_whitespace(text)
    body = soup.find("body")
    if body:
        return _clean_whitespace(body.get_text(separator="\n", strip=True))
    return _clean_whitespace(soup.get_text(separator="\n", strip=True))


def _clean_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
