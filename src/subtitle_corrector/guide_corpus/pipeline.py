"""攻略语料构建主流程。

串联 加载URL → 抓取 → 清洗 → 分句 → 输出JSONL 全流程。
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from subtitle_corrector.guide_corpus.sources import load_source_urls
from subtitle_corrector.guide_corpus.fetcher import fetch_articles
from subtitle_corrector.guide_corpus.cleaner import clean_articles
from subtitle_corrector.guide_corpus.sentence_splitter import split_sentences


def build_guide_corpus(
    source_urls_path: str | Path,
    output_dir: str | Path,
    known_entities: set[str] | None = None,
) -> dict:
    """从 source_urls.txt 构建攻略语料库。

    Returns:
        dict with articles_count, sentences_count, output_dir
    """
    source_urls_path = Path(source_urls_path)
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 加载白名单 URL
    urls = load_source_urls(source_urls_path)
    if not urls:
        logger.warning("source_urls.txt 中没有有效 URL")
        return {"articles_count": 0, "sentences_count": 0, "output_dir": str(output_dir)}

    logger.info("加载 {} 个白名单 URL", len(urls))

    # Step 2: 抓取 HTML
    html_paths = fetch_articles(urls, raw_dir)
    if not html_paths:
        logger.warning("没有成功抓取到任何文章")
        return {"articles_count": 0, "sentences_count": 0, "output_dir": str(output_dir)}

    # Step 3: 清洗提取正文
    source_label = _guess_source_label(urls)
    articles = clean_articles(html_paths, source_label)

    # Step 4: 分句
    sentence_list = split_sentences(articles, known_entities)

    # Step 5: 写出 JSONL
    articles_path = output_dir / "cleaned_articles.jsonl"
    with open(articles_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(article.model_dump_json(ensure_ascii=False) + "\n")
    logger.info("写出 {} 篇文章到 {}", len(articles), articles_path)

    sentences_path = output_dir / "sentences.jsonl"
    with open(sentences_path, "w", encoding="utf-8") as f:
        for sent in sentence_list:
            f.write(sent.model_dump_json(ensure_ascii=False) + "\n")
    logger.info("写出 {} 个句子到 {}", len(sentence_list), sentences_path)

    return {
        "articles_count": len(articles),
        "sentences_count": len(sentence_list),
        "output_dir": str(output_dir),
    }


def _guess_source_label(urls: list[str]) -> str:
    """从 URL 中猜测来源平台标签。"""
    for url in urls[:1]:
        if "miyoushe" in url:
            return "miyoushe"
        if "nga" in url:
            return "nga"
        if "bilibili" in url:
            return "bilibili"
        if "zhihu" in url:
            return "zhihu"
        if "tieba" in url:
            return "tieba"
    return "unknown"
