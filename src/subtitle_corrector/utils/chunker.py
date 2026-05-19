"""Sliding-window SRT chunker with configurable overlap for context preservation.

Produces Chunk objects that separate each chunk into context_before (preceding
overlap), target_lines (the payload for downstream processing), and
context_after (trailing overlap). All indices are anchored to the original
SRT cue order — no coordinate drift is possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class Chunk(Generic[T]):
    """A single window over an ordered sequence of items.

    *target_lines* is the payload that should be processed (e.g. sent to the
    LLM). *context_before* / *context_after* are overlap regions that provide
    continuity but were already part of the previous / next chunk's payload.
    """

    chunk_id: int
    context_before: list[T] = field(default_factory=list)
    target_lines: list[T] = field(default_factory=list)
    context_after: list[T] = field(default_factory=list)
    start_index: int = 0
    end_index: int = 0


@dataclass
class ChunkResult(Generic[T]):
    """Container for all chunks produced by SubtitleChunker, plus metadata."""

    chunks: list[Chunk[T]]
    total_items: int
    chunk_size: int
    overlap_size: int


class SubtitleChunker:
    """Split an ordered sequence into overlapping windows.

    Each window (Chunk) is composed of three regions::

        [ … context_before … ][ …… target_lines …… ][ … context_after … ]

    - *context_before*: the last *overlap_size* items of the previous chunk's
      target. Provides continuity so downstream consumers (e.g. an LLM) can
      resolve cross-chunk references.
    - *target_lines*: the payload — exactly *chunk_size* items (except possibly
      the last chunk, which may be shorter).
    - *context_after*: the first *overlap_size* items of the next chunk's
      target. Symmetric to context_before.

    The caller receives Chunk objects whose indices remain anchored to the
    original sequence, so there is never any index drift.
    """

    def __init__(self, chunk_size: int = 100, overlap_size: int = 15) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap_size < 0:
            raise ValueError("overlap_size must be >= 0")
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def chunk(self, items: list[T]) -> ChunkResult[T]:
        """Split *items* into overlapping windows.

        Returns a :class:`ChunkResult` with the full list of chunks and
        summary metadata.
        """
        n = len(items)
        chunks: list[Chunk[T]] = []
        chunk_id = 0
        cursor = 0

        while cursor < n:
            # Determine the payload slice
            payload_start = cursor
            payload_end = min(cursor + self.chunk_size, n)
            target = items[payload_start:payload_end]

            # context_before: the overlap_size items immediately preceding
            # this chunk's payload (i.e. the tail of the previous target).
            ctx_before_start = max(0, payload_start - self.overlap_size)
            ctx_before = (
                items[ctx_before_start:payload_start]
                if payload_start > 0
                else []
            )

            # context_after: the overlap_size items immediately following
            # this chunk's payload (i.e. the head of the next target).
            ctx_after_end = min(payload_end + self.overlap_size, n)
            ctx_after = (
                items[payload_end:ctx_after_end]
                if payload_end < n
                else []
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    context_before=ctx_before,
                    target_lines=target,
                    context_after=ctx_after,
                    start_index=payload_start,
                    end_index=payload_end - 1,
                )
            )

            chunk_id += 1
            cursor = payload_end

        return ChunkResult(
            chunks=chunks,
            total_items=n,
            chunk_size=self.chunk_size,
            overlap_size=self.overlap_size,
        )
