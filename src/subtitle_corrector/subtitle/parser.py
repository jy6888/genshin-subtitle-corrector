from __future__ import annotations

from pathlib import Path

import pysubs2

from subtitle_corrector.schemas import SubtitleCue, SubtitleDocument, SubtitleFormat


class SubtitleParser:
    """Parse common subtitle files into a unified cue model."""

    def parse_file(self, path: str | Path) -> SubtitleDocument:
        file_path = Path(path)
        fmt = self._detect_format(file_path)
        subs = pysubs2.load(str(file_path), encoding="utf-8")
        cues = [
            SubtitleCue(
                index=i,
                start_ms=int(event.start),
                end_ms=int(event.end),
                text=event.text,
                style=getattr(event, "style", None),
                metadata={"pysubs2_type": type(event).__name__},
            )
            for i, event in enumerate(subs.events)
        ]
        return SubtitleDocument(format=fmt, cues=cues, source_path=str(file_path))

    def save_file(self, document: SubtitleDocument, path: str | Path) -> None:
        output = pysubs2.SSAFile()
        for cue in document.cues:
            output.events.append(
                pysubs2.SSAEvent(start=cue.start_ms, end=cue.end_ms, text=cue.text)
            )
        output.save(str(path))

    @staticmethod
    def _detect_format(path: Path) -> SubtitleFormat:
        suffix = path.suffix.lower().lstrip(".")
        if suffix == "srt":
            return SubtitleFormat.SRT
        if suffix == "vtt":
            return SubtitleFormat.VTT
        if suffix == "ass":
            return SubtitleFormat.ASS
        return SubtitleFormat.UNKNOWN
