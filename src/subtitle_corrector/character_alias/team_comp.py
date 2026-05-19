"""Team composition shorthand parser."""

from __future__ import annotations

import re

from pypinyin import lazy_pinyin
from rapidfuzz import fuzz

from subtitle_corrector.character_alias.lexicon import CharacterAliasEntry, CharacterAliasLexicon

_TEAM_COMP_TRIGGERS = re.compile(
    r"配队|阵容|队伍|队|组队|战队|组合|搭配|加|带|那芙|芙万|芙白|万白|"
    r"草行久|万达|胡行钟|雷国|雷九|神鹤|融甘|胡夜|提八|三神|"
    r"纯水队|蒸发队|融化队|感电队|超载队|绽放队|激化队|永冻队|岩队|风队|雷队|火队"
)

_SINGLE_CHAR_SEQUENCE = re.compile(r"[\u4e00-\u9fff]{2,6}")

_AMBIGUOUS_SLOTS: set[str] = {"莱", "夏", "爱", "玛", "娜", "琳", "一", "九", "五", "七"}

_TEAM_MARKERS_AFTER = ("队", "组合", "阵容", "配队", "队伍")
_TEAM_MARKERS_BEFORE = ("带", "加", "配", "和")
_LOADOUT_NOISE_TERMS = ("标准", "输出", "生成", "输出套", "输出装", "生成装", "标准输出", "标准生成")

# Reviewed pair repairs for ASR-corrupted team shorthand that cannot be safely
# inferred from single-character slot pinyin alone.
_REVIEWED_TEAM_COMP_REPAIRS: dict[str, tuple[str, list[str]]] = {
    "琴谱": ("琴芙", ["琴", "芙宁娜"]),
}


class TeamCompParser:
    def __init__(self, lexicon: CharacterAliasLexicon) -> None:
        self.lexicon = lexicon

    def parse(self, text: str) -> list[tuple[str, str]]:
        if not self._has_team_context(text):
            return []

        results: list[tuple[str, str]] = []
        results.extend(self._parse_single_char_sequence(text))
        nicknames = self.lexicon.find_nicknames_in_text(text)
        for entry in nicknames:
            results.append((entry.alias_surface, entry.canonical_term))

        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for alias, canonical in results:
            if canonical not in seen:
                seen.add(canonical)
                unique.append((alias, canonical))
        return unique

    def requery_alias_candidate(self, text: str, suspect_surface: str) -> dict | None:
        """Build a team-comp alias repair candidate after an LLM REQUERY.

        The LLM has already decided this surface is worth repairing — we only
        validate it can be rebuilt from team_slot characters.
        """
        if not suspect_surface or suspect_surface not in text:
            return None
        if len(suspect_surface) < 2 or len(suspect_surface) > 6:
            return None

        start = text.find(suspect_surface)
        end = start + len(suspect_surface)
        override = self._reviewed_override_candidate(text, suspect_surface, start, end)
        if override is not None:
            return override

        if not self._has_local_team_marker(text, start, end):
            return None
        if self._is_loadout_noise_surface(text, start, end):
            return None

        repaired = self._repair_slot_sequence(suspect_surface)
        if repaired is None:
            return None
        value, activated = repaired

        return self._build_requery_candidate(
            surface=suspect_surface,
            value=value,
            activated=activated,
            start=start,
            end=end,
            source="team_comp_requery",
            score=0.86,
            match_kind="team_comp_slot_phonetic",
        )

    def find_requery_hints(self, text: str, max_hints: int = 2) -> list[str]:
        """Find broken team-comp shorthand surfaces worth asking Phase2 about.

        Requires local team wording so ordinary loadout text does not turn into
        character/team aliases. Reviewed overrides are tried before fuzzy slots.
        """
        hints: list[str] = []
        occupied: list[tuple[int, int]] = []
        for surface in _REVIEWED_TEAM_COMP_REPAIRS:
            pos = text.find(surface)
            while pos != -1:
                end = pos + len(surface)
                if self._has_local_team_marker(text, pos, end):
                    hints.append(surface)
                    occupied.append((pos, end))
                    if len(hints) >= max_hints:
                        return hints
                pos = text.find(surface, pos + 1)

        for surface, start, end in self._iter_candidate_surfaces(text):
            if surface in hints:
                continue
            if any(start < occ_end and end > occ_start for occ_start, occ_end in occupied):
                continue
            if not self._has_local_team_marker(text, start, end):
                continue
            if self._is_loadout_noise_surface(text, start, end):
                continue
            repaired = self._repair_slot_sequence(surface)
            if repaired is None:
                continue
            hints.append(surface)
            if len(hints) >= max_hints:
                break
        return hints

    def _has_team_context(self, text: str) -> bool:
        return bool(_TEAM_COMP_TRIGGERS.search(text))

    def _parse_single_char_sequence(self, text: str) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        for m in _SINGLE_CHAR_SEQUENCE.finditer(text):
            seq = m.group(0)
            if not self._is_valid_single_char_seq(seq):
                continue
            for ch in seq:
                entry = self.lexicon.team_slots.get(ch)
                if entry and ch not in _AMBIGUOUS_SLOTS:
                    results.append((ch, entry.canonical_term))
        return results

    def _is_valid_single_char_seq(self, seq: str) -> bool:
        if len(seq) < 2:
            return False
        valid_count = sum(
            1
            for ch in seq
            if ch in self.lexicon.team_slots and ch not in _AMBIGUOUS_SLOTS
        )
        return valid_count >= len(seq) * 0.5

    @staticmethod
    def _iter_candidate_surfaces(text: str):
        for match in _SINGLE_CHAR_SEQUENCE.finditer(text):
            seq = match.group(0)
            length = 2
            for start in range(len(seq) - length + 1):
                surface = seq[start : start + length]
                if "队" in surface:
                    continue
                absolute_start = match.start() + start
                absolute_end = absolute_start + length
                yield surface, absolute_start, absolute_end

    @staticmethod
    def _has_local_team_marker(text: str, start: int, end: int) -> bool:
        before = text[max(0, start - 2):start]
        after = text[end:end + 2]
        return (
            any(after.startswith(marker) for marker in _TEAM_MARKERS_AFTER)
            or any(before.endswith(marker) for marker in _TEAM_MARKERS_BEFORE)
        )

    @staticmethod
    def _is_loadout_noise_surface(text: str, start: int, end: int) -> bool:
        window = text[max(0, start - 4): min(len(text), end + 4)]
        surface = text[start:end]
        return any(term in surface or term in window for term in _LOADOUT_NOISE_TERMS)

    @staticmethod
    def _reviewed_override_candidate(
        text: str,
        suspect_surface: str,
        start: int,
        end: int,
    ) -> dict | None:
        repaired = _REVIEWED_TEAM_COMP_REPAIRS.get(suspect_surface)
        if repaired is None:
            return None
        if not TeamCompParser._has_local_team_marker(text, start, end):
            return None
        value, activated = repaired
        return TeamCompParser._build_requery_candidate(
            surface=suspect_surface,
            value=value,
            activated=activated,
            start=start,
            end=end,
            source="team_comp_reviewed_override",
            score=0.95,
            match_kind="team_comp_reviewed_override",
        )

    @staticmethod
    def _build_requery_candidate(
        surface: str,
        value: str,
        activated: list[str],
        start: int,
        end: int,
        source: str,
        score: float,
        match_kind: str,
    ) -> dict:
        return {
            "surface": surface,
            "value": value,
            "score": score,
            "source": source,
            "category": "team_comp_alias",
            "parent_entity": "|".join(activated),
            "start_char": start,
            "end_char": end,
            "metadata": {
                "intent": "alias_repair_target",
                "activated_entities": activated,
                "match_kind": match_kind,
            },
        }

    def _repair_slot_sequence(self, surface: str) -> tuple[str, list[str]] | None:
        repaired_chars: list[str] = []
        activated: list[str] = []
        changed = False
        exact_slots = 0

        for ch in surface:
            exact = self.lexicon.team_slots.get(ch)
            if exact and ch not in _AMBIGUOUS_SLOTS:
                repaired_chars.append(ch)
                activated.append(exact.canonical_term)
                exact_slots += 1
                continue

            replacement = self._nearest_team_slot(ch)
            if replacement is None:
                return None
            repaired_chars.append(replacement.alias_surface)
            activated.append(replacement.canonical_term)
            changed = True

        if not changed or exact_slots == 0:
            return None
        value = "".join(repaired_chars)
        if value == surface:
            return None
        return value, activated

    def _nearest_team_slot(self, char: str) -> CharacterAliasEntry | None:
        char_py = "".join(lazy_pinyin(char, errors="ignore"))
        if not char_py:
            return None
        best: CharacterAliasEntry | None = None
        best_score = 0
        for alias, entry in self.lexicon.team_slots.items():
            if alias in _AMBIGUOUS_SLOTS:
                continue
            alias_py = "".join(lazy_pinyin(alias, errors="ignore"))
            score = fuzz.ratio(char_py, alias_py)
            if score > best_score:
                best = entry
                best_score = score
        return best if best_score >= 88 else None
