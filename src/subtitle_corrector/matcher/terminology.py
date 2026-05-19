from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rapidfuzz import fuzz, process

from subtitle_corrector.schemas import Candidate

if TYPE_CHECKING:
    from subtitle_corrector.memory.entity import EntityMemoryManager


@dataclass(frozen=True)
class TerminologyEntry:
    term: str
    aliases: tuple[str, ...] = ()
    category: str | None = None
    game_title: str | None = None
    source: str | None = None
    trust_level: float = 0.5
    parent_entity: str | None = None


class TerminologyRepository:
    def list_terms(self) -> list[TerminologyEntry]:
        raise NotImplementedError


class InMemoryTerminologyRepository(TerminologyRepository):
    def __init__(self, entries: list[TerminologyEntry] | None = None) -> None:
        self._entries = entries or []

    def list_terms(self) -> list[TerminologyEntry]:
        return list(self._entries)


class FuzzyTerminologyMatcher:
    def __init__(
        self,
        repository: TerminologyRepository,
        threshold: float = 82.0,
        pinyin_threshold: float = 75.0,
        boost_factor: float = 0.0,
    ) -> None:
        self.repository = repository
        self.threshold = threshold
        self.pinyin_threshold = pinyin_threshold
        self.boost_factor = boost_factor
        self.entity_memory: EntityMemoryManager | None = None
        self._choices_cache: dict[str, TerminologyEntry] | None = None
        self._pinyin_index: dict[str, str] | None = None  # surface -> pinyin

    def _build_choices(self) -> dict[str, TerminologyEntry]:
        if self._choices_cache is None:
            entries = self.repository.list_terms()
            choices: dict[str, TerminologyEntry] = {}
            # Pass 1: register all term names (highest priority).
            for entry in entries:
                choices[entry.term] = entry
            term_keys = set(choices.keys())
            # Pass 2: register aliases that do NOT shadow an existing term.
            for entry in entries:
                for alias in entry.aliases:
                    if alias not in term_keys:
                        choices[alias] = entry
            self._choices_cache = choices
        return self._choices_cache

    def _build_pinyin_index(self) -> dict[str, str]:
        """Build a mapping from each surface form to its pinyin string.

        Lazy-built and cached. This allows pinyin-based matching where
        character-level fuzzy matching fails (e.g. 艾克非 vs 爱可菲).
        """
        if self._pinyin_index is not None:
            return self._pinyin_index

        from pypinyin import lazy_pinyin

        choices = self._build_choices()
        index: dict[str, str] = {}
        for surface in choices:
            if len(surface) < 2:
                continue
            index[surface] = " ".join(lazy_pinyin(surface, errors="ignore"))
        self._pinyin_index = index
        return index

    def _extract_ngrams(self, text: str, min_n: int = 2, max_n: int = 8) -> list[str]:
        """Extract character n-grams from *text* for substring-level matching."""
        ngrams: list[str] = []
        # Strip punctuation so n-grams don't start/end with commas etc.
        clean = "".join(ch for ch in text if ch.isalnum() or ch in "\u4e00-\u9fff")
        for n in range(min_n, min(max_n + 1, len(clean) + 1)):
            for i in range(len(clean) - n + 1):
                ngrams.append(clean[i : i + n])
        return ngrams

    def _apply_entity_boost(self, score: float, entry: TerminologyEntry) -> float:
        """Boost *score* if *entry* belongs to a currently active entity."""
        if self.entity_memory is None or not entry.parent_entity:
            return score
        weight = self.entity_memory.get_weight(entry.parent_entity)
        if weight <= 0:
            return score
        boosted = score + weight * self.boost_factor * 100.0
        return min(boosted, 100.0)

    def get_parent_entities(self) -> set[str]:
        """Return all unique parent_entity values across the terminology."""
        choices = self._build_choices()
        return {
            entry.parent_entity
            for entry in choices.values()
            if entry.parent_entity
        }

    def get_parent_for_term(self, term: str) -> str | None:
        """Return the parent_entity for *term* if one exists in the terminology."""
        choices = self._build_choices()
        entry = choices.get(term)
        return entry.parent_entity if entry else None

    def lookup(self, text: str, limit: int = 8) -> list[Candidate]:
        choices = self._build_choices()
        if not choices:
            return []

        candidates_by_term: dict[str, Candidate] = {}

        # --- Strategy 1: full-text character matching ---
        full_matches = process.extract(
            text, choices.keys(), scorer=fuzz.WRatio, limit=limit
        )
        for matched, score, _ in full_matches:
            if score < self.threshold:
                continue
            entry = choices[matched]
            boosted = self._apply_entity_boost(score, entry)
            self._upsert_candidate(candidates_by_term, entry, matched, boosted)

        # --- Strategy 2: n-gram character matching ---
        ngrams = self._extract_ngrams(text)
        choice_keys = list(choices.keys())
        seen_ngrams: set[str] = set()
        for ngram in ngrams:
            if ngram in seen_ngrams:
                continue
            seen_ngrams.add(ngram)
            if ngram in text and any(
                term in text for term in choice_keys if ngram == term
            ):
                continue
            matches = process.extract(
                ngram, choice_keys, scorer=fuzz.WRatio, limit=3
            )
            for matched, score, _ in matches:
                if score < self.threshold:
                    continue
                if matched in text:
                    continue
                entry = choices[matched]
                boosted = self._apply_entity_boost(score, entry)
                self._upsert_candidate(candidates_by_term, entry, matched, boosted)

        # --- Strategy 3: pinyin-based matching ---
        # Convert n-grams to pinyin and compare against term pinyin strings.
        # This catches ASR errors where the characters are completely
        # different but the pronunciation is identical or very similar
        # (e.g. "艾克非" → "爱可菲", pinyin both "ai ke fei").
        self._pinyin_lookup(text, ngrams, candidates_by_term)

        return sorted(
            candidates_by_term.values(),
            key=lambda c: c.score,
            reverse=True,
        )[:limit]

    def _pinyin_lookup(
        self,
        text: str,
        ngrams: list[str],
        candidates_by_term: dict[str, Candidate],
    ) -> None:
        """Match n-grams by pinyin similarity against the terminology index."""
        from pypinyin import lazy_pinyin

        pinyin_index = self._build_pinyin_index()
        if not pinyin_index:
            return

        choices = self._build_choices()
        pinyin_surfaces = list(pinyin_index.keys())
        pinyin_values = list(pinyin_index.values())

        seen: set[str] = set()
        for ngram in ngrams:
            if ngram in seen or len(ngram) < 2:
                continue
            seen.add(ngram)

            # Skip if the ngram exactly matches a known term already in text
            if ngram in choices and ngram in text:
                continue

            ngram_py = " ".join(lazy_pinyin(ngram, errors="ignore"))
            if not ngram_py.strip():
                continue

            # Compare this n-gram's pinyin against all term pinyin strings
            # using rapidfuzz process.extract for speed
            py_matches = process.extract(
                ngram_py, pinyin_values, scorer=fuzz.ratio, limit=5
            )
            for _, score, idx in py_matches:
                if score < self.threshold:
                    continue
                surface = pinyin_surfaces[idx]
                # Skip if this term appears verbatim in the text (it's correct)
                if surface in text:
                    continue
                # Skip if the ngram itself IS the term (exact match, not an error)
                if ngram == surface:
                    continue
                entry = choices[surface]
                # Pinyin matches get a small bonus for being phonetically motivated
                adjusted_score = min(score * 1.05, 100.0)
                boosted = self._apply_entity_boost(adjusted_score, entry)
                self._upsert_candidate(
                    candidates_by_term,
                    entry,
                    surface,
                    boosted,
                    pinyin_match=f"{ngram}({ngram_py}) ≈ {surface}({pinyin_index[surface]})",
                )

    @staticmethod
    def _upsert_candidate(
        store: dict[str, Candidate],
        entry: TerminologyEntry,
        matched: str,
        score: float,
        pinyin_match: str | None = None,
    ) -> None:
        norm_score = score / 100.0
        existing = store.get(entry.term)
        if existing is not None and existing.score >= norm_score:
            return
        metadata = {
            "category": entry.category,
            "game_title": entry.game_title,
            "source": entry.source,
            "trust_level": entry.trust_level,
            "parent_entity": entry.parent_entity,
        }
        if pinyin_match:
            metadata["matched_pinyin"] = pinyin_match
        store[entry.term] = Candidate(
            value=entry.term,
            source="terminology",
            score=norm_score,
            explanation=f"matched terminology alias: {matched}",
            metadata=metadata,
        )
