from __future__ import annotations

import threading

import jieba
from pypinyin import lazy_pinyin
from rapidfuzz import fuzz

from subtitle_corrector.detector.base import Detector
from subtitle_corrector.matcher.terminology import FuzzyTerminologyMatcher
from subtitle_corrector.schemas import Candidate, DetectionResult, Span, SubtitleCue, SubtitleDocument

_JIEBA_INJECTION_LOCK = threading.RLock()


class JiebaSpanDetector(Detector):
    """Detector that uses jieba tokenization to find ASR errors with
    precise character coordinates (spans).

    Level 1: exact match via jieba + terminology lookup.
    Level 2: pinyin-based fuzzy match with dynamic threshold (stricter
             for short tokens) and hot/cold pool filtering.
    """

    name = "jieba_span"

    def __init__(
        self,
        matcher: FuzzyTerminologyMatcher,
        hot_categories: set[str] | None = None,
        cold_categories: set[str] | None = None,
        category_activation: object | None = None,
        alias_lexicon: object | None = None,
        character_lexicon: object | None = None,
    ) -> None:
        self.matcher = matcher
        self.hot_categories = hot_categories or set()
        self.cold_categories = cold_categories or set()
        self.category_activation = category_activation
        self.alias_lexicon = alias_lexicon
        self.character_lexicon = character_lexicon
        self._terms_injected = False

    def _ensure_terms_injected(self) -> None:
        """Lazily inject all terminology surface forms (len >= 2) into
        jieba's dictionary so they are recognised during tokenization.

        Thread-safe: double-checked locking with module-level RLock so
        concurrent DiscoveryEngine workers never race on the global TRIE.
        """
        if self._terms_injected:
            return
        with _JIEBA_INJECTION_LOCK:
            if self._terms_injected:
                return
            choices = self.matcher._build_choices()
            for surface in choices:
                if len(surface) >= 2:
                    jieba.add_word(surface)
            self._terms_injected = True

    def _effective_threshold(self, word: str) -> float:
        """Dynamic threshold for exact matching: shorter words need stricter."""
        if len(word) <= 2:
            return 90.0
        if len(word) == 3:
            return 86.0
        return self.matcher.threshold  # 82.0 for 4+ chars

    def _effective_pinyin_threshold(self, word: str) -> float:
        """Pinyin matching threshold — lower than exact match, length-based.

        Pinyin edit distances are inherently larger, so thresholds are ~10
        points lower than exact match.  Scope is already limited to active
        entity terms, so false-positive risk is low.
        """
        if len(word) <= 2:
            return 80.0   # catches homophones (ratio≈100), filters noise
        if len(word) == 3:
            return 76.0   # catches 弗丽娜→芙宁娜 (77.8), 芙琳娜→芙宁娜 (84.2)
        return 72.0       # 4+ chars: even larger edit distances tolerated

    def _category_for_entry(self, entry) -> str:
        """Return the category string for a TerminologyEntry."""
        return getattr(entry, "category", "") or ""

    def _category_is_hot(self, entry) -> bool:
        return self._category_for_entry(entry) in self.hot_categories

    def _nearest_nickname_pinyin(self, word: str):
        """搜索与 word 拼音最接近的角色多字外号。

        Returns (ratio, CharacterAliasEntry) 或 None。
        """
        if self.character_lexicon is None:
            return None
        word_py = " ".join(lazy_pinyin(word, errors="ignore"))
        if not word_py.strip():
            return None
        threshold = self._effective_threshold(word)
        best_ratio = 0.0
        best_entry = None
        for surface, entry in self.character_lexicon.nicknames.items():
            surface_py = " ".join(lazy_pinyin(surface, errors="ignore"))
            ratio = fuzz.ratio(word_py, surface_py)
            if ratio >= threshold and ratio > best_ratio:
                best_ratio = ratio
                best_entry = entry
        if best_entry is not None:
            return (best_ratio, best_entry)
        return None

    def detect(self, cue: SubtitleCue, document: SubtitleDocument) -> DetectionResult:
        self._ensure_terms_injected()
        choices = self.matcher._build_choices()
        pinyin_index = self.matcher._build_pinyin_index()

        tokens = list(jieba.tokenize(cue.text, mode="search"))

        # Determine which cold categories are currently active
        active_cold: set[str] = set()
        if self.category_activation is not None:
            active_cold = self.category_activation.active_categories

        # Active entities: only pinyin-match against terms belonging to
        # entities that EntityMemory has confirmed (plus general terms
        # that have no parent_entity).
        active_entities: set[str] = set()
        if self.matcher.entity_memory is not None:
            active_entities = {
                name for name, w in self.matcher.entity_memory.active_entities.items()
                if w > 0
            }

        # Track best span per term for deduplication
        best_spans: dict[str, Span] = {}

        # Per-cold-category probe tracking: best span per cold category
        cold_probes: dict[str, Span] = {}

        for word, start, end in tokens:
            if len(word) < 2:
                continue

            term_entry = choices.get(word)

            intent = "repair_candidate"
            expansion_policy = "repair_to_canonical"
            evidence_type = "exact_term"
            # --- Level 1: exact match ---
            if term_entry is not None:
                raw_score = 100.0
                source = "exact"
                matched_entry = term_entry
                if self.alias_lexicon is not None and word in self.alias_lexicon:
                    policy = self.alias_lexicon.get_policy(word)
                    if policy in ("context_only", "exact_context_only"):
                        intent = "context_alias"
                        expansion_policy = "preserve_surface"
                        evidence_type = "known_alias"
                    candidate_value = word  # 别名 surface 自身，不扩写
                elif (
                    self.character_lexicon is not None
                    and word in self.character_lexicon.nicknames
                ):
                    # 多字角色外号可作为修复候选，指向标准名
                    entry = self.character_lexicon.nicknames[word]
                    candidate_value = entry.canonical_term
                    intent = "repair_candidate"
                    expansion_policy = "repair_to_canonical"
                    evidence_type = "character_nickname"
                else:
                    candidate_value = term_entry.term
            else:
                # --- Level 2: pinyin match ---
                token_pinyin = " ".join(lazy_pinyin(word, errors="ignore"))
                if not token_pinyin.strip():
                    continue

                effective_threshold = self._effective_pinyin_threshold(word)

                best_ratio = 0.0
                best_entry = None
                for surface, surface_pinyin in pinyin_index.items():
                    ratio = fuzz.ratio(token_pinyin, surface_pinyin)
                    if ratio < effective_threshold or ratio <= best_ratio:
                        continue
                    entry = choices[surface]
                    cat = self._category_for_entry(entry)

                    # Cold pool filtering: skip dormant cold categories
                    if cat in self.cold_categories and cat not in active_cold:
                        continue

                    # Entity scope: skip terms whose parent_entity is NOT
                    # currently active.  Short words (2 chars) require active
                    # entity context — no parent_entity means no context,
                    # so they are excluded from pinyin matching.
                    pe = getattr(entry, "parent_entity", None)
                    if len(word) <= 2:
                        if not pe or pe not in active_entities:
                            continue
                    elif pe and pe not in active_entities:
                        continue

                    best_ratio = ratio
                    best_entry = entry

                if best_entry is None:
                    # --- Probe: check cold (dormant) categories ---
                    best_probe_ratio = 0.0
                    best_probe_entry = None
                    for surface, surface_pinyin in pinyin_index.items():
                        ratio = fuzz.ratio(token_pinyin, surface_pinyin)
                        if ratio < effective_threshold or ratio <= best_probe_ratio:
                            continue
                        entry = choices[surface]
                        cat = self._category_for_entry(entry)

                        if cat not in self.cold_categories or cat in active_cold:
                            continue  # skip hot or already-active cold

                        best_probe_ratio = ratio
                        best_probe_entry = entry

                    if best_probe_entry is not None:
                        probe_cat = self._category_for_entry(best_probe_entry)
                        raw_score = min(best_probe_ratio * 1.05, 100.0)
                        span = Span(
                            surface_text=word,
                            start_char=start,
                            end_char=end,
                            candidate_value=best_probe_entry.term,
                            score=min(raw_score / 100.0, 1.0),
                            source="pinyin_probe",
                            metadata={
                                "category": best_probe_entry.category,
                                "parent_entity": best_probe_entry.parent_entity,
                                "expansion_policy": "repair_to_canonical",
                                "evidence_type": "pinyin_probe",
                            },
                        )
                        # Keep only the best probe per cold category
                        existing = cold_probes.get(probe_cat)
                        if existing is None or span.score > existing.score:
                            cold_probes[probe_cat] = span
                    # --- character nickname pinyin fallback ---
                    if best_entry is None and self.character_lexicon is not None:
                        best_nick = self._nearest_nickname_pinyin(word)
                        if best_nick is not None:
                            ratio, nick_entry = best_nick
                            score = min(min(ratio * 1.05, 100.0) / 100.0, 1.0)
                            span = Span(
                                surface_text=word,
                                start_char=start,
                                end_char=end,
                                candidate_value=nick_entry.alias_surface,
                                score=score,
                                source="pinyin",
                                metadata={
                                    "category": "character",
                                    "parent_entity": nick_entry.canonical_term,
                                    "intent": "repair_candidate",
                                    "surface_text": word,
                                    "expansion_policy": "repair_to_canonical",
                                    "evidence_type": "character_nickname_pinyin",
                                },
                            )
                            key = nick_entry.alias_surface
                            if key not in best_spans or span.score > best_spans[key].score:
                                best_spans[key] = span
                    continue

                raw_score = min(best_ratio * 1.05, 100.0)
                source = "pinyin"
                candidate_value = best_entry.term
                matched_entry = best_entry
                expansion_policy = "repair_to_canonical"
                evidence_type = "pinyin_match"

            # Apply entity memory boost
            boosted = self.matcher._apply_entity_boost(raw_score, matched_entry)
            final_score = min(boosted / 100.0, 1.0)

            span = Span(
                surface_text=word,
                start_char=start,
                end_char=end,
                candidate_value=candidate_value,
                score=final_score,
                source=source,
                metadata={
                    "category": matched_entry.category,
                    "parent_entity": matched_entry.parent_entity,
                    "intent": intent,
                    "surface_text": word,
                    "expansion_policy": expansion_policy,
                    "evidence_type": evidence_type,
                },
            )

            term_key = matched_entry.term
            if term_key not in best_spans or span.score > best_spans[term_key].score:
                best_spans[term_key] = span

        # Merge hot/active-cold spans with cold probes
        all_spans = list(best_spans.values()) + list(cold_probes.values())

        if not all_spans:
            return DetectionResult(
                detector=self.name,
                cue_index=cue.index,
                risk_score=0.0,
                reason="no spans matched",
            )

        risk_score = max(s.score for s in all_spans)

        candidates: list[Candidate] = []
        for s in all_spans:
            candidates.append(
                Candidate(
                    value=s.candidate_value,
                    source=s.source,
                    score=s.score,
                    explanation=f"matched via jieba span: {s.surface_text} -> {s.candidate_value}",
                    metadata=s.metadata,
                )
            )

        return DetectionResult(
            detector=self.name,
            cue_index=cue.index,
            risk_score=risk_score,
            reason=f"found {len(all_spans)} jieba span(s)",
            candidates=candidates,
            metadata={"spans": [s.model_dump() for s in all_spans]},
        )
