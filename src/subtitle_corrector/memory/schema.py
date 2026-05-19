from __future__ import annotations

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS terminology (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'zh',
    category TEXT,
    game_title TEXT,
    source TEXT,
    parent_entity VARCHAR(255),
    trust_level REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS terminology_alias (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    terminology_id INTEGER NOT NULL REFERENCES terminology(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    alias_type TEXT NOT NULL DEFAULT 'surface',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (terminology_id, alias)
);

CREATE TABLE IF NOT EXISTS correction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_text TEXT NOT NULL,
    corrected_text TEXT NOT NULL,
    selected_candidate TEXT,
    confidence REAL NOT NULL,
    reason TEXT,
    source_file TEXT,
    cue_index INTEGER,
    accepted INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entity_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    entity_type TEXT,
    game_title TEXT,
    first_seen_context TEXT,
    last_seen_context TEXT,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (entity, canonical_name, game_title)
);

CREATE INDEX IF NOT EXISTS idx_terminology_game ON terminology(game_title);
CREATE INDEX IF NOT EXISTS idx_alias_value ON terminology_alias(alias);
CREATE INDEX IF NOT EXISTS idx_history_source_text ON correction_history(source_text);
CREATE INDEX IF NOT EXISTS idx_entity_canonical ON entity_memory(canonical_name);
"""
