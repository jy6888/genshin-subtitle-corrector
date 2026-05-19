from __future__ import annotations

import sqlite3
from pathlib import Path

from subtitle_corrector.matcher.terminology import TerminologyEntry, TerminologyRepository
from subtitle_corrector.memory.history import CorrectionHistoryStore
from subtitle_corrector.memory.schema import SCHEMA_SQL
from subtitle_corrector.schemas import RepairResult, Terminology


class SQLiteMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._ensure_migrations(conn)

    @staticmethod
    def _ensure_migrations(conn: sqlite3.Connection) -> None:
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'terminology'"
        ).fetchone()["sql"]
        if "term TEXT NOT NULL UNIQUE" in table_sql:
            SQLiteMemory._rebuild_terminology_without_term_unique(conn)

        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(terminology)").fetchall()
        }
        if "parent_entity" not in columns:
            conn.execute("ALTER TABLE terminology ADD COLUMN parent_entity VARCHAR(255)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_terminology_parent ON terminology(parent_entity)"
        )

    @staticmethod
    def _rebuild_terminology_without_term_unique(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP TABLE IF EXISTS terminology_new")
        conn.execute("DROP TABLE IF EXISTS terminology_alias_new")
        conn.execute(
            """
            CREATE TABLE terminology_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'zh',
                category TEXT,
                game_title TEXT,
                source TEXT,
                trust_level REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                parent_entity VARCHAR(255)
            )
            """
        )
        old_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(terminology)").fetchall()
        }
        parent_select = "parent_entity" if "parent_entity" in old_columns else "NULL"
        conn.execute(
            f"""
            INSERT INTO terminology_new (
                id,
                term,
                language,
                category,
                game_title,
                source,
                trust_level,
                created_at,
                updated_at,
                parent_entity
            )
            SELECT
                id,
                term,
                language,
                category,
                game_title,
                source,
                trust_level,
                created_at,
                updated_at,
                {parent_select}
            FROM terminology
            """
        )
        conn.execute(
            """
            CREATE TABLE terminology_alias_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                terminology_id INTEGER NOT NULL REFERENCES terminology(id) ON DELETE CASCADE,
                alias TEXT NOT NULL,
                alias_type TEXT NOT NULL DEFAULT 'surface',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (terminology_id, alias)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO terminology_alias_new (
                id,
                terminology_id,
                alias,
                alias_type,
                created_at
            )
            SELECT id, terminology_id, alias, alias_type, created_at
            FROM terminology_alias
            """
        )
        conn.execute("DROP TABLE terminology_alias")
        conn.execute("DROP TABLE terminology")
        conn.execute("ALTER TABLE terminology_new RENAME TO terminology")
        conn.execute("ALTER TABLE terminology_alias_new RENAME TO terminology_alias")
        conn.execute("PRAGMA foreign_keys = ON")


class SQLiteTerminologyRepository(TerminologyRepository):
    def __init__(self, memory: SQLiteMemory) -> None:
        self.memory = memory

    def list_terms(self) -> list[TerminologyEntry]:
        with self.memory.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    t.id,
                    t.term,
                    t.category,
                    t.game_title,
                    t.source,
                    t.trust_level,
                    t.parent_entity,
                    a.alias
                FROM terminology t
                LEFT JOIN terminology_alias a ON a.terminology_id = t.id
                ORDER BY t.id
                """
            ).fetchall()
        grouped: dict[int, dict[str, object]] = {}
        for row in rows:
            item = grouped.setdefault(
                row["id"],
                {
                    "term": row["term"],
                    "category": row["category"],
                    "game_title": row["game_title"],
                    "source": row["source"],
                    "trust_level": row["trust_level"],
                    "parent_entity": row["parent_entity"],
                    "aliases": [],
                },
            )
            if row["alias"]:
                item["aliases"].append(row["alias"])
        return [
            TerminologyEntry(
                term=str(item["term"]),
                aliases=tuple(item["aliases"]),
                category=item["category"],
                game_title=item["game_title"],
                source=item["source"],
                trust_level=float(item["trust_level"]),
                parent_entity=item["parent_entity"],
            )
            for item in grouped.values()
        ]


class SQLiteTerminologyStore:
    def __init__(self, memory: SQLiteMemory) -> None:
        self.memory = memory

    def upsert(self, item: Terminology) -> int:
        with self.memory.connect() as conn:
            existing = conn.execute(
                """
                SELECT id
                FROM terminology
                WHERE term = ?
                  AND COALESCE(category, '') = COALESCE(?, '')
                  AND COALESCE(game_title, '') = COALESCE(?, '')
                  AND COALESCE(parent_entity, '') = COALESCE(?, '')
                """,
                (item.term, item.category, item.game_title, item.parent_entity),
            ).fetchone()
            if existing:
                terminology_id = existing["id"]
                conn.execute(
                    """
                    UPDATE terminology
                    SET source = ?,
                        trust_level = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (item.source, item.trust_level, terminology_id),
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO terminology (
                        term,
                        category,
                        game_title,
                        source,
                        trust_level,
                        parent_entity,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        item.term,
                        item.category,
                        item.game_title,
                        item.source,
                        item.trust_level,
                        item.parent_entity,
                    ),
                )
                terminology_id = cursor.lastrowid
            for alias in item.aliases:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO terminology_alias (terminology_id, alias)
                    VALUES (?, ?)
                    """,
                    (terminology_id, alias),
                )
            return int(terminology_id)


class SQLiteCorrectionHistoryStore(CorrectionHistoryStore):
    def __init__(self, memory: SQLiteMemory) -> None:
        self.memory = memory

    def record(self, repair: RepairResult, source_file: str | None = None) -> None:
        with self.memory.connect() as conn:
            conn.execute(
                """
                INSERT INTO correction_history (
                    source_text,
                    corrected_text,
                    selected_candidate,
                    confidence,
                    reason,
                    source_file,
                    cue_index,
                    accepted
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    repair.original_text,
                    repair.repaired_text,
                    repair.repaired_text if repair.original_text != repair.repaired_text else None,
                    repair.confidence,
                    repair.explanation,
                    source_file,
                    repair.cue_index,
                    None,
                ),
            )
