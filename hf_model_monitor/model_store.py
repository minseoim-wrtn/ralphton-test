"""SQLite-based local state store for tracking known HuggingFace models.

Provides persistent storage with timestamps to determine new vs already-seen
models.  Designed for daily batch polling by a single PM — no concurrency
concerns, simple schema, easy to inspect with any SQLite browser.

Usage:
    store = ModelStore()                      # uses default path
    store = ModelStore("path/to/models.db")   # custom path

    # Record models from a poll
    new = store.upsert_models(model_records)

    # Check if a model has been seen before
    store.is_known("meta-llama/Llama-4-Scout-17B-16E")

    # Query
    store.get_model("meta-llama/Llama-4-Scout-17B-16E")
    store.get_all_models()
    store.get_models_since("2025-01-01T00:00:00")
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "models.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS known_models (
    model_id       TEXT PRIMARY KEY,
    author         TEXT NOT NULL DEFAULT '',
    first_seen_at  TEXT NOT NULL,
    last_checked_at TEXT NOT NULL,
    created_at     TEXT DEFAULT 'N/A',
    last_modified  TEXT DEFAULT 'N/A',
    pipeline_tag   TEXT DEFAULT 'N/A',
    library_name   TEXT DEFAULT 'N/A',
    downloads      INTEGER DEFAULT 0,
    likes          INTEGER DEFAULT 0,
    tags           TEXT DEFAULT '[]',
    status         TEXT DEFAULT 'new',
    metadata_json  TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS poll_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    poll_timestamp  TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT '',
    models_found    INTEGER DEFAULT 0,
    new_models      INTEGER DEFAULT 0,
    errors          TEXT DEFAULT ''
);
"""

# v2: added 'notified' column to track whether Slack notification was sent
_MIGRATIONS = [
    # Migration 1: add 'notified' column (safe to re-run — IF NOT EXISTS)
    """
    ALTER TABLE known_models ADD COLUMN notified INTEGER DEFAULT 0;
    """,
]


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------
class ModelStore:
    """SQLite-backed store for tracking known HuggingFace models.

    All timestamps are stored as ISO-8601 UTC strings for readability and
    portability (easy to inspect in any SQLite browser or export to JSON).
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema & migrations
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create tables if they don't exist and run pending migrations."""
        self._conn.executescript(_SCHEMA_SQL)
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Apply schema migrations that haven't been applied yet."""
        for i, sql in enumerate(_MIGRATIONS):
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError as exc:
                # "duplicate column name" means migration already applied
                if "duplicate column" in str(exc).lower():
                    continue
                logger.warning("Migration %d skipped: %s", i, exc)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def is_known(self, model_id: str) -> bool:
        """Check whether a model has been seen before."""
        cursor = self._conn.execute(
            "SELECT 1 FROM known_models WHERE model_id = ?", (model_id,)
        )
        return cursor.fetchone() is not None

    def get_model(self, model_id: str) -> dict | None:
        """Retrieve a single model record as a dict, or None if not found."""
        cursor = self._conn.execute(
            "SELECT * FROM known_models WHERE model_id = ?", (model_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def get_all_models(self) -> list[dict]:
        """Return all known models, ordered by first_seen_at descending."""
        cursor = self._conn.execute(
            "SELECT * FROM known_models ORDER BY first_seen_at DESC"
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_models_since(self, since_iso: str) -> list[dict]:
        """Return models first seen on or after *since_iso* (ISO-8601)."""
        cursor = self._conn.execute(
            "SELECT * FROM known_models WHERE first_seen_at >= ? "
            "ORDER BY first_seen_at DESC",
            (since_iso,),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_new_models(self) -> list[dict]:
        """Return models with status='new' (not yet fully processed)."""
        cursor = self._conn.execute(
            "SELECT * FROM known_models WHERE status = 'new' "
            "ORDER BY first_seen_at DESC"
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def count_models(self) -> int:
        """Return the total number of known models."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM known_models")
        return cursor.fetchone()[0]

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_models(self, models: list[dict]) -> list[dict]:
        """Insert or update a batch of models.  Returns list of NEW models.

        For each model:
        - If not seen before → insert with status='new', record first_seen_at
        - If already seen → update last_checked_at, downloads, likes

        Args:
            models: List of dicts with at minimum a 'model_id' key.
                    Accepted keys match ModelRecord.to_dict() output.

        Returns:
            List of model dicts that were newly inserted (not seen before).
        """
        now = _now_iso()
        new_models: list[dict] = []

        for model in models:
            model_id = model.get("model_id", "")
            if not model_id:
                continue

            if self.is_known(model_id):
                self._update_existing(model, now)
            else:
                self._insert_new(model, now)
                new_models.append(model)

        self._conn.commit()

        if new_models:
            logger.info(
                "Upserted %d models: %d new, %d updated",
                len(models),
                len(new_models),
                len(models) - len(new_models),
            )

        return new_models

    def upsert_model(self, model: dict) -> bool:
        """Insert or update a single model.  Returns True if it was new."""
        result = self.upsert_models([model])
        return len(result) > 0

    def _insert_new(self, model: dict, now: str) -> None:
        """Insert a model record that hasn't been seen before."""
        model_id = model["model_id"]
        author = model.get("author", "")
        if not author and "/" in model_id:
            author = model_id.split("/")[0]

        tags = model.get("tags", [])
        tags_json = json.dumps(tags) if isinstance(tags, list) else "[]"

        # Store any extra fields in metadata_json for extensibility
        metadata = {
            k: v
            for k, v in model.items()
            if k
            not in (
                "model_id",
                "author",
                "created_at",
                "last_modified",
                "pipeline_tag",
                "library_name",
                "downloads",
                "likes",
                "tags",
                "private",
                "gated",
            )
        }

        self._conn.execute(
            """
            INSERT INTO known_models
                (model_id, author, first_seen_at, last_checked_at,
                 created_at, last_modified, pipeline_tag, library_name,
                 downloads, likes, tags, status, metadata_json, notified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, 0)
            """,
            (
                model_id,
                author,
                now,
                now,
                model.get("created_at", "N/A"),
                model.get("last_modified", "N/A"),
                model.get("pipeline_tag", "N/A"),
                model.get("library_name", "N/A"),
                model.get("downloads", 0),
                model.get("likes", 0),
                tags_json,
                json.dumps(metadata),
            ),
        )

    def _update_existing(self, model: dict, now: str) -> None:
        """Update a model that was already seen — refresh mutable fields."""
        self._conn.execute(
            """
            UPDATE known_models
            SET last_checked_at = ?,
                downloads = ?,
                likes = ?,
                last_modified = CASE
                    WHEN ? != 'N/A' THEN ? ELSE last_modified END
            WHERE model_id = ?
            """,
            (
                now,
                model.get("downloads", 0),
                model.get("likes", 0),
                model.get("last_modified", "N/A"),
                model.get("last_modified", "N/A"),
                model["model_id"],
            ),
        )

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def mark_as_known(self, model_id: str) -> None:
        """Mark a model as fully processed (status → 'known')."""
        self._conn.execute(
            "UPDATE known_models SET status = 'known' WHERE model_id = ?",
            (model_id,),
        )
        self._conn.commit()

    def mark_as_notified(self, model_id: str) -> None:
        """Mark a model as having had its Slack notification sent."""
        self._conn.execute(
            "UPDATE known_models SET notified = 1 WHERE model_id = ?",
            (model_id,),
        )
        self._conn.commit()

    def get_unnotified_models(self) -> list[dict]:
        """Return models that haven't had Slack notifications sent yet."""
        cursor = self._conn.execute(
            "SELECT * FROM known_models WHERE notified = 0 "
            "ORDER BY first_seen_at DESC"
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Poll history
    # ------------------------------------------------------------------

    def record_poll(
        self,
        source: str,
        models_found: int,
        new_models: int,
        errors: str = "",
    ) -> None:
        """Record a poll event in the history table."""
        self._conn.execute(
            """
            INSERT INTO poll_history
                (poll_timestamp, source, models_found, new_models, errors)
            VALUES (?, ?, ?, ?, ?)
            """,
            (_now_iso(), source, models_found, new_models, errors),
        )
        self._conn.commit()

    def get_poll_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent poll history entries."""
        cursor = self._conn.execute(
            "SELECT * FROM poll_history ORDER BY poll_timestamp DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_last_poll_time(self) -> str | None:
        """Return the timestamp of the most recent poll, or None."""
        cursor = self._conn.execute(
            "SELECT poll_timestamp FROM poll_history "
            "ORDER BY poll_timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row["poll_timestamp"] if row else None

    # ------------------------------------------------------------------
    # Migration from JSON
    # ------------------------------------------------------------------

    def import_from_json(self, json_path: str) -> int:
        """Import model IDs from the legacy seen_models.json file.

        Models imported this way get status='known' and first_seen_at set to
        the file's modification time (best available approximation).

        Returns:
            Number of models imported.
        """
        if not os.path.exists(json_path):
            logger.info("No legacy JSON file at %s, nothing to import", json_path)
            return 0

        try:
            with open(json_path) as f:
                model_ids = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to read legacy JSON file %s", json_path)
            return 0

        if not isinstance(model_ids, list):
            logger.warning("Legacy JSON file does not contain a list")
            return 0

        # Use file mtime as approximate first_seen_at
        try:
            mtime = os.path.getmtime(json_path)
            import_time = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except OSError:
            import_time = _now_iso()

        imported = 0
        for mid in model_ids:
            if not isinstance(mid, str) or not mid:
                continue
            if self.is_known(mid):
                continue

            author = mid.split("/")[0] if "/" in mid else ""
            self._conn.execute(
                """
                INSERT INTO known_models
                    (model_id, author, first_seen_at, last_checked_at,
                     status, notified)
                VALUES (?, ?, ?, ?, 'known', 1)
                """,
                (mid, author, import_time, import_time),
            )
            imported += 1

        self._conn.commit()
        logger.info(
            "Imported %d models from legacy JSON (%d were already known)",
            imported,
            len(model_ids) - imported,
        )
        return imported

    # ------------------------------------------------------------------
    # Export (for debugging / backup)
    # ------------------------------------------------------------------

    def export_model_ids(self) -> list[str]:
        """Export all known model IDs as a flat list (JSON-compatible)."""
        cursor = self._conn.execute(
            "SELECT model_id FROM known_models ORDER BY first_seen_at DESC"
        )
        return [row["model_id"] for row in cursor.fetchall()]

    def export_to_json(self, json_path: str) -> None:
        """Export all model IDs to a JSON file (backward-compatible format)."""
        ids = self.export_model_ids()
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(ids, f, indent=2)

    # ------------------------------------------------------------------
    # Filtering & search
    # ------------------------------------------------------------------

    def get_models_by_author(self, author: str) -> list[dict]:
        """Return all models from a specific author/org."""
        cursor = self._conn.execute(
            "SELECT * FROM known_models WHERE author = ? "
            "ORDER BY first_seen_at DESC",
            (author,),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_models_by_status(self, status: str) -> list[dict]:
        """Return all models with a specific status ('new' or 'known')."""
        cursor = self._conn.execute(
            "SELECT * FROM known_models WHERE status = ? "
            "ORDER BY first_seen_at DESC",
            (status,),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def search_models(self, query: str) -> list[dict]:
        """Search models by model_id substring (case-insensitive)."""
        cursor = self._conn.execute(
            "SELECT * FROM known_models WHERE model_id LIKE ? "
            "ORDER BY first_seen_at DESC",
            (f"%{query}%",),
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_model(self, model_id: str) -> bool:
        """Remove a model from the store. Returns True if it existed."""
        cursor = self._conn.execute(
            "DELETE FROM known_models WHERE model_id = ?", (model_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict with parsed JSON fields."""
        d = dict(row)
        # Parse JSON-encoded fields back to Python objects
        for json_field in ("tags", "metadata_json"):
            if json_field in d and isinstance(d[json_field], str):
                try:
                    d[json_field] = json.loads(d[json_field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
