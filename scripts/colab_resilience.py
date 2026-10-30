"""Durable recovery primitives for the CREMA-D Colab experiment.

The module intentionally uses SQLite only for operational state and evidence
metadata. Large arrays, model checkpoints, figures, and reports remain files
on Google Drive and are referenced by path, size, and SHA-256 when practical.

The code is standard-library first so the same implementation can be tested
locally and embedded verbatim in a generated Colab notebook.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import traceback as traceback_module
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional


SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return a sortable UTC timestamp without requiring third-party modules."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def safe_stage_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "stage"


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_is_valid(path: os.PathLike[str] | str, min_bytes: int = 1) -> bool:
    candidate = Path(path)
    return candidate.is_file() and candidate.stat().st_size >= min_bytes


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")


def atomic_write_bytes(path: os.PathLike[str] | str, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(target)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def atomic_write_text(
    path: os.PathLike[str] | str, text: str, encoding: str = "utf-8"
) -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> Path:
    return atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def atomic_npz_save(path: os.PathLike[str] | str, **arrays: Any) -> Path:
    """Save a compressed NumPy archive through an atomic sibling rename."""

    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f"{target.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def atomic_torch_save(payload: Any, path: os.PathLike[str] | str) -> Path:
    """Save a torch payload atomically without importing torch at module import."""

    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(target)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


class SQLiteLedger:
    """Small transactional ledger designed for a Google Drive-backed Colab run."""

    def __init__(self, path: os.PathLike[str] | str, busy_timeout_ms: int = 30_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = int(busy_timeout_ms)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=max(1.0, self.busy_timeout_ms / 1000),
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection: Optional[sqlite3.Connection] = None
        last_error: Optional[sqlite3.OperationalError] = None
        for attempt in range(6):
            candidate = self._connect()
            try:
                candidate.execute("BEGIN IMMEDIATE")
                connection = candidate
                break
            except sqlite3.OperationalError as exc:
                candidate.close()
                last_error = exc
                if "locked" not in str(exc).lower() or attempt == 5:
                    raise
                time.sleep(0.2 * (2**attempt))
        if connection is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Could not open SQLite transaction")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            experiment_name TEXT NOT NULL,
            config_hash TEXT,
            config_json TEXT,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS stages (
            run_id TEXT NOT NULL,
            stage_id TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            completed_at TEXT,
            duration_sec REAL,
            error_type TEXT,
            error_message TEXT,
            traceback TEXT,
            required_artifacts_json TEXT,
            metadata_json TEXT,
            PRIMARY KEY (run_id, stage_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            stage_id TEXT,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS items (
            run_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            item_key TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 1,
            path TEXT,
            bytes INTEGER,
            sha256 TEXT,
            error TEXT,
            metadata_json TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, domain, item_key),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS training_epochs (
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            metrics_json TEXT NOT NULL,
            checkpoint_path TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, model_name, epoch),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            run_id TEXT NOT NULL,
            stage_id TEXT NOT NULL,
            path TEXT NOT NULL,
            kind TEXT,
            bytes INTEGER NOT NULL,
            sha256 TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, stage_id, path),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_runs_experiment_status
            ON runs(experiment_name, status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_events_run_stage
            ON events(run_id, stage_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_items_run_domain_status
            ON items(run_id, domain, status);
        """
        with contextlib.closing(self._connect()) as connection:
            # DELETE is more conservative than WAL on a mounted Google Drive
            # filesystem. Set it once during initialization to avoid requiring
            # an exclusive lock for every heartbeat connection.
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.executescript(schema)
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            connection.commit()

    def close(self) -> None:
        """Connections are short-lived; the method exists for a stable test API."""

    @staticmethod
    def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        return dict(row) if row is not None else None

    def start_run(
        self,
        run_id: str,
        experiment_name: str,
        config_hash: str = "",
        config: Optional[Mapping[str, Any]] = None,
    ) -> str:
        now = utc_now()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, experiment_name, config_hash, config_json,
                    status, started_at, updated_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    experiment_name=excluded.experiment_name,
                    config_hash=excluded.config_hash,
                    config_json=excluded.config_json,
                    status='running',
                    updated_at=excluded.updated_at,
                    error=NULL
                """,
                (
                    str(run_id),
                    str(experiment_name),
                    str(config_hash or ""),
                    json_dumps(dict(config or {})),
                    now,
                    now,
                ),
            )
        return str(run_id)

    def start_or_resume_run(
        self,
        experiment_name: str,
        preferred_run_id: Optional[str] = None,
        config_hash: str = "",
        config: Optional[Mapping[str, Any]] = None,
        force_new: bool = False,
    ) -> tuple[str, bool]:
        if preferred_run_id and not force_new:
            with contextlib.closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM runs WHERE run_id=?", (str(preferred_run_id),)
                ).fetchone()
            if row is not None and row["status"] != "completed":
                self.start_run(
                    str(preferred_run_id), experiment_name, config_hash, config
                )
                return str(preferred_run_id), True
            if row is not None and row["status"] == "completed":
                # The active pointer is authoritative. Once it is complete, a
                # fresh Run all must not resurrect an older failed execution.
                force_new = True

        if not force_new:
            query = """
                SELECT * FROM runs
                WHERE experiment_name=? AND status IN ('running','failed','interrupted')
            """
            params: list[Any] = [str(experiment_name)]
            if config_hash:
                query += " AND config_hash=?"
                params.append(str(config_hash))
            query += " ORDER BY updated_at DESC LIMIT 1"
            with contextlib.closing(self._connect()) as connection:
                row = connection.execute(query, params).fetchone()
            if row is not None:
                self.start_run(row["run_id"], experiment_name, config_hash, config)
                return str(row["run_id"]), True

        new_run_id = preferred_run_id or time.strftime(
            "%Y%m%d_%H%M%S", time.localtime()
        )
        if force_new or preferred_run_id is None:
            new_run_id = f"{new_run_id}_{uuid.uuid4().hex[:8]}"
        self.start_run(new_run_id, experiment_name, config_hash, config)
        return str(new_run_id), False

    def mark_run(self, run_id: str, status: str, error: Optional[str] = None) -> None:
        now = utc_now()
        completed_at = now if status == "completed" else None
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status=?, updated_at=?, completed_at=?, error=?
                WHERE run_id=?
                """,
                (str(status), now, completed_at, error, str(run_id)),
            )

    def begin_stage(
        self,
        run_id: str,
        stage_id: str,
        required_artifacts: Iterable[os.PathLike[str] | str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        now = utc_now()
        required = [str(Path(path)) for path in required_artifacts]
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO stages(
                    run_id, stage_id, status, attempts, started_at,
                    required_artifacts_json, metadata_json
                ) VALUES (?, ?, 'running', 1, ?, ?, ?)
                ON CONFLICT(run_id, stage_id) DO UPDATE SET
                    status='running',
                    attempts=stages.attempts + 1,
                    started_at=excluded.started_at,
                    completed_at=NULL,
                    duration_sec=NULL,
                    error_type=NULL,
                    error_message=NULL,
                    traceback=NULL,
                    required_artifacts_json=excluded.required_artifacts_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    str(run_id),
                    str(stage_id),
                    now,
                    json_dumps(required),
                    json_dumps(dict(metadata or {})),
                ),
            )

    def complete_stage(
        self,
        run_id: str,
        stage_id: str,
        duration_sec: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        now = utc_now()
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE stages
                SET status='completed', completed_at=?, duration_sec=?,
                    error_type=NULL, error_message=NULL, traceback=NULL,
                    metadata_json=COALESCE(?, metadata_json)
                WHERE run_id=? AND stage_id=?
                """,
                (
                    now,
                    duration_sec,
                    json_dumps(dict(metadata)) if metadata is not None else None,
                    str(run_id),
                    str(stage_id),
                ),
            )
            connection.execute(
                "UPDATE runs SET status='running', updated_at=?, error=NULL WHERE run_id=?",
                (now, str(run_id)),
            )

    def fail_stage(
        self,
        run_id: str,
        stage_id: str,
        error: BaseException,
        traceback_text: str,
        duration_sec: Optional[float] = None,
    ) -> None:
        now = utc_now()
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE stages
                SET status='failed', completed_at=?, duration_sec=?,
                    error_type=?, error_message=?, traceback=?
                WHERE run_id=? AND stage_id=?
                """,
                (
                    now,
                    duration_sec,
                    type(error).__name__,
                    str(error)[:4000],
                    traceback_text,
                    str(run_id),
                    str(stage_id),
                ),
            )
            connection.execute(
                "UPDATE runs SET status='failed', updated_at=?, error=? WHERE run_id=?",
                (now, f"{stage_id}: {type(error).__name__}: {error}"[:4000], str(run_id)),
            )

    def get_stage(self, run_id: str, stage_id: str) -> Optional[dict[str, Any]]:
        with contextlib.closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM stages WHERE run_id=? AND stage_id=?",
                (str(run_id), str(stage_id)),
            ).fetchone()
        return self._row_to_dict(row)

    def stage_is_reusable(
        self,
        run_id: str,
        stage_id: str,
        required_artifacts: Iterable[os.PathLike[str] | str] = (),
        min_bytes: int = 1,
    ) -> bool:
        row = self.get_stage(run_id, stage_id)
        if row is None or row["status"] != "completed":
            return False
        return all(file_is_valid(path, min_bytes=min_bytes) for path in required_artifacts)

    def reset_stage(self, run_id: str, stage_id: str) -> None:
        """Reset ledger state only. Artifact deletion is deliberately not implicit."""

        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE stages
                SET status='pending', started_at=NULL, completed_at=NULL,
                    duration_sec=NULL, error_type=NULL, error_message=NULL, traceback=NULL
                WHERE run_id=? AND stage_id=?
                """,
                (str(run_id), str(stage_id)),
            )

    def event(
        self,
        run_id: str,
        level: str,
        message: str,
        stage_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO events(run_id, timestamp, stage_id, level, message, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    utc_now(),
                    stage_id,
                    str(level),
                    str(message),
                    json_dumps(dict(payload or {})),
                ),
            )

    def upsert_item(
        self,
        run_id: str,
        domain: str,
        item_key: str,
        status: str,
        path: Optional[os.PathLike[str] | str] = None,
        byte_count: Optional[int] = None,
        error: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        compute_hash: bool = True,
    ) -> None:
        path_text = str(Path(path)) if path is not None else None
        digest = None
        if path is not None and file_is_valid(path):
            byte_count = int(Path(path).stat().st_size)
            if compute_hash:
                digest = sha256_file(path)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO items(
                    run_id, domain, item_key, status, attempts, path, bytes,
                    sha256, error, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, domain, item_key) DO UPDATE SET
                    status=excluded.status,
                    attempts=items.attempts + 1,
                    path=excluded.path,
                    bytes=excluded.bytes,
                    sha256=COALESCE(excluded.sha256, items.sha256),
                    error=excluded.error,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(run_id),
                    str(domain),
                    str(item_key),
                    str(status),
                    path_text,
                    byte_count,
                    digest,
                    error,
                    json_dumps(dict(metadata or {})),
                    utc_now(),
                ),
            )

    def get_item(
        self, run_id: str, domain: str, item_key: str
    ) -> Optional[dict[str, Any]]:
        with contextlib.closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM items WHERE run_id=? AND domain=? AND item_key=?",
                (str(run_id), str(domain), str(item_key)),
            ).fetchone()
        return self._row_to_dict(row)

    def record_training_epoch(
        self,
        run_id: str,
        model_name: str,
        epoch: int,
        metrics: Mapping[str, Any],
        checkpoint_path: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO training_epochs(
                    run_id, model_name, epoch, metrics_json, checkpoint_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    str(model_name),
                    int(epoch),
                    json_dumps(dict(metrics)),
                    str(Path(checkpoint_path)) if checkpoint_path is not None else None,
                    utc_now(),
                ),
            )

    def record_training_history(
        self,
        run_id: str,
        model_name: str,
        rows: Iterable[Mapping[str, Any]],
        checkpoint_path: Optional[os.PathLike[str] | str] = None,
    ) -> int:
        """Backfill multiple epoch rows in one Drive-friendly transaction."""

        timestamp = utc_now()
        checkpoint_text = (
            str(Path(checkpoint_path)) if checkpoint_path is not None else None
        )
        records = [
            (
                str(run_id),
                str(model_name),
                int(row["epoch"]),
                json_dumps(dict(row)),
                checkpoint_text,
                timestamp,
            )
            for row in rows
        ]
        if not records:
            return 0
        with self._transaction() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO training_epochs(
                    run_id, model_name, epoch, metrics_json, checkpoint_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                records,
            )
        return len(records)

    def training_history(self, run_id: str, model_name: str) -> list[dict[str, Any]]:
        with contextlib.closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM training_epochs
                WHERE run_id=? AND model_name=? ORDER BY epoch
                """,
                (str(run_id), str(model_name)),
            ).fetchall()
        return [dict(row) for row in rows]

    def register_artifact(
        self,
        run_id: str,
        stage_id: str,
        path: os.PathLike[str] | str,
        kind: str = "file",
        metadata: Optional[Mapping[str, Any]] = None,
        compute_hash: bool = True,
    ) -> dict[str, Any]:
        candidate = Path(path)
        if not file_is_valid(candidate):
            raise FileNotFoundError(f"Artifact missing or empty: {candidate}")
        record = {
            "path": str(candidate),
            "bytes": int(candidate.stat().st_size),
            "sha256": sha256_file(candidate) if compute_hash else None,
        }
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO artifacts(
                    run_id, stage_id, path, kind, bytes, sha256, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    str(stage_id),
                    record["path"],
                    str(kind),
                    record["bytes"],
                    record["sha256"],
                    json_dumps(dict(metadata or {})),
                    utc_now(),
                ),
            )
        return record

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with contextlib.closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM artifacts WHERE run_id=? ORDER BY stage_id, path",
                (str(run_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def stage_rows(self, run_id: str) -> list[dict[str, Any]]:
        with contextlib.closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM stages
                WHERE run_id=?
                ORDER BY COALESCE(started_at, ''), stage_id
                """,
                (str(run_id),),
            ).fetchall()
        return [dict(row) for row in rows]


class TeeStream:
    def __init__(self, original: Any, log_handle: Any):
        self.original = original
        self.log_handle = log_handle

    def write(self, text: str) -> int:
        written = self.original.write(text)
        self.log_handle.write(text)
        self.log_handle.flush()
        return written

    def flush(self) -> None:
        self.original.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())


class Heartbeat:
    def __init__(
        self,
        ledger: SQLiteLedger,
        run_id: str,
        stage_id: str,
        interval_seconds: float,
        resource_probe: Optional[Callable[[], Mapping[str, Any]]] = None,
        progress_dir: Optional[os.PathLike[str] | str] = None,
    ):
        self.ledger = ledger
        self.run_id = run_id
        self.stage_id = stage_id
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.resource_probe = resource_probe
        self.progress_dir = Path(progress_dir) if progress_dir is not None else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=min(5.0, self.interval_seconds))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            payload: dict[str, Any] = {}
            if self.resource_probe is not None:
                try:
                    payload.update(dict(self.resource_probe()))
                except Exception as exc:  # heartbeat must never stop the experiment
                    payload["resource_probe_error"] = str(exc)[:500]
            self.ledger.event(
                self.run_id,
                level="heartbeat",
                message=f"Stage {self.stage_id} is still running",
                stage_id=self.stage_id,
                payload=payload,
            )
            print(f"[HEARTBEAT] {self.stage_id} ainda em execução | {utc_now()}")
            if self.progress_dir is not None:
                save_progress_snapshot(
                    self.ledger,
                    self.run_id,
                    self.progress_dir,
                    self.stage_id,
                    "running",
                    keep_history=False,
                )


def save_progress_snapshot(
    ledger: SQLiteLedger,
    run_id: str,
    progress_dir: os.PathLike[str] | str,
    stage_id: str,
    status: str,
    keep_history: bool = True,
) -> Optional[Path]:
    """Save a compact PNG dashboard. Failure to render never fails the run."""

    try:
        import matplotlib.pyplot as plt

        rows = ledger.stage_rows(run_id)
        progress_path = Path(progress_dir)
        progress_path.mkdir(parents=True, exist_ok=True)
        completed = sum(row["status"] == "completed" for row in rows)
        failed = sum(row["status"] == "failed" for row in rows)
        running = sum(row["status"] == "running" for row in rows)
        labels = [row["stage_id"] for row in rows[-18:]] or [stage_id]
        colors = {
            "completed": "#2E8B57",
            "failed": "#C0392B",
            "running": "#F39C12",
            "pending": "#95A5A6",
        }
        row_statuses = [row["status"] for row in rows[-18:]] or [status]
        fig_height = max(4.5, 0.34 * len(labels) + 2.5)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        y_positions = list(range(len(labels)))
        ax.barh(
            y_positions,
            [1] * len(labels),
            color=[colors.get(value, "#7F8C8D") for value in row_statuses],
        )
        ax.set_yticks(y_positions, labels)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(
            f"CREMA-D pipeline | run={run_id}\n"
            f"último estágio={stage_id} ({status}) | concluídos={completed} "
            f"falhos={failed} em execução={running}"
        )
        for index, value in enumerate(row_statuses):
            ax.text(0.02, index, value, va="center", color="white", weight="bold")
        fig.tight_layout()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        def save_figure_atomic(path: Path) -> None:
            temporary = path.with_name(
                f"{path.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}{path.suffix}"
            )
            try:
                fig.savefig(temporary, dpi=140, bbox_inches="tight")
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()

        latest = progress_path / "latest_progress.png"
        save_figure_atomic(latest)
        target = latest
        if keep_history:
            target = progress_path / (
                f"{timestamp}_{safe_stage_name(stage_id)}_{safe_stage_name(status)}.png"
            )
            save_figure_atomic(target)
        plt.close(fig)
        return target
    except Exception as exc:
        try:
            ledger.event(
                run_id,
                level="warning",
                message="Could not render progress snapshot",
                stage_id=stage_id,
                payload={"error": str(exc)[:1000]},
            )
        except Exception:
            pass
        return None


@contextlib.contextmanager
def tracked_stage(
    ledger: SQLiteLedger,
    run_id: str,
    stage_id: str,
    required_artifacts: Iterable[os.PathLike[str] | str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
    log_dir: Optional[os.PathLike[str] | str] = None,
    progress_dir: Optional[os.PathLike[str] | str] = None,
    heartbeat_seconds: Optional[float] = None,
    resource_probe: Optional[Callable[[], Mapping[str, Any]]] = None,
) -> Iterator[None]:
    """Record a stage transaction, tee its text log, and persist failures."""

    stage_id = str(stage_id)
    required_paths = [Path(path) for path in required_artifacts]
    ledger.mark_run(run_id, "running")
    ledger.begin_stage(run_id, stage_id, required_paths, metadata)
    ledger.event(
        run_id,
        level="info",
        message="Stage started",
        stage_id=stage_id,
        payload=dict(metadata or {}),
    )
    started = time.time()
    log_handle = None
    stdout_before, stderr_before = sys.stdout, sys.stderr
    heartbeat = None
    try:
        if log_dir is not None:
            log_path = Path(log_dir) / "stage_logs" / f"{safe_stage_name(stage_id)}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(f"\n===== {utc_now()} | run={run_id} | start =====\n")
            log_handle.flush()
            sys.stdout = TeeStream(stdout_before, log_handle)
            sys.stderr = TeeStream(stderr_before, log_handle)
        if heartbeat_seconds:
            heartbeat = Heartbeat(
                ledger,
                run_id,
                stage_id,
                heartbeat_seconds,
                resource_probe=resource_probe,
                progress_dir=progress_dir,
            )
            heartbeat.start()
        print(f"[STAGE] {stage_id}: início")
        yield
        if heartbeat is not None:
            heartbeat.stop()
            heartbeat = None
        missing = [str(path) for path in required_paths if not file_is_valid(path)]
        if missing:
            raise RuntimeError(
                "Stage finished without required artifacts: " + ", ".join(missing)
            )
        for path in required_paths:
            ledger.register_artifact(run_id, stage_id, path)
        duration = time.time() - started
        ledger.complete_stage(run_id, stage_id, duration_sec=duration, metadata=metadata)
        ledger.event(
            run_id,
            level="info",
            message="Stage completed",
            stage_id=stage_id,
            payload={"duration_sec": round(duration, 3)},
        )
        print(f"[STAGE] {stage_id}: concluído em {duration:.1f}s")
        if progress_dir is not None:
            save_progress_snapshot(
                ledger, run_id, progress_dir, stage_id, "completed"
            )
    except BaseException as exc:
        if heartbeat is not None:
            heartbeat.stop()
            heartbeat = None
        duration = time.time() - started
        traceback_text = "".join(
            traceback_module.format_exception(type(exc), exc, exc.__traceback__)
        )
        ledger.fail_stage(
            run_id,
            stage_id,
            error=exc,
            traceback_text=traceback_text,
            duration_sec=duration,
        )
        ledger.mark_run(
            run_id,
            "failed",
            error=f"{stage_id}: {type(exc).__name__}: {exc}",
        )
        ledger.event(
            run_id,
            level="error",
            message=f"Stage failed: {type(exc).__name__}: {exc}",
            stage_id=stage_id,
            payload={"duration_sec": round(duration, 3)},
        )
        print(f"[STAGE] {stage_id}: FALHOU após {duration:.1f}s")
        print(traceback_text)
        if progress_dir is not None:
            save_progress_snapshot(ledger, run_id, progress_dir, stage_id, "failed")
        raise
    finally:
        if heartbeat is not None:
            heartbeat.stop()
        sys.stdout, sys.stderr = stdout_before, stderr_before
        if log_handle is not None:
            log_handle.flush()
            log_handle.close()


def print_stage_dashboard(ledger: SQLiteLedger, run_id: str) -> None:
    rows = ledger.stage_rows(run_id)
    if not rows:
        print("Nenhuma etapa registrada ainda.")
        return
    columns = ["stage_id", "status", "attempts", "duration_sec", "error_message"]
    try:
        import pandas as pd

        display(pd.DataFrame(rows)[columns])  # type: ignore[name-defined]
    except Exception:
        print(" | ".join(columns))
        for row in rows:
            print(" | ".join(str(row.get(column, "")) for column in columns))
