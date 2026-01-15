# RICE13_FS/output/coalition_store.py
"""
Persistent cache for coalition solutions using a single SQLite database.

Key behavior:
- Cache is MANDATORY and the single source of truth for exports.
- One SQLite file stores BOTH the registry and the full solution blobs.
- meta.json pins a run fingerprint; mismatches are rejected unless allow_mismatch=True.
- PUT requires `solution["disc"]` (discounted-only pipeline).
- No cache-scope gating: reuse is governed solely by the fingerprint in meta.json.

Public API:
  vec_to_bitmask(vec) -> '101...'
  CoalitionStore(path, namespace, fingerprint, allow_mismatch=False)
    .get(vec, spec_id) -> {"payoff": [...], "solution": dict|None, "meta": {...}} | None
    .put(vec, spec_id, *, label, payoff, solution, meta) -> None
    .has(vec, spec_id) -> bool
    .iter_rows(spec_id: str | None = None) -> Iterator[CacheEntry]
    .peek(vec, spec_id) -> CacheEntry | None
    .summary_df() -> pd.DataFrame
    .close() -> None
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time, secrets, stat
import logging
import pandas as pd

from RICE13_FS.common.utils import normalize_fingerprint,vec_to_bitmask


# ------------------------------
# Small helpers
# ------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    tmpdir = path.parent
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Make target writable if it exists (Windows may set read-only bits)
    try:
        if path.exists():
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except Exception:
        pass

    # Serialize once to bytes (faster retries)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    # Retry replacing to dodge brief sync/AV locks
    last_err = None
    for attempt in range(8):  # ~1–2 seconds total
        tmppath = tmpdir / f"{path.stem}_{secrets.token_hex(4)}.tmp"
        try:
            with open(tmppath, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(tmppath, path)  # atomic if not locked
                return
            except PermissionError as e:
                last_err = e
                # tiny backoff; sync tools usually release quickly
                time.sleep(0.1 * (attempt + 1))
                continue
        finally:
            try:
                if tmppath.exists():
                    tmppath.unlink()
            except OSError:
                pass

    # If we get here, something keeps holding the file
    raise PermissionError(f"Could not atomically write {path}: {last_err}")


# ------------------------------
# Data structures
# ------------------------------

@dataclass
class CacheEntry:
    vector: Tuple[int, ...]
    spec_id: str           # solution spec id (compatibility key)
    label: str             # "US_EU", "GRAND", ...
    payoff: List[float]    # len = #regions (discounted utilities per region)
    meta: Dict[str, Any]   # e.g., {"converged": True, "iterations": 37, "S_tag": "..."}
    created_at: str        # ISO timestamp


# ------------------------------
# CoalitionStore (SQLite backend)
# ------------------------------

class CoalitionStore:
    """
    Persistent cache with a SQLite registry + blobs to avoid Windows replace/lock issues.
    Reuse is controlled by the on-disk fingerprint in meta.json.

    Note: for quick inspection/filtering, we mirror lightweight identity fields
    from the solution into the meta_json at write time (e.g., "utility", "disc_tag").
    """

    def __init__(
        self,
        path: Path,
        namespace: str,
        fingerprint: Dict[str, Any],
        *,
        allow_mismatch: bool = False,
    ) -> None:
        self.base = Path(path).expanduser().resolve() / namespace
        self.base.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.base / "meta.json"

        self.fingerprint = normalize_fingerprint(fingerprint)
        self.allow_mismatch = bool(allow_mismatch)

        # Validate / write meta.json (atomic; Windows-friendly)
        self._init_or_check_meta()

        # SQLite DB path and connection
        self.db_path = self.base / "cache_index.sqlite"
        self._conn = self._open_db(self.db_path)
        self._init_schema(self._conn)

    # ---------- public API ----------

    def get(self, vec: Iterable[int], spec_id: str) -> Optional[Dict[str, Any]]:
        """
        Return {"payoff": [...], "solution": dict|None, "meta": {...}} or None.
        Fingerprint compatibility is enforced by meta.json on store initialization.
        """
        bitmask = vec_to_bitmask(vec)
        row = self._lookup_row(bitmask, spec_id)
        if row is None:
            return None

        # Deserialize solution blob (if present)
        solution = None
        blob = row.get("solution_blob")
        if blob is not None:
            try:
                solution = pickle.loads(blob)
            except Exception:
                solution = None

        return {
            "payoff": json.loads(row["payoff_json"]),
            "solution": solution,
            "meta": json.loads(row["meta_json"]),
        }

    def put(
        self,
        vec: Iterable[int],
        spec_id: str,
        *,
        label: str,
        payoff: List[float],
        solution: Dict[str, Any],
        meta: Optional[Dict[str, Any]],
    ) -> None:
        """
        Upsert registry row + solution blob (single-DB transaction; no file rename).

        Strict checks (discounted-only pipeline):
          • len(payoff) must equal len(fingerprint["regions"]) if provided
          • solution must include 'disc' mapping (region, t) -> discount factor
        """
        # Normalize
        spec_id = str(spec_id)
        bitmask = vec_to_bitmask(tuple(int(x) for x in vec))
        created_at = now_iso()

        # Structural checks
        regions = list(self.fingerprint.get("regions", []))
        if regions and len(payoff) != len(regions):
            raise ValueError(
                f"payoff length {len(payoff)} != #regions {len(regions)} from fingerprint"
            )
        if not isinstance(solution, dict) or "disc" not in solution:
            raise ValueError("solution must be a dict containing 'disc' for discounted payoffs (strict pipeline).")

        # Serialize solution to BLOB (resilient). If serialization fails,
        # we still write the registry row with NULL blob so exporters can
        # select the entry via mirrored metadata.
        solution_blob: Optional[bytes] = None
        try:
            solution_blob = pickle.dumps(solution, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Cache: could not serialize solution for vec=%s spec_id=%s; "
                "writing row without blob. Error: %s",
                bitmask, spec_id, e
            )

        # Mirror lightweight identity fields into meta for quick filtering without blob decoding.
        # (No fallbacks; only copy if explicitly present in the solution.)
        meta_out = dict(meta) if meta is not None else {}
        util = solution.get("utility", None)
        if isinstance(util, str) and util:
            meta_out.setdefault("utility", util)
        disc_tag = solution.get("disc_tag", None)
        if isinstance(disc_tag, str) and disc_tag:
            meta_out.setdefault("disc_tag", disc_tag)

        # Upsert into SQLite (single statement with ON CONFLICT)
        self._upsert_row(
            bitmask=bitmask,
            spec_id=spec_id,
            label=str(label),
            payoff_json=json.dumps(list(payoff)),
            meta_json=json.dumps(meta_out),
            created_at=created_at,
            solution_blob=solution_blob,
        )

    def has(self, vec: Iterable[int], spec_id: str) -> bool:
        """Cheap existence check for a (vec, spec_id) registry row."""
        return self._lookup_row(vec_to_bitmask(vec), spec_id) is not None

    def iter_rows(self, spec_id: Optional[str] = None) -> Iterable[CacheEntry]:
        """Yield all cached rows as CacheEntry (payoff/meta parsed)."""
        cur = self._conn.cursor()
        if spec_id:
            cur.execute(
                "SELECT vector_bitmask, spec_id, label, payoff_json, meta_json, created_at "
                "FROM registry WHERE spec_id=? ORDER BY created_at", (spec_id,)
            )
        else:
            cur.execute(
                "SELECT vector_bitmask, spec_id, label, payoff_json, meta_json, created_at "
                "FROM registry ORDER BY created_at"
            )
        for vb, sid, label, payoff_json, meta_json, created_at in cur.fetchall():
            yield CacheEntry(
                vector=tuple(int(ch) for ch in vb),
                spec_id=sid,
                label=label,
                payoff=list(json.loads(payoff_json)),
                meta=json.loads(meta_json),
                created_at=created_at,
            )

    def peek(self, vec: Iterable[int], spec_id: str) -> Optional[CacheEntry]:
        """
        Return registry metadata (no blob decode) or None if absent.
        Useful for quick summaries without touching large blobs.
        """
        row = self._lookup_row(vec_to_bitmask(vec), spec_id)
        if row is None:
            return None
        return CacheEntry(
            vector=tuple(int(ch) for ch in row["vector_bitmask"]),
            spec_id=row["spec_id"],
            label=row["label"],
            payoff=list(json.loads(row["payoff_json"])),
            meta=json.loads(row["meta_json"]),
            created_at=row["created_at"],
        )

    def summary_df(self) -> pd.DataFrame:
        """Return a snapshot DataFrame of the registry (excluding blobs)."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT vector_bitmask, spec_id, label, payoff_json, meta_json, created_at FROM registry"
        )
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["vector_bitmask", "spec_id", "label", "payoff_json", "meta_json", "created_at"]
            )
        return pd.DataFrame(
            rows, columns=["vector_bitmask", "spec_id", "label", "payoff_json", "meta_json", "created_at"]
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with contextlib.suppress(Exception):
            self._conn.close()

    # ---------- internals ----------

    def _init_or_check_meta(self) -> None:
        meta = {"fingerprint": self.fingerprint}
        if self.meta_path.exists():
            try:
                on_disk = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                if not self.allow_mismatch:
                    raise RuntimeError(
                        "Corrupt cache meta.json and allow_mismatch is False; refusing to continue.\n"
                        f"Path: {self.meta_path}"
                    )
                on_disk = None

            if isinstance(on_disk, dict):
                fp_disk = normalize_fingerprint(on_disk.get("fingerprint", {}))
                ok = (fp_disk == self.fingerprint)
                if not ok and not self.allow_mismatch:
                    # fail fast; caller can relaunch with allow_mismatch=True
                    raise RuntimeError(
                        "Cache fingerprint mismatch.\n"
                        f"  disk.fingerprint: {json.dumps(fp_disk, sort_keys=True)}\n"
                        f"  run.fingerprint : {json.dumps(self.fingerprint, sort_keys=True)}\n"
                        "Use --cache-allow-mismatch to override."
                    )
        # Write/refresh meta.json atomically (Windows-friendly)
        _write_json_atomic(self.meta_path, meta)

    # --- SQLite plumbing ---

    @staticmethod
    def _open_db(db_path: Path) -> sqlite3.Connection:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # timeout allows waiting for short concurrent locks; autocommit mode
        conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None, detect_types=0)
        cur = conn.cursor()
        # WAL journaling + NORMAL synchronous → robust + fast enough
        with contextlib.suppress(Exception):
            cur.execute("PRAGMA journal_mode=WAL;")
        with contextlib.suppress(Exception):
            cur.execute("PRAGMA synchronous=NORMAL;")
        with contextlib.suppress(Exception):
            cur.execute("PRAGMA foreign_keys=ON;")
        return conn

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS registry (
                vector_bitmask TEXT NOT NULL,
                spec_id        TEXT NOT NULL,
                label          TEXT NOT NULL,
                payoff_json    TEXT NOT NULL,
                meta_json      TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                solution_blob  BLOB,
                PRIMARY KEY (vector_bitmask, spec_id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_registry_created ON registry(created_at);")

    def _lookup_row(self, bitmask: str, spec_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT vector_bitmask, spec_id, label, payoff_json, meta_json, created_at, solution_blob "
            "FROM registry WHERE vector_bitmask=? AND spec_id=?",
            (bitmask, spec_id),
        )
        row = cur.fetchone()
        if row is None:
            return None
        vb, sid, label, payoff_json, meta_json, created_at, solution_blob = row
        return {
            "vector_bitmask": vb,
            "spec_id": sid,
            "label": label,
            "payoff_json": payoff_json,
            "meta_json": meta_json,
            "created_at": created_at,
            "solution_blob": solution_blob,
        }

    def _upsert_row(
        self,
        *,
        bitmask: str,
        spec_id: str,
        label: str,
        payoff_json: str,
        meta_json: str,
        created_at: str,
        solution_blob: Optional[bytes],
    ) -> None:
        cur = self._conn.cursor()
        # UPSERT (SQLite >= 3.24; available on standard Windows Python)
        cur.execute(
            """
            INSERT INTO registry (vector_bitmask, spec_id, label, payoff_json, meta_json, created_at, solution_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(vector_bitmask, spec_id) DO UPDATE SET
              label=excluded.label,
              payoff_json=excluded.payoff_json,
              meta_json=excluded.meta_json,
              created_at=excluded.created_at,
              solution_blob=excluded.solution_blob
            ;
            """,
            (
                bitmask,
                spec_id,
                label,
                payoff_json,
                meta_json,
                created_at,
                sqlite3.Binary(solution_blob) if solution_blob is not None else None,
            ),
        )

    
    def get_latest_fs(self, vec, disc_tag: str | None = None):
        """
        Return the most recent FS entry for this coalition vector.
        If disc_tag is provided, require meta.disc_tag == disc_tag.
        Returns {"payoff": [...], "solution": dict|None, "meta": {...}} or None.
        """
        bitmask = vec_to_bitmask(vec)
    
        sql = (
            "SELECT payoff_json, meta_json, solution_blob "
            "FROM registry "
            "WHERE vector_bitmask=? "
            "AND json_extract(meta_json,'$.utility')='fs' "
        )
        params = [bitmask]
        if disc_tag is not None:
            sql += "AND json_extract(meta_json,'$.disc_tag')=? "
            params.append(str(disc_tag))
    
        sql += "ORDER BY datetime(created_at) DESC LIMIT 1"
    
        cur = self._conn.cursor()
        row = cur.execute(sql, params).fetchone()
        if not row:
            return None
    
        payoff_json, meta_json, solution_blob = row
        try:
            solution = pickle.loads(solution_blob) if solution_blob is not None else None
        except Exception:
            solution = None
    
        return {
            "payoff": json.loads(payoff_json),
            "solution": solution,
            "meta": json.loads(meta_json),
        }
