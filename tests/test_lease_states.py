"""Unit tests for the agent2agent plugin — lease state machine.

Guards that:
  (a) the database schema (LEASE_STATES, CREATE TABLE CHECK, JSON enum) does not
      contain the dead 'released' state,
  (b) wait rejects until_states=['released'],
  (c) the database migration is idempotent and correctly maps 'released' -> 'free',
  (d) the CHECK constraint enforces the allowed states.

Run with the Hermes venv interpreter (isolates HERMES_HOME so the production
database is never touched):
    python -m unittest -v <plugin_dir>/tests/test_lease_states.py
or:
    cd <plugin_dir> && python -m unittest -v tests.test_lease_states
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR.parent))

# Redirect HERMES_HOME to an isolated directory BEFORE importing core.py,
# so _lease_db() never touches the production database.
_TMP = Path(tempfile.mkdtemp(prefix="a2a_test_"))
os.environ["HERMES_HOME"] = str(_TMP)

# core.py imports hermes_constants (lazy, inside _state_db_path/_lease_db) —
# the Hermes venv has it; stub get_hermes_home just in case.
import hermes_constants  # noqa: E402


def _stub_get_hermes_home() -> str:
    return str(_TMP)


hermes_constants.get_hermes_home = _stub_get_hermes_home

import core  # noqa: E402  (import after HERMES_HOME stub)

# Test session identity — must look like a real session_id (16+ char hash),
# because _our_identity() and me_holder use our_sid[:12].
SID = "testsession0001111"


def _open_db() -> sqlite3.Connection:
    """Open the lease database through production code (_lease_db) — runs migration."""
    return core._lease_db()


# ---------------------------------------------------------------------------
# (a) Schema: 'released' is not a lease state
# ---------------------------------------------------------------------------
class TestNoReleasedState(unittest.TestCase):
    def test_lease_states_tuple(self):
        """LEASE_STATES does not contain 'released'."""
        self.assertNotIn("released", core.LEASE_STATES)
        self.assertEqual(set(core.LEASE_STATES), {"free", "requested", "held"})

    def test_create_table_check_constraint(self):
        """CREATE TABLE in the code has a 3-state CHECK (no 'released')."""
        src = (PLUGIN_DIR / "core.py").read_text()
        checks = re.findall(r"CHECK\s*\(state IN \(([^)]+)\)\)", src)
        self.assertTrue(checks, "No CHECK constraint found in core.py")
        for c in checks:
            self.assertNotIn("released", c, f"CHECK contains 'released': {c}")

    def test_json_schema_enum_until_states(self):
        """The until_states enum in the tool JSON schema does not contain 'released'."""
        conn = _open_db()
        try:
            out = core.lease({"action": "wait", "resource": "res:test", "timeout": 1},
                             session_id=SID)
            # We do not check the result content here — only that the tool schema is consistent.
            # The enum itself is verified statically:
            src = (PLUGIN_DIR / "core.py").read_text()
            m = re.search(r'"until_states".*?"enum":\s*\[([^\]]+)\]', src, re.DOTALL)
            self.assertIsNotNone(m, "No until_states enum found in JSON schema")
            self.assertNotIn("released", m.group(1))
        finally:
            conn.close()

    def test_fresh_db_check_constraint(self):
        """A freshly created database has a 3-state CHECK."""
        conn = _open_db()
        try:
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='leases'"
            ).fetchone()[0]
            self.assertNotIn("'released'", sql)
            self.assertIn("CHECK (state IN ('free','requested','held'))", sql)
        finally:
            conn.close()

    def test_check_rejects_released_insert(self):
        """INSERT state='released' is rejected by the CHECK constraint (IntegrityError)."""
        conn = _open_db()
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO leases (resource, state) VALUES ('res:reject', 'released')"
                )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# (b) wait rejects the unknown state 'released'
# ---------------------------------------------------------------------------
class TestWaitRejectsReleased(unittest.TestCase):
    def test_wait_until_states_released_only(self):
        """wait(until_states=['released']) returns an error about unknown states."""
        conn = _open_db()
        try:
            out = json.loads(core._wait_lease(
                {"until_states": ["released"]},
                "res:anything", conn, our_sid=SID,
                ours={"display_name": "tester", "chat_id": "", "session_key": ""},
            ))
            self.assertFalse(out["ok"])
            self.assertIn("until_states empty or unknown", out["error"])
            self.assertIn("free/requested/held", out["error"])
            self.assertNotIn("released", out["error"],
                             "Error message still mentions 'released'")
        finally:
            conn.close()

    def test_wait_until_states_mixed_filters_released(self):
        """wait(until_states=['free','released']) filters out 'released' and works."""
        conn = _open_db()
        try:
            # Row in 'free' state -> wait should return success immediately.
            import time as _t
            now = _t.time()
            conn.execute(
                "INSERT OR REPLACE INTO leases "
                "(resource, state, note, created_at, updated_at) "
                "VALUES ('res:mixed', 'free', 'ok', ?, ?)", (now, now)
            )
            conn.commit()
            out = json.loads(core._wait_lease(
                {"until_states": ["free", "released"]},
                "res:mixed", conn, our_sid=SID,
                ours={"display_name": "tester", "chat_id": "", "session_key": ""},
            ))
            self.assertTrue(out["ok"])
            self.assertFalse(out.get("timed_out"))
            self.assertEqual(out["lease"]["state"], "free")
        finally:
            conn.close()

    def test_wait_until_states_all_unknown(self):
        """wait(until_states=['released','bogus']) -> error (all states outside LEASE_STATES)."""
        conn = _open_db()
        try:
            out = json.loads(core._wait_lease(
                {"until_states": ["released", "bogus"]},
                "res:anything", conn, our_sid=SID,
                ours={"display_name": "tester", "chat_id": "", "session_key": ""},
            ))
            self.assertFalse(out["ok"])
            self.assertIn("until_states empty or unknown", out["error"])
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# (c) Migration: idempotency + 'released' -> 'free' mapping
# ---------------------------------------------------------------------------
class TestMigration(unittest.TestCase):
    OLD_SCHEMA = """CREATE TABLE leases (
            resource TEXT PRIMARY KEY,
            holder TEXT NOT NULL DEFAULT '',
            holder_session TEXT NOT NULL DEFAULT '',
            holder_display TEXT NOT NULL DEFAULT '',
            requestor TEXT NOT NULL DEFAULT '',
            requestor_session TEXT NOT NULL DEFAULT '',
            requestor_display TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL DEFAULT 'free' CHECK (state IN ('free','requested','held','released')),
            note TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            granted_by TEXT NOT NULL DEFAULT '',
            granted_by_session TEXT NOT NULL DEFAULT ''
        )"""

    @staticmethod
    def _db_path() -> Path:
        return Path(os.environ["HERMES_HOME"]) / "agent2a" / "leases.db"

    def _reset_db(self) -> None:
        """Remove the lease database so the next open creates it from scratch."""
        p = self._db_path()
        for suffix in ("", "-wal", "-shm"):
            f = Path(str(p) + suffix)
            if f.exists():
                f.unlink()

    def _make_old_db(self) -> None:
        """Create a synthetic 'old' database with the dead state and a 'released' row."""
        db = self._db_path()
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db))
        conn.execute(self.OLD_SCHEMA)
        import time as _t
        now = _t.time()
        conn.execute(
            "INSERT INTO leases (resource, state, note, created_at, updated_at) "
            "VALUES ('res:old', 'released', 'note-rel', ?, ?)", (now, now)
        )
        conn.execute(
            "INSERT INTO leases (resource, state, note, created_at, updated_at) "
            "VALUES ('res:held', 'held', 'note-held', ?, ?)", (now, now)
        )
        conn.commit()
        conn.close()

    def test_migration_maps_released_to_free(self):
        """Migration: 'released' row -> 'free' (note preserved), 'held' row untouched."""
        self._reset_db()
        self._make_old_db()
        # Opening through production code triggers the migration.
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT state, note FROM leases WHERE resource='res:old'"
            ).fetchone()
            self.assertIsNotNone(row, "row 'res:old' vanished after migration")
            self.assertEqual(row[0], "free", "'released' was not migrated to 'free'")
            self.assertEqual(row[1], "note-rel", "note was not preserved")
            row2 = conn.execute(
                "SELECT state FROM leases WHERE resource='res:held'"
            ).fetchone()
            self.assertEqual(row2[0], "held", "'held' row was modified")
            # No orphaned table.
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            self.assertNotIn("leases_old", tables)
        finally:
            conn.close()

    def test_migration_idempotent(self):
        """Second open after migration is a no-op (data and schema unchanged)."""
        self._reset_db()
        self._make_old_db()
        c1 = _open_db()
        c1.close()
        c2 = _open_db()
        try:
            rows = c2.execute(
                "SELECT resource, state, note FROM leases ORDER BY resource"
            ).fetchall()
            self.assertEqual(len(rows), 2)
            states = {r[0]: (r[1], r[2]) for r in rows}
            self.assertEqual(states["res:old"], ("free", "note-rel"))
            self.assertEqual(states["res:held"], ("held", "note-held"))
            sql = c2.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='leases'"
            ).fetchone()[0]
            self.assertNotIn("'released'", sql)
        finally:
            c2.close()

    def test_new_db_no_migration_needed(self):
        """A new database (without the old schema) does not go through migration."""
        self._reset_db()
        conn = _open_db()
        try:
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='leases'"
            ).fetchone()[0]
            self.assertNotIn("'released'", sql)
            self.assertNotIn("leases_old", sql)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# (d) Full request -> approve -> release cycle (API regression)
# ---------------------------------------------------------------------------
class TestLeaseCycle(unittest.TestCase):
    def test_request_approve_release(self):
        """The basic negotiation cycle works end-to-end."""
        conn = _open_db()
        try:
            res = "res:cycle"
            # request (from B's perspective)
            out = json.loads(core.lease(
                {"action": "request", "resource": res, "note": "B needs X"},
                session_id="B" + SID[1:]))
            self.assertTrue(out["ok"])
            self.assertEqual(out["lease"]["state"], "requested")
            # approve (from A's perspective — the owner)
            out = json.loads(core.lease(
                {"action": "approve", "resource": res, "also_steer": True,
                 "note": "conditions"},
                session_id=SID))
            self.assertTrue(out["ok"])
            self.assertEqual(out["lease"]["state"], "held")
            # release (B finishes)
            out = json.loads(core.lease(
                {"action": "release", "resource": res},
                session_id="B" + SID[1:]))
            self.assertTrue(out["ok"])
            self.assertEqual(out["lease"]["state"], "free")
            self.assertNotEqual(out["lease"]["state"], "released")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
