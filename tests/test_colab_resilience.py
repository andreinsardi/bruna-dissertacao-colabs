import json
import sqlite3
import unittest
import uuid
from pathlib import Path

from scripts.colab_resilience import (
    SQLiteLedger,
    atomic_write_json,
    sha256_file,
    tracked_stage,
)


TEST_TMP_ROOT = Path(__file__).resolve().parent


class SQLiteLedgerTests(unittest.TestCase):
    def setUp(self):
        self.test_key = f".colab-resilience-{uuid.uuid4().hex}"
        self.root = TEST_TMP_ROOT
        self.db_path = self.root / f"{self.test_key}.sqlite"
        self.ledger = SQLiteLedger(self.db_path)
        self.run_id = "run-test-001"
        self.ledger.start_run(
            run_id=self.run_id,
            experiment_name="cremad-test",
            config_hash="abc123",
            config={"seed": 42},
        )

    def tearDown(self):
        self.ledger.close()
        for path in self.root.glob(f"{self.test_key}*"):
            if path.is_file():
                path.unlink()

    def test_schema_contains_operational_tables(self):
        conn = sqlite3.connect(self.db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            conn.close()
        self.assertTrue(
            {
                "runs",
                "stages",
                "events",
                "items",
                "training_epochs",
                "artifacts",
            }.issubset(tables)
        )

    def test_stage_success_is_reusable_only_with_valid_artifacts(self):
        artifact = self.root / f"{self.test_key}-artifact.csv"
        artifact.write_text("a,b\n1,2\n", encoding="utf-8")

        with tracked_stage(
            self.ledger,
            self.run_id,
            "unit_success",
            required_artifacts=[artifact],
        ):
            pass

        self.assertTrue(
            self.ledger.stage_is_reusable(
                self.run_id, "unit_success", [artifact]
            )
        )
        artifact.unlink()
        self.assertFalse(
            self.ledger.stage_is_reusable(
                self.run_id, "unit_success", [artifact]
            )
        )

    def test_stage_failure_persists_traceback_and_can_resume(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with tracked_stage(self.ledger, self.run_id, "unit_failure"):
                raise RuntimeError("boom")

        row = self.ledger.get_stage(self.run_id, "unit_failure")
        self.assertEqual(row["status"], "failed")
        self.assertIn("RuntimeError: boom", row["traceback"])
        self.assertGreaterEqual(row["attempts"], 1)
        conn = sqlite3.connect(self.db_path)
        try:
            run_status = conn.execute(
                "SELECT status FROM runs WHERE run_id=?", (self.run_id,)
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(run_status, "failed")

    def test_items_training_epochs_and_artifacts_are_persisted(self):
        artifact = self.root / f"{self.test_key}-checkpoint.pt"
        artifact.write_bytes(b"checkpoint")
        digest = sha256_file(artifact)

        self.ledger.upsert_item(
            self.run_id,
            domain="download",
            item_key="video-001",
            status="completed",
            path=artifact,
            byte_count=artifact.stat().st_size,
            metadata={"source": "test"},
        )
        self.ledger.record_training_epoch(
            self.run_id,
            model_name="EmoT",
            epoch=7,
            metrics={"train_loss": 0.5, "val_macro_f1": 0.7},
            checkpoint_path=artifact,
        )
        self.ledger.register_artifact(
            self.run_id,
            stage_id="train_EmoT",
            path=artifact,
            kind="checkpoint",
        )

        item = self.ledger.get_item(self.run_id, "download", "video-001")
        self.assertEqual(item["status"], "completed")
        self.assertEqual(item["sha256"], digest)

        epochs = self.ledger.training_history(self.run_id, "EmoT")
        self.assertEqual(len(epochs), 1)
        self.assertEqual(epochs[0]["epoch"], 7)
        self.assertAlmostEqual(
            json.loads(epochs[0]["metrics_json"])["val_macro_f1"], 0.7
        )

        artifacts = self.ledger.list_artifacts(self.run_id)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["sha256"], digest)

    def test_start_or_resume_returns_latest_incomplete_run(self):
        self.ledger.mark_run(self.run_id, "failed", error="interrupted")
        resumed, was_resumed = self.ledger.start_or_resume_run(
            experiment_name="cremad-test",
            preferred_run_id=None,
            config_hash="abc123",
            config={"seed": 42},
        )
        self.assertEqual(resumed, self.run_id)
        self.assertTrue(was_resumed)

    def test_completed_active_run_starts_a_new_run(self):
        self.ledger.mark_run(self.run_id, "completed")
        new_run, was_resumed = self.ledger.start_or_resume_run(
            experiment_name="cremad-test",
            preferred_run_id=self.run_id,
            config_hash="abc123",
            config={"seed": 42},
        )
        self.assertNotEqual(new_run, self.run_id)
        self.assertFalse(was_resumed)

    def test_training_history_can_be_backfilled_in_one_transaction(self):
        inserted = self.ledger.record_training_history(
            self.run_id,
            "EmoT",
            [
                {"epoch": 8, "train_loss": 0.4, "val_macro_f1": 0.71},
                {"epoch": 9, "train_loss": 0.3, "val_macro_f1": 0.72},
            ],
        )
        self.assertEqual(inserted, 2)
        epochs = self.ledger.training_history(self.run_id, "EmoT")
        self.assertEqual([row["epoch"] for row in epochs], [8, 9])

    def test_atomic_json_write_never_leaves_temporary_file(self):
        target = self.root / f"{self.test_key}-state.json"
        atomic_write_json(target, {"ok": True, "n": 3})
        self.assertEqual(
            json.loads(target.read_text(encoding="utf-8")),
            {"ok": True, "n": 3},
        )
        self.assertEqual(list(target.parent.glob("*.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
