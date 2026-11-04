import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from report_title_translation_status import build_status


class ReportTitleTranslationStatusTests(unittest.TestCase):
    def test_reads_aggregate_cache_metrics_without_leaking_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            db_path = workdir / "cache_gemma4.sqlite3"
            now = datetime(2026, 10, 24, 7, 0, tzinfo=timezone.utc)
            rows = [
                ("k1", "SENSITIVE_TITLE", "ok", 0, "2026-10-24T06:58:00+00:00", "SENSITIVE_TRANSLATION"),
                ("k2", "OTHER_TITLE", "ok", 0, "2026-10-24T06:52:00+00:00", "OTHER_TRANSLATION"),
                ("k3", "FAILED_TITLE", "error", 0, "2026-10-24T06:57:00+00:00", "PRIVATE_ERROR"),
                ("k4", "TRUNCATED_TITLE", "error", 1, "2026-10-24T06:56:00+00:00", "PRIVATE_OUTPUT"),
            ]
            with closing(sqlite3.connect(db_path)) as connection:
                connection.execute(
                    "CREATE TABLE translations (cache_key TEXT, source_title TEXT, "
                    "status TEXT, truncated INTEGER, updated_at_utc TEXT, output_text TEXT)"
                )
                connection.executemany(
                    "INSERT INTO translations VALUES (?, ?, ?, ?, ?, ?)", rows
                )
                connection.commit()
            (workdir / "hybrid_run_state.json").write_text(
                json.dumps({"models": {"gemma4": {"stages": {"full": {
                    "total_unique_titles": 10,
                    "status": "checkpoint_saved_continue",
                    "completed_unique_titles": 1,
                    "remaining_unique_titles": 9,
                }}}}}),
                encoding="utf-8",
            )
            (workdir / "gemma4_background_test.stdout.log").write_text(
                "[gemma4] transient request error on attempt 1 (HTTP 500); "
                "retrying after 3s: SECRET_PROVIDER_DETAIL\n"
                "[gemma4] uncached 4/10; new complete=4; errors=0; "
                "api_requests=3; paired_requests=2; fallback=0; "
                "SECRET_TITLE SECRET_TRANSLATION\n",
                encoding="utf-8",
            )

            result = build_status(workdir, now=now, window_minutes=15)

            self.assertEqual(result["completed_ok"], 2)
            self.assertEqual(result["completed_unique_titles"], 2)
            self.assertEqual(result["remaining_unique_titles"], 8)
            self.assertEqual(result["run_state_completed_unique_titles"], 1)
            self.assertEqual(result["run_state_remaining_unique_titles"], 9)
            self.assertEqual(result["latest_logged_batch_progress"]["completed"], 4)
            self.assertEqual(result["http_retry_counts_in_log_tail"], {"http_500": 1})
            self.assertEqual(result["truncated_records"], 1)
            self.assertEqual(result["recent_completed_titles"], 2)
            self.assertEqual(result["recent_rate_titles_per_hour"], 8.0)
            serialized = json.dumps(result)
            for secret in ("SENSITIVE_TITLE", "SENSITIVE_TRANSLATION", "PRIVATE_ERROR", "SECRET_TITLE", "SECRET_TRANSLATION", "SECRET_PROVIDER_DETAIL"):
                self.assertNotIn(secret, serialized)

    def test_missing_database_is_reported_without_mutating_workdir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            before = sorted(path.name for path in workdir.iterdir())
            result = build_status(workdir)
            after = sorted(path.name for path in workdir.iterdir())
            self.assertEqual(result["database"], "missing")
            self.assertEqual(before, after)

    def test_naive_clock_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                build_status(Path(temp_dir), now=datetime(2026, 10, 24))


if __name__ == "__main__":
    unittest.main()
