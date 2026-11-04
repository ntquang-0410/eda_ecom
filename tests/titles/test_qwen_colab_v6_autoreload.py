import ast
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/templates/title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload.ipynb"


class QwenColabV6AutoreloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = {
            index: "".join(cell.get("source", []))
            for index, cell in enumerate(cls.notebook.get("cells", []))
            if cell.get("cell_type") == "code"
        }

    def test_all_python_cells_compile(self):
        for index, source in self.code_cells.items():
            if source.lstrip().startswith(("!", "%")):
                continue
            with self.subTest(cell=index):
                compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")

    def test_drive_control_is_allowlisted_and_reports_active_bundle_sha(self):
        control_source = "\n".join(self.code_cells.values())
        self.assertIn("{'run', 'pause', 'stop', 'reload'}", control_source)
        self.assertIn("'active_bundle_sha256': ACTIVE_BUNDLE_SHA", control_source)
        self.assertIn("'requested_bundle_sha256': _control_expected_bundle_sha256", control_source)
        self.assertNotIn("exec(command", control_source)

    def test_drive_status_strips_titles_translations_and_exception_messages(self):
        module = ast.parse(self.code_cells[7])
        wanted = {"_qwen_safe_recent_errors", "_qwen_safe_error_class", "qwen_drive_control"}
        body = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        namespace = {
            "json": json,
            "time": SimpleNamespace(time=lambda: 123.0),
            "datetime": datetime,
            "timezone": timezone,
            "DRIVE_CACHE": Path("drive-cache.sqlite3"),
            "ACTIVE_BUNDLE_SHA": "active-bundle-sha",
            "_control_last_status_write": 0.0,
            "_control_last_action": None,
            "_control_expected_bundle_sha256": None,
            "_SAFE_CONTROL_ERROR_REASONS": {"repeated_phrase_guard", "truncated_output", "empty_output"},
            "_SAFE_EXCEPTION_CLASSES": {"RuntimeError", "ValueError"},
        }
        exec(compile(ast.Module(body=body, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            control_path = root / "qwen_control.json"
            status_path = root / "qwen_live_status.json"
            control_path.write_text(json.dumps({"action": "run"}), encoding="utf-8")
            namespace.update({"CONTROL_FILE": control_path, "LIVE_STATUS_FILE": status_path})
            namespace["qwen_drive_control"]({
                "model_key": "qwen3_14b_awq",
                "phase": "generating",
                "processed_this_session": 1,
                "total_this_session": 10,
                "completed_this_session": 0,
                "errors_this_session": 1,
                "completed_before_session": 0,
                "recent_errors": [{
                    "reason": "repeated_phrase_guard",
                    "source": "SECRET_SOURCE_TITLE",
                    "output_preview": "SECRET_TRANSLATION",
                    "truncated": False,
                }],
                "last_error": "RuntimeError: SECRET_TITLE_AND_TRANSLATION",
            })
            stored = status_path.read_text(encoding="utf-8")
            payload = json.loads(stored)

        for secret in ("SECRET_SOURCE_TITLE", "SECRET_TRANSLATION", "SECRET_TITLE_AND_TRANSLATION"):
            self.assertNotIn(secret, stored)
        self.assertEqual(payload["recent_errors"], [{"reason": "repeated_phrase_guard", "truncated": False}])
        self.assertEqual(payload["last_error_class"], "RuntimeError")
        self.assertNotIn("traceback_tail", payload)

    def test_v6_builder_emits_loading_heartbeat_and_private_status_only(self):
        from build_title_translation_qwen3_14b_awq_colab_v6 import rewrite_pipeline

        source = (ROOT / "src/title_translation_hybrid.py").read_text(encoding="utf-8")
        generated = rewrite_pipeline(source)
        compile(generated, "generated_qwen3_14b_awq_pipeline.py", "exec")
        self.assertIn("heartbeat_stop.wait(30.0)", generated)
        self.assertNotIn('"source": title[:120]', generated)
        self.assertNotIn("output_preview", generated)

    def _load_reload_function(self, bundle_path, control_path, active_sha, expected_sha, clock):
        source = "\n".join(self.code_cells.values())
        module = ast.parse(source)
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_apply_drive_code_update"
        )
        namespace = {
            "hashlib": hashlib,
            "json": json,
            "time": clock,
            "BUNDLE_PATH": bundle_path,
            "CONTROL_FILE": control_path,
            "CONTROL_POLL_SECONDS": 5,
            "ACTIVE_BUNDLE_SHA": active_sha,
            "_control_expected_bundle_sha256": expected_sha,
            "pipeline": "old-pipeline",
            "load_pipeline_bundle": lambda expected_sha256=None: ("new-pipeline", expected_sha256),
            "qwen_drive_control": lambda progress: None,
            "print": lambda *args, **kwargs: None,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
        return namespace

    def test_reload_accepts_only_matching_sha_and_resets_control_to_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / "bundle.zip"
            control = root / "qwen_control.json"
            bundle.write_bytes(b"verified-new-bundle")
            expected_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
            control.write_text(json.dumps({"action": "reload", "bundle_sha256": expected_sha}), encoding="utf-8")
            clock = SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _: None)
            namespace = self._load_reload_function(bundle, control, "old-sha", expected_sha, clock)

            self.assertTrue(namespace["_apply_drive_code_update"](timeout_seconds=60))
            self.assertEqual(namespace["pipeline"], "new-pipeline")
            self.assertEqual(namespace["ACTIVE_BUNDLE_SHA"], expected_sha)
            self.assertEqual(json.loads(control.read_text(encoding="utf-8")), {
                "action": "run", "bundle_sha256": expected_sha,
            })

    def test_reload_refuses_sha_mismatch_and_keeps_current_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / "bundle.zip"
            control = root / "qwen_control.json"
            bundle.write_bytes(b"bundle-a")
            control.write_text(json.dumps({"action": "reload", "bundle_sha256": "wrong-sha"}), encoding="utf-8")
            ticks = iter((0.0, 0.0, 61.0))
            clock = SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
            namespace = self._load_reload_function(bundle, control, "old-sha", "wrong-sha", clock)

            self.assertFalse(namespace["_apply_drive_code_update"](timeout_seconds=60))
            self.assertEqual(namespace["pipeline"], "old-pipeline")
            self.assertEqual(namespace["ACTIVE_BUNDLE_SHA"], "old-sha")
            self.assertEqual(json.loads(control.read_text(encoding="utf-8"))["action"], "reload")


if __name__ == "__main__":
    unittest.main()
