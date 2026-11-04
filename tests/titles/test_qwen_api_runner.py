import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_title_translation_qwen3_hf_api as runner


class QwenApiRunnerMirrorTests(unittest.TestCase):
    def test_default_source_resolves_configured_or_flat_dataset_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            configured = root / "data" / "cleaned" / "bilingual_zh_vi_cado.parquet"
            flat = root / "bilingual_zh_vi_cado.parquet"

            self.assertEqual(
                runner.default_source_path(root, "data/cleaned/bilingual_zh_vi_cado.parquet"),
                configured,
            )
            flat.parent.mkdir(parents=True, exist_ok=True)
            flat.touch()
            self.assertEqual(
                runner.default_source_path(root, "data/cleaned/bilingual_zh_vi_cado.parquet"),
                flat,
            )
            configured.parent.mkdir(parents=True, exist_ok=True)
            configured.touch()
            self.assertEqual(
                runner.default_source_path(root, "data/cleaned/bilingual_zh_vi_cado.parquet"),
                configured,
            )

    def test_status_is_mirrored_without_raw_titles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workdir = root / "run"
            mirror = root / "drive"
            status = {
                "state": "running",
                "recent_errors": [{"title_sha256_prefix": "0123456789ab", "error": "timeout"}],
            }

            runner._write_status(workdir, status, mirror)

            local = json.loads((workdir / "qwen_api_live_status.json").read_text(encoding="utf-8"))
            mirrored = json.loads((mirror / "qwen_api_live_status.json").read_text(encoding="utf-8"))
            self.assertEqual(local["state"], "running")
            self.assertEqual(mirrored["state"], "running")
            self.assertEqual(local["recent_errors"], status["recent_errors"])
            self.assertNotIn("title", local["recent_errors"][0])

    def test_sqlite_backup_is_readable_and_connections_are_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "cache.sqlite3"
            mirror = root / "drive"
            cache = runner.TranslationCache(source, "test-config")
            cache.put("源标题", {
                "status": "ok", "output_text": "Tiêu đề tiếng Việt",
                "input_tokens": 4, "output_tokens": 5,
            })
            runner._mirror_cache(source, mirror)

            db = sqlite3.connect(mirror / source.name)
            try:
                row = db.execute(
                    "SELECT source_title,output_text,status FROM translations"
                ).fetchone()
            finally:
                db.close()
            self.assertEqual(row, ("源标题", "Tiêu đề tiếng Việt", "ok"))

            renamed = root / "cache-renamed.sqlite3"
            source.replace(renamed)
            self.assertTrue(renamed.exists())

    def test_retry_cost_and_tokens_accumulate_across_the_same_config(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = runner.TranslationCache(Path(temp) / "cache.sqlite3", "same-config")
            cache.put("源标题", {
                "status": "error", "output_text": "", "input_tokens": 10,
                "output_tokens": 20, "attempts": 1, "estimated_cost_usd": 0.01,
            })
            cache.put("源标题", {
                "status": "ok", "output_text": "Tiêu đề mới",
                "input_tokens": 11, "output_tokens": 21,
                "attempts": 2, "estimated_cost_usd": 0.02,
            })
            row = cache.get_many(["源标题"])["源标题"]
            self.assertEqual(row["input_tokens"], 21)
            self.assertEqual(row["output_tokens"], 41)
            self.assertEqual(row["attempts"], 3)
            self.assertAlmostEqual(row["estimated_cost_usd"], 0.03)
            self.assertAlmostEqual(cache.total_estimated_cost_usd(), 0.03)

    def test_changed_code_requests_reload_only_after_saving_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []

            def fake_request(title, token):
                calls.append(title)
                return {
                    "status": "ok", "output_text": "Tiêu đề tiếng Việt",
                    "truncated": False, "input_tokens": 10, "output_tokens": 8,
                    "estimated_cost_usd": 0.001, "attempts": 1,
                    "provider": "mock", "model": "mock-qwen",
                }

            with patch.dict("os.environ", {"HF_TOKEN": "unit-test-only-token"}), \
                 patch.object(runner, "check_source", return_value=pd.DataFrame({"title_zh": ["甲", "乙"]})), \
                 patch.object(runner, "source_titles", return_value=["甲", "乙"]), \
                 patch.object(runner, "_current_code_sha256", side_effect=["before", "after"]):
                result = runner.run(
                    source_path=root / "unused.parquet",
                    workdir=root / "run",
                    stage="smoke",
                    workers=1,
                    max_cost_usd=1.0,
                    allow_paid_inference=True,
                    request_fn=fake_request,
                )

            self.assertEqual(result["state"], "reload_requested")
            self.assertEqual(result["phase"], "code_changed_at_batch_boundary")
            self.assertEqual(calls, ["甲"])
            cache = runner.TranslationCache(root / "run" / "qwen3_14b_hf_api.sqlite3", result["config_id"])
            self.assertEqual(cache.totals()["completed_records"], 1)


class QwenApiRequestTests(unittest.TestCase):
    def test_cache_signature_contains_prompt_hash_and_generation_settings(self):
        payload = runner.config_payload("deepinfra", 8)
        self.assertEqual(
            payload["system_prompt_sha256"],
            runner.hashlib.sha256(runner.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(payload["generation"]["temperature"], runner.TEMPERATURE)
        self.assertEqual(payload["generation"]["top_p"], runner.TOP_P)

    def test_provider_request_uses_qwen_nonthinking_prompt_and_returns_usage(self):
        captured = {}

        class Response:
            status_code = 200
            ok = True
            headers = {"x-inference-provider": "test-provider"}
            text = ""

            @staticmethod
            def json():
                return {
                    "model": "Qwen/Qwen3-14B",
                    "choices": [{
                        "message": {"content": "Túi xách nữ phong cách retro"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                }

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return Response()

        result = runner.call_provider("复古女包", "hf_test_token", post=fake_post)

        self.assertEqual(captured["url"], runner.API_URL)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer hf_test_token")
        self.assertIn("/no_think", captured["json"]["messages"][1]["content"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["provider"], "test-provider")
        self.assertEqual(result["input_tokens"], 12)
        self.assertEqual(result["output_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
