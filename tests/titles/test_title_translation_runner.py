import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_title_translation_local as runner


class RunnerWorkflowTests(unittest.TestCase):
    def test_key_is_read_from_process_environment_without_echoing_or_prompting(self):
        with (
            patch.dict(runner.os.environ, {"GEMINI_API_KEY": "local-secret"}, clear=True),
            patch.object(runner, "getpass") as hidden_prompt,
        ):
            self.assertEqual(runner._get_google_api_key(), "local-secret")
        hidden_prompt.assert_not_called()

    def test_key_falls_back_to_hidden_prompt_when_not_configured(self):
        with (
            patch.dict(runner.os.environ, {}, clear=True),
            patch.object(runner, "getpass", return_value="one-time-secret") as hidden_prompt,
        ):
            self.assertEqual(runner._get_google_api_key(), "one-time-secret")
        hidden_prompt.assert_called_once()

    def test_gemma_full_resumes_checkpointed_sessions_until_complete(self):
        with (
            patch.object(
                runner,
                "_context",
                return_value={
                    "mode": "full",
                    "lock": {"config_id": "cfg"},
                    "state": {
                        "gemma_batch_canary": {
                            "config_id": "cfg",
                            "passed": True,
                            "min_interval_seconds": runner.GEMMA_MIN_INTERVAL_SECONDS,
                            "selected_batch_size": 2,
                        },
                        "models": {},
                    },
                },
            ),
            patch.object(
                runner.pipeline,
                "run_stage",
                side_effect=[
                    {"status": "checkpoint_saved_continue"},
                    {"status": "all_titles_complete"},
                ],
            ) as run_stage,
        ):
            status = runner._run_full("gemma4", "rotated-google-api-key")

        self.assertEqual(status, "all_titles_complete")
        self.assertEqual(run_stage.call_count, 2)
        for call in run_stage.call_args_list:
            self.assertEqual(call.kwargs["full_titles_per_session"], 200)
            self.assertEqual(call.kwargs["gemma_batch_size"], 2)

    def test_batch_canary_must_match_current_request_interval(self):
        matching = {
            "config_id": "cfg",
            "passed": True,
            "min_interval_seconds": runner.GEMMA_MIN_INTERVAL_SECONDS,
        }
        old_interval = {**matching, "min_interval_seconds": runner.GEMMA_MIN_INTERVAL_SECONDS + 1}
        self.assertTrue(runner._canary_matches_runtime(matching, "cfg"))
        self.assertFalse(runner._canary_matches_runtime(old_interval, "cfg"))
        self.assertFalse(runner._canary_matches_runtime({k: v for k, v in matching.items() if k != "min_interval_seconds"}, "cfg"))

    def test_gemma_smoke_runs_without_local_gpu_and_only_calls_gemma(self):
        with (
            patch.dict(runner.os.environ, {}, clear=True),
            patch.object(runner.sys, "stdin") as stdin,
            patch.object(runner, "SOURCE") as source,
            patch.object(runner, "getpass", return_value="rotated-google-api-key"),
            patch.object(runner, "input", return_value="START", create=True),
            patch.object(runner, "_local_qwen_gpu_ready") as gpu_check,
            patch.object(runner, "_run_gated_stage", return_value=True) as gated,
            patch.object(runner, "_run_full") as full,
        ):
            stdin.isatty.return_value = True
            source.exists.return_value = True
            result = runner.main(["gemma-smoke"])

        self.assertEqual(result, 0)
        gpu_check.assert_not_called()
        gated.assert_called_once_with("smoke", "gemma4", "rotated-google-api-key")
        full.assert_not_called()

    def test_gemma_full_is_explicit_and_never_runs_qwen_or_uploads(self):
        with (
            patch.dict(runner.os.environ, {}, clear=True),
            patch.object(runner.sys, "stdin") as stdin,
            patch.object(runner, "SOURCE") as source,
            patch.object(runner, "getpass", return_value="rotated-google-api-key"),
            patch.object(runner, "input", return_value="START FULL", create=True),
            patch.object(runner, "_run_full", return_value="all_titles_complete") as full,
            patch.object(runner, "_print_local_outputs") as print_outputs,
        ):
            stdin.isatty.return_value = True
            source.exists.return_value = True
            result = runner.main(["gemma-full"])

        self.assertEqual(result, 0)
        full.assert_called_once_with(
            "gemma4",
            "rotated-google-api-key",
            defer_full_quality_review=True,
        )
        print_outputs.assert_not_called()

    def test_run_uses_google_api_and_never_uploads(self):
        with (
            patch.dict(runner.os.environ, {}, clear=True),
            patch.object(runner.sys, "stdin") as stdin,
            patch.object(runner, "SOURCE") as source,
            patch.object(runner, "_local_qwen_gpu_ready", return_value=True),
            patch.object(runner, "getpass", return_value="google-api-key") as getpass,
            patch.object(runner, "input", return_value="START", create=True),
            patch.object(runner, "_run_gated_stage", return_value=True) as gated,
            patch.object(runner, "_run_full", return_value="all_titles_complete") as full,
            patch.object(runner, "_print_local_outputs") as print_outputs,
            patch.object(runner.pipeline, "upload_outputs_to_hf") as upload,
        ):
            stdin.isatty.return_value = True
            source.exists.return_value = True
            result = runner.main(["run"])

        self.assertEqual(result, 0)
        self.assertEqual(getpass.call_count, 1)
        self.assertIn("GEMINI_API_KEY", getpass.call_args.args[0])
        self.assertEqual(gated.call_count, 6)
        self.assertEqual(full.call_count, 2)
        print_outputs.assert_called_once()
        upload.assert_not_called()

    def test_run_without_local_gpu_stops_before_requesting_api_key(self):
        with (
            patch.object(runner.sys, "stdin") as stdin,
            patch.object(runner, "_local_qwen_gpu_ready", return_value=False),
            patch.object(runner, "getpass") as getpass,
        ):
            stdin.isatty.return_value = True
            result = runner.main(["run"])

        self.assertEqual(result, 2)
        getpass.assert_not_called()

    def test_upload_is_a_separate_explicit_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "hybrid_run_state.json").write_text("{}", encoding="utf-8")
            with (
                patch.object(runner, "WORKDIR", root),
                patch.object(runner.sys, "stdin") as stdin,
                patch.object(runner, "getpass", return_value="write-token") as getpass,
                patch.object(runner, "input", return_value="UPLOAD", create=True),
                patch.object(runner, "_print_local_outputs"),
                patch.object(
                    runner.pipeline,
                    "upload_outputs_to_hf",
                    return_value={"readback_verified": True},
                ) as upload,
            ):
                stdin.isatty.return_value = True
                result = runner.main(["upload"])

        self.assertEqual(result, 0)
        getpass.assert_called_once()
        upload.assert_called_once_with(
            root,
            hf_token="write-token",
            source_path=runner.SOURCE,
        )

    def test_upload_cancellation_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "hybrid_run_state.json").write_text("{}", encoding="utf-8")
            with (
                patch.object(runner, "WORKDIR", root),
                patch.object(runner.sys, "stdin") as stdin,
                patch.object(runner, "getpass", return_value="write-token"),
                patch.object(runner, "input", return_value="NO", create=True),
                patch.object(runner, "_print_local_outputs"),
                patch.object(runner.pipeline, "upload_outputs_to_hf") as upload,
            ):
                stdin.isatty.return_value = True
                result = runner.main(["upload"])

        self.assertEqual(result, 0)
        upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
