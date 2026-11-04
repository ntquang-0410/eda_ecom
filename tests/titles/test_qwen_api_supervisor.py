import argparse
import unittest
from unittest.mock import Mock, patch

import supervise_title_translation_qwen3_hf_api as supervisor


class QwenApiSupervisorTests(unittest.TestCase):
    def test_full_stage_starts_with_smoke_gate_command(self):
        args = argparse.Namespace(
            source=supervisor.ROOT / "source.parquet",
            workdir=supervisor.ROOT / "run",
            stage="full",
            workers=8,
            provider_policy="deepinfra",
            max_cost_usd=1.0,
            mirror_dir=None,
            allow_paid_inference=False,
            dry_run=False,
        )
        command = supervisor.build_worker_command(args, "smoke")
        self.assertIn("--stage", command)
        self.assertEqual(command[command.index("--stage") + 1], "smoke")
        self.assertNotIn("--allow-paid-inference", command)
        self.assertNotIn("Bearer", " ".join(command))
        self.assertNotIn("HF_TOKEN", " ".join(command))

    def test_source_digest_tracks_runner_and_quality_module(self):
        self.assertEqual(len(supervisor.source_digest()), 64)

    def test_reload_requested_revalidates_then_relaunches_from_same_workdir(self):
        args = argparse.Namespace(
            source=supervisor.ROOT / "source.parquet",
            workdir=supervisor.ROOT / "run",
            stage="smoke",
            workers=8,
            provider_policy="deepinfra",
            max_cost_usd=1.0,
            mirror_dir=None,
            allow_paid_inference=True,
            dry_run=False,
            poll_seconds=0.25,
        )
        first_process, second_process = Mock(), Mock()
        first_process.wait.return_value = 0
        second_process.wait.return_value = 0

        with patch.object(supervisor, "source_digest", return_value="same-source"), \
             patch.object(supervisor, "validate_code", side_effect=[True, True]) as validate, \
             patch.object(supervisor, "read_status", side_effect=[
                 {"state": "reload_requested", "remaining": 1},
                 {"state": "all_titles_complete", "remaining": 0},
             ]), \
             patch.object(supervisor.subprocess, "Popen", side_effect=[
                 first_process, second_process,
             ]) as popen, \
             patch("builtins.print"):
            result = supervisor.supervise(args)

        self.assertEqual(result, 0)
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(popen.call_args_list[0].args[0], popen.call_args_list[1].args[0])
        self.assertEqual(popen.call_args_list[1].args[0][
            popen.call_args_list[1].args[0].index("--workdir") + 1
        ], str(args.workdir))


if __name__ == "__main__":
    unittest.main()
