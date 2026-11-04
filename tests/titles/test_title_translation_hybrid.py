import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pandas as pd

import title_translation_hybrid as hybrid
import title_translation_pipeline as quality


class HybridModelConfigurationTests(unittest.TestCase):
    def test_models_are_the_selected_hybrid_pair(self):
        self.assertEqual(MODELS := hybrid.MODELS["gemma4"]["model_id"], "google/gemma-4-26B-A4B-it")
        self.assertEqual(hybrid.MODELS["qwen35"]["model_id"], "Qwen/Qwen3.5-9B")
        self.assertEqual(hybrid.MODELS["gemma4"]["backend"], "google_gemini_api")
        self.assertEqual(hybrid.MODELS["gemma4"]["api_model_id"], "gemma-4-26b-a4b-it")
        self.assertEqual(hybrid.MODELS["gemma4"]["provider"], "google_ai_studio")
        self.assertEqual(hybrid.MODELS["qwen35"]["backend"], "local_transformers_cuda")
        self.assertEqual(hybrid.MODELS["qwen35"]["provider"], "local_cuda_8bit")
        self.assertTrue(MODELS)

    def test_gemma_client_disables_sdk_retries_so_paced_retry_loop_is_authoritative(self):
        captured = {}
        fake_genai = ModuleType("google.genai")
        fake_genai.types = SimpleNamespace(
            HttpRetryOptions=lambda **kwargs: kwargs,
            HttpOptions=lambda **kwargs: kwargs,
        )
        fake_genai.Client = lambda **kwargs: captured.update(kwargs) or "client"
        fake_google = ModuleType("google")
        fake_google.genai = fake_genai

        with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai}):
            client = hybrid._new_gemma_client("test-key")

        self.assertEqual(client, "client")
        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["http_options"]["retry_options"]["attempts"], 1)

    def test_qwen_prompt_contains_source_and_translation_constraints(self):
        title = "儿童书包 4G TS03 2斤"
        message = hybrid._qwen_messages(title)[0]["content"]
        self.assertIn(title, message)
        self.assertIn("numbers", message)
        self.assertIn("Vietnamese", message)

    def test_gemma_google_sdk_contract(self):
        fake_types = SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: kwargs,
            ThinkingConfig=lambda **kwargs: kwargs,
        )
        fake_genai = SimpleNamespace(types=fake_types)
        response = SimpleNamespace(
            text="Áo khoác trẻ em",
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(candidates_token_count=8),
            model_version="managed-version",
            response_id="response-123",
        )
        captured = {}

        def generate_content(**kwargs):
            captured.update(kwargs)
            return response

        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        with patch.dict("sys.modules", {"google.genai": fake_genai}):
            parsed = hybrid._gemma_response(client, "儿童外套", 256)
        self.assertEqual(captured["model"], "gemma-4-26b-a4b-it")
        self.assertEqual(captured["contents"], "儿童外套")
        self.assertEqual(captured["config"]["temperature"], 0.0)
        self.assertEqual(captured["config"]["max_output_tokens"], 256)
        self.assertEqual(captured["config"]["thinking_config"]["thinking_level"], "minimal")
        self.assertEqual(parsed["output_text"], "Áo khoác trẻ em")
        self.assertFalse(parsed["truncated"])
        self.assertEqual(parsed["model_version"], "managed-version")
        self.assertEqual(parsed["response_id"], "response-123")

    def test_daily_quota_classifier_does_not_mark_generic_429_as_daily(self):
        daily = RuntimeError("GenerateRequestsPerDayPerProjectPerModel-FreeTier quota exceeded")
        minute = RuntimeError("rate exceeded, retry in 5.0s")
        self.assertTrue(hybrid._gemma_is_daily_quota_error(daily))
        self.assertFalse(hybrid._gemma_is_daily_quota_error(minute))

    def test_bad_api_credentials_fail_fast_instead_of_retrying_every_title(self):
        invalid = RuntimeError("API key not valid. Please pass a valid API key.")
        transient = RuntimeError("temporary server error")
        self.assertTrue(hybrid._gemma_is_configuration_error(invalid))
        self.assertFalse(hybrid._gemma_is_configuration_error(transient))

    def test_gemma_billing_error_fails_fast_without_retrying(self):
        class Response:
            status_code = 402
            headers = {}

        class BillingError(RuntimeError):
            response = Response()

        calls = []

        class FakeClient:
            pass

        with patch.object(hybrid, "_gemma_response", side_effect=BillingError("payment required")) as generate:
            with self.assertRaises(hybrid.GemmaConfigurationError):
                hybrid._call_gemma_with_retries(
                    FakeClient(), "\u513f\u7ae5\u5916\u5957", 256, hybrid.RequestPacer(0), "secret"
                )
        calls.append(generate.call_count)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], 1)

    def test_gemma_transient_retry_logs_status_without_api_key(self):
        failure = RuntimeError("temporary server failure; api_key=secret-api-key")
        failure.response = SimpleNamespace(status_code=500)
        success = {
            "output_text": "Áo khoác trẻ em",
            "input_tokens": 10,
            "output_tokens": 5,
            "truncated": False,
        }
        with (
            patch.object(hybrid, "_gemma_response", side_effect=[failure, success]),
            patch.object(hybrid.time, "sleep"),
            patch("builtins.print") as log,
        ):
            result = hybrid._call_gemma_with_retries(
                object(), "儿童外套", 256, hybrid.RequestPacer(0), "secret-api-key"
            )
        self.assertEqual(result["api_attempts"], 2)
        messages = " ".join(str(call.args[0]) for call in log.call_args_list)
        self.assertIn("HTTP 500", messages)
        self.assertNotIn("secret-api-key", messages)

class HybridCacheAndGateTests(unittest.TestCase):
    def test_cache_survives_reopen_with_provider_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gemma.sqlite3"
            cache = hybrid.Cache(path, "gemma4", "provider_managed", "cfg")
            cache.put(
                "儿童外套",
                {
                    "output_text": "Áo khoác trẻ em",
                    "status": "ok",
                    "truncated": False,
                    "input_tokens": 19,
                    "output_tokens": 8,
                    "attempts": 1,
                    "model_version": "managed-v1",
                    "response_id": "resp-1",
                },
            )
            reopened = hybrid.Cache(path, "gemma4", "provider_managed", "cfg")
            result = reopened.get_many(["儿童外套", "không có"])
            self.assertEqual(result["儿童外套"]["model_version"], "managed-v1")
            self.assertEqual(result["儿童外套"]["response_id"], "resp-1")
            self.assertEqual(result["儿童外套"]["input_tokens"], 19)
            self.assertEqual(result["儿童外套"]["generation_prompt_version"], hybrid.PROMPT_VERSION)
            self.assertNotIn("không có", result)

    def test_cache_retains_repair_prompt_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = hybrid.Cache(Path(directory) / "gemma.sqlite3", "gemma4", "provider", "cfg")
            cache.put(
                "儿童外套",
                {
                    "output_text": "Áo khoác trẻ em",
                    "status": "ok",
                    "generation_prompt_version": hybrid.GEMMA_QA_REPAIR_PROMPT_VERSION,
                },
            )
            row = hybrid.Cache(Path(directory) / "gemma.sqlite3", "gemma4", "provider", "cfg").get_many(
                ["儿童外套"]
            )["儿童外套"]
            self.assertEqual(row["generation_prompt_version"], hybrid.GEMMA_QA_REPAIR_PROMPT_VERSION)

    def test_cache_roundtrips_fractional_paired_api_request_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gemma.sqlite3"
            cache = hybrid.Cache(path, "gemma4", "provider", "cfg")
            cache.put(
                "儿童外套",
                {
                    "output_text": "Áo khoác trẻ em",
                    "status": "ok",
                    "attempts": 1,
                    "api_requests": 0.5,
                },
            )
            row = hybrid.Cache(path, "gemma4", "provider", "cfg").get_many(["儿童外套"])["儿童外套"]
            self.assertEqual(row["attempts"], 1)
            self.assertEqual(row["api_requests"], 0.5)

    def test_gemma_paired_response_maps_exact_ids_and_splits_usage(self):
        fake_types = SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: kwargs,
            ThinkingConfig=lambda **kwargs: kwargs,
        )
        response = SimpleNamespace(
            text="[[1]]\tÁo khoác trẻ em\n[[2]]\tÁo khoác nam",
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(prompt_token_count=30, candidates_token_count=12),
            model_version="gemma-test",
            response_id="paired-response",
        )
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response))
        with patch.dict("sys.modules", {"google": SimpleNamespace(), "google.genai": SimpleNamespace(types=fake_types)}):
            result = hybrid._gemma_batch_response(client, ["儿童外套", "男士外套"], 512)

        self.assertIsNone(result["parse_error"])
        self.assertEqual([item["output_text"] for item in result["items"]], ["Áo khoác trẻ em", "Áo khoác nam"])
        self.assertEqual(sum(item["input_tokens"] for item in result["items"]), 30)
        self.assertEqual(sum(item["output_tokens"] for item in result["items"]), 12)
        self.assertEqual(result["response_id"], "paired-response")

    def test_gemma_paired_response_rejects_missing_duplicate_or_unindexed_lines(self):
        fake_types = SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: kwargs,
            ThinkingConfig=lambda **kwargs: kwargs,
        )
        cases = (
            "[[1]]\tÁo khoác trẻ em",
            "[[1]]\tÁo khoác trẻ em\n[[1]]\tÁo khoác nam",
            "Bản dịch: Áo khoác trẻ em\n[[2]]\tÁo khoác nam",
        )
        for raw_text in cases:
            with self.subTest(raw_text=raw_text):
                response = SimpleNamespace(
                    text=raw_text,
                    candidates=[SimpleNamespace(finish_reason="STOP")],
                    usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
                )
                client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response))
                with patch.dict("sys.modules", {"google": SimpleNamespace(), "google.genai": SimpleNamespace(types=fake_types)}):
                    result = hybrid._gemma_batch_response(client, ["儿童外套", "男士外套"], 512)
                self.assertIsNone(result["items"])
                self.assertTrue(result["parse_error"])

    def test_gemma_canary_metrics_measure_request_retry_overhead(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = hybrid.Cache(Path(directory) / "gemma.sqlite3", "gemma4", "provider", "cfg")
            titles = [f"title-{index}" for index in range(6)]
            for title in titles:
                cache.put(title, {"output_text": "bản dịch", "status": "ok", "api_requests": 1 / 3})
            with patch.object(
                hybrid.quality,
                "check_translation",
                return_value={"status": "pass", "hard_flags": []},
            ):
                baseline = hybrid._gemma_canary_metrics(titles, cache, batch_size=3)
            self.assertTrue(baseline["quality_passed"])
            self.assertTrue(baseline["retry_overhead_passed"])
            self.assertAlmostEqual(baseline["retry_overhead_rate"], 0.0)

            cache.put(titles[0], {"output_text": "bản dịch", "status": "ok", "api_requests": 1.0})
            with patch.object(
                hybrid.quality,
                "check_translation",
                return_value={"status": "pass", "hard_flags": []},
            ):
                retried = hybrid._gemma_canary_metrics(titles, cache, batch_size=3)
            self.assertFalse(retried["retry_overhead_passed"])
            self.assertGreater(retried["retry_overhead_rate"], hybrid.MAX_GEMMA_BATCH_RETRY_OVERHEAD)

    def test_dashcam_repair_prompt_explains_channel_and_spec_rules(self):
        instruction = hybrid._gemma_qa_repair_instruction(
            "行车记录仪 四录 12寸",
            "camera dashcam",
            ["number_mismatch", "cjk_remaining"],
        )
        self.assertIn("四录 mean three-channel and four-channel recording", instruction)
        self.assertIn("12", instruction)
        self.assertIn("number_mismatch", instruction)

    def test_source_anchored_dashcam_repairs_preserve_explicit_specs(self):
        cases = [
            (
                "ADAS12寸行车记录仪",
                "Camera hành trình ADAS12",
                "màn hình 12 inch",
            ),
            (
                "前4K后2K行车记录仪",
                "Camera hành trình 4K后2K",
                "4K trước, 2K sau",
            ),
            (
                "高清单镜头行车记录仪",
                "Camera 高清 với ống kính",
                "độ nét cao với một ống kính",
            ),
            (
                "双录行车记录仪",
                "Camera hành trình",
                "ghi hình hai kênh",
            ),
            (
                "2分割行车记录仪",
                "Camera hành trình",
                "chia màn hình thành 2 phần",
            ),
        ]
        for source, draft, expected in cases:
            with self.subTest(source=source):
                repaired, rules = hybrid._repair_gemma_specs_from_source(source, draft)
                self.assertIn(expected.casefold(), repaired.casefold())
                self.assertTrue(rules)

    def test_source_anchored_title_repairs_fix_duration_variants_and_trailing_duplicates(self):
        duration_source = "韩国GIK5分钟面膜Gik血清胶原蛋白面膜21片新款补水保湿"
        duration_draft = "Mặt nạ GIK5 21 phút Gik serum collagen 21 miếng mẫu mới cấp ẩm"
        duration_fixed, duration_rules = hybrid._repair_gemma_specs_from_source(
            duration_source, duration_draft
        )
        self.assertIn("GIK5 5 phút", duration_fixed)
        self.assertEqual(quality.check_translation(duration_source, duration_fixed)["status"], "pass")
        self.assertIn("source-duration:5-minutes", duration_rules)

        variant_source = "韩国进口诞兴薇清潭洞小鲸鱼面膜补水B款M款保湿呵护肌肤"
        variant_draft = "Mặt nạ cá voi nhỏ Danxingwei Cheongdam-dong nhập khẩu Hàn Quốc cấp ẩm B款M款"
        variant_fixed, variant_rules = hybrid._repair_gemma_specs_from_source(
            variant_source, variant_draft
        )
        self.assertIn("loại B loại M", variant_fixed)
        self.assertFalse(quality.contains_cjk(variant_fixed))
        self.assertEqual(quality.check_translation(variant_source, variant_fixed)["status"], "pass")
        self.assertIn("letter-variant:B", variant_rules)
        self.assertIn("letter-variant:M", variant_rules)

        huaxi_title = "2.0米蓓尔蓝绷带涂抹面膜华熙生物"
        huaxi_draft = "Mặt nạ bôi Belle Blue Bandage 2.0m 华熙生物"
        huaxi_fixed, huaxi_rules = hybrid._repair_gemma_specs_from_source(huaxi_title, huaxi_draft)
        self.assertIn("Huaxi Bio", huaxi_fixed)
        self.assertFalse(quality.contains_cjk(huaxi_fixed))
        self.assertEqual(quality.check_translation(huaxi_title, huaxi_fixed)["status"], "pass")
        self.assertIn("brand:huaxi-bio", huaxi_rules)

        cka_title = "c咖泥膜小罐第三代清洁泥膜"
        cka_draft = "Mặt nạ bùn c咖 thế hệ thứ ba làm sạch"
        cka_fixed, cka_rules = hybrid._repair_gemma_specs_from_source(cka_title, cka_draft)
        self.assertIn("C-Ka", cka_fixed)
        self.assertFalse(quality.contains_cjk(cka_fixed))
        self.assertEqual(quality.check_translation(cka_title, cka_fixed)["status"], "pass")
        self.assertIn("brand:c-ka-pinyin", cka_rules)

        combo_source = "【严选377组合】淡斑美白面膜377系列美白精华次抛爽肤水面霜"
        combo_draft = "Combo 377 tuyển chọn mặt nạ trắng da dòng 377 tinh chất dùng một lần 377 377"
        combo_fixed, combo_rules = hybrid._repair_gemma_specs_from_source(combo_source, combo_draft)
        self.assertFalse(combo_fixed.endswith("377"))
        self.assertEqual(quality.check_translation(combo_source, combo_fixed)["status"], "pass")
        self.assertEqual(combo_rules.count("trim-trailing-duplicate-number:377"), 2)

    def test_dashcam_repair_does_not_duplicate_existing_video_count_and_restores_hours(self):
        title = "行车记录仪后视镜三镜头WiFi高清1080P倒车影像全屏触控三录同显"
        draft = (
            "Gương chiếu hậu camera hành trình 3 ống kính WiFi HD 1080P màn hình cảm ứng "
            "toàn màn hình hỗ trợ camera lùi hiển thị đồng thời 3 video"
        )
        repaired, rules = hybrid._repair_gemma_specs_from_source(title, draft)
        self.assertNotIn("ghi hình ba kênh", repaired)
        self.assertEqual(quality.check_translation(title, repaired)["status"], "pass")
        self.assertEqual(rules, [])

        old_normalization_draft = draft + "; ghi hình ba kênh"
        cleaned, cleanup_rules = hybrid._repair_gemma_specs_from_source(title, old_normalization_draft)
        self.assertNotIn("ghi hình ba kênh", cleaned)
        self.assertEqual(quality.check_translation(title, cleaned)["status"], "pass")
        self.assertIn("duplicate-recording-channel:三录", cleanup_rules)

        duration_title = "WIFI倒车影像GPS24小时监控"
        duration_draft = "Kết nối WIFI, camera lùi GPS24 giám sát trong nhiều giờ"
        duration_repaired, duration_rules = hybrid._repair_gemma_specs_from_source(
            duration_title, duration_draft
        )
        self.assertIn("24 giờ", duration_repaired)
        self.assertEqual(quality.check_translation(duration_title, duration_repaired)["status"], "pass")
        self.assertIn("monitoring-duration:24-hours", duration_rules)

    def test_quality_gate_stop_is_resolved_only_after_window_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            titles = [f"儿童外套 {index}" for index in range(200)]
            workdir = Path(directory)
            context = {
                "frame": pd.DataFrame({"title_zh": titles}),
                "workdir": workdir,
                "lock": {
                    "config_id": "cfg",
                    "models": {"gemma4": {"revision": "provider"}},
                },
            }
            cache = hybrid.Cache(workdir / "cache_gemma4.sqlite3", "gemma4", "provider", "cfg")
            for index, title in enumerate(titles):
                output = "错误翻译 999" if index < 12 else f"Áo khoác trẻ em {index}"
                cache.put(title, {"output_text": output, "status": "ok", "truncated": False})
            stopped = {
                "status": "stopped_quality_gate",
                "failed_window": {
                    "start": 0,
                    "end": 200,
                    "incomplete_rate": 0.0,
                    "technical_qa_failure_rate": 0.06,
                },
            }
            self.assertFalse(hybrid._full_gate_stop_is_resolved(stopped, context, "gemma4"))
            for index in range(12):
                cache.put(
                    titles[index],
                    {"output_text": f"Áo khoác trẻ em {index}", "status": "ok", "truncated": False},
                )
            self.assertTrue(hybrid._full_gate_stop_is_resolved(stopped, context, "gemma4"))

    def test_smoke_and_pilot_quality_gates(self):
        titles = ["儿童外套", "女装", "车载屏 4G TS03"]
        with tempfile.TemporaryDirectory() as directory:
            cache = hybrid.Cache(Path(directory) / "qwen.sqlite3", "qwen35", "sha", "cfg")
            for title, text in zip(titles, ["Áo khoác trẻ em", "Thời trang nữ", "Màn hình xe 4G TS03"]):
                cache.put(title, {"output_text": text, "status": "ok", "truncated": False})
            smoke = hybrid._stage_gate("smoke", "qwen35", titles[:2], {}, cache)
            pilot = hybrid._stage_gate(
                "pilot",
                "qwen35",
                titles,
                {"pilot": {"representative": titles[:2], "challenge": titles[2:]}},
                cache,
            )
        self.assertTrue(smoke["passed"])
        self.assertTrue(pilot["passed"])
        self.assertEqual(pilot["representative_technical_pass_rate"], 1.0)
        self.assertFalse(pilot["semantic_verified"] if "semantic_verified" in pilot else False)

    def test_smoke_does_not_pass_when_all_requests_complete_but_technical_qa_is_low(self):
        title = "车载屏 4G TS03"
        with tempfile.TemporaryDirectory() as directory:
            cache = hybrid.Cache(Path(directory) / "gemma.sqlite3", "gemma4", "sha", "cfg")
            cache.put(title, {"output_text": "Màn hình xe 5G TS04", "status": "ok", "truncated": False})
            result = hybrid._stage_gate("smoke", "gemma4", [title], {}, cache)
        self.assertEqual(result["completion_rate"], 1.0)
        self.assertEqual(result["technical_pass_rate"], 0.0)
        self.assertFalse(result["passed"])

    def test_full_requires_only_the_selected_model_validation(self):
        context = {
            "lock": {"config_id": "cfg"},
            "state": {
                "config_id": "cfg",
                "models": {
                    "gemma4": {"stages": {"validate": {"passed": True, "config_id": "cfg"}}},
                    "qwen35": {"stages": {}},
                },
            },
        }
        hybrid._check_prerequisites(context, "gemma4", "full")
        with self.assertRaisesRegex(RuntimeError, "qwen35 validation"):
            hybrid._check_prerequisites(context, "qwen35", "full")
        context["state"]["models"]["qwen35"]["stages"]["validate"] = {
            "passed": True,
            "config_id": "cfg",
        }
        hybrid._check_prerequisites(context, "gemma4", "full")

    def test_explicit_qwen_full_can_defer_validation_and_allow_gpu_reassignment(self):
        context = {
            "lock": {"config_id": "cfg"},
            "state": {"config_id": "cfg", "models": {"qwen35": {"stages": {}}}},
        }
        with self.assertRaisesRegex(RuntimeError, "qwen35 validation"):
            hybrid._check_prerequisites(context, "qwen35", "full")
        hybrid._check_prerequisites(
            context,
            "qwen35",
            "full",
            allow_unvalidated_full=True,
            defer_full_quality_review=True,
        )

    def test_deferred_qwen_full_exports_without_window_qa_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            titles = ["儿童外套", "女装"]
            context = {
                "mode": "full",
                "model_key": "qwen35",
                "workdir": root,
                "frame": pd.DataFrame({"title_zh": titles}),
                "lock": {
                    "config_id": "cfg",
                    "models": {"qwen35": {"revision": "hub-sha", "model_id": "Qwen/Qwen3.5-9B"}},
                },
                "state": {
                    "config_id": "cfg",
                    "models": {"qwen35": {"stages": {}}, "gemma4": {"stages": {}}},
                },
                "samples": {},
                "qwen_batch_size": 4,
            }
            cache = hybrid.Cache(root / "cache_qwen35.sqlite3", "qwen35", "hub-sha", "cfg")

            def generate(batch, run_context, hf_token, batch_size=1):
                for title in batch:
                    cache.put(title, {"output_text": "Ứng viên cần rà soát", "status": "ok", "truncated": False})
                return {"status": "finished_current_batch", "completed": len(batch)}

            output_path = root / "bilingual_zh_vi_cado_title_qwen35.parquet"
            output_path.write_bytes(b"test-output")
            with (
                patch.object(hybrid, "_check_or_write_runtime_lock"),
                patch.object(hybrid, "_cache", return_value=cache),
                patch.object(hybrid, "_run_qwen", side_effect=generate),
                patch.object(hybrid, "_full_window_gates", side_effect=AssertionError("QA gate should be deferred")),
                patch.object(hybrid, "export_model_parquet", return_value=output_path),
            ):
                result = hybrid.run_stage(
                    context,
                    batch_size=4,
                    full_titles_per_session=10,
                    allow_unvalidated_full=True,
                    defer_full_quality_review=True,
                )

        self.assertEqual(result["status"], "all_titles_complete")
        self.assertTrue(result["quality_review_deferred"])
        self.assertTrue(result["validation_bypassed"])
        self.assertEqual(result["completed_unique_titles"], 2)
        self.assertEqual(result["remaining_unique_titles"], 0)

    def test_deferred_gemma_full_resumes_quality_stop_and_keeps_review_deferred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            titles = ["儿童外套"]
            context = {
                "mode": "full",
                "model_key": "gemma4",
                "workdir": root,
                "frame": pd.DataFrame({"title_zh": titles}),
                "lock": {
                    "config_id": "cfg",
                    "models": {"gemma4": {"revision": "api-rev", "model_id": "google/gemma-4-26B-A4B-it"}},
                },
                "state": {
                    "config_id": "cfg",
                    "models": {
                        "gemma4": {
                            "stages": {
                                "validate": {"passed": True, "config_id": "cfg"},
                                "full": {
                                    "status": "stopped_quality_gate",
                                    "failed_window": {
                                        "start": 0,
                                        "end": 1,
                                        "incomplete_rate": 0.0,
                                        "technical_qa_failure_rate": 1.0,
                                    },
                                },
                            }
                        },
                        "qwen35": {"stages": {}},
                    },
                },
                "samples": {},
            }
            cache = hybrid.Cache(root / "cache_gemma4.sqlite3", "gemma4", "api-rev", "cfg")

            def generate(batch, run_context, api_key, min_interval, *, batch_size=1):
                for title in batch:
                    cache.put(title, {"output_text": "Ứng viên cần rà soát", "status": "ok", "truncated": False})
                return {"status": "finished_current_batch", "completed": len(batch)}

            output_path = root / "bilingual_zh_vi_cado_title_gemma4.parquet"
            output_path.write_bytes(b"test-output")
            with (
                patch.object(hybrid, "_check_or_write_runtime_lock"),
                patch.object(hybrid, "_cache", return_value=cache),
                patch.object(hybrid, "_run_gemma", side_effect=generate),
                patch.object(hybrid, "_full_window_gates", side_effect=AssertionError("QA gate should be deferred")),
                patch.object(hybrid, "export_model_parquet", return_value=output_path),
            ):
                result = hybrid.run_stage(
                    context,
                    google_api_key="test-key",
                    full_titles_per_session=10,
                    defer_full_quality_review=True,
                )

        self.assertEqual(result["status"], "all_titles_complete")
        self.assertTrue(result["quality_review_deferred"])
        self.assertEqual(result["window_gates"], [])
        self.assertEqual(result["completed_unique_titles"], 1)
        self.assertEqual(result["remaining_unique_titles"], 0)

    def test_qwen_full_can_use_larger_runtime_batch_without_invalidating_cache_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            titles = ["儿童外套"]
            context = {
                "mode": "full",
                "model_key": "qwen35",
                "workdir": root,
                "frame": pd.DataFrame({"title_zh": titles}),
                "lock": {
                    "config_id": "cfg",
                    "models": {"qwen35": {"revision": "hub-sha", "model_id": "Qwen/Qwen3.5-9B"}},
                },
                "state": {
                    "config_id": "cfg",
                    "models": {"qwen35": {"stages": {}}, "gemma4": {"stages": {}}},
                },
                "samples": {},
                "qwen_batch_size": 4,
            }
            cache = hybrid.Cache(root / "cache_qwen35.sqlite3", "qwen35", "hub-sha", "cfg")

            def generate(batch, run_context, hf_token, batch_size=1):
                self.assertEqual(batch_size, 8)
                for title in batch:
                    cache.put(title, {"output_text": "Áo khoác trẻ em", "status": "ok", "truncated": False})
                return {"status": "finished_current_batch", "completed": len(batch), "batch_size_used": batch_size}

            output_path = root / "bilingual_zh_vi_cado_title_qwen35.parquet"
            output_path.write_bytes(b"test-output")
            with (
                patch.object(hybrid, "_check_or_write_runtime_lock"),
                patch.object(hybrid, "_cache", return_value=cache),
                patch.object(hybrid, "_run_qwen", side_effect=generate) as runner,
                patch.object(hybrid, "_full_window_gates", side_effect=AssertionError("QA gate should be deferred")),
                patch.object(hybrid, "export_model_parquet", return_value=output_path),
            ):
                result = hybrid.run_stage(
                    context,
                    batch_size=8,
                    full_titles_per_session=10,
                    allow_unvalidated_full=True,
                    defer_full_quality_review=True,
                    allow_qwen_batch_change=True,
                )

        self.assertEqual(runner.call_args.kwargs["batch_size"], 8)
        self.assertEqual(result["generation"]["batch_size_locked_in_config"], 4)
        self.assertEqual(result["generation"]["batch_size_requested"], 8)
        self.assertEqual(result["status"], "all_titles_complete")

    def test_qwen_runtime_lock_can_record_gpu_change_but_not_package_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = {"workdir": root, "mode": "smoke"}
            original = {
                "python": "3.12",
                "packages": {"transformers": "5.10.1"},
                "device": {
                    "cuda_available": True,
                    "cuda_version": "12.8",
                    "gpus": [{"name": "T4", "memory_bytes": 16}],
                },
            }
            reassigned = {
                **original,
                "device": {
                    "cuda_available": True,
                    "cuda_version": "12.8",
                    "gpus": [{"name": "L4", "memory_bytes": 24}],
                },
            }
            lock_path = root / "runtime_qwen35.json"
            lock_path.write_text(json.dumps({"signature": original}), encoding="utf-8")
            with patch.object(hybrid, "_runtime_signature", return_value=reassigned):
                lock = hybrid._check_or_write_runtime_lock(
                    context, "qwen35", allow_qwen_hardware_change=True
                )
            self.assertEqual(lock["signature"], reassigned)
            self.assertEqual(len(lock["gpu_hardware_history"]), 2)

            changed_packages = {**reassigned, "packages": {"transformers": "5.11.0"}}
            with patch.object(hybrid, "_runtime_signature", return_value=changed_packages):
                with self.assertRaisesRegex(RuntimeError, "Runtime changed"):
                    hybrid._check_or_write_runtime_lock(
                        context, "qwen35", allow_qwen_hardware_change=True
                    )

    def test_full_window_gate_stops_above_failure_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = hybrid.Cache(Path(directory) / "qwen.sqlite3", "qwen35", "sha", "cfg")
            titles = [f"商品 {index}" for index in range(20)]
            for index, title in enumerate(titles):
                if index == 0:
                    cache.put(title, {"output_text": "", "status": "error"})
                else:
                    cache.put(title, {"output_text": f"Sản phẩm {index}", "status": "ok", "truncated": False})
            gate = hybrid._full_window_gates(titles, titles, cache)
        self.assertEqual(len(gate), 1)
        self.assertFalse(gate[0]["passed"])
        self.assertAlmostEqual(gate[0]["incomplete_rate"], 0.05)

    def test_full_window_gate_allows_exact_one_percent_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = hybrid.Cache(Path(directory) / "qwen.sqlite3", "qwen35", "sha", "cfg")
            titles = [f"商品 {index}" for index in range(200)]
            for index, title in enumerate(titles):
                if index < 2:
                    cache.put(title, {"output_text": "", "status": "error"})
                else:
                    cache.put(title, {"output_text": f"Sản phẩm {index}", "status": "ok", "truncated": False})
            gate = hybrid._full_window_gates(titles, titles, cache)
        self.assertTrue(gate[0]["passed"])
        self.assertAlmostEqual(gate[0]["incomplete_rate"], 0.01)

    def test_only_incomplete_full_gate_stops_are_recoverable(self):
        self.assertTrue(
            hybrid._full_gate_stop_is_incomplete_only(
                {
                    "status": "stopped_quality_gate",
                    "failed_window": {"incomplete_rate": 0.015, "technical_qa_failure_rate": 0.02},
                }
            )
        )
        self.assertFalse(
            hybrid._full_gate_stop_is_incomplete_only(
                {
                    "status": "stopped_quality_gate",
                    "failed_window": {"incomplete_rate": 0.0, "technical_qa_failure_rate": 0.06},
                }
            )
        )
        self.assertFalse(
            hybrid._full_gate_stop_is_incomplete_only(
                {
                    "status": "all_titles_complete",
                    "failed_window": {"incomplete_rate": 0.02, "technical_qa_failure_rate": 0.0},
                }
            )
        )

    def test_full_resume_allows_only_incomplete_provider_failures(self):
        context = {
            "lock": {"config_id": "cfg"},
            "state": {
                "config_id": "cfg",
                "models": {
                    "gemma4": {
                        "stages": {
                            "validate": {"passed": True, "config_id": "cfg"},
                            "full": {
                                "status": "stopped_quality_gate",
                                "failed_window": {
                                    "incomplete_rate": 0.015,
                                    "technical_qa_failure_rate": 0.02,
                                },
                            },
                        }
                    }
                },
            },
        }
        hybrid._check_prerequisites(context, "gemma4", "full")
        context["state"]["models"]["gemma4"]["stages"]["full"]["failed_window"][
            "technical_qa_failure_rate"
        ] = 0.06
        with self.assertRaisesRegex(RuntimeError, "previously stopped"):
            hybrid._check_prerequisites(context, "gemma4", "full")


class HybridOutputTests(unittest.TestCase):
    @staticmethod
    def source_frame():
        rows = []
        for product_id, title in ((1, "儿童外套"), (2, "儿童外套"), (3, "女装")):
            rows.append({column: None for column in quality.ORIGINAL_COLUMNS})
            rows[-1]["product_id"] = product_id
            rows[-1]["title_zh"] = title
            rows[-1]["title_vi"] = ""
            rows[-1]["title_vi_cleaned"] = ""
            rows[-1]["category"] = "test"
        return pd.DataFrame(rows)

    def test_qwen_eos_ids_include_all_model_stop_tokens(self):
        tokenizer = SimpleNamespace(
            eos_token_id=151645,
            unk_token_id=0,
            convert_tokens_to_ids=lambda token: 151645 if token == "<|im_end|>" else 0,
        )
        model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[151645, 151643]),
            config=SimpleNamespace(eos_token_id=151645),
        )

        self.assertEqual(hybrid._qwen_eos_token_ids(model, tokenizer), [151643, 151645])

    def test_qwen_generate_stops_at_secondary_configured_eos(self):
        class FakeScalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FakeMaskRow:
            def sum(self):
                return FakeScalar(2)

        class FakeTensor:
            shape = (1, 2)

            def to(self, device):
                return self

            def __getitem__(self, index):
                return FakeMaskRow()

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeSequence:
            def __init__(self, values):
                self.values = values

            def __getitem__(self, key):
                return FakeSequence(self.values[key])

            def tolist(self):
                return self.values

        class FakeTokenizer:
            eos_token_id = 151645
            pad_token_id = 151643
            unk_token_id = 0

            def __init__(self):
                self.decoded_ids = None

            def apply_chat_template(self, *args, **kwargs):
                return FakeBatch(input_ids=FakeTensor(), attention_mask=FakeTensor())

            def convert_tokens_to_ids(self, token):
                return {"<|im_end|>": 151645}.get(token, 0)

            def decode(self, token_ids, **kwargs):
                self.decoded_ids = token_ids
                return "Túi xách"

        class FakeModel:
            generation_config = SimpleNamespace(eos_token_id=[151645, 151643])
            config = SimpleNamespace(eos_token_id=151645)

            def __init__(self):
                self.generate_kwargs = None

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs
                return [FakeSequence([11, 12, 7, 151643, 99])]

        tokenizer = FakeTokenizer()
        model = FakeModel()
        fake_torch = SimpleNamespace(inference_mode=lambda: __import__("contextlib").nullcontext())
        with patch.dict("sys.modules", {"torch": fake_torch}):
            generated = hybrid._qwen_generate(model, tokenizer, ["女包"], "cuda:0", 256)

        self.assertEqual(model.generate_kwargs["eos_token_id"], [151643, 151645])
        self.assertEqual(tokenizer.decoded_ids, [7, 151643])
        self.assertFalse(generated[0]["truncated"])
        self.assertEqual(generated[0]["output_tokens"], 2)

    def test_repetitive_cached_translation_is_not_complete(self):
        repeated = "túi xách thời trang, " * 4
        record = {"status": "ok", "truncated": False, "output_text": repeated.strip()}

        self.assertTrue(quality._repeated_phrase(record["output_text"]))
        self.assertFalse(hybrid._complete(record))

    def test_qwen_runner_marks_repetitive_output_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            context = {
                "workdir": Path(directory),
                "lock": {
                    "config_id": "cfg",
                    "models": {"qwen35": {"revision": "hub-sha"}},
                },
            }
            generated = [{
                "output_text": "túi xách thời trang, " * 4,
                "input_tokens": 20,
                "output_tokens": 40,
                "truncated": False,
                "repetition_detected": True,
            }]
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
            with (
                patch.object(hybrid, "_load_qwen", return_value=(object(), object())),
                patch.object(hybrid, "_qwen_generate", return_value=generated),
                patch.dict("sys.modules", {"torch": fake_torch}),
            ):
                result = hybrid._run_qwen(["女包"], context, None, batch_size=1)

            cached = hybrid.Cache(Path(directory) / "cache_qwen35.sqlite3", "qwen35", "hub-sha", "cfg").get_many(["女包"])["女包"]

        self.assertEqual(result["completed"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(cached["status"], "error")
        self.assertEqual(cached["error_text"], "repeated_phrase_guard")

    def test_export_preserves_rows_order_source_columns_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = self.source_frame()
            lock = {
                "config_id": "cfg",
                "models": {"qwen35": {"revision": "sha"}},
            }
            context = {"workdir": Path(directory), "frame": frame, "lock": lock}
            cache = hybrid.Cache(Path(directory) / "cache_qwen35.sqlite3", "qwen35", "sha", "cfg")
            cache.put("儿童外套", {"output_text": "Áo khoác trẻ em", "status": "ok", "truncated": False})
            cache.put("女装", {"output_text": "Thời trang nữ", "status": "ok", "truncated": False})
            path = hybrid.export_model_parquet(context, "qwen35")
            output = pd.read_parquet(path)
        self.assertEqual(path.name, "bilingual_zh_vi_cado_title_qwen35.parquet")
        self.assertEqual(output["product_id"].tolist(), [1, 2, 3])
        self.assertEqual(output["title_vi_candidate_qwen35"].tolist(), ["Áo khoác trẻ em", "Áo khoác trẻ em", "Thời trang nữ"])
        for column in quality.ORIGINAL_COLUMNS:
            self.assertTrue(quality._series_values_equal(frame[column], output[column]), column)
        self.assertTrue((output["title_vi_qwen35_semantic_verified"] == False).all())

    def test_qwen_local_runner_records_token_usage_without_inference_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = {
                "workdir": root,
                "lock": {
                    "config_id": "cfg",
                    "models": {"qwen35": {"revision": "hub-sha"}},
                },
            }
            generated = [{
                "output_text": "Áo khoác trẻ em",
                "input_tokens": 20,
                "output_tokens": 7,
                "truncated": False,
            }]
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
            with (
                patch.object(hybrid, "_load_qwen", return_value=(object(), object())) as load,
                patch.object(hybrid, "_qwen_generate", return_value=generated),
                patch.dict("sys.modules", {"torch": fake_torch}),
            ):
                result = hybrid._run_qwen(["儿童外套"], context, None, batch_size=1)
            self.assertEqual(result["status"], "finished_current_batch")
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["input_tokens_this_session"], 20)
            self.assertEqual(result["output_tokens_this_session"], 7)
            load.assert_called_once_with("hub-sha", None)

    def test_gemma_google_api_runner_records_usage_and_retries_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            context = {
                "workdir": Path(directory),
                "lock": {
                    "config_id": "cfg",
                    "models": {"gemma4": {"revision": "hub-sha"}},
                },
            }
            responses = [
                {"output_text": "Áo khoác", "input_tokens": 21, "output_tokens": 256, "truncated": True},
                {"output_text": "Áo khoác trẻ em", "input_tokens": 22, "output_tokens": 8, "truncated": False},
            ]
            fake_genai = SimpleNamespace(
                Client=lambda **kwargs: SimpleNamespace(close=lambda: None),
                types=SimpleNamespace(
                    HttpRetryOptions=lambda **kwargs: kwargs,
                    HttpOptions=lambda **kwargs: kwargs,
                ),
            )
            with (
                patch.dict("sys.modules", {"google": SimpleNamespace(genai=fake_genai)}),
                patch.object(
                    hybrid, "_gemma_response", side_effect=lambda *args, **kwargs: responses.pop(0)
                ),
            ):
                result = hybrid._run_gemma(
                    ["\u513f\u7ae5\u5916\u5957"], context, "google-api-key", 0
                )
            self.assertEqual(result["status"], "finished_current_batch")
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["input_tokens_this_session"], 43)
            self.assertEqual(result["output_tokens_this_session"], 264)
            self.assertEqual(result["provider"], "google_ai_studio")
            cached = hybrid.Cache(
                Path(directory) / "cache_gemma4.sqlite3", "gemma4", "hub-sha", "cfg"
            ).get_many(["\u513f\u7ae5\u5916\u5957"])["\u513f\u7ae5\u5916\u5957"]
            self.assertEqual(cached["attempts"], 2)
            self.assertEqual(cached["input_tokens"], 43)
            self.assertEqual(cached["output_tokens"], 264)

    def test_gemma_paired_runner_attributes_one_request_across_two_titles(self):
        with tempfile.TemporaryDirectory() as directory:
            context = {
                "workdir": Path(directory),
                "lock": {"config_id": "cfg", "models": {"gemma4": {"revision": "hub-sha"}}},
            }
            client = SimpleNamespace(close=lambda: None)
            fake_genai = SimpleNamespace(
                Client=lambda **kwargs: client,
                types=SimpleNamespace(
                    HttpRetryOptions=lambda **kwargs: kwargs,
                    HttpOptions=lambda **kwargs: kwargs,
                ),
            )

            def paired_call(*args, batch_titles=None, **kwargs):
                self.assertEqual(batch_titles, ["儿童外套", "男士外套"])
                return {
                    "items": [
                        {"output_text": "Áo khoác trẻ em", "truncated": False, "input_tokens": 20, "output_tokens": 6},
                        {"output_text": "Áo khoác nam", "truncated": False, "input_tokens": 10, "output_tokens": 4},
                    ],
                    "input_tokens": 30,
                    "output_tokens": 10,
                    "truncated": False,
                    "api_attempts": 1,
                }

            with (
                patch.dict("sys.modules", {"google": SimpleNamespace(genai=fake_genai)}),
                patch.object(hybrid, "_call_gemma_with_retries", side_effect=paired_call),
            ):
                result = hybrid._run_gemma(
                    ["儿童外套", "男士外套"], context, "google-api-key", 0, batch_size=2
                )

            cache = hybrid.Cache(Path(directory) / "cache_gemma4.sqlite3", "gemma4", "hub-sha", "cfg")
            rows = cache.get_many(["儿童外套", "男士外套"])
            self.assertEqual(result["api_attempts_this_session"], 1)
            self.assertEqual(result["completed"], 2)
            self.assertEqual(sum(row["api_requests"] for row in rows.values()), 1.0)
            self.assertEqual([rows[title]["api_requests"] for title in ("儿童外套", "男士外套")], [0.5, 0.5])

    def test_gemma_bad_paired_format_falls_back_to_single_titles_without_double_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            context = {
                "workdir": Path(directory),
                "lock": {"config_id": "cfg", "models": {"gemma4": {"revision": "hub-sha"}}},
            }
            client = SimpleNamespace(close=lambda: None)
            fake_genai = SimpleNamespace(
                Client=lambda **kwargs: client,
                types=SimpleNamespace(
                    HttpRetryOptions=lambda **kwargs: kwargs,
                    HttpOptions=lambda **kwargs: kwargs,
                ),
            )
            calls = 0

            def mixed_call(*args, batch_titles=None, **kwargs):
                nonlocal calls
                calls += 1
                if batch_titles is not None:
                    return {
                        "items": None,
                        "parse_error": "missing item id",
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "api_attempts": 1,
                    }
                title = args[1]
                return {
                    "output_text": "Áo khoác trẻ em" if title == "儿童外套" else "Áo khoác nam",
                    "truncated": False,
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "api_attempts": 1,
                }

            with (
                patch.dict("sys.modules", {"google": SimpleNamespace(genai=fake_genai)}),
                patch.object(hybrid, "_call_gemma_with_retries", side_effect=mixed_call),
            ):
                result = hybrid._run_gemma(
                    ["儿童外套", "男士外套"], context, "google-api-key", 0, batch_size=2
                )

            cache = hybrid.Cache(Path(directory) / "cache_gemma4.sqlite3", "gemma4", "hub-sha", "cfg")
            rows = cache.get_many(["儿童外套", "男士外套"])
            self.assertEqual(calls, 3)
            self.assertEqual(result["api_attempts_this_session"], 3)
            self.assertEqual(result["batch_fallbacks_this_session"], 1)
            self.assertEqual(sum(row["api_requests"] for row in rows.values()), 3.0)
            self.assertEqual([rows[title]["api_requests"] for title in ("儿童外套", "男士外套")], [1.5, 1.5])
            self.assertEqual(result["input_tokens_this_session"], 20)
            self.assertEqual(result["output_tokens_this_session"], 10)

    def test_upload_is_blocked_until_both_models_are_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hybrid_environment_lock.json").write_text(json.dumps({"config_id": "cfg"}))
            (root / "hybrid_run_state.json").write_text(
                json.dumps({"models": {"gemma4": {"stages": {}}, "qwen35": {"stages": {}}}})
            )
            with self.assertRaisesRegex(RuntimeError, "both model full runs"):
                hybrid.upload_outputs_to_hf(root, hf_token="dummy")

    def test_upload_uses_one_commit_and_verifies_remote_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source_frame()
            source_path = root / "source.parquet"
            source.to_parquet(source_path, index=False)
            source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            lock = {
                "config_id": "cfg",
                "models": {
                    "gemma4": {"revision": "provider_managed"},
                    "qwen35": {"revision": "qwen-sha"},
                },
            }
            for key in hybrid.MODELS:
                cache = hybrid.Cache(root / f"cache_{key}.sqlite3", key, lock["models"][key]["revision"], "cfg")
                cache.put("儿童外套", {"output_text": f"{key} áo khoác", "status": "ok", "truncated": False})
                cache.put("女装", {"output_text": f"{key} thời trang nữ", "status": "ok", "truncated": False})
            context = {"workdir": root, "frame": source, "lock": lock}
            for key in hybrid.MODELS:
                hybrid.export_model_parquet(context, key)
            state = {
                "models": {
                    key: {"stages": {"full": {"status": "all_titles_complete"}}}
                    for key in hybrid.MODELS
                }
            }
            hybrid._export_comparison_if_ready({**context, "state": state})
            for key in hybrid.MODELS:
                output_path = root / f"bilingual_zh_vi_cado_title_{key}.parquet"
                state["models"][key]["stages"]["full"]["output_sha256"] = hashlib.sha256(
                    output_path.read_bytes()
                ).hexdigest()
            (root / "hybrid_environment_lock.json").write_text(json.dumps(lock), encoding="utf-8")
            (root / "hybrid_run_state.json").write_text(json.dumps(state), encoding="utf-8")

            captured = {}

            class FakeApi:
                def repo_info(self, **kwargs):
                    captured["repo"] = kwargs

                def create_commit(self, **kwargs):
                    captured["operations"] = kwargs["operations"]
                    return SimpleNamespace(oid="commit-sha", commit_url="https://hf.test/commit/commit-sha")

                def file_exists(self, **kwargs):
                    return True

            class FakeAdd:
                def __init__(self, **kwargs):
                    self.path_in_repo = kwargs["path_in_repo"]
                    self.path_or_fileobj = kwargs["path_or_fileobj"]

            def fake_download(**kwargs):
                return str(root / Path(kwargs["filename"]).name)

            with (
                patch.object(hybrid, "EXPECTED_ROWS", len(source)),
                patch.object(hybrid, "DATASET_SHA256", source_hash),
                patch.object(quality, "validate_input_frame", lambda frame: None),
                patch("huggingface_hub.HfApi", return_value=FakeApi()),
                patch("huggingface_hub.CommitOperationAdd", FakeAdd),
                patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
            ):
                result = hybrid.upload_outputs_to_hf(root, hf_token="dummy", source_path=source_path)
            manifest_exists = (root / "title_translation_upload_manifest.json").exists()

        self.assertTrue(result["readback_verified"])
        self.assertEqual(result["commit_oid"], "commit-sha")
        self.assertEqual(len(captured["operations"]), 2)
        self.assertTrue(manifest_exists)


if __name__ == "__main__":
    unittest.main()
