# Graph Report - eda_ecom  (2026-11-24)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1790 nodes · 4372 edges · 113 communities (85 shown, 28 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 107 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c463e695`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 86
- Community 100
- Community 101
- Community 102
- Community 103
- Community 104
- Community 106
- Community 107
- Community 108
- Community 109
- Community 110
- Community 111
- Community 112

## God Nodes (most connected - your core abstractions)
1. `normalize_text()` - 55 edges
2. `resolve_path()` - 44 edges
3. `extract_human_value_pairs()` - 34 edges
4. `run_cado()` - 31 edges
5. `run_tiki_eda()` - 30 edges
6. `build_tiki_units()` - 28 edges
7. `normalize_key()` - 27 edges
8. `BitextConfig` - 25 edges
9. `Whitelist` - 25 edges
10. `extract_quantities()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `command_stats()` --uses--> `Cedict`  [INFERRED]
  run_pipeline.py → cado_core/dictionary.py
- `command_stats()` --uses--> `Whitelist`  [INFERRED]
  run_pipeline.py → cado_core/dictionary.py
- `TitleSmoothingTest` --uses--> `LengthPriorV2Config`  [INFERRED]
  tests/test_length_prior_v3.py → cado_core/length_prior_v2.py
- `ExtractV2Test` --uses--> `LengthPriorV2Config`  [INFERRED]
  tests/test_length_prior_v2.py → cado_core/length_prior_v2.py
- `q_items()` --uses--> `Quantity`  [INFERRED]
  tools/contextual_quantity_decision.py → cado_core/quantity_norm.py

## Import Cycles
- None detected.

## Communities (113 total, 28 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (53): apply_length_prior(), _as_list(), _attribute_text(), _describe(), extract_human_length_pairs(), add_pair(), fit_length_prior(), add_group() (+45 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (39): _add_flags(), BitextConfig, BitextResult, classify(), compute_cosine(), compute_entity_signals(), compute_hybrid(), compute_lexical() (+31 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (39): HF_HOME, HF_HUB_OFFLINE, PYTORCH_CUDA_ALLOC_CONF, run_job(), run_queue.sh script, TOKENIZERS_PARALLELISM, TRANSFORMERS_OFFLINE, main() (+31 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (33): load_dotenv(), main(), _alarm(), kernel_status(), load_state(), main(), save_state(), _Timeout (+25 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (32): cluster_boot(), load_scores(), main(), fmt(), load_gemma(), load_scores(), main(), mean_table() (+24 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (33): copy_file(), kaggle_username(), load_dotenv(), main(), sha256(), main(), sha256(), main() (+25 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (31): normalize_brackets(), add_features(), brand_retention(), build_frame(), cjk_count(), clean_text(), compact_len(), cross_model() (+23 more)

### Community 7 - "Community 7"
Cohesion: 0.05
Nodes (39): 1688_description, 1688_title, tiki_description, tiki_title, A_natural, B_balanced, inference, batch_size (+31 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (39): 1688_description, 1688_title, tiki_description, tiki_title, A_natural, B_balanced, inference, batch_size (+31 more)

### Community 9 - "Community 9"
Cohesion: 0.10
Nodes (24): chunk_pair(), compact_len(), main(), pack(), process(), sentence_weight(), slice_proportional(), split_long_sentence() (+16 more)

### Community 10 - "Community 10"
Cohesion: 0.06
Nodes (9): build(), build(), make(), build(), build(), build(), main(), build() (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (16): build_product_state(), build_tiki_key_map(), build_translation_cache(), _cache_key(), export_plan_a(), KeyMapResult, _load_existing_manual(), _normalize_keys() (+8 more)

### Community 12 - "Community 12"
Cohesion: 0.05
Nodes (36): accum_count, adam_beta2, attention_dropout, batch_size, batch_type, bucket_size, path_src, path_tgt (+28 more)

### Community 13 - "Community 13"
Cohesion: 0.05
Nodes (36): accum_count, adam_beta2, attention_dropout, batch_size, batch_type, bucket_size, path_src, path_tgt (+28 more)

### Community 14 - "Community 14"
Cohesion: 0.05
Nodes (36): accum_count, adam_beta2, attention_dropout, batch_size, batch_type, bucket_size, path_src, path_tgt (+28 more)

### Community 15 - "Community 15"
Cohesion: 0.05
Nodes (36): accum_count, adam_beta2, attention_dropout, batch_size, batch_type, bucket_size, path_src, path_tgt (+28 more)

### Community 16 - "Community 16"
Cohesion: 0.10
Nodes (11): main(), main(), resolve_path(), main(), build_parser(), main(), main(), _validate_product_ids() (+3 more)

### Community 17 - "Community 17"
Cohesion: 0.11
Nodes (18): clean_text(), _as_list(), bt_method_for_class(), build_tiki_units(), classify_key(), _clean_value(), _describe(), extract_attributes() (+10 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (14): CadoConfig, CadoResult, cleaned_column_name(), _log(), run_cado(), _run_tier5(), _tier_of(), classify_occurrence() (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.10
Nodes (11): main(), needs_repair(), is_good_translation(), main(), main(), main(), main(), main() (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.13
Nodes (20): _as_list(), attributes_to_text(), _clean_score(), _clip(), _content_tokens(), _entity_score(), is_vi_token(), latin_metrics() (+12 more)

### Community 21 - "Community 21"
Cohesion: 0.22
Nodes (22): collect_coverage(), delete_kernel(), download_outputs(), ensure_kernel(), find_report(), harvest(), incomplete_models(), is_running() (+14 more)

### Community 22 - "Community 22"
Cohesion: 0.11
Nodes (16): load(), main(), boot(), fmt(), load(), main(), mean(), subset() (+8 more)

### Community 23 - "Community 23"
Cohesion: 0.15
Nodes (16): length_bucket(), measure_text(), apply_length_prior_v2(), _as_list(), _attribute_key(), _attribute_value(), _describe(), _estimated_key_chars() (+8 more)

### Community 24 - "Community 24"
Cohesion: 0.14
Nodes (12): apply_length_prior_v3(), _bucket_order(), _confidence(), evaluate_length_prior_v3(), fit_length_prior_v3(), LengthPriorV3, LengthPriorV3ApplyResult, LengthPriorV3Config (+4 more)

### Community 25 - "Community 25"
Cohesion: 0.11
Nodes (10): extract_human_value_pairs(), add_pair(), HumanValuePairs, measure_zh(), _numeric_overlap(), split_for_product(), split_value_list(), ExtractV2Test (+2 more)

### Community 26 - "Community 26"
Cohesion: 0.12
Nodes (9): find_file(), main(), main(), build_parser(), main(), main(), read_token(), create_dataset_card_content() (+1 more)

### Community 27 - "Community 27"
Cohesion: 0.19
Nodes (13): aggregate_reviews(), apply_reviews(), build_review_input(), build_silver_input(), evaluate_silver(), normalize_key_zh(), _review_messages(), run_reviews() (+5 more)

### Community 28 - "Community 28"
Cohesion: 0.16
Nodes (9): main(), make_eval_1688(), make_eval_tiki(), make_train_view(), target_tokens(), main(), scan_series(), discover() (+1 more)

### Community 29 - "Community 29"
Cohesion: 0.15
Nodes (8): clean_vi_text(), _inside_latin_phrase(), _is_brand_token(), load_vi_map(), run_cado_vi(), ViCleanConfig, ViCleanResult, word_pattern()

### Community 30 - "Community 30"
Cohesion: 0.18
Nodes (9): build_candidates(), pattern_keep(), count_vi_tokens(), extract_latin_words(), has_cjk(), has_latin(), normalize_text(), cache_key() (+1 more)

### Community 31 - "Community 31"
Cohesion: 0.20
Nodes (9): build_key_alignment_map(), _alignment_summary(), _alignment_table(), build_parser(), main(), run_from_args(), _stratified_sample(), _write_phase_evaluation() (+1 more)

### Community 32 - "Community 32"
Cohesion: 0.18
Nodes (7): fit_length_prior_v2(), add_group(), LengthPriorV2Config, FitApplyV2Test, main(), _write_prior(), _write_split_audit()

### Community 33 - "Community 33"
Cohesion: 0.20
Nodes (12): clean_noise_characters(), remove_unmatched_brackets(), as_list(), attr_key(), attr_values(), char_sim(), main(), _clean_attrs() (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.18
Nodes (10): _close(), cn_to_number(), extract_quantities(), _in_code_token(), _match_unit(), _normalize_fractions(), parse_number(), _propagate_range_units() (+2 more)

### Community 35 - "Community 35"
Cohesion: 0.19
Nodes (6): _extract_json(), validate_key_zh(), key_map_frame(), ReviewAggregationTest, units_frame(), ValidationTest

### Community 36 - "Community 36"
Cohesion: 0.26
Nodes (9): Quantity, artifact_reason(), canonical(), classify(), find_quantity(), is_cjk(), lenient_match(), main() (+1 more)

### Community 37 - "Community 37"
Cohesion: 0.17
Nodes (4): extract_protected_spans(), _looks_like_brand(), PipelineTest, TextTest

### Community 38 - "Community 38"
Cohesion: 0.22
Nodes (4): remove_words(), strip_cjk(), _tidy_after_removal(), TextBoundaryTest

### Community 39 - "Community 39"
Cohesion: 0.27
Nodes (5): add(), main(), main(), ngrams(), similarity()

### Community 40 - "Community 40"
Cohesion: 0.25
Nodes (3): ProfileTest, sample_frame(), UnitsTest

### Community 41 - "Community 41"
Cohesion: 0.40
Nodes (6): compact_len(), load_a(), load_b(), main(), normalize(), token_est()

### Community 42 - "Community 42"
Cohesion: 0.33
Nodes (7): build_1688_context(), build_tiki_context(), close(), decide(), find_in(), main(), q_items()

### Community 43 - "Community 43"
Cohesion: 0.42
Nodes (6): compare(), test_chinese_numerals(), test_ideal_models_not_units(), test_quantity_match(), test_quantity_mismatch(), test_weak_quantities_ignored()

### Community 44 - "Community 44"
Cohesion: 0.36
Nodes (6): signature(), as_json(), collect(), main(), naive_digits(), reason_for()

### Community 45 - "Community 45"
Cohesion: 0.22
Nodes (3): classify_value(), _looks_numeric_value(), KeyTest

### Community 46 - "Community 46"
Cohesion: 0.22
Nodes (9): optimizer, betas, grad_clip, lr, name, scheduler, total_steps, warmup_steps (+1 more)

### Community 47 - "Community 47"
Cohesion: 0.22
Nodes (9): optimizer, betas, grad_clip, lr, name, scheduler, total_steps, warmup_steps (+1 more)

### Community 48 - "Community 48"
Cohesion: 0.22
Nodes (9): optimizer, betas, grad_clip, lr, name, scheduler, total_steps, warmup_steps (+1 more)

### Community 49 - "Community 49"
Cohesion: 0.31
Nodes (4): is_good(), looks_degenerate(), main(), recover()

### Community 50 - "Community 50"
Cohesion: 0.42
Nodes (5): api(), discover_chat(), load_config(), main(), send()

### Community 51 - "Community 51"
Cohesion: 0.25
Nodes (4): chunk_sentence(), _protect_sentence_markers(), _restore_sentence_markers(), split_sentences()

### Community 52 - "Community 52"
Cohesion: 0.54
Nodes (5): apply_cap(), group_of(), main(), required_weights(), shares()

### Community 54 - "Community 54"
Cohesion: 0.29
Nodes (7): 1688_description, 1688_title, tiki_description, tiki_title, A_natural, B_balanced, configs

### Community 55 - "Community 55"
Cohesion: 0.29
Nodes (7): lora, alpha, bias, dropout, r, target_modules, task_type

### Community 56 - "Community 56"
Cohesion: 0.29
Nodes (7): training, loss, micro_batch, seed, steps, token_budget, tokens_per_step

### Community 57 - "Community 57"
Cohesion: 0.29
Nodes (7): 1688_description, 1688_title, tiki_description, tiki_title, A_natural, B_balanced, configs

### Community 58 - "Community 58"
Cohesion: 0.29
Nodes (7): lora, alpha, bias, dropout, r, target_modules, task_type

### Community 59 - "Community 59"
Cohesion: 0.29
Nodes (7): training, loss, micro_batch, seed, steps, token_budget, tokens_per_step

### Community 60 - "Community 60"
Cohesion: 0.29
Nodes (7): 1688_description, 1688_title, tiki_description, tiki_title, A_natural, B_balanced, configs

### Community 61 - "Community 61"
Cohesion: 0.29
Nodes (7): lora, alpha, bias, dropout, r, target_modules, task_type

### Community 62 - "Community 62"
Cohesion: 0.29
Nodes (7): training, loss, micro_batch, seed, steps, token_budget, tokens_per_step

### Community 63 - "Community 63"
Cohesion: 0.43
Nodes (4): dataset_of(), load_records(), main(), pair_type_of()

### Community 65 - "Community 65"
Cohesion: 0.33
Nodes (6): model, base, dtype, gradient_checkpointing, max_seq_len, source

### Community 66 - "Community 66"
Cohesion: 0.33
Nodes (6): model, base, dtype, gradient_checkpointing, max_seq_len, source

### Community 67 - "Community 67"
Cohesion: 0.33
Nodes (6): model, base, dtype, gradient_checkpointing, max_seq_len, source

### Community 68 - "Community 68"
Cohesion: 0.47
Nodes (3): audit(), cache_key(), main()

### Community 69 - "Community 69"
Cohesion: 0.53
Nodes (4): build(), code(), _fix(), md()

### Community 70 - "Community 70"
Cohesion: 0.53
Nodes (4): build(), code(), md(), run_cell()

### Community 71 - "Community 71"
Cohesion: 0.53
Nodes (4): main(), ngrams(), similarity(), tier_weight()

### Community 72 - "Community 72"
Cohesion: 0.67
Nodes (4): clean(), main(), tok_vi(), tok_zh()

### Community 74 - "Community 74"
Cohesion: 0.53
Nodes (4): evaluate(), has_quantity(), main(), naive_digits()

### Community 75 - "Community 75"
Cohesion: 0.40
Nodes (4): prompt, chat_template, system, user

### Community 76 - "Community 76"
Cohesion: 0.40
Nodes (5): sampling, group_target, method, per_group_uniform, sample_with_replacement

### Community 77 - "Community 77"
Cohesion: 0.40
Nodes (4): prompt, chat_template, system, user

### Community 78 - "Community 78"
Cohesion: 0.40
Nodes (5): sampling, group_target, method, per_group_uniform, sample_with_replacement

### Community 79 - "Community 79"
Cohesion: 0.40
Nodes (4): prompt, chat_template, system, user

### Community 80 - "Community 80"
Cohesion: 0.40
Nodes (5): sampling, group_target, method, per_group_uniform, sample_with_replacement

### Community 81 - "Community 81"
Cohesion: 0.60
Nodes (3): build(), code(), md()

### Community 82 - "Community 82"
Cohesion: 0.60
Nodes (3): build(), code(), md()

### Community 83 - "Community 83"
Cohesion: 0.70
Nodes (3): compact_len(), main(), tokens_est()

### Community 84 - "Community 84"
Cohesion: 0.50
Nodes (4): inference, batch_size, do_sample, max_new

### Community 85 - "Community 85"
Cohesion: 0.50
Nodes (4): inference, batch_size, do_sample, max_new

### Community 86 - "Community 86"
Cohesion: 0.50
Nodes (4): inference, batch_size, do_sample, max_new

### Community 101 - "Community 101"
Cohesion: 0.50
Nodes (3): run_ab_eval.sh script, USE_FLAX, USE_TF

## Knowledge Gaps
- **7 isolated node(s):** `USE_FLAX`, `USE_TF`, `HF_HOME`, `HF_HUB_OFFLINE`, `PYTORCH_CUDA_ALLOC_CONF` (+2 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 756 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **28 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `normalize_text()` connect `Community 30` to `Community 0`, `Community 1`, `Community 68`, `Community 5`, `Community 38`, `Community 39`, `Community 11`, `Community 17`, `Community 18`, `Community 51`, `Community 20`, `Community 23`, `Community 25`, `Community 27`, `Community 29`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **What connects `USE_FLAX`, `USE_TF`, `HF_HOME` to the rest of the system?**
  _7 weakly-connected nodes found - possible missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05231724796942188 - nodes in this community are weakly interconnected._
- **Why does `build_tiki_units()` connect `Community 17` to `Community 0`, `Community 39`, `Community 40`, `Community 10`, `Community 45`, `Community 16`, `Community 27`, `Community 28`, `Community 30`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.0639269406392694 - nodes in this community are weakly interconnected._
- **Why does `main()` connect `Community 6` to `Community 28`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.059154929577464786 - nodes in this community are weakly interconnected._