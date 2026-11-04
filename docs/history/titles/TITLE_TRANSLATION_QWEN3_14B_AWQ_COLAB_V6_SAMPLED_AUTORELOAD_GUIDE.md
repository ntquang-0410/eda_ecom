# Qwen3-14B-AWQ v6: sampled decoding with Drive checkpoint resume

The v6 notebook keeps the existing `title_translation_cado_qwen3_14b_awq_full_v2` directory untouched and writes to a new `title_translation_cado_qwen3_14b_awq_full_v3_sampled` directory. It uses the same pinned Qwen3-14B-AWQ weights and dataset, with the model-card non-thinking sampling recommendations: temperature 0.7, top-p 0.8, top-k 20. A mild repetition penalty (1.05) is added to reduce loops. Greedy decoding was removed because Qwen's official model guidance warns it may produce endless repetition.

## Resume without overwriting the old checkpoint

- On first run, v6 reads the v2 cache in read-only mode and pins the v2 model revision.
- It imports only completed, non-truncated, non-repetitive outputs into its new cache. Incomplete or repetitive rows are intentionally retried.
- v2 files are not modified. The new cache import is recorded in `legacy_cache_import.json` and can be repeated safely if interrupted.
- Do not run v2 and v6 at the same time against the same source cache.

## Run and observe

1. Open `title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload.ipynb` in Colab, choose a CUDA GPU, and select **Runtime → Run all**. Authorize the Drive mount.
2. Keep the notebook, bundle, `qwen_control.json`, checkpoint and status in `MyDrive/title_translation_cado_qwen3_14b_awq_full_v3_sampled`. The notebook loads the bundle from this Drive folder at startup, so it does not need to be uploaded into Colab's temporary storage each session.
3. The run writes `qwen_live_status.json` and checkpoints in the v6 folder. Status contains progress, throughput, error categories and exception class only; it never writes source titles, generated translations, or raw tracebacks. A heartbeat updates status while the model weights load.
4. Drive control accepts `run`, `pause`, `stop`, and `reload`. To apply a reviewed fix while Colab is running, upload/replace the bundle first, then set `qwen_control.json` to `{"action":"reload","bundle_sha256":"<SHA-256 of uploaded bundle>"}`. At a batch boundary, Colab saves its checkpoint, verifies the hash, reloads the approved modules, and resumes. The hash-bound control file is not an arbitrary code executor.
5. Drive cannot start or keep a Colab VM alive. If Colab disconnects or terminates, reconnect and select **Runtime → Run all** to resume. The background Codex monitor can inspect the Drive status and, after tests pass, publish a reviewed bundle and reload command only while the runtime remains alive.
6. The model still requires CUDA. The local laptop's 4-GB GPU cannot host this 14B AWQ model. This version does not call a paid inference API and does not upload output to Hugging Face.

The change addresses the observed repeated-phrase failures at the decoding layer. It still requires the smoke gate before full; technical QA does not verify semantic accuracy, and `semantic_verified` remains false.
