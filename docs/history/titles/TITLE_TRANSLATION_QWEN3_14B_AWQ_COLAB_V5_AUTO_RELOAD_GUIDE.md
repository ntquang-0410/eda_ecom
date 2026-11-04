# Qwen3-14B-AWQ title translation on Google Colab

This is a separate run configuration for `Qwen/Qwen3-14B-AWQ`. It uses the repository's official AWQ 4-bit configuration through Transformers `AutoModelForCausalLM`; it does not use bitsandbytes, Qwen3.5-specific FLA kernels, Gemma, or an API.

## Before switching

- The v5 notebook reuses the existing Qwen3-14B-AWQ folder, cache, run state, source Parquet, and model lock. It does not overwrite the SQLite checkpoint.
- If another full cell is currently running, let it reach a checkpoint or stop it cleanly before starting v5; do not run two workers against the same SQLite file at once.

## Run

1. Put `title_translation_qwen3_14b_awq_colab_v5_auto_reload.ipynb` in Google Drive and open it with Colab. Put `title_translation_qwen3_14b_awq_colab_v5_auto_reload_bundle.zip` in `MyDrive/title_translation_cado_qwen3_14b_awq_full_v2`; the notebook reads it there, so it is not uploaded on every session. The control file is created in the same folder on first run.
2. Select a CUDA GPU runtime and run all cells. It installs/checks `gptqmodel`, verifies the pinned source Parquet and resumes the existing cache. Smoke must have at least 99% complete outputs before full.
3. The package lock compares the PyTorch core version while ignoring the `+cpu`/`+cuXXX` build suffix, and pins the currently installed Torch build during dependency installation. This lets an old CPU-tagged lock resume on a CUDA Colab runtime without replacing its GPU-enabled Torch.
4. Smoke keeps the locked batch size 1. Full starts at batch size 4 and automatically backs off to 2 then 1 if generation runs out of VRAM. A T4 16 GB is close to the model's memory limit; after a model-loading OOM, restart the Colab runtime before resuming so the smoke model is fully cleared. Do not change quantization or the checkpoint's locked smoke configuration.
5. Checkpoints are mirrored every 100 processed titles, and immediately on `pause`/`stop`/`reload`. `qwen_live_status.json` records progress/rate, recent repeated/truncated outputs, and errors from any notebook cell (traceback is recorded with common token patterns redacted). The `qwen_control.json` allowlist is `run`, `pause`, `stop`, or `reload`; `pause` waits and `stop` saves and exits.
6. The notebook exports a separate Parquet locally on Drive and does not upload to Hugging Face.

## Observe/edit and automatic safe reload

- The run writes progress, throughput, recent generation errors, and cell tracebacks to Drive. I can inspect those files and upload a reviewed replacement bundle through Drive.
- To apply a reviewed fix while the run is active, upload/replace the bundle first, then set `qwen_control.json` to `{"action":"reload","bundle_sha256":"<new ZIP SHA-256>"}`. At the next completed batch the notebook saves its checkpoint, verifies the requested hash, reloads the module, and resumes. It never patches a live generation call; model weights must reload, which costs time.
- The loader accepts only the two expected module filenames and checks bundle size, file names, and SHA-256. The control JSON is an allowlisted command file; it never executes code.
- `run`, `pause`, `stop`, and `reload` are polled between generation batches. If Colab has already terminated or the cell exited, Drive cannot restart its runtime; reconnect and run all to resume. This is Drive-mediated control, not SSH/direct kernel access.
- This AWQ configuration requires CUDA. Running Qwen3-14B-AWQ on CPU is not a practical fallback. For direct agent-triggered notebook runs, connect the Deepnote integration; API access and GPU machine availability depend on its plan and billing.

The 4-bit AWQ model is larger than the previous 9B model and is **not a speed upgrade**. Expected throughput must be measured in the 20-title preflight on the actual Colab GPU before deciding whether to run all 15,863 unique titles. `semantic_verified` remains false until human review.
