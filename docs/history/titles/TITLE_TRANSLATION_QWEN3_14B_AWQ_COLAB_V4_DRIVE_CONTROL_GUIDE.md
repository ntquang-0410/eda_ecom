# Qwen3-14B-AWQ title translation on Google Colab

This is a separate run configuration for `Qwen/Qwen3-14B-AWQ`. It uses the repository's official AWQ 4-bit configuration through Transformers `AutoModelForCausalLM`; it does not use bitsandbytes, Qwen3.5-specific FLA kernels, Gemma, or an API.

## Before switching

- The v4 notebook reuses the existing Qwen3-14B-AWQ folder, cache, run state, source Parquet, and model lock. It does not overwrite the SQLite checkpoint.
- If another full cell is currently running, let it reach a checkpoint or stop it cleanly before starting v4; do not run two workers against the same SQLite file at once.

## Run

1. Put `title_translation_qwen3_14b_awq_colab_v4_drive_control.ipynb` in Google Drive and open it with Colab. Put `title_translation_qwen3_14b_awq_colab_v4_drive_control_bundle.zip` in `MyDrive/title_translation_cado_qwen3_14b_awq_full_v2`; the notebook reads it there, so it is not uploaded on every session. The control file is created in the same folder on first run.
2. Select a CUDA GPU runtime and run all cells. It installs/checks `gptqmodel`, verifies the pinned source Parquet and resumes the existing cache. Smoke must have at least 99% complete outputs before full.
3. The package lock compares the PyTorch core version while ignoring the `+cpu`/`+cuXXX` build suffix, and pins the currently installed Torch build during dependency installation. This lets an old CPU-tagged lock resume on a CUDA Colab runtime without replacing its GPU-enabled Torch.
4. Smoke keeps the locked batch size 1. Full starts at batch size 4 and automatically backs off to 2 then 1 if generation runs out of VRAM. A T4 16 GB is close to the model's memory limit; after a model-loading OOM, restart the Colab runtime before resuming so the smoke model is fully cleared. Do not change quantization or the checkpoint's locked smoke configuration.
5. Checkpoints are mirrored every 100 processed titles, and immediately on `pause`/`stop`. `qwen_live_status.json` records progress/rate, recent repeated/truncated outputs, and errors from any notebook cell (traceback is recorded with common token patterns redacted). The `qwen_control.json` allowlist is only `run`, `pause`, or `stop`; `pause` waits between batches, `stop` saves and exits. No Python from Drive is executed.
6. The notebook exports a separate Parquet locally on Drive and does not upload to Hugging Face.

## Observe/edit loop

- I can read `qwen_live_status.json` and the cache/report files directly through the connected Drive tools when you ask me to inspect a run.
- I can issue `pause`/`run`/`stop` by updating the control JSON in Drive. The running kernel checks it between generation batches (not mid-kernel call).
- I can create and upload a reviewed replacement bundle to Drive. Python modules are loaded once per notebook run, so code changes take effect after a clean stop/checkpoint and re-running the notebook's code-load cell (or Run all). A live batch is never hot-patched.
- This provides Drive-mediated observation/control, not SSH or direct access to Colab's process/GPU/memory. Colab must remain connected and its Drive mount healthy.

The 4-bit AWQ model is larger than the previous 9B model and is **not a speed upgrade**. Expected throughput must be measured in the 20-title preflight on the actual Colab GPU before deciding whether to run all 15,863 unique titles. `semantic_verified` remains false until human review.
