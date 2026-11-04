# Resume Qwen3.5 full title run with CUDA kernels and larger batches

This fast notebook resumes the existing Qwen run in `MyDrive/title_translation_cado_qwen35_full_v1`. It keeps the pinned dataset, model revision, prompt, generation settings, quality-review-after-full choice, run folder, and SQLite cache. It changes only the execution environment and runtime batch size.

## What changed

- Migrates stale package locks to Transformers 5.20.2, which includes the nested optional-kernel resolver fix, then installs pinned `flash-linear-attention` / `fla-core` and `causal-conv1d` CUDA packages while constraining the existing PyTorch and Triton versions.
- Verifies both imports and Transformers' own resolution of each fast kernel, and records package versions in `qwen35_fast_kernel_lock.json`. It stops before model loading if it detects a PyTorch fallback risk.
- Requests a larger batch based on GPU VRAM: 8 for <22 GiB, 16 for 22–35 GiB, 32 for 36–71 GiB, and 64 for 72+ GiB. On CUDA OOM, the pipeline halves it automatically until it fits.
- Allows that runtime-only batch override without changing the pinned prompt/model configuration or invalidating prior cache entries.
- Still uses only `cuda:0`. When Colab shows several GPUs, a second GPU is not used by this notebook; prefer the strongest single GPU available.

## Safely switch from the currently running notebook

1. In the old Colab tab, stop the full cell. The cache is mirrored every 100 titles and the worker's `finally` block attempts one last mirror on interruption.
2. Before restarting the runtime, run this in a new cell in the old notebook and wait for it to finish:

   ```python
   persist_qwen_cache(LOCAL_CACHE)
   ```

   This copies the latest local SQLite checkpoint to Drive. If the runtime has already been reset, the latest available Drive checkpoint is used; work after its last sync may need to be repeated.
3. Choose **Runtime → Restart session**. This is required so Transformers loads after the fast kernels are installed.
4. Upload/open `title_translation_qwen35_colab_fast.ipynb`, choose the strongest single GPU available (at least 14 GiB), and select **Runtime → Run all**.
5. When asked, upload `title_translation_qwen35_colab_fast_bundle.zip`. Keep the existing Drive run folder unchanged. Smoke should reuse cached results; full should resume the missing titles.

Do not run the old and fast notebooks simultaneously against the same Drive cache. Keep the API/HF tokens out of the notebook; neither is needed for this run.

## What to verify

- The fast-kernel cell prints resolved implementation names before model loading; it must not stop with an unresolved-kernel error.
- During generation, no `falling back to its reference PyTorch implementation` warnings should appear. If they do, interrupt and preserve the cache rather than letting the full run continue slowly.
- Full progress uses a denominator near `15843` and advances in `[qwen35] uncached ...` messages. The 20 smoke titles are already cached.
- The first output reports `batch_size_requested`; `batch_size_used` shows the actual value after any automatic OOM fallback.
- If kernel setup fails, do not continue with the same runtime lock. Preserve the checkpoint and share the setup error; the original notebook/bundle remain available.

This is an execution-speed profile, not a semantic-quality guarantee. The complete dataset QA remains scheduled after translation, as requested.
