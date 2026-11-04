# Qwen3-14B: Codex live supervisor (no local GPU)

This route runs Qwen3-14B through Hugging Face Inference Providers from a local Python process. Codex can keep the process in a persistent terminal session, read its progress and errors, edit the Python pipeline, and let the supervisor resume after validating the changed code. The provider-hosted model is not the local `Qwen/Qwen3-14B-AWQ` checkpoint; it is provider-managed Qwen3-14B.

## Data and billing boundary

- Product titles are sent to the selected hosted inference provider. Do not use this route unless that transfer is acceptable.
- Provider calls are paid. The local budget ledger stops before starting another batch when its estimated cumulative spend reaches `--max-cost-usd`; one already in-flight batch and provider price/usage reporting can cause a small overrun. Set an account-side provider spending limit as well.
- The runner refuses to call a provider unless `--allow-paid-inference` is present. It reads `HF_TOKEN` from the process environment or the Hugging Face CLI credential store. Never paste a token into a notebook cell, command line, chat, or logs; use `hf auth login` interactively or a protected environment secret.
- This setup does not upload results to Hugging Face. It writes the Parquet and reports locally; an optional `--mirror-dir` mirrors state into a Google Drive for Desktop synced folder.

## Start

After explicitly approving paid hosted inference and configuring a fresh token with Inference Providers permission, Codex should launch the supervisor from this project directory:

```powershell
python supervise_title_translation_qwen3_hf_api.py `
  --stage full `
  --workers 8 `
  --max-cost-usd 1.00 `
  --allow-paid-inference
```

`--stage full` runs the 20-title smoke gate first, then proceeds to the full source if smoke completes. A user-chosen `--mirror-dir "<Google Drive for Desktop synced folder>"` may be added if that client is installed and configured. A Drive mirror path is not inferred automatically.

## Observe, fix, resume

1. The supervisor runs Python compilation and the project unit tests before starting a worker.
2. The worker uses parallel requests, saves each completed title into SQLite, and reports rolling progress to `qwen_api_live_status.json`.
3. At checkpoints, the SQLite cache is backed up safely to the optional Drive mirror. Status is mirrored after each batch. A completed Parquet/report is mirrored after export verification.
4. If a batch has unresolved titles, the worker stops at a checkpoint. The supervisor remains alive and waits for the watched code files to change.
5. After Codex updates code, the worker exits at a batch boundary. The supervisor runs compile/tests; if they pass, it starts a new worker and resumes from SQLite. Changes to the system prompt alter its signature so old outputs are not silently reused under a different prompt.
6. If the supervisor is stopped, run the same command again with the same `--workdir` to resume. Preserve the SQLite ledger: it tracks completed work and cumulative estimated provider spend across run configurations.

The supervisor can automate checkpoint/reload/test/restart, but it cannot independently make the model-quality judgment that a particular failure requires a prompt/code change. Codex must remain actively monitoring the live session to diagnose those errors and apply a fix. A provider auth error or budget stop requires user action and is not retried automatically.

## Offline verification

```powershell
python supervise_title_translation_qwen3_hf_api.py --stage smoke --workers 8 --max-cost-usd 1.00 --dry-run
```

`--dry-run` runs preflight only and starts zero provider requests. The standard full suite is `python -m unittest discover -s tests`.
