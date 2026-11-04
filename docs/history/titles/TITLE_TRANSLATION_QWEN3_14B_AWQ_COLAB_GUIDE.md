# Qwen3-14B-AWQ title translation on Google Colab

This is a separate run configuration for `Qwen/Qwen3-14B-AWQ`. It uses the repository's official AWQ 4-bit configuration through Transformers `AutoModelForCausalLM`; it does not use bitsandbytes, Qwen3.5-specific FLA kernels, Gemma, or an API.

## Before switching

- Stop the currently running Qwen3.5-9B Colab cell manually. This notebook cannot stop another browser/runtime session.
- Keep the old Qwen3.5 Drive folder and ZIP untouched. Its checkpoints and the 280 completed titles are not reusable as Qwen3-14B-AWQ translations.
- The AWQ notebook uses a separate Drive folder, SQLite cache, run state, output Parquet, and model revision lock.

## Run

1. Upload `title_translation_qwen3_14b_awq_colab_v4_drive_control.ipynb` and `title_translation_qwen3_14b_awq_colab_v4_drive_control_bundle.zip` to Drive. The notebook keeps the ZIP in the MyDrive root and reuses it after runtime resets; replace the Drive ZIP whenever you install an updated bundle.
2. Open the notebook in Colab and select a CUDA GPU runtime. Run all cells. It installs and checks `gptqmodel`, verifies the pinned source Parquet, smoke-tests 20 titles, and proceeds to full only if at least 99% of smoke outputs are complete. Technical QA flags can be reviewed later, but truncated/repetitive outputs block full.
3. The package lock compares the PyTorch core version while ignoring the `+cpu`/`+cuXXX` build suffix, and pins the currently installed Torch build during dependency installation. This lets an old CPU-tagged lock resume on a CUDA Colab runtime without replacing its GPU-enabled Torch.
4. Smoke keeps the locked batch size 1. Full starts at batch size 4 and automatically backs off to 2 then 1 if generation runs out of VRAM. A T4 16 GB is close to the model's memory limit; after a model-loading OOM, restart the Colab runtime before resuming so the smoke model is fully cleared. Do not change quantization or the checkpoint's locked smoke configuration.
5. Generation respects every EOS token in the model's official generation config (including `<|im_end|>` and the additional configured stop token). A repeated-phrase guard prevents looping output from being counted as complete. Checkpoints are mirrored to Drive every 100 titles. After a runtime reset, reconnect the same Drive and run all to resume.
6. The notebook exports a separate Parquet locally on Drive and does not upload to Hugging Face.

The 4-bit AWQ model is larger than the previous 9B model and is **not a speed upgrade**. Expected throughput must be measured in the 20-title preflight on the actual Colab GPU before deciding whether to run all 15,863 unique titles. `semantic_verified` remains false until human review.
