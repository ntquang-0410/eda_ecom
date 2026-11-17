# CHECKPOINT — trạng thái trước khi ngắt GPU (2026-11-17 10:55 UTC)

Mục đích: lưu đủ artefact để có thể ngắt máy A100/Colab bất cứ lúc nào và chạy tiếp ở phiên sau.

---

## 1. Đã hoàn thành (không cần chạy lại)

| Hạng mục | Trạng thái | Nơi lưu |
|---|---|---|
| Qwen3-4B A-natural (seed 42) | hyp + COMET | `.artifacts/hyp/hyp_qwen4b_a_seed42.parquet`, `.artifacts/comet_gpu/scores/` |
| Qwen3-4B A-balanced | hyp + COMET | `hyp_qwen4b_a_balanced_seed42.parquet` |
| Qwen3-4B B-balanced (seed 42) | hyp + COMET | `hyp_qwen4b_b_seed42.parquet` |
| Qwen3-4B B-natural | hyp + COMET | `hyp_qwen4b_b_natural_seed42.parquet` |
| Qwen3-4B B-balanced (seed 123) | hyp + COMET | `hyp_qwen4b_b_balanced_seed123.parquet` |
| NLLB A / B | hyp + COMET | `hyp_nllb_a_seed42.parquet`, `hyp_nllb_b_seed42.parquet` |
| 10 cell matrix gốc | hyp + COMET | `.artifacts/hyp/`, `.artifacts/matrix_out/comet5_v2_common/scores/` |
| Báo cáo 2×2 / seed / NLLB | xong | `reports/CONTROLS_2X2.md` |
| Báo cáo matrix v2 + §10 control | xong | `reports/MATRIX_5X2_EVAL_F_V2.md` |

## 2. Việc DUY NHẤT còn thiếu

**Inference Qwen3-4B A seed 123** (`hyp_qwen4b_a_seed123.parquet`).

- Train đã xong 500 step; adapter đã lưu: `/content/dl/a123/out_A_qwen4b_natural/adapter_natural_seed123` (127 MB)
- Bản backup adapter trên Kaggle: dataset `nguynthanhquang/qwen4b-a-s123-adapter`
- Lần chạy trước bị ngắt ở `k≈106/300` dòng description; **không checkpoint được** vì `out_map` chỉ nằm trong RAM → phải sinh lại từ đầu.
- Script mới `tools/infer_resume.py` ghi JSONL sau **từng batch** và resume được, batch 4 cho description + 16 cho prompt ngắn → dự kiến **~60–75 phút** thay vì 3,5h.

### Lệnh chạy lại

```bash
cd /content/eda_ecom
python tools/infer_resume.py \
  --adapter /content/dl/a123/out_A_qwen4b_natural/adapter_natural_seed123 \
  --eval /content/dl/a123/matrix_data/eval_A.parquet \
  --out .artifacts/hyp/hyp_qwen4b_a_seed123.parquet \
  --state .artifacts/hyp/a123_rows.jsonl \
  --seed 123 --config natural --batch-long 4 --batch-short 16
```

Nếu `/content/dl/a123` không còn (Colab reset), tải lại:

```bash
KAGGLE_API_TOKEN=$(cat /root/.kaggle_01/access_token) \
  kaggle kernels output anhtamxinhdep/tiki-m4b-a-s123 -p /content/dl/a123
# hoặc: kaggle datasets download nguynthanhquang/qwen4b-a-s123-adapter
```

### Sau khi có hypothesis

```bash
USE_TF=0 USE_FLAX=0 /content/cometgpu/bin/python tools/score_comet_local.py \
  --hyp-dir .artifacts/hyp --out .artifacts/comet_gpu/scores \
  --labels qwen4b_a --batch 8 --gpus 1
python tools/analyze_controls.py \
  --scores-dir .artifacts/comet_gpu/scores \
  --eval data/cleaned/eval_F_v2_common.parquet \
  --baseline qwen4b_a@42 --out reports/CONTROLS_2X2.md
```

## 3. Kết quả đã chốt (eval_F_v2_common, 831 dòng)

| Cell | 1688 da | Tiki QE |
|---|---:|---:|
| Qwen3-4B A-natural | **0,8584** | **0,0573** |
| Qwen3-4B A-balanced | 0,8477 | 0,0524 |
| Qwen3-4B B-balanced | 0,8471 | 0,0461 |
| Qwen3-4B B-natural | 0,8436 | 0,0471 |
| Qwen3-8B A / B | 0,8567 / 0,8572 | 0,0521 / 0,0509 |
| EcomGPT-7B A / B | 0,8250 / 0,7634 | 0,0077 / −0,1366 |
| Moses A / B | 0,7862 / 0,8037 | −0,2439 / −0,1593 |
| NLLB A / B | 0,6344 / 0,5517 | −0,5379 / −0,6753 |
| OpenNMT A / B-natural | 0,3488 / 0,3387 | −1,4177 / −1,4074 |

Seed 42 vs 123 (B-balanced): +0,0008 [−0,0030; +0,0048] trên 1688 → không ý nghĩa.

## 4. Tài nguyên

- GPU Kaggle: `anhtamxinhdep` −13,6h, `nadsa091` −10,2h, `nguynthanhquang` +0,9h (đã vượt hạn mức, không chạy thêm được).
- Máy hiện tại: A100-SXM4-40GB (Colab), không giới hạn quota.
- Môi trường chấm COMET GPU: `/content/cometgpu` (torch 2.6+cu124, comet 2.2.7). Nếu Colab reset, dựng lại:

```bash
uv venv --python 3.11 /content/cometgpu
uv pip install --python /content/cometgpu/bin/python torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python /content/cometgpu/bin/python 'numpy<2' pandas 'pyarrow==16.1.0' unbabel-comet sacrebleu 'setuptools<81'
```

## 5. Lưu ý khi resume trên Colab mới

1. `pip uninstall -y torchao` trước khi import peft (torchao 0.10 không tương thích peft 0.21).
2. `setuptools<81` cho `pkg_resources` mà torchmetrics cần.
3. `tools/score_comet_local.py` phải là bản có regex `seed(\d+)` (đã commit trong repo) — bản cũ chỉ đọc `*_seed42.parquet`.
