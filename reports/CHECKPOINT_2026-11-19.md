# CHECKPOINT — 2026-11-19 11:30 UTC (trước khi ngắt GPU)

## 1. Đang chạy dở: Qwen3-8B × A-balanced

| Chỉ số | Giá trị |
|---|---|
| Tiến độ lúc checkpoint | **~150/500 step** (log: step 150 @ 102,4 phút) |
| Adapter lưu được | **step 100** (`adapter_balanced_seed42_ckpt`, 167MB) |
| Tốc độ | 0,67 phút/step, GPU 28,6GB, micro-batch 4 |
| Nếu chạy liên tục | train xong ~15:15, hypothesis xong ~17:00–17:30 |

**Không resume được giữa chừng theo nghĩa đầy đủ**: script gốc chỉ lưu adapter, không lưu optimizer state. Đã bổ sung warm-start (`--init-adapter` + `--start-step`) để **tiếp tục từ step 100** với Adam reset — chấp nhận được cho LoRA, nhưng phải ghi rõ trong báo cáo.

### Lệnh chạy tiếp (GPU mới)

```bash
cd /content/eda_ecom
mkdir -p /content/runs/qwen8b_a_balanced
python tools/train_weight_quota_gpu.py \
  --dataset A --data-root .artifacts/matrix_out/m4b_a/matrix_data \
  --out /content/runs/qwen8b_a_balanced --model Qwen/Qwen3-8B --seed 42 \
  --tokens 4000000 --steps 500 --configs balanced \
  --micro-batch 4 --dtype bf16 --max-seq 960 --save-every 100 \
  --init-adapter /content/ckpt8b/adapter_balanced_seed42_ckpt --start-step 100
```

Plan lấy mẫu là **tất định** (cùng units/config/seed/budget/steps) nên step 100–499 trùng đúng thứ tự dữ liệu.

Nếu muốn sạch tuyệt đối (không reset Adam), chạy lại từ step 0: bỏ `--init-adapter/--start-step`, tốn thêm ~100 phút.

## 2. Đã lưu ở đâu

| Nơi | Nội dung |
|---|---|
| **Kaggle** `nguynthanhquang/tiki-matrix-artifacts` | bản mới: `ckpt8b/` (adapter step 100 + log), `reports/runs_2026-11-19/` (6 run JSON đã backfill hash+exposure), `reports/*.md`, `tools/*.py`; giữ nguyên `hyps/` (16), `scores/` (32) |
| Local | `/content/ckpt8b/`, `/content/eda_ecom/` (git commit local) |
| GitHub / HF | **chưa đẩy được** — VM không có credential; cần token nếu muốn đồng bộ |

## 3. Trạng thái 3 phản biện (không thay đổi từ audit)

| Phản biện | Đạt | Còn thiếu |
|---|---|---|
| 1 — Leakage | 5/5 về khoa học, preflight 4 khoá đã thêm | hash trong run JSON đã backfill 6 cell; các cell gốc dùng manifest |
| 2 — OpenNMT | 3/4 | seed 2–3 cho NLLB; recipe chuẩn (tùy chọn) |
| 3 — Quota/length | 3/4 | 2×2 cho Qwen3-8B/EcomGPT/Moses/OpenNMT; seed thứ 3 |

## 4. Hàng đợi khi có GPU lại

1. `Qwen3-8B A-balanced` — resume từ step 100 (lệnh trên).
2. `Qwen3-8B B-natural` — script sẵn: `/content/run_8b_B_natural.sh` (dataset B, config natural).
3. `EcomGPT-7B A-balanced + B-natural` — dùng cùng script, đổi `--model`.
4. Seed thứ 3: Qwen3-4B A-natural (`--seed 7`) + NLLB seed 7.
5. Moses A-balanced/B-natural (CPU).
6. OpenNMT recipe chuẩn 6/512 + BPE 16k.

## 5. Môi trường khi resume

```bash
pip uninstall -y torchao          # bắt buộc, nếu không peft 0.21 lỗi
python -c "import torch;print(torch.cuda.is_available())"
# tải model: Qwen/Qwen3-8B (16GB) — snapshot_download
# COMET GPU (nếu cần chấm): /content/cometgpu (torch 2.6+cu124, comet 2.2.7, setuptools<81)
```
