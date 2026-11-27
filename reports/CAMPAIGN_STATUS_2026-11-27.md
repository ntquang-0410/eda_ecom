# Campaign status — 2026-11-27

Đối chiếu 5 việc trong bảng yêu cầu.

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Train lại với `max_seq=1792` | 🔄 **đã khởi động, chưa xong** | Registry `campaign/jobs.json`: 30 run LLM (6 cell × 5 seed) ở 1792; 0 run xong. A100 đang 503, Kaggle hết quota tuần (reset ~28/11). |
| Campaign ≥5 seed | 🔄 **đã khởi động, chưa xong** | 54 job / ~216,7 giờ GPU còn lại; NLLB B seed 42 đang chạy, 53 job pending. |
| NLLB B với tập subsample 4,84 M | 🔄 **đang chạy seed 42** | Đã build `sample_B_sub4842_balanced_seed{7,42,123,2024,2025}.parquet` (500 step, 4,00 M token). Train seed 42 trên T4 từ 13:57 UTC, ETA ~8–9 h. |
| EcomGPT B-natural retry (LR thấp hơn) | ⏳ **sẵn sàng, chờ GPU** | `tools/train_ecomgpt.py` đã có `--lr`; runner `auto_resume_ecomgpt_b_retry.sh` (lr 5e-5, 500 step); A100 auto-acquire đang chạy. |
| Sửa harness eval (truncate + `--eval-max-seq`) | ✅ **xong** | Đã dùng cho các run mới; qwen4b_a đã sinh lại 9 dòng. |

## 1. Chi tiết campaign (item 1–2)

- `campaign/jobs.json`: 54 job.
  - 50 job LLM 1792: Qwen3-8B A/B, Qwen3-4B A/B, EcomGPT A/B, Moses A/B, OpenNMT A/B × 5 seed (7, 42, 123, 2024, 2025).
  - 4 job còn lại: NLLB B subsample (3 seed) + EcomGPT B-natural retry.
- Tổng **~216,7 giờ GPU**; trong đó 30 job LLM 1792 ≈ 172 h.
- Trạng thái hiện tại: 1 running (NLLB B s42), 53 pending.
- Nguồn GPU: Colab A100 (503 tạm thời, đang retry nền), Colab T4 (đang chạy NLLB), Kaggle (reset quota ~28/11).

## 2. NLLB B subsample (item 3)

- Units: `data/cleaned/nllb_train_b_subsampled.parquet` — 17.695 dòng, **4,841.873 token**, 0,826 epoch @4 M (A: 0,826).
- Sample: `data/cleaned/nllb_samples/sample_B_sub4842_balanced_seed{...}.parquet` — 500 step, 4,00 M token/draw, ~12k unique rows/seed.
- Đang chạy: seed 42, T4, `tools/train_nllb.py`, checkpoint mỗi 25 step (`runs/b_sub42/adapter_ckpt`), inference 1.500 dòng sau khi train.
- Nếu phiên T4 bị cắt: resume bằng `--init-adapter runs/b_sub42/adapter_ckpt --start-step N`.

## 3. EcomGPT B-natural retry (item 4)

- Đã thêm `--lr` vào trainer; recipe retry: `--configs natural --lr 5e-5 --steps 500 --max-seq 960`.
- Runner: `/content/auto_resume_ecomgpt_b_retry.sh`; upload adapter: `upload_ecomgpt_ckpt_b_lr5e5.sh` → dataset `nguynthanhquang/ecomgpt-b-natural-lr5e5-ckpt`.
- Chờ A100 (7B cần ≥24 GB; T4 16 GB không đủ an toàn).

## 4. Ước tính hoàn tất

| Nguồn GPU | Throughput | Thời gian cho 217 h |
|---|---|---|
| Colab A100 (12 h/phiên, ~50% thành công) | ~6 h/ngày | ~5–6 tuần |
| Kaggle 2 account (60 h/tuần) + Colab | ~12–15 h/ngày | ~2–3 tuần |

Không thể hoàn tất trong một phiên; registry + runner cho phép chạy tiếp qua nhiều phiên mà không lặp lại việc đã xong.
