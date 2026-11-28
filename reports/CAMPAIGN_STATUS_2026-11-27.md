# Campaign status — 2026-11-27

Đối chiếu 5 việc trong bảng yêu cầu.

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Train lại với `max_seq=1792` | 🔄 **đã có A100, job đầu đang chờ chạy** | Qwen3-4B A-natural seed 2024 @1792 xếp hàng sau EcomGPT retry (`chain_4b_1792.sh`); 30 run LLM trong registry. |
| Campaign ≥5 seed | 🔄 **đã khởi động, chưa xong** | 54 job / ~216,7 giờ GPU còn lại; NLLB B seed 42 đang chạy, 53 job pending. |
| NLLB B với tập subsample 4,84 M | 🔄 **đang chạy seed 42 trên A100** | Sample 5 seed đã build. T4 bị thu hồi khi A100 vào; restart trên A100 14:47 UTC (nhanh hơn ~2×), checkpoint 25 step. |
| EcomGPT B-natural retry (LR thấp hơn) | 🔄 **đang chạy** | A100 14:44 UTC: tải model 28 GB → train lr 5e-5, 500 step, upload checkpoint 25 step. |
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

## 5. Cập nhật 15:00 UTC — A100 đã vào

- A100 `gpu-a100-s-ass1c0-maczc7fy7k95` (14:40 UTC), hạn phiên ~02:40 UTC.
- Đang chạy song song: **EcomGPT B-natural retry** (bootstrap 28 GB) và **NLLB B s42** (train từ 14:47).
- Chain: khi EcomGPT xong → tự chạy **Qwen3-4B A-natural seed 2024 @1792** (~6 h, có checkpoint upload).
- T4 bị Colab thu hồi khi tạo A100 (giới hạn 2 GPU session) — NLLB mất ~45 phút train trên T4, đã restart trên A100.

## 6. Cập nhật 17:55 UTC — A100 bị preempt, đã chuẩn bị resume

- A100 `gpu-a100-s-ass1c0-maczc7fy7k95` **bị thu hồi** sau ~3 h (không phải hết 12 h). Toàn bộ
  process trên máy đó dừng theo.
- **EcomGPT retry**: đã kịp upload checkpoint **step 75** → `nguynthanhquang/ecomgpt-b-natural-lr5e5-ckpt`.
  Runner đã sửa để tự tải ckpt và resume `--init-adapter --start-step 75`.
- **NLLB B s42**: lần chạy trên A100 **mất trắng** vì chưa có upload (bài học: phải upload ngay).
  Đã thêm `--upload-script` vào `train_nllb.py` + dataset `nllb-b-sub4842-ckpt`; runner mới
  `run_nllb_b_sub4842.sh` tự resume.
- **4B 1792 seed 2024**: chưa từng chạy (chain chờ EcomGPT).
- Đang chạy lại vòng xin A100 (`a100_campaign2.log`); khi vào sẽ chạy `chain_all.sh`:
  NLLB + EcomGPT song song, rồi 4B 1792 khi EcomGPT xong.
- Tất cả 54 job trở về `pending` trong registry; không có kết quả nào bị tính sai.

## 7. Cập nhật 18:35 UTC — A100 mới đã vào, resume chạy

- A100 mới `gpu-a100-s-ass1c0-3s6xzguhic3uu` (18:13 UTC). Đã sync lại repo + scripts + NLLB stage.
- **NLLB B s42**: chạy từ 18:16; step 25 lúc 18:25 (~24 s/step khi tải model song song), **đã upload ckpt step 25**
  (`nllb-b-sub4842-ckpt`, xác minh STEP=25). ETA train ~21:45, inference ~23:45.
- **EcomGPT retry**: đang tải model 17,8/28 GB (18:31), resume từ step 75; ETA xong ~22:10.
- **4B 1792 seed 2024**: chain cũ fire sớm (chỉ kiểm tra `train_ecomgpt`, không thấy bootstrap) nên đã
  kill job 4B vừa khởi động; đã thay bằng waiter kiểm tra cả bootstrap/download/train, sẽ chạy sau EcomGPT.
- Cả 3 job đều upload checkpoint 25 step ⇒ preempt không mất quá 25 step.

## 8. Cập nhật 01:20 UTC 28/11 — chuyển sang Kaggle (quota đã reset)

- A100 lại bị thu hồi (~sau 20:00 27/11). Cứu được: **NLLB ckpt step 100**, **EcomGPT ckpt step 125**;
  4B chưa từng chạy.
- Kaggle quota reset 00:00 28/11 → đã đẩy 2 kernel GPU (song song):
  - `nguynthanhquang/tiki-nllb-b-sub4842-s42` — resume step 100, L4, model NLLB mount sẵn.
  - `nguynthanhquang/tiki-ecomgpt-b-natural-lr5e5-resume` — resume step 125, lr 5e-5, RTX Pro 6000.
- `nguynthanhquang/tiki-m4b-a-s2024-1792` đã build xong, **chờ slot** (Kaggle giới hạn 2 GPU session);
  vòng lặp nền tự push khi một kernel kết thúc.
- Dữ liệu cho Kaggle: `nllb-b-sub4842-data` (units 4,84M + sample seed 42/7 + eval_B) và
  `tiki-matrix-tools-v2` (trainer có `--eval-max-seq`).
- Không còn phụ thuộc Colab cho các job này; kernel Kaggle chạy tới 12 h, có output để tải.

## 9. Cập nhật 01:40 UTC 28/11 — dùng account thứ 2 (anhtamxinhdep)

- Đã đưa **Qwen3-4B A seed 2024 @1792** lên account `anhtamxinhdep` (Kaggle), shape **NvidiaRtxPro6000**.
- Dataset riêng cho acc02: `anhtamxinhdep/tiki-matrix-data` (đã có) + `anhtamxinhdep/tiki-matrix-tools-v2` (mới upload).
- **Bug phát hiện & sửa:** `BUCKETS` dừng ở 1024 → tại `max_seq=1792` collate lỗi
  `expected sequence of length 1246 at dim 1 (got 1024)`. Đã mở rộng
  `BUCKETS = (…, 1024, 1280, 1536, 1792, 2048)` trong `tpu_weight_train.py`, version lại
  tools-v2 cho cả 2 account; kernel 4B đã chạy lại (v3) và qua điểm crash cũ.
- Hiện có **3 kernel GPU chạy song song**:
  | Account | Kernel | Nội dung |
  |---|---|---|
  | nguynthanhquang | `tiki-nllb-b-sub4842-s42` | NLLB resume step 100 |
  | nguynthanhquang | `tiki-ecomgpt-b-natural-lr5e5-resume` | EcomGPT lr 5e-5 resume 125 |
  | anhtamxinhdep | `tiki-m4b-a-s2024-1792` | Qwen3-4B A seed 2024, max_seq 1792 |
- Acc02 còn 1 slot GPU (Kaggle cho 2/account) nếu muốn chạy thêm cell.

## 10. Cập nhật 01:42 UTC — chính sách 1 kernel/account

- **acc01 (`nguynthanhquang`)** chỉ còn `tiki-ecomgpt-b-natural-lr5e5-resume` (resume 125).
  Kernel NLLB đã **xoá** để tuân thủ 1 kernel/account; sẽ **tự push lại** (resume từ ckpt step 100)
  ngay khi EcomGPT xong — vòng lặp nền `after_ecom_loop.sh`.
- **acc02 (`anhtamxinhdep`)** chỉ có `tiki-m4b-a-s2024-1792` (RTX Pro 6000, max_seq 1792).
- Hệ quả: NLLB mất ~35 phút train trên Kaggle (từ 100), nhưng ckpt 100 vẫn còn nên resume sạch.
