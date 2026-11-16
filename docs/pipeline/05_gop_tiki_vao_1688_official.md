# 05. Title, gộp Tiki vào 1688 và bộ official (v1, v2)

[← 04](04_ra_soat_theo_cum_bo_3_4.md) · [Tổng quan](00_TONG_QUAN.md) · [06 →](06_loc_luat_ft_clean.md)

Thời gian: 26/10 – 29/10/2026. Nguồn:
- [`docs/gop_tiki_1688/KE_HOACH_GOP_TIKI_VAO_1688.md`](../gop_tiki_1688/KE_HOACH_GOP_TIKI_VAO_1688.md);
- WALKTHROUGH Prompt 510–557 và Trả lời 600.

## 1. Phía title (Huy)

Title của cả 1688 (zh→vi) và Tiki (vi→zh) do **Huy** dịch. Mã và tài liệu nằm ở nhánh **`giahuy`** của repo GitHub `ntquang-0410/eda_ecom`, Huy push ngày 04/11/2026 (commit `c4864fc` đến `b8afe40`). Phần dưới đây tóm tắt từ các tài liệu đó; Nhật Anh không chạy lại.
- `README.md`, `docs/ARCHITECTURE.md`, `docs/OPERATIONS.md`;
- `docs/history/titles/TITLE_TRANSLATION_PIPELINE.md`, `TIKI_TITLE_TRANSLATION_GUIDE.md`;
- các hướng dẫn `TITLE_TRANSLATION_QWEN3_14B_AWQ_*`.

**Nguồn title.** Khác phía mô tả, title đi từ bản làm sạch CADO (của Quang) trên HF `data/cleaned/`:

| Nguồn | File | Dòng | Title duy nhất (dịch một lần rồi ánh xạ lại) |
|---|---|---|---|
| 1688 | `bilingual_zh_vi_cado.parquet` (revision `e098c4db`) | 16.348 | 15.863 `title_zh` |
| Tiki | `tiki/tiki_vi_cado.parquet` (revision `51540c9e`) | 13.200 | 13.102 `title_vi_cleaned` |

**Hai mô hình**, mỗi mô hình có cache/checkpoint SQLite riêng:
- **Gemma 4 26B-A4B** (`gemma-4-26b-a4b-it`), gọi qua Google AI Studio API. Chỉ gửi title, không gửi `product_id`.
- **Qwen**, chạy trên GPU Colab/Kaggle:
  - lúc đầu là Qwen3.5-9B 8-bit;
  - sau đó viết lại thành **Qwen3-14B-AWQ** (notebook v4–v6 cho Colab và Kaggle).
  - `docs/OPERATIONS.md` ghi Qwen3.5-9B là cấu hình cũ, nên bản Qwen của title cuối nhiều khả năng là Qwen3-14B-AWQ. Tài liệu không có dòng nào khẳng định thẳng điều này.

**Các bậc chạy có cổng:**
1. smoke (20 title), rồi pilot (150), rồi validate (200 title mới). Mỗi bậc phải có ≥99% đầu ra hoàn chỉnh và ≥95% mẫu qua QA kỹ thuật.
2. full chạy từng cửa sổ 200 title, dừng nếu đầu ra không hoàn chỉnh vượt 1% hoặc lỗi QA vượt 5%.
3. Lượt Tiki có cổng nhẹ hơn: smoke 10, pilot 50, QA ≥90%. Lượt này cho phép chạy full khi còn cờ QA (`--defer-full-quality-review`), nhưng ghi rõ trong báo cáo.

**QA và sửa:** tìm chữ nguồn còn sót, số liệu lệch, đầu ra rỗng/cắt cụt/lặp. Sửa có giới hạn và giữ dấu vết bản gốc. Mọi bản dịch gắn `semantic_verified=False`.

**Chọn bản bằng COMETKiwi:** chấm hai ứng viên trên cùng nguồn.
- Script `compare_1688_cometkiwi.py` giữ chính sách "strict" ban đầu: QA trước, floor 0,72, có thể HOLD.
- Chính sách **xuất title cuối** khác: **lấy bản điểm cao hơn, không floor, QA chỉ cảnh báo** (`archive/release_history/export_comet_best_without_floor.py`).
- Kết quả ghi vào cột `title_nguon_dich`.
- Về sau Judge DeepSeek còn đổi 7.604 title (file 07). `judge_chon_title` là mô hình của bản nháp title cuối, không luôn là lựa chọn trực tiếp của Judge.

- Ngày 28/10 Huy đưa lên HF 4 file "full" 9 cột, mỗi nguồn 2 bản (commit `246ba7e482`):

| File | Dòng |
|---|---|
| `data/1688/full_dataset/1688_full_attributes_context_zh_vi.parquet` (mô tả **cụm**, bộ 4) | 16.348 |
| `data/1688/full_dataset/1688_full_attributes_independent_zh_vi.parquet` (mô tả **pair**, bộ 1) | 16.348 |
| `data/tiki/full_dataset/tiki_full_attributes_context_vi_zh.parquet` (cụm, bộ 3) | 13.200 |
| `data/tiki/full_dataset/tiki_full_attributes_independent_vi_zh.parquet` (pair, bộ 2) | 13.200 |

- 9 cột đã chốt, đúng thứ tự: `product_id, source, category, title_zh, title_vi, description_zh, description_vi, title_nguon_dich, desc_nguon_dich`.
- Đã kiểm 28/10: mô tả và `desc_nguon_dich` khớp 100% bảng sản phẩm Nhật Anh đã đẩy (16.342 và 13.187 sản phẩm).
- Gộp category Tiki vào hệ 1688: Tiki có 6 nhóm, đều trùng tên nhóm của 1688.

## 2. Quy tắc gộp (Nhật Anh soạn, Huy và Nhật Anh duyệt)

- Danh sách giữ/loại quyết định **một lần trên bảng sản phẩm bộ cụm**, rồi áp **giống hệt** cho bản pair. Vì vậy hai bản luôn cùng tập sản phẩm.
- Mỗi sản phẩm bị loại ghi đúng một `ly_do_loai`, theo quy tắc đầu tiên nó vi phạm.

| Bước | Quy tắc | Loại | Còn lại |
|---|---|---|---|
| R0 | Không có mô tả (1688 6, Tiki 13), từ 29.548 sản phẩm | 19 | 29.529 |
| R1 | Sai nhóm hàng theo danh mục lá (Tiki food 377 thú cưng/dược phẩm, mother_baby 163 đồ người lớn, home 17 nhạc cụ; 1688 37) | 594 | 28.935 |
| R2 | Tỉ lệ cặp yếu `so_cap_thap / so_cap` của bộ cụm ≥ 0,25 | 214 | 28.721 |
| R3 | Mô tả trùng hệt quá 3 bản (cùng nguồn, nhóm, `description_zh`; giữ 3 theo crc32 của `product_id`) | 825 | 27.896 |
| R4 | Title 1688 còn chữ Hán trong `title_vi` | 31 | 27.865 |
| R5 | Vượt trần **3.500 sản phẩm mỗi nhóm** (chỉ home bớt 944, mother_baby bớt 881) | 1.825 | 26.040 |

Huy thêm 2 quy tắc title khi chạy: 4 title Tiki chưa dịch, 1 title 1688 là câu từ chối của mô hình. Kết quả cuối là **26.038**.

## 3. Official v1 (29/10, HF commit `ef346872ec`, Huy chạy và đẩy)

- `data/bilingual_official/`: **26.038 sản phẩm** (1688 14.996, Tiki 11.042), loại 3.510.
- Các file: `bilingual_official_context.parquet` (bản cụm), `bilingual_official_pair.parquet` (bản pair), `excluded_products.parquet`, `selection_report.json`, manifest.
- Nhật Anh kiểm chéo độc lập: chạy lại cùng đầu vào và quy tắc, dấu vân tay và từng ô dữ liệu khớp.

Phân bố dự kiến theo nhóm (tính trước khi Huy thêm 2 quy tắc title):

| Nhóm | 1688 | Tiki | Cộng |
|---|---|---|---|
| home | 1.779 | 1.721 | 3.500 |
| mother_baby | 1.957 | 1.543 | 3.500 |
| beauty | 1.292 | 2.096 | 3.388 |
| electronics | 1.085 | 2.094 | 3.179 |
| food | 1.406 | 1.710 | 3.116 |
| bags | 2.874 | 0 | 2.874 |
| auto | 960 | 1.880 | 2.840 |
| shoes | 2.309 | 0 | 2.309 |
| fashion | 1.334 | 0 | 1.334 |

## 4. Official v2 strict (29/10 17:21, HF commit `4a23574`)

- Thư mục chuyển sang `data/LLM/bilingual_official/`; bản cũ lưu ở `archive_v1`.
- `criteria.json` version `2026-10-30-strict-v2`.
- **Huy** làm và đẩy bản này. Commit hiện tên `ntquang0410` vì nhóm đẩy chung qua tài khoản HF đó. `HUONG_DAN_JUDGE.md` ghi "bộ official v2 do Huy gộp (commit `4a23574`)", và Nhật Anh xác nhận ngày 16/11.
- Bản v2 là chính bản v1 siết thêm bộ lọc kỹ thuật. Commit có thông điệp "Tighten official technical quality filters and preserve previous release".
- Kết quả: **25.956 sản phẩm** (1688 14.928, Tiki 11.028), bớt 82 sản phẩm so với v1:
  - 57 trùng song ngữ y hệt;
  - 25 lỗi kỹ thuật: vượt giới hạn ký tự (title_zh 192, title_vi 256, description_zh 3.000, description_vi 6.000), ký tự Unicode/điều khiển lỗi, lặp cụm.
- v2 là tập con của v1. Các dòng giữ lại **không đổi nội dung ở cột nào** [ĐO, Trả lời 600].
- Bản v2 (cụm và pair) là đầu vào của bước lọc luật (file 06). Bản sao trên máy ở `data/snapshot/bilingual_official_v2/`, kiểm SHA256 theo manifest trước khi dùng.

## 5. Báo cáo với thầy

Nhật Anh báo cáo bộ song ngữ đã gộp với GVHD bằng một bộ slide ngắn quanh sơ đồ phễu (Prompt 558–601).

Chọn bản nào để fine-tune:
- Lúc đầu Nhật Anh nghiêng về bản cụm (Prompt 568–569).
- Đến khi lên kế hoạch fine-tune trên bộ sau Judge, quyết huấn luyện và đánh giá **cả hai bản** (Prompt 698, 707).
