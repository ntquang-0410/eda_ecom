# KẾ HOẠCH PHÂN TÍCH P1 (khóa trước khi xem số)

Ngày 17/11/2026. Người lập: OpenCode (theo giao việc `PROMPT_OPENCODE_P0_P1.md`). Mọi phân tích chỉ trên **eval fold 0** (train = fold 1–4). Không đọc test. Không huấn luyện, không sinh bản dịch mới.

## 0. Phạm vi và nguồn

- Bộ: `pair` (1.198 mẫu) và `cum` (1.200 mẫu), eval fold 0.
- Tham chiếu: `cham_co_dien.tham_chieu(hg, nhanh)`; điền `category` từ pool theo `product_id` (id = `<product_id>_<loai>_<...>`).
- Bản dịch: đúng các file `tong_hop_5mh.cac_mo_hinh()` dùng.
- Tập "dòng đã thấy": `cham_diem.xay_tap_da_thay(pool_df, fold_kiem=0, nhanh=set)`, đúng định nghĩa hiện hành: cặp `(category, dòng nguồn đã chuẩn hoá)` của mọi mẫu `loai=mo_ta` trong pool ở fold 1–4, tách dòng theo `nhanh`.

## 1. Tái lập 5 mô hình (3.1)

- Chạy lại đúng quy trình `tong_hop_5mh.py` (import hàm, không chép logic chấm) cho cả `pair` và `cum`.
- So từng ô `tb_chinh`, `tb_macro`, `chrf_corpus` với `tong_hop_5mh{,_cum}/ket_qua.json`.
- **Luật dừng:** lệch tuyệt đối > 0,01 ở bất kỳ ô nào thì dừng, báo, không tự chỉnh.
- Đầu ra: `data/processed/ft_run/phan_bien_v2/tai_lap_{pair,cum}.json`.

## 2. Tách dòng khuôn / dòng mới (3.2)

- Đơn vị: chỉ **mô tả** (`loai=mo_ta`); title không tách dòng.
- Chỉ dùng sản phẩm `can_dong=True` (số dòng hyp = ref = nguồn). Báo số sản phẩm bị loại theo từng mô hình.
- **Tỉ lệ dòng đã thấy:**
  - theo dòng = `tổng dòng đã thấy / tổng dòng` trên các sản phẩm `can_dong`;
  - theo sản phẩm = trung bình đều của `(số dòng đã thấy / số dòng)` của từng sản phẩm `can_dong` (mỗi sản phẩm một phiếu).
  - Báo riêng từng bộ × hướng (zh→vi, vi→zh).
- **chrF:**
  - `chrF corpus` trên dòng khuôn và dòng mới: lấy từ `th["dong_da_thay_moi"][h][nhom]["chrf_corpus"]` (đúng hàm `cham`), tính cả hai hướng.
  - `chrF câu TB` theo sản phẩm: với mỗi sản phẩm, chrF corpus của các dòng mới (hoặc khuôn) của sản phẩm đó; trung bình đều qua sản phẩm. Dùng cho bootstrap.
- **So sánh:** chênh `mô hình − Qwen3-4B` trên dòng mới, theo sản phẩm, chỉ giữ sản phẩm có dòng mới ở **cả hai** mô hình; bootstrap 2.000 lần, seed 42, khoảng percentile 2,5–97,5. Làm tương tự cho dòng khuôn (phụ).
- **Giả thuyết ghi trước:** khoảng cách của OpenNMT/Moses so với Qwen3-4B **lớn hơn** trên dòng mới so với dòng khuôn. Nếu sai, ghi đúng.
- Kết cục dự kiến:
  - Nếu CI chứa 0: "không kết luận được khác biệt".
  - Nếu CI hoàn toàn dương/âm: ghi theo dấu.
- **Đầu ra:** `data/processed/ft_run/phan_bien_v2/tach_dong_{pair,cum}.json`.

## 3. Vòng lặp tham chiếu (3.3)

- Nguồn nhóm (theo sản phẩm):
  - Title: `judge_chon_title` ∈ {gemma, qwen}. Bỏ sản phẩm thiếu.
  - Mô tả: đọc `judge_chon_description` dạng `"gemma: a | qwen: b | quy_tac: c | chua_ro: d"`; `qwen_share = b / (a + b)`.
    - Qwen chủ đạo: `qwen_share ≥ 2/3`.
    - Gemma chủ đạo: `qwen_share ≤ 1/3`.
    - Còn lại: trộn. `a + b = 0`: không xác định.
- **Chỉ số:** với mô tả zh→vi, mỗi sản phẩm có chrF corpus của các dòng mô tả; nhóm lấy trung bình đều qua sản phẩm.
- **Chênh lệch kép:** `(M − baseline | nhóm Qwen) − (M − baseline | nhóm Gemma)`, với `M` ∈ {Qwen3-4B, Qwen3-8B, zero-shot Qwen3-4B}; `baseline` ∈ {Moses học đủ, OpenNMT, EcomGPT-7B}.
  - Bootstrap: lấy lại sản phẩm **trong từng nhóm** (độc lập), 2.000 lần, seed 42; khoảng percentile.
- Kết cục:
  - CI chứa 0: không kết luận.
  - CI dương: mô hình họ Qwen có lợi thế lớn hơn ở nhóm tham chiếu Qwen.
  - CI âm: ngược lại.
- Giới hạn ghi kèm: Judge chọn Qwen khi Gemma kém nên hai nhóm khác độ khó; chênh lệch kép chỉ khử phần độ khó chung; bản title "qwen" có thể là Qwen3-14B hoặc Qwen3.5-9B [CHƯA KIỂM].
- **Đầu ra:** `data/processed/ft_run/phan_bien_v2/vong_tham_chieu_{pair,cum}.json`.

## 4. Theo độ dài (3.4)

- Với mỗi bộ × hướng × loại (title/mô tả), tính tứ phân vị độ dài **ký tự nguồn** trên eval (Q1, Q2, Q3).
- Chia 4 nhóm: `≤Q1`, `(Q1,Q2]`, `(Q2,Q3]`, `>Q3`.
- Với mỗi mô hình, báo chrF corpus theo nhóm (và số dòng). Không bootstrap (chỉ mô tả).
- Kết cục: ghi nhận mô hình nào tụt mạnh ở nhóm dài; không dùng để xếp hạng tổng.
- **Đầu ra:** `data/processed/ft_run/phan_bien_v2/do_dai_{pair,cum}.json`.

## 5. Đường học OpenNMT (3.5)

- Đọc `kaggle_onmt_that/onmt/{zh-vi,vi-zh}_train/train.log`, `dev.json`, `onmt_report.json`.
- Nếu log có điểm dev theo bước/epoch: vẽ PNG (matplotlib) và bảng số; ghi bước dừng sớm, checkpoint trung bình.
- Nếu không có: ghi rõ "log không có điểm dev theo thời gian", không bịa.
- **Đầu ra:** `docs/phan_bien_vong2/hinh/opennmt_{zhvi,vizh}.png` (nếu vẽ được) + mục trong KET_QUA_P1.

## 6. Compute và hợp nhất OpenNMT (3.6)

- Bảng compute 5 mô hình: tham số, token đã học, epoch/lượt, GPU-giờ nếu có; cột pretraining:
  - Qwen3: ~36 nghìn tỷ token (báo cáo kỹ thuật Qwen3) **[CHƯA KIỂM]**.
  - EcomGPT: dựa trên BLOOMZ-7b1-mt; BLOOM ~366 tỷ token **[CHƯA KIỂM]**.
  - OpenNMT, Moses: 0.
- Bảng hợp nhất OpenNMT: nhánh Nhật Anh lấy từ D3/onmt_report; nhánh Quang lấy **từ biên bản** (37M, 4 lớp/256, char 32k, beam 512, replace_unk, 30k/22.990 step, eval_F, COMET 0,349), ghi rõ "theo biên bản, chưa đối chiếu cấu hình gốc".
- Câu bắt buộc: hai tập đo và hai thước đo khác nhau, không so điểm trực tiếp.

## 7. Nhãn và cách viết

- Mọi số mới gắn [ĐO]/[ƯỚC]/[CHƯA KIỂM]. Số kiểu Việt trong chữ; bảng giữ kiểu số của bảng gốc.
- Không dùng "—" trong câu văn; không dùng "tối ưu", "vượt trội", "chứng minh" nếu không có số.
- Kết quả ra sao ghi vậy, kể cả khi trái giả thuyết.

## 8. Thứ tự chạy

1. Tái lập (mục 1), dừng nếu lệch > 0,01.
2. Tách dòng (mục 2).
3. Vòng tham chiếu (mục 3).
4. Độ dài (mục 4).
5. Đường học + compute (mục 5–6).
6. Viết `docs/phan_bien_vong2/KET_QUA_P1.md`.
