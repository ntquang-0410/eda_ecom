# scripts/ft_train: fine-tune Qwen3-4B + LoRA trên hai bộ sau Judge (cụm, pair)

Kế hoạch đầy đủ (B0–B10): `C:\Users\ADMIN\.claude\plans\enumerated-questing-crystal.md` (ngoài repo) và tóm tắt ở `CLAUDE.md` §5.3b.
Dữ liệu vào: `data/processed/bo_chon_hf/` (HF `ntquang0410/zh-vie_ecom`, `data/LLM/bilingual_judged`, commit `82aa9c9`, hash kiểm trong `chung.py`).
Đầu ra: `data/processed/ft_run/` (gitignore). Chạy mọi script bằng `python -I -X utf8 <script>` (cần pandas, pyarrow, numpy; B3 cần `tokenizers`, B4 cần `sacrebleu`).

## Khóa tập test (bắt buộc)

- `chia_tap.py` tạo phép chia **một lần**: `ft_run/split.parquet` + `ft_run/DU_LIEU_DONG_BANG.json` (hash dữ liệu HF, hash split, hash danh sách test) + `ft_run/test_khoa/ds_test.txt`. Chạy lại bị từ chối nếu không có `--ghi-de`.
- Mọi script đọc phép chia qua `chung.doc_split()`: lệch hash là dừng.
- Mọi thứ dẫn xuất từ test nằm trong `ft_run/test_khoa/`. Gói dữ liệu huấn luyện (B1b) chỉ lấy `ft_run/mau/` (pool), không lấy `test_khoa/`.
- Tập test chỉ dùng để đánh giá (zero-shot, B9, B10). Không dùng để huấn luyện, dò siêu tham số hay chọn cấu hình.
- `kiem_ro_ri.py` kiểm độc lập, viết mã riêng, không dùng mã gom nhóm của `chia_tap.py`. Thoát mã 1 nếu có rò rỉ.

## Các file

| File | Bước | Việc |
|---|---|---|
| `chung.py` | – | đường dẫn, hash HF, `doc_bo()`, `doc_split()`, `kiem_khong_giao()` |
| `chia_tap.py` | B1 | gom nhóm gần-trùng (phép nối tập chính xác, 6 góc nhìn hai phía ngôn ngữ + luật cùng thương hiệu), khóa 10% test, chia 5 fold, phân tầng hướng × category |
| `kiem_ro_ri.py` | B1 | vét cạn mọi cặp test × pool và fold × fold khác |
| `rubric_kiem.md` | B2 | rubric chấm chung cho Gemini và Claude (Đạt / Nhẹ / Nặng); hash ghi vào lời nhắc |
| `lay_mau_kiem.py` | B2/B9 | lấy mẫu M1, M1c, thu20, M2, M2c, B9 từ test; dựng CSV gửi Gemini và lô JSONL cho Claude |
| `kiem_ket_qua.py` | B2/B9 | đọc kết quả, kiểm id/dòng, tỉ lệ lỗi nặng (title theo mục, mô tả theo mảnh), Wilson, kappa, hiệu chỉnh, cổng 5%, kiểm định dấu |
| `kiem_title.py` | B2 | 4 lô Gemini 200 title mới + 40 title Claude chấm mù; báo cáo lỗi nặng hai tầng (lõi, do thiếu) |
| `dung_mau.py` | B3 | mẫu chat (title, mô tả) mỗi nhánh; pool → `ft_run/mau/pool_{cum,pair}.jsonl`, test → `ft_run/test_khoa/test_{cum,pair}.jsonl` |
| `cham_diem.py` | B4/B8 | chrF++ (đích Việt) / chrF ký tự (đích Trung), COMET tùy chọn, kiểm số, Latin, chữ viết, số dòng, sinh lặp; cắt lời dẫn; điểm macro theo hướng |
| `loc_title.py` | B2 | bộ lọc tự động tìm title có dấu hiệu lỗi, chỉ đọc; chỉ dùng để sàng |
| `dong_goi_kaggle.py` | B0, B1b, B5–B8 | đóng gói thư viện, mô hình; Dataset CHỈ có mẫu pool; ghi đè và chạy notebook `nlp-group3` (`chay_b5`, `b6_tu_dong`, `b78_thu`, `b78_tu_dong`); `luu_b6` lưu adapter sang notebook riêng, phải chạy trước khi đẩy lại `nlp-group3` |
| `kaggle_train.py` | B5–B8 | mã chạy trên Kaggle: thử môi trường, đo tốc độ, dò siêu tham số loại dần, B7 + B8 (`chay_b78`), đều chạy tiếp được qua nhiều phiên |
| `tong_hop_b78.py` | B7, B8 | chạy trên máy: đọc `b78_trang_thai.json`, tính chốt chặn C1, C2, C3, túi lỗi category, kết luận LoRA attention; không tự bật thêm việc |
| `tong_hop_ti_le.py` | Tỉ lệ, bước 2, 3 | chạy trên máy: chấm lại bản dịch từng mức, áp cổng fold 0, viết `bao_cao_ti_le.md`; thử trên Kaggle bằng `dong_goi_kaggle.py tl_b2 moi\|tiep` |
| `kiem_dong_moi.py` | Tỉ lệ, kiểm tín hiệu | chạy trên máy, không GPU: so mức ~11% với 1 : 1 trên dòng mới của mô tả, kèm độ nhạy chữ hoa và cụm sản phẩm chung dòng, viết `kiem_dong_moi.md`; không đọc test |
| `kiem_fold_ti_le.py` | Tỉ lệ, kiểm trên fold khác | chạy trên máy, không GPU: áp luật khóa trên một fold (`--fold F --moc <sinh> --muc <sinh> --ra <thu_muc>`), viết `kiem_fold<F>.md`; bản dịch lấy từ `dong_goi_kaggle.py b8f1 moi\|tiep`, `b8f2 moi\|tiep`; lưu đầu ra trước lần đẩy kế bằng `luu_tl`, `luu_b78`, `luu_b8f1` |
| `kiem_3fold_ti_le.py` | Tỉ lệ, kết luận bản cụm | chạy trên máy, không GPU: áp tiêu chí 3 fold, viết `kiem_3fold.json`; không đọc test |
| `nhat_ky_chay.py` | Nhật ký mọi lần chạy | chạy trên máy, không GPU: gom thông số, kết quả, so sánh theo cặp, giờ GPU, hash mã/dữ liệu/bản dịch của MỌI lần chạy từ file kết quả, viết `ft_run/NHAT_KY_CHAY.{md,json}` để tra cứu khi viết báo cáo và phản biện; **chạy lại sau mỗi phiên mới** (thêm phiên vào `PHIEN_CHINH` và `NGUON_TRANG_THAI` trong file); không đọc test |
| `test_*.py` | – | test trên dữ liệu tự dựng, chạy trực tiếp |

## Thứ tự chạy

```
python -I -X utf8 scripts/ft_train/chia_tap.py          # một lần (đã chạy, v2, 05/11/2026)
python -I -X utf8 scripts/ft_train/kiem_ro_ri.py
python -I -X utf8 scripts/ft_train/dung_mau.py --tokenizer <tokenizer.json của Qwen3>
python -I -X utf8 scripts/ft_train/lay_mau_kiem.py
python -I -X utf8 scripts/ft_train/kiem_ket_qua.py [--thu20]
python -I -X utf8 scripts/ft_train/cham_diem.py --tham-chieu ... --du-doan ... --ra-thu-muc ...
```

## Lịch sử phép chia

- v1 (04/11): 6 góc nhìn Jaccard (title 0,5; mô tả 0,7). Bị thay trước khi được dùng.
- v2 (05/11, hiện hành): thêm luật "cùng thương hiệu cụ thể + title chứa trong nhau ≥0,7", vì đã thấy sp test "Trà kỷ tử" là tin đăng anh em của 3 sp pool cùng hiệu. Test 2.558 sp (zh→vi 1.485, vi→zh 1.073), pool 23.017.
