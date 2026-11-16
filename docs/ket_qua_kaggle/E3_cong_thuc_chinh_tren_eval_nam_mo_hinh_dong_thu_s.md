# E3. Công thức chính trên eval năm mô hình ("dòng thứ sáu", T4 fp16)

[← Mục lục](INDEX.md) · [← E2](E2_thu_cheo_dang_pair_va_dang_cum.md) · [F1 →](F1_cong_nen_de_chay_demo_tren_laptop_khong_phai_kag.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | E3 |
| Giai đoạn | E. Thử chéo và công thức chính trên eval năm mô hình |
| Notebook Kaggle | `honghuynanh/nlp-dong6` |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 14/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_dong6` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_dong6/bao_cao_dong6.json` | Tesla T4 | 14.6 | 2.10.0+cu128 | 5.0.0 | 0.18.1 | 5.758 giây (1.60 giờ) | có | – |

## Mục đích

Cho adapter công thức chính học fold 1–4 (E1) dịch đúng 1.200 mẫu eval của năm mô hình, để biết công thức chính hơn vòng sàng lọc bao nhiêu trên cùng mẫu.

## Sinh và chấm trên Kaggle (`bao_cao_dong6.json`)

Cấu hình: `{"tap": ["cum", "pair"], "dtype": "fp16", "bs": 32, "ngan_sach": 20000, "gio_toi_da": 8.0}`. Môi trường: `{"torch": "2.10.0+cu128", "transformers": "5.0.0", "peft": "0.18.1", "gpu": "Tesla T4", "vram_gb": 14.6, "dtype": "fp16"}`. Đạt: có. Tổng 5.758 giây (1.60 giờ).

| Bộ | Adapter | Hàng eval | Lượt gọi | Khối lệch dòng | Rỗng | chrF macro (corpus) | chrF macro (câu) | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả | Giây nạp | Giây/lượt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum | cum_f0_p11 | 1.200 | 1.200 | 0 | 0.00% | 76.72 | 76.93 | 62.48 | 93.48 | 60.35 | 90.58 | 1756 | 1.699 |
| pair | pair_f0_p11 | 1.198 | 1.165 | 0 | 0.00% | 74.99 | 78.62 | 62.37 | 85.60 | 60.95 | 91.03 | 9 | 1.663 |

Đường dẫn adapter trên Kaggle: `/kaggle/input/notebooks/honghuynanh/nlp-group3/cheo_cum/cum_f0_p11/ep2`; `/kaggle/input/notebooks/honghuynanh/nlp-group3-pair/cheo_pair/pair_f0_p11/ep2` (bản huấn luyện lại ở tài khoản 2, phiên E1).

## Đặt cạnh Qwen3-4B vòng sàng lọc (`tong_hop_5mh.py`)

Eval fold 0. Pair: 1.198 mẫu (từng dòng); cụm: 1.200 mẫu. chrF corpus; `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi; `Macro` = trung bình đều hai hướng. Chênh so với Qwen3-4B (vòng sàng lọc) tính trên chrF câu trung bình, bootstrap theo sản phẩm (2.000 lần, cùng mẫu rút cho mọi mô hình). LLM là trung bình 2 seed.

| Bộ | Mô hình | Số lần chạy | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả | Chính (corpus) | Macro (corpus) | Chính (câu TB) | Chênh chính (câu) so với 4B [95%] | Chênh macro (câu) so với 4B [95%] | Sinh lặp (TB 4 nhóm) | Lệch chính giữa 2 seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair | Qwen3-4B | 2 | 49.98 | 83.31 | 54.20 | 86.58 | 79.98 | 68.52 | 90.03 | mốc | mốc | 0.62% | 1.63 |
| pair | Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16) | 1 | 62.37 | 85.60 | 60.95 | 91.03 | 83.28 | 74.99 | 93.99 | +3.95 [+2.98; +4.99] | +6.05 [+5.31; +6.79] | 0.00% | – |
| cum | Qwen3-4B | 2 | 54.19 | 90.44 | 56.33 | 86.54 | 86.81 | 71.87 | 87.65 | mốc | mốc | 0.42% | 0.20 |
| cum | Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16) | 1 | 62.48 | 93.48 | 60.35 | 90.58 | 90.38 | 76.72 | 91.02 | +3.36 [+2.96; +3.78] | +4.48 [+3.92; +5.06] | 0.08% | – |

SacreBLEU / TER (trung bình các lần chạy; sacrebleu 2.6.0, zh→vi `tok:13a`, vi→zh `tok:zh`, TER `tercom`; BLEU kiểu mteval-v13a trùng SacreBLEU):

| Bộ | Mô hình | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả |
|---|---|---|---|---|---|
| pair | Qwen3-4B | 25.70 / 65.42 | 76.04 / 20.18 | 35.68 / 56.77 | 84.86 / 9.97 |
| pair | Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16) | 41.95 / 45.50 | 78.68 / 15.93 | 45.36 / 42.06 | 90.56 / 5.55 |
| cum | Qwen3-4B | 31.35 / 59.12 | 86.73 / 12.30 | 39.04 / 48.14 | 87.58 / 9.87 |
| cum | Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16) | 42.73 / 45.12 | 90.50 / 8.37 | 45.44 / 44.03 | 91.21 / 6.64 |

## Kết luận của phiên

- Công thức chính hơn 4B sàng lọc rõ ở cả hai bộ (khoảng tin cậy xa 0).

## Ghi chú, giới hạn, sự cố

- Chạy trên Tesla T4 fp16 (đã mất RTX Pro 6000), khác bf16 lúc huấn luyện.
- Lời dẫn theo `dung_mau.py`, pair dịch cả khối rồi tách dòng, nên không xếp hạng chung với năm dòng sàng lọc.
- Hai lần chạy thử trước đó (`kaggle_dong6_khoi`, `kaggle_dong6_khoi_fp16`) nằm ở nhóm khói trong INDEX.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_dong6/bao_cao_dong6.json` | 7.040 | `1a8ca526c9816e6a` |
| `ft_run/kaggle_dong6/dong6/eval_items_cum.jsonl` | 1.146.496 | `e990e9adf4769761` |
| `ft_run/kaggle_dong6/dong6/eval_items_pair.jsonl` | 405.943 | `2ebe2521748f789d` |
| `ft_run/kaggle_dong6/dong6/sinh_cum_f0_p11_cum.jsonl` | 1.095.130 | `744375c1ccb04469` |
| `ft_run/kaggle_dong6/dong6/sinh_khoi_cum_f0_p11_cum.jsonl` | 567.391 | `81ba2eaeec924d44` |
| `ft_run/kaggle_dong6/dong6/sinh_khoi_pair_f0_p11_pair.jsonl` | 555.443 | `bbc0c482ad6d7773` |
| `ft_run/kaggle_dong6/dong6/sinh_pair_f0_p11_pair.jsonl` | 245.844 | `c6d59574945c0b36` |
| `ft_run/kaggle_dong6/nlp-dong6.log` | 5.122 | `628de732f4535236` |
