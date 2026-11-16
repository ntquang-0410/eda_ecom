# D6. Tổng hợp năm mô hình (trên máy, không GPU)

[← Mục lục](INDEX.md) · [← D5](D5_moses_vong_2_hoc_du_du_lieu_va_khop_4m_token_kem.md) · [E1 →](E1_huan_luyen_lai_adapter_fold_0_o_tai_khoan_2_cum.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | D6 |
| Giai đoạn | D. So sánh năm mô hình (vòng sàng lọc) |
| Notebook Kaggle | – (`scripts/ft_train/tong_hop_5mh.py`) |
| Tài khoản | – (Qwen3-4B lấy từ C4, tài khoản 1; các mô hình khác tài khoản 2) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 12–14/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/tong_hop_5mh`, `data/processed/ft_run/tong_hop_5mh_cum` |

## Mục đích

Chấm chung bản dịch của C4 (Qwen3-4B) và D1–D5 bằng một bộ chấm (`cham_diem.py`, `cham_bleu.py`), bootstrap theo sản phẩm, thêm BLEU/SacreBLEU/TER theo cách đo của bài VLSP.

## Bảng năm mô hình

Eval fold 0. Pair: 1.198 mẫu (từng dòng); cụm: 1.200 mẫu. chrF corpus; `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi; `Macro` = trung bình đều hai hướng. Chênh so với Qwen3-4B (vòng sàng lọc) tính trên chrF câu trung bình, bootstrap theo sản phẩm (2.000 lần, cùng mẫu rút cho mọi mô hình). LLM là trung bình 2 seed.

| Bộ | Mô hình | Số lần chạy | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả | Chính (corpus) | Macro (corpus) | Chính (câu TB) | Chênh chính (câu) so với 4B [95%] | Chênh macro (câu) so với 4B [95%] | Sinh lặp (TB 4 nhóm) | Lệch chính giữa 2 seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair | Qwen3-4B | 2 | 49.98 | 83.31 | 54.20 | 86.58 | 79.98 | 68.52 | 90.03 | mốc | mốc | 0.62% | 1.63 |
| pair | Qwen3-8B | 2 | 52.03 | 79.86 | 56.70 | 77.84 | 77.08 | 66.61 | 90.48 | +0.44 [-0.24; +1.14] | +1.24 [+0.69; +1.82] | 0.75% | 0.26 |
| pair | EcomGPT-7B | 2 | 38.45 | 71.45 | 44.18 | 73.03 | 68.15 | 56.78 | 85.23 | -4.81 [-5.95; -3.79] | -6.89 [-7.61; -6.15] | 1.75% | 3.03 |
| pair | OpenNMT (tu dau, train day du) | 1 | 48.27 | 77.28 | 29.64 | 64.38 | 74.38 | 54.89 | 90.43 | +0.39 [-1.09; +1.90] | -8.33 [-9.40; -7.26] | 0.25% | – |
| pair | Moses (train day du) | 1 | 54.93 | 83.65 | 51.66 | 82.07 | 80.78 | 68.08 | 91.90 | +1.86 [+0.67; +3.04] | +0.75 [+0.01; +1.50] | 0.00% | – |
| pair | Moses (khop 4M token) | 1 | 51.97 | 82.34 | 47.58 | 79.55 | 79.30 | 65.36 | 90.10 | +0.05 [-1.14; +1.23] | -1.82 [-2.60; -1.05] | 0.00% | – |
| pair | Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16) | 1 | 62.37 | 85.60 | 60.95 | 91.03 | 83.28 | 74.99 | 93.99 | +3.95 [+2.98; +4.99] | +6.05 [+5.31; +6.79] | 0.00% | – |
| cum | Qwen3-4B | 2 | 54.19 | 90.44 | 56.33 | 86.54 | 86.81 | 71.87 | 87.65 | mốc | mốc | 0.42% | 0.20 |
| cum | Qwen3-8B | 2 | 56.52 | 91.34 | 58.84 | 88.98 | 87.86 | 73.92 | 88.68 | +1.03 [+0.71; +1.36] | +1.74 [+1.26; +2.28] | 0.38% | 0.03 |
| cum | EcomGPT-7B | 2 | 45.85 | 82.49 | 46.88 | 82.55 | 78.83 | 64.44 | 81.24 | -6.42 [-7.36; -5.55] | -6.34 [-6.98; -5.72] | 2.50% | 0.09 |
| cum | OpenNMT (tu dau, train day du) | 1 | 47.01 | 84.34 | 29.64 | 73.89 | 80.61 | 58.72 | 82.08 | -5.57 [-6.42; -4.85] | -12.42 [-13.29; -11.57] | 3.33% | – |
| cum | Moses (train day du) | 1 | 54.93 | 88.48 | 51.66 | 84.02 | 85.12 | 69.77 | 86.14 | -1.51 [-2.06; -1.02] | -2.04 [-2.69; -1.39] | 0.00% | – |
| cum | Moses (khop 4M token) | 1 | 51.97 | 85.71 | 47.58 | 80.68 | 82.33 | 66.48 | 83.48 | -4.17 [-4.79; -3.60] | -5.01 [-5.66; -4.38] | 0.08% | – |
| cum | Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16) | 1 | 62.48 | 93.48 | 60.35 | 90.58 | 90.38 | 76.72 | 91.02 | +3.36 [+2.96; +3.78] | +4.48 [+3.92; +5.06] | 0.08% | – |

SacreBLEU / TER (trung bình các lần chạy; sacrebleu 2.6.0, zh→vi `tok:13a`, vi→zh `tok:zh`, TER `tercom`; BLEU kiểu mteval-v13a trùng SacreBLEU):

| Bộ | Mô hình | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả |
|---|---|---|---|---|---|
| pair | Qwen3-4B | 25.70 / 65.42 | 76.04 / 20.18 | 35.68 / 56.77 | 84.86 / 9.97 |
| pair | Qwen3-8B | 27.70 / 63.05 | 72.78 / 22.35 | 39.29 / 50.52 | 83.00 / 13.87 |
| pair | EcomGPT-7B | 15.32 / 83.72 | 58.81 / 45.18 | 25.28 / 72.99 | 68.94 / 31.19 |
| pair | OpenNMT (tu dau, train day du) | 27.13 / 59.64 | 72.48 / 25.68 | 16.51 / 75.20 | 68.12 / 26.71 |
| pair | Moses (train day du) | 28.83 / 60.94 | 69.78 / 27.81 | 28.99 / 61.67 | 81.12 / 25.20 |
| pair | Moses (khop 4M token) | 24.81 / 64.66 | 69.73 / 28.92 | 20.68 / 72.03 | 77.85 / 27.08 |
| pair | Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16) | 41.95 / 45.50 | 78.68 / 15.93 | 45.36 / 42.06 | 90.56 / 5.55 |
| cum | Qwen3-4B | 31.35 / 59.12 | 86.73 / 12.30 | 39.04 / 48.14 | 87.58 / 9.87 |
| cum | Qwen3-8B | 33.10 / 54.30 | 87.78 / 11.16 | 43.94 / 46.44 | 89.67 / 7.58 |
| cum | EcomGPT-7B | 21.92 / 72.47 | 74.61 / 32.03 | 27.55 / 69.61 | 81.91 / 17.60 |
| cum | OpenNMT (tu dau, train day du) | 26.08 / 60.18 | 80.02 / 21.91 | 16.51 / 75.20 | 76.03 / 19.75 |
| cum | Moses (train day du) | 28.83 / 60.94 | 80.69 / 21.13 | 28.99 / 61.67 | 83.45 / 22.58 |
| cum | Moses (khop 4M token) | 24.81 / 64.66 | 77.78 / 24.19 | 20.68 / 72.03 | 79.97 / 25.49 |
| cum | Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16) | 42.73 / 45.12 | 90.50 / 8.37 | 45.44 / 44.03 | 91.21 / 6.64 |

## Kết luận của phiên

- Cụm: Qwen3-8B hơn 4B rõ; 4B hơn Moses; EcomGPT và OpenNMT kém xa.
- Pair: 4B, 8B, OpenNMT, Moses khớp 4M token nằm trong khoảng nhiễu; Moses học đủ nhỉnh hơn.
- Nhận xét đầy đủ: `docs/KE_HOACH_VA_KET_QUA_NAM_MO_HINH.md`; khoảng tin cậy SacreBLEU/TER: `tong_hop_5mh*/bang_metric_bai_bao.md` (số chép ở `docs/TONG_HOP_KET_QUA_NHAT_ANH.md` mục 6).

## Ghi chú, giới hạn, sự cố

- Dòng `Qwen3-4B cong thuc cuoi` đến từ phiên E3: khác lời dẫn và cách dịch nên không xếp hạng chung với năm dòng còn lại.
- Năm mô hình chưa đo trên tập test khóa.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/tong_hop_5mh/bang.md` | 1.023 | `aff97697eb1a9fbd` |
| `ft_run/tong_hop_5mh/bang_metric_bai_bao.md` | 5.229 | `6f76855f9ad8a0bc` |
| `ft_run/tong_hop_5mh/ket_qua.json` | 28.385 | `23f608da4e4aff14` |
| `ft_run/tong_hop_5mh_cum/bang.md` | 1.023 | `5f4401f673de9eae` |
| `ft_run/tong_hop_5mh_cum/bang_metric_bai_bao.md` | 5.230 | `7bd00ec7f874fd7e` |
| `ft_run/tong_hop_5mh_cum/ket_qua.json` | 28.581 | `5cf76498389ffd88` |
