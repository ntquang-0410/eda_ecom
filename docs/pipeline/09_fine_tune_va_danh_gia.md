# 09. Fine-tune Qwen3-4B và đánh giá (cụm và pair)

[← 08](08_chia_tap_va_dung_mau.md) · [Tổng quan](00_TONG_QUAN.md) · [10 →](10_demo_hien_trang_va_gioi_han.md)

Thời gian: 05 – 14/11/2026, Kaggle (RTX Pro 6000 đến 13/11, sau đó chỉ còn T4). Nguồn:
- [`docs/TONG_HOP_KET_QUA_NHAT_ANH.md`](../TONG_HOP_KET_QUA_NHAT_ANH.md) mục 3–8;
- [`docs/ket_qua_kaggle/`](../ket_qua_kaggle/INDEX.md), phiên B1–B9, C1–C6, D0–D6, E1–E3;
- [`docs/PHU_LUC_NHAT_KY_CHAY.md`](../PHU_LUC_NHAT_KY_CHAY.md);
- `docs/TRONG_SO_TITLE_MO_TA_DA_CHOT.md`, `docs/QUOTA_KET_LUAN.md`.

**Lưu ý chung:**
- Tham chiếu là bản dịch máy (file 07). chrF đo độ giống tham chiếu đó, không phải chất lượng tuyệt đối.
- Khoảng tin cậy [95%] là bootstrap theo sản phẩm. Chúng chỉ tính dao động do chọn mẫu, không tính dao động khi huấn luyện lại.

## 1. Cấu hình cố định (công thức chính)

- **Mô hình nền:** `Qwen/Qwen3-4B` revision `1cfa9a72`, bf16, tắt suy luận, sinh greedy.
- **LoRA:** trên mọi lớp tuyến tính (q, k, v, o, gate, up, down); rank 64, alpha 128, dropout 0,05.
- **Tối ưu:** lr 2e-4, AdamW, clip 1,0, warmup 3% rồi giảm cosine; **2 epoch**.
- **Lô:** `max_len` 2.048; 4.096 token mỗi lô nhỏ; seed 42.
- **Đánh giá:** sinh greedy trên 1.000 mẫu cố định của fold kiểm. Thước đo: chrF++ cho zh→vi, chrF ký tự cho vi→zh.
- **Mỗi bản (cụm, pair) có một adapter riêng**, dịch cả title lẫn mô tả, cả hai hướng; chỉ đổi lời dẫn.
- Tổng giờ GPU các phiên chính: 32,02 giờ.

## 2. Dò siêu tham số (B6, fold 0, bản cụm) [ĐO, phiên B1–B2]

| Cấu hình | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung | Title | Mô tả |
|---|---|---|---|---|---|
| lr 5e-5, r16 | 0.4090 | – | – | – | – |
| lr 1e-4, r16 | 0.3900 | – | – | – | – |
| lr 2e-4, r16 | 0.3770 | – | – | – | – |
| lr 5e-5, r64 | 0.3808 | – | – | – | – |
| lr 1e-4, r64 | 0.3680 | 0.3587 | 75.51 | 60.21 | 90.81 |
| **lr 2e-4, r64 (chọn)** | 0.3703 | **0.3512** | **75.76** | 60.22 | 91.30 |

- lr 2e-4 so với 1e-4: +0,33 [−0,12; +0,81], tức hòa. Chọn 2e-4 vì loss thấp hơn.
- LoRA chỉ trên attention kém LoRA mọi lớp: −2,20 [−2,75; −1,65].

## 3. Bộ trọng số đóng góp title : mô tả (đã chốt) [ĐO, phiên B3–B7]

**"Trọng số" ở đây là hệ số nhân vào loss, không phải trọng số mô hình.** Mẫu mô tả nhân 1, mẫu title nhân `w_title`.

- **Mốc cũ 1 : 1 theo mẫu:** title chiếm khoảng 50% loss, dù title chỉ khoảng 11% token đích.
- **Mức mới:** đưa title về đúng tỉ trọng token đích.

| | Cụm | Pair |
|---|---|---|
| % title trong loss (theo token đích) | **10,74%** | **10,86%** |
| `w_title` fold 0 | 0,12029 | 0,11818 |
| `w_title` bản cuối (sau khi loại title có cờ) | **0,17247** | **0,16915** |

Kiểm trên 3 fold (mức mới trừ mốc 1 : 1):

| Bản | Fold | chrF chung | Dòng mô tả mới zh→vi [95%] | Title hai hướng |
|---|---|---|---|---|
| cụm | 0 | +0.38 | +1.43 [+0.54; +2.32] | +0.14 [−0.99; +1.34] |
| cụm | 1 | +0.40 | +1.95 [+1.13; +2.82] | +0.33 [−0.64; +1.30] |
| cụm | 2 | +1.15 | +2.69 [+1.76; +3.69] | +0.33 [−0.46; +1.16] |
| pair | 0 | −0.08 (câu) | +1.36 [+0.57; +2.19] | **−0.96 [−1.81; −0.15]** |
| pair | 1 | +0.28 (câu) | +2.10 [+1.23; +2.94] | +0.01 [−0.96; +1.00] |
| pair | 2 | +0.47 (câu) | +1.58 [+0.72; +2.44] | +0.34 [−0.65; +1.25] |

- "Dòng mới" là dòng mô tả chưa từng xuất hiện nguyên văn trong train.
- Cụm đạt luật khóa trước. Pair đạt luật 3 fold riêng của pair (khóa ở Prompt 804). Lợi chủ yếu nằm ở dòng mới zh→vi.
- Mức 75% title thua ở pair (−0,58 [−0,99; −0,19]) và hòa ở cụm, nên bỏ.
- Kiểm lại lr 1e-4 ở mức 11%: không đạt, giữ 2e-4.
- **Giới hạn:** chỉ thử 3 mức (50%, 11%, 75%). Vì vậy 11% là "tốt hơn 50%", chưa chứng minh là tối ưu.

**Kiểm chéo bằng khảo sát lấy mẫu theo hạn ngạch** (quota, phiên C1–C4: rank 16, một lượt, 2 seed, eval 1.200 mẫu fold 0):

| Bộ | Cấu hình | % title | Chính | Macro | Title zh→vi | Mô tả zh→vi |
|---|---|---|---|---|---|---|
| cụm | **natural (chốt)** | 10.8 | **86.81** | 71.87 | 54.19 | 90.44 |
| cụm | balanced | 50 | 85.04 | 73.11 | 57.98 | 88.04 |
| cụm | title_heavy | 80 | 81.84 | 72.43 | 59.12 | 84.36 |
| pair | **natural (chốt)** | 10.3 | **79.98** | 68.52 | 49.98 | 83.31 |
| pair | balanced | 50 | 75.44 | 68.60 | 54.00 | 77.82 |
| pair | attr_heavy | 15 | 78.86 | 68.19 | 51.38 | 81.92 |

- Tăng tỉ trọng title thì title tăng và mô tả giảm, ở cả hai seed.
- **w hòa vốn** là trọng số title tối thiểu để cấu hình kia thắng natural: cụm balanced 0,39; cụm title_heavy 0,55; pair balanced 0,58; pair attr_heavy 0,50.
- Điểm "Chính" phải tính lại bằng 0,1 × title + 0,9 × mô tả. Trường `diem_chinh` trong báo cáo v55 dùng sai trọng số.

## 4. Đo test khóa MỘT lần (07/11, Kaggle bản 41) [ĐO, phiên B8]

- So adapter fold 0 (`cum_f0_p11`, `pair_f0_p11`, học fold 1–4) với zero-shot, trên 5.116 mục / 2.558 sản phẩm mỗi nhánh.
- Luật khóa trước khi đo: thắng khi cận dưới 95% của chênh lớn hơn 0.

| Nhánh | Hướng | Zero-shot | Fine-tune | Chênh [95%] |
|---|---|---|---|---|
| cụm | vi→zh | 41.49 | 75.63 | +34.14 [+33.30; +34.95] |
| cụm | zh→vi | 47.22 | 77.36 | +30.14 [+29.66; +30.63] |
| pair | vi→zh | 43.92 | 75.41 | +31.50 [+30.73; +32.24] |
| pair | zh→vi | 46.34 | 77.65 | +31.31 [+30.81; +31.83] |

- Kiểm riêng zh→vi của bản cụm: tỉ lệ sai chữ Việt giảm từ 0,308 xuống 0,058; rò lời dẫn giảm từ 0,326 xuống 0.
- Chênh khoảng 30 điểm **bị phóng đại một phần**, vì tham chiếu là bản dịch máy cùng kiểu với dữ liệu huấn luyện. Không nói "dịch tốt 77%".

## 5. Bản cuối (07/11, Kaggle bản 43) [ĐO, phiên B9]

- Huấn luyện trên **toàn pool 23.017 sản phẩm**, 2 epoch, **loại 6.386 mẫu title có cờ**. `w_title` tính lại để giữ đúng tỉ trọng.

| | `cum_cuoi` | `pair_cuoi` |
|---|---|---|
| Mẫu / token / bước | 39.734 / 15,65M / 1.990 | 39.724 / 14,83M / 2.008 |
| Loss ep2 | 0,1599 | 0,1655 |
| sha256 adapter | `fc8481ad…5b131` | `7fd7f157…2319b` |

- Không còn dữ liệu giữ lại, nên **không có điểm chất lượng**, và **không đo trên test** (cố ý). Chỉ sinh thử 100 mẫu để bắt lỗi hỏng.
- Lưu ở Kaggle `trinhnhatanh/ft-qwen3-4b-cuoi-luu` (riêng tư) và trên máy `data/processed/ft_run/ban_cuoi/cuoi/`. Mỗi adapter nặng 528.550.256 byte.
- Bảng đầy đủ các adapter (TK1/TK2, fold 0, kho lưu) ở `TONG_HOP` mục 10.

## 6. So năm mô hình (eval fold 0, không đụng test) [ĐO, phiên D0–D6]

Cách đo:
- Huấn luyện trên fold 1–4, đo trên 1.200 mẫu cố định của fold 0.
- **Ba LLM** dùng cấu hình **sàng lọc**: LoRA r16, lr 2e-4, lấy mẫu natural 4M token, một lượt, trung bình 2 seed. Cấu hình này không phải công thức chính.
- **Hai mô hình cổ điển** học từ đầu trên dòng của bản pair:
  - OpenNMT: Transformer nhỏ;
  - Moses: phrase-based, chữ Hán tách từng chữ.

**Pair (1.198 mẫu)**

| Mô hình | Title | Mô tả | Chính | Macro | Chênh chính (câu) so với 4B |
|---|---|---|---|---|---|
| Qwen3-4B | 49.98 | 83.31 | 79.98 | 68.52 | mốc |
| Qwen3-8B | 52.03 | 79.86 | 77.08 | 66.61 | +0.44 [−0.24; +1.14] |
| EcomGPT-7B | 38.45 | 71.45 | 68.15 | 56.78 | −4.81 [−5.95; −3.79] |
| OpenNMT (học đủ) | 48.27 | 77.28 | 74.38 | 54.89 | +0.39 [−1.09; +1.90] |
| Moses (học đủ) | 54.93 | 83.65 | 80.78 | 68.08 | +1.86 [+0.67; +3.04] |
| Moses (khớp 4M token) | 51.97 | 82.34 | 79.30 | 65.36 | +0.05 [−1.14; +1.23] |

**Cụm (1.200 mẫu)**

| Mô hình | Title | Mô tả | Chính | Macro | Chênh chính (câu) so với 4B |
|---|---|---|---|---|---|
| Qwen3-4B | 54.19 | 90.44 | 86.81 | 71.87 | mốc |
| Qwen3-8B | 56.52 | 91.34 | 87.86 | 73.92 | **+1.03 [+0.71; +1.36]** |
| EcomGPT-7B | 45.85 | 82.49 | 78.83 | 64.44 | −6.42 [−7.36; −5.55] |
| OpenNMT (học đủ) | 47.01 | 84.34 | 80.61 | 58.72 | −5.57 [−6.42; −4.85] |
| Moses (học đủ) | 54.93 | 88.48 | 85.12 | 69.77 | −1.51 [−2.06; −1.02] |
| Moses (khớp 4M token) | 51.97 | 85.71 | 82.33 | 66.48 | −4.17 [−4.79; −3.60] |

- **Cụm:** 8B hơn 4B rõ; 4B hơn Moses; EcomGPT và OpenNMT kém xa.
- **Pair:** 4B, 8B, OpenNMT và Moses khớp 4M token nằm trong khoảng nhiễu; Moses học đủ dữ liệu nhỉnh hơn.
  - Khoảng 87% dòng thuộc tính trong eval pair đã có nguyên văn trong train. Vì vậy Moses mạnh ở pair, và mất lợi thế khi phải dịch cả khối (cụm).
- Có thêm BLEU/SacreBLEU/TER theo cách đo của bài VLSP: `TONG_HOP` mục 6.2.

## 7. Thử chéo dạng dữ liệu và công thức chính trên eval fold 0 [ĐO, phiên E1–E3]

- Hai adapter `cum_f0_p11`, `pair_f0_p11` được huấn luyện lại ở tài khoản 2 với cùng cấu hình (E1).

Thử chéo (600 sản phẩm fold 0, mô tả zh→vi):

| So sánh | Chênh chrF câu [95%] |
|---|---|
| pair + adapter pair so với cụm + adapter cụm | +0.04 [−0.16; +0.23], hòa |
| adapter cụm nhận đầu vào dạng pair (sai dạng) | −1.01 [−1.39; −0.67] |
| adapter pair nhận đầu vào dạng cụm (sai dạng) | −0.47 [−0.68; −0.28] |
| adapter so với zero-shot | +37.7 đến +37.9 |

Hai dạng **ngang nhau** khi dùng đúng adapter của dạng đó. Chọn pair hay cụm cho extension là vì tiện, không vì điểm.

Công thức chính (rank 64, 2 epoch, học fold 1–4) trên cùng eval với năm mô hình, chạy T4 fp16:

| Bộ | chrF chính (corpus) | Chênh chrF câu so với 4B sàng lọc |
|---|---|---|
| Cụm | 90.38 | +3.36 [+2.96; +3.78] |
| Pair | 83.28 | +3.95 [+2.98; +4.99] |

Dòng này khác lời dẫn và cách dịch (pair dịch cả khối rồi tách dòng), nên **không xếp hạng chung** với năm dòng sàng lọc ở mục 6.
