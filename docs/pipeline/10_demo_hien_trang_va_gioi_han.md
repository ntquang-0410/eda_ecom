# 10. Demo, hiện trạng (16/11/2026), việc còn mở và giới hạn

[← 09](09_fine_tune_va_danh_gia.md) · [Tổng quan](00_TONG_QUAN.md)

Nguồn:
- [`docs/TONG_HOP_KET_QUA_NHAT_ANH.md`](../TONG_HOP_KET_QUA_NHAT_ANH.md) mục 9–14;
- `demo/CHAY_DEMO.md`;
- phiên [F1](../ket_qua_kaggle/F1_cong_nen_de_chay_demo_tren_laptop_khong_phai_kag.md), [F2](../ket_qua_kaggle/F2_may_chu_dich_demo_tren_kaggle_thu_13_trang_1688.md), [F3](../ket_qua_kaggle/F3_qwen3_8b_theo_cong_thuc_chinh_khong_chay_duoc_bi.md).

## 1. Demo: extension Chrome + máy chủ dịch (14 – 15/11)

```
Trang detail.1688.com ── content script ── background ── HTTP ── máy chủ dịch (Flask) ── Qwen3-4B + LoRA pair
```

- Mô hình demo là **Qwen3-4B + `pair_cuoi`**.
- Phạm vi dịch: **title + bảng thuộc tính**, đúng thứ mô hình đã học. Không dịch ảnh, đánh giá, tên phân loại SKU.
- Máy chủ là lớp duy nhất giữ lời dẫn. Lời dẫn import từ `scripts/ft_train/dung_mau.py` nên giống lúc huấn luyện.

**Cổng nén để chạy trên laptop** (RTX 3050 4 GB; adapter `pair_f0_p11`; 150 sản phẩm fold 0, zh→vi; so với fp16 trên Kaggle). Ngưỡng đặt trước khi đo: chrF tụt ≤ 1,0; lệch dòng ≤ 2%; không có bản rỗng.

| Bản | chrF câu TB | Tụt so với fp16 | Khối lệch dòng | Rỗng | Title (giây) | Khối ~22 dòng (giây) | Kết luận |
|---|---|---|---|---|---|---|---|
| fp16 Kaggle (mốc) | 79.76 | – | 0/72 | 0% | – | – | – |
| Q4_K_M + LoRA | 76.95 | −2.81 | 1/72 | 0.6% | 0.99 | 12.09 | loại |
| **Q5_K_M + LoRA** | 79.62 | −0.14 | 0/72 | 0% | 1.09 | 14.08 | **đạt** |

**Thử trên 13 trang 1688 thật** (15/11, máy chủ Kaggle T4 fp16 qua đường hầm):
- 13/13 trang đạt kiểm cấu trúc. Số dòng thuộc tính đọc từ DOM khớp JSON nhúng trong trang.
- Đọc tay 8 trang (không có tham chiếu): khoảng 95% dòng đúng và tự nhiên. Lỗi gặp:
  - tự thêm "(Thượng Hải)";
  - 秋葵 dịch thành "khoai môn";
  - 树脂 dịch thành "Nhựa polyester";
  - rụng chữ "外贸";
  - phiên âm tên thương hiệu.
- Có bộ đệm dịch sẵn 13 trang và chế độ chỉ dùng bộ đệm (không cần GPU). Đã chọn 5 trang để demo. Chưa thử đầy đủ trong Chrome thật.

## 2. Qwen3-8B theo công thức chính: chưa chạy được

- Ngày 13/11, hai lần chạy khói ở tài khoản 2 bị cấp Tesla T4 vì mất RTX Pro 6000:
  - lần 1 hết bộ nhớ khi nạp mô hình;
  - lần 2 dừng ở chốt chặn VRAM.
- Lần chạy khói trước đó đo được đỉnh bộ nhớ 25,2 GB, vượt một card 24 GB. Kaggle "4×L4" là bốn card 24 GB riêng, nên không đủ nếu không chia mô hình.
- Notebook `trinhnhatanh/notebook59a0de1490` (gắn cuộc thi Gemma 4) chưa chạy gì. Chờ Nhật Anh chọn accelerator và xem điều khoản cuộc thi.

## 3. Đã chốt và còn mở

**Đã chốt:**
- lr 2e-4, rank 64, 2 epoch, LoRA mọi lớp tuyến tính.
- Tỉ trọng title 10,74% (cụm) và 10,86% (pair).
- Loại 6.386 title có cờ khỏi bản cuối.
- Test đo đúng một lần.
- Bộ thước đo cho năm mô hình: chrF + BLEU/SacreBLEU/TER, tách title và mô tả, cả hai hướng.

**Còn mở (chỉ làm khi có lệnh):**
- 8B theo công thức chính cho bản cụm: cần GPU hơn 24 GB một card.
- Extension dùng pair hay cụm. Hiện demo dùng `pair_cuoi`.
- Đo test cho mô hình được chọn; năm mô hình chưa đo trên test.
- B9: LLM chấm so cặp mù bản fine-tune với zero-shot, để biết chất lượng thật ngoài chrF.
- Sửa title thật (khoảng 1.729 title zh→vi bị cờ) rồi huấn luyện bản dữ liệu v2.
- Người song ngữ kiểm một tập nhỏ.

## 4. Giới hạn phải ghi khi trích số

1. **Tham chiếu là bản dịch máy** (Gemma/Qwen qua Judge).
   - Tham chiếu title có lỗi lõi khoảng 5–14%, nên chrF title chỉ khoảng 50–62.
   - Qwen3-14B (tạo dữ liệu) **cùng họ** với Qwen3-4B (được fine-tune).
2. Khoảng tin cậy chỉ tính dao động do chọn mẫu. Vòng sàng lọc có 2 seed; các lần chạy 4B chỉ 1 seed.
3. Năm mô hình chỉ đo trên eval fold 0. Fold 0 cũng là nơi phát hiện tín hiệu "dòng mới", nên chỉ fold 1 và 2 là kiểm độc lập cho tỉ trọng title.
4. Hai mô hình cổ điển chỉ chạy một lần, học trên dòng của bản pair. EcomGPT chạy ít bước hơn vì tokenizer BLOOM cho ít token hơn.
5. Độ phủ Judge ở bản pair thấp (khoảng 14,5% đơn vị tranh chấp được chấm). Bộ dữ liệu **không phải tập vàng**.
6. Mã `kaggle_train.py` bản trước 05/11 23:06 (dùng cho B6 và Kaggle bản 25) không còn bản lưu riêng.

## 5. Đặt cạnh hướng của Quang

| | Hướng Nhật Anh | Hướng Quang (`main`) |
|---|---|---|
| Dữ liệu | bản dịch Gemma/Qwen chọn bằng Judge DeepSeek; dạng cụm và pair | làm sạch CADO, lọc bitext; bộ A/B/C, Tiki dịch ngược bằng Qwen3 |
| Tập đo | eval fold 0 (1.200 mẫu) và test khóa 2.558 sản phẩm | eval_F 1.500 dòng |
| Thước chính | chrF++ (kèm BLEU, SacreBLEU, TER) | COMET-da (1688), COMET-QE (Tiki), kèm BLEU và chrF |

Điểm hai bên **không so thẳng được**. Muốn chọn theo điểm thì cách công bằng là cho mô hình cuối của mỗi bên dịch cùng một tập, rồi chấm bằng cùng một thước đo.
