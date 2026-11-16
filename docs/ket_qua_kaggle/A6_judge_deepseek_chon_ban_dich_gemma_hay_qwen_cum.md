# A6. Judge DeepSeek chọn bản dịch Gemma hay Qwen (cụm và pair)

[← Mục lục](INDEX.md) · [← A5](A5_loc_luat_cho_fine_tune_tren_may.md) · [A7 →](A7_chia_tap_theo_san_pham_kiem_ro_ri_gan_co_title_t.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | A6 |
| Giai đoạn | A. Dữ liệu: dịch, lọc, Judge |
| Notebook Kaggle | – (`scripts/judge/`: `so_cap_du_lieu.py`, `so_cap_chon.py`, `ghep_bo.py`, `chay_lon.ps1`) |
| Tài khoản | – (gọi API DeepSeek, không chạy trên Kaggle; thí điểm: Nhật Anh trên máy; lần chạy đầy đủ: Huy) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | thí điểm 29–31/10/2026; bản phát hành tải về 03/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/judge`, `data/processed/so_cap`, `data/processed/bo_chon_hf` |

## Mục đích

Với mỗi mảnh mà bản Gemma và bản Qwen khác nhau (title nguyên câu; mô tả cụm theo từng cặp tên–giá trị; mô tả pair theo từng tên hoặc giá trị), cho DeepSeek (`deepseek-v4.1-flash`) chấm mù theo rubric rồi chọn một bản. Chỉ thay bản hiện có khi hai lượt hoán vị cùng chọn một mô hình; mục khai tương đương, không nhất quán hoặc chưa chấm thì giữ hiện trạng; chặn bản mới nếu gây sót chữ Hán (1688) hoặc thêm chữ Việt (Tiki).

## 1. Thí điểm trên máy (29–31/10/2026, `data/processed/judge/`, `data/processed/so_cap/`)

Mỗi dòng là một lần chạy thử trên mẫu nhỏ để chọn giao thức (một hay hai lượt hoán vị, có ngữ cảnh hay không, quy tắc v1 hay v2, gom nhiều mục mỗi lời gọi). Số đọc từ `tom_tat_*.json`.

| File | Lần chạy | Mô hình chấm | Phiên bản lời nhắc | Chế độ | Mục xong | Lỗi | Token vào | Token ra | Giây |
|---|---|---|---|---|---|---|---|---|---|
| `judge/tom_tat_nt_3.json` | nt_3 | deepseek-v4.1-flash | – | – | 3 | 0 | 4.676 | 624 | 4 |
| `judge/tom_tat_nv_3.json` | nv_3 | deepseek-ai/deepseek-v4.1-flash | – | – | 3 | 0 | 4.751 | 242 | 576 |
| `judge/tom_tat_nv_8.json` | nv_8 | deepseek-ai/deepseek-v4.1-flash | – | – | 0 | 8 | 0 | 0 | 1.218 |
| `judge/tom_tat_t3_low.json` | t3_low | deepseek-v4.1-flash | – | – | 0 | 0 | 0 | 0 | 1 |
| `judge/tom_tat_thu_300.json` | thu_300 | deepseek-v4.1-flash | – | – | 44 | 6 | 97.315 | 111.808 | 179 |
| `judge/tom_tat_thu_300_nothink.json` | thu_300_nothink | deepseek-v4.1-flash | – | – | 0 | 0 | 0 | 0 | 1 |
| `judge/tom_tat_thu_5.json` | thu_5 | deepseek-v4.1-flash | – | – | 5 | 0 | 7.506 | 6.408 | 16 |
| `so_cap/tom_tat_title_20.json` | title_20 | deepseek-v4.1-flash | so_cap_v1 | – | 20 | 0 | 19.180 | 1.930 | 8 |
| `so_cap/tom_tat_cum_20.json` | cum_20 | deepseek-v4.1-flash | so_cap_v1 | – | 20 | 0 | 20.055 | 1.803 | 9 |
| `so_cap/tom_tat_pair_20.json` | pair_20 | deepseek-v4.1-flash | so_cap_v1 | – | 20 | 0 | 16.650 | 1.842 | 8 |
| `so_cap/tom_tat_cumctx_40.json` | cumctx_40 | deepseek-v4.1-flash | so_cap_v1 | – | 40 | 0 | 70.735 | 3.620 | 15 |
| `so_cap/tom_tat_cumkhong_40.json` | cumkhong_40 | deepseek-v4.1-flash | so_cap_v1 | – | 39 | 0 | 37.075 | 3.583 | 15 |
| `so_cap/tom_tat_cumkhong_40_lap.json` | cumkhong_40_lap | deepseek-v4.1-flash | so_cap_v1 | – | 0 | 0 | 0 | 0 | 0 |
| `so_cap/tom_tat_title_100.json` | title_100 | deepseek-v4.1-flash | so_cap_v1 | – | 100 | 0 | 95.962 | 10.033 | 20 |
| `so_cap/tom_tat_cum_100.json` | cum_100 | deepseek-v4.1-flash | so_cap_v1 | – | 100 | 0 | 96.749 | 8.947 | 19 |
| `so_cap/tom_tat_pair_100.json` | pair_100 | deepseek-v4.1-flash | so_cap_v1 | – | 100 | 0 | 87.378 | 8.976 | 19 |
| `so_cap/tom_tat_title_100_t2.json` | title_100_t2 | deepseek-v4.1-flash | so_cap_v1 | – | 28 | 0 | 41.668 | 2.720 | 9 |
| `so_cap/tom_tat_pair_100_t2.json` | pair_100_t2 | deepseek-v4.1-flash | so_cap_v1 | – | 38 | 0 | 61.536 | 3.439 | 8 |
| `so_cap/tom_tat_qt_title_100.json` | qt_title_100 | deepseek-v4.1-flash | so_cap_v2 | hai_luot | 100 | 0 | 170.562 | 21.049 | 28 |
| `so_cap/tom_tat_qt_cum_100.json` | qt_cum_100 | deepseek-v4.1-flash | so_cap_v2 | hai_luot | 100 | 0 | 171.349 | 16.091 | 25 |
| `so_cap/tom_tat_qt_pair_100.json` | qt_pair_100 | deepseek-v4.1-flash | so_cap_v2 | hai_luot | 100 | 0 | 161.978 | 15.214 | 24 |
| `so_cap/tom_tat_m2_v1_title_150.json` | m2_v1_title_150 | deepseek-v4.1-flash | so_cap_v1 | hai_luot | 150 | 0 | 143.280 | 14.825 | 30 |
| `so_cap/tom_tat_m2_v2_title_150.json` | m2_v2_title_150 | deepseek-v4.1-flash | so_cap_v2 | hai_luot | 150 | 0 | 255.180 | 32.356 | 44 |
| `so_cap/tom_tat_m2_v1_cum_ctx_150.json` | m2_v1_cum_ctx_150 | deepseek-v4.1-flash | so_cap_v1 | hai_luot | 150 | 0 | 266.421 | 13.649 | 30 |
| `so_cap/tom_tat_m2_v2_cum_ctx_150.json` | m2_v2_cum_ctx_150 | deepseek-v4.1-flash | so_cap_v2 | hai_luot | 150 | 0 | 378.021 | 21.955 | 36 |
| `so_cap/tom_tat_m2_v1_pair_150.json` | m2_v1_pair_150 | deepseek-v4.1-flash | so_cap_v1 | hai_luot | 150 | 0 | 131.788 | 13.189 | 28 |
| `so_cap/tom_tat_m2_v2_pair_150.json` | m2_v2_pair_150 | deepseek-v4.1-flash | so_cap_v2 | hai_luot | 150 | 0 | 243.688 | 22.982 | 35 |
| `so_cap/tom_tat_b_title_150.json` | b_title_150 | deepseek-v4.1-flash | so_cap_v2 | mot_luot | 150 | 0 | 29.775 | 6.141 | 7 |
| `so_cap/tom_tat_b_cum_ctx_150.json` | b_cum_ctx_150 | deepseek-v4.1-flash | so_cap_v2 | mot_luot | 150 | 0 | 90.318 | 5.954 | 6 |
| `so_cap/tom_tat_b_pair_150.json` | b_pair_150 | deepseek-v4.1-flash | so_cap_v2 | mot_luot | 150 | 0 | 24.110 | 5.869 | 6 |

Kết quả so cặp mù hai thứ tự trên mẫu 100 mục mỗi loại (`bao_cao_*_100.json`, quy tắc v1): `gemma`/`qwen` = cả hai lượt cùng chọn; `khong_nhat_quan` = hai lượt chọn khác nhau.

| Loại | Mục | Gemma | Qwen | Hòa thật | Không nhất quán | Chọn vị trí 1 | Gemma trong số có người thắng | Bản dài hơn thắng | Trùng CometKiwi |
|---|---|---|---|---|---|---|---|---|---|
| title | 100 | 67 | 5 | 0 | 28 | 62.8% | 93.1% | 57.7% | 61.1% |
| cum | 100 | 62 | 9 | 1 | 28 | 62.6% | 87.3% | 60.0% | – |
| pair | 100 | 46 | 5 | 11 | 38 | 67.1% | 90.2% | 51.2% | 52.9% |

Thử gom 15 mục mỗi lời gọi (`so_sanh_gom.json`, 450 mục, quy tắc v2) so với một mục mỗi lời gọi:

| Chỉ số | Một mục / lời gọi | Gom |
|---|---|---|
| Token mỗi mục | 1.060 | 360 |
| Gemma / Qwen | 313 / 137 | 314 / 136 |
| Chọn vị trí 1 | 73.1% | 73.8% |
| Khai tương đương | 36.9% | 39.8% |
| Trùng kết quả trên mục bền vững hai thứ tự | – | 223/247 (90.3%) |

## 2. Kết quả trên bản phát hành (HF `ntquang0410/zh-vie_ecom`, `data/LLM/bilingual_judged`, commit `82aa9c9`)

Đọc từ bốn file đã tải về `data/processed/bo_chon_hf/` (sha256 khớp `manifest.json`). Theo README của bản phát hành: DeepSeek chấm **4.439/4.446** mục cụm (7 mục bị nhà cung cấp chặn được rà riêng: 2 chọn Gemma, 4 tương đương, 1 chưa chắc); bản pair chấm **5.000/34.534** đơn vị tranh chấp dùng nhiều nhất (khoảng 84% mảnh), đơn vị chưa chấm giữ bản trước.

Nguồn của nội dung cuối (`bo_*_chon_deepseek.parquet`). `judge_chon_title` là mô hình tạo bản title được giữ, không nhất thiết là lựa chọn trực tiếp của DeepSeek; `quy_tac` = mảnh khóa bằng từ điển/luật; `chua_ro` = chưa chấm hoặc không quyết, giữ hiện trạng. Mảnh cụm là một cặp tên–giá trị, mảnh pair là tên hoặc giá trị riêng, nên không so số đếm giữa hai bộ.

| Bộ | Sản phẩm | Theo nguồn | Title Gemma | Title Qwen | Mảnh mô tả Gemma | Mảnh mô tả Qwen | Mảnh quy tắc | Mảnh chưa rõ |
|---|---|---|---|---|---|---|---|---|
| cum | 25.575 | `{"1688": 14837, "tiki": 10738}` | 19.261 | 6.314 | 217.818 | 493 | 140.080 | 83.451 |
| pair | 25.575 | `{"1688": 14837, "tiki": 10738}` | 19.261 | 6.314 | 316.057 | 12.756 | 237.909 | 305.312 |

Trạng thái Judge theo file đánh dấu (`bo_*_danh_dau.parquet`, cộng trên mọi sản phẩm):

| Bộ | Nhãn title | Title đã đổi | Mảnh tranh chấp | Judge tầng 1 | Judge tầng 2 | Khai tương đương | Không nhất quán | Chưa chấm | Mảnh đã đổi | Áp Gemma | Áp Qwen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cum | `{"judge_t1": 20353, "judge_tuong_duong": 5021, "khong_tranh_chap": 201}` | 7.604 | 17.170 | 9.755 | 0 | 7.408 | 0 | 7 | 615 | 9.110 | 642 |
| pair | `{"judge_t1": 20353, "judge_tuong_duong": 5021, "khong_tranh_chap": 201}` | 7.604 | 195.311 | 59.470 | 0 | 104.331 | 0 | 31.510 | 9.054 | – | – |

## Kết luận của phiên

- Bản cuối của hai bộ (cụm, pair) là dữ liệu huấn luyện của mọi phiên B–F (HF commit `82aa9c9`, hash ở `docs/TONG_HOP_KET_QUA_NHAT_ANH.md` mục 2.5).
- Gemma thắng phần lớn mục có người thắng ở thí điểm; Judge gần như tất định (chạy lặp giống khoảng 97%), nên mục không nhất quán chỉ gỡ được khi thêm ngữ cảnh (tầng 2).
- Ở bản phát hành, Judge chỉ đổi một phần nhỏ nội dung so với bản đầu vào (cột `Title đã đổi`, `Mảnh đã đổi`); phần lớn mảnh giữ hiện trạng vì tương đương, chưa chấm hoặc là mảnh luật. Cột `Judge tầng 2` bằng 0 nghĩa là lần chạy đầy đủ không dùng tầng 2.

## Ghi chú, giới hạn, sự cố

- **Phân công:** Nhật Anh thiết kế giao thức, viết mã (`scripts/judge/`), chạy các lần thí điểm (phần 1) và soạn `HUONG_DAN_JUDGE.md` + gói dữ liệu riêng; **Huy chạy lần Judge đầy đủ** theo hướng dẫn đó và đưa bản kết quả lên HF (phần 2, Nhật Anh xác nhận 16/11/2026). Output thô của lần chạy đầy đủ không nằm trên máy Nhật Anh; số ở phần 2 đọc từ bản phát hành.
- Các file truy vết chi tiết của lần chạy đầy đủ nằm ở revision HF `126a564f` (theo README bản phát hành), chưa tải về máy.
- Judge thiên vị vị trí 1 (khoảng 63–74% ở thí điểm) nên bắt buộc hỏi hai thứ tự; tham chiếu cuối vẫn là bản dịch máy, chưa người kiểm toàn bộ.
- Thư mục `so_cap/` còn các bảng mục `muc_*.parquet` (đầu vào của Judge) và kết quả thô `ket_qua_*.jsonl` của thí điểm.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `judge/tom_tat_nt_3.json` | 343 | `2a70813d6f0b3013` |
| `judge/tom_tat_nv_3.json` | 303 | `be20ab98e92aa0f8` |
| `judge/tom_tat_nv_8.json` | 296 | `a8c7bb404c97ded7` |
| `judge/tom_tat_t3_low.json` | 586 | `b381dc01a76341f8` |
| `judge/tom_tat_thu_300.json` | 302 | `42ca2ad5209093b4` |
| `judge/tom_tat_thu_300_nothink.json` | 595 | `39ac8db2e586aabf` |
| `judge/tom_tat_thu_5.json` | 292 | `96c0b39a709f5e90` |
| `so_cap/tom_tat_b_cum_ctx_150.json` | 354 | `178f15743ae9b464` |
| `so_cap/tom_tat_b_pair_150.json` | 364 | `8add7c35bc1d18aa` |
| `so_cap/tom_tat_b_title_150.json` | 364 | `0d61830d939f4a4f` |
| `so_cap/tom_tat_cum_100.json` | 311 | `12b45b399ca657b4` |
| `so_cap/tom_tat_cum_20.json` | 308 | `a28c56c1e3919a2c` |
| `so_cap/tom_tat_cumctx_40.json` | 312 | `c866d168a65b29fa` |
| `so_cap/tom_tat_cumkhong_40.json` | 314 | `ba7ca70b9b0f6175` |
| `so_cap/tom_tat_cumkhong_40_lap.json` | 307 | `3ea88776c2edbeca` |
| `so_cap/tom_tat_m2_v1_cum_ctx_150.json` | 347 | `617ec45737b7972b` |
| `so_cap/tom_tat_m2_v1_pair_150.json` | 344 | `d958d66c1a30c835` |
| `so_cap/tom_tat_m2_v1_title_150.json` | 345 | `6db93c329ddc7a55` |
| `so_cap/tom_tat_m2_v2_cum_ctx_150.json` | 347 | `63d937f332b9706f` |
| `so_cap/tom_tat_m2_v2_pair_150.json` | 344 | `e35e166c29f64764` |
| `so_cap/tom_tat_m2_v2_title_150.json` | 345 | `dd6596775e455c86` |
| `so_cap/tom_tat_pair_100.json` | 312 | `d7bac3de090a740d` |
| `so_cap/tom_tat_pair_100_t2.json` | 313 | `f45157219d1ecb2f` |
| `so_cap/tom_tat_pair_20.json` | 309 | `545e27486bd9975a` |
| `so_cap/tom_tat_qt_cum_100.json` | 340 | `e53b55cf296887a9` |
| `so_cap/tom_tat_qt_pair_100.json` | 341 | `80a9daef5c87aa08` |
| `so_cap/tom_tat_qt_title_100.json` | 342 | `327e1b9045a0c92a` |
| `so_cap/tom_tat_title_100.json` | 314 | `52b26b7214e18deb` |
| `so_cap/tom_tat_title_100_t2.json` | 314 | `9ea575252f208ccc` |
| `so_cap/tom_tat_title_20.json` | 309 | `fe41582aacbad4d9` |
| `so_cap/bao_cao_title_100.json` | 1.112 | `f524fbb181d57ccf` |
| `so_cap/bao_cao_cum_100.json` | 950 | `42f5d09238d05adc` |
| `so_cap/bao_cao_pair_100.json` | 1.089 | `4664d211f5a6299e` |
| `so_cap/so_sanh_gom.json` | 4.361 | `4c61fc180a72540d` |
| `so_cap/so_sanh_quy_tac.json` | 73.996 | `a293c3e65e5484ba` |
| `bo_chon_hf/README.md` | 2.771 | `c22dbe10437ac376` |
| `bo_chon_hf/manifest.json` | 1.529 | `4e9c148232dfd178` |
