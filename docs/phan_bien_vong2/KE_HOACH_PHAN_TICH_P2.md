# KẾ HOẠCH PHÂN TÍCH P2 (khóa trước khi xem số)

Ngày 19/11/2026. Người làm: Claude. Ba phân tích bổ sung trên các đầu ra GPU đã có (Kaggle, Colab), để trả lời đúng câu hỏi của biên bản vòng 2. Chỉ **eval fold 0**, CPU, không train, không sinh bản dịch mới, không đọc test khóa. Số thô ghi vào `data/processed/ft_run/phan_bien_v2/` (tệp mới, không ghi đè tệp cũ). sha256 của tệp này được ghi vào `WALKTHROUGH.md` **trước** khi chạy bất kỳ phép chấm nào của P2. Sau khi có số không sửa kế hoạch; muốn phân tích thêm thì ghi là "phân tích thêm sau khi xem số".

Đây là các phân tích **thêm** sau khi đã có điểm chrF corpus của từng việc (nên điểm corpus đã biết khi viết kế hoạch). Kế hoạch khóa **cách tính khoảng tin cậy và luật diễn giải**, không khóa kỳ vọng về chiều của kết quả.

## A. "OpenNMT hòa Qwen3-4B": tính lại cho 3 seed (biên bản 2.2(b), 2.3.3)

- **Chỉ số:** đúng chỉ số của D3: chrF **câu trung bình**, "chính" = 0,10 × title + 0,90 × mô tả (zh→vi) và "macro" = trung bình đều hai hướng, mỗi hướng trung bình đều title và mô tả. Bộ pair (eval pair 1.198 mẫu) và bộ cụm (eval cụm 1.200 mẫu).
- **So với:** Qwen3-4B, trung bình chrF câu của 2 seed (42, 123) từng mẫu, như `tong_hop_5mh.py`.
- **Bootstrap:** theo mã sản phẩm, 2.000 lần, `numpy.random.default_rng(7).multinomial`, **cùng một ma trận rút** cho mọi mô hình (đúng `tong_hop_5mh.main`).
- **Kiểm tái lập trước:** OpenNMT seed 42 một mình so với 4B phải ra đúng số trong `tong_hop_5mh{,_cum}/ket_qua.json` (pair chính +0,39 [−1,09; +1,90]; cụm chính −5,57 [−6,42; −4,85]); lệch quá 0,01 ở bất kỳ số nào thì dừng và không báo số P2.
- **Báo:** seed 42, 43, 44 từng seed, và OpenNMT **trung bình chrF câu theo mẫu của 3 seed** (như LLM trung bình 2 seed), chỉ số chính và macro, pair và cụm.
- **Luật diễn giải:**
  - Nếu khoảng 95% của chênh "chính" pair (trung bình 3 seed) chứa 0: câu "hòa" ở pair được giữ, kèm khoảng và miền dao động giữa 3 seed, và ghi rõ chỉ đúng với thước chrF câu trung bình.
  - Nếu không chứa 0: bỏ chữ "hòa", ghi chiều và độ lớn.
  - Nếu một seed riêng lẻ có khoảng không chứa 0 trong khi trung bình 3 seed chứa 0 (hoặc ngược lại): ghi là "kết luận phụ thuộc seed".
  - Cụm: ghi số như đo được, không có luật riêng.

## B. Khoảng tin cậy cho tỉ trọng title 25% và 35% so với 11% (biên bản 3.3.5)

- **Dữ liệu:** `kaggle_tlc_v2_tiep/tlc_cum_f0_p{25,35,11}/.../sinh_ep2.jsonl` (1.000 mẫu, 250 mỗi nhóm hướng × loại). Mẫu và tham chiếu dựng lại bằng chính `kaggle_train.chon_mau_sinh(pool_cum, pool_cum, fold 0, 250, seed 42)` trên `pool_cum.jsonl` (pool, không phải test) và chấm bằng `cham_diem.cham(..., tap_da_thay=...)` như `chay_b78`.
- **Kiểm tái lập trước:** `chrf_macro_huong` và bốn `chrf_corpus` theo nhóm phải khớp báo cáo Kaggle (`bao_cao_tlc_*.json`) đến sai số 1e-6; không khớp thì dừng.
- **Chỉ số:** chrF **câu trung bình** (xấp xỉ của chrF corpus, như `tong_hop_5mh`): "chính" 0,10/0,90 (zh→vi), macro hai hướng, và từng nhóm hướng × loại.
- **Bootstrap:** ghép cặp theo mã sản phẩm, 2.000 lần, `default_rng(42)`, cùng ma trận rút cho cả ba việc. Đơn vị là mã sản phẩm (một sản phẩm có thể có cả title và mô tả).
- **Cặp so sánh:** 25% − 11%, 35% − 11%, 35% − 25%. Không hiệu chỉnh đa so sánh (ghi rõ).
- **Luật diễn giải:**
  - Khoảng 95% chứa 0: "không phân biệt được ở mức dao động do chọn mẫu eval".
  - Khoảng không chứa 0: ghi chiều và độ lớn, nhưng vẫn kèm giới hạn "một seed, chưa phủ dao động khởi tạo".
  - Khoảng tin cậy chỉ phủ dao động do chọn mẫu eval, **không** phủ dao động khởi tạo; không dùng để nói về mức tối ưu.

## C. Chênh lệch kép của title bằng zero-shot Qwen3-4B (biên bản 1.2(d)(ii), 1.3.4)

- **Dữ liệu:** bản dịch zero-shot cụm trên toàn bộ eval cụm 1.200 hàng: `kaggle_tlc_song_song/zero_shot/sinh_zero_shot_cum.jsonl`. Giữ nguyên bản sinh (không làm sạch lời thừa). Bản pair không dùng được, không phân tích.
- **Nhóm tham chiếu và cách tính:** đúng `p1_phan_tich.vong_tham_chieu` (title theo `judge_chon_title`: qwen / gemma; trong eval zh→vi 55 sản phẩm Qwen, 245 Gemma; điểm từng sản phẩm là chrF corpus của title zh→vi; bootstrap theo sản phẩm 2.000 lần, seed 42).
- **Chênh lệch kép loại 1 (thiên vị sẵn có của họ mô hình):** `(zero-shot − mốc)` ở nhóm Qwen trừ `(zero-shot − mốc)` ở nhóm Gemma, mốc = Moses (học đủ), OpenNMT, EcomGPT-7B.
- **Chênh lệch kép loại 2 (phần do fine-tune):** `(Qwen3-4B tinh chỉnh − zero-shot)` ở nhóm Qwen trừ cùng đại lượng ở nhóm Gemma.
- **Luật diễn giải:**
  - Loại 1 khoảng 95% chứa 0: zero-shot không cho thấy lợi thế sẵn có theo họ ở title. Dương và không chứa 0: có lợi thế sẵn có ở nhóm tham chiếu Qwen, độc lập với fine-tune.
  - Loại 2 chứa 0: không thấy fine-tune hưởng lợi nhiều hơn ở nhóm tham chiếu Qwen. Dương và không chứa 0: fine-tune hưởng lợi nhiều hơn ở nhóm tham chiếu Qwen, nhất quán với vòng lặp tham chiếu qua dữ liệu huấn luyện, nhưng chưa chứng minh nguyên nhân.
  - Mô tả không phân tích: eval fold 0 không có sản phẩm Qwen chủ đạo (đã ghi ở P1).
- **Giới hạn ghi trước:** Judge chọn Qwen khi bản Gemma kém, nên hai nhóm khác độ khó; chênh lệch kép chỉ khử phần độ khó chung cho mọi mô hình. Nhóm Qwen chỉ 55 sản phẩm, khoảng tin cậy sẽ rộng. Zero-shot yếu hơn nhiều so với các mốc, nên "mốc" ở loại 1 là mô hình mạnh hơn zero-shot; chênh lệch kép so sánh khoảng cách, không so sánh mức.
