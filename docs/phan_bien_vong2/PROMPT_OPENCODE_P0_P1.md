# Giao việc cho OpenCode: trả lời phản biện vòng 2, phần nhánh Nhật Anh (P0 sửa tuyên bố + P1 chấm lại trên CPU)

Ngày giao: 17/11/2026. Người giao: Nhật Anh. Repo: `D:\nam4_ky1\e-commerce - EDA` (Windows 11, PowerShell, git nhánh `nhatanh_updating`).

## 0. Bối cảnh trong 5 câu

1. Đề tài: dịch máy thương mại điện tử Trung–Việt. Nhánh Nhật Anh dịch title + bảng thuộc tính sản phẩm (1688 zh→vi, Tiki vi→zh) bằng LLM (Gemma 4 / Qwen3-14B, Judge DeepSeek chọn bản), rồi fine-tune Qwen3-4B + LoRA trên hai dạng dữ liệu: **cụm** (thuộc tính nối bằng `"; "`) và **pair** (mỗi thuộc tính một dòng `\n`).
2. Hội đồng (Quang đóng vai) vừa ra biên bản phản biện vòng 2: `REVIEW_VONG2_3_PHAN_BIEN.md` ở gốc repo. Biên bản nhắm cả nhánh Quang và nhánh Nhật Anh; **bạn chỉ làm phần của nhánh Nhật Anh**.
3. Số liệu hội đồng trích là đúng. Vấn đề nằm ở **cách tuyên bố** và ở chỗ **bảng chính chưa tách dòng thuộc tính khuôn mẫu khỏi dòng mới**.
4. Việc của bạn: (P0) sửa câu chữ trong tài liệu cho đúng mức bằng chứng; (P1) chấm lại các bản dịch **đã có sẵn trên máy** của eval fold 0 để tạo bằng chứng mới. Tất cả chạy CPU, không cần GPU, không huấn luyện, không sinh bản dịch mới.
5. Song song, Claude (một AI khác) đang làm phần GPU trên Colab: thêm 2 mức `w_title` (khoảng 25% và 35%) và 2 seed OpenNMT. Bạn **không** làm phần đó và không động vào file của phần đó (mục 2.3).

## 1. Đọc trước, theo thứ tự

| # | File | Đọc để biết |
|---|---|---|
| 1 | `REVIEW_VONG2_3_PHAN_BIEN.md` | Toàn bộ biên bản. Phần của nhánh Nhật Anh: 1.2(c), 1.2(d)(ii), 1.2(b) áp cho fold 0, 1.3 mục 3–4; 2.2(b), 2.2(d), 2.3 mục 2–5; 3.2(c) câu về bootstrap, 3.2(e), 3.3 mục 5–6; mục 5 "Kết luận" |
| 2 | `docs/pipeline/00_TONG_QUAN.md` rồi `08_…`, `09_…`, `10_…` | Pipeline, phép chia, fine-tune, 5 mô hình, giới hạn đã ghi |
| 3 | `docs/ket_qua_kaggle/A7_chia_tap_theo_san_pham_kiem_ro_ri_gan_co_title_t.md` | Kiểm rò rỉ cấp sản phẩm; dòng "85,0% dòng mô tả test có nguyên văn trong pool" |
| 4 | `docs/ket_qua_kaggle/D3_opennmt_transformer_nho_hoc_tu_dau_pair_dich_ban.md` | OpenNMT nhánh mình; bảng chênh so với Qwen3-4B |
| 5 | `docs/QUOTA_KET_LUAN.md`, `docs/TRONG_SO_TITLE_MO_TA_DA_CHOT.md`, `docs/BANG_KET_QUA_CUOI.md` (đã bị thay) | Khảo sát quota, `w_title`, w hòa vốn |
| 6 | `docs/TONG_HOP_KET_QUA_NHAT_ANH.md` | Bảng kết quả gom một chỗ; nơi nhóm trích số |
| 7 | `scripts/ft_train/cham_diem.py` (đặc biệt `xay_tap_da_thay`, `cham`, cột `can_dong`, `_dong`) | Cách chấm và định nghĩa "dòng đã thấy" = cặp (category, dòng nguồn đã chuẩn hóa) có trong train |
| 8 | `scripts/ft_train/kiem_dong_moi.py`, `tong_hop_ti_le.py` | Cách các phiên B đã chấm "dòng mới" (bootstrap theo sản phẩm, 2.000 lần, seed 42) |
| 9 | `scripts/ft_train/tong_hop_5mh.py`, `cham_co_dien.py`, `cham_bleu.py` | Cách dựng bảng 5 mô hình trên eval fold 0; đường dẫn file sinh của từng mô hình |
| 10 | `scripts/ft_train/xuat_ket_qua_kaggle.py` | Bộ sinh `docs/ket_qua_kaggle/*.md`. Phần chữ viết tay (mục đích, kết luận, ghi chú) nằm trong bảng `PHIEN` của file này (A7 quanh dòng 696–701, D3 quanh dòng 801) |
| 11 | `scripts/ft_train/README.md`, `scripts/ft_train/chung.py` | Quy ước chung, `doc_split()` |

Dữ liệu dùng được (chỉ đọc):
- File sinh fold 0 của 5 mô hình (đường dẫn đúng như trong `tong_hop_5mh.py`):
  - Qwen3-4B: `data/processed/ft_run/kaggle_quota_step1/quota/{cum,pair}_natural_s{42,123}/sinh.jsonl`
  - Qwen3-8B: `data/processed/ft_run/kaggle_quota_nlp-8b/quota/...`
  - EcomGPT-7B: `data/processed/ft_run/kaggle_quota_nlp-ecom/quota/...`
  - OpenNMT: `kaggle_onmt_that/onmt/{zh-vi,vi-zh}_train/sinh.jsonl` (pair), `kaggle_onmt_cum/onmt_cum/{zh-vi,vi-zh}/sinh_cum.jsonl` (cụm)
  - Moses: `kaggle_moses_v2_{zhvi,vizh}/moses/*_char_{train,khop4m_s42}/sinh{,_cum}.jsonl`
  - Zero-shot Qwen3-4B và công thức chính trên fold 0: `kaggle_cheo_that/cheo/sinh_*.jsonl`, `kaggle_dong6/dong6/sinh_*.jsonl`
- Tham chiếu eval: `data/processed/ft_run/song_ngu/<huong>/...` (xem `cham_co_dien.tham_chieu`).
- Kết quả đã công bố để đối chiếu: `data/processed/ft_run/tong_hop_5mh{,_cum}/ket_qua.json`, `bang.md`, `bang_metric_bai_bao.md`.
- Nguồn tham chiếu (mô hình nào sinh bản dịch được giữ): `data/processed/bo_chon_hf/bo_{cum,pair}_chon_deepseek.parquet`:
  - cột `judge_chon_title` (`gemma`/`qwen`, mỗi sản phẩm);
  - cột `judge_chon_description`: chuỗi đếm mảnh kiểu `"gemma: 4 | quy_tac: 2 | chua_ro: 4"`;
  - thêm `bo_{cum,pair}_danh_dau.parquet` (`mo_ta_so_manh_gemma_duoc_ap_dung`, `mo_ta_so_manh_qwen_duoc_ap_dung`, …).
  - `chua_ro` là mảnh giữ bản trước Judge. Mô hình gốc của bản đó có thể tra qua cột `desc_nguon_dich` của bộ official v2 (`data/snapshot/bilingual_official_v2/`, xem `scripts/release/doc_desc_nguon_dich.py` và doc `docs/pipeline/05_…`). Tra được thì dùng, không tra được thì ghi rõ.
- OpenNMT train log (để vẽ đường học): `data/processed/ft_run/kaggle_onmt_that/onmt/{zh-vi,vi-zh}_train/train.log`, `dev.json`, `onmt_report.json`.

## 2. Luật cứng (vi phạm là dừng)

### 2.1. Không đụng tập test
- **Không mở, không đọc, không liệt kê nội dung**:
  - `data/processed/ft_run/test_khoa/`
  - `data/processed/ft_run/kaggle_do_test/`, `kaggle_do_test_thu/`
  - `data/processed/ft_run/goi_do_khoa/`
  - bất kỳ `split.parquet`, `ds_test.txt`
- Test khóa đã đo đúng một lần (07/11). Mọi số test được trích **chỉ từ tài liệu đã có** (ví dụ `docs/pipeline/09_…` mục 4, con số 85,0% trong A7). Không tính lại gì trên test.
- Mọi phân tích P1 chỉ dùng **eval fold 0** (train = fold 1–4).

### 2.2. Không sửa số đo, không chạy mô hình
- Không sửa file kết quả đã có (`*.json`, `*.jsonl`, `*.log` trong `data/processed/`). Kết quả mới ghi vào thư mục mới `data/processed/ft_run/phan_bien_v2/`.
- Không gọi GPU, không huấn luyện, không sinh bản dịch, không gọi API LLM nào.
- Không mạng ra ngoài: không tải/đẩy Hugging Face, Kaggle, GitHub.

### 2.3. Không đụng phần của người khác
- Phần GPU của Claude: không sửa `scripts/ft_train/kaggle_*.py`, `chay_cuoi_8b.py`, `dung_mau.py`, `dong_goi_*.py`, `quota_sampling.py`, `notebooks/colab/`, `scripts/colab/` (nếu có).
- Không sửa `WALKTHROUGH.md` (Claude ghi nhật ký). Ghi nhật ký của bạn vào `docs/phan_bien_vong2/NHAT_KY_OPENCODE.md`.
- Không sửa tài liệu của nhánh Quang. Không sửa file `.pptx`/`.docx`; nếu thấy câu sai trong `docs/bao_cao_thay/*.pptx` thì **chỉ liệt kê** (slide số mấy, câu nào, sửa thành gì).

### 2.4. Git và bí mật
- Không `git commit`, không `git push`. Có thể dùng `git mv` để chuyển file, `git diff` để tự kiểm.
- Không mở `api_deepseek_alibaba.txt`, không in token/khóa nào. Trước khi xong, quét diff tìm `KGAT_`, `sk-`, `Bearer`, `api_key`, `hf_`, `ghp_`, `AIza`, `trycloudflare`, `@gmail` và báo kết quả.

### 2.5. Môi trường
- Python: `.\.venv\Scripts\python.exe -I -X utf8 <script>`; đặt `$env:OPENBLAS_NUM_THREADS=1`. Máy 15,6 GB RAM, đọc jsonl theo luồng nếu file lớn.
- Thiếu thư viện thì báo, không cài vào Python toàn cục.
- Script mới đặt ở `scripts/phan_bien_v2/`. Viết kèm `test_*.py` (unittest, dữ liệu giả nhỏ) theo phong cách `scripts/ft_train/test_*.py`. Tái dùng hàm có sẵn (`cham_diem`, `cham_co_dien`, `cham_bleu`, `tong_hop_5mh`) bằng import, không chép lại logic chấm.

### 2.6. Cách viết tài liệu
- Tiếng Việt, xưng hô trung tính, viết ngắn, mỗi câu một ý. Số kiểu Việt (dấu chấm ngăn nghìn, dấu phẩy thập phân) trong chữ; bảng kết quả mô hình giữ kiểu số như bảng gốc.
- Nhãn bắt buộc cho mọi con số mới: **[ĐO]** đo trên dữ liệu thật; **[ƯỚC]** ước tính; **[CHƯA KIỂM]** chưa xác minh.
- Không dùng gạch ngang dài "—" trong câu văn. Không dùng từ "tối ưu", "vượt trội", "production", "chứng minh" trừ khi đúng nghĩa đen và có số đi kèm.
- Trung thực tuyệt đối: số nào không ra như kỳ vọng thì ghi đúng như vậy.

## 3. P1 (làm trước P0, vì P0 cần số của P1)

### 3.0. Khóa kế hoạch phân tích trước khi xem số
Trước khi chạy bất kỳ phép chấm mới nào, viết `docs/phan_bien_vong2/KE_HOACH_PHAN_TICH_P1.md` gồm: chỉ số chính của từng phân tích, cách bootstrap, câu diễn giải cho từng kết cục (ví dụ "nếu khoảng tin cậy chứa 0 thì viết …"). Ghi sha256 của file này vào `NHAT_KY_OPENCODE.md` **trước** khi chạy. Sau khi có số, không sửa kế hoạch; muốn phân tích thêm thì ghi là "phân tích thêm sau khi xem số".

### 3.1. Kiểm tái lập trước khi tách
Chạy lại phép chấm toàn phần của 5 mô hình trên fold 0 (cả `pair` và `cum`) bằng chính `tong_hop_5mh.py` hoặc hàm của nó, ghi ra thư mục mới (không ghi đè `tong_hop_5mh/`). So với `tong_hop_5mh{,_cum}/ket_qua.json`. **Lệch quá 0,01 điểm ở bất kỳ ô nào thì dừng và báo**, không tự chỉnh.

### 3.2. Tách dòng khuôn / dòng mới cho 5 mô hình (trả lời 1.2(c), 1.3 mục 3, 2.2(b))
- Định nghĩa "dòng đã thấy" giữ đúng `cham_diem.xay_tap_da_thay` với train = fold 1–4. Ghi lại định nghĩa trong báo cáo.
- Với mỗi bộ (cụm, pair) × hướng (zh→vi, vi→zh): báo **tỉ lệ dòng mô tả eval đã thấy** (dòng khuôn) theo dòng và theo sản phẩm, kèm theo nhóm hàng. Đối chiếu với con số 85,0% (test, A7) và 87% (eval pair, doc 09).
- Với mỗi mô hình (Qwen3-4B, Qwen3-8B, EcomGPT-7B, OpenNMT, Moses train đầy đủ, Moses khớp 4M; thêm zero-shot Qwen3-4B và công thức chính 4B nếu file khớp id): chrF mô tả trên **dòng khuôn** và **dòng mới** riêng, cả chrF câu TB và chrF corpus; thêm SacreBLEU/TER nếu hàm có sẵn hỗ trợ.
- Chênh so với Qwen3-4B trên dòng mới, bootstrap theo sản phẩm 2.000 lần, seed 42, cùng mẫu rút cho mọi mô hình (như `tong_hop_5mh.py`). LLM lấy trung bình 2 seed như bảng gốc.
- Chỉ dùng sản phẩm có số dòng hyp khớp số dòng ref (`can_dong`); báo số sản phẩm bị loại vì lệch dòng theo từng mô hình (OpenNMT/Moses dịch từng mảnh nên gần như luôn khớp; LLM có thể lệch).
- Giả thuyết ghi trước (trong 3.0): khoảng cách giữa mô hình học từ đầu (OpenNMT, Moses) và LLM **lớn hơn** trên dòng mới so với dòng khuôn. Kết quả ra thế nào ghi thế ấy.

### 3.3. Kiểm vòng lặp tham chiếu (trả lời 1.2(d)(ii), 1.3 mục 4)
- Chia eval fold 0 theo nguồn tham chiếu:
  - Title: `judge_chon_title` = `gemma` hay `qwen`.
  - Mô tả: theo tỉ phần mảnh Qwen trong các mảnh có nguồn xác định. Đặt ngưỡng trong kế hoạch 3.0 (gợi ý: "Qwen chủ đạo" ≥ 2/3, "Gemma chủ đạo" ≤ 1/3, còn lại "trộn"), và báo phần "không xác định".
- Với mỗi nhóm: chrF của từng mô hình, và chỉ số chính là **chênh lệch kép**:
  `(Qwen3-4B − mốc)` ở nhóm tham chiếu Qwen trừ `(Qwen3-4B − mốc)` ở nhóm tham chiếu Gemma, với mốc lần lượt là Moses (train đầy đủ), OpenNMT, EcomGPT-7B (họ BLOOM). Làm tương tự cho Qwen3-8B và zero-shot Qwen3-4B (zero-shot cho biết thiên vị có sẵn từ họ mô hình, không do fine-tune). Bootstrap theo sản phẩm như 3.2.
- Báo cáo theo **họ mô hình**: Qwen (4B, 8B, zero-shot 4B), BLOOM (EcomGPT), không phải LLM (OpenNMT, Moses).
- Ghi giới hạn: Judge chọn Qwen khi bản Gemma kém, nên hai nhóm khác nhau về độ khó; chênh lệch kép chỉ khử được phần độ khó chung cho mọi mô hình. Bản title "qwen" có thể là Qwen3-14B-AWQ hoặc Qwen3.5-9B (doc 05 mục 1, chưa khẳng định). Tham chiếu mô tả do Gemma 4 hoặc Qwen3-14B sinh (doc 03).

### 3.4. Kết quả theo độ dài (2.3 mục 2)
Chia eval fold 0 theo độ dài nguồn (số ký tự; ngưỡng tứ phân vị tính trên eval, ghi trong kế hoạch 3.0), báo chrF từng mô hình theo nhóm độ dài, riêng title và mô tả, zh→vi và vi→zh.

### 3.5. Đường học OpenNMT (2.2(d))
Từ `train.log`/`dev.json`/`onmt_report.json` của `kaggle_onmt_that`: rút điểm dev (perplexity/accuracy/chrF, cái nào có) theo bước hoặc epoch cho hai hướng; vẽ PNG (matplotlib) vào `docs/phan_bien_vong2/hinh/`, kèm bảng số. Ghi rõ bước dừng sớm và checkpoint được lấy trung bình. Nếu log không có điểm dev theo thời gian thì ghi đúng như vậy, không bịa.

### 3.6. Bảng compute và bảng hợp nhất OpenNMT (2.3 mục 3, 5)
- Bảng compute nhánh mình cho 5 mô hình: số tham số, token đã học (đọc từ report: ví dụ `onmt_report.json` có "Token SP train", report quota có ngân sách token), số epoch/lượt, GPU-giờ nếu report có. Cột **pretraining**: Qwen3 khoảng 36 nghìn tỷ token (báo cáo kỹ thuật Qwen3); EcomGPT dựa trên BLOOMZ-7b1-mt, BLOOM khoảng 366 tỷ token; OpenNMT, Moses: 0. Hai con số pretraining gắn **[CHƯA KIỂM]** kèm tên nguồn công bố, vì không kiểm được ngoại tuyến.
- Bảng hợp nhất hai baseline OpenNMT: cột nhánh Nhật Anh (D3: d_model 256, 4 lớp, 8 đầu, ff 1024, SentencePiece unigram 16k mỗi ngôn ngữ, Noam 2e-3, ≤20 epoch + dừng sớm, trung bình 5 checkpoint, beam 5, học trên dòng của bản pair, 1 seed, eval fold 0 chrF) và cột nhánh Quang lấy **từ chính biên bản** (37M tham số, 4 lớp/256 hidden, char-level 32k, beam 512, `replace_unk`, 30k step A / 22.990 B, eval_F, COMET 0,349), ghi "theo biên bản vòng 2, chưa đối chiếu cấu hình gốc của Quang". Thêm một dòng: hai tập đo và hai thước đo khác nhau nên không so điểm trực tiếp, chỉ so cấu hình.

### 3.7. Đầu ra P1
- Script: `scripts/phan_bien_v2/*.py` + test.
- Số: `data/processed/ft_run/phan_bien_v2/*.json` (kèm sha256 đầu vào đã đọc).
- Báo cáo: `docs/phan_bien_vong2/KET_QUA_P1.md` với các bảng 3.2–3.6, mỗi bảng có câu "đọc bảng" 1–3 dòng và nhãn [ĐO]/[ƯỚC]/[CHƯA KIỂM].

## 4. P0: sửa tuyên bố trong tài liệu

Nguyên tắc: sửa tại chỗ, giữ nguyên số đo, chỉ đổi cách nói và thêm phần thiếu. Mỗi sửa đổi ghi một dòng vào `NHAT_KY_OPENCODE.md` (file, mục, câu cũ → câu mới, vì sao, mục biên bản tương ứng).

### 4.1. A7: rò rỉ cấp sản phẩm khác cấp dòng (1.2(c), 1.3 mục 3)
- Trong `xuat_ket_qua_kaggle.py`, phần A7: giữ nguyên dòng "Kết luận trong file: …" (đọc từ JSON), nhưng thêm ngay sau đó câu diễn giải: không có sản phẩm gần-trùng giữa test và pool **ở cấp sản phẩm** (Jaccard), **nhưng ở cấp dòng**, 85,0% dòng mô tả của test đã có nguyên văn trong pool. Vì vậy test là held-out theo sản phẩm, **không** phải held-out cho từng dòng mô tả; điểm mô tả toàn phần chủ yếu đo khả năng tái tạo thuộc tính khuôn mẫu.
- Sửa `ket_luan` của A7 theo đúng ý trên. Chạy lại bộ sinh: `.\.venv\Scripts\python.exe -I -X utf8 scripts/ft_train/xuat_ket_qua_kaggle.py`. Bộ sinh xóa và ghi lại toàn bộ `docs/ket_qua_kaggle/*.md`; sau khi chạy, dùng `git diff --stat docs/ket_qua_kaggle` để chắc chỉ A7, D3 (và INDEX nếu cần) đổi. File khác đổi thì tìm nguyên nhân và báo, không để lẫn.
- Sửa câu tương ứng trong `docs/pipeline/08_…` và `docs/TONG_HOP_KET_QUA_NHAT_ANH.md` nếu có chữ "không rò rỉ" đứng một mình.

### 4.2. Bảng chính phải có cột dòng mới (1.3 mục 3)
Mọi bảng mô tả dùng để kết luận trong `docs/pipeline/09_…` mục 6, `docs/TONG_HOP_KET_QUA_NHAT_ANH.md` và `docs/KE_HOACH_VA_KET_QUA_NAM_MO_HINH.md`: thêm cột (hoặc bảng ngay dưới) "Mô tả, dòng mới" lấy từ `KET_QUA_P1.md`. Câu "chrF mô tả 83–91" và các câu xếp hạng mô tả phải kèm số dòng mới.

### 4.3. D3 và câu "OpenNMT hòa 4B" (2.2(b), 2.3 mục 3–4)
- Sửa `ket_luan` của D3 trong `xuat_ket_qua_kaggle.py` (và câu tương ứng ở `docs/pipeline/09_…` mục 6): nêu đủ bốn số của pair so với Qwen3-4B (chrF câu TB +0,39 [−1,09; +1,90]; chrF corpus chính 74,38 so với 79,98; macro câu −8,33 [−9,40; −7,26]) và cụm (−5,57 [−6,42; −4,85]); thêm câu về dòng mới từ P1.
- Câu kết luận được phép: "Với cấu hình Transformer nhỏ học từ đầu (D3), OpenNMT hòa Qwen3-4B ở pair theo chrF câu trung bình trên eval fold 0, nơi phần lớn dòng đã có trong train; kém rõ ở cụm và ở vi→zh. Kết quả cho thấy độ nhạy theo cấu hình và theo loại dòng, không kết luận về paradigm NMT." Điều chỉnh theo số dòng mới của P1.
- Rà toàn bộ `docs/` của nhánh mình: không còn câu kiểu "NMT yếu", "LLM > NMT", "NMT không cạnh tranh".

### 4.4. Tỉ trọng title: tách loss weighting khỏi quota sampling (3.2(e), 3.3 mục 5–6)
Sửa `docs/QUOTA_KET_LUAN.md`, `docs/TRONG_SO_TITLE_MO_TA_DA_CHOT.md`, `docs/pipeline/09_…` mục 3, `docs/TONG_HOP_KET_QUA_NHAT_ANH.md`:
- **Bằng chứng chính** cho `w_title` của bản cuối là thí nghiệm **trọng số loss** (phiên B/TL: 3 mức 1:1 ≈ 50%, ≈11%, 75%; 3 fold; khoảng tin cậy; luật khóa trước). Trong cơ chế này mọi title được thấy đúng số lần như nhau (2 epoch), chỉ đổi trọng số, nên **không dính** confound "tỉ trọng vs số lần thấy" mà hội đồng nêu cho quota. **Trước khi viết câu này, kiểm lại trong `scripts/ft_train/kaggle_train.py` (chỉ đọc)** rằng `w_title` là hệ số nhân loss từng mẫu và dữ liệu không bị lấy mẫu lại theo tỉ lệ; nếu không đúng thì báo, không viết.
- **Khảo sát quota** (C1–C4, rank 16, một lượt, 2 seed, 300 mẫu/nhóm) chỉ là **kiểm chéo**, có confound tần suất, không dùng làm căn cứ chính.
- Câu kết luận được phép: "Trong 3 mức đã thử, khoảng 11% title tốt nhất dưới thước 10/90 (zh→vi). Kết luận đổi nếu trọng số title của thước đo từ khoảng 0,39 (cụm) và 0,50 (pair) trở lên. Chưa thử mức 11–50%; mức 25% và 35% đang chạy (phần GPU)." Bỏ mọi chữ "tối ưu".
- Thêm câu về khoảng tin cậy: bootstrap chỉ phủ dao động do chọn mẫu eval, không phủ dao động khởi tạo; số seed của từng bảng ghi rõ (sàng lọc 2 seed, công thức chính 1 seed).

### 4.5. Fold 0 là tập chọn, 5 mô hình không phải test (1.2(b), mục 5)
- Mọi bảng 5 mô hình ghi rõ: "eval fold 0 (tập dùng để chọn cấu hình, train = fold 1–4), cấu hình sàng lọc r16, chưa đo trên test".
- Thêm một câu: các lựa chọn đã dựa trên fold 0 (tín hiệu dòng mới cho `w_title`, so 5 mô hình, chọn adapter demo) nên số trên fold 0 bị ảnh hưởng bởi việc chọn; tỉ trọng title có thêm kiểm độc lập ở fold 1 và 2.
- Số test duy nhất là B8 (07/11, adapter fold 0 so với zero-shot). Ghi rõ chênh khoảng 30 điểm bị phóng đại một phần (tham chiếu máy cùng kiểu) **và** chưa tách dòng mới trên test.

### 4.6. Vòng lặp tham chiếu (1.2(d)(ii), 1.3 mục 4)
Thêm một mục riêng (trong `docs/pipeline/10_…` mục 4 và `TONG_HOP_…`): tham chiếu title và mô tả do Gemma 4 hoặc Qwen (Qwen3-14B, có thể Qwen3.5-9B cho một phần title) sinh, Judge DeepSeek chọn; dữ liệu huấn luyện cùng nguồn đó; Qwen3-4B/8B cùng họ với một mô hình sinh tham chiếu. Dẫn số của P1 mục 3.3. Ghi việc còn mở: tham chiếu độc lập do người (100–300 mẫu) chưa có.

### 4.7. Demo (mục 5)
Rà `docs/pipeline/10_…`, `demo/CHAY_DEMO.md`, `TONG_HOP_…`: demo là "thử trên 13 trang 1688 thật, đọc tay 8 trang, không có tham chiếu, một người dùng, máy chủ trên Kaggle T4 hoặc laptop". Không có chữ "triển khai", "production", "sẵn sàng sử dụng".

### 4.8. Tài liệu đã bị thay
`git mv docs/BANG_KET_QUA_CUOI.md docs/archive/BANG_KET_QUA_CUOI_09-11_DA_THAY.md` (tạo `docs/archive/` nếu chưa có), sửa mọi liên kết trỏ tới nó. Giữ banner cảnh báo ở đầu file.

### 4.9. Các chỗ cần rà theo grep đã thấy
Các dòng sau có từ khóa nhạy ("tối ưu", "production", "không rò rỉ", …), đọc từng chỗ và sửa nếu là tuyên bố quá mức:
`docs/KE_HOACH_VA_KET_QUA_NAM_MO_HINH.md:55`, `docs/PHU_LUC_NHAT_KY_CHAY.md:33`, `:213`, `docs/PROJECT_CHARTER_team_CLAUDE.md:23`, `docs/PROMPT_BAN_GIAO_AI.md:9`, `docs/TONG_HOP_KET_QUA_NHAT_ANH.md:233`, `docs/TRONG_SO_TITLE_MO_TA_DA_CHOT.md:57`, `docs/pipeline/01_thu_thap_crawl.md:9`, `docs/pipeline/08_chia_tap_va_dung_mau.md:29`, `docs/pipeline/09_fine_tune_va_danh_gia.md:19`, `:67`. Grep thêm với các từ: "thắng", "vượt", "tốt nhất", "held-out", "test" (ở bảng 5 mô hình).
Slide `docs/bao_cao_thay/Bao_cao_bo_song_ngu_chinh_thuc.pptx`: chỉ đọc chữ (ví dụ bằng `python-pptx` nếu có trong `.venv`), liệt kê câu cần sửa, không sửa file.

## 5. Bản trả lời phản biện (đầu ra cuối)

Viết `docs/phan_bien_vong2/TRA_LOI_V2_NHANH_NHAT_ANH.md`, dành cho hội đồng đọc. Cấu trúc: một bảng ở đầu (mục biên bản | hội đồng nói | nhánh mình chấp nhận gì | đã sửa gì, ở file nào | số mới | còn mở), rồi mỗi mục một đoạn ngắn. Các mục cần có:
1. 1.2(c) + 1.3.3: rò rỉ cấp dòng; bảng dòng khuôn/dòng mới (P1 3.2).
2. 1.2(d)(ii) + 1.3.4: vòng lặp tham chiếu; chênh lệch kép (P1 3.3); người chấm chưa có.
3. 1.2(b) áp cho nhánh mình: fold 0 là tập chọn; test dùng một lần (B8).
4. 2.2(b) + 2.3.3–5: D3, bảng hợp nhất, bảng compute, đường học, theo độ dài.
5. 3.2(e) + 3.3.5–6: loss weighting vs quota; câu kết luận hạ mức; mức 25%/35% và 2 seed OpenNMT **đang chạy ở phần GPU (Claude)**, chưa có số.
6. Mục 5 biên bản: 5 mô hình là eval fold 0; demo không phải production.
7. Việc chờ Nhật Anh quyết (chỉ liệt kê, không làm): dùng test khóa lần hai có đăng ký trước để tách dòng mới trên test; người song ngữ chấm 100–300 mẫu.

Không nhận thay nhánh Quang các mục chỉ thuộc về họ (eval_F, NLLB, 2×2 A/B, hash 10 cell, README_EVIDENCE).

## 6. Khi xong, báo lại cho Nhật Anh

Trong tin nhắn cuối, gồm:
- Danh sách file đã tạo/sửa/chuyển (đường dẫn), và `git diff --stat`.
- Kết quả kiểm tái lập 3.1 (khớp hay lệch, lệch bao nhiêu).
- 3–6 con số quan trọng nhất của P1 kèm nhãn, đặc biệt: tỉ lệ dòng khuôn trên fold 0; khoảng cách OpenNMT/Moses so với 4B trên dòng mới; chênh lệch kép của vòng lặp tham chiếu.
- Mọi chỗ không làm được hoặc phải đoán, ghi rõ lý do.
- Kết quả quét bí mật trong diff.
- Kết quả chạy test (`python -m unittest` cho test mới và các test cũ liên quan tới file đã sửa, ví dụ `scripts/ft_train/test_*.py` nếu có cho bộ sinh).

Dừng và hỏi Nhật Anh (không tự quyết) khi: tái lập lệch > 0,01; một file sinh thiếu hoặc id không khớp tham chiếu; phải đọc bất kỳ thứ gì trong danh sách cấm ở 2.1; câu "mọi title thấy 2 epoch" không đúng với mã; bộ sinh `xuat_ket_qua_kaggle.py` đổi những file ngoài A7/D3/INDEX.
