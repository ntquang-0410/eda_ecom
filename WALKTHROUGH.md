# WALKTHROUGH — NHẬT KÝ TIẾN ĐỘ (NHẬT ANH)

File này ghi lại: repo có gì, luồng chạy ra sao, mình đã làm/tìm ra được gì, và
nhật ký prompt dùng AI trong ngày (kể cả chỗ sai). Cập nhật thêm mỗi ngày làm
việc, không viết lại từ đầu — thêm mục mới ở cuối phần "Nhật ký theo ngày".

> **Quy ước (từ 28/09/2026, tới hết project):** mọi prompt và mọi thay đổi ở
> mọi phiên làm việc đều được ghi vào mục 7 "NHẬT KÝ THEO NGÀY" của file này.

> Đây là file cá nhân của Nhật Anh, mục đích để Quang/Huy nắm được đang làm
> tới đâu, không phải tài liệu kỹ thuật chính thức (cái đó nằm ở `README.md`
> của repo `ecom_crawler`). File này cũng được đẩy sang repo `eda_ecom`
> (nhánh `nhatanh_updating`) — xem mục 3, điểm 9.

---

## 0. ĐANG Ở ĐÂU (CẬP NHẬT 01/10/2026)

**Folder làm việc từ 01/10/2026: `D:\nam4_ky1\e-commerce - EDA`** (bước EDA và
Xử lý dữ liệu). File này là **nguồn thật duy nhất**; bản trong `ecom_crawler`
đã đóng băng và có dòng trỏ sang đây.

- Mục 1–2 và nhật ký trước 01/10 mô tả `ecom_crawler` và đường dẫn cũ; đó là
  lịch sử, giữ nguyên. Đường dẫn `data/eda/...` vẫn đúng ở folder mới. Mục 3
  điểm 7–8 ("script chưa nằm trong repo") nay không còn đúng: script nằm ở
  `scripts/` (xem `README.md`).
- Nguồn gốc từng file chuyển sang: `docs/PROVENANCE.md` và
  `docs/copy_manifest.csv` (41 file: 33 giống hệt byte, 7 script chỉ sửa đường
  dẫn, và chính file này sau khi thêm mục 0). Dữ liệu đầu vào đóng băng:
  `data/snapshot/SNAPSHOT.md` (kèm SHA256).
- Môi trường: `.venv` riêng, Python 3.13.7, pandas 3.0.5, numpy 2.5.3, pyarrow
  25.0.1, pypinyin 0.55.0 (`requirements.txt`). Không cần `.env`: HF dataset
  công khai. Folder này **không phải git repo**.
- Đã kiểm chứng ngày chuyển: chạy lại 4 script audit ở folder mới ra 8 file
  kết quả **giống hệt từng byte** bản gốc.

**Bức tranh tổng thể dự án.** Nguồn chính: `docs/PROJECT_CHARTER_team_CLAUDE.md`
(CLAUDE.md gốc của nhóm, 15/09). Chi tiết về nhóm, GVHD và lịch lấy từ ghi chú
memory của Claude ngày 15–17/09 (tức lời Nhật Anh nói lúc đó), chưa có tài liệu
chính thức để kiểm lại.
- **Đề tài** (TLCN, nhóm 3 người: Quang PM, Nhật Anh, Huy; GVHD ThS. Đoàn Minh
  Trí; 16 tuần, tháng 9–12/2026): tối ưu dịch máy Việt–Trung cho TMĐT xuyên
  biên giới. Tiêu đề TMĐT dồn từ khoá, vi phạm ngữ pháp có chủ đích, nên mô hình
  dịch tổng quát hay thêm diễn giải thừa. Không có bộ dữ liệu song ngữ Vi–Zh
  miền TMĐT công khai, nên tự crawl là đóng góp chính.
- **Mô hình:** charter ghi fine-tune Sailor2-8B, so sánh Seed-X-7B. **Từ 01/10 Nhật
  Anh gỡ mô hình fine-tune khỏi pipeline, chờ nhóm bàn lại** (Prompt 23): không thiết
  kế phân tích hay xử lý dữ liệu theo một mô hình cụ thể. GPU huấn luyện theo
  charter là RTX PRO 6000 96 GB ở giai đoạn sau; máy này chỉ có RTX 3050 4 GB
  (torch CPU-only).
- **Quy trình** (WBS ghi nhận 15–17/09): khảo sát → thu thập đơn ngữ Việt/Trung
  → làm sạch → mining ghép cặp bằng LaBSE + FAISS → back-translation → gộp/chia
  → fine-tune → đánh giá BLEU/chrF/COMET → demo → báo cáo. Charter nói "8 giai
  đoạn" nhưng chỉ liệt kê GĐ1, GĐ3, GĐ4. **Bước hiện tại: làm sạch (GĐ1) và chuẩn
  bị chia dữ liệu.**
- **Vai trò từng nguồn dữ liệu:** 1688 song ngữ (16.348 dòng) là máy dịch
  Alibaba, chỉ dùng làm train (silver) và nguồn thuật ngữ, **không bao giờ làm
  test**; test 500–1.000 cặp phải do người dịch hoặc kiểm duyệt. Tiki (13.200
  sản phẩm) là tiếng Việt bản địa cho mining/back-translation; `mono_zh` (6.334)
  là nguồn back-translation.
- Nhật Anh dặn 17/09: đừng để ý phân công theo tên trong timeline, ai có khả
  năng thì làm (không gợi ý bàn giao việc theo tên).
- Quy tắc cứng rút từ charter nằm ở `CLAUDE.md` mục 5.7.

**Nợ ngữ cảnh** (dễ rơi mất, chưa có trong danh sách D00–D17 của mục 5):
1. 2 file thiết kế pipeline `docs/design/pipeline_artifact.html` ("Làm Sạch Dữ
   Liệu 1688") và `eda_pipeline.html` ("EDA Corpus 1688-Tiki", pipeline 6 bước,
   tab đánh giá BGE-M3). Số liệu trong đó chưa kiểm lại.
2. Tiki: đoạn thông báo thuế/VAT lặp cuối `description_prose_vi` ở 13.199/13.200
   dòng (README crawler mục 10).
3. 95 dòng trùng cả `(title_zh, description_zh)` — chưa kiểm lại.
4. 1.629 dòng `title_vi` dài hơn 5 lần `title_zh` — chưa kiểm lại.
5. Số dòng lỗi đánh số: **đã đối chiếu 01/10.** Đo lại trên snapshot hiện tại:
   9.354 dòng (57,2%) nếu tính "Không/Có/Là/Khác" + số ở bất kỳ vị trí; 9.333
   (57,1%) nếu ở cuối giá trị; 9.286 (56,8%) nếu chỉ tính 16.276 dòng ghép được
   (con số D07). Ba số khớp nhau; chênh 47 dòng giữa 9.333 và 9.286 nằm trong
   khoảng 72 dòng không ghép cặp ở D08 (suy luận, chưa kiểm từng dòng). Số 8.494
   của thiết kế cũ **không tái hiện được** bằng regex nào đã thử: coi là lỗi
   thời, dùng D07.
6. Phát hiện của Huy trên nhánh `giahuy` chưa gộp: dấu chấm cuối câu 5.064 dòng,
   ký tự toàn chiều, số thập phân 4,3 vs 4.3, chữ số Hán.
7. Nguồn Kaggle Tiki (~5.360 dòng fashion) chưa tải, cần tài khoản Kaggle.
8. Thông tin cá nhân (charter §6, phải lọc trước khi công bố): cột `shop` (tên
   công ty) có ở 16.348/16.348 dòng; 92 dòng `description_zh` chứa chuỗi 11 chữ
   số dạng số di động Trung Quốc (có thể là mã hàng, chưa kiểm).

**Quyết định đang chờ Nhật Anh:** cách chia train/dev/test (README crawler ghi
theo `product_id`, charter ghi theo `product_id` hoặc shop, D10 cho thấy cùng
`title_zh` rơi vào nhiều dòng → nên nhóm theo `title_zh`); thứ tự các bước làm
sạch (mục 6.1); 回力 (Huili / Warrior / Pull-back, tạm "Warrior"); "dropship"
(tạm "giao hộ"); **D16** đề xuất bỏ 【】 ~ | * nhưng charter dặn không xoá ký tự
trang trí của miền (data hiện không có emoji); **từ mượn quảng cáo** (freeship,
hot, sale...): charter coi `freeship` là cách dịch đúng của 包邮, còn "hot" (323
dòng) đang tạm đổi thành "bán chạy".

**Vướng mắc đang mở:** Ollama không thấy model `bge-m3` (xem Prompt 15), nên
`sim_full.py` và `run_nb_harness.py` chưa chạy lại được ở folder mới.

---

## 1. CẤU TRÚC THƯ MỤC

```
ecom_crawler/
├── config.py                  Load .env -> Settings (HF token, Firebase, tài khoản 1688...)
├── main.py                    Entry point chạy 1 worker (search hoặc detail)
├── keywords_1688.txt          Danh sách từ khoá seed cho search 1688
├── run_crawl.bat               .bat khởi động worker crawl 1688 (search + detail)
├── run_crawl_tiki.bat          .bat khởi động crawl Tiki (mono Việt)
├── sync_to_hf.bat              .bat chạy sync_loop.py (đẩy parquet lên HF định kỳ)
├── eda-bilingual_zh_vi.ipynb  Notebook EDA cho bộ dữ liệu song ngữ (CHƯA commit, xem mục 4)
├── README.md                   Tài liệu chính của repo — kiến trúc, lịch sử đổi schema
│
├── core/                      Engine dùng chung, không chứa logic riêng từng trang
│   ├── base_crawler_engine.py     Lớp cha cho mọi engine (open/close browser...)
│   ├── crawler_engine.py          Điều phối chung 1 lượt crawl (fetch -> parse -> ghi)
│   ├── playwright_crawler_engine.py  Engine chạy Chrome thật qua Playwright
│   ├── browser_1688.py            Quản lý phiên đăng nhập/slider riêng cho 1688
│   ├── engine_1688_search.py      Crawl trang tìm kiếm 1688 -> danh sách sản phẩm
│   ├── engine_1688_detail.py      Crawl trang chi tiết 1688 (song ngữ zh+vi)
│   ├── engine_tiki_detail.py      Crawl trang chi tiết Tiki (mono vi)
│   ├── tiki_client.py             Gọi thẳng JSON API công khai của Tiki (không cần login)
│   ├── queue_manager.py           Claim/release/mark_done trên Firebase Realtime DB
│   ├── raw_writer.py              Ghi JSONL vào data/raw/
│   ├── data_packager.py           Gom record thành đúng schema trước khi ghi
│   ├── hf_uploader.py             Đẩy parquet lên Hugging Face Hub
│   ├── textnorm.py                Chuẩn hoá text (NFC, full-width->ASCII, bỏ HTML/entity)
│   ├── worker.py                  Vòng lặp chính của 1 worker (claim -> crawl -> ghi -> lặp)
│   └── models.py                  Định nghĩa schema record (dataclass)
│
├── parsers/                   Parse HTML/JSON thô -> field có cấu trúc
│   ├── base_parser.py / generic_parser.py
│   ├── parser_1688_search.py, parser_1688_detail.py
│   └── parser_tiki.py
│
├── scripts/                   Script chạy tay, không nằm trong vòng lặp worker
│   ├── seed_1688_keywords.py, seed_1688_details.py, seed_queue.py, seed_tiki_details.py
│   ├── reseed_1688_search.py      Bơm thêm từ khoá tìm kiếm mới vào queue
│   ├── cap_queue_by_category.py   Giới hạn số sản phẩm/category trong queue
│   ├── dedupe_queue.py
│   ├── snapshot_to_hf.py          Gộp raw của cả 3 máy -> 1 file parquet sạch trên HF
│   ├── snapshot_tiki_to_hf.py     Bản snapshot riêng cho Tiki
│   ├── mirror_raw_to_hf.py        Sao lưu thô data/raw/ lên HF (không xử lý gì)
│   ├── sync_loop.py               Lặp định kỳ: snapshot + mirror
│   ├── sample_crawl.py / sample_crawl_tiki.py   Crawl mẫu nhỏ để xem trước, không đụng Firebase/HF
│   ├── run_pipeline.py            Chạy full pipeline (search -> detail, tự seed, tự mirror)
│   ├── login_1688.py, set_1688_language.py, firebase_rules.py
│
├── firebase/
│   ├── database.rules.json
│   └── *-firebase-adminsdk-*.json   ⚠️ Secret, đã .gitignore
│
└── data/
    ├── raw/                    JSONL thô, 1 file/worker/ngày (1688 bi/mono, tiki)
    ├── raw_old/                Data theo schema cũ trước 17/09, không tương thích nữa
    ├── snapshot_tmp/           Bản parquet tải về máy để phân tích local (mono_zh, tiki_vi)
    ├── sample/, sample_tiki/   Mẫu nhỏ để review nhanh
    └── eda/                    Kết quả phân tích (xem mục 3.3)
        ├── title_similarity_full.csv / _summary.json
        ├── full_run/           Log + output chạy toàn bộ notebook EDA trên full data
        ├── title_audit/        Audit tiếng Anh + các lỗi khác trong title_vi
        └── data_issues/        Tổng hợp mọi vấn đề của data + bằng chứng theo product_id (mục 5)
```

## 2. LUỒNG HOẠT ĐỘNG (FLOW)

### 2.1 CRAWL 1688 (SONG NGỮ TRUNG–VIỆT) — NGUỒN CHÍNH

```
keywords_1688.txt
      │ seed_1688_keywords.py
      ▼
queue_search_zh (Firebase)  ──worker: Search1688Engine──▶ data/raw/1688search_zh_*.jsonl
      │ seed_1688_details.py (đọc kết quả search, đẩy sang detail)
      ▼
queue_detail_bi (Firebase)  ──worker: Detail1688Engine──▶ có bản Việt:   1688_bilingual_*.jsonl → HF bronze/bilingual_zh_vi
      │                                                  không bản Việt: 1688_mono_zh_*.jsonl   → HF bronze/mono_zh
      ▼ (định kỳ)
scripts/snapshot_to_hf.py  → gộp raw của cả 3 worker → data/snapshot/bilingual_zh_vi.parquet trên HF
      ▼
eda-bilingual_zh_vi.ipynb  → phân tích chất lượng dịch (mục 3.3)
```

3 máy (`worker_quang`, `worker_nhat_anh`, `worker_huy`) chạy song song, cùng
claim việc từ 1 Central Queue trên Firebase nên không đụng hàng nhau.

### 2.2 CRAWL TIKI (MONO VIỆT) — NGUỒN PHỤ, ĐỂ CÓ VĂN PHONG TIẾNG VIỆT TỰ NHIÊN

```
core/tiki_client.py (gọi JSON API công khai, không cần login)
      ▼
engine_tiki_detail.py → data/raw/tiki_mono_vi_*.jsonl
      ▼
scripts/snapshot_tiki_to_hf.py → data/snapshot/tiki_vi.parquet trên HF
```
Đã crawl xong 6 category × 2.200 = 13.200 sản phẩm (26/09).

### 2.3 PHÂN TÍCH CHẤT LƯỢNG DỊCH (EDA)

```
data/snapshot/bilingual_zh_vi.parquet
      ▼
eda-bilingual_zh_vi.ipynb
  mục 5: encode title_zh + title_vi bằng BAAI/bge-m3 → cosine similarity
         → so với negative baseline (cặp random / cặp cùng category bị xáo)
         → ngưỡng "chắc sai" (p50 negative) và "vùng xám" (p50–p95 negative)
         → review theo category, check chữ Hán còn sót, check cấp thuộc tính
      ▼
data/eda/*.csv, *.json   (số liệu để đưa vào báo cáo luận văn)
```

## 3. CÁC ĐIỂM CẦN LƯU Ý

1. **`eda-bilingual_zh_vi.ipynb` chưa commit** (`git status` báo untracked). Cần
   `git add` + commit nếu muốn Quang/Huy xem được, không thì nó chỉ nằm trên
   máy Nhật Anh.
2. **Máy Nhật Anh không có GPU dùng được cho notebook** (torch global là
   CPU-only). Số liệu full-data trong `data/eda/` hiện tại chạy bằng **Ollama
   bge-m3 local** (`bge-m3:latest`), không phải bản `transformers` gốc trong
   notebook — hai bản khớp nhau tới ~0.001 nên số liệu tin được, nhưng **số
   liệu chính thức nộp báo cáo nên chạy lại 1 lần trên Colab GPU** bằng đúng
   code trong notebook để không phải giải thích thêm.
3. **`data/raw/`, HF `data/bronze/`** chỉ là bản sao lưu thô — **không dùng để
   xem/phân tích**. Luôn dùng `data/snapshot/*.parquet` (đã gộp + làm sạch
   schema).
4. Data crawl bằng schema trước 17/09 (`data/raw_old/`) **không tương thích**,
   không dùng lại được.
5. Phạm vi hiện tại: 8 category (`fashion, electronics, shoes, bags, beauty,
   mother_baby, food, home`), trần 2.500 sp/category cho 1688. `auto` đã crawl
   1.701 dòng, giữ làm dữ liệu bổ sung chứ không tính vào 8 category chính.
   **Không tự seed thêm category ngoài danh sách này.**
6. `.env` chứa HF token + tài khoản đăng nhập 1688 — không đưa lên chat/PR,
   đã có trong `.gitignore`.
7. Script tạo ra 6 file trong `data/eda/title_audit/` (audit tiếng Anh/lỗi
   title) hiện **chưa nằm trong `ecom_crawler`** — mình chạy tay bằng AI trên
   máy, chưa dọn thành script chính thức trong `scripts/`. Nếu Quang/Huy cần
   chạy lại hoặc mở rộng sang `description`, nhắn mình, mình sẽ dọn lại và commit.
8. Tương tự, script tạo `data/eda/data_issues/` (mục 5) cũng đang nằm ngoài
   `ecom_crawler`. Hai file CSV trong đó là **bằng chứng**, chưa sửa gì vào data.
9. **Bản thân các file CSV kết quả** (không phải script) của mục 7, 8, cùng
   notebook và file này, đã đẩy lên repo riêng
   [`eda_ecom`](https://github.com/ntquang-0410/eda_ecom), nhánh
   `nhatanh_updating` (28/09) — xem mục 7 "Nhật ký theo ngày". Đây là nhánh
   độc lập, không chung lịch sử với `main` (của Quang) hay `giahuy`, nên
   **không có sẵn `bilingual_zh_vi.parquet`** — cần tải data riêng nếu muốn
   chạy lại notebook từ nhánh đó.

## 4. ĐÃ LÀM ĐƯỢC GÌ

### 4.1 CRAWL
- 1688 song ngữ: đang chạy tiếp theo 8 category đã chốt, `bags` đã đủ 2.500.
- Tiki mono Việt: xong 13.200 sản phẩm (6 category × 2.200), đã snapshot lên HF.

### 4.2 EDA CHẤT LƯỢNG DỊCH (`eda-bilingual_zh_vi.ipynb`, MỤC 5 CỦA NOTEBOOK)
Viết lại mục 5 (embedding similarity) theo hướng khách quan hơn thay vì so
với ngưỡng 0.5 cố định:
- Thêm **negative baseline** (cặp title random + cặp cùng category bị xáo)
  để tự suy ra ngưỡng thay vì đoán — AUC 0.9769 (so với cùng category),
  0.9964 (so với random) trên toàn bộ 16.348 dòng.
- Xếp hạng theo từng category, đọc tay 2–5% thấp nhất mỗi category thay vì
  áp 1 ngưỡng chung cho cả 9 category (food và electronics có phân bố
  similarity khác hẳn nhau).
- Regex check chữ Hán còn sót trong cột tiếng Việt.
- Chạy full 16.348 dòng (trước đó chỉ test mẫu).
- Mở rộng xuống cấp thuộc tính (`description_zh/vi`, tách theo từng cặp
  `tên: giá trị`) — 405.190 cặp thuộc tính từ 99,56% số dòng.

**Kết quả chính:**
- Similarity trung bình title: 0,704 (Ollama bge-m3). Vùng "chắc sai" (dưới
  ngưỡng p50 negative cùng category): 20 dòng (0,12%) — đa số là lỗi dịch
  thật (鸭屎香→"Phân vịt", balo→"túi du lịch nữ"...). Vùng xám: 2.032 dòng
  (12,4%) — cần đọc tay, không xoá tự động.
- Cấp thuộc tính: numbering artifact (dịch dính số thứ tự kiểu "Không1/Có2")
  8,98%; còn sót chữ Hán 1,23%.
- Kết luận: similarity chỉ bắt được lỗi dịch sai nặng, không bắt được lỗi
  chi tiết (sai số lượng, dịch chưa hết) hay lỗi format → cần thêm regex +
  đọc tay, không xoá dữ liệu chỉ dựa vào similarity thấp.

**Kết quả mục 5.5 notebook (cấp thuộc tính), chạy full 16.348 dòng — xong
28/09 lúc 19:52 (7.936 giây, 165.580 chuỗi embed, Ollama không từ chối chuỗi
nào), ghi vào đây ngày 01/10:**
- Ghép được 16.276/16.348 dòng (99,56%) → 405.190 cặp thuộc tính.
- AUC giá trị thuộc tính = **0,8183** (< 0,9): chuỗi ngắn 1–3 từ thì bge-m3
  phân biệt đúng/sai kém (vd 广州 → "Quảng Châu" vẫn điểm thấp), nên danh sách
  dưới mốc chỉ để **đọc tay, không dùng để xoá tự động**.
- Dưới mốc "chắc sai" (0,486): 34.477 cặp (8,51%).
- Hai cờ regex mới là tín hiệu đáng tin ở cấp thuộc tính: lỗi đánh số
  **36.370 cặp (8,98%)**, chữ Hán còn sót **4.971 cặp (1,23%)**.
- 32 cặp **tên** thuộc tính dưới mốc, xuất hiện tổng 1.301 lần — tên lặp trên
  nhiều sản phẩm nên sửa 1 lần được nhiều dòng.
- File kết quả nằm ở `data/eda/full_run/` (eda_attr_below_threshold.csv,
  eda_attr_names_low_sim.csv, eda_chinese_left_in_vi.csv,
  eda_title_review_by_category.csv, eda_suspect_pairs.csv, eda_summary.csv).
  **Chưa đẩy lên `eda_ecom`.**
- Lần chạy notebook này cho số title hơi khác lần `sim_full.py` ở trên: ngưỡng
  chắc sai 0,504 (17 cặp), vùng xám 11,9%, AUC 0,9779 — so với 0,506 (20 cặp),
  12,4%, 0,9769. Nhiều khả năng do mẫu cặp sai được bốc lại mỗi lần chạy nên
  p50 dịch ±0,002; chưa kiểm chứng. Cả hai đều chạy Ollama, chuỗi cắt 480 ký
  tự — vẫn cần chạy lại trên Colab GPU trước khi đưa vào báo cáo (mục 3 điểm 2).

### 4.3 AUDIT TIẾNG ANH + LỖI KHÁC TRONG `title_vi` (28/09)
Xuất 5 file vào `data/eda/title_audit/` để lọc tay:

| File | Nội dung |
|---|---|
| `title_latin_tokens.csv` | 3.937 từ Latin trong title_vi, phân 10 nhóm (cần dịch / cần xem / nên giữ) kèm đề xuất |
| `title_latin_rows_can_xu_ly.csv` | 4.912 dòng thực sự cần sửa tiếng Anh/pinyin |
| `title_issues_summary.csv` | Tổng hợp 16 loại lỗi khác (dịch sát chữ, lặp từ, mất số, viết hoa lung tung...) |
| `title_issues_rows.csv` | 3.723 dòng có ít nhất 1 lỗi thật, kèm cột trống để điền cách xử lý |
| `title_term_consistency.csv` | Cùng 1 thuật ngữ 1688 (vd 行车记录仪) bị dịch mấy kiểu khác nhau trong data |

Các vấn đề phát hiện được đã chuyển sang **mục 5 (CÁC VẤN ĐỀ CỦA DATA)**,
kèm bằng chứng theo product_id.

### 4.4 QUYẾT ĐỊNH XỬ LÝ TỪNG TỪ LATIN (`title_latin_tokens_editted.csv`, 28/09)
Từ file `title_latin_tokens.csv`, gán cho **từng từ** 1 hành động cụ thể để
Nhật Anh dò tay trước khi áp vào data. **Chưa sửa gì trong `title_vi`**, file
này chỉ là bảng quyết định.

Chính sách Nhật Anh đã chốt (28/09):
- Nhóm "tiếng Anh cần xem": AI tự quyết từng từ, từ nào không chắc thì
  đánh dấu `XEM_TAY`.
- Pinyin: địa danh / tên trà / thuật ngữ → **âm Hán Việt**; tên hãng → giữ pinyin.
- Từ mượn phổ biến (cotton, retro, vintage, size, mini, tote, sneaker...) → giữ hết.

| Hành động | Số từ | Lượt dòng | Nghĩa |
|---|---|---|---|
| `THAY` | 210 | 1.357 | Thay bằng từ tiếng Việt ở cột `thay_bang` |
| `HANVIET` | 162 | 1.260 | Pinyin → âm Hán Việt (Wuyishan → Vũ Di Sơn...) |
| `THEO_CUM` | 246 | 2.645 | Chỉ đúng khi xử lý cả cụm (cột `quy_tac_cum`, dạng `cụm => thay bằng`) |
| `XEM_TAY` | 41 | 99 | Không đủ dữ kiện để quyết, cần người đọc |
| `GIU` | 3.278 | 19.062 | Giữ nguyên (tên hãng, đơn vị, thuật ngữ, từ mượn) |

Tổng cộng ~3.138 dòng sẽ bị đổi nếu áp dụng. Cột `vi_du_sau_sua` cho xem
trước 1 câu trước/sau khi sửa; cột `anh_duyet`, `anh_ghi_chu` để trống cho
người duyệt.

Lỗi dịch "hiểu sai nghĩa" tìm được trong lúc duyệt: xem vấn đề **D03** ở mục 5.

Điểm Nhật Anh cần chốt: 回力 đang có 3 cách viết (Huili 17, Warrior 9,
Pull-back 12 dòng) — tạm để "Warrior"; "dropship" tạm thay bằng "giao hộ".

## 5. CÁC VẤN ĐỀ CỦA DATA

Phạm vi: `bilingual_zh_vi.parquet` (snapshot HF), **16.348 dòng**. Mọi con số
dưới đây đo trên toàn bộ data, không lấy mẫu.

**Bằng chứng xuất ra `data/eda/data_issues/`** (mở bằng Excel được, UTF-8 BOM):

| File | Nội dung |
|---|---|
| `data_issues_summary.csv` | 1 dòng / vấn đề: mức độ, số dòng dính, %, cách phát hiện, 3 category dính nhiều nhất, 5 product_id ví dụ |
| `data_issues_evidence.csv` | 1 dòng / (vấn đề, sản phẩm): `ma_van_de`, `product_id`, `bang_chung` (đoạn lỗi cụ thể), `category`, `title_zh`, `title_vi` — lọc theo `ma_van_de` để xem toàn bộ id dính lỗi |

**Cách xếp mức độ** (theo tác hại khi đưa vào train mô hình dịch):
1. **NGHIÊM TRỌNG**: sai nghĩa / mất thông tin → mô hình học dịch sai.
2. **CAO**: lỗi lan rộng hoặc làm hỏng cả đánh giá (rò rỉ train/test, thuật ngữ lẫn lộn).
3. **TRUNG BÌNH**: đúng ý nhưng chưa phải tiếng Việt tự nhiên (tiếng Anh, pinyin, lặp).
4. **THẤP**: định dạng.

Một dòng có thể dính nhiều vấn đề nên cột "số dòng" **không cộng dồn được**.
Số dòng dính ít nhất 1 lỗi mức 1: **551** (3,4%); dính ít nhất 1 lỗi mức 1–3:
**12.234** (74,8%).

### 5.1 BẢNG TỔNG HỢP (XẾP TỪ NẶNG TỚI NHẸ)

| Mã | Mức | Vấn đề | Số dòng | % | Dính nhiều nhất |
|---|---|---|---|---|---|
| D00 | 0 – Gốc | Phía tiếng Việt là **bản máy dịch của Alibaba**, không phải người dịch | 16.348 | 100 | toàn bộ |
| D01 | 1 | Bản dịch sai nghĩa nặng (similarity dưới ngưỡng "chắc sai") | 20 | 0,12 | food 15, beauty 4 |
| D02 | 1 | Chưa dịch / còn chữ Hán trong `title_vi` | 7 | 0,04 | fashion 2 |
| D03 | 1 | Máy dịch hiểu sai nghĩa từ / tên riêng | 156 | 0,95 | bags 102, food 19 |
| D04 | 1 | Dịch thiếu ý (bản Việt quá ngắn so với bản Trung) | 160 | 0,98 | beauty 37, mother_baby 21 |
| D05 | 1 | Mất số liệu (năm, dung tích, số lượng) | 221 | 1,35 | electronics 57, home 36 |
| D06 | 2 | `description_vi` còn chữ Hán | 2.098 | 12,83 | home 400, beauty 312 |
| D07 | 2 | Giá trị thuộc tính dính số thứ tự ("Không1", "Có2", "Khác1") | 9.286 | 56,80 | home 1.650, mother_baby 1.629 |
| D08 | 2 | Không ghép cặp được thuộc tính zh–vi (số đoạn lệch nhau) | 72 | 0,44 | home 29, food 24 |
| D09 | 2 | Thuật ngữ 1688 dịch sát chữ / không thống nhất | 562 | 3,44 | auto 315, bags 172 |
| D10 | 2 | Trùng lặp + nguy cơ rò rỉ train/test | 876 | 5,36 | home 288, mother_baby 145 |
| D11 | 3 | Tiếng Anh thừa / pinyin chưa Việt hoá | 3.138 | 19,20 | food 775, shoes 741 |
| D12 | 3 | Nguyên cụm tiếng Anh chưa dịch (≥ 3 từ liền) | 269 | 1,65 | beauty 152, food 52 |
| D13 | 3 | Lặp cụm liền nhau (2 từ gốc khác nhau bị dịch thành 1) | 506 | 3,10 | beauty 113, home 97 |
| D14 | 3 | Nghi ngờ: vùng xám similarity, cần đọc | 2.032 | 12,43 | food 747, beauty 438 |
| D15 | 4 | Viết Hoa Mỗi Từ | 518 | 3,17 | food 115, beauty 108 |
| D16 | 4 | Ký hiệu / ngoặc thừa (【】[]~\|*) | 393 | 2,40 | food 70, fashion 59 |
| D17 | 4 | Kết thúc dở dang ("... và", "... cho") | 13 | 0,08 | shoes 7 |

Ngoài lề, **không tính là lỗi**: 3.610 dòng (22,1%) có chữ quảng cáo kiểu
1688 ("bán buôn", "xuyên biên giới", "nhà máy") — dịch đúng nghĩa bản gốc.

### 5.2 BẰNG CHỨNG TỪNG VẤN ĐỀ

**D00 — Bản tiếng Việt là máy dịch.** 1688 tự sinh bản tiếng Việt bằng máy
dịch của Alibaba cho hàng thuộc pool xuyên biên giới (xem README mục nguồn dữ
liệu). Mọi lỗi D01–D17 là dấu vết điển hình của máy dịch. Hệ quả: nếu train
và đánh giá đều trên data này, mô hình chỉ học **bắt chước máy dịch của
Alibaba**, và điểm BLEU cao cũng không chứng minh được bản dịch tốt.

**D01 — Sai nghĩa nặng.** Ngưỡng 0,506 = trung vị điểm của các cặp cố ý ghép
sai trong cùng category (notebook mục 5.2).
- `735300087550`: 鸭屎香 (tên giống trà) → "**Phân vịt** Phoenix Dancong được lựa chọn cẩn thận" (sim 0,439).
- `959363637796`: 面筋 (gluten, món que cay) → "**Xương rồng** nướng kiểu Tứ Xuyên" (0,498).
- `785329535292`: title áo thun dài → "Áo phông nam Áo Phông Nam 2024 năm" (0,487, mất gần hết ý).
- `943730263382`: "香辣鸭翅湖南特色酱麻辣板鸭鸭货翅尖追剧卤味零食小吃" → "[Miễn phí vận chuyển] Cánh vịt cay sốt đặc biệt Hồ Nam" (0,488, mất vịt ép, đầu cánh, đồ ăn vặt).

**D02 — Chưa dịch.** `1043561115166`, `1058721001412`: `title_vi` giống hệt
`title_zh` (sim = 1,0). 3 dòng sót vài chữ Hán: `956583248092` "jacquard**镂空**",
`818980692803` "bị**早产**", `872054230824` "**嘎巴大红袍乌龙茶**". Lưu ý: 2/7 dòng
(`906656433533`, `831394947500`) chỉ chứa ký tự phân cách "丨", thực chất không
phải chữ chưa dịch → mức thật nhẹ hơn con số.

**D03 — Hiểu sai nghĩa** (đếm theo từ lỗi, `bang_chung` ghi rõ):

| Gốc Trung (nghĩa đúng) | Máy dịch thành | Số dòng | Ví dụ id |
|---|---|---|---|
| 密码箱 (vali khóa số) | "hộp mật khẩu" | 97 | `1000087007906` |
| 浓香型 (hương đậm) | "Luzhou" | 14 | xem CSV |
| 回力 (hãng Warrior) | "Pull-back" | 12 | xem CSV |
| 冰岛 (làng trà Băng Đảo) | "Iceland" | 5 | xem CSV |
| 老花 (họa tiết monogram) | "Presbyopic" (lão thị) | 5 | xem CSV |
| 膜 (mặt nạ / màng) | "phim" → "rạp chiếu phim" | 5 | xem CSV |
| 苹果 (quả táo) | "Apple" (hãng) | 2 | `895595225521` |
| 小叮当 (Doraemon), 杰里 (chip Jieli), 大全 (đủ loại)... | "Tinkerbell", "Jerry", "Daquan"... | 1–3 mỗi từ | xem CSV |
| 雪梨 (lê tuyết) | "Sydney" | 1 | xem CSV |

Thêm 1 ví dụ đọc tay, chưa đưa vào CSV: `906656433533` 【一折专区】 (bán còn
10% giá = giảm 90%) → "[Giảm giá 10%]" — **ngược nghĩa**.

**D04 — Dịch thiếu.** Ngưỡng: tỉ lệ (số từ vi / số chữ Hán) < 0,52 (phân vị 1%).
- `41907586179`: "SNJUE8合1泡沫高压洗车水枪车载高压水枪洗车用品家用洗车器" → "Súng nước rửa xe áp lực cao SNJUE8 trong 1 bọt".
- `713481760562`: "...防撞条车用碳纤纹车身划痕防刮蹭加宽保护胶条跨境" → "Dải chống va chạm cản trước và cản sau ô tô" (mất "vân carbon, chống xước, bản rộng").

**D05 — Mất / sai số.** Không chỉ mất mà còn **đổi số**:
- `704753221535`, `714122581154`, `744356768181`: "2026新款" → "mẫu mới **2023**"; `752960812421`: 2026 → "**2025**".
- `1001659294692`: "印花24寸登机密码箱" → mất "24 inch" (kích thước vali).
- `826353007905`: "2024爆款..." → bản Việt mất "2024".

Đã loại các trường hợp hợp lệ như 26年 → 2026, 100.000 ↔ 100000. Còn báo nhầm
lẻ tẻ, vd `749364410345` (zh có cả "70迈" lẫn "70mai", vi gộp còn 1 lần "70mai").

**D06 — Description còn chữ Hán.** Phần lớn là **tên thuộc tính chưa dịch**:
`1000904160405` còn "内存容量 颜色分类 前录高清 ...", `530561037745` còn
"传输数率 电压", `564223667080` còn "产地 浙江".

**D07 — Dính số thứ tự.** 36.370 cặp thuộc tính (8,98% số cặp) nhưng rải trên
**9.286 dòng (56,8%)**. `41907586179`: 是 → "**Là1**"; `1001744080768`:
"速卖通, 独立站, LAZADA, 其他" → "AliExpress, Trạm độc lập, LAZADA, **Khác1**".

**D08 — Không ghép cặp.** `596045303562`: zh 12 đoạn / vi 13 đoạn;
`720257410711`: 28 / 29 → ghép theo vị trí sẽ lệch từ chỗ lệch trở đi.

**D09 — Thuật ngữ.** `1000787279495`: 行车记录仪 → "Máy ghi âm lái xe" (đúng:
camera hành trình). Cùng 1 từ gốc có nhiều cách dịch (`title_term_consistency.csv`):
行车记录仪 519 dòng chia ra "camera hành trình" / "máy ghi âm lái xe" / "máy
ghi hình lái xe"; 拉杆箱 → "vali kéo" / "vali xe đẩy" / "hộp xe đẩy"; 补水 →
"dưỡng ẩm" 619 / "cấp ẩm" 54 / "cấp nước" 33.

**D10 — Trùng lặp.** 402 dòng trùng hẳn cặp (zh, vi) với dòng khác; 474 dòng
cùng `title_zh` nhưng `title_vi` khác (vd `523307015972`, `670224806960`). Nếu
chia ngẫu nhiên, cùng một câu gốc sẽ rơi vào cả train lẫn test → điểm test bị
thổi phồng.

**D11 — Tiếng Anh / pinyin.** Theo bảng quyết định `title_latin_tokens_editted.csv`
(mục 4.4): 3.138 dòng chứa ≥ 1 từ cần THAY / HANVIET / THEO_CUM / XEM_TAY.
Cột `bang_chung` liệt kê từng từ + hành động, vd `wuyishan[HANVIET]`, `casual[THAY]`.

**D12 — Cụm tiếng Anh.** `694399617240` "Net Red Ribbon" (网红丝带);
`609186431641` "Jeep Compass Free Light Grand Cherokee Renegade Hidden"
(tên xe đúng, nhưng "Hidden" = 隐藏式 bị để tiếng Anh).

**D13 — Lặp cụm.** `523007667240` "rèm che nắng rèm che nắng" (gốc: 遮阳帘 +
遮光 là 2 ý khác nhau); `611247055426` "đệm thắt lưng, đệm thắt lưng ô tô"
(护腰靠垫 + 汽车腰靠). Điển hình nhất: 补水保湿 → "dưỡng ẩm, dưỡng ẩm".

**D14 — Vùng xám.** Similarity trong [0,506; 0,635) = giữa trung vị và phân
vị 95% của cặp sai cố ý. Chưa chắc sai, cần đọc; food chiếm 37%.

**D15–D17 — Định dạng.** Viết hoa mỗi từ ≥ 60% số từ; ký hiệu 【】[]~|*;
kết thúc bằng "và/cho/của" hoặc dấu phẩy. Id đầy đủ trong CSV.

## 6. GIẢI PHÁP CHI TIẾT CHO TỪNG VẤN ĐỀ

**Nguyên tắc chung**, có cơ sở từ tài liệu:
- **Sửa rẻ và chắc trước, lọc sau, đọc tay cuối cùng.** Nhiều lỗi ở đây có
  quy luật (cùng 1 từ gốc bị dịch sai cùng một kiểu), nên sửa bằng từ điển có
  điều kiện theo bản gốc sẽ rẻ hơn nhiều so với đọc từng dòng.
- **Ưu tiên lọc bỏ cặp sai nghĩa.** NMT bị hại bởi cặp dịch lệch nghĩa và cặp
  chưa dịch nhiều hơn các loại nhiễu khác; cặp chưa dịch còn khiến mô hình học
  cách chép nguyên câu nguồn (Khayrallah & Koehn, 2018).
- **Chọn ít mà tốt thắng nhiều mà bẩn.** Lọc bằng điểm chất lượng (QE) có thể
  bỏ một nửa data mà chất lượng dịch vẫn tăng (Peter et al., 2023).
- **Mọi thay thế tự động phải có điều kiện bên tiếng Trung.** Ví dụ chỉ đổi
  "hộp mật khẩu" → "vali khóa số" khi `title_zh` có 密码箱, để không sửa nhầm
  ngữ cảnh khác.

### 6.1 THỨ TỰ THỰC HIỆN ĐỀ XUẤT

| Bước | Việc | Vấn đề giải quyết | Chi phí |
|---|---|---|---|
| 1 | Sửa cấu trúc: ghép lại thuộc tính theo `fid`, xoá số thứ tự dính | D08, D07 | Thấp (script) |
| 2 | Từ điển thay thế có điều kiện theo bản gốc | D03, D09, D11, D13 | Trung bình (duyệt từ điển) |
| 3 | Lọc cứng: sai nghĩa, chưa dịch, thiếu ý, mất số | D01, D02, D04, D05 | Thấp + đọc ~560 dòng |
| 4 | Khử trùng lặp + chia train/dev/test theo nhóm câu gốc | D10 | Thấp |
| 5 | Chấm điểm QE lần 2, đọc giao của 2 tín hiệu nghi ngờ | D14, D12 | Trung bình |
| 6 | Chuẩn hoá định dạng (cả 2 phía) | D15, D16, D17 | Thấp |
| 7 | Dựng tập test do người hiệu đính + gắn tag chất lượng | D00 | Cao nhưng bắt buộc |

### 6.2 GIẢI PHÁP THEO TỪNG VẤN ĐỀ

**D00 — Bản Việt là máy dịch**
- (a) **Tập dev/test do người hiệu đính**, khoảng 500–1.000 cặp chia đều
  category, lấy từ các cặp *không* nằm trong train. Đây là cách duy nhất để đo
  "dịch tốt" chứ không phải "giống máy Alibaba". Có thể tham khảo thêm tập test
  zh–vi 1.000 cặp của VLSP 2022–2023 (Tran et al., 2025) làm đánh giá ngoài
  miền (cần kiểm tra quyền truy cập).
- (b) **Gắn tag chất lượng** cho từng cặp khi train (vd `<raw>` = máy dịch
  nguyên bản, `<fixed>` = đã sửa/duyệt), lúc dịch thì dùng tag tốt. Đây là cùng
  ý tưởng với tagged back-translation (Caswell et al., 2019) và tag
  translationese (Riley et al., 2020): cho mô hình biết đâu là dữ liệu kém tự
  nhiên để nó không bắt chước.
- (c) **Tận dụng 13.200 sản phẩm Tiki (tiếng Việt thật)**: back-translation
  vi→zh có tag để thêm phía đích tự nhiên; hoặc train bộ phân loại "máy dịch
  vs tiếng Việt gốc" như Riley et al. (2020).
- *Rủi ro:* (a) tốn công người, cần 2 người đọc chéo; (b)(c) thêm bước train,
  back-translation cũng là máy dịch nên phải có tag, nếu không sẽ thêm nhiễu.

**D01 — Sai nghĩa nặng (20 dòng)**
- Số lượng nhỏ: đọc cả 20 dòng. Dòng sai hẳn thì bỏ khỏi train (hoặc dịch lại
  tay nếu muốn giữ, 20 dòng chỉ mất ~30 phút). Không xoá tự động theo điểm.
- *Rủi ro:* có dòng điểm thấp chỉ vì bản Việt để nhiều tiếng Anh/pinyin chứ
  không sai nghĩa (vd `822969947362` "Yizhichun Hyaluronic Acid Hydrating
  Mask...") → phải đọc trước khi bỏ.

**D02 — Chưa dịch (7 dòng)**
- Bỏ 2 dòng chưa dịch hẳn (`title_vi` = `title_zh`), vì cặp chép nguyên dạy mô
  hình chép nguồn (Khayrallah & Koehn, 2018). 3 dòng còn sót vài chữ Hán
  thì dịch tay đoạn đó. Riêng "丨" thì thay bằng "|" ở cả 2 phía.
- Thêm luật chặn vào pipeline: tỉ lệ ký tự Hán trong `title_vi` > 0 → cờ đỏ
  (bicleaner-hardrules cũng có luật "sai ngôn ngữ" / "câu trùng nguồn").
- *Rủi ro:* không đáng kể.

**D03 — Hiểu sai nghĩa (156 dòng)**
- Lỗi có quy luật → **từ điển sửa lỗi có điều kiện**: mỗi mục gồm (chữ Hán
  nguồn, cụm sai, cụm đúng), chỉ áp khi `title_zh` chứa chữ Hán nguồn. Ví dụ
  (密码箱, "hộp mật khẩu", "vali khóa số"); (雪梨, "Sydney", "lê tuyết").
  Danh sách ban đầu lấy từ `bang_chung` của D03 + cột `ghi_chu` của bảng từ.
- Sau khi áp, chạy lại similarity cho các dòng đã sửa để kiểm tra điểm tăng.
- *Rủi ro:* áp sai ngữ cảnh (vd "Apple" vừa là hãng vừa là 苹果 quả táo) —
  chính vì vậy phải có điều kiện bên tiếng Trung. Từ điển chỉ phủ lỗi đã biết;
  lỗi kiểu này chắc chắn còn ở dòng khác chưa phát hiện.

**D04 — Dịch thiếu (160 dòng)**
- Bỏ khỏi train, hoặc dịch lại những dòng thuộc category ít data. Briakou &
  Carpuat (2021) chỉ ra cặp "gần tương đương nhưng lệch vài chỗ" làm NMT sinh
  câu thoái hoá và kém tự tin, nên không nên giữ nguyên.
- *Rủi ro:* bỏ 160 dòng (~1%) không đáng kể; riêng beauty mất 37 dòng.
  Ngưỡng 0,52 là phân vị nên có vài dòng ngắn mà vẫn đủ ý → đọc lướt trước.

**D05 — Mất số (221 dòng)**
- Luật "số phải khớp" (giống `NonZeroNumeralsFilter` của OpusFilter và luật
  "inconsistent numbers" của bicleaner-hardrules) sau khi chuẩn hoá (26年 →
  2026, 100.000 → 100000, chữ số Hán 三 → 3). Dòng còn lệch: số là thông số
  sản phẩm (dung tích, kích thước, số lượng) thì sửa hoặc bỏ; số là năm/mã
  quảng cáo thì có thể bỏ qua.
- Riêng lỗi **đổi năm** (2026 → 2023/2025) là lỗi hệ thống của máy dịch → sửa
  tự động được: nếu `title_zh` có năm 20xx mà `title_vi` có năm khác thì thay
  bằng năm bên Trung.
- *Rủi ro:* báo nhầm khi số lặp trong tên model (zh có cả "70迈" lẫn "70mai",
  vi chỉ giữ 1 lần). Luật số không bắt được số bị **dịch sai đơn vị** (2斤 → "2 con").

**D06 — Description còn chữ Hán (2.098 dòng)**
- Phần lớn là **tên thuộc tính** (传输数率, 电压, 产地...), mà số lượng tên
  thuộc tính khác nhau là hữu hạn → dựng **từ điển tên thuộc tính zh→vi**
  (dịch 1 lần, duyệt 1 lần) rồi áp cho mọi dòng. Giá trị thuộc tính còn chữ Hán
  thì **bỏ riêng cặp thuộc tính đó**, không bỏ cả dòng.
- *Rủi ro:* từ điển dịch sai 1 tên sẽ lan ra hàng trăm dòng → bắt buộc duyệt tay.

**D07 — Dính số thứ tự (9.286 dòng)**
- Regex xoá đuôi `(Không|Có|Là|Khác)\.?\d+$` → "Không", "Có", "Là", "Khác".
  **Chỉ xoá khi giá trị bên Trung không có con số đó**, để không xoá số thật.
- *Rủi ro:* thấp; cần kiểm ngẫu nhiên khoảng 100 cặp sau khi sửa.

**D08 — Không ghép cặp (72 dòng)**
- Snapshot đã có `attributes_zh` / `attributes_vi` kèm `fid` → **ghép theo
  `fid`** thay vì theo vị trí; `fid` nào chỉ có 1 phía thì bỏ.
- *Rủi ro:* không đáng kể.

**D09 — Thuật ngữ sát chữ / không thống nhất (562 dòng)**
- **Bảng thuật ngữ chuẩn zh→vi** cho các từ 1688 hay gặp (khởi đầu từ
  `title_term_consistency.csv`). Chọn cách dịch theo tần suất người bán Việt
  thật dùng (đếm trên Tiki), vd 行车记录仪 → "camera hành trình". Chuẩn hoá phía
  Việt có điều kiện theo từ gốc.
- Về lâu dài: huấn luyện mô hình tôn trọng thuật ngữ bằng cách chèn thuật ngữ
  đích vào câu nguồn lúc train (Dinu et al., 2019), để khi dùng thật chỉ cần
  đưa bảng thuật ngữ vào, không phải train lại.
- *Rủi ro:* chọn thuật ngữ chuẩn mang tính chủ quan, nhóm nên chốt chung; chuẩn
  hoá làm giảm đa dạng từ vựng phía Việt.

**D10 — Trùng lặp + rò rỉ (876 dòng)**
- Trùng hẳn: giữ 1. Cùng zh khác vi: giữ bản có điểm chất lượng cao nhất, hoặc
  giữ cả hai **chỉ trong train** (coi như nhiều bản tham chiếu).
- **Chia train/dev/test theo nhóm**: mọi dòng cùng `title_zh` (và các dòng gần
  trùng theo MinHash, pipeline đã có) phải rơi vào cùng 1 tập. Lee et al.
  (2022) cho thấy data gần trùng làm mô hình thuộc lòng và gây chồng lấn
  train–test, làm điểm đánh giá bị thổi phồng.
- *Rủi ro:* bỏ bản dịch khác nhau thì mất đa dạng diễn đạt; ngưỡng MinHash quá
  thấp có thể gộp nhầm 2 sản phẩm khác nhau nhưng tiêu đề na ná.

**D11 — Tiếng Anh thừa / pinyin (3.138 dòng)**
- Áp bảng `title_latin_tokens_editted.csv` sau khi Nhật Anh duyệt xong (mục 4.4).
- Pinyin → Hán Việt: tra theo chữ Hán gốc (đã dò bằng `pypinyin`) trong bộ
  âm Hán Việt mở, vd `chinese-hanviet-cognates` (dựa trên từ điển Thiều Chửu)
  hoặc `standard-han-nom`; tên thương hiệu thì giữ pinyin (chính sách đã chốt).
- *Rủi ro:* chữ đa âm (1 chữ Hán có nhiều âm Hán Việt, vd 莆 = "Bồ", AI từng
  đọc nhầm "Phủ") → tên quen dùng (Phổ Nhĩ, Thiết Quan Âm...) phải có danh
  sách cố định duyệt tay. CVDICT (122.000 mục, CC BY-SA 4.0) có nghĩa tiếng
  Việt nhưng phần lớn được dịch bằng ChatGPT-4o, nên chỉ dùng tham khảo. Thay
  từng từ có thể làm sai trật tự từ tiếng Việt → cụm dài xử lý ở D12.

**D12 — Nguyên cụm tiếng Anh (269 dòng)**
- Dịch lại **riêng cụm đó**: người dịch, hoặc LLM hậu biên tập (APE) kèm ràng
  buộc "giữ nguyên tên hãng / model" rồi người duyệt. Raunak et al. (2023)
  cho thấy GPT-4 hậu biên tập hiệu quả, nhưng có lúc **bịa nội dung** khi sửa.
- *Rủi ro:* LLM bịa hoặc dịch lại cả những phần đúng → chỉ cho sửa đúng cụm bị
  đánh dấu, rồi so similarity trước/sau.

**D13 — Lặp cụm (506 dòng)**
- Không xoá lặp một cách máy móc, vì bên Trung thực sự có 2 ý (补水 ≠ 保湿):
  xoá thì bản Việt thành thiếu ý (quay lại lỗi D04). Cách đúng là đưa vào bảng
  thuật ngữ D09 để 2 từ gốc ra 2 từ Việt khác nhau (补水 → "cấp ẩm", 保湿 →
  "dưỡng ẩm").
- *Rủi ro:* chỉ sửa được khi biết cặp từ gốc; lặp do chính bản Trung nhồi từ
  khoá thì giữ nguyên là đúng.

**D14 — Vùng xám (2.032 dòng)**
- Không xoá. Chấm thêm một tín hiệu độc lập với bge-m3: điểm QE không cần bản
  tham chiếu như CometKiwi (Peter et al., 2023), hoặc LaBSE. **Chỉ đọc tay
  những dòng bị cả 2 tín hiệu đánh giá thấp**, như vậy số dòng cần đọc giảm
  mạnh.
- *Rủi ro:* CometKiwi train trên câu văn bình thường, còn title 1688 là chuỗi
  từ khoá → điểm có thể lệch; cần hiệu chỉnh trên khoảng 200 cặp đã gán nhãn
  tay. Model `wmt22-cometkiwi-da` dùng giấy phép phi thương mại (hợp với luận
  văn, nhưng cần kiểm lại trên Hugging Face). Chạy trên máy không GPU sẽ chậm
  → nên chạy trên Colab.

**D15–D17 — Định dạng (518 / 393 / 13 dòng)**
- D15: chuẩn hoá về viết thường đầu câu, nhưng **giữ nguyên** từ nằm trong danh
  sách GIU/HANVIET của bảng từ (tên hãng, tên riêng) và từ viết hoa toàn bộ
  (USB, LED). bicleaner-hardrules cũng coi titlecase là tín hiệu nhiễu.
- D16: 【】 → [ ], bỏ ~ | * trang trí; áp **cả 2 phía** để giữ song song.
- D17: đọc tay 13 dòng.
- *Rủi ro:* hạ chữ hoa nhầm tên riêng chưa có trong danh sách → chạy thử trên
  mẫu trước.

### 6.3 TÀI LIỆU THAM KHẢO

- Khayrallah & Koehn (2018), *On the Impact of Various Types of Noise on NMT* — https://aclanthology.org/W18-2709/
- Koehn et al. (2019), *Findings of the WMT 2019 Shared Task on Parallel Corpus Filtering* — https://aclanthology.org/W19-5404/
- Peter et al. (2023), *There's No Data like Better Data: Using QE Metrics for MT Data Filtering* — https://aclanthology.org/2023.wmt-1.50/
- Caswell et al. (2019), *Tagged Back-Translation* — https://aclanthology.org/W19-5206/
- Riley et al. (2020), *Translationese as a Language in "Multilingual" NMT* — https://aclanthology.org/2020.acl-main.691/
- Dinu et al. (2019), *Training NMT to Apply Terminology Constraints* — https://aclanthology.org/P19-1294/
- Briakou & Carpuat (2021), *Beyond Noise: Fine-grained Semantic Divergences in NMT* — https://aclanthology.org/2021.acl-long.562/
- Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better* — https://aclanthology.org/2022.acl-long.577/
- Raunak et al. (2023), *Leveraging GPT-4 for Automatic Translation Post-Editing* — https://aclanthology.org/2023.findings-emnlp.804/
- Feng et al. (2022), *Language-agnostic BERT Sentence Embedding (LaBSE)* — https://aclanthology.org/2022.acl-long.62/
- Aulamo et al. (2020), *OpusFilter* — https://aclanthology.org/2020.acl-demos.20/ · https://github.com/Helsinki-NLP/OpusFilter
- Bicleaner hard-rules / Bicleaner AI — https://github.com/bitextor/bicleaner-hardrules · https://github.com/bitextor/bicleaner-ai (model đa ngữ hiện chỉ ghép với tiếng Anh; zh–vi phải tự train)
- COMET / CometKiwi — https://github.com/Unbabel/COMET
- Tran et al. (2025), *VLSP 2022–2023 MT Shared Tasks (Vietnamese–Chinese)* — https://arxiv.org/abs/2501.08621
- Âm Hán Việt: https://github.com/ryanphung/chinese-hanviet-cognates · https://github.com/votaquangnhat/standard-han-nom · CVDICT https://github.com/ph0ngp/CVDICT

## 7. NHẬT KÝ THEO NGÀY

### 28/09/2026

Session bị gián đoạn giữa chừng (task chạy nền `run_nb_harness.py` bị dừng
theo phiên trước), phải khởi động lại lúc 16:21 bằng tiến trình tách rời
(WMI `Win32_Process.Create`) để không chết theo cửa sổ Claude — output vẫn
đang chạy tới mục 5.5 (description theo cặp thuộc tính), chưa có số liệu.

**Prompt 1:**
> "Tui thấy trong title có những từ Tiếng anh lẫn lộn không cần thiết vì mô
> hình tui không sử dụng tiếng anh, ngoại trừ tên hãng, đơn vị cm, ml, các
> tên như wifi á. Bạn hãy xuất những trường hợp tiếng anh đó kèm số lượng
> dòng sử dụng từ đó, có thể đề xuất cách xử lý vì tui dự định lọc bằng tay.
> Sau đó bạn hãy coi trong tille đó còn vấn đề gì nữa không, hãy báo cho tui
> và đề xuất cách xử lý"

→ Viết script `title_audit.py`: tách token Latin trong `title_vi`, phân
nhóm bằng luật (đơn vị/số, thuật ngữ kỹ thuật, tên riêng, từ mượn phổ biến —
đối chiếu tần suất trên chính bộ Tiki để biết người Việt có hay dùng từ đó
không, phần còn lại tách tiếp bằng bộ tách âm tiết pinyin để phân biệt "tên
riêng Trung Quốc" với "từ tiếng Anh thật"), cộng thêm 1 loạt kiểm tra khác
(chữ Hán còn sót, số bị mất, lặp từ, thuật ngữ dịch không thống nhất...).

Vài chỗ bị sai/phải sửa lại trong lúc làm — ghi lại cho minh bạch:
- Regex dò chữ Hán (khoảng mã U+4E00–U+9FFF) viết kiểu raw string bị
  pandas/Arrow (backend RE2) từ chối vì không hiểu escape `\u` — lỗi này đã
  gặp 1 lần trước đó rồi mà vẫn dính lại, phải chuyển các chỗ dùng
  `.str.contains/.fullmatch` kiểu này sang regex Python thường
  (`re.compile` + `.map`).
- Bản tách pinyin đầu tiên quá "nhạy", bắt nhầm cả từ tiếng Anh/hoạt chất
  (hoodie, berlin, nicotinamide) vì ghép phụ âm đầu + vần quá rộng, sinh ra
  những âm không có thật — phải viết lại bảng âm tiết hợp lệ và giới hạn
  chỉ áp dụng cho từ ít gặp trên Tiki.
- Bảng "term consistency" ban đầu đưa số bị đếm chồng lấn (mẫu ngắn nằm lọt
  trong mẫu dài, ví dụ "xe đẩy" bao luôn cả "vali xe đẩy") — đã đính chính
  lại số cho anh ngay trong cùng lượt trả lời.

**Prompt 2 (sau khi task nền báo dừng giữa chừng):**
> "lỗi gì vậy, sao không chạy nữa"

→ Không phải lỗi code — phiên làm việc trước bị ngắt nên tiến trình chạy
nền chết theo. Khởi động lại bằng tiến trình tách rời khỏi phiên Claude
(WMI), vẫn đang chạy.

**Prompt 3:**
> "giải thích ý nghĩa của các file csv mà bạn xuất ra để tui hiểu, ý nghĩa"

→ Giải thích từng cột trong 5 file ở `data/eda/title_audit/`, gồm cả cách
đọc số liệu và cột nào để trống cho anh tự điền khi lọc tay.

**Prompt 4:**
> "Bạn viết cho tui file WALKTHROUGH.md để cho team tui biết tiến độ làm
> việc..."

→ File này.

**Prompt 5:**
> "Bạn sửa file số 1 title_latin_tokens.csv theo đề xuất của bạn nha, khi
> sửa xong hãy xuất file title_latin_tokens_editted.csv để tui dò lại và
> kiểm tra bằng tay, sau đó sẽ kêu bạn sửa thêm. Nếu không rõ chỗ nào thì
> phải hỏi tui cho tới khi bạn hiểu rõ"

→ Hỏi lại 3 câu (nhóm "cần xem" xử lý thế nào, chính sách pinyin, từ mượn),
Nhật Anh chọn cả 3 phương án đề xuất (xem 4.4). Duyệt tay đủ 825 từ pinyin
(dùng `pypinyin` dò chữ Hán gốc trong `title_zh` rồi đọc âm Hán Việt) và
1.277 từ tiếng Anh, xuất `data/eda/title_audit/title_latin_tokens_editted.csv`.

Sai/sửa trong lúc làm:
- Báo cáo hôm trước ghi 莆田 = "Phủ Điền" là **sai**, âm Hán Việt đúng là
  **Bồ Điền** — đã sửa trong file mới.
- Bản xem trước lần 1 viết hoa từ thay vào giữa câu ("Wuyi Hương đậm"),
  "Luzhou hương vị" ra "hương đậm hương vị", quy tắc giữ "3 Series" bị bước
  thay từ đơn đè lên thành "3 dòng", "Baigou New City" thành "... mới City"
  → sửa: chỉ viết hoa ở đầu câu; cụm đã xử lý được khoá lại; với
  `THEO_CUM` chỉ thay từ đơn khi hai bên là tiếng Việt, cạnh từ Latin khác
  (dễ là tên riêng) thì để nguyên cho người duyệt.
- Cài `pypinyin` vào thư mục tạm của phiên (không cài vào `.venv` của repo).

**Prompt 6 (lệnh mặc định tới hết project):**
> "Mỗi lần tui prompt hay bạn thay đổi những gì, tất cả những phiên làm việc
> của tui và bạn, bạn đều phải cập nhật vào file WALKTHROUGH.md để tránh bị
> thất lạc tiến độ..."

→ Từ giờ mọi prompt / thay đổi đều ghi vào file này (mục nhật ký, nay là
mục 7 sau khi đánh số lại), cuối mỗi câu trả lời AI ghi dòng "ĐÃ CẬP NHẬT VÀO
file WALKTHROUGH".

**Chạy nền mục 5.5 (description):** lần chạy lúc 16:21 **chết lúc 17:22** ở
cell 5.5 — Ollama trả HTTP 400 với 1 lô chuỗi thuộc tính (log lỗi giữ ở
`data/eda/full_run/run_failed_http400.log`). Đã sửa script chạy thử: lô nào
lỗi thì tách ra thử từng chuỗi, chuỗi bị từ chối ghi log + thay bằng "." rồi
chạy tiếp. Chạy lại lúc ~17:40.

**Prompt 7:**
> "Trong file WALKTHROUGH.MD, thực hiện các yêu cầu sau nếu nội dung không
> nằm ở những yêu cầu này, đề nghị giữ nguyên nội dung trong đó: 1. Hãy tách
> mục CÁC VẤN ĐỀ CỦA DATA ra mục riêng ... mỗi vấn đề phải đi kèm bằng chứng
> ... xếp theo thứ tự các vấn đề lỗi nặng tới nhẹ nhất ... có thể xuất file
> csv để dễ quan sát 2. Thêm mục GIẢI PHÁP CHI TIẾT CHO TỪNG VẤN ĐỀ ĐÃ NÊU ở
> trên, giải pháp phải tối ưu, bạn phải search trên toàn bộ các link, bài
> báo,... và các giải pháp đó có rủi ro gì không ... 3. Các đề mục viết in
> hoa lên để làm nổi bật"

→ Đã làm:
- Viết script `data_issues.py` gom mọi vấn đề (D00–D17) thành bảng xếp mức
  độ + bằng chứng theo product_id → `data/eda/data_issues/data_issues_summary.csv`
  và `data_issues_evidence.csv`.
- Thêm **mục 5 CÁC VẤN ĐỀ CỦA DATA** và **mục 6 GIẢI PHÁP CHI TIẾT**. Giải pháp
  dựa trên tài liệu đã tra (WMT filtering, OpusFilter, bicleaner, QE filtering,
  tagged back-translation, translationese, thuật ngữ, khử trùng lặp, LLM
  post-editing, dữ liệu âm Hán Việt), có nêu rủi ro từng giải pháp.
- Chuyển đoạn "Phát hiện đáng chú ý" (4.3) và "Lỗi dịch nặng" (4.4) sang mục 5,
  để lại dòng trỏ tới mục 5. Đánh số lại: nhật ký từ mục 5 → **mục 7**.
- Viết hoa toàn bộ đề mục; tên file trong dấu `...` giữ nguyên để link không hỏng.

Sai/sửa trong lúc làm:
- Lần chạy đầu, D01 dùng ngưỡng riêng từng category (ra 6 dòng) và D14
  (1.188 dòng), không khớp số 20 / 2.032 đã ghi ở mục 4.2 (ngưỡng chung của
  notebook). Đã đổi về ngưỡng chung cho nhất quán.
- Soát lại ví dụ: `834457793386` (sim 0,510) bị đưa nhầm vào D01 dù trên
  ngưỡng → thay ví dụ khác. D02 đếm "丨" là chữ Hán chưa dịch (2/7 dòng thực
  ra chỉ là ký tự phân cách) → đã ghi chú. Ví dụ D05 đầu tiên (`749364410345`)
  là báo nhầm → thay bằng ví dụ **đổi năm 2026 → 2023** do máy dịch.
- Phát hiện mới khi soát: 【一折专区】 (bán còn 10% giá = giảm 90%) bị dịch
  thành "[Giảm giá 10%]" — ngược nghĩa.
- Công cụ chạy lệnh / tìm web bị lỗi kiểm duyệt phía server một lúc (không
  phải lỗi của script), phải chờ rồi chạy lại.

**Prompt 8:**
> "Bạn hãy commit, tạo nhánh tên nhatanh_updating trên repo:
> https://github.com/ntquang-0410/eda_ecom, sau đó push các file cần thiết,
> quan trọng lên nhánh đó, bắt buộc phải có file WALKTHROUGH.md, thắc mắc gì
> cứ hỏi tui nhé"

→ Repo `eda_ecom` (khác `ecom_crawler`) do Quang tạo riêng cho phần EDA, có
sẵn nhánh `main` (data + notebook + audit tiếng Anh của Quang) và nhánh
`giahuy` (notebook + walkthrough riêng của Huy, không chung lịch sử với
main). Hỏi lại 2 câu trước khi push (nền nhánh dựa trên gì, có kèm script
Python hay không) — Nhật Anh chọn: nhánh độc lập kiểu `giahuy`, chưa kèm
script (script hiện đang hard-code đường dẫn máy Nhật Anh, để sau).

Đã tạo nhánh `nhatanh_updating` (orphan, không chung lịch sử với `main`/`giahuy`),
push: `eda-bilingual_zh_vi.ipynb`, `WALKTHROUGH.md`, `README.md`, `.gitignore`,
`data/eda/title_similarity_full.csv`, `data/eda/title_similarity_summary.json`,
6 file trong `data/eda/title_audit/`, 2 file trong `data/eda/data_issues/` —
giữ nguyên cấu trúc thư mục `data/eda/...` như trong `ecom_crawler` để các
đường dẫn nhắc trong file này vẫn đúng. **Không** kèm `bilingual_zh_vi.parquet`
(25MB, đã có sẵn ở nhánh `main`) vì chọn nhánh độc lập.

Nhân tiện sửa 2 chỗ trong file này cho khớp thực tế: mục 3 điểm 7 (5 file →
6 file, vì đã có thêm `title_latin_tokens_editted.csv` từ Prompt 5) và câu mở
đầu (link `README.md` trỏ nhầm sang chính nó thay vì repo `ecom_crawler`).

**Prompt 9:**
> "Tui mới thêm vào project này file CLAUDE.md, tập hợp các quy tắc mà bạn
> phải tuân theo từ đầu tới khi kết thúc project, bạn đọc kĩ đi và cho tui ý
> kiến, nó viết bằng Tiếng anh á"

→ Đọc `ecom_crawler/CLAUDE.md` (4 mục quy tắc hành vi chung: Think Before
Coding, Simplicity First, Surgical Changes, Goal-Driven Execution — dạng
template giảm lỗi LLM code, không phải quy tắc riêng dự án). Nhận xét: Rule 3
(Surgical Changes) khớp với quy tắc VXB đã có; Rule 1 ("không chắc thì phải
hỏi") có thể khác với cách làm bên VXB (tự chốt mặc định + ghi "chờ góp ý")
nhưng 2 dự án khác nhau nên không xung đột; file thiên về code, ít khớp với
việc phân tích data/viết báo cáo đang làm; file tự ghi "merge with
project-specific instructions as needed" → đề xuất thêm 1 mục riêng cho
`ecom_crawler`.

**Prompt 10:** "Bạn định thêm những gì" → liệt kê đề xuất (2 nhóm: quy tắc
hành vi rút từ thực tế + known gotchas kỹ thuật đã dính ≥2 lần), chưa sửa file.

**Prompt 11:** "9 category chứ" → Nhật Anh chỉnh đúng: **9 category** đang có
trong data (8 category chính đã chốt + `auto` là dữ liệu bổ sung giữ lại,
không tính vào 8 category chính nhưng vẫn có trong data/phân tích), không
phải 8 như em ghi nhầm ban đầu. Đã sửa lại đề xuất.

Hỏi lại 1 câu (viết mục mới bằng tiếng Anh hay tiếng Việt) — Nhật Anh chọn
tiếng Anh để khớp văn phong file. Đã thêm **mục 5 "Project-Specific:
ecom_crawler"** vào `CLAUDE.md`: 9 quy tắc hành vi (WALKTHROUGH bắt buộc,
phạm vi 9 category, không dùng `data/raw`/`bronze`/`raw_old` để phân tích,
không để lộ `.env`, không kill tiến trình của anh, chạy nền qua WMI, không
cài package vào `.venv` repo, số liệu CPU/Ollama chỉ tạm — phải đối chiếu
Colab GPU trước khi báo cáo, quy ước nhánh độc lập + tài khoản GitHub của
repo `eda_ecom`) + 2 known gotchas (heredoc PowerShell không chạy được với
`git commit -m`; regex `\u` bị pandas Arrow/RE2 từ chối, phải dùng `re` của
Python thường).

**Việc còn nợ:**
- Số liệu full-data của notebook mục 5.5 (description theo cặp thuộc tính) —
  chạy nền xong sẽ cập nhật vào mục 4.2, kèm danh sách chuỗi Ollama từ chối.
- Nhóm chốt thứ tự làm ở mục 6.1 trước khi viết script sửa data thật.
- Nhật Anh duyệt `title_latin_tokens_editted.csv` (41 từ `XEM_TAY` + cột
  `anh_duyet`), sau đó mới viết script áp vào `title_vi`.
- Script audit (title_audit.py, data_issues.py, tok_prep.py, dec_pinyin.py,
  dec_en.py, build_tokens_edited.py) vẫn chưa nằm trong repo nào — chỉ có kết
  quả CSV. Dọn đường dẫn + commit khi nhóm cần chạy lại.
- `CLAUDE.md` đã sửa (mục 5 mới) nhưng chưa commit — Nhật Anh xem lại nội
  dung mục 5 trước khi commit/push.

### 01/10/2026

**Prompt 12:**
> "Giờ tui xong phần crawl dữ liệu rồi, tới bước EDA, tui có nên chuyển qua 1
> folder mới hay ko, có điểm mạnh hay rủi ro gì ko, hãy cho tui ý kiến nhé"

→ Chỉ tư vấn, **chưa hành động gì**. Khuyến nghị: nên chuyển sang làm trong
folder clone riêng của `eda_ecom` (nhánh `nhatanh_updating`), đúng quy ước
nhóm đã có, nhưng cần xử lý trước 6 điểm: (1) `WALKTHROUGH.md` đang rẽ
nhánh — bản trong `ecom_crawler` (có mục 5 CLAUDE.md, chưa commit) mới hơn
bản đã push lên `eda_ecom`; phải chọn 1 nơi làm nguồn thật và đổi điểm trỏ
của quy tắc "bắt buộc ghi WALKTHROUGH"; (2) cần chốt ngày/phiên bản snapshot
data dùng cho EDA để con số không lệch nếu Quang/Huy crawl thêm; (3) chỉ cần
`HF_TOKEN`, không cần tài khoản 1688 nữa; (4) `eda_ecom` có `requirements.txt`
riêng, nhẹ hơn; (5) script audit (title_audit.py...) vẫn chưa commit đâu —
tiện thể dọn vào `scripts/` của folder mới; (6) notebook của mình (28 cell)
đang lệch bản `main` của Quang (25 cell), cần đối chiếu trước khi coi 1 bản
là chính.

**Prompt 13:**
> "Sau đó, tui có nên đổi phiên chat mới không, nếu có thì tui nhờ bạn viết
> cho tui đoạn prompt đủ dài để phiên chat mới vẫn hiểu project tui đang làm,
> hãy cho tui ý kiến nhé"

→ Tư vấn: **nên đổi**, nhưng phải xử lý trước khi đổi — (1) scratchpad của
phiên này (script audit, `bi.parquet`, `attr_pairs.parquet`...) gắn với phiên,
phiên mới sẽ có scratchpad khác → phải copy script ra folder repo trước; (2)
bộ nhớ (memory) gắn với **folder mở phiên**, không phải folder làm việc: thư
mục `~/.claude/projects/` chỉ có 3 mục (`...AI-4-IOT-D4`, `...NLP-PROJECT`,
`...Module-5---PROJECT`), không có mục riêng cho `ecom_crawler` — nghĩa là
mọi quy tắc của dự án này đang nằm lẫn trong memory của `D4`; mở phiên ở
folder mới thì memory trống → quy tắc phải nằm trong `CLAUDE.md` của folder
mới; (3) `WALKTHROUGH.md` + `CLAUDE.md` phải commit trước. Đã viết đoạn prompt
bàn giao cho phiên mới (Nhật Anh dán vào phiên mới).

Phát hiện khi kiểm tra: tiến trình chạy nền mục 5.5 **đã xong từ 28/09 19:52**
nhưng em chưa báo (lần báo cuối lúc 19:25 vẫn ghi "đang chạy") — em báo muộn
~2 ngày; số liệu đã ghi vào mục 4.2. Ngoài ra lần chạy shell đầu ngày 01/10
`git` báo không nhận (PATH hiển thị `D:\Tr?nh\Git\cmd`), chưa rõ nguyên nhân,
chưa kiểm tra lại `git status` của `ecom_crawler`.

**Prompt 14:**
> "Tui có tạo một folder mới tên e-commerce - EDA, bạn hãy đọc kĩ project này
> và đề xuất tất cả file nào nên chuyển qua folder mới đó (chắc chắn phải có
> WALKTHROUGH.md và CLAUDE.md), vì folder mới đó sẽ qua Bước lớn tiếp theo và
> quan trọng đó là EDA và Xử lý dữ liệu, nên bạn hãy đảm các file chuyển qua
> đó không bị gián đoạn, gây mất ngữ cảnh, phù hợp với Bước lớn này, đề xuất
> kĩ nhé"

→ Chỉ khảo sát và đề xuất, **chưa chuyển/sửa file nào**. Folder
`D:\nam4_ky1\e-commerce - EDA` đang rỗng. Điều khảo sát được:
- Trong `ecom_crawler`, `CLAUDE.md`, `WALKTHROUGH.md`, notebook đều **untracked**;
  `.gitignore` còn loại `/data/` và `*.parquet` → toàn bộ `data/eda/` (27 MB,
  gồm `full_run/` chỉ có 1 bản) không được git bảo vệ; `CLAUDE.md` không nằm
  ở đâu khác. → quy trình chuyển phải là copy, kiểm hash, rồi mới để stub.
- Snapshot `bilingual_zh_vi.parquet`: 16.348 dòng, `crawl_time` 17/09 19:28 →
  21/09 21:24 (UTC), worker_nhat_anh 10.396 + worker_huy 5.952, SHA256
  `B4E5CE75…35A29` trùng với bản trong cache HF. `mono_zh` 6.334 dòng,
  `tiki_vi` 13.200 dòng. HF lần sửa cuối 01/10 10:12 UTC nhưng kích thước 3
  file snapshot không đổi so với bản local (chỉ so hash riêng bilingual).
- **Đính chính Prompt 13:** em viết "chỉ cần HF_TOKEN để tải" là **sai** — repo
  HF `ntquang0410/zh-vie_ecom` là **công khai** (`private: False`), tải snapshot
  không cần token; notebook cũng đọc thẳng URL HF. Folder mới không cần `.env`.
- Scratchpad có 2 file thiết kế từ 26–28/09: `pipeline_artifact.html` ("Làm
  Sạch Dữ Liệu 1688") và `eda_pipeline.html` ("EDA Corpus 1688-Tiki", pipeline
  6 bước + tab đánh giá BGE-M3) — chứa thiết kế bước Xử lý dữ liệu, chưa có
  trong WALKTHROUGH. README `ecom_crawler` mục 10 "Việc còn lại" cũng là TODO
  của bước này (`data/processed/`: lọc đuôi số lạ, MinHash dedup, tách dev/test
  theo `product_id`; Tiki: đoạn VAT lặp cuối `description_prose_vi`).
- Môi trường: `.venv` cũ = Python 3.13.7, pandas 3.0.5, numpy 2.5.3, pyarrow
  25.0.1 (không có jupyter/matplotlib/torch); `pypinyin` chỉ cài trong thư mục
  tạm. 5 script phụ thuộc `config.py` của crawler chỉ để tải parquet; nhiều
  script hard-code `D:\...\ecom_crawler` và đường dẫn scratchpad.
- Gotcha mới: sáng 01/10 `git` không nhận trong shell, dùng được bằng đường
  dẫn đầy đủ `D:\Trịnh\Git\cmd\git.exe`.

"Nợ ngữ cảnh" có nguy cơ rơi mất khi chuyển (cần đưa vào mục đầu WALKTHROUGH
mới): (1) 2 file thiết kế pipeline ở trên; (2) Tiki: đoạn VAT lặp ở 13.199/13.200
mô tả — chưa có trong danh sách D00–D17; (3) 95 dòng trùng cả `(title_zh,
description_zh)` — chưa kiểm lại; (4) 1.629 dòng `title_vi` dài >5 lần `title_zh`
— chưa kiểm lại; (5) số dòng lỗi đánh số lệch giữa thiết kế cũ (8.494) và D07
(9.286), khác regex, chưa đối chiếu; (6) phát hiện của Huy trên nhánh `giahuy`
(dấu chấm cuối câu 5.064 dòng, ký tự toàn chiều, số thập phân 4,3 vs 4.3, chữ số
Hán) chưa gộp; (7) nguồn Kaggle Tiki (~5.360 dòng fashion) chưa tải, cần tài
khoản Kaggle.

Đề xuất chi tiết (cây thư mục, bảng chuyển/không chuyển, các chỗ phải chỉnh,
quy trình + tiêu chí kiểm chứng) nằm trong câu trả lời của phiên này; chờ
Nhật Anh duyệt 4 quyết định: folder là clone git `eda_ecom`/`nhatanh_updating`
hay folder thường; parquet có vào git không; cho phép sửa đường dẫn script;
copy memory sang namespace mới hay chỉ dựa `CLAUDE.md`.

**Prompt 15:**
> "1. folder đang rỗng á, nhưng clone để làm gì nữa, trước tui có add thêm file
> notebook eda-bilingual_zh_vi.ipynb của Quang thôi, ròi tui lại chỉnh sửa trên
> file đó
> 2. Không
> 3. Có
> 4. đường dẫn folder mới D:\nam4_ky1\e-commerce - EDA, các quy tắc đã cập nhật
> trong CLAUDE.md nha, nếu thiếu quy tắc thống nhất nào tui đã bàn với bạn thì
> hãy cập nhật vào file .md trước khi chuyển"

→ Quyết định của Nhật Anh trả lời 4 câu ở Prompt 14: (1) **không clone**
`eda_ecom`, dùng folder thường (notebook gốc là file của Quang, anh thêm vào rồi
tự chỉnh); (2) parquet **không** vào git; (3) **cho phép** sửa đường dẫn script;
(4) dựa vào `CLAUDE.md` (không chép memory), bổ sung quy tắc còn thiếu **trước
khi chuyển**.

Đã làm, theo thứ tự:
1. `CLAUDE.md` mới ở folder này: mục 1–4 giữ nguyên byte so với bản của anh
   (đã kiểm), mục 5 viết lại cho bước EDA + Xử lý dữ liệu, giữ mọi quy tắc cũ
   còn đúng và thêm các quy tắc đã thống nhất mà file chưa có: xưng hô, trung
   thực + báo job nền xong ngay, đề xuất thì dừng, việc anh hoãn thì không làm,
   văn bản nộp thầy; bằng chứng + CSV, quy trình duyệt tay (`_editted`, cột
   trống, `XEM_TAY`), similarity chỉ là bộ dò (ngưỡng theo negative, không xoá
   tự động), CSV `utf-8-sig`; chính sách làm sạch 28/09 (tiếng Anh, pinyin → Hán
   Việt, từ mượn); snapshot bất biến, đường dẫn tương đối, kiểm lại sau khi
   sửa script, sửa notebook an toàn; các bẫy máy (127.0.0.1, Ollama, WMI,
   PowerShell, regex `\u`, git PATH). Phần **chưa chốt** (thứ tự bước, chia
   train/test, 回力, dropship) đặt riêng ở 5.5, ghi rõ "chưa thống nhất" để file
   không khẳng định điều anh chưa đồng ý. Bản `CLAUDE.md` ở `ecom_crawler` giữ nguyên.
2. Copy 41 file, kiểm SHA256 từng file: 41/41 khớp lúc copy.
3. Sửa đường dẫn 7 script (bỏ `import config`/tải HF, bỏ đường dẫn cứng
   `D:\...\ecom_crawler` và thư mục tạm; `bi.parquet` → đọc thẳng snapshot;
   `attr_pairs.parquet`, `review_*.tsv` → `data/interim/`); grep lại: không còn
   đường dẫn cứng; `py_compile` 9 script: sạch.
4. Dựng `.venv` từ `requirements.txt`: `pip check` sạch.
5. Chạy lại `attr_split_check` → `title_audit` → `build_tokens_edited` →
   `data_issues`: 8 file kết quả **giống hệt từng byte** bản gốc.
6. Viết `README.md`, `requirements.txt`, `.gitignore`, `data/snapshot/SNAPSHOT.md`,
   `docs/PROVENANCE.md`, `docs/copy_manifest.csv`.
7. Thêm mục 0 vào file này; thêm dòng trỏ sang đây ở đầu `WALKTHROUGH.md` cũ.

Chưa làm: chưa chạy lại `sim_full.py` (khoảng 20 phút) và chạy end-to-end
`run_nb_harness.py`; **không xoá gì** ở `ecom_crawler` (anh tự xoá khi thấy ổn).

**Vướng: Ollama không thấy model.** Smoke test notebook 40 dòng chạy qua phần
đọc parquet và các cell 0–14 (chứng tỏ đường dẫn mới đúng) rồi dừng ở lần gọi
embedding đầu tiên với HTTP 404. `ollama list` rỗng, trong khi `D:\ollama-models`
có `bge-m3:latest` (manifest 15/09). Server Ollama khởi động lúc 13:49 hôm nay;
`OLLAMA_MODELS` mức user = `D:\ollama-models`. Nhiều khả năng server khởi động mà
không nạp biến này nên đang đọc thư mục mặc định (rỗng) — **chưa kiểm chứng**. Em
không tắt Ollama của anh; cần anh thoát Ollama từ khay hệ thống rồi mở lại, sau
đó em chạy lại smoke test.

Sai/sửa trong lượt này: lần chạy đầu script sửa đường dẫn báo lỗi vì
`attr_split_check.py` có 1 dòng CRLF lẫn trong file LF nên em đoán sai kiểu xuống
dòng; script có `assert` nên không file nào bị sửa dở (4 file đã xong nguyên vẹn,
3 file còn lại chưa động tới); em sửa script rồi chạy nốt 3 file.

**Prompt 16:**
> "Đoạn prompt này có thực sự đầy đủ để phiên chat mới bắt kịp tiến độ của
> phiên này, nắm ngữ cảnh và hiểu hết từ đầu project"

→ Trả lời: **không đầy đủ**. Em kiểm bằng cách tìm trong `CLAUDE.md`,
`WALKTHROUGH.md`, `README.md`, `PROVENANCE.md`, `SNAPSHOT.md` ở folder mới các từ
khoá của đề tài (Sailor2, Seed-X, GVHD, 16 tuần, WBS, mining, back-translation
như một bước của kế hoạch): **không có**. Những thông tin đó chỉ nằm trong memory
của folder cũ mà phiên mới không thấy. Lúc trả lời Prompt 15 em khẳng định
"ngữ cảnh nằm trong 2 file đó, prompt chỉ cần ngắn" mà chưa kiểm — **sai**.
Đính chính thêm: em cũng viết "`CLAUDE.md` và `WALKTHROUGH.md` tự nạp" — chỉ
`CLAUDE.md` tự nạp (từ folder mở phiên); `WALKTHROUGH.md` phải được đọc chủ động,
nên prompt đầu và dòng mở đầu mục 5 của `CLAUDE.md` đều bảo phiên mới đọc nó.

Khôi phục: `CLAUDE.md` gốc của nhóm (anh đính kèm 15/09, chưa từng nằm trong
thư mục project) còn nguyên trong transcript của phiên (dòng 158, dạng tài liệu
đính kèm). Đã lưu nguyên văn vào `docs/PROJECT_CHARTER_team_CLAUDE.md` (SHA256 văn
bản gốc ghi ở `docs/PROVENANCE.md`), chỉ thêm khung chú thích đầu file: §1, §4
(nguyên tắc thư mục dữ liệu), §6, §7 còn hiệu lực; §2, §3, §5, §8 và phần
schema/định dạng lưu của §4 đã lỗi thời.

Đã sửa: mục 0 của file này thêm "Bức tranh tổng thể dự án" và nợ ngữ cảnh số 8
(thông tin cá nhân); `CLAUDE.md` thêm mục 5.7 (quy tắc cứng của charter) và 2 mục
chờ quyết định ở 5.5; `PROVENANCE.md` thêm dòng charter và chỉ đường tới
transcript (không copy vì có thể chứa nội dung nhạy cảm).

Phát hiện khi đối chiếu charter với việc đã làm (chưa quyết, đã ghi vào mục chờ
quyết định ở mục 0): (a) đề xuất D16 bỏ 【】 ~ | * mâu thuẫn charter "không xoá ký
tự trang trí" (data hiện không có emoji: 0/16.348 dòng); (b) charter coi
`freeship` là cách dịch đúng của 包邮, trong khi `title_vi` không có chữ `freeship`
nào (0 dòng, trong 57 dòng zh có 包邮) và em đang đề xuất "hot" → "bán chạy";
(c) charter ghi chia theo `product_id` hoặc shop, README crawler ghi `product_id`,
D10 cho thấy chưa đủ; (d) charter có mục thông tin cá nhân chưa ai kiểm: `shop` có
ở mọi dòng, 92 dòng `description_zh` có chuỗi 11 chữ số dạng số di động (chưa xác
nhận là số điện thoại).

**Prompt 17:**
> "Nó phản hồi như vậy:" + nội dung phản hồi của một phiên Claude mới (anh dán
> vào) cho bài tự kiểm 6 câu (a)–(f) của Prompt 16, kèm 6 câu hỏi ngược lại anh.

→ Chấm 6 câu (a)–(f) với file và dữ liệu: **đúng cả 6**, không thấy bịa. Chi
tiết đã đối chiếu: mục tiêu, mô hình, GPU (RTX PRO 6000 chỉ có trong charter, không
có trong memory, nên phiên mới đã đọc file); lý do không dùng làm test; số dòng và
category của snapshot; D07 (36.370 cặp, 9.286 dòng); chính sách 28/09 và con số
bảng quyết định (3.278 / 246 / 210 / 162 / 41); danh sách quyết định đang chờ khớp
mục 0. Phiên mới còn tự nêu đúng 2 chỗ lệch số mà em chưa giải thích.

Hai chỗ lệch số, em kiểm lại bằng dữ liệu:
- `auto` 1.701 vs 971: **đúng như phiên mới đoán**: 971 (bilingual) + 730 (mono_zh)
  = 1.701. Đã ghi vào `data/snapshot/SNAPSHOT.md`, kèm category của `mono_zh` và
  `tiki_vi` (phiên mới báo thiếu: Tiki là food, home, beauty, mother_baby,
  electronics, auto, mỗi loại 2.200, không có fashion/shoes/bags).
- D07 8.494 vs 9.286: giải xong, xem nợ ngữ cảnh số 5 ở mục 0 (đo lại ra 9.354 /
  9.333 / 9.286 tuỳ cách tính; 8.494 không tái hiện được, coi là lỗi thời).

Điểm hở bài tự kiểm lộ ra:
1. Prompt của em nói "chưa sửa file hay chạy gì" nên phiên mới không ghi nhật ký,
   trong khi `CLAUDE.md` bắt ghi mọi prompt. Đã thêm vào `CLAUDE.md` mục 5.1: ghi
   WALKTHROUGH là ngoại lệ duy nhất của lệnh "chưa sửa file".
2. Hiện có **hai phiên cùng ghi** vào bộ WALKTHROUGH (phiên cũ này và phiên mới),
   dễ trùng số "Prompt N" hoặc ghi đè nhau. Đã thêm quy tắc: đọc mục nhật ký cuối
   trước khi ghi, đánh số tiếp, không để hai phiên cùng sửa. Khuyến nghị: từ giờ
   chỉ làm việc ở phiên mới.
3. Phiên mới mở từ folder `D4`, không phải folder `e-commerce - EDA`: nó phải đọc
   `CLAUDE.md` bằng tay, chưa kiểm được việc tự nạp, và còn thấy memory của `D4`.
   Bằng chứng nó vẫn đọc file thật là câu (a)/(d)/(e) chứa chi tiết chỉ có trong
   file. Muốn bài kiểm sạch: mở phiên từ `D:\nam4_ky1\e-commerce - EDA` rồi chạy lại.

Chưa làm (chờ anh): trả lời 6 câu hỏi ngược của phiên mới; ý kiến của em ở câu trả
lời trong chat: đóng các quyết định đang chờ trước, vì nó chặn mọi bước sau và
không cần chạy máy.

**Prompt 18:**
> "Vậy phiên chat mới sẽ lấy data hơn 16k dòng đã crawl từ nguồn nào để xử lý"

→ Trả lời: từ file local `data/snapshot/bilingual_zh_vi.parquet` trong folder này
(25.591.612 byte, SHA256 `B4E5CE75…35A29`, đã kiểm lại khớp ngày 01/10; 16.348
dòng). 8 script đọc đường dẫn này (tương đối theo gốc folder); không script nào
còn tải từ HF hay đọc `ecom_crawler`. Nguồn gốc: bản copy của snapshot công khai
`ntquang0410/zh-vie_ecom` (`data/snapshot/bilingual_zh_vi.parquet`); nếu mất file
thì tải lại theo lệnh trong `data/snapshot/SNAPSHOT.md` rồi đối chiếu SHA256.
Không dùng `data/raw/` (không chuyển sang đây, README crawler cấm dùng để phân tích).

**Prompt 19:**
> Anh trả lời 6 câu hỏi ngược của phiên này: (1) ghi nhật ký ngay mỗi lượt, đọc
> mục cuối trước, ghi từ Prompt 18; (2) làm gì trước: đóng các quyết định đang
> chờ (cách chia tập; D16 và từ mượn quảng cáo; thứ tự các bước làm sạch), sau
> đó anh duyệt 41 từ `XEM_TAY`, 回力 và dropship làm cùng lúc; (3) Ollama để
> sau, chưa cần embedding tới khi tính lại similarity sau làm sạch hoặc chạy lại
> số chính thức; (4) D07 và (5) `auto` đã giải xong, không hỏi nữa; (6) CLAUDE.md
> tự nạp được, nên mở phiên mới từ folder EDA.

→ **Đánh số:** anh dặn ghi từ Prompt 18, nhưng khi em ghi thì Prompt 18 đã bị phiên
khác dùng (câu hỏi về nguồn data); em đọc lại mục cuối rồi ghi tiếp thành Prompt
19. Đây đúng là chuyện hai phiên cùng ghi mà Prompt 17 đã cảnh báo; lần ghi đầu của
em bị từ chối vì file đã đổi, không có gì bị ghi đè. Phiên này mở từ folder `D4`,
không phải folder EDA. Lượt trước của phiên này (trả lời 6 câu a–f) chưa được ghi vì
em hiểu "chưa sửa file" là cấm ghi nhật ký; nội dung đó đã nằm trong Prompt 17 do
phiên kia ghi, không ghi bù lần nữa.

Việc đã làm: **chỉ đọc**, chưa sửa data, chưa sửa file nào ngoài mục này. Chạy
2 script đọc-không-ghi trên `data/snapshot/` (`bilingual_zh_vi` và `tiki_vi`) và
đọc `data/eda/title_audit/title_latin_tokens_editted.csv` để lấy bằng chứng cho
các quyết định; script nằm ở scratchpad của phiên (`decide_evidence.py`), **không
nằm trong repo**, kết quả chỉ in ra màn hình, chưa xuất CSV. Không dùng Ollama,
không dùng GPU.

Bằng chứng thu được (đo trên toàn bộ 16.348 dòng; Tiki 13.200 dòng làm mốc tiếng
Việt bản địa):

1. **Cách chia tập.** 16.348 `product_id` khác nhau nhưng chỉ 15.863 `title_zh`
   khác nhau: 391 nhóm `title_zh` có ≥ 2 dòng, gồm 876 dòng (5,4%), nhóm lớn
   nhất 16 dòng; không nhóm nào nằm ở 2 category; 325/391 nhóm trải trên ≥ 2 shop
   (cùng mẫu hàng nhiều shop đăng). Shop: 7.094 shop, shop lớn nhất 74 dòng
   (0,45%), top 10 shop chiếm 2,34%, 4.349 shop chỉ có 1 dòng, 35 shop bán ≥ 2
   category. Nối `title_zh` với shop (thành phần liên thông): 6.781 nhóm, nhóm lớn
   nhất 183 dòng (1,1%), 5 nhóm lớn nhất 183/153/139/112/107. Nghĩa là nhóm theo
   `title_zh` + shop **không** làm vón cục: không có nhóm nào đủ lớn để chiếm một
   tập test.
2. **D16 (ký hiệu).** Tiếng Trung có 【 298 dòng, 】 306 dòng; tiếng Việt có 【 89,
   】 91, `[` 200, `]` 207, `~` 53, `|` 18, `*` 24 (bản Trung: `~` 71, `|` 13, `*`
   26). Máy dịch đổi phần lớn 【】 thành `[ ]` (281/298 dòng có 【 ở bản Trung thì
   bản Việt có ngoặc). Tiki (người bán Việt thật): `[ ]` ở 525/13.200 dòng (4,0%),
   `|` 187, `*` 33, `~` 3, 【】 0 dòng. Theo regex của em, 391 dòng Việt có ký hiệu
   (bảng D16 ghi 393, chưa đối chiếu); **384/391 dòng thì bản Trung cũng có ký
   hiệu**, chỉ 7 dòng là ký hiệu do máy dịch tự thêm.
3. **Từ mượn quảng cáo.** `freeship`: 0 dòng trong 1688 (máy dịch viết "Miễn phí
   vận chuyển" 46 dòng, bản Trung có 包邮 57 dòng), Tiki có 6 dòng `freeship` và 1
   dòng "miễn phí vận chuyển". `hot`: 323 dòng (1,98%) so với Tiki 37 dòng (0,28%);
   trong 323 dòng này có 199 dòng bản Trung chứa 爆款, 2 dòng 热卖, 0 dòng 热销;
   "bán chạy" đã có 191 dòng. `sale`: 0 dòng (Tiki 6); `new`: 23 dòng (Tiki 51). Mẫu
   Tiki dùng `hot` kiểu "HÀNG HOT", "Hot Deal", "Siêu Hot"; mẫu 1688 là kiểu máy
   dịch "Douyin Hot Model", "xuyên biên giới hot model".
4. **回力.** Bản Trung có 回力 ở 91 dòng: shoes 43 (hãng giày, mẫu "Giày Pull-back",
   "Giày Nam Warrior") và mother_baby 48 (**đồ chơi lên cót / kéo lùi**, không phải
   hãng). Quy tắc hiện tại trong bảng 41 từ đổi `pull-back` → "Warrior" không phân
   biệt category, sẽ sai nếu dính dòng đồ chơi. Em chưa kiểm 12 dòng `pull-back`
   thuộc category nào.
5. **dropship.** 91 dòng `title_vi` chứa `dropship` (74 "dropshipping", 17 "dropship");
   bản Trung có 代发 ở 466 dòng, tức máy dịch bỏ rơi khoảng 375 dòng. Máy dịch cũng
   tự dịch 代发 thành "giao hàng hộ" ở 20 dòng ("bán sỉ, giao hàng hộ"). Tiki: cả
   "dropship" lẫn "giao hộ" lẫn "giao hàng hộ" đều 0 dòng, nên **không có mốc bản
   địa để chọn**; chưa tra nguồn ngoài.

Khuyến nghị của em (đang chờ anh chốt, chưa áp dụng gì), chi tiết nằm trong câu trả
lời của chat: chia theo thành phần liên thông (`title_zh`, shop), sau này gộp thêm
cụm MinHash khi anh duyệt `datasketch`; D16 giữ ký hiệu theo charter, chỉ xử lý 7
dòng máy dịch tự thêm; `hot` đổi sang "bán chạy" khi bản Trung có 爆款/热款/热卖/热销,
không chủ động thêm `freeship`; 回力 chỉ đổi khi category shoes; dropship giữ hay đổi
là quyết định của anh vì không có mốc; thứ tự bước mới đề xuất ở câu trả lời (đưa
việc gán nhóm lên sớm, định dạng trước khử trùng lặp, gỡ thông tin cá nhân sau khi
chia). Cần anh chốt 4 nhóm câu hỏi trước khi em viết bảng duyệt `_editted`.

**Prompt 20** (anh chuyển model sang Opus trước prompt này):
> "Bạn bỏ qua phần Title, giờ tới phần EDA Description, bạn hãy đọc kĩ toàn bộ
> decription và đề xuất cho tui file notebook gồm các block để tìm hiểu, trực quan
> bộ data đó (phần decription), hoặc đề xuất cho tui các phương pháp để tui hiểu kĩ
> về chúng, hiểu cặn kẽ các vấn đề của decription đã crawl"
>
> Nhắn thêm giữa lượt: "bạn làm việc với toàn bộ decription của hơn 16 nghìn dòng nha"

→ Các quyết định về title ở Prompt 19 **tạm gác** theo lời anh (chưa chốt, vẫn còn
trong danh sách chờ). Lượt này **chỉ đọc và đề xuất**: chưa tạo notebook, chưa sửa
data. Đọc `docs/ecom_crawler_README.md` (schema description), `attr_split_check.py`,
danh sách cell notebook (description mới chỉ có ở 5.4 và 5.5). Chạy 4 script đọc-không-ghi
trên **toàn bộ** 16.348 dòng `bilingual_zh_vi` (cộng `mono_zh` 6.334 và `tiki_vi`
13.200); script và file tạm `pairs_fixed.parquet` nằm ở scratchpad của phiên, không
nằm trong repo. Không dùng Ollama.

Phát hiện chính (số đo toàn bộ, chưa đọc tay mẫu nào để kiểm độ chính xác):
1. **Cấu trúc.** Không dòng nào rỗng `description_zh/vi`. Mỗi dòng có 4–53 thuộc tính
   (trung vị 22), zh và vi luôn bằng số lượng. Độ dài trung vị: zh 421, vi 1.132 ký tự.
   `description_zh` dựng lại đúng từ `attributes_zh` ở 16.347/16.348 dòng.
2. **`attributes_vi` KHÔNG cùng thứ tự với `description_vi`/`attributes_zh`**: chỉ 21/16.348
   dòng trùng thứ tự. Struct chỉ có `name` và `values`, **không có `fid`**. Ghép
   `attributes_zh[i]` với `attributes_vi[i]` theo chỉ số là ghép sai. Khôi phục được thứ tự
   bằng cách dò từng đoạn `tên: giá trị` của `attributes_vi` trong `description_vi`:
   16.342/16.348 dòng, ra 406.871 cặp, không cặp nào lệch số giá trị.
3. **Đính chính D08 (mục 5 và 6.2).** Cả 72 dòng D08 đều do giá trị tiếng Việt chứa
   `"; "` (dấu phân cách), nên tách chuỗi bằng `"; "` bị lệch. Không phải lỗi ghép của
   crawler. Cả 72 dòng khôi phục được bằng cách ở điểm 2. Giải pháp "ghép theo `fid`"
   ở mục 6.2 **không làm được** vì snapshot không có `fid`.
4. **Tên thuộc tính.** Có 2.053 tên zh, nhưng 3.885 tên vi. 866 tên zh bị dịch nhiều
   kiểu, trong đó có 99/100 tên phổ biến nhất (vd 货号 → "Số mặt hàng" 7.165 / "Mã sản
   phẩm" 3.214 / "Mã hàng" 3.093 / để nguyên "货号" 130). 100 tên đầu phủ 73,2% số lần
   xuất hiện, 300 tên đầu phủ 96,5%, 1.006 tên chỉ xuất hiện 1 lần. Tên vi của 1688 chỉ
   trùng 92/3.616 tên của Tiki.
5. **Thuộc tính giao dịch của 1688** (是否跨境出口专供货源 9.001 cặp, 有可授权的自有品牌
   7.261, 主要下游平台 5.614, 主要销售地区 5.320, 是否进口 5.023...). Đây là thông tin nền
   tảng, không phải mô tả hàng, giá trị lặp gần như y hệt. 14 tên em liệt kê thử (tính cả
   货号, 品牌) chiếm 20,4% số cặp. 11,4% số cặp có giá trị zh = vi (chủ yếu mã hàng 货号).
6. **Lỗi đánh số: tìm ra cơ chế.** Máy dịch đánh số những giá trị **giống nhau trong
   cùng một sản phẩm**. Nếu '否' xuất hiện 1 lần trong sản phẩm, có 6/4.293 cặp bị đánh số.
   Nếu xuất hiện từ 2 lần, có 18.842/26.166 cặp bị đánh số. 11.490/13.121 nhóm (sản phẩm,
   giá trị) được đánh số đúng 1..k. **Regex D07 bắt thiếu**: chỉ bắt
   Không/Có/Là/Khác viết hoa đầu, trong khi có cả "KHÔNG4", "không2", "Đúng vậy2", "bông1",
   "Silicon cấp thực phẩm1". Regex D07 trên cặp đã khôi phục: 36.524 cặp, 9.332 dòng.
   Luật rộng hơn (vi tận cùng bằng số, zh không có chữ số, giá trị zh lặp trong sản
   phẩm): 66.881 cặp, 13.675 dòng (83,7%). Luật rộng **chưa đọc tay** để đo báo nhầm.
7. **Thông tin cá nhân (cập nhật nợ ngữ cảnh số 8).** 93 cặp, 91 dòng có giá trị dạng số
   di động Trung Quốc. Đã thấy số điện thoại thật (产品厂家联系方式 19 cặp, 定做电话,
   联系方式), nhưng 24 cặp nằm ở 净含量 (khả năng là mã, chưa kiểm). `description_extra_zh`:
   URL 86 dòng, địa chỉ 61, tên công ty 605, số di động 8.
8. **`description_extra_zh`** có chữ ở 3.749 dòng, chỉ có tiếng Trung, lẫn lộn thông số
   sản phẩm, giới thiệu công ty, chính sách hậu mãi.
9. **Trùng.** 193 dòng trùng `description_zh` với dòng khác (87 nhóm). 70 nhóm có cùng zh
   mà khác vi.
10. **Tiki.** `description_prose_vi` trung vị 1.563 ký tự. 13.199/13.200 dòng có VAT/thuế,
    12.439 dòng có cùng 200 ký tự cuối. Tiki có 843 tên thuộc tính, người bán dùng "Xuất
    xứ (Made in)", "Họa tiết" (821) thay vì "Hoa văn" (6). Có 1 dòng chữ bị tách rời ("n à
    o m à ch ủ y ế u"), 78 dòng có hotline/số điện thoại. `mono_zh` có cùng cấu trúc
    description, lệch category (bags 2.048, mother_baby 1).

Đề xuất bộ khung notebook mới `notebooks/eda-description.ipynb` (không sửa notebook của
Quang) nằm ở câu trả lời trong chat. Chờ anh duyệt trước khi tạo. Notebook cần
`matplotlib` (và `ipykernel` nếu chạy ở VS Code), mà `requirements.txt` chưa có: phải hỏi
anh trước khi thêm. Mục 5 và 6 chưa sửa; đính chính D07/D08 ở trên là ghi chú sửa sai theo quy tắc.

**Prompt 21:**
> "1. Làm riêng nha, làm như notebook bạn đề xuất 2. Tui sẽ chạy trên colab
> 3. Phạm vi: chỉ description song ngữ. Nhưng trước khi tạo file notebook, bạn hãy
> kiểm tra từng cell bạn đề xuất đã đầy đủ, kĩ, hiệu quả để tìm hiểu, trực quan toàn
> bộ 16 nghìn dòng description song ngữ chưa, chưa hiểu thì hỏi tui tới khi hiểu kĩ
> nhé, vì bước này rất quan trọng"

→ Quyết định của anh: notebook riêng `notebooks/eda-description.ipynb`; chạy trên
**Colab** (nên không cần thêm `matplotlib`/`ipykernel` vào `requirements.txt` local);
phạm vi **chỉ description song ngữ** (bỏ block `mono_zh` và Tiki). **Chưa tạo notebook**:
anh dặn rà từng block và hỏi cho rõ trước.

Rà lại đề xuất Prompt 20, có chạy thêm 1 script đọc-không-ghi trên toàn bộ 16.348 dòng
(scratchpad, `desc_profile5.py`) để kiểm 3 chỗ nghi là hổng. Phát hiện mới:
1. **Thứ tự giá trị bên trong một thuộc tính cũng có thể lệch.** Trong 6.030 thuộc tính
   nhiều giá trị mà mỗi giá trị có số riêng (vd "1号…" ↔ "Số 1…"), 776 (12,9%) có dãy số
   zh và vi khác nhau. Chưa rõ là do đảo thứ tự, mất số hay đổi số; chưa đọc tay.
   Tổng cộng có 691.128 cặp ở cấp giá trị.
2. **Dữ liệu lặp rất nhiều:** 406.871 cặp thuộc tính nhưng chỉ có 72.392 cặp (giá trị zh,
   giá trị vi) khác nhau. 20 cặp lặp nhiều nhất chiếm 11,1% (vd 否 → "Không.1" 4.438 lần,
   纯色 → "Màu trơn" 3.517 lần).
3. **Tên hãng (品牌):** 13.952/15.834 cặp có tên hãng bằng chữ Hán, và máy dịch xử lý
   không thống nhất. Có lúc dịch nghĩa (悍狮 → "Sư tử khổng lồ", 绵之情 → "Cảm giác của
   bông"), có lúc ra pinyin ("Bocheng Yipin"), Hán Việt (方靖 → "Phương Tĩnh") hay để
   nguyên chữ Hán. Có giá trị 品牌 thực ra là nguyên một tiêu đề sản phẩm.
4. Trong 866 tên thuộc tính có nhiều cách dịch, chỉ 32 tên là khác nhau do chữ hoa hay
   dấu "?" "."; 834 tên khác nhau thật.
5. Số thuộc tính trung vị tăng theo ngày crawl (17/09: 15 → 21/09: 35). Nhiều khả năng do
   thứ tự crawl category chứ không phải parser đổi, chưa kiểm.

Tự nhận: con số "20,4%" ở Prompt 20 (thuộc tính giao dịch) gộp cả 货号 và 品牌, mà hai tên
này là thông tin sản phẩm, không phải thông tin giao dịch, nên con số này bị thổi lên.
Block phân loại vai trò phải làm lại bằng một bảng phân loại có người duyệt.

Đã gửi anh bản rà từng block (thêm, sửa, bỏ) và các câu hỏi cần chốt trước khi tạo
notebook (xem câu trả lời trong chat).

**Prompt 22:**
> "là sao vậy, làm rõ 8 câu hỏi trên cho tui hiểu"

→ Giải thích lại 8 câu hỏi của Prompt 21 bằng lời dễ hiểu, kèm ví dụ thật từ dữ liệu
(chỉ đọc). Ví dụ câu 4 ở Prompt 21 ("camera hành trình" ở title, "máy ghi âm lái xe" ở
description) là em tự đặt, **chưa kiểm**. Nay đã kiểm: trong 519 dòng có 行车记录仪 ở
`title_zh`, 52 dòng có title dịch "camera hành trình" mà description lại dịch "máy ghi
âm/ghi hình lái xe" (vd `811048395243`: 产品名: 行车记录仪后视镜 → "Tên sản phẩm: Máy ghi
âm lái xe gương chiếu hậu"). Vẫn chờ anh trả lời 8 câu, chưa tạo notebook.

**Prompt 23:**
> "khi fine-tune Sailor2, bạn loại ra khỏi bộ nhớ pipeline nha, vì tui sẽ bàn lại với
> nhóm sau về model finetune, bạn có đề xuất gì khác cho câu 5 ko / tui sẽ chạy file
> notebook trong vs code lun (colab trong vs code), có thể xuất output sau mỗi cell để
> tui đọc và hiểu / bạn thắc mắc gì nữa ko"

→ **Gỡ mô hình fine-tune khỏi pipeline** (ghi chú, không xoá lịch sử): `CLAUDE.md` mục
5.7 dòng "Goal" (nay là: không thiết kế phân tích hay xử lý theo mô hình hay tokenizer
cụ thể, chờ nhóm bàn lại); mục 0 của file này (dòng "Mô hình"); khung chú thích đầu
`docs/PROJECT_CHARTER_team_CLAUDE.md` (thêm 1 dòng, thân văn bản gốc giữ nguyên); memory
của phiên (`project-ecom-crawler-vi-zh.md`). Nhật ký cũ có nhắc Sailor2 (Prompt 16) giữ
nguyên vì là lịch sử. Lần sửa `CLAUDE.md` đầu tiên bị từ chối vì file vừa đổi; em đọc
lại thì thấy dòng cần sửa vẫn y như cũ, rồi mới sửa.

Câu 5 đổi thành block **"Độ dài và đơn vị dữ liệu" không phụ thuộc mô hình**: đo ký tự
(zh, vi), số âm tiết tiếng Việt, số thuộc tính, số giá trị, ở 3 cấp (cả description,
từng cặp thuộc tính, từng giá trị), vẽ đường tích luỹ để sau này chọn mô hình nào cũng
đọc được ngay có bao nhiêu % vượt giới hạn. Cùng block có tỉ lệ độ dài vi/zh từng cặp
thuộc tính để bắt dịch thiếu/thừa ở cấp thuộc tính. Không tải tokenizer nào.

Anh chạy notebook bằng **Colab trong VS Code**. Hệ quả (chưa kiểm trên máy anh): kernel
chạy trên máy ảo Colab, nên file CSV notebook ghi ra nằm trên máy ảo, không nằm trong
folder EDA; output in ngay dưới cell thì được lưu vào file `.ipynb` ở máy anh. Em đề
xuất mỗi cell in kết luận bằng tiếng Việt kèm con số ngay dưới cell. Còn hỏi anh: 7 câu
còn lại (1–4, 6–8) có theo đề xuất không; cách lấy CSV về máy (câu 8 đổi lại vì Colab
trong VS Code có thể không dùng được nút tải file, em chưa kiểm).

**Prompt 24:**
> "bạn kiểm tra file notebook của eda title á, coi cách xuất ntn"

→ Chỉ đọc `notebooks/eda-bilingual_zh_vi.ipynb` (code và output đã lưu), không chạy,
không sửa. Cách notebook title làm:
- **Đầu vào:** cell 3 đọc parquet thẳng từ URL HF (`resolve/main/...`), cell 5 có
  phương án dự phòng là duckdb. **Không kiểm SHA256**, nên nếu HF đổi file (đã có commit
  mới ngày 01/10) thì số liệu đổi theo mà không báo.
- **Kết quả xem trong notebook:** bảng hiện bằng `display(...)`, kết luận bằng `print(...)`
  (cell 27 có "BẢNG TỔNG HỢP EDA", dòng ✅ và "NHẬN XÉT NHANH"), biểu đồ bằng `plt.show()`.
  Toàn bộ được lưu trong file `.ipynb`. Kiểu output (`vnd.microsoft.datawrangler`) cho
  thấy notebook được chạy trong VS Code.
- **File xuất:** chỉ cell 27 ghi 6 CSV (`utf-8-sig`, bỏ cột lồng nhau), tên file không có
  thư mục, nên file rơi vào thư mục làm việc của kernel. Không có code tải về hay ghi vào
  Drive.

**Phát hiện:** output đã lưu trong notebook là **một lần chạy trên GPU** ("Thiết bị:
cuda", pandas 2.2.3, torch 2.11.0+cu128), đủ 16.348/16.348 title và 405.190 cặp thuộc
tính. File sửa lần cuối 28/09 18:31 và giống hệt từng byte bản trong `copy_manifest.csv`.
Nhiều khả năng anh đã chạy bằng Colab trong VS Code ngày 28/09, sau khi Claude sửa mục 5,
nhưng em **chưa xác nhận với anh**. Số của lần chạy GPU này: title AUC 0,9770, mốc chắc sai
0,506, 19 cặp chắc sai (0,12%), vùng chồng lấn 2.011 (12,30%); thuộc tính AUC 0,8191, mốc
0,487, 34.432 cặp dưới mốc (8,50%). Số chạy bằng Ollama ghi ở mục 4.2: 0,9779 / 0,504 / 17
/ 11,9% / 0,8183 / 34.477 (8,51%). **6 file CSV của lần chạy GPU không có trong folder**:
CSV trong `data/eda/full_run/` là của lần chạy Ollama local (ghi lúc 19:52 ngày 28/09).
Điều này khớp với điều em nghi ở Prompt 23: chạy trên máy ảo Colab thì CSV nằm lại trên
máy ảo. Việc nợ "chạy lại số chính thức trên Colab GPU" có thể đã xong một phần (có số in
ra, mất CSV); chờ anh xác nhận rồi mới sửa danh sách việc nợ.

**Prompt 25:**
> "Notebook description chạy bằng kernel local hay Colab + Drive?, kernel colab là giống
> với title đúng ko / Các câu 1–4, 6–7 ở lượt trước có theo đề xuất của em không? có"

→ **Anh chốt các câu 1–4, 6–7 theo đề xuất của em** (notebook description):
1. `description_extra_zh`: 1 block ngắn (phân loại nội dung + thông tin cá nhân).
2. Tiếng Anh/pinyin trong `description_vi`: đo theo y chính sách title 28/09.
3. Tên hãng 品牌: pinyin là đúng; dịch nghĩa hoặc để nguyên chữ Hán là lỗi; Hán Việt chưa
   chốt (đề xuất cũ để anh quyết). Giới hạn em nói thêm: máy không tách tự động được "Hán
   Việt" với "dịch nghĩa" vì cả hai đều là tiếng Việt có dấu, nên notebook gộp chung một
   nhóm và xuất mẫu để đọc tay.
4. So thuật ngữ title với description: 1 block nhỏ, chỉ đếm.
6. Cho dùng `datasketch` để tìm description gần trùng.
7. Bảng 300 tên thuộc tính và 200 mẫu lỗi đánh số: notebook chỉ xuất file có cột trống,
   anh duyệt sau.

Câu "kernel Colab có giống title không": **đúng**, notebook title chạy bằng Colab trong VS
Code (Prompt 24). Đã giải thích lại hệ quả: CSV ghi ra nằm trên máy ảo, nên mất như lần
chạy title. Vẫn chờ anh chọn **kernel local** hay **Colab + Drive** rồi mới tạo notebook.
Câu "lần chạy GPU 28/09 có phải anh chạy" anh chưa trả lời.

**Prompt 26:**
> "bạn đề xuất tui chạy ở đâu thì hiệu quả nhất"

→ Đề xuất **kernel local** (`.venv` của folder EDA, chọn kernel trong VS Code). Lý do: không
cần GPU; file duyệt tay và CSV bằng chứng ghi thẳng vào `data/eda/description/`, không mất
như 6 CSV của notebook title; đường dẫn tương đối và kiểm SHA256 trên snapshot local chạy
ngay, không phải tải 25 MB mỗi phiên; em chạy thử và kiểm được từng cell trước khi giao.
Cần anh đồng ý thêm `matplotlib`, `ipykernel`, `datasketch` vào `requirements.txt` rồi cài
vào `.venv` (CLAUDE.md 5.4). Chưa cài, chưa tạo notebook.

**Prompt 27:**
> "đồng ý, bạn tạo đi và hướng dẫn tui chạy" / về lần chạy GPU 28/09 của notebook title:
> "ko quan trọng nên đừng hỏi nữa nha"

→ Câu hỏi về lần chạy GPU ngày 28/09: **không hỏi lại**, việc nợ "chạy lại số chính thức
trên Colab GPU" giữ nguyên như cũ.

Đã làm:
1. Cài vào `.venv`: `matplotlib==3.11.2`, `ipykernel==7.4.0`, `datasketch==2.0.0` (kéo theo
   scipy, ipython, jupyter_client...), `pip check` sạch; ghi 3 dòng vào `requirements.txt`.
2. Tạo **`notebooks/eda-description.ipynb`** (33 ô: 17 ô chữ, 16 ô code, 16 block 0–15 như đã
   chốt ở Prompt 21–25, phạm vi chỉ description song ngữ). Notebook dựng bằng script tạm ở
   scratchpad (`build_desc_nb.py`, không nằm trong repo); từ giờ sửa thẳng trên notebook.
   Notebook title `eda-bilingual_zh_vi.ipynb` không bị đụng (vẫn 560.516 byte).
3. Chạy thử toàn bộ bằng kernel `.venv` thật (qua `jupyter_client`), output đã lưu sẵn trong
   notebook: **58 giây, không lỗi, bảng kiểm 38/38 khớp số lúc khảo sát**. Biểu đồ đã mở ra xem:
   chữ Hán và tiếng Việt hiện đúng (font Microsoft YaHei), không chồng chữ.
4. File xuất: 13 CSV trong `data/eda/description/` (5 file `_editted` có cột trống để anh
   duyệt: vai trò 300 tên thuộc tính, 200 mẫu lỗi đánh số (thực tế 193 vì nhóm "chỉ D07"
   chỉ có 33 cặp), 200 mẫu tên hãng, ứng viên thông tin cá nhân, từ tiếng Anh trong
   description), cache `data/interim/desc_attr_pairs.parquet`,
   `desc_value_pairs.parquet`.

Phát hiện mới trong lúc viết notebook (số đo toàn bộ 16.348 dòng):
- **Cơ chế lỗi đánh số đúng hơn giả thuyết lúc khảo sát:** máy dịch đánh số những **bản
  dịch tiếng Việt trùng nhau** trong 1 sản phẩm, không phải giá trị Trung trùng nhau. Có số đuôi:
  87.512/88.185 giá trị có bản dịch trùng (99,2%) so với 1.356/602.943 giá trị không trùng.
  Giả thuyết cũ bỏ sót 7.648 giá trị (vd 补水 và 保湿 cùng ra "Dưỡng ẩm" → "Dưỡng ẩm1/2"; 其他
  và 其它 → "Khác1/2"). **Luật v2** (xét từng giá trị, kể cả trong danh sách, không đụng số
  thật): 87.512 giá trị, 82.092 cặp thuộc tính, **14.338 dòng (87,7%)**; D07 cũ 9.332 dòng.
  Chưa đo độ chính xác, chờ anh duyệt 200 mẫu.
- Số thuộc tính chênh theo máy crawl (vd mother_baby: Huy 33, Nhật Anh 22) **không phải do
  crawler**: trong cùng danh mục con 1688, 2 máy chênh không quá 3 thuộc tính; chênh là do mỗi
  máy crawl danh mục con khác nhau.
- Bỏ số đuôi rồi so lại: số lệch zh/vi giảm từ 14.645 cặp (lần chạy đầu) xuống **2.849 cặp ở
  2.152 dòng**; máy đổi chữ số Hán (二层 → "2 lớp") và 斤 sang kg (36斤 → "18kg") được coi là đúng.
- Thứ tự giá trị trong danh sách: kiểm theo từng vị trí còn 286 thuộc tính lệch (285 dòng).
- Tóm tắt (đề xuất mức độ, chưa duyệt): mức 1 (DS10 số lệch) 2.152 dòng (13,16%); mức 1–2
  15.812 dòng (96,72%, chủ yếu do lỗi đánh số và tên thuộc tính dịch không thống nhất).

Sai/sửa trong lúc làm (đều sửa **trước khi giao**, notebook giao là bản đã sửa):
- Lần chạy đầu, kết luận block 1 ghi "số thuộc tính tăng theo ngày là do thứ tự crawl" mà
  chưa kiểm; thêm phép so trong cùng danh mục con mới ra kết luận ở trên.
- Lần chạy thứ hai, em ghi "776 thuộc tính lệch thứ tự phần lớn do lỗi đánh số" nhưng bỏ số
  đuôi chỉ giảm còn 768: **sai**. Nguyên nhân thật là máy đổi chữ số Hán/đơn vị; đổi sang kiểm
  theo từng vị trí.
- Block 7 ban đầu xếp "other/其他" vào nhóm tiếng Việt có dấu, và đếm "---1" là hãng bị đổi tên;
  đã sửa (nhận dạng "không có tên hãng" rộng hơn, so trên bản đã bỏ số đuôi).
- Block 9 ban đầu tính cả tên hãng và mã hàng, ra 7.159 từ "chưa có trong bảng title"; đã bỏ 2
  nhóm đó và chỉ tính là lỗi những từ bảng title xếp vào nhóm phải thay (4.040 dòng); 5.922 từ
  chưa phân loại để anh duyệt, không tính là lỗi.
- Số thập phân trong kết luận in kiểu "2.7" → đổi sang kiểu Việt "2,7".

Giới hạn còn lại (đã ghi trong notebook): cụm gần trùng (MinHash) là gần trùng bảng thuộc
tính, có thể là sản phẩm khác nhau cùng khuôn; `pinyin_dung` phụ thuộc pypinyin đọc đúng chữ
đa âm; số Hán ghép (二十, 百) chưa quy đổi.

**Việc mới (từ Prompt 27):** anh chạy notebook trong VS Code bằng kernel `.venv`; duyệt 5 file
`_editted` trong `data/eda/description/`; sau khi duyệt 200 mẫu lỗi đánh số mới chốt luật v2.

**Prompt 28:**
> "TỪ NÃY GIỜ TÌM HIỂU DATA, HÃY NÊU CÁC VẤN ĐỀ CỦA mục decription, mỗi vấn đề phải đi kèm
> bằng chứng để tăng độ xác thực (ví dụ vấn đề này ở những id nào, số lượng id dính lỗi, xếp
> theo thứ tự các vấn đề lỗi nặng tới nhẹ nhất,...), trình bày dạng table, có thể xuất file
> csv để dễ quan sát"

→ Trước khi lập bảng, em đọc mẫu bằng chứng của từng vấn đề và thấy còn báo nhầm có quy
luật, nên sửa notebook rồi chạy lại (vẫn 38/38 khớp, không lỗi):
- DS03 (thứ tự/số trong danh sách lệch) nâng lên **mức 1** (ghép theo vị trí ra cặp sai); bỏ qua
  giá trị 斤 → kg (90-105斤 → 45-52.5kg là quy đổi đúng). Còn 73 dòng (trước 285).
- DS10 (số lệch): thêm quy đổi đúng 万 (1000万 → "10 million") và 纯棉 → "100% cotton"; so trên
  bản đã bỏ cả số đuôi nghi đánh số. Còn **727 dòng** (lần chạy Prompt 27: 2.152).
- DS04 (nghi dịch thiếu/thừa): tỉ lệ đổi sang (chữ cái vi − chữ Latin có sẵn ở zh) / số chữ Hán,
  để số và chữ Latin không làm lệch. Còn 1.111 dòng (trước 1.815).
- Luật đánh số v2 không coi số đuôi là lỗi khi nó trùng chữ số Hán trong bản Trung (二级 →
  "Cấp độ 2" là dịch đúng). Luật v2 còn 14.329 dòng (Prompt 27: 14.338).
- Thêm **DS06B** (mức 3): 1.185 giá trị / 895 dòng có số đuôi kiểu đánh số nhưng không thấy bản
  dịch trùng trong description ("Bộ1", "áo thun1", "Là1"), tách riêng để không đếm sai.
- Thêm **DS14** (mức 4): số đo thập phân có đơn vị, bản Việt lúc viết 1,5m lúc 1.5m (1.036 cặp
  giữ dấu chấm, 1.027 cặp dấu phẩy), 1.877 dòng. Lần đầu em tính cả mã tiêu chuẩn
  (GB/T30357.2) và số phiên bản (Bluetooth 5.3) là sai, đã chỉ giữ số đo có đơn vị.
- `desc_issues_summary.csv` thêm cột `ten_muc_do`, `vi_du_bang_chung` (3 ví dụ ngẫu nhiên), xếp
  nặng → nhẹ, mục cấu trúc DS01 (mức 0) để cuối. `desc_issues_evidence.csv`: 63.092 dòng (vấn
  đề, product_id, bằng chứng), thêm cột `muc_do`.

Bảng cuối (16 mục, đã gửi trong chat). Số dòng / % trên 16.348:
- Mức 1: DS10 số lệch 727 (4,45%); DS03 danh sách lệch thứ tự/số 73 (0,45%).
- Mức 2: DS06 lỗi đánh số 14.329 (87,65%); DS05 tên thuộc tính dịch không thống nhất 14.005
  (85,67%); DS07 còn chữ Hán 2.097 (12,83%); DS11 trùng/gần trùng 1.319 (8,07%); DS12 ứng viên
  thông tin cá nhân 297 (1,82%); DS13 thuật ngữ title ≠ description 180 (1,10%); DS09A tên hãng
  còn chữ Hán 49 (0,30%); DS02 không ghép được 6 (0,04%).
- Mức 3: DS09B tên hãng ra tiếng Việt có dấu 5.766 (35,27%); DS08 tiếng Anh/pinyin phải thay
  4.040 (24,71%); DS04 nghi dịch thiếu/thừa 1.111 (6,80%); DS06B nghi đánh số lẻ 895 (5,47%).
- Mức 4: DS14 dấu thập phân 1.877 (11,48%). Mức 0 (cấu trúc): DS01 16.321 (99,83%).
- Dòng dính ≥ 1 vấn đề mức 1: 744 (4,55%); mức 1–2: 15.799 (96,64%); mức 1–3: 15.982 (97,76%).

Ghi rõ với anh: DS10, DS04, DS06B, DS09B là danh sách **để đọc**, còn lẫn báo nhầm (vd DS10 "9月 →
tháng chín" là đúng; DS04 "塑料/塑胶 → Nhựa" là đúng), chưa dùng để xoá. Mức độ là đề xuất,
chưa duyệt. Chưa đưa bảng này vào mục 5 của file này (anh chưa yêu cầu).

### 02/10/2026

**Prompt 29:**
> "15 vấn đề đó đã thực sự đã hết chưa, bạn hãy đọc kĩ data và tìm có thể còn những lỗi nghiêm
> trọng hay ít nghiêm trọng nữa á"

→ **Chưa hết.** 15 vấn đề cũ chủ yếu dò bằng luật máy; chưa ai đọc nghĩa của bản dịch. Lượt
này em **đọc tay** 220 tên thuộc tính phổ biến nhất và 700 cặp giá trị phổ biến nhất (file tạm
`names_top220.txt`, `values_top700.txt` ở scratchpad), rồi chạy thêm các bộ dò cho những gì
chưa kiểm. Thêm **block 15 "Đọc thêm"** vào notebook (block tổng hợp thành 16), chạy lại toàn
bộ: không lỗi, **bảng kiểm 49/49 khớp** (11 chỉ số mới lấy từ script khảo sát cùng định nghĩa).

11 vấn đề mới (số đo toàn bộ 16.348 dòng):
- **DS15 (mức 1) Tên thuộc tính dịch sai nghĩa** (33 tên đọc tay, nên là mức tối thiểu):
  21.917 cặp / 10.878 dòng (66,5%). Vd 毛重 → "Trọng lượng tịnh" (ngược nghĩa, đúng là cả bì) 877,
  "Trọng lượng lông" 756; 容量 → "Công suất" 1.579; 皮质特征 → "Đặc điểm vỏ não" 739; 适用场景 →
  "Kịch bản áp dụng" 3.573; 上市年份季节 → "niêm yết" ~4.000; 适用/适合人群 → "đám đông" 1.837;
  安全等级 → "mức độ bảo mật" 813. File `desc_ten_dich_sai_nghia.csv`.
- **DS16 (mức 1)** 是否X ("có phải X không") → "Có nên X" / "cho dù X": 4.047 cặp / 2.877 dòng
  (vd 是否进口 → "Có nên nhập khẩu hay không" 974).
- **DS17 (mức 1) Giá trị dịch sai nghĩa** (32 lỗi đọc tay): 6.143 giá trị / 3.944 dòng. Vd 苹果
  (dòng máy) → "táo" 925; 单根 → "Một gốc" 616; 实拍无模特 → "không có mô hình" 368; 前系带 → "cà vạt"
  366; 基础大众 → "Volkswagen" 342; 长爬 → "Leo núi dài" 340; 鸭屎香 → "phân vịt" 119 (lỗi D03 của
  title lặp lại); 单鞋 → "Căn hộ". Nhiều lỗi đi vòng qua tiếng Anh (sole → "duy nhất", flats →
  "căn hộ", brushed → "đánh răng"). File `desc_gia_tri_dich_sai_nghia.csv`.
- **DS18 (mức 1) Sai địa danh** nơi sản xuất: 997 cặp / 990 dòng (温州 → "Vũ Hán" 224; 台州 → "Thái
  Châu"/"Chợp Châu"; 义乌 → "Nhật Bản" 38; 莆田 → "Phủ Điền" 68).
- **DS20 (mức 1)** số thập phân bị tách ("53.5" → "53, 5"): 51 cặp / 39 dòng.
- **DS19 (mức 2) Giá trị phổ biến dịch không thống nhất**: 726 giá trị gặp ≥ 50 lần; 67.932 cặp /
  13.652 dòng dùng cách dịch thiểu số (vd 通用 → thông dụng / đa năng / phổ quát / tổng quan). 明前
  sai ở mọi cách dịch ("trước Tết Trung Thu", "trước thời nhà Minh", "trước ngày mai").
- **DS21 (mức 3) Giá trị rỗng nghĩa** (其他, 无, 见包装, 咨询客服, "/", "-", "."): 20.792 cặp (5,1%) /
  10.036 dòng.
- **DS22 (mức 4)** dấu chấm cuối do máy thêm: 7.855 giá trị / 5.645 dòng. **DS23 (mức 4)** "KHÔNG" in
  hoa: 4.456 giá trị / 1.986 dòng. **DS24 (mức 4)** ký tự lạ: 683 giá trị / 167 dòng (1 ký tự hỏng
  "�": 意境-橙 → "Tâm trạng-O�"; 4 chữ Hàn: 德绒 → "Đề롱"; 9 ký tự vùng riêng PUA; dấu câu Trung,
  khoảng trắng thừa). **DS25 (mức 0, cấu trúc)** tên thuộc tính lặp trong cùng sản phẩm: 47 dòng.

Đã kiểm, **không có lỗi**: đảo ngược có/không (是 → "Không", 否 → "Có"): 0 cặp; đảo giới tính: 1 cặp
(男士 → "các bạn nữ", `592488420290`); bản Việt rỗng khi bản Trung có giá trị: 0; dạng Unicode NFC: đúng hết.

Bộ dò được sửa trước khi đưa vào notebook (báo nhầm em thấy khi đọc mẫu): 苹果 → "táo" chỉ tính ở
thuộc tính dòng máy (适用型号, 适用机型...), vì ở 颜色/花型 thì "táo" là đúng; 领标 chỉ tính "Không có
cổ áo" (vì "Có nhãn cổ áo" là đúng); 粘胶 chỉ tính 粘胶鞋 (粘胶 trong vải là viscose, đúng); "IN HOA"
lần đầu viết regex sai (đếm cả "Là"); dấu câu Trung không tính khi bản Việt chép nguyên mã hàng.

Tổng sau khi thêm: 27 mục (25 lỗi + 2 cấu trúc). Dòng dính ≥ 1 lỗi mức 1: **12.790 (78,24%)**
(trước: 744), chủ yếu do DS15 (tên dịch sai lặp ở rất nhiều sản phẩm, sửa 1 lần bằng từ điển
tên là hết); mức 1–2: 16.333 (99,91%). `desc_issues_evidence.csv`: 113.353 dòng.

**Prompt 30:**
> "Vậy tổng hợp lại là có 25 lỗi từ nghiêm trọng tới ít nghiêm trọng"

→ Xác nhận: **25 vấn đề** (mức 1: 7, mức 2: 9, mức 3: 5, mức 4: 4) + 2 mục cấu trúc không tính
là lỗi (DS01, DS25) = 27 dòng trong `desc_issues_summary.csv`. Nói rõ thêm: trong 25 vấn đề có
5 mục là "nghi, cần đọc" (DS04, DS06B, DS09B; DS10 còn lẫn báo nhầm; DS21 là giá trị ít thông
tin chứ không phải dịch sai) và 3 mục không phải lỗi dịch (DS11 trùng lặp, DS12 thông tin cá
nhân, DS02 không ghép được). Không chạy hay sửa gì.

**Prompt 31:**
> "Tui đã coi qua các lỗi đó và đúng là có các lỗi như vậy. 1. Vậy với từng lỗi đó, bạn hãy đề
> xuất các giải pháp tối ưu nhất, bạn phải tìm kiếm trên các bài báo, trên web ... Với mỗi giải
> pháp, bạn phải đưa ra ví dụ (input, kỹ thuật thực hiện, output, có thể đưa ra rủi ro ...). Có
> thể xuất toàn bộ mục 1 ra file csv riêng, đặt tên là pro_solu_descriptions.csv ... 2. ... đưa
> ra pipeline xử lý description (đi qua bước nào trước, bước nào sau,...) ... trình bày bằng hình
> vẽ các khối ... Nếu bạn có thắc mắc thì phải hỏi lại tui tới khi hiểu kĩ nhé"

→ Anh **xác nhận các lỗi là có thật**. Em hỏi lại 4 câu, anh chọn cả 4 theo đề xuất (**quyết định
của anh, áp cho pipeline description**):
1. Lỗi có quy luật thì **sửa** bằng luật/từ điển đã duyệt; cặp lỗi lẻ không sửa được thì **gắn mã
   lý do và loại khỏi train** (vẫn giữ trong file để truy vết).
2. LLM (Claude/GPT/Qwen...) **chỉ để gợi ý**, người duyệt trước khi áp; không cho LLM sửa thẳng.
3. Đầu ra gồm **cả 2 dạng**: bảng cặp thuộc tính sạch và description_zh/vi dựng lại.
4. Thuộc tính giao dịch 1688: **giữ, gắn nhãn vai trò** (theo bảng anh duyệt).

Tra tài liệu (WebSearch/WebFetch): OpusFilter, Bifixer/Bicleaner, bicleaner-hardrules, sacremoses,
Presidio, cn2an, CLDR (định dạng số tiếng Việt), Dinu 2019, WMT 2023 Terminology, Jia et al. 2022
(thuật ngữ thuộc tính TMĐT), Zhang et al. 2024 (RAG cho LLM dịch tiêu đề TMĐT), xCOMET (Guerreiro
2024), GEMBA-MQM (Kocmi & Federmann 2023), Peter 2023, Raunak 2023, Khayrallah & Koehn 2018, Briakou
& Carpuat 2021, Lee 2022, Wu & Wang 2007 (dịch vòng qua ngôn ngữ trung gian). Một trích dẫn WMT 2023
em không chắc số hiệu ACL Anthology nên dùng link đã kiểm được (ResearchGate).

Đã làm (chưa chạy pipeline, chưa sửa dữ liệu):
1. **`data/eda/description/pro_solu_descriptions.csv`** (32 dòng, `utf-8-sig`): mỗi dòng 1 giải pháp
   cho 25 vấn đề + 2 mục cấu trúc + vai trò, bước P1 + P11 + P13; cột: bước pipeline, mã, mức, số
   dòng dính, giải pháp, quyết định (sửa/loại/gắn nhãn/nhóm), product_id ví dụ, input thật, kỹ
   thuật, output dự kiến, rủi ro, cách giảm rủi ro, cần anh duyệt gì, nguồn. Output là **dự kiến**,
   chưa chạy. Script tạo file ở scratchpad (`write_solutions.py`).
2. **Pipeline 14 bước** P0–P13: Đầu vào → Ghép cặp (DS01, DS25, DS02) → Ký tự (DS24) → Thông tin cá
   nhân (DS12, trước mọi bước gửi ra LLM) → Gắn nhãn vai trò + giá trị rỗng nghĩa (DS21) → Gỡ lỗi
   đánh số (DS06, DS06B) → Từ điển tên (DS15, DS16, DS05, DS07) → Từ điển giá trị có điều kiện (DS17,
   DS18, DS19, DS13, DS09A/B, DS08, DS07) → Số và đơn vị (DS10, DS03, DS20) → Định dạng (DS22, DS23,
   DS14) → Kiểm nghĩa phần đuôi dài (DS04 + đuôi DS15/17, QE + LLM gợi ý) → Loại có mã lý do → Khử
   trùng lặp và nhóm (DS11) → Dựng output và kiểm định. 7 cổng duyệt tay (P3, P4, P5, P6, P7, P10,
   P13); vòng lặp: lỗi mới ở P10/P13 thêm vào từ điển rồi chạy lại từ P6.
   Trang sơ đồ: `docs/design/description_pipeline.html`, đã đăng artifact riêng tư
   https://claude.ai/artifact/C1UfherrsUVvnKXkZtweV2 (lần đăng đầu bị từ chối vì trang chứa ký tự
   '�' làm ví dụ, đã đổi sang `&#xFFFD;`).

Chờ anh chốt trước khi viết code pipeline: tên hãng Hán Việt hay pinyin (DS09B); dấu thập phân số
đo (DS14); ký hiệu 【】 ~ | * (D16, mặc định giữ theo charter); chọn LLM và bộ dò P10 (xCOMET cần
GPU); cách chia tập (chung với title).

**Prompt 32:**
> "Bạn hãy làm rõ 5 mục cần tui chốt, để tui hiểu rõ và quyết định, có thể recommend cho tui cách
> giải quyết tối ưu, hiệu quả cho từng mục"

→ Đo thêm bằng 1 script đọc-không-ghi (`decide2.py`, scratchpad) để có căn cứ:
- **Dấu thập phân trên Tiki** (số đo có đơn vị, 13.200 sản phẩm): dấu chấm 5.213 lần / 2.439 sản
  phẩm, dấu phẩy 1.933 lần / 1.011 sản phẩm. → **Đính chính** khuyến nghị Prompt 31 (dấu phẩy theo
  CLDR): nay khuyến nghị **giữ dấu chấm** như bản Trung. Đã sửa dòng DS14, DS20 trong
  `pro_solu_descriptions.csv` và trang pipeline (đăng lại, Version 2).
- **Ký hiệu trên Tiki**: 【 ở title 0, ở mô tả 11; [ ở title 525, mô tả 202; ~ mô tả 458; | 180 + 187;
  * mô tả 2.301; ★ 66. → Khuyến nghị D16: không xoá; bản Trung giữ nguyên, bản Việt đổi 【】 → [ ],
  giữ ~ | *.
- **Tên hãng trên Tiki**: hãng Trung Quốc viết Latin/pinyin (Ugreen 126, Hoco 80, Deli 69, Baseus 46,
  Xiaomi 23); 240/2.185 hãng có dấu tiếng Việt đều là hãng Việt (Trung Nguyên, Điện Quang...), không
  thấy hãng Trung nào viết Hán Việt. → Khuyến nghị DS09B: **pinyin**.
- **Quy mô phần đuôi dài P10**: 97.439 cặp có giá trị chữ Hán ngoài từ điển giá trị phổ biến, tức
  50.608 cặp (zh, vi) khác nhau (~2,54 triệu ký tự Hán); 1.752 tên thuộc tính ngoài top 300.
- **Chia tập**: nhóm theo (title_zh + shop) có 6.781 nhóm, lớn nhất 183 dòng; gộp thêm description
  trùng y hệt + cụm gần trùng thì 6.679 nhóm, lớn nhất **283 dòng (1,73%)** → vẫn chia được.
- Giá LLM lấy từ tài liệu Claude API (không theo trí nhớ): Opus 5.5 $4/$20, Sonnet 5.5 $2/$10,
  Haiku 4.5 $1/$5 mỗi triệu token input/output; Batch API giảm 50%. Ước tính chi phí trong câu trả
  lời là **ước lượng thô**, chưa đếm token thật.

Đã gửi anh giải thích 5 mục kèm khuyến nghị (xem câu trả lời trong chat). Chưa chạy hay sửa dữ liệu.

**Prompt 33:**
> "4. ... Dùng LLM để làm gì, tui đang sử dụng gói Claude pro, liệu có sử dụng được được. Hay còn
> phương pháp nào khác nếu Claude quá tốn / 5. Chia train / dev / test để làm gì, tui chưa đưa vào
> fine tune mà"

→ Đo thêm (đọc-không-ghi): phần đuôi dài P10 có 97.439 lượt / 50.608 cặp khác nhau, trong đó
**40.732 cặp chỉ gặp 1 lần**; 1.000 / 3.000 / 5.000 / 10.000 cặp phổ biến nhất phủ 27,4% / 40,6% /
47,7% / 58,3% số lượt; 15.457 dòng có ít nhất 1 cặp đuôi dài chỉ gặp 1 lần → đuôi dài phân tán, kiểm
hết bằng tay hay LLM đều nặng.

Trả lời (chi tiết trong chat):
- Gói **Claude Pro** dùng được cho claude.ai và Claude Code (phiên này) trong giới hạn lượt dùng,
  nhưng **không gồm API** (API trả tiền riêng theo token). Đề xuất: việc gợi ý từ điển (~1.100 mục)
  và đọc ~3.000 cặp đuôi dài phổ biến nhất làm **ngay trong Claude Code**, chia nhiều phiên, không tốn
  thêm tiền; phần còn lại (~47.600 cặp hiếm) **không gửi LLM trả phí**: gắn nhãn `chua_kiem_nghia`,
  vẫn cho vào train, không bao giờ lấy làm test; nếu còn thời gian thì sàng lọc miễn phí bằng mô hình
  mở (Qwen) trên Colab GPU hoặc Ollama local (máy anh 4 GB VRAM chỉ chạy tốt mô hình ~3–4B).
- **Chia tập chưa cần chốt bây giờ**: làm sạch không cần biết dòng nào thuộc train/test; pipeline chỉ
  gắn `nhom_trung` (nhóm trùng) ở P12 để lúc fine-tune chia trong vài phút. Lý do duy nhất nên nghĩ sớm:
  tập test 500–1.000 cặp phải do người duyệt, tốn công nhất, nên chọn trước để duyệt dần. Đề xuất
  **hoãn**, đánh dấu chờ đến khi nhóm chốt mô hình.

**Prompt 34:**
> "ở mục 4 việc b tầng 2, tui có thể chạy collab Qwen mà, có quá mất nhìu thời gian hay rủi ro, khó
> khăn gì ko / mục 5 hoãn"

→ **Quyết định của anh: mục 5 (chia train/dev/test) HOÃN** tới khi nhóm chốt mô hình; pipeline chỉ
gắn `nhom_trung` ở P12. Không bắt đầu việc chia tập.

Tra nhanh (WebSearch): vLLM chạy được trên T4 của Colab (compute capability 7.5 ≥ 7.0) nhưng phải
đặt `dtype=float16` vì T4 không hỗ trợ bfloat16; Qwen2.5-7B-Instruct giấy phép Apache 2.0 (bản 3B
và 72B có giấy phép khác).

Trả lời về Qwen trên Colab cho tầng 2 (~47.600 cặp hiếm): làm được; thời gian máy chạy **chưa đo**,
ước lượng thô vài giờ nếu dùng vLLM gom lô, lâu hơn nhiều nếu dùng transformers thường; rủi ro chính:
chất lượng tiếng Việt của mô hình 7B (bắt nhầm/bỏ sót), Colab miễn phí hay ngắt và không bảo đảm GPU,
file ra nằm trên máy ảo (bài học notebook title), JSON hỏng, cài vLLM lệch phiên bản. Đề xuất: chạy thử
500 cặp có nhãn trước để đo tốc độ và độ chính xác, ghi kết quả từng lô vào Google Drive để chạy tiếp
được khi bị ngắt; Qwen chỉ làm bộ lọc (gắn cờ), không sửa. Chi tiết trong chat. Chưa viết code.

**Prompt 35:**
> "ủa chứ không phải nếu tui chốt thì bạn viết file notebook pipeline để xử lý hử" → em xác nhận đúng
> và mở bảng hỏi 4 điểm còn chờ (tên hãng, dấu thập phân, ký hiệu, cách gợi ý từ điển); **anh từ chối
> bảng hỏi** và hỏi tiếp: "Tui cần duyệt tay những gì, có phương pháp nào tối ưu hơn duyệt tay không"

→ Chưa chốt 4 điểm (bảng hỏi bị từ chối), chưa viết notebook. Trả lời: liệt kê toàn bộ việc duyệt tay
theo từng cổng, và đề xuất giảm công duyệt bằng: (1) Claude (phiên Claude Code) duyệt trước và điền
sẵn, anh chỉ **kiểm mẫu ngẫu nhiên** (chấp nhận cả bảng nếu mẫu đạt; quy tắc 3/n: 0 lỗi trên n mẫu →
tỉ lệ lỗi thật < 3/n với độ tin cậy 95%); (2) chỉ đọc kỹ những mục **lan rộng** (từ điển tên/giá
trị) và những mục Claude **thay đổi** so với cách dịch đa số; (3) bước nào hành động an toàn (che số
điện thoại, gắn nhãn vai trò, loại cặp) thì làm mặc định thận trọng, không cần duyệt; (4) tầng 2: Claude
gắn nhãn 300 cặp để đo Qwen, anh kiểm lại 30. Chỉ còn 2 việc không thay được: kiểm mẫu cuối trước/sau
và tập test do người duyệt (đã hoãn cùng mục 5). Chi tiết và ước lượng thời gian trong chat.

**Prompt 36:**
> "Vậy tóm lại đề xuất của bạn cho mục 4 là gì"

→ Tóm tắt đề xuất mục 4 (chưa chốt): không mua API; (a) gợi ý từ điển ~1.100 mục và (b) tầng 1 đọc
~3.000 cặp đuôi dài phổ biến nhất làm bằng Claude Code (gói Pro), chia nhiều phiên; tầng 2 ~47.600
cặp hiếm chạy Qwen2.5-7B-Instruct trên Colab (vLLM, float16, ghi từng lô vào Google Drive) làm bộ lọc
gắn cờ, sau khi chạy thử 500 cặp; Claude gắn nhãn 300 cặp để đo Qwen; anh chỉ kiểm mẫu ngẫu nhiên;
cặp Qwen gắn SAI_NGHIA bị loại nếu độ chính xác đủ, còn không thì chỉ gắn `chua_kiem_nghia`.

**Prompt 37:**
> "Về việc khoảng 4–6 giờ anh đọc mẫu, cho toàn bộ pipeline chứ không riêng mục 4. Tui áp dụng những
> nguyên tắc bạn đưa được không" (kèm đoạn 3 nguyên tắc giảm duyệt tay em viết ở Prompt 35)

→ Trả lời: **áp dụng được**, kèm điều kiện: mẫu do notebook bốc ngẫu nhiên với seed cố định và ghi lại;
quy tắc 3/n chỉ đúng khi mẫu ngẫu nhiên và bảng đồng nhất (n = 30 → dưới ~10%, n = 50 → dưới ~6%,
độ tin cậy 95%); bảng tác động lớn (từ điển tên/giá trị) không dùng kiểm mẫu thay cho đọc mục đã sửa,
mà lấy mẫu theo trọng số số dòng; có lỗi thì sửa rồi bốc mẫu MỚI; ghi số mẫu và số lỗi vào báo cáo;
tập test vẫn phải người duyệt toàn bộ (charter §7). Đề xuất ghi các nguyên tắc này vào `CLAUDE.md` mục
5.2 (đang ghi "Nhật Anh duyệt tay") nếu anh đồng ý. Chưa sửa `CLAUDE.md`, chưa viết notebook.

**Prompt 38:**
> "bạn viết code file notebook theo pipe artifact v2 bạn đã gửi cho tui đúng ko, đi theo các khối trong
> đó đúng ko"

→ Xác nhận: notebook đi đúng 14 khối P0–P13 của trang pipeline (Version 2), mỗi khối một phần, mỗi khối
ghi một Parquet mới vào `data/processed/desc/`. Liệt kê các chỗ trang v2 chưa cập nhật theo các trao
đổi Prompt 32–37 (P2 đổi 【】 → [ ] ở bản Việt; P3 che hết theo hướng thận trọng, không cần duyệt; P5
Claude chấm mẫu, anh kiểm 30; P6/P7 Claude điền gợi ý trong Claude Code, anh đọc mục đã sửa + mẫu theo
trọng số; P7 tên hãng chữ Hán → pinyin, bỏ 200 mẫu Hán Việt; P8/P9 dấu chấm; P10 chia 2 tầng, tầng 2
Qwen là notebook Colab riêng; P12 chỉ gắn `nhom_trung` vì mục 5 hoãn; cổng duyệt dùng kiểm mẫu ngẫu
nhiên). Đề xuất cập nhật trang lên Version 3 cho khớp code, sau khi anh chốt 5 điểm đang chờ.

**Prompt 39 (quyết định chính thức của anh):**
> 1. Tên hãng: pinyin, lập danh sách từ chung để loại trừ, anh đọc 200 mẫu trước khi áp. 2. Dấu thập
> phân: dấu chấm; chỉ đổi số 1–2 chữ số sau dấu và có đơn vị (anh duyệt cách xử lý). 3. Ký hiệu: không
> xoá; bản Trung giữ; bản Việt 【】 → [ ], giữ ~ | *. 4. LLM: như đề xuất (chia phiên, theo tầng, Claude
> Code, Colab Qwen, kiểm tay). 5. Ghi 5 điều kiện kiểm mẫu vào CLAUDE.md. "bắt đầu làm".
> Giữa lượt anh nhắn: "tạm dừng để tui nhấn compact được ko, rồi bạn chạy tiếp tiến trình".

→ Đã làm:
1. **`CLAUDE.md`**: mục 5.2 thêm quy trình kiểm mẫu 5 điều kiện; thêm mục **5.3b** (mọi quyết định pipeline
   description 02/10); mục 5.5: D16 đã chốt, chia tập hoãn (kèm khuyến nghị nhóm khi tới lúc).
2. **`notebooks/desc-pipeline.ipynb`** (29 ô, 14 khối P0–P13 theo trang pipeline): dựng bằng script tạm
   `build_pipe_nb.py` trong scratchpad của phiên (không nằm trong repo; từ giờ nên sửa thẳng notebook).
   Chạy bằng kernel `.venv`: **không lỗi, ~4 phút**. Output trong `data/processed/desc/` (~199 MB): mỗi
   khối 2 file `pN_cap.parquet`, `pN_gia_tri.parquet` (zstd), `p12_san_pham.parquet`,
   **`desc_pairs_clean.parquet`**, **`desc_clean.parquet`**, `log_thay_doi.parquet`; 15 file duyệt/mẫu trong
   `data/processed/desc/duyet/`.
3. Kết quả lần chạy cuối (dữ liệu **CHƯA chính thức**, 15 cổng chờ + mẫu cuối P13):
   - Giữ **401.409/406.871 cặp (98,7%)** cho train; 16.342 sản phẩm có description sạch; loại nhiều
     nhất ở food 2,7%, home 2,4%, beauty 2,2%.
   - Mã lý do: DS07 2.915, RONG 1.373, DS10 1.043, DS17 165, DS03 62, PII 29, DS18 12, DS09_dai ~11, DS24 5,
     DS25 2.
   - Đo lại trên phần giữ (trước → sau): chữ Hán sót 1.792 → 0; số đuôi đánh số 42.230 → 0; dấu chấm cuối
     do máy thêm 7.118 → 0; IN HOA 4.454 → 0; 【】 4.511 → 14 (chỉ còn trong mã chép nguyên); tên zh có >1
     cách dịch 637 → 356 (tên ngoài top 300).
   - Số thay đổi: P5 đánh số 72.137 giá trị; P6 tên 167.644 cặp; P7 từ điển 134.483 giá trị, tên hãng
     10.011, tiếng Anh theo bảng title 4.824, thuật ngữ 258, địa danh 783; P8 chép mã 16.134; P9 7.463.
   - `nhom_trung`: 6.674 nhóm, lớn nhất 283 sản phẩm (1,73%); 72 sản phẩm trùng y hệt sau khi sạch.
4. **Đuôi dài lớn hơn ước tính**: 182.731 giá trị / **134.705 cặp khác nhau** (đếm cả giá trị trong danh
   sách; ước tính cũ 50.608 chỉ tính cặp 1 giá trị). Tầng 1 (Claude): 4.044 cặp (gồm 1.070 cặp độ dài bất
   thường); tầng 2 (Qwen): 130.661 cặp → nặng gấp ~2,5 lần ước tính. *(Sửa 02/10 ở Prompt 40: bản ghi đầu
   chép nhầm 182.679 / 134.682 / 130.637 của một lần chạy trước; số trên là output thật của lần chạy cuối.
   Sau khi sửa luật ở Prompt 40, số mới xem Prompt 40.)*

Sai/sửa trong lúc làm (đều sửa trước khi chốt, đã chạy lại sạch):
- Bốc mẫu theo trọng số bằng pandas lỗi khi trọng số lệch → đổi sang Efraimidis–Spirakis (numpy, seed cố định).
- **Lỗi do pipeline gây ra (đã sửa)**: luật "chép nguyên mã" lúc đầu áp cho mọi giá trị của vai trò mã/giấy
  tờ, nên chép chữ Hán vào bản Việt cho giá trị mô tả dưới 型号/标准号 (vd "Từ tính thơm không viền" →
  "无边框香薰磁吸", "Xem bao bì bên ngoài" → "见外包装"). Nay chỉ chép khi giá trị Trung không có chữ Hán,
  hoặc là số giấy phép/đăng ký có ≥ 4 chữ số.
- Bảng thuật ngữ thay cả từ tiếng Anh trong cụm tiếng Anh ("Removal mặt nạ-Sheet") → chỉ thay biến thể tiếng
  Việt; 密码箱/拉杆箱 chỉ áp cho category bags (ở mẹ và bé có thể là đồ chơi); giữ viết hoa đầu câu.
- Dấu thập phân chỉ đổi số sát đơn vị ("7,5 * 7,5 * 2.5CM") → đổi mọi số a,b trong giá trị có đơn vị khi bản
  Trung có đúng a.b (diễn giải "có đơn vị đi kèm" là "giá trị có số đo kèm đơn vị"; anh kiểm lại cách hiểu này).
- DS03 thiếu dung sai 斤 → kg (274 → 62 cặp sau khi thêm).
- Bỏ dấu chấm cuối làm mất dấu của chữ viết tắt ("P.U.", "v.v.") → thêm ngoại lệ.
- File duyệt tạo từ các lần chạy lỗi đã **xoá theo tên** (lệnh xoá dùng `*` bị môi trường chặn) rồi chạy lại
  sạch; không file nào có phần anh đã điền.

**Còn lại, làm tiếp sau khi anh compact:**
1. Cập nhật trang pipeline lên **Version 3** cho khớp notebook (P2 【】→[ ], P3 mặc định, P5/P6/P7 cổng kiểm
   mẫu, P7 tên hãng pinyin + 200 mẫu, P8 luật mã thật, P9 dấu chấm, P10 hai tầng và con số đuôi dài mới,
   P12 chỉ gắn nhóm).
2. Phiên Claude Code: điền `goi_y_claude` cho `duyet/P6_ten_thuoc_tinh_editted.csv` (300 tên) và
   `duyet/P7_gia_tri_editted.csv` (914 giá trị), rồi `claude_nhan` cho `duyet/P10_tang1_editted.csv` (4.044 cặp).
3. Viết notebook Colab Qwen cho tầng 2 (130.637 cặp, ghi từng lô vào Google Drive) + 300 cặp hiệu chỉnh.
4. Anh duyệt các file trong `data/processed/desc/duyet/` theo bảng cổng ở cuối notebook, rồi chạy lại notebook.

**Prompt 40 (sau khi anh compact):**
> "Bạn hãy tiếp tục tiến trình nhé, cẩn thận đừng mắc lỗi"

→ Đã làm:
1. **Đối chiếu nhật ký với output notebook**: Prompt 39 chép nhầm số đuôi dài của lần chạy trước → đã sửa
   ở trên.
2. **Hai lỗ hổng luật do em gây ra, tìm thấy khi lấy ví dụ cho trang V3, đã sửa**:
   - Số giấy phép mỹ phẩm/thiết bị y tế bị dịch nghĩa thay vì chép nguyên (vd 国妆特字G20191510 → "Trang
     điểm quốc gia từ đặc biệt G20191510"): 147 giá trị, do regex tên mã thiếu 证号/准字 và luật còn đòi vai
     trò tạm là mã/giấy tờ (特殊用途化妆品证号 đang là mo_ta_hang). Luật mới: (vai trò mã + giá trị Trung không
     có chữ Hán) HOẶC (tên là số giấy phép/đăng ký + giá trị có ≥ 4 chữ số liền nhau). Đã xem trước: cả 147
     giá trị thêm đều là mã thật (国妆特字, 湘械注准, số bằng sáng chế); không giá trị cũ nào bị mất.
   - Dấu phẩy thập phân còn sót vì danh sách đơn vị thiếu "mét", "%", "W", "CBM"...: thêm mét, W, V, mAh,
     CBM, %. Đã xem trước 177 giá trị đổi thêm, đều đúng (1,2 mét → 1.2 mét, 76,5% → 76.5%).
   - Còn đúng 3 giá trị có đơn vị chưa đổi (đều là cặp lệch số kiểu DS10). **345 giá trị không có đơn vị**
     ("1,2*1,2", "F=2,0", "Rộng 11,5") mà bản Trung có đúng a.b: theo luật anh chốt thì KHÔNG đổi → hỏi anh.
3. **Lỗ hổng ở cổng P6/P7 (đã sửa)**: `da_sua` chỉ tính lúc tạo file, nên tên/giá trị Claude gợi ý khác cách
   dịch đa số sẽ lọt khỏi nhóm "anh đọc hết". Nay tính lại mỗi lần chạy (`da_sua` = 1 HOẶC `goi_y_claude`
   khác đa số). Mẫu 30 của P6/P7 chỉ bốc **sau khi** Claude điền xong: thêm cờ `CLAUDE_XONG` ở P0 (đổi thành
   True sau mỗi phiên), mẫu ghi vào file `_mau_v2`. Tầng 1 cũng chỉ bốc mẫu khi `CLAUDE_XONG["P10_tang1"]`.
4. **Không xoá được file duyệt cũ**: lệnh xoá 15 file duyệt chưa điền (đã kiểm: mọi cột `anh_duyet`,
   `anh_ghi_chu`, `ket_qua`, `claude_nhan`, `claude_sua` đều trống) bị môi trường chặn. Em không vòng qua;
   thay vào đó dùng **phiên bản file**: `mau_kiem(..., phien="v2")`, tầng 1 ghi `P10_tang1_v2_editted.csv`,
   mẫu cuối `P13_mau_cuoi_mau_v2.csv`. Các file cũ **không còn được đọc**, anh xoá tay khi muốn:
   `P10_tang1_editted.csv`, `P13_mau_cuoi_mau_v1.csv`, `P6_ten_mau_mau_v1.csv`, `P7_gia_tri_mau_mau_v1.csv`.
   Các file duyệt khác không bị ảnh hưởng bởi hai chỗ sửa nên giữ nguyên.
5. Dựng lại notebook từ script `build_pipe_nb.py` (ô không đổi giữ nguyên nguồn; không sao lưu riêng vì
   notebook do script sinh và anh chưa sửa, đã kiểm thời gian sửa file) và chạy trọn: **không lỗi, 244 giây**.
   Số mới (vẫn CHƯA chính thức): mã thật 16.281 (+147); P8 3.073; P9 7.618; đuôi dài 182.589 giá trị /
   134.613 cặp; tầng 1 **3.997** cặp (1.020 độ dài bất thường); tầng 2 **130.616**; `chua_kiem_nghia`
   58.821 cặp; DS07 2.914; giữ **401.410/406.871 (98,7%)**.
6. **Trang pipeline Version 3** (`docs/design/description_pipeline.html`, cùng URL
   https://claude.ai/artifact/C1UfherrsUVvnKXkZtweV2): khớp notebook (P2 【】→[ ], P3 không cổng, cổng kiểm
   mẫu P4–P7/P13, tên hãng pinyin + 200 mẫu, luật mã thật, dấu chấm thập phân, P10 hai tầng, P12 chỉ gắn
   nhóm), thêm bảng kết quả trước/sau và bảng trạng thái 17 cổng; mọi ví dụ lấy từ `log_thay_doi.parquet`
   và đã đối chiếu lại; bỏ cn2an khỏi nguồn vì code không dùng.
7. **`CLAUDE.md` 5.3b**: sửa số tầng 2 (130.616, thay ước tính ~47.600); thêm luật mã thật; ghi rõ danh sách
   đơn vị và điều kiện "bản Trung có đúng a.b".

**Còn lại:**
1. Phiên Claude Code: `goi_y_claude` cho P6 (300 tên) và P7 (914 giá trị), rồi `claude_nhan` cho
   `P10_tang1_v2_editted.csv` (3.997 cặp); xong bảng nào thì đổi `CLAUDE_XONG` và chạy lại.
2. Notebook Colab Qwen tầng 2 (130.616 cặp) + 300 cặp hiệu chỉnh.
3. Anh: quyết 345 số thập phân không đơn vị; xoá tay 4 file cũ (nếu muốn); duyệt các file trong `duyet/`.

**Prompt 41:**
> "Bạn hãy tiếp tục tiến trình nhé, cẩn thận đừng mắc lỗi"

→ Đã làm:
1. **Gợi ý P6 (300 tên)**: em đọc hết 300 tên kèm các cách dịch và giá trị ví dụ; gợi ý **86 tên** vào
   `goi_y_claude`, lý do ở cột mới `ly_do_claude` (script tạm `fill_p6.py`: chặn nếu cột `anh_duyet`,
   `anh_ghi_chu`, `goi_y_claude` đã có dữ liệu; không đụng cột của anh). Nguyên tắc: tên tạm ổn thì để trống;
   chỉ gợi ý khi sai nghĩa (包装体积 'Khối lượng' → 'Thể tích đóng gói', 误差范围 'Phạm vi lỗi' → 'Sai số',
   记录仪安装类型 'máy ghi âm' → 'camera hành trình', 滑轮 'Ròng rọc' → 'Bánh xe'...), câu 是否 lủng củng
   ('Nó có...', 'đôi tai đơn'), thiếu nhất quán (适用机型/适用型号 → 'Dòng máy tương thích'; 规格 → 'Quy cách'),
   hoặc viết hoa từ viết tắt (AQL, 3C, AI, Bluetooth). 淘货类别 → 'Phân khúc khách hàng' em ghi rõ "anh xem kỹ".
   Tên '1', '2', '3' (rác) để nguyên.
2. Bật `CLAUDE_XONG["P6"] = True`, chạy lại: **115 tên** cho anh đọc hết (33 sai nghĩa + 82 gợi ý khác đa số;
   4 gợi ý chỉ đổi chữ hoa không tính là sửa); bốc 30 mẫu phiên mới `P6_ten_mau_mau_v2.csv`. Kiểm: 86/86 gợi ý
   áp đúng vào `desc_pairs_clean.parquet`; mẫu v2 không chứa tên đã sửa; cột của anh vẫn trống. Đổi tên ở
   212.537 cặp (trước 167.644).
3. **Lỗi thứ tự (đã sửa)**: mẫu cuối `P13_mau_cuoi_mau_v2.csv` bốc ở lần chạy trước, trước khi áp gợi ý P6,
   nên nội dung đã cũ. Nay mẫu P13 chỉ bốc khi **mọi cổng khác đạt** (file `_mau_v3`); hiện trạng thái
   `chua_toi`. Thêm `P13_mau_cuoi_mau_v2.csv` vào danh sách file cũ anh xoá tay.
4. Trang pipeline cập nhật số P6 và trạng thái cổng (Version 4 của artifact, cùng URL).

**Còn lại:** P7 `goi_y_claude` (914 giá trị) → tầng 1 `claude_nhan` (3.997 cặp) → notebook Colab Qwen tầng 2;
anh: 345 số thập phân không đơn vị, xoá tay 5 file cũ, duyệt `duyet/`.

*(Prompt hỏi "các file cần duyệt nằm ở đâu" không ghi theo yêu cầu của anh.)*

**Prompt 42:**
> "Hãy cho tui cách duyệt, có từ điển gì không, tui không biết tiếng Trung"

→ Trả lời (không sửa file): phần lớn việc duyệt chỉ cần đọc tiếng Việt (P5, P4, mẫu P6, thuật ngữ); phần cần
tra dùng Google Dịch (có pinyin), từ điển Hán Việt Thi Viện, Wikipedia tiếng Việt (địa danh), MDBG, Tiki. Hướng
dẫn tiêu chí DUNG/SAI từng file; không chắc thì ghi SAI + ghi chú (notebook coi mọi ô `ket_qua` có chữ là đã
kiểm). Đề xuất thêm cột hỗ trợ khách quan (`pinyin`, `link_1688`, `gia_tri_vi_vd`), chờ anh đồng ý; không thêm
cột "giải thích nghĩa" của Claude để giữ việc kiểm tra độc lập. Đã kiểm `P5_danh_so_le_mau_v1.csv`: 100/100
dòng có `truoc` khác `sau` (không lỗi).

**Prompt 43:**
> "các file bạn cần tui duyệt thì bạn đã xử lý kĩ rồi phải không"

→ Trả lời trung thực: **chưa phải tất cả**. Kỹ: P6 (đọc hết 300 tên, 86 gợi ý), P2 (5). Chưa kỹ: vai trò P4 do luật
EDA gán, em chưa đọc từng tên (đã biết 特殊用途化妆品证号 bị gán mo_ta_hang); mẫu P5 em chưa đọc; bảng địa danh,
từ chung, thuật ngữ em tự viết, chưa đối chiếu nguồn (thuật ngữ còn 5 mục trống); tên hãng pinyin tự động, rủi ro
chữ đa âm (长, 行, 乐) chưa kiểm; P7 giá trị chưa làm (đa số cũng có thể sai: 浙江台州 → 'Thái Châu').
Đề xuất: em soát trước mọi file (P4 → P5 → địa danh → đa âm tên hãng → thuật ngữ → P7), lỗi luật thì sửa và bốc
mẫu phiên mới, rồi anh mới kiểm độc lập. Chờ anh đồng ý.

**Prompt 44:**
> "Bạn hãy bắt đầu đi, duyệt kĩ từng file nhé, tránh bỏ sót các lỗi"

→ Em soát từng file, sửa lỗi có quy luật rồi chạy lại notebook sau mỗi nhóm sửa (mọi lần chạy: không lỗi):
1. **P4 vai trò (đọc hết 300 tên)**: 12 gợi ý ở cột mới `goi_y_claude` + `ly_do_claude` của
   `data/eda/description/desc_vai_tro_thuoc_tinh_editted.csv` (3 tên là mã: 特殊用途化妆品证号, 商品3C认证码,
   质检报告编号; 4 thành giao dịch: 装箱数量, 加工方式, 是否有导购视频, 是否支持到店安装; 5 thành mô tả hàng:
   化妆品功效, 适用范围, 收纳场景, 能力培养, 跨境风格类型). Lỗi luật: luật giao dịch xét trước luật mã nên 专利号
   thành giao dịch → nay luật mã xét trước, thêm 证号/准字/专利号/认证码. Lỗi tiềm ẩn: pipeline lấy thẳng
   `anh_duyet` làm vai trò, anh ghi "OK" sẽ thành vai trò "OK" → nay "OK" = đồng ý gợi ý, giá trị lạ thì cảnh báo.
   Mã thật 16.281 → 16.705 (kiểm từng nhóm tăng: ZL…, 3C, 国妆特字, 械注准; 质检报告编号 có giá trị rác '8', '-'
   chép nguyên vô hại). Cổng mới `P4_vai_tro_da_sua` (anh đọc 12 tên) + 30 mẫu `P4_vai_tro_mau_v2.csv` (em đã đọc:
   30/30 đúng).
2. **Nguy cơ mất dữ liệu (đã sửa)**: notebook EDA `eda-description.ipynb` ghi đè vô điều kiện mọi file `_editted`
   khi chạy lại. Sửa đúng 1 ô (`ghi_csv`): file `_editted` đã có thì ghi bản mới ra `_editted_moi.csv`. Sao lưu:
   `notebooks/eda-description.backup_2026-10-02.ipynb`; so với bản sao lưu chỉ khác 4 dòng thêm.
3. **P5 (đọc 30 + 100 mẫu)**: 30/30 đúng; 100 mẫu số đuôi lẻ có 1 lỗi có quy luật: 款式十一 "Phong cách 11" bị bỏ
   "11" vì luật chỉ nhận số Hán từng chữ. Đo lại thì luật v2 đã áp nhầm 10 giá trị (图十一 "Hình 11" → "Hình").
   Sửa: nhận số Hán ghép (十一, 二十七, 一百二十). Bốc lại mẫu v2 (`P5_danh_so_mau_v2`, `P5_danh_so_le_mau_v2`),
   em đọc lại: 30/30 và 100/100 đúng.
4. **Địa danh (43 mục đúng âm Hán Việt nhưng phủ thiếu)**: 1.544 giá trị nơi sản xuất không tách được, nhiều bản
   dịch sai: 新余 'Tân Ngu' (98 giá trị), 澄海 'Thành Hải', 徐州 'Tô Châu', 荆州 'Cương Châu', 张家界 'Trương Gia
   Kiệt', 温岭 'Vân Lĩnh', 内蒙古 'Mongolia Trung'... Bổ sung 44 địa danh (đã kiểm từng âm) → bảng 87 mục ở
   `P7_dia_danh_v2_editted.csv`; tách hậu tố 省/市/区/县/镇 sau địa danh, nhận dấu ';'; thêm 产地-国内 (bị thiếu).
   Dựng lại 783 → 1.089 giá trị; DS18 12 → 3. Giá trị đã dựng lại không đưa vào hàng đợi kiểm nghĩa P10.
5. **Tên hãng**: đọc 4 lượt × 200 mẫu (v2 ~5% sai, v3 ~2,5%, v4 1%, v5 ~1,5% còn lại khó bắt bằng luật) và quét
   toàn bộ 15.834 giá trị. Lỗi có quy luật đã sửa: tên Latin bị dịch nghĩa (234 giá trị: zx → 'từ đó', LIFE HOME →
   'NHÀ CUỘC SỐNG') → chép nguyên; mất chữ số (70迈 → 'Mai') → '70mai'; chú thích trong ngoặc bị phiên âm (天王
   （食品）→ 'Tianwangshipin') → 'Tianwang (thực phẩm)'; tên công ty/mô tả → DS09_dai (86 cặp); từ chung bị phiên
   âm (自主 'Zizhu', 否 'Fou', 咨询客服 'Zixunkefu', 贴牌, 白牌, 见产品包装...) → bảng từ chung 16 → 88 mục
   (`P7_tu_chung_ten_hang_v2_editted.csv` + `_bo_sung.csv`); tên + hậu tố loại hàng (美凤箱包 → 'Túi xách Meifeng',
   箱包/玩具/童装/包装/服饰/皮具/茶叶/茶业/鞋业/制衣/食品); bỏ ký hiệu đầu/cuối, dấu câu Trung. Còn lại trong
   `P7_ten_hang_mau_v5.csv` ~3/200: hãng ngoại viết chữ Hán (奥利奥 = Oreo → 'Aoliao'), 东莞点线面数码, 春秋纽巴仑酷跑.
   Chữ đa âm (203 giá trị): pypinyin chọn âm phổ biến, không có quy luật sửa.
6. **Thuật ngữ**: điền 爆款 → 'bán chạy', 网红 → 'nổi tiếng trên mạng'; 一件代发, 补水, 老爹鞋 để trống có chủ đích
   (ghi lý do ở `ghi_chu`). Kiểm trên dữ liệu thì phép thay gây lặp ý ('… nổi tiếng trên mạng trên Internet',
   '[Bán hàng bán chạy]') → thêm cụm thay riêng xét trước, giữ chữ hoa của cụm gốc; kiểm lại: 0 lặp ý.
7. **Thứ tự tạo file (đã sửa)**: file tầng 1 P10 chỉ tạo sau khi Claude xong P7 (`P10_tang1_v3_editted.csv`);
   `file_duyet` nay áp tạm mục mới trong `_bo_sung.csv` và giữ phần anh điền ở đó.
8. Số mới (CHƯA chính thức): giữ 401.362/406.871 cặp (98,6%); DS09_dai 86; DS10 1.023; DS18 3; tầng 1 dự kiến
   3.995, tầng 2 130.446.

File cũ không còn được đọc (anh xoá tay khi tiện): `P4_vai_tro_mau_v1`, `P5_danh_so_mau_v1`, `P5_danh_so_le_mau_v1`,
`P6_ten_mau_mau_v1`, `P7_gia_tri_mau_mau_v1`, `P7_dia_danh_editted`, `P7_tu_chung_ten_hang_editted`,
`P7_ten_hang_mau_v1..v4`, `P10_tang1_editted`, `P10_tang1_v2_editted`, `P13_mau_cuoi_mau_v1`, `_v2` (đều `.csv`).

9. **P7 từ điển giá trị (đọc hết 914)**: 264 gợi ý ở `goi_y_claude` + `ly_do_claude` (script tạm `fill_p7.py`, chặn
   nếu cột của anh có dữ liệu): 192 sửa nghĩa/thống nhất (vd 车缝线 'Chỉ khâu xe' → 'Đường chỉ may', 散装 'Số lượng
   lớn' → 'Bán rời', 莫代尔 'Phương thức' → 'Modal', 40支 '40 sợi' → 'Chi số 40', 90码 '90 thước' → 'Cỡ 90',
   大红袍 'Áo choàng đỏ' → 'Đại Hồng Bào', 浙江台州 → 'Đài Châu, Chiết Giang'), 28 dòng máy 苹果N → 'iPhone N
   (Pro/Plus/Pro Max/Air)', 44 chỉ viết hoa chữ đầu (không tính là sửa). Luật ngữ cảnh 中 → 'Trung bình' mở rộng cho
   鞋底硬度, 防滑性能, 定型性能. Bật `CLAUDE_XONG["P7"]` → 235 giá trị cho anh đọc hết; 30 mẫu `P7_gia_tri_mau_mau_v2`
   (em đã đọc: chấp nhận được, 中性/男女均可 'Trung tính/nam và nữ' hơi gượng).
   - **Lỗi do gợi ý của em (đã sửa)**: 万向轮 → 'Bánh xe xoay 360°' thêm số 360 không có ở bản Trung → 513 cặp bị
     DS10, túi xách bị loại 0,9% → 1,9%. Đổi thành 'Bánh xe xoay mọi hướng'; quét cả 264 gợi ý: còn 7 gợi ý khác số
     nhưng đều nằm trong dung sai (纯 → 100%, 万 → /100, số Hán 三). DS10 sau sửa: 827 (trước mọi sửa: 1.023).
10. File tầng 1 đã tạo: `P10_tang1_v3_editted.csv` (3.994 cặp, gồm 1.019 độ dài bất thường).
11. Số mới (CHƯA chính thức): giữ **401.625/406.871 cặp (98,7%)**; DS07 2.913, RONG 1.373, DS10 827, DS17 110,
    DS09_dai 74, DS03 62, PII 29, DS24 5, DS18 3, DS25 2; tầng 2 130.455 cặp.

**Còn lại:** gắn nhãn tầng 1 (3.994 cặp, `claude_nhan`/`claude_sua`) → notebook Colab Qwen tầng 2 → cập nhật trang
pipeline theo số mới. Anh: 345 số thập phân không đơn vị; ngưỡng sai cho phép của 200 mẫu tên hãng (em đề xuất ≤ 5%
như DS06B, hiện code để 0); duyệt các file trong `duyet/`.

**Prompt 45:**
> "Trong pipeline xử lý data sạch, có bước nào xử lý Tiếng anh không"

→ Trả lời (không sửa file): có một bước, P7.6 (DS08) thay từ THAY (210) / HANVIET (162) của bảng
`title_latin_tokens_editted.csv` khi từ đó không có trong bản Trung; 4.748 giá trị đổi; bảng chưa được anh duyệt
(0 dòng, còn 41 XEM_TAY). Ba lỗ hổng thấy trong log: thay từng từ tạo câu lai ('Good Jin Junmei' → 'Good Jin Tuấn
Mi'); lặp chữ ('dệt flyknit' → 'dệt dệt kim'); tiếng Anh ngoài bảng không được phát hiện ('Ancient Early Taste',
'Alpine Cloud Mist Tea'). Đề xuất: thay theo cụm (THEO_CUM, 246 từ, chưa dùng) trước từng từ + chống lặp chữ; luật
phát hiện ≥ 2 từ tiếng Anh liền nhau không có ở bản Trung → đưa vào hàng đợi kiểm nghĩa P10 (không thay tự động);
anh duyệt bảng title. Chờ anh đồng ý rồi đo quy mô trước khi sửa.

**Prompt 46:**
> "À, những bước nào xử lý Tiếng Anh, thì bạn bỏ qua cho tui nha, không làm, vấn đề xử lý Tiếng Anh bạn Quang sẽ xử
> lý sau. Còn lại giữ như cũ"

→ Đã làm:
1. Tắt P7.6 (thay từ tiếng Anh/pinyin theo bảng title) bằng cờ `XU_LY_TIENG_ANH = False` ở P0 (bật lại được); cổng
   `P7_tieng_anh` = `khong_can`; thêm dòng `tieng_anh` vào `QUYET_DINH`. Không làm 3 đề xuất ở Prompt 45.
   Lưu ý: bước này gồm cả phần pinyin → Hán Việt của bảng (vd Anxi → An Khê) nên cũng tắt theo; luật tên hãng
   (chép nguyên tên Latin, 'other' → 'Khác') thuộc bước tên hãng nên giữ.
2. `CLAUDE.md` 5.3b: ghi quyết định; khi gắn nhãn đuôi dài không coi tiếng Anh sót là lỗi.
3. Chạy lại notebook: không lỗi. P7.6 đổi 0 giá trị (trước 4.748); giữ **401.635/406.871 (98,7%)**; DS07 2.896.
   File tầng 1 lập lại phiên v4 `P10_tang1_v4_editted.csv` (3.996 cặp) vì khoá (zh, vi) đổi; v3 chưa điền, không
   còn được đọc (thêm vào danh sách file cũ anh xoá tay).

**Prompt 47:**
> "Vậy file code notebook pipeline xử lý toàn bộ vấn đề của data (description) đã xong phải không, có thể đưa hơn 16
> nghìn dòng vào chạy được chưa, nếu chạy thì bạn đưa data cleaned lên HF luôn hay sao, hãy cho tui ý kiến nhé"

→ Trả lời (không sửa file): code đủ 14 khối và mọi lần chạy đều là toàn bộ 16.348 dòng (4–9 phút, không lỗi);
output đã có (`desc_pairs_clean.parquet` 406.871 cặp, `desc_clean.parquet` 16.348 sản phẩm). Dữ liệu CHƯA chính thức:
còn anh duyệt + kiểm mẫu, tầng 1 (3.996, em), tầng 2 Qwen (130.469, notebook Colab chưa viết; không bắt buộc vì có
nhãn `chua_kiem_nghia`), 3 quyết định mở, mẫu cuối P13; tiếng Anh do Quang.
**Phát hiện mới:** `desc_clean.parquet` có cột `shop`; 1.473 giá trị tên/địa chỉ nhà sản xuất (产品厂家厂址, 厂名,
生产厂家...) chưa được P3 xử lý → vi phạm charter §6 nếu công bố.
Ý kiến: chưa đẩy HF (HF dataset của nhóm là public; dữ liệu chưa chính thức; phải hỏi trước khi công bố). Lộ trình:
(1) P3 loại cặp tên/địa chỉ nhà sản xuất (mã PII) + bản công bố bỏ cột `shop` — chờ anh đồng ý; (2) duyệt + tầng 1
(+ tầng 2 nếu kịp) tới khi cổng đạt; (3) dựng bản phát hành có phiên bản (chỉ cột cần, SHA256, README/data card);
(4) anh xem và đồng ý mới đẩy, vào thư mục riêng, không ghi đè `bronze/`. Nếu cần sớm: bản nháp nội bộ, không công khai.

**Prompt 48:**
> "Vậy bây giờ tui cần làm những công việc gì nữa để hoàn thành, hãy nêu chi tiết nhé"

→ Trả lời (không sửa file), danh sách việc của anh (~8–9 giờ đọc):
1. Chốt 4 quyết định: P3 loại tên/địa chỉ nhà sản xuất + bỏ cột `shop` khi công bố; 345 số thập phân không đơn vị;
   ngưỡng 200 mẫu tên hãng (đề xuất ≤ 10/200); 3 thuật ngữ trống.
2. Đọc hết mục đã sửa: vai trò 12, P2 5, P6 119 (`da_sua`=1 hoặc có `goi_y_claude`), P7 280 (cùng cách lọc; 44 dòng
   chỉ đổi chữ hoa), địa danh 87, từ chung 25 + 76 (`_bo_sung`), thuật ngữ 11.
3. Kiểm mẫu: P4 30, P5 30 + 100 (≤ 5 sai), P6 30, P7 30, tên hãng 200; rồi Run All và xem bảng cổng.
4. Sau phần của em: 30 + 30 mẫu tầng 1; (tuỳ chọn) chạy notebook Colab Qwen; 200 mẫu cuối P13; duyệt bản phát hành
   trước khi đẩy HF; xoá file cũ nếu muốn.
Phần em song song: sửa P3 + quyết định, gắn nhãn tầng 1, notebook Colab Qwen + 300 cặp hiệu chỉnh, trang pipeline,
bản phát hành + README.

**Prompt 49:**
> "Phần 1. Chốt 4 quyết định (~15 phút, làm trước), theo như bạn đề xuất đi, 3 thuật ngữ cũng bạn chọn nhưng chọn cho
> hợp lý ( tra từ điển ngữ cảnh)"

→ Đã làm (chạy lại notebook: không lỗi; 1 lần MemoryError ở P9 do máy còn ít RAM khi nhiều ứng dụng mở, chạy lại thì qua):
1. **P3**: loại cặp tên/địa chỉ nhà sản xuất, mã mới `PII_NSX` (生产厂家, 产品厂家厂址, 产品厂家厂名, 生产企业, 厂址, 厂名,
   生产地址, 公司, 制造商...; không gồm 是否源头厂家) → **1.478 cặp**; bản công bố sẽ bỏ cột `shop`. Hệ quả: food bị loại
   2,5% → **7,2%** (phần lớn 生产厂家 = '见包装'/'其他' ở hàng thực phẩm).
2. **Số thập phân**: bỏ điều kiện "có đơn vị", vẫn yêu cầu bản Trung có đúng a.b (1–2 chữ số sau dấu). Đã xem 25 mẫu
   trong 345 giá trị đổi thêm: đúng hết (vd 'F1,55' → 'F1.55', '28,8 nhân dân tệ/jin' → '28.8'). P9 7.576 → 7.910.
3. **Ngưỡng 200 mẫu tên hãng**: sai ≤ 10 (5%).
4. **3 thuật ngữ** (đã tra cách dùng của người bán và trang mỹ phẩm Việt):
   - 一件代发 → 'dropship' (nhà cung cấp gửi thẳng từng đơn); chỉ thay cụm dịch sai: 'vận chuyển thả', 'giao hàng một
     món/chiếc/mảnh', 'giao hàng tận nơi' (trừ khi bản Trung có 上门); không thay 'giao hàng' đứng riêng.
   - 补水 → 'cấp nước' (khác 保湿 = 'dưỡng ẩm'); sửa 'Uống nước', 'Hấp thụ nước', 'Hydrat hóa', 'dưỡng ẩm và dưỡng ẩm';
     'dưỡng ẩm' chỉ đổi khi bản Trung không có 保湿/润.
   - 老爹鞋 → 'giày bố' ('dad shoes (giày bố)'; tiếng Anh để Quang); description không có biến thể sai nên 0 thay đổi.
   Kiểm trên log: thay đúng, 0 lặp chữ. Sửa thêm: hàm thay giữ kiểu viết hoa từng chữ ('Mặt Nạ Dưỡng Ẩm' → 'Mặt Nạ
   Cấp Nước', trước đó ra 'Mặt Nạ Cấp nước'). Thuật ngữ đổi 272 → 612 giá trị.
5. `CLAUDE.md` 5.3b ghi đủ 4 quyết định. File tầng 1 lập lại `P10_tang1_v6_editted.csv` (3.993 cặp); v4, v5 chưa điền,
   không còn được đọc.
6. Số mới (CHƯA chính thức): giữ **400.179/406.871 cặp (98,4%)**; mã: DS07 2.896, PII_NSX 1.478, RONG 1.373, DS10 827,
   DS17 110, DS09_dai 74, DS03 62, PII 29, DS24 5, DS18 3, DS25 2; tầng 2 129.802 cặp.

**Prompt 50:**
> "Phần 2. Duyệt các mục em đã sửa (~4 giờ, đọc hết), tui đã đọc OKK hết rồi á / Phần 3. Kiểm mẫu ngẫu nhiên (~2 giờ),
> tui đã kiểm mẫu ngẫu nhiên, OKK hết"

→ Kiểm 14 file duyệt/mẫu: **0 ô đã điền** ở `anh_duyet`/`ket_qua`; thời gian sửa file đều là lúc Claude ghi; không có
file Excel tạm (`~$`) hay `.xlsx`. Không điền thay khi chưa được anh xác nhận cách làm. Hỏi anh: (1) đã đọc nhưng chưa
lưu → mở đúng file, điền, lưu; (2) muốn Claude ghi thay `OK`/`DUNG` kèm ghi chú "Nhật Anh xác nhận qua chat 02/10,
Claude ghi thay" để truy vết. Nói rõ: hai cổng này là lớp kiểm độc lập duy nhất với Claude (Claude đã từng gây lỗi,
vd '360°'), ghi OK khi chưa đọc thật phải ghi rõ trong báo cáo (charter §7). Đề xuất giảm tải: bắt buộc đọc 33 + 82
tên P6, ~100 giá trị P7 nhãn "sai nghĩa", 87 địa danh, 200 mẫu tên hãng; mục chỉ đổi định dạng/chữ hoa ghi OK nhanh.

**Prompt 51:**
> "Bạn ghi OKK ở phần 2 và Phần 3 hết cho tui nha, tui đọc và OKK hết rồi á"

→ Đã làm (script tạm `fill_anh_ok.py`; không ghi đè ô có sẵn; mỗi dòng ghi chú "Nhật Anh xác nhận OK qua chat
02/10/2026, Claude ghi thay" ở `anh_ghi_chu`/`ghi_chu` để truy vết):
- Phần 2 `anh_duyet` = OK: vai trò 12, P2 5, P6 119, P7 280, địa danh 87, từ chung 25 + 76, thuật ngữ 11.
- Phần 3 `ket_qua` = DUNG: P4 30, P5 30, P5 lẻ 100, P6 30, P7 30, tên hãng 200.
- **Lưu ý cho báo cáo**: các ô này do Claude ghi theo xác nhận của anh trong chat, không phải anh điền trực tiếp.
- File tầng 1 lập lại `P10_tang1_v7_editted.csv` (DS06B được áp làm đổi bản Việt; v6 chưa điền).
- Chạy lại notebook: không lỗi, 246 giây. **Mọi cổng phần 2 và 3 = `dat`**; DS06B áp 1.334 giá trị; P2 áp 5 giá trị
  (hết DS24); DS10 827 → 555. Giữ **400.444/406.871 (98,4%)**. Còn 2 cổng: `P10_tang1` (Claude gắn nhãn 3.996 cặp),
  `P10_tang2_Qwen` (chưa chạy); P13 chờ hai cổng này.

**Prompt 52:**
> "Tầng 2 Qwen (P10_tang2_Qwen): 130.277 cặp ... có nhiều sản phẩm mắc lỗi này không"

→ Trả lời (không sửa code; script tạm `tang2_stats.py`, mẫu lưu ở scratchpad `tang2_mau100.csv`): tầng 2 là cặp CHƯA
kiểm, không phải lỗi đã biết. Phủ: 145.175 giá trị = 21,6% giá trị giữ (34.543 cặp = 8,6%); 90,1% sản phẩm (14.728/16.342)
có ≥ 1 cặp, trung vị 2 cặp/sản phẩm (~9%); food 16,4%, auto 13,7%, beauty 12,3%; chủ yếu giá trị dài ở 颜色/花型/尺寸规格,
mỗi chuỗi ~1 lần. Claude đọc 100 giá trị ngẫu nhiên (không tính tiếng Anh sót): **25/100 sai nghĩa** (KTC 95% ~18–34%),
vd 150豪升 → '150 lít', 冰箱保鲜盒 → 'tủ lạnh giòn', 喷油 → 'phun nhiên liệu', 浪漫粉 → 'Bột lãng mạn', 锅铲 → 'thìa'
→ ước ~36 nghìn giá trị sai ≈ 5% dữ liệu train (ước tính một người chấm). Khuyên chạy Qwen (cờ chỉ dùng để loại khi
độ chính xác ≥ 80% trên 300 cặp hiệu chỉnh; ~2–4 giờ Colab T4); dự phòng: loại hẳn tầng 2 (mất 8,6% cặp, chủ yếu tên
biến thể). Chờ anh đồng ý.

**Prompt 53:**
> Đồng ý chạy Qwen; "Sau khi chạy QWen là có thể đẩy toàn bộ data cleaned lên HF đúng ko:
> https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/tree/main/data/cleaned"

→ Đã làm:
1. **Bộ hiệu chỉnh 300 cặp** `duyet/P10_qwen_hieu_chinh.csv`: 100 cặp đã chấm ở Prompt 52 + 200 cặp ngẫu nhiên mới
   (seed 7, không trùng); Claude chấm nghĩa TRƯỚC khi có nhãn Qwen: 79 SAI / 221 DUNG (đợt 2: 54/200 = 27%).
2. **Pipeline P10**: tự lấy `qwen_nhan` của 300 cặp từ `P10_tang2_ket_qua.csv`; cổng Qwen đạt khi precision ≥ 80% VÀ
   ≥ 20 cờ; in thêm recall. Thông báo thiếu file ghi đúng file còn thiếu.
3. **Notebook Colab** `notebooks/desc-qwen-tang2-colab.ipynb` (13 ô, script tạm `build_qwen_nb.py`): Qwen2.5-7B-Instruct
   nạp 4-bit (bitsandbytes) qua transformers — **đổi so với kế hoạch** (vLLM float16): bản float16 ~15 GB không vừa T4,
   vLLM mới không chắc hỗ trợ T4. Không sinh chữ: đọc xác suất đáp án A/B/C ở token đầu (cố định, có `p_sai` để chỉnh
   ngưỡng). Câu hỏi tiếng Anh, có 4 ví dụ lấy từ từ điển (không nằm trong 300 cặp hiệu chỉnh), dặn không tính tiếng Anh
   sót là lỗi. Chạy thử 500 cặp (in ước tính giờ), chạy hết theo lô 1.000 cặp ghi vào Drive (chạy tiếp được), ô cuối gộp
   `P10_tang2_ket_qua.csv`. Đã kiểm cú pháp; CHƯA chạy thật (máy không có GPU).
4. `CLAUDE.md` 5.3b: ghi cách chạy Qwen mới.
5. HF: thư mục `data/cleaned` đang có `bilingual_zh_vi_cleaned(huy).parquet` của Huy (27 MB) → bản của mình phải đặt tên
   riêng, không ghi đè. Trả lời: chưa đẩy ngay sau Qwen; còn tầng 1, mẫu cuối P13, dựng bản phát hành, anh duyệt.

**Prompt 54:**
> (ảnh chụp Colab) "lỗi gì khi chạy colab vậy" — ô 3 báo `ValueError: mount failed` ở `drive.mount("/content/drive")`
> sau ~2 phút.

→ Giải thích: Colab không kết nối được Google Drive (chưa liên quan dữ liệu/Qwen); thường do hộp thoại cấp quyền bị bỏ
qua/hết giờ, không tick đủ quyền, trình duyệt chặn popup/cookie, hoặc chạy qua extension Colab trong VS Code (không hỗ
trợ mount). Sửa notebook Colab: mount lại với `force_remount=True`, chờ 5 phút; lỗi thì tự chuyển chế độ không Drive
(`files.upload` file vào `/content/zh-vi-desc-qwen`, ô cuối `files.download` kết quả); hướng dẫn mở trên trình duyệt.
Kiểm cú pháp: OK.

**Prompt 55:**
> (ảnh kết quả chạy thử) "500 cặp: 115 giây → ước tính 8.3 giờ"; SAI_NGHIA 280 / DUNG 220; bảng 13 cặp bị cờ.

→ Nhận định: Qwen gắn cờ quá tay (56% so với ~26% ước tính); trong 13 cặp bị cờ chỉ ~6 sai thật, 7 cặp đúng vẫn bị cờ với
p_sai ≥ 0,99 ('Sữa rửa mặt dạng ống đôi', 'Vỏ chăn đơn 200X230cm', 'Rose Pink' dù đã dặn bỏ qua tiếng Anh) → precision
~50% < 80%, chạy hết sẽ trượt cổng; 8,3 giờ quá chậm (câu hỏi ~250 token). Sửa notebook Colab:
1. Câu hỏi ngắn ~80 token, bỏ ví dụ, cỡ lô 32 (ước nhanh hơn 2–3 lần).
2. Ô chạy thử thay bằng ô **chỉnh ngưỡng**: chấm 100 cặp hiệu chỉnh đợt 1 (`mau100_02-10`), in precision/recall theo
   ngưỡng p_sai, chọn ngưỡng nhỏ nhất có precision ≥ 85% (≥ 5 cờ), in ước tính giờ; anh gửi bảng cho Claude trước khi
   chạy hết. Lô lưu xác suất ở `lo_v2/`; ô cuối gắn nhãn theo ngưỡng rồi gộp.
3. Pipeline đo cổng Qwen CHỈ trên 200 cặp đợt 2 (`mau200_03-10`) để không "học tủ".
Kiểm cú pháp: OK; chưa chạy thật.

**Prompt 56:**
> (ảnh) "lỗi" — ô nạp mô hình báo `CUDA out of memory`: tiến trình đã dùng 12,26 GiB, còn trống 2,30 GiB; ô số [20].

→ Nguyên nhân: chạy lại ô nạp mô hình trong phiên cũ, mô hình của lần chạy thử trước còn trên GPU → nạp bản thứ hai.
Cách xử lý ngay: Runtime → Restart session rồi chạy lại từ đầu. Sửa notebook: ô nạp mô hình dùng lại `model` nếu đã có
trong phiên (không nạp thêm), dọn bộ nhớ trước khi nạp. Kiểm cú pháp: OK.

**Prompt 57:**
> (ảnh) `KeyError: False` ở ô chỉnh ngưỡng.

→ **Lỗi code của Claude**: viết `hc.dot` để lấy cột `dot`, nhưng `DataFrame.dot` là hàm nhân ma trận của pandas →
`hc.dot == "..."` ra `False` → lọc lỗi. Pipeline cũng mắc đúng lỗi này (`h.dot`, chưa chạy tới) → sửa cả hai sang
`["dot"]`. Thêm phép thử trên máy (`test_qwen_cells.py`): chạy các ô Colab với Qwen/torch/Drive giả lập trên 3.000 cặp
+ file hiệu chỉnh thật → đọc dữ liệu, chỉnh ngưỡng, chạy lô, gộp đều chạy trọn, file kết quả đúng cột. Bài học: từ nay
chạy thử logic bằng giả lập trước khi giao notebook Colab, không chỉ kiểm cú pháp.

**Prompt 58:**
> (ảnh) ô "chạy hết" báo `AssertionError: Chưa có ngưỡng đạt yêu cầu`.

→ Giải thích: chốt an toàn hoạt động đúng thiết kế — hoặc ô chỉnh ngưỡng không tìm được ngưỡng có precision ≥ 85% trên
100 cặp (Qwen vẫn cờ nhầm nhiều), hoặc ô đó chưa chạy/lỗi trong phiên. Xin ảnh output ô chỉnh ngưỡng (ước tính giờ,
bảng ngưỡng, ngưỡng chọn, 25 cặp theo p_sai) để quyết: chỉnh câu hỏi thêm một lần nếu gần đạt; nếu tách biệt kém ở mọi
ngưỡng thì bỏ Qwen, chọn dự phòng (loại hẳn tầng 2 khỏi train, hoặc giữ với nhãn `chua_kiem_nghia`).

**Prompt 59:**
> (ảnh bảng ngưỡng) 100 cặp: 13 giây → ước 4,8 giờ; precision 0,284–0,322 ở mọi ngưỡng 0,5–0,999 (59–75 cờ / 100 cặp,
> recall 0,76–0,88); Ngưỡng chọn: None.

→ Kết luận: Qwen2.5-7B (4-bit, câu hỏi ngắn) gần như không phân biệt được: precision ≈ tỉ lệ sai nền 25% của 100 cặp,
xác suất bão hoà → đề nghị dừng Qwen (chỉnh câu hỏi khó cứu; 14B lâu gấp đôi, không chắc tốt hơn). Đưa 3 hướng:
**A (khuyên)** gắn mã `T2_CHUA_KIEM` cho cặp có giá trị tầng 2 → không vào train, vẫn giữ trong file (mất ~34.500 cặp
= 8,6%, chủ yếu tên biến thể); B giữ với nhãn `chua_kiem_nghia` (~5% train sai); C Claude Haiku qua API (~12 USD với
Batch, ~23 USD không Batch; anh từng chọn không dùng API trả phí). Chờ anh chọn.

**Prompt 60:**
> "Vậy làm phần A nha"

→ Đã làm:
1. Pipeline P10: cờ `DUNG_QWEN = False` ở P0 (+ dòng `tang2` trong `QUYET_DINH`); cặp có giá trị tầng 2 gắn mã
   `T2_CHUA_KIEM` (không vào train, vẫn giữ trong file); cổng `P10_tang2_Qwen` = `khong_can`. Nhánh Qwen giữ lại sau cờ
   (bật lại được); vẫn xuất `P10_tang2_cho_qwen.csv` để chấm lại sau.
2. Chạy lại: không lỗi, 271 giây. **36.249 cặp** `T2_CHUA_KIEM`; giữ **365.947/406.871 (89,9%)**; loại theo category:
   food 22,2%, auto 14,9%, beauty 14,1%, home 14,1%, electronics 11,0%, mother_baby 8,6%, bags 7,7%, fashion 6,3%,
   shoes 5,1%. Mọi cổng đạt trừ `P10_tang1` (Claude). Hệ quả phụ: description dựng lại ngắn hơn nên trùng y hệt 72 → 165
   sản phẩm, nhóm `nhom_trung` lớn nhất 283 → 629 sản phẩm (3,85%) — cần để ý khi chia tập.
3. Đo: trong 34.497 cặp chỉ có mã T2, 19.780 cặp một giá trị, 10.376 cặp nhiều giá trị toàn hiếm, 4.341 cặp chỉ một
   phần hiếm (chủ yếu 颜色) → food cao là do chính giá trị dài hiếm; lựa chọn để sau: cắt riêng giá trị hiếm khỏi danh sách
   để cứu 4.341 cặp (21.825 giá trị). Chưa làm.
4. `CLAUDE.md` 5.3b ghi quyết định.

**Prompt 61:**
> "Lúc nào bạn xong"

→ Ước lượng: tầng 1 có 3.996 cặp, em đọc theo đợt 250–350 cặp, làm liên tục trong lúc trả lời (không chạy ngầm được).
Đã làm hết trong Prompt 61–62:
1. Gắn nhãn tầng 1 (`duyet/P10_tang1_v7_editted.csv`): **DUNG 2.837, SAI 1.144 (cặp SAI nào cũng có `claude_sua`),
   KHONG_CHAC 15**. Thêm 10 cặp mới sinh ra sau khi sửa luật (file `P10_tang1_v7_editted_bo_sung.csv`): 7 SAI, 3 DUNG (Prompt 63 sửa lại, trước ghi nhầm 6 SAI)
   (cột `claude_nhan` đã điền đủ). Nguyên tắc khi gắn: chữ Anh/pinyin chưa dịch không tính là lỗi; "Apple N"/"Táo N" →
   "iPhone N"; size ghi 斤 thì chia đôi ra kg; còn sót chữ Hán thì SAI và dịch lại; chữ đầu viết thường không tính lỗi;
   bản dịch lệch dòng (vi là của sản phẩm khác) thì SAI và dịch lại từ zh; tên họa tiết dịch sang tiếng Anh thì dịch
   lại sang tiếng Việt.
2. 15 cặp KHONG_CHAC (vẫn vào train với nhãn `chua_kiem_nghia`, không dùng làm test): PP裤 (2 lần), 人字型 (2), 杀手包,
   五环黑, L可拆16空盒, 华棉, 米兰 (màu), 仙草 (họa tiết), 药用层孔菌, 乳贴面膜, 奶油-无刷桶款, 蝴网RiFit款, 蝴网连杆.
3. Sửa luật P3: regex `TEN_NSX` thêm 生产厂商, 供货商, 工厂 (trước đó 29 cặp tên công ty lọt vào train).
   `PII_NSX` tăng 1.478 → **1.567 cặp**.
4. Bật `CLAUDE_XONG["P10_tang1"] = True`, dựng lại và chạy pipeline (lần đầu MemoryError ở ô cuối do RAM máy còn
   3,3 GB, chạy lại thì qua; 2 lần chạy sau đều không lỗi). Tầng 1 sửa **7.088 giá trị**. Giữ **366.016/406.871 cặp
   (90,0%)**; 16.342 sản phẩm có description sạch. Mọi cổng đạt, trừ 2 cổng mẫu tầng 1 đang chờ anh.
5. File mẫu cho anh: `duyet/P10_tang1_SAI_mau_v1.csv` (30 cặp em gắn SAI, xem `claude_sua` có đúng hơn bản cũ không)
   và `duyet/P10_tang1_DUNG_mau_v1.csv` (30 cặp em gắn DUNG). Anh điền cột `ket_qua` (OK hoặc ghi chỗ sai). Hai cổng
   này đạt thì P13 tự bốc 200 mẫu cuối.

**Prompt 62:**
> "Bạn hãy tiếp tục tiến trình đi"

→ Làm tiếp các đợt 2700–3995 và các bước 3–5 của Prompt 61 (số liệu ghi ở trên).

**Prompt 63:**
> "Bạn đã gán những những gì vậy"

→ Tổng kết nhãn tầng 1 (file chính + bổ sung, 4.006 cặp): DUNG 2.840, SAI 1.151, KHONG_CHAC 15. Đếm theo nhóm lỗi
(một cặp có thể thuộc nhiều nhóm): bản vi còn chữ Hán/lệch dòng 136; Táo/Vinh quang → iPhone/Honor 81; tên trà 63;
size 斤 → kg 42; vali mẹ con/vali đơn 39; mAh và camera hành trình 32; chăn ga 28; dây màu (…带) 19. Theo thuộc tính
nhiều nhất: 颜色 239, 花型 94, 适用型号 83, 包装规格 59. Phát hiện ghi nhầm "6 SAI" ở file bổ sung (đúng là 7), đã sửa.

**Prompt 64:**
> "Tiến trình tới đâu ròi, tiếp theo tui cần phải làm những gì"

→ Kiểm: 2 file mẫu tầng 1 (`P10_tang1_SAI_mau_v1.csv`, `P10_tang1_DUNG_mau_v1.csv`) còn 0/30 dòng có `ket_qua`.
Báo tiến độ: mọi bước pipeline xong, chỉ chờ 2 cổng mẫu tầng 1 → P13 (200 mẫu cuối) → đóng gói → hỏi trước khi
đẩy HF. Việc của anh: (1) kiểm 60 mẫu tầng 1; (2) kiểm 200 mẫu cuối P13; (3) đồng ý đẩy HF (tên file riêng); tuỳ
chọn: xoá tay các file phiên bản cũ, quyết có cứu 4.341 cặp T2 lẫn hay không.

**Prompt 65:**
> "1. Kiểm 60 mẫu tầng 1 ..., Ok hết rồi á, bạn điền giúp tui"

→ Đã làm:
1. Ghi `ket_qua = OK` cho 30 + 30 dòng ở `P10_tang1_SAI_mau_v1.csv` và `P10_tang1_DUNG_mau_v1.csv`, `ghi_chu` = "Nhật Anh
   xác nhận OK qua chat 03/10/2026, Claude ghi thay" (script kiểm trước là cột còn trống, không ghi đè).
2. Chạy lại pipeline: 2 cổng tầng 1 đạt → P13 bốc 200 sản phẩm vào `duyet/P13_mau_cuoi_mau_v3.csv`.
3. Phát hiện và sửa lỗi pipeline: biến `chinh_thuc` được tính TRƯỚC khi thêm cổng P13 nên notebook báo "Dữ liệu là
   CHÍNH THỨC" dù P13 còn 0/200. Đã tính lại sau cổng P13; chạy lại: "CHƯA chính thức: còn 1 cổng" (đúng).
4. Quét tự động 200 mẫu P13: 16 mẫu còn chữ Hán đều là số giấy phép mỹ phẩm (粤妆…, 粤G妆网备字…) giữ nguyên theo luật mã
   thật; 14 mẫu có dãy giống số điện thoại 11 số, xem từng cái: đều là mã giấy phép SC…, số chứng nhận 3C, mã sản phẩm,
   không phải số điện thoại. Không rỗng, không còn 【】.
5. Việc của anh: đọc 200 mẫu P13 (so `description_vi_goc` với `description_vi_sach_cat`, chỉ cần tiếng Việt), điền
   `ket_qua`. Sai ≤ 10 thì đạt.

**Prompt 66:**
> "Bạn xử lý description_vi_sach_cat bằng những phương pháp nào vậy"

→ Giải thích (không đổi code): `description_vi_sach` = ghép lại các cặp "tên: giá trị" còn `dung_cho_train = 1` theo
thứ tự gốc, nối bằng "; "; `_cat` chỉ là 600 ký tự đầu để đọc mẫu. Các cặp đi qua P1–P12: tách/ghép cặp, làm sạch ký
tự, lọc thông tin cá nhân, vai trò + giá trị rỗng nghĩa, gỡ số đuôi, từ điển tên (P6), từ điển giá trị + luật ngữ cảnh
+ địa danh + tên hãng + thuật ngữ (P7), số và đơn vị (P8), định dạng (P9), kiểm nghĩa đuôi dài (P10), loại có mã lý do
(P11), khử trùng lặp (P12). Phương pháp: luật + từ điển (anh duyệt) + Claude đọc tầng 1; không dùng mô hình dịch lại.

**Prompt 67:**
> "Tui đã đọc ròi, còn một số dòng bên description_vi_sach_cat còn chữ Hán, giải thích"

→ Liệt kê toàn bộ chữ Hán trong 200 mẫu P13: 16 dòng, tất cả là mã cơ quan cấp, giữ nguyên theo luật "mã thật" (P8):
粤妆2017… (số giấy phép sản xuất mỹ phẩm), 粤G妆网备字2025… (số công bố mỹ phẩm thông thường), 粤XK16-204-… (giấy phép
sản xuất công nghiệp). Không có chữ Hán mô tả nào sót. Giải thích lý do giữ nguyên; đưa 2 hướng khác nếu anh muốn
(loại các cặp này khỏi train, hoặc thêm chú thích Việt sau mã). Chưa ghi `ket_qua` P13: chờ anh xác nhận ngoài chuyện
này 200 mẫu có OK không.

**Prompt 68:**
> "tất cả 200 mẫu P13 tui thấy OK hết rồi á, điền giúp tui nha. Và việc một số dòng chữ Hán thì Thêm chú thích tiếng
> Việt sau mã, ví dụ "粤妆20170110 (Quảng Đông)""

→ Đã làm:
1. Ghi `ket_qua = OK` cho 200 dòng `duyet/P13_mau_cuoi_mau_v3.csv`, `ghi_chu` = "Nhật Anh xác nhận OK cả 200 mẫu qua chat
   03/10/2026, Claude ghi thay".
2. Thống kê mã còn chữ Hán trên toàn bộ dữ liệu: 2.479 giá trị, 51 dạng (粤G妆网备字 1.009, 粤妆 869, 粤 167, 国妆特字 134,
   沪…, 浙…, 国妆网备进字（沪）…, 湘械注准…, XK16 …), cộng 1 giá trị ngày "2023年05月24日" lọt vào nhóm mã.
3. P8 thêm `chu_thich_ma`: mã còn chữ Hán → thêm "(tỉnh)" theo chữ viết tắt tỉnh (bảng 31 tỉnh/thành), mã bắt đầu
   bằng 国 thêm "quốc gia" (+ tỉnh trong ngoặc nếu có). Ví dụ "粤妆20161271 (Quảng Đông)", "国妆特字20241670 (quốc gia)",
   "国妆网备进字（京）2023000197 (quốc gia, Bắc Kinh)". Ngày → "24/05/2023". Kết quả: 2.478/2.478 mã có chú thích,
   0 mã không chú thích được. Bản Trung không đổi.
4. Chạy lại pipeline: không lỗi. Cổng P13 đạt (200 mẫu, 0 sai) → **mọi cổng đạt, dữ liệu CHÍNH THỨC**. Giữ 366.016/406.871
   cặp (90,0%), 16.342 sản phẩm có description sạch. Lưu ý: 16 dòng trong file mẫu P13 vẫn hiện mã chưa có chú thích (mẫu
   bốc trước khi thêm chú thích), dữ liệu thật đã có.
5. `CLAUDE.md` 5.3b ghi quyết định.
6. Bước tiếp: cập nhật trang pipeline (V5), đóng gói bản công bố, hỏi anh trước khi đẩy HF.

**Prompt 69:**
> "Okk bạn hãy làm đi nhé, tiện cho tui hỏi Sau khi đưa qua pipeline làm sạch đã thông nhất với nhau, thì data
> description còn bao nhiêu dòng vậy"

→ Trả lời: đầu vào 16.348 sản phẩm / 406.871 cặp → còn **16.342 sản phẩm** có description sạch (6 bị loại DS02), trong
đó 165 trùng y hệt (còn 16.177 không trùng); **366.016 cặp** dùng cho train (90,0%); trung bình 22,4 cặp/sản phẩm,
description_vi sạch dài trung bình 788 ký tự. Theo category: bags 2.878, mother_baby 2.495, home 2.493, shoes 2.310,
food 1.420, fashion 1.350, beauty 1.317, electronics 1.108, auto 971.

Đã làm:
1. **Đóng gói bản công bố** (script `build_release.py` trong scratchpad, chỉ đọc `data/processed/desc`):
   `data/release/desc_v1/`
   - `bilingual_zh_vi_desc_cleaned(nhatanh).parquet`: 16.342 sản phẩm, cột product_id, category, description_zh_sach,
     description_vi_sach, so_cap_giu, so_cap_loai, trung_y_het, nhom_trung (bỏ shop, title_zh). 3,0 MB.
   - `bilingual_zh_vi_desc_pairs(nhatanh).parquet`: 405.275 cặp (bỏ hẳn 1.596 cặp PII/PII_NSX; bỏ cột bản gốc
     name_vi_goc/gia_tri_vi_goc vì có thể còn thông tin chưa che). 7,2 MB.
   - `README.md` (data card: cột, mã lý do, nhãn, các bước, lưu ý chia theo nhom_trung), `SHA256SUMS.txt`.
   - Quét thông tin cá nhân lần cuối: 0 số điện thoại, 0 email, 0 URL; 4 chỗ "weixin/微信" đều là tên sàn bán hàng
     ("Cửa hàng WeChat") hoặc tên hãng (Ruiweixing), không phải ID liên hệ.
2. **Trang pipeline Version 5** (`docs/design/description_pipeline.html`, cùng link artifact cũ): dữ liệu chính thức,
   số liệu 03/10, phương án A thay Qwen, tắt tiếng Anh, PII_NSX, chú thích mã, kết quả tầng 1, bảng cổng đều đạt,
   bản công bố, mục "Còn mở" mới (đẩy HF, chia tập, cứu 4.341 cặp, kiểm lại tầng 2).
3. **Chưa đẩy HF**: chờ anh đồng ý.

**Prompt 70:**
> "Hãy cho tui biết chi tiết tình trạng của 16.342 có description sạch ( tức là data đã cleaned)"

→ Đo trên bản công bố (script `tinh_trang_16342.py` trong scratchpad, chỉ đọc):
- Quy mô: 16.342 sp; 366.016 cặp giữ, 39.259 cặp loại. Sp không mất cặp nào 1.362 (8,3%); giữ ≥ 90% cặp: 9.021
  (55,2%); 70–90%: 6.767 (41,4%); 50–70%: 506 (3,1%); < 50%: 48 (0,3%). Cặp giữ/sp: trung vị 19 (10%: 13, 90%: 37);
  34 sp còn ≤ 5 cặp. Độ dài description_vi: trung vị 704 ký tự (424–1.277); zh trung vị 260.
- Theo category (sp, % cặp giữ): bags 2.878 92,4%; mother_baby 2.495 91,4%; home 2.493 85,9%; shoes 2.310 94,9%;
  food 1.420 77,8%; fashion 1.350 93,7%; beauty 1.317 85,9%; electronics 1.108 88,9%; auto 971 85,1%.
- Lý do mất cặp: T2_CHUA_KIEM 36.184 (90,6% sp có ít nhất 1 cặp T2), DS07 2.762, RONG 1.369, DS10 554, DS17 110,
  DS09_dai 74, DS03 62, DS18 3, DS25 2.
- Cặp giữ theo vai trò: mô tả hàng 66,9%, giao dịch 1688 21,6%, mã/giấy tờ 5,0%, thương hiệu 4,2%, lựa chọn SKU 2,3%.
  Nhãn: it_thong_tin 5,2%, chep_ma 4,5%, chua_kiem_nghia 47 cặp (43 sp), vai_tro_tam 98,5%.
- Trùng: 165 trung_y_het; 6.566 nhóm (4.121 nhóm 1 sp; 221 nhóm ≥ 10 sp chứa 5.028 sp); nhóm lớn nhất 629 sp
  (home 334, mother_baby 135, electronics 72...).
- Còn tồn: 9.883 cặp giữ (2,7%) có giá trị Việt toàn chữ Latin không dấu (ước lượng trên, gồm cả từ hợp lệ như PPSU,
  Polyester, inch); 2.476 cặp còn chữ Hán đều là mã có chú thích; 67 cặp có nhãn che [SĐT]/[URL]/[LIÊN HỆ].

**Prompt 71:**
> "Sau khi có data description sạch rồi, Bạn có đề xuất cách xử lý tiếp theo gì không, ví dụ tính Cosin theo mô hình
> bg-m3 ... như phần data Title Huy đã up lên HF ... Bạn hãy đề xuất tui các phương pháp tiếp theo"

→ Chỉ đề xuất, chưa làm gì. Điểm chính: có sẵn 4.006 cặp tầng 1 (DUNG/SAI) + 300 cặp tầng 2 đã gắn nhãn → đo trước
được bộ lọc nào thật sự bắt lỗi (AUC, precision ở từng ngưỡng) thay vì chọn ngưỡng cảm tính. Đề xuất theo thứ tự:
(1) đo bge-m3/LaBSE cosine và CometKiwi trên tập đã gắn nhãn; (2) nếu đạt, dùng để cứu một phần 36.184 cặp T2 và gắn
điểm chất lượng cho 366.016 cặp (cột điểm, không xoá); (3) giới hạn tần suất câu mẫu lặp (giao dịch 1688 21,6%);
(4) chia train/dev/test theo nhom_trung + category, test do người kiểm (charter §7); (5) xuất bảng thuật ngữ từ
P6/P7 cho fine-tune và đo độ chính xác thuật ngữ; (6) phối hợp Quang phần tiếng Anh/pinyin; (7) cắt description thành
đoạn huấn luyện vừa độ dài mô hình. Chờ anh chọn.

**Prompt 72:**
> "Khi tui chưa yêu cầu chia thì chưa chia nha, tui chỉ đang xử lý phần data, chia tính sau, hãy ghi nhớ điều đó.
> Okk, vậy hãy làm bước 1 với bge-m3 trên máy tui trước"

→ Đã ghi nhớ (memory): không chia train/dev/test khi anh chưa yêu cầu; giai đoạn này chỉ xử lý data.
Bước 1 (đo trước bge-m3):
1. Viết `scripts/eda/desc_bge_do_truoc.py`: lấy 3.991 cặp tầng 1 (DUNG 2.840, SAI 1.151; bỏ 15 KHONG_CHAC) và 300 cặp
   tầng 2 (DUNG 221, SAI 79); tính cosine bge-m3 (Ollama, cắt 480 ký tự) theo 2 cách: chỉ giá trị, và "tên: giá trị";
   đo AUC, bảng ngưỡng (precision/recall khi gắn cờ, tỉ lệ sai còn lại trong phần giữ), mốc cặp lệch cùng thuộc tính,
   ví dụ bỏ lọt/cờ nhầm. Ghi vào `data/eda/description/bge_do_truoc/` (file mới).
2. Chạy thử với vector giả để bắt lỗi code: chạy hết, không lỗi (số liệu giả, không dùng). Thư mục thử
   `data/eda/description/_thu_gia/` nằm lại (em không xoá được file), anh xoá tay giúp.
3. Chưa chạy thật: Ollama đang chạy nhưng không thấy model (`/api/tags` 0 model, `/api/embed` 404), dù bge-m3 vẫn có
   trong `D:\ollama-models` và biến OLLAMA_MODELS (User) đúng → giống lỗi 01/10. Nhờ anh thoát Ollama ở khay hệ thống
   rồi mở lại; em không tắt tiến trình của anh.

**Prompt 73:**
> "mở rồi"

→ Ollama vẫn 0 model; tiến trình đang giữ cổng 11434 mở từ 9:08 (chưa khởi động lại). Không đụng tiến trình của anh:
em mở một Ollama riêng ở cổng 11435 với `OLLAMA_MODELS=D:\ollama-models` (tác vụ nền), script đọc `OLLAMA_URL`; chạy xong
đã tắt tác vụ đó (chỉ còn Ollama của anh). Ghi cách này vào `CLAUDE.md`.

**Kết quả đo bge-m3** (`data/eda/description/bge_do_truoc/`, CPU/Ollama, cắt 480 ký tự, 15.386 chuỗi, ~8 phút):
- AUC (0,5 = đoán mò): tầng 1 chỉ giá trị 0,580, "tên: giá trị" 0,626; **tầng 2 0,574 / 0,570**.
- Cosine trung bình DUNG 0,71 vs SAI 0,67 (chỉ giá trị); cặp ghép lệch cùng thuộc tính (chắc chắn sai) trung vị 0,46, p95 0,78.
- Tầng 1, ngưỡng 0,50: cờ 39 cặp, 38 sai thật (97%) nhưng chỉ bắt 3,3% số cặp sai; ngưỡng 0,70: precision 47%, recall 37%.
- **Tầng 2 (quyết định cứu cặp T2):** ở mọi ngưỡng precision 27–38%, gần tỉ lệ sai nền 26%; phần giữ lại vẫn sai
  18–26% → không cứu được tầng 2 bằng bge-m3.
- Vì sao: giá trị thuộc tính quá ngắn; bản Việt chưa dịch (giống hệt bản Trung) cho cosine 1,0; phiên âm/thuật ngữ
  đúng (寿眉 → Thọ Mi, 浮世绘 → Ukiyo-e, 熊果苷 → Arbutin) cho cosine 0,27–0,29; lỗi nhỏ (米色 → "m màu", 字 → "từ")
  vẫn 0,87–0,89.
- Kết luận: không dùng bge-m3 cosine để lọc hay cứu cặp description. Hướng tiếp (chờ anh chọn): thử CometKiwi trên
  Colab bằng cùng bộ đo, hoặc giữ phương án A.

**Prompt 74:**
> "Vậy hãy làm theo hướng 1 cho tui nhé"

→ Đã làm (chờ anh chạy Colab):
1. File đầu vào `data/eda/description/kiwi_do_truoc/kiwi_dau_vao.csv`: 4.291 cặp đã gắn nhãn (tầng 1 DUNG 2.840 /
   SAI 1.151; tầng 2 DUNG 221 / SAI 79), cột tap, ten_thuoc_tinh, gia_tri_zh, gia_tri_vi, zh_ten, vi_ten, nhan (678 KB).
2. Notebook `notebooks/desc-cometkiwi-do-truoc-colab.ipynb` (11 ô, Colab T4): cài `unbabel-comet`, đăng nhập Hugging
   Face (Colab Secret `HF_TOKEN` hoặc dán token), tải file đầu vào, chấm `Unbabel/wmt22-cometkiwi-da` theo 2 cách (chỉ
   giá trị; "tên: giá trị"), in AUC nhanh, tự tải về `kiwi_ket_qua.csv`. Mô hình bị khoá: anh phải bấm "Agree and access
   repository" trên trang mô hình và tạo token Read. Giấy phép CC BY-NC-SA 4.0 (nghiên cứu, không thương mại).
3. Chạy thử toàn bộ ô trên máy với Colab/comet/torch giả: không lỗi (số giả, không dùng).
4. Việc của anh: chạy notebook, chép `kiwi_ket_qua.csv` vào `data/eda/description/kiwi_do_truoc/`, báo em để phân
   tích đầy đủ (AUC, bảng ngưỡng, ví dụ, so với bge-m3).

**Prompt 75:**
> (ảnh trang huggingface.co/Unbabel/wmt22-cometkiwi-da, khung "Acknowledge license to accept the repository")

→ Hướng dẫn: nút trên trang hiện tên là **Acknowledge license** (không phải "Agree and access repository" như em ghi);
bấm nút đó là đồng ý điều khoản, Hugging Face chia sẻ email + username của anh cho Unbabel. Sau đó tạo token Read
(Settings → Access Tokens → Create new token → Read) và chạy notebook.

**Prompt 76:**
> (ảnh trang "Create new Access Token", Token type đang chọn Fine-grained, tài khoản HF trinhnhatanh)

→ Hướng dẫn: ở "Token type" bấm sang **Read** (đang để Fine-grained), đặt tên (vd `colab-kiwi`), bấm Create token,
chép token ngay (chỉ hiện một lần), không gửi token cho Claude.

**Prompt 77:**
> (ảnh hộp thoại "Save your Access Token": token `colab-kiwi`, quyền READ; anh dán cả chuỗi token vào chat)

→ Token đã lộ trong lịch sử chat (em KHÔNG ghi token vào file nào). Khuyên: dùng token này cho lần chạy Colab, chạy
xong vào Settings → Access Tokens xoá (Revoke) `colab-kiwi`; lần sau cần thì tạo token mới. Rủi ro thấp vì chỉ quyền
Read, nhưng vẫn nên xoá. Bước tiếp: chạy notebook CometKiwi trên Colab.

**Prompt 78:**
> (ảnh ô `pip install unbabel-comet` trên Colab: "ERROR: pip's dependency resolver ..." liệt kê opencv, jax, tensorflow,
> gradio, shap... cần numpy ≥ 2 / protobuf ≥ 5.28, trong khi đã cài numpy 1.26.4, protobuf 4.25.9, huggingface-hub 0.36.2)

→ Giải thích: đây là cảnh báo xung đột của pip, không phải cài thất bại; unbabel-comet cần numpy 1.x nên pip hạ numpy,
các gói báo xung đột (opencv, jax, tensorflow, gradio...) notebook không dùng. Việc cần làm: Runtime → Restart session,
rồi chạy từ ô kiểm tra (ô thứ 3, in phiên bản), không chạy lại ô cài. Nếu ô kiểm tra lỗi numpy (Colab đang Python 3.13,
numpy 1.26.4 phải tự build) thì gửi ảnh để em đổi cách cài.

**Prompt 79:**
> (ảnh ô kiểm tra `import comet, torch`: ImportError: cannot import name 'runtime_version' from 'google.protobuf',
> lỗi nằm trong tensorflow/core/framework/attr_value_pb2.py)

→ Nguyên nhân: cài unbabel-comet hạ protobuf xuống 4.25.9; khi import comet, pytorch-lightning/torchmetrics kéo theo
tensorflow có sẵn trên Colab, mà tensorflow 2.20 cần protobuf ≥ 5.28. numpy 1.26.4 trên Python 3.13 không phải
nguyên nhân. Sửa: cài lại `protobuf>=5.28.3,<6` sau unbabel-comet, Restart session, chạy lại ô kiểm tra. Đã sửa ô cài
trong `notebooks/desc-cometkiwi-do-truoc-colab.ipynb` (thêm dòng cài protobuf và ghi chú).

**Prompt 80:**
> (ảnh: ô `pip install "protobuf>=5.28.3,<6"` đã cài protobuf 5.29.6, chỉ cảnh báo unbabel-comet muốn protobuf < 5;
> ô `import comet, torch` vẫn hiện ImportError cũ, cùng tiến trình ipykernel_10014)

→ Protobuf đã cài đúng; lỗi còn hiện là do phiên Python chưa khởi động lại (cùng số tiến trình 10014, protobuf cũ vẫn
nằm trong bộ nhớ). Việc cần làm: Runtime → Restart session rồi chạy lại ô kiểm tra. Cảnh báo "unbabel-comet requires
protobuf<5" bỏ qua được (mô hình không dùng protobuf khi chấm). Dự phòng nếu vẫn lỗi: `!pip -q uninstall -y
tensorflow` (notebook không dùng tensorflow) rồi Restart session.

**Prompt 81:**
> (ảnh: sau Restart session, tiến trình mới ipykernel_11774, `import comet` lỗi mới ở
> torchmetrics/functional/text/helper_embedding_metric.py dòng `from transformers import AutoModelForMaskedLM, ...`)

→ Lỗi protobuf đã hết; lỗi mới là import transformers. Nguyên nhân (suy từ ảnh đầu: "diffusers requires
huggingface-hub>=1.23 but you have 0.36.2"): unbabel-comet hạ huggingface-hub xuống 0.36.2, không hợp với transformers
5.x có sẵn trên Colab. Sửa: `!pip -q install "transformers>=4.45,<5"` → Restart session → chạy lại ô kiểm tra. Đã cập
nhật ô cài trong notebook (thêm transformers 4.x cùng dòng protobuf). Nếu vẫn lỗi: cuộn xuống chụp dòng lỗi cuối cùng.

**Prompt 82:**
> (ảnh: ô `!pip -q install "transformers>=4.45,<5"` còn ký hiệu `[ ]`, không có output; ô kiểm tra lỗi
> `ImportError: cannot import name 'PreTrainedModel' from 'transformers'`)

→ Ô cài transformers chưa được chạy (ô có `[ ]` trống, không output); lỗi đúng như dự đoán (transformers 5.x không
nạp được với huggingface-hub 0.36). Hướng dẫn: bấm nút chạy ở ô cài, đợi xong (hiện số trong ngoặc), Restart session,
chạy lại ô kiểm tra.

**Prompt 83:**
> (ảnh: tiến trình mới ipykernel_13546, vẫn `ImportError: cannot import name 'PreTrainedModel' from 'transformers'`)

→ Đổi cách: vá từng gói trên Python có sẵn của Colab (Python 3.13, transformers 5.x, tensorflow, numpy 2) không ổn định.
Viết lại `notebooks/desc-cometkiwi-do-truoc-colab.ipynb` (bản 2, 10 ô): cài `uv`, tạo môi trường riêng Python 3.11 ở
`/content/kiwi_env`, cài `unbabel-comet` + pandas vào đó; Python của Colab chỉ đăng nhập HF (token lưu ở
~/.cache/huggingface, môi trường riêng đọc lại), tải file, xem AUC, tải kết quả; chấm điểm chạy bằng script
`/content/cham_kiwi.py` trong môi trường riêng. Không cần Restart session. Chạy thử script chấm + ô AUC trên máy với
comet giả: không lỗi. Việc của anh: Runtime → Disconnect and delete runtime, mở notebook bản 2 (tải lại file từ máy),
chọn T4, chạy lần lượt.

**Prompt 84:**
> (ảnh ô đăng nhập: huggingface_hub mới của Colab hiện "To log in, open this URL and enter the code:
> https://hf.co/oauth/device", mã thiết bị, "Waiting for authorization...")

→ Đăng nhập kiểu mới (mã thiết bị) thay cho ô dán token. Hướng dẫn: mở hf.co/oauth/device, đăng nhập tài khoản HF đã
Acknowledge license, nhập mã đang hiện, bấm cho phép; ô tự chạy xong. Dự phòng: dừng ô, chạy `login(token="...")` dán
token Read.

**Prompt 85:**
> (ảnh trang huggingface.co/oauth/device "Authorize Device", 8 ô nhập mã, nút Continue)

→ Hướng dẫn: gõ 8 ký tự mã đang hiện trong ô Colab (bỏ dấu gạch), bấm Continue, trang sau bấm cho phép (Authorize);
quay lại Colab chạy tiếp ô tải dữ liệu. Mã hết hạn thì chạy lại ô đăng nhập lấy mã mới.

**Prompt 86:**
> (ảnh ô "Xem nhanh": FileNotFoundError /content/kiwi_ket_qua.csv)

→ Ô xem nhanh chỉ đọc kết quả; file không có nghĩa là ô chấm điểm (`!/content/kiwi_env/bin/python /content/cham_kiwi.py`)
chưa chạy hoặc đã lỗi (lệnh `!` lỗi không làm ô báo đỏ). Nhờ anh xem/chụp phần chữ dưới ô chấm điểm (và chắc ô
`%%writefile` đã chạy). Nghi lỗi nhiều nhất: script trong môi trường riêng không dùng được phiên đăng nhập kiểu mã thiết
bị → tải mô hình bị từ chối (401/403/GatedRepo). Dự phòng: chạy `!HF_TOKEN=hf_... /content/kiwi_env/bin/python
/content/cham_kiwi.py` với token Read.

**Prompt 87:**
> (ảnh ô `%%writefile /content/cham_kiwi.py`, output "Writing /content/cham_kiwi.py")

→ Ô ghi script đã chạy đúng. Cần xem tiếp ô ngay dưới (`!/content/kiwi_env/bin/python /content/cham_kiwi.py`): chạy
ô đó (nếu chưa) và chụp phần chữ bên dưới.

**Prompt 88:**
> (ảnh ô chấm điểm trong môi trường riêng Python 3.11: Traceback khi `import comet` → pytorch_lightning → torchmetrics
> → torchmetrics/functional/audio/pit.py; ảnh bị cắt trước dòng lỗi cuối)

→ Môi trường riêng đã tách khỏi gói Colab (đường dẫn kiwi_env/python3.11), nhưng bộ phiên bản uv chọn (torch /
torchmetrics / pytorch-lightning) không khớp nhau. Chưa thấy dòng lỗi cuối nên chưa kết luận; nhờ anh chạy 1 ô chẩn
đoán in 6 dòng lỗi cuối + phiên bản các gói (torch, lightning, torchmetrics, numpy, scipy, comet, transformers).

**Prompt 89:**
> (ảnh ô chẩn đoán: `ModuleNotFoundError: No module named 'pkg_resources'` ở torchmetrics/utilities/imports.py; phiên
> bản: numpy 1.26.4, pytorch-lightning 2.6.6, scipy 1.17.1, torch 2.14.1, torchmetrics 0.10.3, transformers 4.57.6,
> unbabel-comet 2.2.7)

→ Nguyên nhân: môi trường uv không có setuptools (pkg_resources), torchmetrics 0.10.3 cần nó. Sửa trên Colab:
`!uv pip install -q --python /content/kiwi_env/bin/python "setuptools<81"` rồi chạy lại ô chấm điểm. Đã cập nhật ô cài
trong notebook bản 2 (thêm `"setuptools<81"`), chạy lại thử trên máy với comet giả: không lỗi.

**Prompt 90:**
> "tui đã làm xong, bạn kiểm tra nhé"

→ Có `data/eda/description/kiwi_do_truoc/kiwi_ket_qua.csv` (4.291 dòng, khớp từng cặp với file bge). Viết
`scripts/eda/desc_kiwi_phan_tich.py` (chỉ đọc; ghi `tom_tat_auc_kiwi_vs_bge.csv`, `bang_giu_tang1/2.csv`,
`cap_da_gan_nhan_kiwi_bge.csv`). Kết quả (CometKiwi chạy trên Colab T4, thật; bge-m3 chạy Ollama CPU):
- AUC: CometKiwi "tên: giá trị" tầng 1 **0,733**, tầng 2 **0,724** (chỉ giá trị 0,692 / 0,705); bge-m3 0,57–0,63;
  kết hợp hai cái không hơn (0,70 / 0,67).
- Tầng 2 (tỉ lệ sai nền 26,3%), giữ x% cặp điểm cao nhất: 50% → còn sai 14,0%; 30% → 10,0% (90 cặp); 20% → 8,3%
  (60 cặp, 5 sai; mẫu nhỏ nên khoảng tin cậy rộng, khoảng 3–18%).
- Tầng 1: đáy 2% điểm thấp nhất (79 cặp) có 87% là SAI; đáy 5% 82%; đáy 10% 71%.
- CometKiwi bỏ lọt: size 斤 để nguyên số thành kg (0,85–0,86), 苹果6S → "Apple 6S", 老年 → "tuổi già"; cờ nhầm thuật
  ngữ đúng nhưng ngắn (灯芯绒 → Nhung kẻ, 魔术贴 → Băng dính gai, 0,25–0,28).
- Kiểm thêm dữ liệu công bố: 50 cặp giữ có 斤, đọc tay đều đã quy đổi đúng (regex nghi 21 cặp là báo nhầm).
- Kết luận đề xuất: CometKiwi tốt hơn bge-m3 rõ nhưng chưa đủ để cứu tầng 2 (phần cứu được vẫn sai khoảng 8–14%, trong
  khi dữ liệu đang giữ ước dưới 1,5%) → giữ phương án A. Chờ anh quyết.

**Prompt 91:**
> "Là hiện có những hướng nào, nói chi tiết, 2 mô hình đó chưa ổn hử"

→ Xác nhận: cả bge-m3 lẫn CometKiwi đều chưa đủ để cứu tầng 2. Số liệu tầng 2 (P10_tang2_cho_qwen.csv): 130.229 giá
trị khác nhau / 149.123 lượt; 116.897 giá trị (90%) chỉ xuất hiện 1 lần; muốn phủ 50% lượt cần 55.668 giá trị; 46% là
颜色 (59.803), tiếp 花型 15.139, 包装规格 10.138; giá trị zh trung vị 11 ký tự. 36.184 cặp T2 (34.455 chỉ có mã T2);
theo category: home 7.026, mother_baby 5.069, food 4.824, shoes 3.983, bags 3.816, beauty 3.568, auto 2.862, fashion
2.814, electronics 2.222. Đưa 6 hướng (giữ A; cứu 4.341 cặp bằng luật cắt giá trị hiếm; QE lớn hơn CometKiwi-XL/
MetricX-24; LLM mạnh hơn: Qwen 14B Colab hoặc Claude Haiku API có phí; luật "giá trị an toàn" theo từ điển; để tầng 2
cho bước huấn luyện quyết định), em khuyên làm hướng 2 trước rồi hướng 5. Chờ anh chọn.

**Prompt 92:**
> "Data cleaned hiện tại đã có thể đẩy lên HF rồi phải không"

→ Trả lời: đã sẵn sàng về mặt dữ liệu (mọi cổng đạt, quét PII sạch, README + SHA256). Kiểm HF (API công khai, chỉ
đọc): `data/cleaned/` hiện chỉ có `bilingual_zh_vi_cleaned(huy).parquet` (27,4 MB) → tên file của mình không trùng.
Đề xuất đổi tên README/SHA khi đẩy thành `README_desc(nhatanh).md`, `SHA256SUMS_desc(nhatanh).txt` để không chiếm tên
chung. Cần anh chốt: (1) đẩy ngay bản desc_v1 hay đợi hướng 2/5 (sẽ thành v2); (2) cách đẩy: anh tự tải lên qua web
(khuyên, không cần cài gì) hay em đẩy bằng script (cần token Write + cài huggingface_hub vào .venv, phải xin phép);
(3) tài khoản HF trinhnhatanh có quyền ghi repo của Quang chưa (nếu chưa: mở Pull Request trên HF hoặc nhờ Quang thêm).
Nhắc xoá token `colab-kiwi` đã lộ trong chat.

**Prompt 93:**
> "Đẩy thẳng HF cho tui , nhớ đừng đè vào bất kỹ file gì đã có sẵn trên HF nhé"

→ Máy không có git/git-lfs/hf CLI toàn cục và chưa đăng nhập HF; dùng lại Python của `e-commerce - crawl/ecom_crawler/.venv`
(đã có huggingface_hub 1.31.0 + `hf.exe`), không cài gì thêm. Viết `scripts/release/push_desc_hf.py`:
`kiem` = so SHA256 file trên máy, liệt kê repo, dừng nếu 1 trong 4 đường dẫn đích đã có; `day` = whoami, kiem, 1 commit
chỉ thêm 4 file (`parent_commit` = phiên bản vừa kiểm), rồi kiểm lại 4 file mới đúng kích thước và mọi mục cũ giữ nguyên
(kích thước + sha256 LFS). Đích: `data/cleaned/bilingual_zh_vi_desc_cleaned(nhatanh).parquet`,
`..._desc_pairs(nhatanh).parquet`, `README_desc(nhatanh).md`, `SHA256SUMS_desc(nhatanh).txt`. Chạy `kiem`: repo @
f5e2fa2abc, 79 mục, `data/cleaned/` chỉ có file của Huy, 4 đích chưa có, SHA khớp. README bản công bố sửa dòng trỏ tên
file SHA mới. Chờ anh đăng nhập HF bằng token có quyền ghi (`hf auth login` trong terminal, không dán token vào chat).

**Prompt 94:**
> "Ủa sao Huy đẩy thẳng lên đó được"

→ Giải thích (theo ảnh trang `data/cleaned` anh gửi): commit file của Huy đứng tên **ntquang0410** ("1 contributor"), tức
được đẩy bằng tài khoản/token của Quang, không phải tài khoản riêng của Huy. Repo thuộc tài khoản cá nhân Quang, HF không
cho thêm người ghi vào repo cá nhân → tài khoản trinhnhatanh trên trình duyệt chỉ mở được Pull Request. Cách: (1) Quang
merge PR; (2) dùng token ghi của Quang như Huy (bộ crawl có HF_TOKEN trong .env; Claude bị chặn đọc .env, anh tự chạy);
(3) chuyển repo sang Organization để mọi người ghi trực tiếp. (Các prompt về cách đẩy HF ngay trước đó anh dặn không ghi.)

**Prompt 95:**
> "Hãy commit và push lên nhánh https://github.com/ntquang-0410/eda_ecom/tree/nhatanh_updating"

→ Đã làm:
1. git: không có trên PATH; dùng git của GitHub Desktop (`%LOCALAPPDATA%\GitHubDesktop\app-3.4.19\resources\app\git\cmd\
   git.exe`, 2.47.1); tài khoản git `nhatanh20022005 <bena20022005@gmail.com>`, credential manager. Nhánh từ xa:
   `main`, `giahuy`, `nhatanh_updating` (095bb03, bản 28/09, 14 file).
2. Thư mục làm việc vẫn KHÔNG là repo git (CLAUDE.md 01/10): clone riêng nhánh `nhatanh_updating` vào scratchpad, dùng
   `--git-dir=<clone>/.git --work-tree="D:\nam4_ky1\e-commerce - EDA"` để `add -A` theo `.gitignore` (bỏ .venv, *.parquet,
   data/interim, .env). `.gitignore` thêm `data/release/desc_v1_tai_len_hf/` (bản chép tạm) và
   `data/eda/description/_thu_gia/` (số giả).
3. Repo công khai → quét các file sắp commit: không có token; có số điện thoại thật trong ví dụ EDA/notebook/trang
   pipeline, cột `shop` ở 2 CSV. Chỉ che trong BẢN COMMIT (git hash-object + update-index; file trên máy giữ nguyên, kể cả
   file anh đã điền): số di động → `138****1990`, máy bàn, ID WeChat/QQ (bắt buộc có ":"/khoảng trắng sau từ khoá; lần
   đầu che nhầm mã hàng `WXY-K262C` nên sửa luật và làm lại), bỏ cột `shop`; không đưa
   `desc_thong_tin_ca_nhan_editted.csv`. 14 file được che; quét lại toàn bộ index (cả file cũ từ 28/09): 0 sót. 6 chỗ
   "liên hệ" còn lại là mã hàng (VXL8414, VXY-1012), giữ nguyên.
4. Commit `10095a7` trên `nhatanh_updating`: 111 file (thêm notebooks/, scripts/, src/, docs/, data/eda/description/,
   data/processed/desc/duyet/, data/release/desc_v1 README+SHA, CLAUDE.md; chuyển eda-bilingual_zh_vi.ipynb vào
   notebooks/). Commit thứ hai: WALKTHROUGH này. Đẩy lên `origin nhatanh_updating` (không đụng main, giahuy).

**Việc còn nợ (cập nhật 01/10, sau khi chuyển folder):**
- Anh thoát và mở lại Ollama từ khay → em chạy lại smoke test notebook (40 dòng)
  và `sim_full.py check`.
- Mở phiên Claude mới **từ folder này** (`CLAUDE.md` và `WALKTHROUGH.md` tự nạp).
- Nợ ngữ cảnh (7 mục) và quyết định đang chờ: xem mục 0.
- Duyệt `data/eda/title_audit/title_latin_tokens_editted.csv` (41 từ `XEM_TAY`)
  rồi mới viết script áp vào `title_vi`.
- Đẩy CSV kết quả `data/eda/full_run/` và các file mới lên `eda_ecom` nếu muốn
  team thấy (folder này không phải git repo: cần hỏi anh cách nối lại).
- Gộp 3 bản notebook (anh, Quang, Huy); chạy lại số chính thức trên Colab GPU.
- Mục 5.5 notebook đã xong từ 28/09 (kết quả ở mục 4.2).
