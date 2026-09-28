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

**Việc còn nợ:**
- Số liệu full-data của notebook mục 5.5 (description theo cặp thuộc tính) —
  chạy nền xong sẽ cập nhật vào mục 4.2, kèm danh sách chuỗi Ollama từ chối.
- Nhóm chốt thứ tự làm ở mục 6.1 trước khi viết script sửa data thật.
- Nhật Anh duyệt `title_latin_tokens_editted.csv` (41 từ `XEM_TAY` + cột
  `anh_duyet`), sau đó mới viết script áp vào `title_vi`.
- Script audit (title_audit.py, data_issues.py, tok_prep.py, dec_pinyin.py,
  dec_en.py, build_tokens_edited.py) vẫn chưa nằm trong repo nào — chỉ có kết
  quả CSV. Dọn đường dẫn + commit khi nhóm cần chạy lại.
