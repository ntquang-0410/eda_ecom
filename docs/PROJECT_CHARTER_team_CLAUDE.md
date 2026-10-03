> **LƯU Ý KHI ĐỌC (thêm ngày 01/10/2026; phần bên dưới dòng `---` ngay sau khung này là NGUYÊN VĂN).**
> Đây là `CLAUDE.md` gốc do nhóm viết cho giai đoạn crawl; Nhật Anh đính kèm vào phiên Claude ngày 15/09/2026, trước khi có `CLAUDE.md` hiện tại ở thư mục gốc (bộ quy tắc hành vi).
>
> **Còn hiệu lực và quan trọng cho bước EDA + Xử lý dữ liệu:** §1 (bối cảnh, vì sao phải tự crawl, mô hình), §4 "Nguyên tắc thư mục dữ liệu" (`raw/` bất biến, `processed/` mỗi bước một file mới, `final/`), §6 (bẫy: nhiễm dữ liệu máy dịch, trùng lặp/MinHash, cân bằng danh mục, emoji, thông tin cá nhân), §7 (tập test phải do người dịch hoặc kiểm duyệt).
>
> **Đã lỗi thời, đừng làm theo:** §2 "Trạng thái hiện tại" (nay đã crawl xong 1688 song ngữ và Tiki đơn ngữ Việt); §3 kiến trúc crawler (đã đổi nhiều, xem `ecom_crawler_README.md`); §4 "Định dạng lưu" và "Schema tối thiểu" (schema thật nằm trong `ecom_crawler_README.md`, snapshot dùng Parquet); §5 quy tắc sửa code crawler (không áp dụng ở folder này); §8 checklist; đường dẫn thư mục dự án ở §6.
>
> Quy tắc nào của file này mâu thuẫn với quyết định sau của Nhật Anh (xem `CLAUDE.md` mục 5.3) thì theo quyết định sau.
>
> **01/10/2026:** mô hình nền ở §1 (Sailor2-8B, Seed-X-7B) đã được gỡ khỏi pipeline, chờ nhóm bàn lại (WALKTHROUGH Prompt 23).

---

# CLAUDE.md — ecom_crawler

Tài liệu ngữ cảnh cho Claude Code. Đọc trước khi sửa bất kỳ file nào trong project.

---

## 1. BỐI CẢNH NGHIÊN CỨU

### Đề tài
**Tối ưu hoá hệ thống dịch máy Việt–Trung chuyên biệt cho thương mại điện tử xuyên biên giới.**

Đây là Tiểu luận chuyên ngành (TLCN), nhóm 3 người, chạy song song với NCKH và Khoá luận tốt nghiệp. Thời gian còn lại: khoảng 2-3 tháng.

### Vấn đề cốt lõi
Các mô hình dịch máy hiện tại được huấn luyện trên văn bản tổng quát (tin tức, sách, hội thoại) — văn phong đúng ngữ pháp. Nhưng văn bản thương mại điện tử **vi phạm ngữ pháp có chủ đích**: người bán dồn nén từ khoá thay vì viết câu hoàn chỉnh.

Ví dụ minh hoạ xuyên suốt dự án:
- Tiếng Việt: `Áo thun nam cotton form rộng, freeship toàn quốc`
- Tiếng Trung: `男士纯棉宽松T恤，全国包邮`

Mô hình tổng quát khi dịch câu trên có xu hướng thêm diễn giải thừa (`这是一件...` = "đây là một chiếc..."), làm hỏng văn phong tiêu đề sản phẩm.

### Vì sao phải tự crawl
**Không tồn tại bộ dữ liệu song ngữ Việt–Trung miền TMĐT công khai nào.** Đây là lý do duy nhất khiến việc crawl là bắt buộc, và cũng là đóng góp chính của đề tài.

Các nguồn công khai đã có và KHÔNG thay thế được việc crawl:
- **VLSP 2022** (300K câu Vi-Zh) — miền TIN TỨC, chỉ dùng làm benchmark đối chiếu, không phải dữ liệu chính
- **WanJuanSiLu** (74,1 tỷ token tiếng Việt) — webtext đơn ngữ, không có tiếng Trung, không phải tiêu đề sản phẩm
- **OPUS** — có vi-zh nhưng chủ yếu phụ đề phim

### Phần cứng & mô hình (giai đoạn sau, không liên quan trực tiếp crawler)
- GPU: **RTX PRO 6000 Blackwell 96GB** — không bị nghẽn tính toán, nút thắt là DỮ LIỆU
- Mô hình nền chính: **Sailor2-8B** (Apache 2.0, nền Qwen2.5, mạnh cả tiếng Việt lẫn tiếng Trung)
- Nhánh so sánh: **ByteDance Seed-X-7B** (OpenMDW, chuyên dịch máy, hỗ trợ vi↔zh trực tiếp)

---

## 2. CRAWLER NÀY NẰM Ở ĐÂU TRONG DỰ ÁN

Dự án có 8 giai đoạn. Crawler thuộc **Giai đoạn 1 — Xây dựng kho ngữ liệu**, là giai đoạn **quan trọng nhất**: chất lượng dữ liệu quyết định trần chất lượng của toàn bộ mô hình cuối cùng.

### Dữ liệu crawler này tạo ra sẽ được dùng ở đâu

| Loại dữ liệu | Dùng ở giai đoạn | Mục đích |
|---|---|---|
| Tiếng Trung đơn ngữ (Alibaba) | GĐ 1 → mining | Ghép cặp Vi-Zh bằng LaBSE + FAISS |
| Tiếng Trung đơn ngữ | GĐ 4 | Nguồn cho back-translation |
| Tiếng Việt đơn ngữ (Shopee/Tiki/Lazada) | GĐ 1, GĐ 4 | Mining + back-translation |
| Cặp thuật ngữ (cây danh mục) | GĐ 3 bước 1 | Fine-tune học từ vựng chuyên ngành |
| Cặp câu song ngữ | GĐ 3 bước 2 | Fine-tune học văn phong TMĐT |

### Trạng thái hiện tại
- **Đang crawl Alibaba → chỉ lấy tiếng Trung (đơn ngữ)**
- Chưa crawl phía tiếng Việt
- Chưa crawl cây danh mục
- Việc ghép cặp Vi-Zh sẽ làm SAU bằng LaBSE + FAISS, không phải trong crawler này

---

## 3. KIẾN TRÚC CRAWLER

### Mô hình phân tán nhiều worker
Mỗi thành viên nhóm chạy một worker riêng trên máy của mình (đặt tên theo mẫu `worker_<tên>`, ví dụ `worker_nhat_anh`). Các worker chia sẻ **một hàng đợi trung tâm** đặt trên Firebase Realtime Database.

### Firebase Realtime Database — hàng đợi trung tâm

Thư mục `firebase/` chứa 2 file:

**1. `ecom-crawler-*-firebase-adminsdk-*.json`** — khoá service account
- Đường dẫn khai báo trong `.env` tại biến `FIREBASE_CRED_PATH`
- Khởi tạo tại `core/queue_manager.py` (~dòng 79-81):
  ```python
  cred = credentials.Certificate(cred_path)
  firebase_admin.initialize_app(cred, {"databaseURL": db_url})
  ```
- `config.py` (~dòng 126-129) kiểm tra file tồn tại lúc khởi động, thiếu là lỗi ngay
- **TUYỆT ĐỐI KHÔNG commit file này lên git** — đây là khoá admin

**2. `database.rules.json`** — quy tắc bảo mật + index
```json
"queue": { ".indexOn": ["status"], ".read": false, ".write": false }
```
- `.indexOn: ["status"]` là **BẮT BUỘC**: `claim_urls()` và `recover_stale_locks()` truy vấn bằng `order_by_child("status")`. Thiếu index → Firebase từ chối query → lỗi `missing the required index` (xem `core/queue_manager.py` ~dòng 99-107)
- `.read`/`.write` = `false` chặn mọi client thường; chỉ service account (quyền admin) bỏ qua được rules
- File này **không được code đọc tự động** — là bản mẫu, phải copy dán vào Firebase Console → Realtime Database → Rules → Publish (làm một lần)

### Các thao tác hàng đợi (`core/queue_manager.py`)
| Hàm | Vai trò |
|---|---|
| `claim_urls()` | Worker nhận một lô URL, đánh dấu đang xử lý (tránh 2 worker trùng việc) |
| `release()` | Trả URL về hàng đợi khi worker gặp lỗi/dừng |
| `mark_done()` | Đánh dấu URL đã crawl xong |
| `recover_stale_locks()` | Giải phóng URL bị worker chết giữa chừng giữ khoá |

### Cấu trúc thư mục
```
ecom_crawler/
├── core/          # Logic lõi: queue_manager.py, ...
├── parsers/       # Bóc tách dữ liệu theo từng site
├── scripts/       # Script tiện ích, chạy tay
├── firebase/      # Khoá service account + database.rules.json
├── config.py      # Nạp .env, validate cấu hình lúc khởi động
├── main.py        # Điểm vào chạy worker
├── urls.txt       # Danh sách URL nguồn / seed
├── .env           # Bí mật — KHÔNG commit
├── .env.example   # Mẫu để thành viên mới copy
└── requirements.txt
```

---

## 4. QUY ƯỚC DỮ LIỆU (BẮT BUỘC TUÂN THỦ)

### Định dạng lưu
- **Giai đoạn crawl: JSONL** (mỗi dòng một JSON object), ghi chế độ append
- **KHÔNG dùng CSV/TSV** — tiêu đề sản phẩm chứa dấu phẩy, ngoặc kép, emoji, xuống dòng → vỡ cấu trúc âm thầm, không báo lỗi
- **KHÔNG dùng JSON một file lớn** — không append được, đứt giữa chừng là hỏng toàn bộ

### Ghi tiếng Trung — lỗi hay gặp nhất
```python
# ĐÚNG
f.write(json.dumps(record, ensure_ascii=False) + "\n")

# SAI — ký tự Hán thành \u5305\u90ae, file phình 5-6 lần, không đọc được bằng mắt
f.write(json.dumps(record) + "\n")
```
Luôn mở file với `encoding="utf-8"`.

### Schema tối thiểu mỗi record
```json
{
  "product_id": "...",
  "title_zh": "男士纯棉宽松T恤，全国包邮",
  "category": "...",
  "source_site": "alibaba",
  "url": "...",
  "crawl_time": "2026-09-15T10:30:00Z",
  "worker": "worker_nhat_anh"
}
```
Khi crawl được cả hai ngôn ngữ thì thêm `title_vi`. Giữ `product_id` vì bước chia train/dev/test sau này phải chia **theo product_id**, không chia ngẫu nhiên theo dòng.

### Nguyên tắc thư mục dữ liệu
```
data/
  raw/          # JSONL — BẤT BIẾN, không bao giờ sửa file đã ghi
  state/        # trạng thái local (nếu có, ngoài Firebase)
  processed/    # Parquet — sản phẩm từng bước làm sạch
  final/        # train.jsonl / dev.jsonl / test.jsonl
```
**`raw/` là bất biến.** Mỗi bước làm sạch tạo file MỚI ở `processed/`. Lý do: nếu phát hiện luật lọc sai ở tuần thứ 6, chỉ cần chạy lại từ `raw/`, không phải crawl lại — khác biệt giữa 1 tuần và 3 tuần làm lại.

### Chia nhỏ file (sharding)
Đặt tên theo site + ngày + số lô: `alibaba_zh_20260915_001.jsonl`. Mỗi shard khoảng 100-500 MB. Không dồn tất cả vào một file.

---

## 5. QUY TẮC KHI SỬA CODE

1. **Rate limit là bắt buộc**: `time.sleep(random.uniform(1, 3))` giữa các request, không dùng số cố định. Gặp HTTP 429/403 → exponential backoff, không retry dồn dập.
2. **Tôn trọng `robots.txt`** của từng site.
3. **Luôn giữ khả năng resume**: mọi thay đổi phải bảo toàn cơ chế claim/mark_done của Firebase. Crawl vài chục nghìn trang chắc chắn sẽ đứt giữa chừng.
4. **Không bao giờ ghi đè hay sửa file trong `raw/`.**
5. **Không commit `.env` và file service account JSON.** Kiểm tra `.gitignore` đã chặn cả hai.
6. **Ưu tiên API nội bộ hơn render trình duyệt**: mở DevTools → tab Network → lọc XHR/Fetch, tìm endpoint trả JSON. Nhanh hơn Playwright hàng chục lần và dữ liệu đã có cấu trúc.
7. **Thêm parser mới thì đặt trong `parsers/`**, không nhét vào `core/`.

---

## 6. BẪY ĐÃ BIẾT — CẦN CẢNH GIÁC

### Nguy cơ nhiễm dữ liệu máy dịch (quan trọng nhất)
Các sàn xuyên biên giới (AliExpress, Temu, Shein, listing xuyên biên giới trên Lazada) **dịch tiêu đề tự động bằng máy**. Nếu lấy cặp song ngữ từ đó làm dữ liệu huấn luyện, mô hình của nhóm học từ một hệ dịch máy khác → trần chất lượng bị khoá.

Nếu sau này crawl cặp song ngữ, phải kiểm tra thủ công 50-100 mẫu, tìm 3 dấu hiệu:
- Trật tự từ còn sót kiểu tiếng Trung (`cotton nam áo thun` thay vì `áo thun nam cotton`)
- Lượng từ dịch máy móc (`一件` → `một cái` ở chỗ tiếng Việt không cần)
- Thuật ngữ dịch word-by-word (`包邮` → `bao bưu` thay vì `freeship`)

Phát hiện là máy dịch thì vẫn dùng được cho tập TRAIN, nhưng **tuyệt đối không dùng cho tập TEST**, và phải ghi rõ trong báo cáo.

### Trùng lặp cực nặng
Hàng trăm shop đăng cùng một mẫu sản phẩm với tiêu đề gần giống nhau. Không dedup kỹ → tập train bị vài mẫu phổ biến thống trị, và tập test rò rỉ từ train → BLEU cao giả tạo.
Dùng **MinHash/LSH** (thư viện `datasketch`), ngưỡng Jaccard ~0,8. Đây là bước cắt nhiều dữ liệu nhất, phải dự trù crawl dư vài lần.

### Cân bằng danh mục
Nếu 80% dữ liệu là thời trang thì mô hình dịch tệ ở điện tử, gia dụng. Crawl **theo hạn ngạch từng danh mục**, không crawl tự do rồi mới đếm.

### Emoji và ký tự trang trí
Tiêu đề TMĐT đầy `🔥HOT🔥 FREESHIP⚡`. **Đừng xoá sạch** — đó là đặc trưng thật của miền, mô hình cần học xử lý. Chỉ chuẩn hoá ký tự điều khiển và khoảng trắng lặp.

### Thông tin cá nhân
Mô tả sản phẩm có thể chứa tên shop, số điện thoại, địa chỉ. Phải lọc trước khi công bố dữ liệu.

### Đường dẫn Windows có dấu cách
Project nằm ở `d:/nam4_ky1/e-commerce - crawl/ecom_crawler/` — tên thư mục có dấu cách. Mọi lệnh shell phải bọc đường dẫn trong dấu ngoặc kép.

---

## 7. NGUYÊN TẮC VÀNG VỀ TẬP TEST

Chuẩn cho tập TEST khác hẳn chuẩn cho tập TRAIN.

- **Train**: chấp nhận dữ liệu nhiễu, mining tự động, thậm chí có gốc máy dịch (có ghi chú)
- **Test**: **bắt buộc do người dịch hoặc người kiểm duyệt**. Nếu bản tham chiếu là máy dịch thì điểm BLEU chỉ đo "mô hình giống máy dịch của sàn đến đâu" — vô nghĩa về mặt khoa học, làm hỏng toàn bộ phần thực nghiệm.

Quy mô test cần: 500-1.000 cặp là đủ (tham khảo: bài G2ST chỉ dùng 2.000 câu test, toàn bộ do 3 người có kinh nghiệm TMĐT dịch tay).

Chia train/dev/test **theo `product_id` hoặc theo shop**, không chia ngẫu nhiên theo dòng — nếu cùng một sản phẩm xuất hiện ở cả train và test thì điểm test sẽ ảo.

---

## 8. CẦN BỔ SUNG (nhóm tự điền)

Các mục dưới đây Claude Code chưa có thông tin, nhóm nên bổ sung để tăng độ chính xác:

- [ ] Nội dung cụ thể của `core/` (ngoài `queue_manager.py`)
- [ ] Danh sách parser hiện có trong `parsers/` và site tương ứng
- [ ] Các script trong `scripts/` dùng để làm gì
- [ ] Schema output thực tế hiện tại (các trường đang lưu)
- [ ] `urls.txt` chứa gì — seed URL danh mục hay danh sách product URL
- [ ] Cách chạy worker (lệnh cụ thể, tham số của `main.py`)
- [ ] Các biến trong `.env` và ý nghĩa
