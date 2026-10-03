# SNAPSHOT: dữ liệu đầu vào đã đóng băng

Ba file parquet trong thư mục này là bản sao của snapshot trên Hugging Face
(`ntquang0410/zh-vie_ecom`, **công khai**, tải không cần token). **Không sửa,
không ghi đè.** Kết quả xử lý đi vào `data/processed/`, không đi vào đây.

| File | Dòng | SHA256 | `crawl_time` (UTC) |
|---|---|---|---|
| `bilingual_zh_vi.parquet` | 16.348 | `B4E5CE75D5AE8AF523235B5C23F5C7289E405A8230D879F11136AFC938135A29` | 2026-09-17 19:28 → 2026-09-21 21:24 |
| `mono_zh.parquet` | 6.334 | `34A568997506C6AB9C95EEE33611F0A1EC7BC72E82071FAD010A4268637D24CB` | 2026-09-17 19:28 → 2026-09-20 19:40 |
| `tiki_vi.parquet` | 13.200 | `4BD8CE3559FB93F35FEC3EF8149FC991BD0E1DD2DD4BB2FA324D2374C4DD9710` | 2026-09-26 04:04 → 2026-09-26 11:06 |

- `bilingual_zh_vi`: worker_nhat_anh 10.396 dòng + worker_huy 5.952 dòng; 9
  category: bags 2.878, mother_baby 2.496, home 2.493, shoes 2.311, food 1.421,
  fashion 1.351, beauty 1.318, electronics 1.109, auto 971.
- `mono_zh` (chỉ tiếng Trung): bags 2.048, electronics 1.378, fashion 1.145,
  auto 730, beauty 664, food 368, mother_baby 1.
- `tiki_vi` (tiếng Việt bản địa): 6 category × 2.200 dòng: food, home, beauty,
  mother_baby, electronics, auto. **Không có** fashion, shoes, bags (theo file
  thiết kế cũ, 3 category đó định lấy từ nguồn Kaggle Tiki, chưa tải).
- `auto`: 971 dòng ở `bilingual_zh_vi` + 730 dòng ở `mono_zh` = 1.701 dòng crawl
  từ 1688 (đã kiểm 01/10). Con số 1.701 trong README crawler là tổng này, không
  phải riêng bảng song ngữ.
- Phía tiếng Việt của `bilingual_zh_vi` là **bản máy dịch của Alibaba**.
- Bản copy này lấy từ `ecom_crawler/data/snapshot_tmp/` ngày 01/10/2026. Riêng
  `bilingual_zh_vi.parquet` đã so hash với bản trong cache HF tải ngày 28/09:
  trùng. Lúc kiểm tra, HF có commit mới (01/10 10:12 UTC, revision `f5e2fa2a`)
  nhưng kích thước 3 file snapshot không đổi so với bản này; chưa so hash
  `mono_zh` và `tiki_vi` với HF.

Tải lại khi cần (công khai):

```
curl.exe -L -o data\snapshot\bilingual_zh_vi.parquet https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/resolve/main/data/snapshot/bilingual_zh_vi.parquet
```

Sau khi tải, đối chiếu SHA256 ở bảng trên. Nếu lệch là HF đã đổi: **không ghi đè**,
lưu thành bản mới có ngày và ghi vào file này.
