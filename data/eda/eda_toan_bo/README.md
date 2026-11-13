# Hai tập dữ liệu EDA riêng: cụm và pair

## Hai tập, không gộp

Thư mục này có **hai tập riêng**, không có file gộp:

| Tập | File | Số dòng | Số sản phẩm | Cột |
|---|---|---:|---:|---:|
| Cụm | `bo_cum_full_25575.parquet` | 25.575 | 25.575 | 9 |
| Pair | `bo_pair_full_25575.parquet` | 25.575 | 25.575 | 9 |

Hai file là **bản sao nguyên byte** từ hai file gốc trong `data/processed/bo_chon_hf/`:

- `bo_cum_chon_deepseek.parquet` → `bo_cum_full_25575.parquet`
- `bo_pair_chon_deepseek.parquet` → `bo_pair_full_25575.parquet`

Đã kiểm bằng so sánh byte: giống hoàn toàn. Không sửa nội dung, không thêm cột, không có cột `ban`.

## Không được gộp hai tập làm một

Hai tập là **hai cách biểu diễn của cùng 25.575 sản phẩm**, không phải hai tập sản phẩm độc lập. Nếu nối lại sẽ thành 51.150 dòng với mỗi sản phẩm lặp hai lần, làm sai mọi thống kê và mọi phép chia tập. Vì vậy:

- Phân tích cụm: dùng `bo_cum_full_25575.parquet`.
- Phân tích pair: dùng `bo_pair_full_25575.parquet`.
- Muốn so sánh hai bản: ghép theo `product_id`, không nối dọc.

## Cảnh báo quan trọng

Mỗi tập có đủ **25.575 sản phẩm**, tức **gồm cả 2.558 sản phẩm đang nằm trong tập test khóa**. Hai file **không có cột đánh dấu test**, vì muốn thêm cờ test phải mở `split.parquet` trong `test_khoa`.

- Chỉ dùng cho **EDA, mô tả dữ liệu, kiểm tra chất lượng, viết báo cáo**.
- **Không dùng để chọn mô hình, chọn trọng số, hay huấn luyện bản cuối.**
- Nếu dùng, test không còn là tập chưa từng thấy và điểm test cũ (cụm 75,63/77,36; pair 75,41/77,65) mất giá trị đánh giá độc lập.
- Nếu cần huấn luyện trên cả 25.575 sản phẩm, đó là quyết định bỏ khóa test; phải ghi rõ và chấp nhận điểm test cũ không còn dùng để kết luận.

## Cột

Cả hai file có đúng 9 cột:

| Cột | Ý nghĩa |
|---|---|
| `product_id` | mã sản phẩm |
| `source` | `1688` hoặc `tiki` |
| `category` | ngành hàng |
| `title_zh` | title tiếng Trung |
| `title_vi` | title tiếng Việt |
| `description_zh` | mô tả/danh sách thuộc tính tiếng Trung |
| `description_vi` | mô tả/danh sách thuộc tính tiếng Việt |
| `judge_chon_title` | model tạo title cuối (`gemma` hoặc `qwen`) |
| `judge_chon_description` | chuỗi đếm nguồn nội dung cuối, không phải điểm |

## Thống kê kèm theo

| File | Nội dung |
|---|---|
| `thong_ke_cum.csv` | Thống kê tập cụm theo nguồn: số dòng, độ dài title/mô tả, số mục, missing/empty |
| `thong_ke_pair.csv` | Thống kê tập pair theo nguồn |
| `phan_bo_category_cum.csv` | Phân bố ngành hàng tập cụm |
| `phan_bo_category_pair.csv` | Phân bố ngành hàng tập pair |

Độ dài trong các file thống kê là **số ký tự**, không phải số token. Muốn số token thì dùng tokenizer Qwen3.

## Số nhanh

| Tập | Nguồn | Số dòng | Title Việt (ký tự) | Mô tả Việt (ký tự) | Số mục/dòng |
|---|---|---:|---:|---:|---:|
| Cụm | 1688 | 14.837 | 112,6 | 1.084,7 | 24,8 mục |
| Cụm | Tiki | 10.738 | 79,4 | 286,1 | 6,9 mục |
| Pair | 1688 | 14.837 | 112,6 | 1.036,7 | 24,4 dòng |
| Pair | Tiki | 10.738 | 79,4 | 279,5 | 6,9 dòng |

- Cụm: mô tả là một chuỗi `name: value; ...`, nên `desc_lines_mean = 1`, `desc_items_mean` là số mục.
- Pair: mô tả là nhiều dòng `name: value`, nên `desc_lines_mean` là số dòng.
- Không có missing hoặc rỗng ở `title_vi` và `description_vi` trong cả hai tập.

## Cách đọc

```python
import pandas as pd

cum = pd.read_parquet(r"D:\nam4_ky1\e-commerce - EDA\data\eda\eda_toan_bo\bo_cum_full_25575.parquet")
pair = pd.read_parquet(r"D:\nam4_ky1\e-commerce - EDA\data\eda\eda_toan_bo\bo_pair_full_25575.parquet")

# lọc theo nguồn
cum_1688 = cum[cum["source"] == "1688"]
pair_tiki = pair[pair["source"] == "tiki"]
```

## Nguyên tắc

- Không sửa hai file gốc trong `data/processed/bo_chon_hf/`.
- Không mở `data/processed/ft_run/test_khoa/`, `split.parquet`, `ds_test.txt`.
- Hai file trong thư mục này chỉ là bản EDA, không phải tập huấn luyện chính thức.
