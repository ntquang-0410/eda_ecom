# 📖 Hướng dẫn sử dụng `query_parquet.py`

> Công cụ truy vấn dữ liệu file `.parquet` bằng SQL, tìm kiếm từ khóa, và xem schema — dành cho dataset song ngữ Trung-Việt 1688.

---

## Mục lục

1. [Cài đặt](#1-cài-đặt)
2. [5 chế độ sử dụng](#2-5-chế-độ-sử-dụng)
3. [Bảng tham chiếu tham số](#3-bảng-tham-chiếu-tham-số)
4. [Bảng dữ liệu & tên cột](#4-bảng-dữ-liệu--tên-cột)
5. [Mẹo & ví dụ nâng cao](#5-mẹo--ví-dụ-nâng-cao)

---

## 1. Cài đặt

```bash
pip install duckdb pandas
```

Script tự động dùng **DuckDB** (nhanh, hỗ trợ SQL đầy đủ). Nếu không có DuckDB, fallback về **Pandas** (chậm hơn, ít tính năng SQL hơn).

---

## 2. 5 chế độ sử dụng

### 2.1 Console tương tác (mặc định)

```bash
python query_parquet.py
python query_parquet.py -f data/cleaned/bilingual_zh_vi_cado.parquet
```

Giao diện dòng lệnh, hỗ trợ:

| Nhập vào | Hành động |
|:---|:---|
| Câu SQL bắt đầu bằng `SELECT`, `DESCRIBE`, `SHOW`... | Thực thi SQL |
| `WHERE category = 'bags' AND price > 50` | Lọc theo điều kiện |
| `944607859983` (từ khóa bất kỳ) | Tìm trong tất cả các cột |
| `schema` | Xem danh sách cột |
| `set format table` / `set format expanded` | Đổi chế độ hiển thị |
| `exit` / `quit` / `q` | Thoát |

**Ví dụ phiên làm việc:**

```
sql> schema
📋 CẤU TRÚC BẢNG (SCHEMA) - data/raw/bilingual_zh_vi.parquet
┌─────────────────────────┬─────────────┐
│       column_name       │ column_type │
├─────────────────────────┼─────────────┤
│ product_id              │ VARCHAR     │
│ title_zh                │ VARCHAR     │
│ title_vi                │ VARCHAR     │
│ ...

sql> SELECT category, count(*) AS total, round(avg(price),2) AS avg_price
     FROM data GROUP BY category ORDER BY total DESC LIMIT 5

sql> bàn chải
🔍 Đang tìm kiếm: 'bàn chải'
📊 Tìm trong 30 cột

sql> exit
Tạm biệt! 👋
```

---

### 2.2 SQL trực tiếp (1 dòng)

```bash
# Cơ bản
python query_parquet.py "SELECT product_id, title_vi, price FROM data WHERE price < 15 LIMIT 5"

# GROUP BY + aggregation
python query_parquet.py "SELECT category, count(*) AS total, round(avg(price),2) AS avg_price FROM data GROUP BY category"

# Chỉ định file khác
python query_parquet.py -f data/cleaned/bilingual_zh_vi_cado.parquet "SELECT * FROM data LIMIT 3"

# Kết hợp với --limit và --output
python query_parquet.py "SELECT * FROM data WHERE category = 'bags'" -n 0 -o bags.csv
```

---

### 2.3 Lọc WHERE nhanh

```bash
# Tìm sản phẩm theo ID
python query_parquet.py -w "product_id = '944607859983'"

# Kết hợp điều kiện
python query_parquet.py -w "category = 'bags' AND price < 20"

# Chỉ chọn một số cột
python query_parquet.py -s "product_id, title_vi, price" -w "price > 100"

# Xuất kết quả
python query_parquet.py -w "category = 'shoes'" -o shoes.csv
```

---

### 2.4 Tìm kiếm từ khóa toàn văn

```bash
# Tìm trong tất cả các cột
python query_parquet.py "bàn chải"

# Giới hạn số kết quả
python query_parquet.py "iPhone" -n 5

# Chỉ tìm trong cột cụ thể
python query_parquet.py "túi xách" -c "title_vi,description_vi"

# Phân biệt hoa/thường
python query_parquet.py "USB" -s

# Xuất kết quả
python query_parquet.py "mặt nạ" -o mask_results.csv
```

---

### 2.5 Xem Schema

```bash
python query_parquet.py --schema
python query_parquet.py --schema -f data/cleaned/bilingual_zh_vi_cado.parquet
```

---

## 3. Bảng tham chiếu tham số

| Tham số | Viết tắt | Mô tả | Ví dụ |
|:---|:---:|:---|:---|
| `query_or_keyword` | (vị trí) | SQL hoặc từ khóa tìm kiếm | `"SELECT * FROM data"` hoặc `"túi xách"` |
| `-f` / `--file` | `-f` | Đường dẫn file parquet | `-f data/cleaned/bilingual_zh_vi_cado.parquet` |
| `--sql` | — | Thực thi SQL trực tiếp | `--sql "SELECT count(*) FROM data"` |
| `-w` / `--where` | `-w` | Điều kiện WHERE | `-w "price < 20 AND category = 'bags'"` |
| `--select` / `-s` | `-s` | Cột cần SELECT (dùng với `-w`) | `-s "product_id, title_vi, price"` |
| `-n` / `--limit` | `-n` | Số dòng tối đa hiển thị (0 = tất cả) | `-n 20` |
| `-x` / `--expanded` | `-x` | Hiển thị dạng key-value từng dòng | `-x` |
| `-t` / `--table` | `-t` | Ép hiển thị dạng bảng ASCII | `-t` |
| `--schema` | — | In danh sách cột + kiểu dữ liệu | `--schema` |
| `-s` / `--case-sensitive` | `-s` | Phân biệt hoa/thường khi tìm | `-s` |
| `-c` / `--columns` | `-c` | Chỉ tìm trong cột chỉ định | `-c "title_vi,description_vi"` |
| `-o` / `--output` | `-o` | Xuất ra file (.csv / .json / .parquet) | `-o ket_qua.csv` |

---

## 4. Bảng dữ liệu & tên cột

Script tự động đăng ký **3 alias** cho cùng một bảng:

| Tên bảng | Dùng trong SQL |
|:---|:---|
| `data` | `SELECT * FROM data WHERE ...` |
| `products` | `SELECT * FROM products WHERE ...` |
| `bilingual_zh_vi` | `SELECT * FROM bilingual_zh_vi WHERE ...` |

### Các cột chính (file CADO đã cleaned):

| Cột | Kiểu | Mô tả |
|:---|:---|:---|
| `product_id` | VARCHAR | Mã sản phẩm 1688 |
| `title_zh` | VARCHAR | Tiêu đề gốc tiếng Trung |
| `title_vi` | VARCHAR | Tiêu đề gốc tiếng Việt |
| `title_vi_cleaned` | VARCHAR | Tiêu đề VI đã CADO làm sạch |
| `description_zh` | VARCHAR | Mô tả gốc tiếng Trung |
| `description_vi` | VARCHAR | Mô tả gốc tiếng Việt |
| `desc_vi_cleaned` | VARCHAR | Mô tả VI đã CADO làm sạch |
| `category` | VARCHAR | Danh mục (bags, shoes, beauty...) |
| `price` | DOUBLE | Giá (CNY) |
| `sales` | DOUBLE | Doanh số |
| `shop` | VARCHAR | Tên cửa hàng |
| `province` / `city` | VARCHAR | Địa điểm nhà cung cấp |
| `attributes_zh` / `attributes_vi` | OBJECT | Thuộc tính sản phẩm (nested) |

> 💡 Dùng `--schema` để xem đầy đủ 30-32 cột.

---

## 5. Mẹo & ví dụ nâng cao

### So sánh trước/sau CADO trên cột tiếng Việt

```bash
# Sản phẩm có title_vi bị CADO thay đổi
python query_parquet.py -f data/cleaned/bilingual_zh_vi_cado.parquet \
  -w "title_vi != title_vi_cleaned" -n 10

# Đếm số dòng bị thay đổi
python query_parquet.py -f data/cleaned/bilingual_zh_vi_cado.parquet \
  "SELECT count(*) FROM data WHERE title_vi != title_vi_cleaned"
```

### Phân tích theo danh mục

```bash
# Top danh mục theo số lượng
python query_parquet.py "SELECT category, count(*) AS cnt FROM data GROUP BY category ORDER BY cnt DESC"

# Giá trung bình mỗi danh mục
python query_parquet.py "SELECT category, round(avg(price),2) AS avg_price, count(*) AS n FROM data GROUP BY category ORDER BY avg_price DESC"
```

### Tìm sản phẩm cụ thể

```bash
# Theo ID
python query_parquet.py -w "product_id = '592582928262'"

# Theo danh mục + giá
python query_parquet.py -w "category = 'bags' AND price BETWEEN 10 AND 50" -n 5

# Tìm từ khóa trong title_vi
python query_parquet.py "túi xách" -c "title_vi" -x
```

### Xuất dữ liệu

```bash
# CSV
python query_parquet.py -w "category = 'beauty'" -o beauty.csv

# JSON
python query_parquet.py "SELECT * FROM data LIMIT 100" -o sample.json

# Parquet (nén)
python query_parquet.py -w "price > 200" -o expensive.parquet
```

### Chế độ hiển thị

| Chế độ | Khi nào dùng |
|:---|:---|
| `auto` (mặc định) | ≤6 cột: bảng ngang; >6 cột: expanded |
| `table` (`-t`) | Muốn xem dạng bảng gọn |
| `expanded` (`-x`) | Muốn xem toàn bộ 30 cột, mỗi cột 1 dòng |
| `-n 0` | Xuất **tất cả** dòng (không giới hạn) |

---

> **Mẹo:** Chạy `python query_parquet.py` không tham số để vào console tương tác — cách nhanh nhất để khám phá dữ liệu!