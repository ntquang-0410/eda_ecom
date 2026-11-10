# Chuẩn hoá đại lượng trước khi tinh chỉnh hyperparameter

Ngày: 2026-11-01. Module: `cado_core/quantity_norm.py` + test `tests/test_quantity_norm.py`.

## 1. Vấn đề

Cách kiểm tra cũ chỉ so **tập chữ số**, nên:

- `144,5 cm` vs `144.5 厘米` → bị coi là lệch (dấu thập phân khác nhau).
- `1.234 kg` vs `1234 克` → lệch (dấu ngăn nghìn + đổi đơn vị).
- `20W` vs `20瓦` → lệch (đơn vị khác ngôn ngữ).
- `10 gói x 16g` vs `10包×16克` → lệch (2 đại lượng, thứ tự khác).

Ngược lại, cách này lại **bỏ qua** các trường hợp sai thật:

- `5cm` vs `8厘米` → cùng tập số? Không, nhưng nếu số trùng mà đơn vị khác
  (`2L` vs `2ml`) thì không phát hiện được.

Hệ quả: nhiễu trong QC và trong tính length prior, ảnh hưởng trực tiếp đến
weight và tham số mix.

## 2. Thiết kế

```text
text → chuẩn hoá NFKC, phân số (½, 1/2)
     → trích số (Ả Rập có locale, số Hán)
     → khớp đơn vị dài nhất
     → quy về đơn vị SI cơ sở
     → signature [(dimension, base_value), ...]
     → so khớp multiset với dung sai 2%
```

### Xử lý số

| dạng | ví dụ | kết quả |
|---|---|---|
| thập phân VI/EU | `144,5` | 144.5 |
| ngăn nghìn US | `1,234.5` | 1234.5 |
| ngăn nghìn VI | `1.234` | 1234 |
| số Hán | `三百`, `十五`, `两`, `半` | 300, 15, 2, 0.5 |
| phân số | `1/2`, `½`, `¾` | 0.5, 0.75 |

### Đơn vị hỗ trợ

| thứ nguyên | đơn vị |
|---|---|
| length | m, cm, mm, km, inch, 英寸, 寸, ft, yard, 尺, 里 |
| mass | kg, g, mg, t, 斤, 两, lb, oz |
| volume | l, ml, cc, 升, 毫升, m³, gallon |
| area | m², cm², 亩 |
| power | W, kW, 瓦, hp |
| voltage | V, kV, 伏 |
| current | A, mA, 安 |
| capacity | Ah, mAh |
| temperature | °C, 摄氏度 |
| time | s, min, h, ngày, 天, 日, 个月, 年 |
| data | b, kb, mb, gb, tb |
| count | cái, chiếc, gói, hộp, miếng, 个, 件, 包, 盒, 袋, 片, 支… |
| currency | đ, vnd, 元, ¥, rmb |

### Chống false positive

```text
- số nằm trong mã sản phẩm (E300, EL0002D189, IN20-24-26M5) -> bỏ
- "20W" vẫn giữ (token = số + đơn vị hợp lệ)
- "千克"/"厘米" là đơn vị, không phải "千" + "克"
- "尺寸" không phải "尺"
- "件套"/"件装"/"一件代发" không phải số đếm
- đơn vị 1 ký tự chỉ nhận khi ký tự sau không phải chữ Latin/Việt
```

## 3. Strict mode

Chỉ so **đại lượng vật lý** (length, mass, volume, area, power, voltage,
current, capacity, temperature, time, data, currency). Bỏ qua:

```text
bare  : số không có đơn vị (3D, 2026, model code)
count : số đếm (cái, 个, 件套)
year  : năm 1900–2100
```

Lý do: số đếm và năm khác cách diễn đạt giữa hai ngôn ngữ
(`bộ bốn mảnh` vs `四件套`, `2026` vs `2026年`), gây nhiễu nặng.

Phân loại mismatch:

```text
strong : cùng thứ nguyên xuất hiện cả hai bên nhưng giá trị lệch  -> tin cậy cao
weak   : chỉ một bên có thứ nguyên đó                             -> gắn cờ mềm
```

## 4. Kết quả trên dữ liệu thật

| nhóm | cặp có đại lượng | naive | quantity | cứu | bắt thêm | strong |
|---|---:|---:|---:|---:|---:|---:|
| 1688 title | 7.632 / 16.133 (47,3%) | 84,66% | **89,92%** | 695 | 274 | **49** |
| 1688 attribute | 104.787 / 395.575 (26,5%) | 68,13% | **96,14%** | 30.823 | 1.473 | **736** |
| Tiki units (8B) | 19.849 / 45.652 (43,5%) | 92,51% | **95,20%** | 1.269 | 735 | **263** |

Diễn giải:

- **12.432 cặp attribute** trước đây bị coi là lệch số nhưng thực chất khớp sau
  khi quy đổi đơn vị (`144,5cm` ↔ `144.5厘米`, `1kg` ↔ `1000克`).
- **836 cặp** có mismatch đại lượng thật (strong) — nên kiểm tra tay hoặc hạ weight.
- Title ít đại lượng hơn (47%) và mismatch strong rất ít (47 cặp).

Ví dụ strong mismatch thật:

```text
"Vali 20inch ... 24inch"  vs  "行李箱20寸 ... 26寸 ... 28寸"
  -> target thêm 26/28 inch, source không có
```

## 5. Cách dùng trong pipeline

### Pair table

```python
from cado_core.quantity_norm import compare

result = compare(pair.source_vi, pair.target_zh, "vi", "zh")
pair.qty_match = result["match"]
pair.qty_strong_mismatch = result["strong_mismatch"]
pair.qty_missing = result["missing"]
pair.qty_extra = result["extra"]
```

### Xuất weak rows

```bash
PYTHONPATH=. python tools/export_weak_quantities.py
# data/cleaned/quantity_weak_rows.csv  1.434 dòng (UTF-8 BOM)
```

Cột: `dataset, id, source_text, target_text, qty_src_all, qty_tgt_all,
missing, extra, reason`.

`reason`:

```text
one_side_only          : đơn vị chỉ xuất hiện một bên (1.324 dòng)
both_sides_diff_dims   : hai bên có thứ nguyên khác nhau (110 dòng)
```

Đây là nhóm **không** bị cách cũ bắt và **không** phải strong mismatch —
phần lớn là khác cách diễn đạt (`38` vs `38厘米`, `một tháng` vs `一个月`),
cần review tay trước khi quyết định weight.

### Weight

```text
strong_mismatch  -> weight 0.0 (loại khỏi train) hoặc 0.3
weak mismatch    -> weight 0.7
match            -> weight 1.0
```

### Length prior

Khi tính `expected_zh_chars`, tách riêng phần số/đơn vị:

```text
length_ratio = (zh_chars - qty_chars_zh) / (vi_chars - qty_chars_vi)
```

Tránh việc `1.234` (6 ký tự) làm lệch tỉ lệ so với `1234` (4 ký tự).

## 6. Hạn chế đã biết

| hạn chế | ảnh hưởng | xử lý |
|---|---|---|
| `寸` vừa là inch vừa là 3,33cm | hiếm | map inch (thương mại) |
| `1m2` (1,2m) bị đọc thành m² | furniture | gắn cờ weak |
| Số đếm tiếng Việt (`một`, `ba`) | "ba lô" = ba lô, không phải 3 | **tắt** |
| `5G` (mạng) vs 5 gram | đồ điện tử | gắn cờ weak |
| `一件代发` | phrase TMĐT | đã loại |
| Mã sản phẩm dạng lạ | hiếm | heuristic token |

## 7. Test

```bash
PYTHONPATH=. python -m pytest tests/test_quantity_norm.py -q
# 32 passed
```

Bao gồm: đổi đơn vị, dấu thập phân, phân số, số Hán, range, mã sản phẩm,
`尺寸`/`件套`, strict mode bỏ count/year.

## 8. Việc tiếp theo

1. Thêm `qty_*` vào pair table builder.
2. Dùng `strong_mismatch` làm hard filter, `weak` làm weight 0,7.
3. Chạy lại Pha 0 với dữ liệu đã chuẩn hoá — token share có thể đổi nhẹ do
   lọc bớt cặp lỗi.
