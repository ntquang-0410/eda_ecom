# Audit & làm sạch ký tự nhiễu

Mục tiêu: loại ký tự không mang nghĩa dịch, có thể làm nhiễu kiến thức model học.

## Còn lại sau làm sạch

| nhóm | bản chất | xử lý |
|---|---|---|
| symbols (+ - × ÷ ¥) | kích thước, toán tử, tiền tệ | giữ |
| mixed_script (iPhone手机壳) | brand + Hán | giữ |
| fullwidth_punct (，。：) | dấu câu Hán chuẩn | giữ |
| long_run (11111111, DL888…) | mã sản phẩm/serial | giữ |
| cjk_ext | chữ Hán mở rộng hiếm | giữ |

## Đã loại / chuẩn hoá

| nhóm | trước | sau |
|---|---:|---:|
| 【】〖〗〔〕丨 | 53.050 【 + 52.910 】 + 489 丨 (1688 attr) | 0 |
| （） full-width | 31.260 (1688 attr_zh) | 0 (đã thành ASCII) |
| emoji/★ | 612–617 dòng attr | 0 |
| PUA (icon font) | 1 dòng (Tiki) | 0 |
| template {var} | 54 attr + 15 Tiki source | 0 (giữ nội dung) |
| HTML tag/entity thật | 8–11 description | 0 |
| URL | 2 description | 0 |
| ngoặc lẻ | 1.291 attr_zh, 46 Tiki source | 0 |
| repeated punctuation | 200 Tiki source | 0 (trừ dấu …) |
| control/zero-width/bidi/variation | 0 phát hiện | 0 |

## Quy tắc chuẩn hoá

```text
NFC
chữ/số full-width -> ASCII
【】〖〗〔〕（）[] -> ASCII
bỏ control, zero-width, bidi, variation selector, PUA, emoji
bỏ HTML tag + entity chuẩn (giữ brand dạng End&start;)
bỏ URL, template {var} (giữ nội dung bên trong)
gộp dấu câu lặp (giữ "…")
bỏ ngoặc lẻ không có cặp
```

Code: `cado_core/text.py::clean_noise_characters`.

## Áp dụng

```text
1688 : title, description, attribute name + value (VI và ZH)
Tiki : source_text_bt, target_text_bt, key_vi, key_zh_effective
       ở full.parquet, train.parquet, units.parquet
```

`raw_output` giữ nguyên để audit.

## Tác động

```text
1688 alignment: 90,21% sản phẩm align đủ attribute (không đổi đáng kể)
qty_status    : phân bố gần như không đổi
HF            : đã upload lại, 31 file, size khớp
```