# Tiki BT v5 (prompt brand-aware)

Sinh lại toàn bộ 29.446 cache MT với prompt v5:
- brand-aware: chỉ giữ proper noun/brand/model code, danh từ Việt phải dịch
- gate loại `target == key_zh` (lỗi 31% ở 0.6B của v3)
- probe 80 dòng/model, retry bằng biến thể nhì

Mỗi model: `full.parquet` (29.446) + `train.parquet` (qc_status=ok) + `units.parquet` (45.709).
