# Audit empty hypothesis — Qwen3-4B A-natural trên 27 dòng bổ sung của 858

**Câu hỏi đặt ra:** "qwen4b_a 27/27 rỗng" — có đúng cell này rỗng toàn bộ 27 dòng không?

**Kết luận: KHÔNG.** Chỉ **9/27 rỗng**, và 9 dòng đó là **toàn bộ 9 dòng description** của nhóm bổ sung; 18/18 dòng attribute đều có hyp.

## 1. Số liệu

| Nhóm | Số dòng | Rỗng | Tỉ lệ |
|---|---:|---:|---:|
| Attribute | 18 | 0 | 0% |
| Description | 9 | **9** | **100%** |
| Tổng | 27 | 9 | 33,3% |

Cùng pattern ở cả 3 seed của `qwen4b_a` (42, 123, 7) và ở `qwen4b_a_balanced`; các cell Qwen khác cũng rỗng đúng 9 dòng description này.

## 2. Nguyên nhân: prompt vượt `max_seq`, bị harness loại trước khi generate

`generate_rows()` trong `tools/tpu_weight_train.py` có guard:

```python
part = part[part["prompt_tokens"] < max_seq - 8]   # max_seq = 960 → ngưỡng 952
```

9 dòng description có prompt (chat template Qwen + source) dài **971–1.196 token**, vượt ngưỡng 952 nên bị loại **trước khi gọi model** — không phải model sinh hỏng.

| eval_id | stratum | src chars | prompt tokens | Kết quả |
|---|---|---:|---:|---|
| tikiv2:00023 | description | 3.262 | 1.059 | bị loại |
| tikiv2:00025 | description | 3.090 | 971 | bị loại |
| tikiv2:00029 | description | 3.331 | 1.067 | bị loại |
| tikiv2:00057 | description | 3.168 | 1.034 | bị loại |
| tikiv2:00069 | description | 3.663 | 1.103 | bị loại |
| tikiv2:00072 | description | 3.320 | 1.064 | bị loại |
| tikiv2:00101 | description | 3.553 | 1.196 | bị loại |
| tikiv2:00125 | description | 3.013 | 1.078 | bị loại |
| tikiv2:00129 | description | 3.109 | 1.108 | bị loại |

Đối chứng: 18 dòng attribute có prompt 57–532 token → tất cả đều có hyp.

## 3. Hệ quả

1. Zero-fill 9 dòng này là **đúng policy** (không lọc), nhưng phải hiểu đây là **giới hạn của harness**
   (`max_seq = 960`), không phải "Qwen không dịch được description dài".
2. Khi báo cáo coverage Tiki của Qwen: 618/627 = 98,6%; trong đó 9 dòng thiếu đều là description dài.
3. EcomGPT/Moses/NLLB **không** bị lọc ở các dòng này (pipeline khác, không dùng guard theo token) nên
   chúng có hyp — và bị điểm rất thấp (QE −0,21…−0,77). Vì vậy so sánh zero-fill giữa Qwen và các model
   khác trên 9 dòng này là **không công bằng cho Qwen**, cần ghi chú khi diễn giải.
4. Nếu muốn đo đúng năng lực dịch description dài: chạy inference riêng với `max_seq = 1280` cho 9 dòng
   (Qwen3 hỗ trợ 32k) rồi báo cáo như một **phụ lục độ dài**, không trộn vào bảng chính.

## 4. Tái lập

```bash
python - <<'PY'
import pandas as pd
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
# tokenize source theo đúng chat template của tools/tpu_weight_train.py
PY
```

Artefact liên quan: `.artifacts/canonical/qwen4b_a/seed42.parquet` (cột `nonempty`), `reports/AUDIT_858.md`.
