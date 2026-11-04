# Hướng dẫn dùng Gemma trên Google AI Studio để chấm bản dịch

Mục tiêu: dùng Gemma làm **judge** chấm chất lượng bản dịch của Qwen3-4B trên 2 tập
(1688 và Tiki), song song với COMET.

## 1. Truy cập

1. Mở https://aistudio.google.com/
2. Đăng nhập bằng tài khoản Google.
3. Ở cột trái chọn **Chat** (hoặc **Create prompt** nếu muốn lưu template).
4. Trong ô chọn model, chọn dòng **Gemma**:
   - `Gemma 3 27B` — chất lượng cao nhất, chậm hơn.
   - `Gemma 3 12B` — cân bằng.
   - `Gemma 3 4B` — nhanh, dùng để chấm số lượng lớn.

Không cần bật billing cho Gemma trong AI Studio (có hạn mức miễn phí).

## 2. Chuẩn bị dữ liệu

Mỗi dòng là một JSON object:

```json
{"id": "1688:123:title", "source": "Áo thun nam cotton", "hypothesis": "男士棉质T恤", "reference": "男士纯棉T恤"}
```

Với Tiki không có bản dịch vàng, bỏ `reference`:

```json
{"id": "tiki:456:title", "source": "Áo thun nam cotton", "hypothesis": "男士棉质T恤"}
```

Khuyến nghị:

```text
- 200 dòng/dataset (100 title + 100 attribute) cho lần chấm đầu
- chia batch 25–50 dòng/lần để tránh vượt context và dễ kiểm soát
- file: data/cleaned/judge/judge_input.jsonl
```

## 3. Prompt chấm điểm

Dán prompt sau vào AI Studio, thay `{{DATA}}` bằng batch JSONL:

```text
You are a strict bilingual evaluator for Vietnamese-to-Simplified-Chinese
e-commerce translation.

For each item, score the hypothesis on three axes (1-5):
- adequacy: does it preserve the meaning of the source?
- fluency: is the Chinese natural and grammatical?
- terminology: are brand names, numbers, units and product terms correct?

Rules:
- Use the reference when provided; otherwise judge against the source only.
- Penalize untranslated Vietnamese, missing numbers/units, and hallucinated content.
- Do not reward literal word-for-word copying.

Return ONLY a JSON array, one object per input, with this schema:
{"id": "...", "adequacy": 1-5, "fluency": 1-5, "terminology": 1-5, "note": "<=20 words"}

Data:
{{DATA}}
```

Ví dụ output:

```json
[
  {"id": "1688:123:title", "adequacy": 5, "fluency": 4, "terminology": 5, "note": "ok"},
  {"id": "1688:124:title", "adequacy": 2, "fluency": 3, "terminology": 1, "note": "missing 500ml"}
]
```

## 4. Cách dán dữ liệu

### Cách A — dán trực tiếp (đơn giản)

1. Mở file `judge_input.jsonl`, copy 25–50 dòng.
2. Dán vào cuối prompt ở vị trí `{{DATA}}`.
3. Enter, đợi Gemma trả JSON.
4. Copy JSON, lưu thành `judge_output_batch01.json`.

Lặp cho các batch tiếp theo.

### Cách B — dùng API (nhanh hơn, nếu có API key)

1. Vào https://aistudio.google.com/app/apikey → **Create API key**.
2. Cài SDK:

```bash
pip install -q google-genai
```

3. Script:

```python
from google import genai
import json

client = genai.Client(api_key="YOUR_KEY")
rows = [json.loads(l) for l in open("judge_input.jsonl")][:50]
prompt = open("judge_prompt.txt").read().replace("{{DATA}}", json.dumps(rows, ensure_ascii=False))

resp = client.models.generate_content(
    model="gemma-3-27b-it",
    contents=prompt,
)
open("judge_output_batch01.json", "w").write(resp.text)
```

4. Lặp cho các batch còn lại.

## 5. Parse và tổng hợp

Script `tools/parse_gemma_judge.py`:

```bash
python tools/parse_gemma_judge.py \
  --inputs judge_output_batch*.json \
  --out data/cleaned/judge/gemma_scores.json
```

Script sẽ:

- đọc tất cả file output, gộp theo `id`
- tính trung bình `adequacy`, `fluency`, `terminology`
- xuất bảng theo dataset (1688/Tiki) và pair_type (title/attribute)
- báo số dòng thiếu / lỗi JSON

## 6. Mẹo để kết quả ổn định

```text
- Chấm 2 lần với 2 model khác nhau (Gemma 3 27B và 12B), so sánh
- Giữ temperature thấp (0.0–0.2) nếu API cho phép
- Không đưa quá 50 dòng/batch: Gemma dễ bỏ dòng khi batch dài
- Luôn kiểm tra số object trả về == số dòng gửi đi
- Với Tiki (không reference), thêm tiêu chí "không được để tiếng Việt"
```

## 7. So sánh với COMET

```text
COMET (1688): reference-based, wmt22-comet-da
COMET (Tiki): reference-free, wmt22-cometkiwi-da
Gemma judge: 1-5, chấm cả 2 tập trên cùng mẫu
```

Báo cáo cuối nên có:

| hệ | COMET 1688 | COMET-QE Tiki | Gemma adequacy | Gemma fluency | Gemma terminology |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | | | | | |
| EcomGPT-7B | | | | | |
| OpenNMT | | | | | |
| Moses | | | | | |

## 8. Lưu ý

- Gemma 3 27B trong AI Studio có thể giới hạn số request/ngày.
- Nếu bị giới hạn, dùng Gemma 3 12B cho phần lớn và 27B cho 50 dòng kiểm tra chéo.
- Kết quả Gemma chỉ là **signal phụ**; quyết định cuối dựa trên COMET + test pass-rate.
