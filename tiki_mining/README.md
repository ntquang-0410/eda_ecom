# Tiki EDA — chuẩn bị back-translation VI → ZH

Module `cado_core/tiki.py` chạy EDA trên **toàn bộ** Tiki và xuất **một parquet
product-level** (1 dòng/sản phẩm). Bước này không gọi model, không dịch; người
dùng có thể mang parquet lên Kaggle/Colab để back-translation.

## File đã publish

- 🤗 Hugging Face: [`ntquang0410/zh-vie_ecom/data/cleaned/tiki/tiki_eda.parquet`](https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/blob/main/data/cleaned/tiki/tiki_eda.parquet)
- Kích thước: **63.558.766 bytes** (~60,6 MiB), **13.200 dòng × 108 cột**
- SHA256: `7af2a2620b8ca2def3b80762cb8f99196bfe7c64baeaeaa8b99d974df570d3bd`
- Manifest toàn bộ cleaned: [`data/cleaned/README.md`](https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/blob/main/data/cleaned/README.md)

## Chạy

```bash
python tiki_mining/tiki_eda.py
# hoặc
python run_pipeline.py tiki-eda
```

Mặc định:

- input: `data/cleaned/tiki_vi_cado.parquet`
- output: `data/cleaned/tiki_eda.parquet`

Test nhanh:

```bash
python tiki_mining/tiki_eda.py --limit 200 --output /tmp/opencode/tiki_eda_test.parquet
```

Yêu cầu thêm `pyvi>=0.1.1` (MIT) để lấy lexicon tiếng Việt. Nếu thiếu pyvi,
script vẫn chạy nhưng tỉ lệ `vi_rate` / `foreign_rate` kém chính xác hơn.

## Grain & quy ước

- **1 dòng = 1 `product_id`**, giữ nguyên toàn bộ cột gốc.
- `bt_title`: title sạch, lấy từ `title_vi` gốc (không dùng
  `title_vi_cleaned` của CADO-VI vì bản đó đã xoá nhầm brand/tên sản phẩm).
- `bt_prose`: prose sạch từ `description_prose_vi`, đã bỏ boilerplate Tiki,
  URL/email/hotline, hashtag, emoji, dòng kẻ. Nếu title lặp lại ở đầu prose và
  đóng vai trò dòng tiêu đề, phần echo được bỏ (flag `title_echo_removed`).
- `bt_sentences`: list câu sạch, tối đa 60 từ/câu; `bt_sentence_freqs` là tần
  suất toàn corpus của từng câu (cùng thứ tự). Dùng để lọc câu template.
- `bt_attributes_text`: các cặp `name: value` đã làm sạch, lấy từ
  `attributes_vi` (fallback `description_vi`).
- `protected_spans`: số/đơn vị, model code, brand — nên giữ nguyên khi dịch.
- `overall_route`:
  - `BT_TITLE+PROSE`: dịch cả title và prose
  - `BT_TITLE`: chỉ title đủ tốt
  - `BT_PROSE`: chỉ prose đủ tốt
  - `BT_ATTR`: chỉ còn attributes
  - `REVIEW` / `DROP`: cần xem lại / bỏ
- `bt_ready` = title **hoặc** prose đủ tốt. `bt_ready_attrs` tách riêng.
- `title_score`, `prose_score`, `quality_score` là điểm heuristic 0–1, không
  phải xác suất. `quality_score` kết hợp title + prose + attributes.
- `eda_flags` / `eda_flags_text`: danh sách cờ EDA. Không xoá dòng nào; người
  dùng tự quyết định filter.
- `split`: train/dev/test 80/10/10 theo hash của `product_id + title + prose`,
  giữ các cặp trùng khít trong cùng split.
- `attributes_vi_json`: bản JSON của attributes để dễ đọc/portable.

## Gợi ý dùng trên Kaggle

```python
import pandas as pd

df = pd.read_parquet("tiki_eda.parquet")

# Title
titles = df.loc[df.bt_ready_title, ["product_id", "bt_title"]].copy()

# Prose theo từng câu
sentences = (
    df.loc[df.bt_ready_prose, ["product_id", "bt_sentences", "bt_sentence_freqs"]]
      .explode(["bt_sentences", "bt_sentence_freqs"])
      .rename(columns={"bt_sentences": "text_vi", "bt_sentence_freqs": "freq"})
)
sentences = sentences[sentences.text_vi.str.split().str.len() >= 4]
sentences = sentences[sentences.freq < 50]  # bỏ câu template lặp nhiều

# Attributes
attributes = df.loc[df.bt_ready_attrs, ["product_id", "bt_attributes_text"]].copy()
```

Khi ghép kết quả dịch, giữ `product_id` và (nếu dịch theo câu) thêm index câu
trước khi `explode` để merge lại chính xác.

## Ghi chú EDA

- Tiki là corpus **đơn ngữ VI**: `title_zh` và `description_zh` rỗng 100%,
  nên không chấm được cosine cặp ZH–VI.
- Boilerplate pháp lý Tiki xuất hiện ở 13.199/13.200 dòng và được bỏ.
- `description_vi` là chuỗi `key: value; ...`; prose thật nằm ở
  `description_prose_vi`.
- Cờ `foreign_high` chỉ cảnh báo khi title có >35% token ngoại lai hoặc prose
  có >15%; brand/model/đơn vị được bảo vệ nên không bị tính là nhiễu.
