# Similarity Trung - Việt với BAAI/bge-m3

Tính độ tương đồng ngữ nghĩa giữa văn bản **tiếng Trung** và **tiếng Việt**
bằng mô hình embedding đa ngôn ngữ [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3).

`bge-m3` hỗ trợ 100+ ngôn ngữ (gồm Trung & Việt) và trả về dense vector
1024 chiều, có thể kết hợp thêm sparse (lexical) và colbert.

## Cài đặt

```bash
# Cách 1: sentence-transformers (đơn giản, chỉ dense) — khuyến nghị bắt đầu
pip install -r requirements.txt

# Cách 2: thư viện chính thức FlagEmbedding (dense + sparse + colbert)
pip install FlagEmbedding
```

Lần chạy đầu tiên, mô hình (~2.2GB) sẽ được tải tự động từ HuggingFace.
Có thể đặt biến môi trường để dùng model local:

```bash
export BGE_M3_MODEL_PATH=/path/to/bge-m3
```

## Chạy

```bash
# Bản sentence-transformers
python similarity_bge_m3.py

# Bản FlagEmbedding (hybrid)
python similarity_bge_m3_flagembedding.py
```

## Sử dụng nhanh (Python)

```python
from similarity_bge_m3 import BgeM3Similarity

engine = BgeM3Similarity()

# 1. Similarity một cặp Trung - Việt
score = engine.similarity("我爱越南", "Tôi yêu Việt Nam")
print(score)  # ~0.7 - 0.9

# 2. So khớp chéo nhiều câu
zh = ["我爱越南", "今天天气很好"]
vi = ["Tôi yêu Việt Nam", "Hôm nay trời mưa"]
matrix = engine.similarity_matrix(zh, vi)  # shape (2, 2)

# 3. Tìm top-k câu tiếng Việt giống nhất
top = engine.top_k_vi("我预订了一个酒店房间", vi, k=2)
```

## Ghi chú

- Embedding đã được chuẩn hoá L2 (`normalize_embeddings=True`), nên
  **cosine similarity = dot product**; giá trị nằm trong khoảng `[-1, 1]`.
- Câu "dịch tương đương" thường cho điểm ~0.7–0.9; câu khác nghĩa cho điểm thấp.
- Nếu muốn chính xác cao hơn cho từ khoá/cụm ngắn, dùng bản FlagEmbedding với
  `hybrid_similarity()` (kết hợp dense + sparse + colbert).
