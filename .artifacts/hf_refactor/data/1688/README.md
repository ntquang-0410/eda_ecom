# data/1688/ — train/dev/test 1688

Dữ liệu song ngữ 1688 (Việt → Trung) đã dedup/chunk, dùng cho track 1688-only và so sánh 4 model.

| file | rows | ghi chú |
|---|---:|---|
| `train.parquet` | 80.811 | |
| `dev.parquet` | 5.254 | |
| `test.parquet` | 5.306 | |
| `test_small.parquet` | 1.000 | 500 title + 500 attribute, **không có description** |

Kèm bản text và tokenized cho OpenNMT/Moses:

```text
{train,dev,test,test_small}.vi / .zh
{train,dev,test,test_small}.tok.vi / .tok.zh
```

- `.vi` / `.zh`: text thô theo dòng.
- `.tok.*`: đã tokenize, chỉ dùng cho OpenNMT/Moses; không cần cho HuggingFace models.

Split theo `product_id`. `test_small` không có description — muốn đánh giá description
phải dùng `data/weight/eval/eval_1688.parquet`.
