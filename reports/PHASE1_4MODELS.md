# Pha 1 — fine-tune 4 model (natural weights)

Cùng harness v2, 5M token/model, LoRA r=16, left padding.
Eval: 2.000 Tiki test + 1.998 1688 test mỗi model.

| model | Tiki pass | 1688 chrF | 1688 qty | steps | phút |
|---|---:|---:|---:|---:|---:|
| qwen3-0.6b | 0.8370 | 0.4266 | 0.7983 | 9,597 | 21.6 |
| qwen3-1.7b | 0.9025 | 0.5268 | 0.8043 | 4,853 | 21.5 |
| qwen3-4b | 0.9565 | 0.5960 | 0.8098 | 2,446 | 31.4 |
| qwen3-8b | 0.9305 | 0.6114 | 0.8048 | 1,208 | 37.9 |

## Kết luận

```text
4B thắng trên Tiki test: pass 0,9565, qty 0,8098
8B cao nhất trên 1688 chrF (0,6114) nhưng Tiki pass thấp hơn 4B (0,9305)
0.6B và 1.7B không cạnh tranh (0,837 / 0,9025)
=> baseline đề xuất: Qwen3-4B
```

## Lưu ý

- 8B có loss spike 0,42 ở step 800; nếu cần có thể chạy lại với LR thấp hơn.
- Test sample 2.000/4.445, CI ±0,9%: chênh 4B–8B (2,6%) vượt CI.
- Chưa lưu adapter nên không re-eval trên cùng tập pair_id; lần sau nên lưu adapter.
