# Pha 1 — kết quả thực nghiệm weight (Qwen3-4B, left padding)

5 cấu hình, 5M token/run, LoRA r=16, eval 2.000 Tiki test + 1.998 1688 test.

| cấu hình | Tiki pass | Tiki qty | untranslated | len ratio | 1688 chrF | phút |
|---|---:|---:|---:|---:|---:|---:|
| natural | 0.9630 | 0.9755 | 0.15% | 0.44 | 0.5707 | 8.2 |
| balanced | 0.9605 | 0.9815 | 0.55% | 0.46 | 0.5664 | 11.5 |
| no_desc | 0.9655 | 0.9815 | 0.45% | 0.45 | 0.4905 | 15.3 |
| tiki_heavy | 0.9610 | 0.9805 | 0.45% | 0.46 | 0.5512 | 13.7 |
| title_heavy | 0.9530 | 0.9785 | 0.75% | 0.46 | 0.5733 | 14.6 |

## Nhận xét

```text
mọi cấu hình đều ~95–97% Tiki pass — chênh lệch nằm trong nhiễu (CI ±0,9%)
no_desc tệ nhất ở 1688 chrF (0,4905) — bỏ description mất 0,08 chrF
title_heavy có chrF cao nhất nhưng Tiki pass thấp nhất
natural: pass cao nhất, untranslated thấp nhất, chrF gần cao nhất
balanced: qty match cao nhất (0,9815), chrF 0,5664
```

## Đề xuất

```text
Mặc định: natural (token-proportional) — đơn giản, kết quả tổng hợp tốt nhất
Nếu ưu tiên quantity: balanced (+0,6% qty, -0,25% pass)
Loại: no_desc (chrF -0,08), title_heavy (pass -1,0%)
tiki_heavy: trung gian, không có lợi thế rõ
```

## Lưu ý

Kết quả v1 (right padding) bị lệch hoàn toàn: pass 0,50–0,74. Sau khi sửa
padding, pass tăng lên 0,95–0,97 — xác nhận lỗi eval cũ.
