# Final 4B — so sánh với baseline 4M

Final: recipe A+B, **8M token / 1.000 step**, seed 42.

Baseline: 4M token / 500 step, seed 42/123. Khác budget nên chỉ tham khảo.

## COMET theo stratum

| stratum | final | n |
|---|---:|---:|
| 1688 title | 0.8405 | 300 |
| 1688 attribute | 0.8758 | 300 |
| 1688 description | 0.9078 | 147 |
| tiki title | -0.0310 | 300 |
| tiki attribute | 0.0894 | 300 |
| tiki description | 0.2245 | 138 |

## Final − natural_4M_seed42

| stratum | Δ | CI 95% | n |
|---|---:|---|---:|
| 1688 title | +0.0107 | [+0.0061; +0.0153] | 300 |
| 1688 attribute | -0.0068 | [-0.0132; -0.0008] | 300 |
| tiki title | +0.0014 | [-0.0191; +0.0226] | 300 |
| tiki description | -0.0253 | [-0.0422; -0.0098] | 138 |

Toàn bộ: **-0.0018** CI [-0.0086; +0.0050], n=1038

## Final − natural_4M_seed123

| stratum | Δ | CI 95% | n |
|---|---:|---|---:|
| 1688 title | +0.0135 | [+0.0083; +0.0191] | 300 |
| 1688 attribute | -0.0039 | [-0.0102; +0.0020] | 300 |
| tiki title | +0.0052 | [-0.0178; +0.0298] | 300 |
| tiki description | -0.0190 | [-0.0381; -0.0013] | 138 |

Toàn bộ: **+0.0017** CI [-0.0056; +0.0094], n=1038

## Final − balanced_4M_seed42

| stratum | Δ | CI 95% | n |
|---|---:|---|---:|
| 1688 title | +0.0086 | [+0.0039; +0.0138] | 300 |
| 1688 attribute | +0.0101 | [+0.0030; +0.0176] | 300 |
| tiki title | -0.0003 | [-0.0145; +0.0134] | 300 |
| tiki description | -0.0221 | [-0.0406; -0.0055] | 138 |

Toàn bộ: **+0.0024** CI [-0.0028; +0.0077], n=1038

## Final − balanced_4M_seed123

| stratum | Δ | CI 95% | n |
|---|---:|---|---:|
| tiki title | +0.0170 | [-0.0034; +0.0403] | 300 |
| tiki description | -0.0161 | [-0.0339; +0.0006] | 138 |

Toàn bộ: **+0.0066** CI [-0.0084; +0.0225], n=438

## Đọc kết quả

- Gain chỉ có ý nghĩa nếu CI không chứa 0 ở stratum tương ứng.
- Final train 8M gấp đôi baseline 4M: chênh lệch dương có thể do budget, không chỉ recipe.
- Muốn tách recipe khỏi budget cần natural 8M/1.000 step (chưa chạy).


## Kết luận

- Final 8M/1.000 **không vượt baseline overall**: Δ so natural 4M là -0,0018 (seed 42) / +0,0017 (seed 123), CI đều chứa 0.
- Điểm mạnh: 1688 title tăng có ý nghĩa (+0,011…+0,014).
- Điểm yếu: tiki description giảm -0,019…-0,025 (CI không chứa 0), do share tiki description bị giảm từ 32% (natural) xuống 20,3%.
- 1688 attribute giảm nhẹ so natural (-0,007) nhưng tăng so balanced (+0,010) — phụ thuộc baseline.

**Khuyến nghị:** không dùng final 8M làm model production. Giữ recipe 4M đã được xác nhận
(natural cho A, balanced cho B), hoặc nếu muốn tách ảnh hưởng budget thì chạy lại đúng recipe
final ở 4M/500 step để so sánh công bằng.
