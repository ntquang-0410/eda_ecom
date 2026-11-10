# Cluster bootstrap theo product_id

Paired bootstrap 4.000 lần, resample **sản phẩm** thay vì `eval_id`.
So sánh với CI theo `eval_id` để thấy mức độ hẹp giả tạo.

## Tập A

| config | dataset | Δ | CI 95% (cluster) | P(win) | n dòng | n sản phẩm |
|---|---|---:|---|---:|---:|---:|
| attr_heavy | 1688 | -0.0049 | [-0.0083; -0.0013] | 0.004 | 600 | 455 |
| attr_heavy | tiki | +0.0048 | [-0.0083; +0.0176] | 0.769 | 600 | 460 |
| attr_heavy | all | -0.0000 | [-0.0067; +0.0063] | 0.488 | 1200 | 915 |
| balanced | 1688 | -0.0074 | [-0.0124; -0.0025] | 0.002 | 600 | 455 |
| balanced | tiki | +0.0005 | [-0.0151; +0.0151] | 0.523 | 600 | 460 |
| balanced | all | -0.0035 | [-0.0113; +0.0044] | 0.202 | 1200 | 915 |
| title_heavy | 1688 | -0.0069 | [-0.0118; -0.0021] | 0.003 | 600 | 455 |
| title_heavy | tiki | +0.0033 | [-0.0098; +0.0159] | 0.689 | 600 | 460 |
| title_heavy | all | -0.0018 | [-0.0085; +0.0053] | 0.299 | 1200 | 915 |
| tiki_heavy | 1688 | -0.0155 | [-0.0205; -0.0109] | 0.000 | 600 | 455 |
| tiki_heavy | tiki | -0.0025 | [-0.0187; +0.0134] | 0.388 | 600 | 460 |
| tiki_heavy | all | -0.0090 | [-0.0175; -0.0004] | 0.021 | 1200 | 915 |

## Tập B

| config | dataset | Δ | CI 95% (cluster) | P(win) | n dòng | n sản phẩm |
|---|---|---:|---|---:|---:|---:|
| balanced | 1688 | +0.0144 | [+0.0081; +0.0213] | 1.000 | 447 | 375 |
| balanced | tiki | +0.0002 | [-0.0141; +0.0144] | 0.507 | 438 | 407 |
| balanced | all | +0.0074 | [-0.0009; +0.0155] | 0.959 | 885 | 782 |
| desc_heavy | 1688 | -0.0054 | [-0.0113; +0.0006] | 0.041 | 447 | 375 |
| desc_heavy | tiki | -0.0026 | [-0.0126; +0.0068] | 0.291 | 438 | 407 |
| desc_heavy | all | -0.0040 | [-0.0099; +0.0017] | 0.086 | 885 | 782 |
| title_heavy | 1688 | +0.0160 | [+0.0097; +0.0227] | 1.000 | 447 | 375 |
| title_heavy | tiki | -0.0034 | [-0.0173; +0.0105] | 0.307 | 438 | 407 |
| title_heavy | all | +0.0064 | [-0.0016; +0.0142] | 0.943 | 885 | 782 |
| tiki_heavy | 1688 | +0.0047 | [-0.0008; +0.0102] | 0.952 | 447 | 375 |
| tiki_heavy | tiki | -0.0045 | [-0.0211; +0.0113] | 0.295 | 438 | 407 |
| tiki_heavy | all | +0.0001 | [-0.0088; +0.0087] | 0.508 | 885 | 782 |

## B: balanced − title_heavy

Δ = +0.0010  CI [-0.0047; +0.0067]  P(win) = 0.619  (782 sản phẩm)

## Đọc kết quả

- Nếu CI cluster rộng hơn CI theo `eval_id`, kết luận trước đó đang lạc quan.
- Chỉ coi config thắng khi CI cluster không chứa 0 **và** P(win) ≥ 0,95.
- A/B dùng chung sản phẩm giữa các config nên đây vẫn là paired bootstrap.

