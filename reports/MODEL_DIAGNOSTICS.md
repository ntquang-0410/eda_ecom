# Chẩn đoán lỗi theo model — matrix 5×2

Sinh từ `eval_F` (1.500 dòng) + hypothesis + score COMET. Các chỉ số: `rep4` = tỉ lệ 4-gram lặp trong câu (càng cao càng máy móc); `vi%` = % ký tự Latin có dấu (dấu hiệu sót tiếng Việt); `digit`/`latin` = recall số/thương hiệu từ nguồn.

## 1. Tổng quan

| model | COMET 1688 | COMET-QE Tiki | BLEU | chrF | rỗng % | <5 ký tự % | rep4 | vi % | digit | latin | hyp unique % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ecomgpt_a | 0.8214 | 0.0102 | 26.37 | 27.43 | 0.0 | 11.5 | 0.062 | 0.1 | 0.88 | 0.16 | 99.4 |
| ecomgpt_b | 0.7661 | -0.1455 | 14.39 | 15.20 | 1.4 | 11.8 | 0.060 | 0.1 | 0.72 | 0.15 | 97.1 |
| moses_a | 0.7947 | -0.2401 | 33.96 | 36.03 | 0.0 | 9.2 | 0.036 | 1.0 | 0.95 | 0.26 | 99.8 |
| moses_b | 0.8103 | -0.1466 | 67.82 | 65.86 | 0.0 | 8.4 | 0.037 | 0.9 | 0.95 | 0.24 | 99.8 |
| onmt_a | 0.3581 | -1.3508 | 0.45 | 1.79 | 0.0 | 15.5 | 0.519 | 4.3 | 0.12 | 0.08 | 71.5 |
| onmt_b | 0.3424 | -1.3909 | 2.05 | 3.08 | 0.0 | 0.1 | 0.389 | 0.0 | 0.00 | 0.00 | 60.1 |
| qwen4b_a | 0.8552 | 0.0647 | 41.66 | 41.68 | 1.0 | 12.7 | 0.042 | 0.1 | 0.92 | 0.16 | 98.9 |
| qwen4b_b | 0.8480 | 0.0570 | 68.77 | 67.36 | 1.0 | 12.0 | 0.044 | 0.1 | 0.93 | 0.15 | 98.8 |
| qwen_a | 0.8560 | 0.0635 | 55.19 | 54.02 | 1.0 | 12.4 | 0.046 | 0.1 | 0.92 | 0.16 | 98.7 |
| qwen_b | 0.8552 | 0.0687 | 70.95 | 69.29 | 1.0 | 11.6 | 0.046 | 0.1 | 0.93 | 0.17 | 98.9 |

## 2. Kiểu lỗi

| model | rep4>0,3 % | len/ref<0,5 % (1688) | len/ref>2 % (1688) | sót tiếng Việt % | mất số % | tần suất hyp top-1 % |
|---|---:|---:|---:|---:|---:|---:|
| ecomgpt_a | 6.5 | 3.3 | 2.3 | 0.5 | 26.0 | 0.2 |
| ecomgpt_b | 6.3 | 8.8 | 5.2 | 0.7 | 41.4 | 1.4 |
| moses_a | 3.4 | 0.4 | 2.3 | 24.0 | 12.8 | 0.2 |
| moses_b | 3.1 | 0.3 | 3.3 | 14.1 | 12.6 | 0.1 |
| onmt_a | 60.2 | 12.3 | 21.2 | 14.7 | 96.9 | 6.0 |
| onmt_b | 53.6 | 0.3 | 39.2 | 0.0 | 100.0 | 4.0 |
| qwen4b_a | 4.2 | 1.6 | 0.7 | 1.1 | 17.0 | 1.0 |
| qwen4b_b | 4.2 | 1.3 | 1.6 | 1.1 | 16.3 | 1.0 |
| qwen_a | 4.5 | 1.9 | 1.3 | 1.1 | 16.9 | 1.0 |
| qwen_b | 4.7 | 1.5 | 2.3 | 1.7 | 16.1 | 1.0 |

## 3. Theo stratum

| model | dataset | stratum | n | COMET | rỗng % | med len | len/ref | rep4 | Hán % | vi % | digit | latin |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ecomgpt_a | 1688 | title | 300 | 0.808 | 0.0 | 30 | 0.94 | 0.012 | 88.1 | 0.0 | 0.90 | 0.03 |
| ecomgpt_a | 1688 | attribute | 300 | 0.852 | 0.0 | 10 | 1.00 | 0.083 | 57.3 | 0.0 | 0.84 | 0.21 |
| ecomgpt_a | 1688 | description | 150 | 0.788 | 0.0 | 328 | 0.89 | 0.210 | 61.5 | 0.0 | 0.80 | 0.05 |
| ecomgpt_a | tiki | title | 300 | -0.057 | 0.0 | 31 | nan | 0.022 | 47.0 | 0.1 | 0.97 | 0.26 |
| ecomgpt_a | tiki | attribute | 300 | 0.104 | 0.0 | 8 | nan | 0.009 | 55.7 | 0.2 | 0.95 | 0.26 |
| ecomgpt_a | tiki | description | 150 | -0.042 | 0.0 | 316 | nan | 0.155 | 70.7 | 0.0 | 0.75 | 0.07 |
| ecomgpt_b | 1688 | title | 300 | 0.817 | 0.0 | 30 | 0.97 | 0.010 | 87.5 | 0.0 | 0.90 | 0.04 |
| ecomgpt_b | 1688 | attribute | 300 | 0.820 | 0.0 | 9 | 1.00 | 0.082 | 56.7 | 0.0 | 0.89 | 0.21 |
| ecomgpt_b | 1688 | description | 150 | 0.557 | 0.0 | 320 | 0.72 | 0.141 | 61.2 | 0.2 | 0.36 | 0.02 |
| ecomgpt_b | tiki | title | 300 | -0.038 | 0.0 | 33 | nan | 0.021 | 45.4 | 0.1 | 0.97 | 0.29 |
| ecomgpt_b | tiki | attribute | 300 | -0.136 | 7.0 | 8 | nan | 0.011 | 56.1 | 0.1 | 0.64 | 0.17 |
| ecomgpt_b | tiki | description | 150 | -0.377 | 0.0 | 512 | nan | 0.214 | 67.0 | 0.0 | 0.43 | 0.03 |
| moses_a | 1688 | title | 300 | 0.797 | 0.0 | 33 | 1.06 | 0.007 | 86.0 | 0.2 | 0.89 | 0.05 |
| moses_a | 1688 | attribute | 300 | 0.843 | 0.0 | 11 | 1.00 | 0.057 | 53.6 | 0.6 | 0.90 | 0.26 |
| moses_a | 1688 | description | 150 | 0.694 | 0.0 | 471 | 1.14 | 0.102 | 54.9 | 1.5 | 0.97 | 0.26 |
| moses_a | tiki | title | 300 | -0.208 | 0.0 | 37 | nan | 0.014 | 42.0 | 1.1 | 0.98 | 0.33 |
| moses_a | tiki | attribute | 300 | 0.006 | 0.0 | 10 | nan | 0.007 | 52.9 | 1.1 | 0.96 | 0.30 |
| moses_a | tiki | description | 150 | -0.797 | 0.0 | 562 | nan | 0.085 | 60.6 | 2.3 | 0.96 | 0.50 |
| moses_b | 1688 | title | 300 | 0.790 | 0.0 | 34 | 1.09 | 0.007 | 85.5 | 0.2 | 0.88 | 0.05 |
| moses_b | 1688 | attribute | 300 | 0.817 | 0.0 | 11 | 1.10 | 0.059 | 52.0 | 1.1 | 0.93 | 0.27 |
| moses_b | 1688 | description | 150 | 0.836 | 0.0 | 528 | 1.24 | 0.103 | 58.8 | 0.6 | 0.96 | 0.17 |
| moses_b | tiki | title | 300 | -0.180 | 0.0 | 38 | nan | 0.014 | 42.2 | 0.9 | 0.97 | 0.34 |
| moses_b | tiki | attribute | 300 | 0.020 | 0.0 | 11 | nan | 0.007 | 48.9 | 1.9 | 0.98 | 0.34 |
| moses_b | tiki | description | 150 | -0.415 | 0.0 | 600 | nan | 0.092 | 64.5 | 0.8 | 0.96 | 0.26 |
| onmt_a | 1688 | title | 300 | 0.269 | 0.0 | 50 | 1.62 | 0.751 | 62.8 | 3.4 | 0.01 | 0.02 |
| onmt_a | 1688 | attribute | 300 | 0.463 | 0.0 | 8 | 1.30 | 0.339 | 29.3 | 6.1 | 0.07 | 0.12 |
| onmt_a | 1688 | description | 150 | 0.328 | 0.0 | 679 | 1.68 | 0.928 | 36.5 | 0.1 | 0.16 | 0.00 |
| onmt_a | tiki | title | 300 | -1.545 | 0.0 | 23 | nan | 0.434 | 28.9 | 6.0 | 0.07 | 0.05 |
| onmt_a | tiki | attribute | 300 | -0.892 | 0.0 | 6 | nan | 0.246 | 28.7 | 6.1 | 0.11 | 0.21 |
| onmt_a | tiki | description | 150 | -1.880 | 0.0 | 507 | nan | 0.718 | 52.0 | 0.0 | 0.28 | 0.00 |
| onmt_b | 1688 | title | 300 | 0.283 | 0.0 | 59 | 1.87 | 0.392 | 80.4 | 0.0 | 0.00 | 0.00 |
| onmt_b | 1688 | attribute | 300 | 0.386 | 0.0 | 18 | 2.15 | 0.239 | 66.6 | 0.0 | 0.00 | 0.00 |
| onmt_b | 1688 | description | 150 | 0.374 | 0.0 | 737 | 1.76 | 0.961 | 61.7 | 0.0 | 0.00 | 0.00 |
| onmt_b | tiki | title | 300 | -1.452 | 0.0 | 39 | nan | 0.259 | 73.7 | 0.0 | 0.00 | 0.00 |
| onmt_b | tiki | attribute | 300 | -1.237 | 0.0 | 18 | nan | 0.128 | 68.3 | 0.0 | 0.00 | 0.00 |
| onmt_b | tiki | description | 150 | -1.577 | 0.0 | 644 | nan | 0.899 | 81.8 | 0.0 | 0.00 | 0.00 |
| qwen4b_a | 1688 | title | 300 | 0.831 | 0.0 | 30 | 0.97 | 0.008 | 89.6 | 0.0 | 0.90 | 0.03 |
| qwen4b_a | 1688 | attribute | 300 | 0.872 | 0.0 | 10 | 1.00 | 0.076 | 58.4 | 0.0 | 0.86 | 0.20 |
| qwen4b_a | 1688 | description | 150 | 0.870 | 2.0 | 367 | 0.89 | 0.121 | 60.7 | 0.0 | 0.93 | 0.05 |
| qwen4b_a | tiki | title | 300 | -0.025 | 0.0 | 33 | nan | 0.017 | 45.8 | 0.2 | 0.97 | 0.29 |
| qwen4b_a | tiki | attribute | 300 | 0.078 | 0.0 | 8 | nan | 0.007 | 60.0 | 0.0 | 0.96 | 0.21 |
| qwen4b_a | tiki | description | 150 | 0.230 | 8.0 | 454 | nan | 0.087 | 69.5 | 0.5 | 0.87 | 0.11 |
| qwen4b_b | 1688 | title | 300 | 0.827 | 0.0 | 30 | 0.97 | 0.007 | 89.3 | 0.0 | 0.90 | 0.03 |
| qwen4b_b | 1688 | attribute | 300 | 0.839 | 0.0 | 9 | 1.00 | 0.075 | 58.1 | 0.1 | 0.91 | 0.20 |
| qwen4b_b | 1688 | description | 150 | 0.909 | 2.0 | 408 | 1.00 | 0.131 | 61.8 | 0.0 | 0.92 | 0.04 |
| qwen4b_b | tiki | title | 300 | -0.048 | 0.0 | 33 | nan | 0.019 | 47.1 | 0.2 | 0.97 | 0.28 |
| qwen4b_b | tiki | attribute | 300 | 0.075 | 0.0 | 9 | nan | 0.010 | 61.5 | 0.0 | 0.95 | 0.19 |
| qwen4b_b | tiki | description | 150 | 0.245 | 8.0 | 448 | nan | 0.088 | 69.2 | 0.5 | 0.87 | 0.12 |
| qwen_a | 1688 | title | 300 | 0.835 | 0.0 | 29 | 0.97 | 0.009 | 89.5 | 0.0 | 0.89 | 0.03 |
| qwen_a | 1688 | attribute | 300 | 0.877 | 0.0 | 10 | 1.00 | 0.076 | 58.0 | 0.0 | 0.84 | 0.20 |
| qwen_a | 1688 | description | 150 | 0.855 | 2.0 | 354 | 0.84 | 0.133 | 60.8 | 0.0 | 0.92 | 0.04 |
| qwen_a | tiki | title | 300 | -0.021 | 0.0 | 33 | nan | 0.019 | 45.8 | 0.2 | 0.98 | 0.29 |
| qwen_a | tiki | attribute | 300 | 0.076 | 0.0 | 8 | nan | 0.007 | 59.4 | 0.0 | 0.97 | 0.21 |
| qwen_a | tiki | description | 150 | 0.220 | 8.0 | 430 | nan | 0.102 | 70.9 | 0.1 | 0.86 | 0.09 |
| qwen_b | 1688 | title | 300 | 0.836 | 0.0 | 30 | 0.97 | 0.007 | 89.3 | 0.0 | 0.90 | 0.03 |
| qwen_b | 1688 | attribute | 300 | 0.849 | 0.0 | 10 | 1.00 | 0.075 | 56.0 | 0.3 | 0.91 | 0.23 |
| qwen_b | 1688 | description | 150 | 0.908 | 2.0 | 410 | 1.00 | 0.134 | 62.1 | 0.0 | 0.92 | 0.04 |
| qwen_b | tiki | title | 300 | -0.037 | 0.0 | 33 | nan | 0.019 | 45.3 | 0.2 | 0.97 | 0.29 |
| qwen_b | tiki | attribute | 300 | 0.107 | 0.0 | 8 | nan | 0.007 | 58.4 | 0.2 | 0.96 | 0.23 |
| qwen_b | tiki | description | 150 | 0.215 | 8.0 | 463 | nan | 0.107 | 70.0 | 0.2 | 0.86 | 0.11 |

## 4. Khoảng trống stratum bị bỏ khỏi training

A chỉ có title+attribute; B chỉ có title+description. `gap` = COMET trung bình các stratum seen − COMET stratum unseen; gap âm nghĩa là stratum bị bỏ lại tốt hơn.

| model | dataset | stratum unseen | COMET seen | COMET unseen | gap |
|---|---|---|---:|---:|---:|
| ecomgpt_a | 1688 | description | 0.830 | 0.788 | +0.042 |
| ecomgpt_a | tiki | description | 0.023 | -0.042 | +0.065 |
| ecomgpt_b | 1688 | attribute | 0.687 | 0.820 | -0.133 |
| ecomgpt_b | tiki | attribute | -0.208 | -0.136 | -0.072 |
| moses_a | 1688 | description | 0.820 | 0.694 | +0.126 |
| moses_a | tiki | description | -0.101 | -0.797 | +0.696 |
| moses_b | 1688 | attribute | 0.813 | 0.817 | -0.004 |
| moses_b | tiki | attribute | -0.297 | 0.020 | -0.317 |
| onmt_a | 1688 | description | 0.366 | 0.328 | +0.038 |
| onmt_a | tiki | description | -1.219 | -1.880 | +0.661 |
| onmt_b | 1688 | attribute | 0.329 | 0.386 | -0.057 |
| onmt_b | tiki | attribute | -1.515 | -1.237 | -0.278 |
| qwen4b_a | 1688 | description | 0.852 | 0.870 | -0.019 |
| qwen4b_a | tiki | description | 0.027 | 0.230 | -0.203 |
| qwen4b_b | 1688 | attribute | 0.868 | 0.839 | +0.028 |
| qwen4b_b | tiki | attribute | 0.099 | 0.075 | +0.024 |
| qwen_a | 1688 | description | 0.856 | 0.855 | +0.001 |
| qwen_a | tiki | description | 0.028 | 0.220 | -0.192 |
| qwen_b | 1688 | attribute | 0.872 | 0.849 | +0.023 |
| qwen_b | tiki | attribute | 0.089 | 0.107 | -0.017 |

## 5. BLEU/chrF theo stratum (1688)

| model | stratum | n | BLEU | chrF |
|---|---|---:|---:|---:|
| ecomgpt_a | title | 300 | 33.47 | 31.92 |
| ecomgpt_a | attribute | 300 | 40.03 | 46.73 |
| ecomgpt_a | description | 150 | 22.97 | 23.43 |
| ecomgpt_b | title | 300 | 36.86 | 34.30 |
| ecomgpt_b | attribute | 300 | 21.82 | 31.39 |
| ecomgpt_b | description | 150 | 9.23 | 9.06 |
| moses_a | title | 300 | 40.40 | 37.13 |
| moses_a | attribute | 300 | 51.92 | 55.62 |
| moses_a | description | 150 | 29.75 | 32.46 |
| moses_b | title | 300 | 38.81 | 35.98 |
| moses_b | attribute | 300 | 38.21 | 44.98 |
| moses_b | description | 150 | 75.97 | 74.07 |
| onmt_a | title | 300 | 0.20 | 0.87 |
| onmt_a | attribute | 300 | 1.85 | 3.59 |
| onmt_a | description | 150 | 0.20 | 1.64 |
| onmt_b | title | 300 | 0.04 | 0.35 |
| onmt_b | attribute | 300 | 0.03 | 1.33 |
| onmt_b | description | 150 | 2.78 | 3.89 |
| qwen4b_a | title | 300 | 42.03 | 38.60 |
| qwen4b_a | attribute | 300 | 51.29 | 55.90 |
| qwen4b_a | description | 147 | 39.96 | 39.73 |
| qwen4b_b | title | 300 | 42.79 | 39.12 |
| qwen4b_b | attribute | 300 | 38.60 | 44.78 |
| qwen4b_b | description | 147 | 77.47 | 75.73 |
| qwen_a | title | 300 | 44.76 | 40.77 |
| qwen_a | attribute | 300 | 50.25 | 54.87 |
| qwen_a | description | 147 | 57.65 | 56.00 |
| qwen_b | title | 300 | 46.02 | 41.84 |
| qwen_b | attribute | 300 | 40.88 | 46.10 |
| qwen_b | description | 147 | 78.43 | 77.60 |

## 6. Ví dụ lỗi nặng nhất (1688, COMET thấp nhất)

### ecomgpt_a — COMET 0.281 (description)

- **Nguồn:** Chất liệu: Nhung sữa; : Roees/; Công nghệ dệt: In và nhuộm; Kiểu dáng: Kiểu bảng; Quy trình in và nhuộm: In và nhuộm hoạt tính; Trọng lượng: 2500g (bao gồm)-300
- **Tham chiếu:** 材质: 牛奶绒; 品牌: Roees/罗布斯; 织造工艺: 印染; 款式: 床单式; 印染工艺: 活性印染; 克重: 2500g(含)-3000g(不含); 风格: 现代简约风; 货号: 2024-08-16-1-LX; 加工定制: 否; 图案: 条纹, 纯色, 心形, 星空, 几何图案, 彩虹, 卡通动漫, 植物花卉
- **Hypothesis:** 材质：牛奶绒 ; : Roes/ ; 印染工艺：活性印染 ; 款式：方格 ; 印染工艺：活性印染 ; 重量：2500g(含)-3000g(不含) ; 风格：现代简约 ; 面单号：2024-08-16-1-LX ; 加工定制：无 ; : , , , ; : , , , ; : , , , ; : , , , ; : , 

### ecomgpt_b — COMET 0.194 (description)

- **Nguồn:** Chất liệu: Nhựa; Thương hiệu: 11; Có nhập khẩu không: Không.1; Quốc gia/Khu vực xuất xứ: Chiết Giang; Số mặt hàng: MC; In LOGO: Có1; Gia công tùy chỉnh: Có2; Th
- **Tham chiếu:** 材质: 塑料; 品牌: 1; 是否进口: 否; 原产国/地区: 浙江; 货号: MC; 加印LOGO: 是; 加工定制: 是; 产品上市时间: 2025; 价格段: 10-20元; 产地: 台州; 是否有专利: 无; 风格: 日韩风; 颜色: 高瓶款大号4层-月光白[热销款式], 高瓶款大号3层-月光白[热销款式], 
- **Hypothesis:** iPad 5/6 (9.7')

### moses_a — COMET 0.499 (description)

- **Nguồn:** Vật liệu: bông1; thương hiệu: kho báu; Số lượng vải: 40 que; Quá trình dệt: Rửa bằng nước, viền, vải chéo, chải; hình dạng: ga trải giường; Quá trình in và nhuộ
- **Tham chiếu:** 材质: 棉; 品牌: 招财宝; 面料支数: 40支; 织造工艺: 水洗, 包边, 斜纹, 磨毛; 款式: 床单式; 印染工艺: 活性印染; 克重: 2500g(含)-3000g(不含); 风格: 现代简约风; 货号: 水洗棉-四件套; 加工定制: 是; 图案: 纯色, 几何图案; 花型: 追/布鲁斯-灰, 追/迪亚斯-
- **Hypothesis:** 材质： 棉 ; 品牌： 宝布 ; 大号： 40 支 ; 工编织： 水洗， 边 ; 斜纹磨形： 床单款 ; 印花工艺染： 活性印染 ; 重量克： 2500g ( 含 ) -3000g ( 专供 ) ;： 现代简约风 ; 产品编号： 水洗棉四件套 ; 加工定制款， ;：： 纯色 , 几何图案款 ;： Chase/Xanh-

### moses_b — COMET 0.497 (description)

- **Nguồn:** Vật liệu: bông1; thương hiệu: kho báu; Số lượng vải: 40 que; Quá trình dệt: Rửa bằng nước, viền, vải chéo, chải; hình dạng: ga trải giường; Quá trình in và nhuộ
- **Tham chiếu:** 材质: 棉; 品牌: 招财宝; 面料支数: 40支; 织造工艺: 水洗, 包边, 斜纹, 磨毛; 款式: 床单式; 印染工艺: 活性印染; 克重: 2500g(含)-3000g(不含); 风格: 现代简约风; 货号: 水洗棉-四件套; 加工定制: 是; 图案: 纯色, 几何图案; 花型: 追/布鲁斯-灰, 追/迪亚斯-
- **Hypothesis:** 材质 : 棉 ; 品牌 : 招财宝 ; 面料支数 : 40 支 ; 织造工艺 : 水洗 , 包边 , 斜纹 , 磨毛 ; 款式 : 床单式 ; 印染工程： 反应印染 ; 克重 : 2500g ( 含 ) -3000g ( 不含 ) ; 风格 : 现代简约风 ; 货号 : 水洗棉四件套 ; 加工定制 : 是 ; 图案 :

### onmt_a — COMET 0.115 (description)

- **Nguồn:** Chất liệu mặt giày: Vải lưới; Chất liệu bên trong: Vải; Thương hiệu: -; Kiểu dáng: Giày lưới1; Chất liệu lót giày: EVA; Chất liệu đế giày: Nhựa; Chức năng: Thoá
- **Tham chiếu:** 鞋面材质: 网层布面; 内里材质: 布; 品牌: -; 款式: 网面鞋; 鞋垫材质: EVA; 鞋底材质: 塑胶; 功能: 透气, 耐磨; 鞋头形状: 圆头; 风格: 休闲风; 流行元素: 交叉绑带; 货源类别: 现货; 产地: 河南 焦作; 适用性别: 女; 鞋跟高度: 低跟(1-3CM); 图案: 拼色; 适合季节
- **Hypothesis:** 件套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套套

### onmt_b — COMET 0.284 (description)

- **Nguồn:** Chất liệu: nylon1; Chức năng: không thấm nước; Kết cấu lót: nylon2; Đặc điểm vỏ não: da thuộc rau; Chứa vừa kích thước máy tính.: 11inch; công suất: Dưới 20L; G
- **Tham chiếu:** 材质: 尼龙; 功能: 防水; 里料质地: 尼龙; 皮质特征: 植鞣皮; 容纳电脑尺寸: 11英寸; 容量: 20L以下; 适用性别: 中性/男女均可; 品牌: 无; 颜色: 黄色笑脸, 藏蓝拼玫红, 粉蓝叶子, 岩石灰, 万花筒, 黑色字母, 玫瑰金, 金属海军蓝, 绿烫银, 银灰色, 金属银色, 三色银拼, 金属银
- **Hypothesis:** 材质 : 否 ; 品牌 : 否 ; 品牌 : 否 ; 品牌 : 否 ; 品牌 : 否 ; 货号 : 否 ; 货号 : 否 ; 货号 : 否 ; 货号 : 否 ; 货源 : 否 ; 货号 : 否 ; 货号 : 否 ; 货号 : 否 ; 货号 : 否 ; 货号 : 否 ; 货号 : 否 ; 货号 : 否 ; 货号 : 否 

### qwen4b_a — COMET 0.710 (description)

- **Nguồn:** Chất liệu: Nhung sữa; : Roees/; Công nghệ dệt: In và nhuộm; Kiểu dáng: Kiểu bảng; Quy trình in và nhuộm: In và nhuộm hoạt tính; Trọng lượng: 2500g (bao gồm)-300
- **Tham chiếu:** 材质: 牛奶绒; 品牌: Roees/罗布斯; 织造工艺: 印染; 款式: 床单式; 印染工艺: 活性印染; 克重: 2500g(含)-3000g(不含); 风格: 现代简约风; 货号: 2024-08-16-1-LX; 加工定制: 否; 图案: 条纹, 纯色, 心形, 星空, 几何图案, 彩虹, 卡通动漫, 植物花卉
- **Hypothesis:** 材质：牛奶绒 ; : 萌趣/ ; 制造工艺：印花染色 ; 型号：床单款 ; 印花染色工艺：活性印染 ; 重量：2500g(含)-3000g(不含) ; 风格：现代简约风 ; 品号：2024-08-16-1-LX ; 定制加工：否 ; : , , , , , , , ; : , , , , , , , , , , , ,

### qwen4b_b — COMET 0.753 (description)

- **Nguồn:** Vật liệu: Chất liệu cotton giặt thân thiện với da; thương hiệu: Không có thương hiệu (không có nhãn hiệu đã đăng ký); Số lượng vải: Khác; Quá trình dệt: Có thể 
- **Tham chiếu:** 材质: 亲肤水洗棉; 品牌: 无品牌(无注册商标); 面料支数: 其它; 织造工艺: 水洗, 斜纹; 款式: 床单式; 印染工艺: 活性印染; 克重: 2000g(含)~2500g(不含); 风格: 极简风, ins风; 货号: ycj -A类母婴级; 加工定制: 否; 图案: 纯色; 花型: 月蓝灰[裸睡触感温柔呵护
- **Hypothesis:** 材质: 亲肤水洗棉; 品牌: 无品牌(无注册商标); 面料支数: 其他; 织造工艺: 可水洗, 斜纹; 形状: 片型; 印染工艺: 活性印染; 克重: 2000g(含)~2500g(专供); 风格: 简约风, 风格1; 货号: ycj-母婴级A类; 加工定制: 否; 图案: 纯色; 花型: 月光蓝灰[裸睡轻柔护理], 

### qwen_a — COMET 0.206 (description)

- **Nguồn:** Thành phần vải chính: Bông; : ; Kiểu dáng: Chui đầu; Thương hiệu: Ameligo; Công nghệ: Sợi quang; Phù hợp với đám đông: Thanh niên; Thể loại: Áo phông áo ba lỗ; 
- **Tham chiếu:** 主面料成分: 棉; 面料名称: 索罗娜; 款式: 套头; 品牌: 阿美利哥; 工艺: 丝光; 适合人群: 青年; 类别: T恤背心; 风格: 轻熟休闲; 货号: STA11-2; 图案: 纯色; 版型: 直筒型; 货源类别: 现货; 淘货类别: 时尚都市(24-35岁); 领型: 圆领; 袖长: 短袖; 是否连帽: 不
- **Hypothesis:** 主成分面料:棉 ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : ; : 

### qwen_b — COMET 0.395 (description)

- **Nguồn:** Chất liệu: Nhung sữa; : Roees/; Công nghệ dệt: In và nhuộm; Kiểu dáng: Kiểu bảng; Quy trình in và nhuộm: In và nhuộm hoạt tính; Trọng lượng: 2500g (bao gồm)-300
- **Tham chiếu:** 材质: 牛奶绒; 品牌: Roees/罗布斯; 织造工艺: 印染; 款式: 床单式; 印染工艺: 活性印染; 克重: 2500g(含)-3000g(不含); 风格: 现代简约风; 货号: 2024-08-16-1-LX; 加工定制: 否; 图案: 条纹, 纯色, 心形, 星空, 几何图案, 彩虹, 卡通动漫, 植物花卉
- **Hypothesis:** 材质: 牛奶绒; 面料: 罗纱/纱; 织造工艺: 印染; 款式: 板式; 印染工艺: 活性印染; 克重: 2500g(含)-3000g(不含); 风格: 现代简约风; 货号: 2024-08-16-1-LX; 加工定制: 否; 颜色: 灰色, 米白, 米黄, 米蓝, 米粉, 米绿, 米紫, 米橙, 米粉, 米粉, 米粉

