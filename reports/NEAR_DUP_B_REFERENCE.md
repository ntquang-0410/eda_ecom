# Near-duplicate text-level: B ↔ reference (1688)

Chuẩn hoá: NFKC, bỏ dấu câu/khoảng trắng, giữ CJK + chữ + số; đơn vị so sánh là **char 3-gram**.
Khử thiên lệch độ dài: data-level chỉ so trong dải 0.6–1.6× độ dài reference; output-level trừ baseline hoán vị reference trong cùng stratum (200 lần).

## 1. Data-level — target train so với reference

| Tập | n target | Jaccard mean | J ≥0,3 | J ≥0,5 | J ≥0,7 | J ≥0,95 | J = 1 | Containment mean | C ≥0,5 | C ≥0,8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 86824 | 0.203 | 19.7% | 9.1% | 3.5% | 1.0% | 1.0% | 0.374 | 31.3% | 7.1% |
| B | 49153 | 0.225 | 26.3% | 8.6% | 1.5% | 0.0% | 0.0% | 0.403 | 35.4% | 3.5% |

### 1b. Data-level theo stratum — chỉ so target cùng nhóm

| Tập | Title J | Title ≥0,5 | Attribute J | Attribute ≥0,5 | Description J | Description ≥0,5 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.231 | 5.6% | 0.271 | 20.6% | — | — |
| B | 0.229 | 4.4% | — | — | 0.405 | 28.9% |

### Cặp gần nhau nhất trong B (theo Jaccard, đã lọc độ dài)

| pair_type | Jaccard | Containment | reference | target gần nhất | group |
|---|---:|---:|---|---|---|
| title | 0.857 | 1.000 | 氨基酸山茶花洁面慕斯洗面奶深层清洁细腻毛孔卸妆控油泡沫洁面乳 | 氨基酸山茶花洁面慕斯洗面奶深层清洁细腻毛孔控油泡沫洁面乳卸妆 | 1688_title |
| title | 0.800 | 0.857 | 230克秋季新疆精梳全棉重磅圆领纯色长袖男t恤宽松打底衫厂家批发 | 230克秋季新疆精梳全棉重磅圆领纯色长袖男t恤宽松打底衫一件代发 | 1688_title |
| title | 0.727 | 0.889 | HERORANGE~低饱和淡颜口红多用棒水光感滋润显白自然镜面持妆唇膏 | HERORANGE低饱和淡颜口红多用棒水光感滋润显白自然持妆镜面唇膏 | 1688_title |
| description | 0.666 | 0.785 | 鞋面材质: 人造革; 内里材质: pu; 品牌: 其他; 款式: 单鞋; 鞋垫材质: 合成革; 鞋底材质: 橡胶; 功能 | 鞋面材质PU内里材质PU品牌其他款式勃肯鞋鞋垫材质合成革鞋底材质橡胶功能轻便鞋头形状圆头风格简约风流行元素休闲货源类别现 | 1688_description |
| description | 0.653 | 0.768 | 材质: 牛津布; 功能: 防水; 容纳电脑尺寸: 16英寸; 品牌: 其他; 背负系统: 弧形肩带; 里料质地: 涤纶; | 材质牛津布功能防水容纳电脑尺寸14英寸品牌其他背负系统弧形肩带里料质地涤纶皮质特征其它颜色黑色灰色适用性别中性男女均可适 | 1688_description |
| description | 0.609 | 0.835 | 主面料成分: 棉; 品牌: 多米牛; 面料名称: 针织布; 厚薄: 普通; 适合年龄段: 婴幼童(1~3岁,80~100 | 主面料成分棉品牌洁森棉面料名称棉厚薄普通适合年龄段婴幼童13岁80100cm款式长爬袖长长袖是否连帽不连帽安全等级A类无 | 1688_description |

## 2. Output-level — hypothesis ↔ reference của chính nó

| Cell | J | J base | J vượt | C | C base | C vượt | C≥0,8 | C≥0,95 | Hyp/Ref | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B A | 0.378 | 0.028 | +0.350 | 0.484 | 0.037 | +0.446 | 14.1% | 13.1% | 0.95 | 198 |
| Qwen3-8B B | 0.433 | 0.027 | +0.406 | 0.541 | 0.036 | +0.505 | 24.2% | 12.6% | 0.95 | 198 |
| Qwen3-4B A | 0.370 | 0.027 | +0.343 | 0.479 | 0.036 | +0.442 | 15.2% | 12.1% | 0.93 | 198 |
| Qwen3-4B B | 0.416 | 0.026 | +0.389 | 0.523 | 0.036 | +0.487 | 24.7% | 9.6% | 0.96 | 198 |
| Qwen3-4B A-balanced | 0.346 | 0.026 | +0.321 | 0.455 | 0.036 | +0.419 | 13.1% | 10.1% | 0.96 | 198 |
| Qwen3-4B B-natural | 0.397 | 0.028 | +0.369 | 0.494 | 0.037 | +0.457 | 24.7% | 11.1% | 0.99 | 198 |
| EcomGPT-7B A | 0.299 | 0.027 | +0.272 | 0.384 | 0.037 | +0.347 | 13.1% | 11.6% | 0.97 | 198 |
| EcomGPT-7B B | 0.227 | 0.026 | +0.201 | 0.304 | 0.036 | +0.268 | 9.6% | 8.6% | 0.98 | 198 |
| Moses A | 0.302 | 0.028 | +0.274 | 0.413 | 0.037 | +0.376 | 13.6% | 11.6% | 1.03 | 198 |
| Moses B | 0.370 | 0.026 | +0.344 | 0.485 | 0.036 | +0.449 | 20.7% | 12.1% | 1.06 | 198 |
| NLLB-600M A | 0.137 | 0.027 | +0.110 | 0.175 | 0.037 | +0.138 | 6.6% | 6.1% | 1.59 | 198 |
| NLLB-600M B | 0.105 | 0.027 | +0.078 | 0.153 | 0.036 | +0.117 | 6.6% | 5.6% | 1.30 | 198 |

### 2b. Theo stratum (J vượt baseline)

| Cell | Title | Attribute | Description |
|---|---:|---:|---:|
| Qwen3-8B A | +0.245 | +0.583 | +0.358 |
| Qwen3-8B B | +0.266 | +0.511 | +0.658 |
| Qwen3-4B A | +0.241 | +0.574 | +0.342 |
| Qwen3-4B B | +0.255 | +0.490 | +0.631 |
| Qwen3-4B A-balanced | +0.248 | +0.516 | +0.305 |
| Qwen3-4B B-natural | +0.201 | +0.481 | +0.673 |
| EcomGPT-7B A | +0.184 | +0.523 | +0.217 |
| EcomGPT-7B B | +0.195 | +0.420 | +0.022 |
| Moses A | +0.211 | +0.517 | +0.184 |
| Moses B | +0.196 | +0.472 | +0.576 |
| NLLB-600M A | +0.060 | +0.298 | +0.064 |
| NLLB-600M B | +0.043 | +0.230 | +0.055 |

## 3. Kết luận

**1. Không có bằng chứng B là near-duplicate của reference.** Ở stratum so sánh được (title), Jaccard data-level của A = 0.231 và B = 0.229 — trùng nhau. Ở ngưỡng chặt, B còn *ít* hơn A: J ≥0,7 là 1.5% (B) so với 3.5% (A); containment ≥0,8 là 3.5% so với 7.1%. Near-dup gần nguyên văn (J ≥0,95) chỉ chiếm 0.0% (B) và 1.0% (A) — tức A: 2/198, B: 0/198 reference; do catalogue 1688 có listing trùng (cùng sản phẩm đăng lại dưới product_id khác), ảnh hưởng cả A lẫn B.
**2. Chênh lệch trung bình nhỏ (+0.022 Jaccard) đến từ khác biệt trường dữ liệu, không phải copy.** A chỉ có title+attribute (attribute J = 0.271, ≥0,5: 20.6%), B chỉ có title+description (description J = 0.405, ≥0,5: 28.9%). Hai tập phủ hai stratum khác nhau nên không so trực tiếp được; mức của B nằm trong cùng bậc với attribute của A.
**3. Ở output-level, lợi thế của B tập trung ở description.** Qwen3-4B: J vượt baseline của description là +0.631 (B) so với +0.342 (A), trong khi title gần như nhau (+0.255 so với +0.241). Đây là hiệu ứng *đúng trường huấn luyện* (B học description nên sinh description giống reference hơn), không phải copy.
**4. Không có copy nguyên văn.** Tỷ lệ reference được phủ ≥95% (C≥0,95) không cao hơn ở B: 9.6% so với 12.1% (Qwen3-4B). Vì vậy BLEU(B) cao chủ yếu do trùng thuật ngữ/boilerplate + đúng trường dữ liệu, không phải gian lận dữ liệu.

**Hệ quả cho báo cáo:** có thể giữ BLEU(B) trong bảng, nhưng phải chú thích rằng BLEU cao của B là *field-matched* (B chứa description, eval có 45/198 reference là description), không phải bằng chứng chất lượng dịch tốt hơn. COMET là thước đo chính; BLEU chỉ dùng để đối chiếu bề mặt.
- Lưu ý kỹ thuật: 33/231 reference bị loại vì ngắn <6 ký tự sau chuẩn hoá (title 1–2 chữ Hán); các title ngắn làm containment bão hoà ở ngưỡng cao, nên dùng Jaccard đã lọc độ dài làm metric chính.
