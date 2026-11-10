# Kỹ thuật khảo sát trọng số cho 2 bộ dữ liệu A/B

**Mục đích:** thống nhất với nhóm cách tạo 2 bài toán, cách gán trọng số, cách train và cách kết luận — trước khi chạy thí nghiệm.

**Nguyên tắc:** chỉ thay **tỷ lệ token lấy từ mỗi nhóm**; giữ nguyên dữ liệu gốc, eval set, seed, token budget và số optimizer step.

---

## 1. Dữ liệu đầu vào: lấy trước EDA/Processing

Không dùng `data/cleaned/bt_tiki_desc/` vì file đó đã qua QC, dedup, chunk, clean boilerplate.

Nguồn dùng để chia A/B:

```text
A: data/cleaned/pairs_dedup/1688.parquet
   data/cleaned/pairs_dedup/tiki_qwen3-4b.parquet

B: thêm Tiki description từ raw BT 4B:
   bt_desc_v1/qwen3-4b/predictions.parquet
```

Tiki description lấy từ **raw predictions**, không lấy từ `full/train/units` đã xử lý.

Hệ quả: dữ liệu B còn boilerplate, dòng rỗng, dòng truncated — đây là dữ liệu thật trước EDA, không được lọc trước khi chia nhóm.

---

## 2. Hai bộ dữ liệu

### Tập A — title + attribute

```text
1688 title + 1688 attribute
Tiki title + Tiki attribute
```

4 nhóm:

```text
1688_title
1688_attribute
tiki_title
tiki_attribute
```

Thống kê hiện tại:

```text
rows           : 118.477 (đã dedup exact pair)
raw rows       : 118.688
effective token: 4,707M
eval           : 1.200 dòng
```

Phân bố natural:

| Nhóm | Token share |
|---|---:|
| 1688_attribute | 65,11% |
| 1688_title | 15,98% |
| tiki_title | 10,37% |
| tiki_attribute | 8,55% |

### Tập B — title + description

```text
1688 title + 1688 description
Tiki title + Tiki description
```

4 nhóm:

```text
1688_title
1688_description
tiki_title
tiki_description
```

Tập B đã đủ raw BT 4B:

```text
rows           : 59.209
effective token: 14,598M
eval           : 900 dòng
```

Phân bố natural:

| Nhóm | Token share |
|---|---:|
| 1688_description | 56,57% |
| tiki_description | 34,93% |
| 1688_title | 5,15% |
| tiki_title | 3,34% |

Description chiếm ~91,5% token; title chỉ ~8,5%.

---

## 3. Một dòng dữ liệu được hiểu như thế nào?

Mỗi dòng là một cặp dịch:

```text
source_text : tiếng Việt
target_text : tiếng Trung
tokens_est  : token ước lượng của cả cặp
group       : một trong 4 nhóm ở trên
```

### Ví dụ thật — tập A

**1688 title** `1688:592582928262:title`, **49,2 token**

```text
SRC: Bàn chải lốp, bàn chải cửa gió, bàn chải trung tâm, bàn chải động cơ, ...
TGT: 轮胎刷 风口刷 轮毂刷发动机刷细节刷缝隙刷洗车清洁小毛刷洗车刷
```

**1688 attribute** `1688:592582928262:attr:0`, **4,2 token**

```text
SRC: sợi nhỏ
TGT: 超细纤维
```

**Tiki title** `tiki:279688539:title`, **41,7 token**

```text
SRC: Đèn UV 20W-200W Diệt Tảo, Diệt Khuẩn Hồ Cá Cao Cấp - Sạch Nước Trong 7 Ngày
TGT: 20W-200W紫外线灯 杀藻 杀菌鱼缸高级 - 7天净水| 杀寄生藻TN2
```

**Tiki attribute** `tiki:279688539:attr:4:0`, **14,7 token**

```text
SRC: 11W, 26cm, bể <1000L
TGT: 11W，26厘米，水箱1000升
```

### Ví dụ thật — tập B

**1688 description** `1688:592582928262:desc:chunk:0`, **237,9 token**

```text
SRC: Chất liệu: sợi nhỏ; Thương hiệu: Perniz; Xuất xứ: Quảng Châu; ...
TGT: 材质: 超细纤维; 品牌: 柏尼士; 产地: 广州; ...
```

**Tiki description raw 4B** `tiki:278499721:desc`, **168,3 token**

```text
SRC: Bàn Phím Led có dây K1 mini nhỏ gọn ... Giá sản phẩm trên Tiki đã bao gồm thuế ...
TGT: 带线LED键盘K1 mini，小巧适用于电脑、笔记本 ... 产品价格已包含现行法律规定的税 ...
```

Dòng Tiki description này cho thấy đúng lý do phải giữ dữ liệu **trước EDA**: nó còn boilerplate Tiki ở cuối.

---

## 4. Kỹ thuật lấy mẫu

### 4.1. Đơn vị là token, không phải dòng

Một title 49 token và một attribute 4 token không đại diện cho cùng lượng học.

Vì vậy mỗi dòng có:

```text
eff_tokens = min(tokens_est, 960)
```

- `tokens_est`: token ước lượng ban đầu.
- `960`: trần token khi train; dòng dài hơn bị cắt khi encode.
- Dùng `eff_tokens` để lập kế hoạch sampling.

### 4.2. Quota-based sampling (bản đã sửa)

Không dùng xác suất `p_i ∝ w_g * t_i` nữa.

Lý do: nếu chọn dòng với xác suất tỷ lệ `t_i`, token thực tế của nhóm tỷ lệ với
`sum(t_i²)`, không phải `sum(t_i)`. Câu dài bị chọn quá nhiều, làm token share
thực tế lệch khỏi mục tiêu và phá phân bố độ dài.

Cách đúng:

```text
budget B = 4.000.000 token
với mỗi nhóm g:
    T_g = s_g * B
    pool_g = tất cả dòng thuộc nhóm g
    lặp:
        chọn đều ngẫu nhiên 1 dòng từ pool_g (có hoàn lại)
        cộng eff_tokens
    dừng khi tổng >= T_g
```

Sau đó gộp các nhóm, xáo trộn và chia thành 500 step ~8.000 token.

Điểm quan trọng:

- Trong mỗi nhóm, mọi dòng có xác suất **như nhau**.
- Câu ngắn vẫn xuất hiện với tần suất tự nhiên.
- Group token share đạt đúng `s_g` (sai số chỉ là overshoot của dòng cuối).
- Không còn hiện tượng câu dài `t_i²` áp đảo.

### 4.2b. Kiểm chứng trên dữ liệu thật

Với `natural`, quota 4M token cho A lấy 60.961 dòng `1688_attribute`,
13.329 `1688_title`, 12.777 `tiki_attribute`, 10.856 `tiki_title`.
Median `eff_tokens` của mẫu trùng khít quần thể (9,9 / 46,5 / 8,9 / 34,6),
tức phân bố độ dài được giữ nguyên, không bị dồn về câu dài.

### 4.3. Bằng chứng: sampler cũ lệch rất mạnh

Mô phỏng 4M token trên chính dữ liệu A/B:

| Tập | Config | Nhóm | Target | Sampler cũ | Quota mới |
|---|---|---|---:|---:|---:|
| A | natural | 1688_attribute | 65,1% | **88,6%** | 65,1% |
| A | natural | 1688_title | 16,0% | **3,3%** | 16,0% |
| A | natural | tiki_title | 10,4% | **2,1%** | 10,4% |
| A | natural | tiki_attribute | 8,6% | 6,1% | 8,6% |
| A | balanced | 1688_attribute | 25% | **53,9%** | 25% |
| A | balanced | 1688_title | 25% | **8,3%** | 25% |
| A | balanced | tiki_title | 25% | **7,9%** | 25% |
| A | balanced | tiki_attribute | 25% | 29,9% | 25% |
| B | natural | 1688_description | 56,6% | **65,3%** | 56,6% |
| B | natural | 1688_title | 5,2% | **0,5%** | 5,2% |
| B | natural | tiki_title | 3,3% | **0,3%** | 3,3% |
| B | natural | tiki_description | 34,9% | 34,0% | 34,9% |

Vì vậy toàn bộ kết quả weight cũ (5 cấu hình 4B) chỉ nên xem là **tham khảo**;
phải chạy lại bằng quota sampler mới.

### 4.3b. Ví dụ quota — tập A, `tiki_heavy`

Với budget 4M:

```text
1688_title      10% -> 400.000 token
1688_attribute  20% -> 800.000 token
tiki_title      30% -> 1.200.000 token
tiki_attribute  40% -> 1.600.000 token
```

Trong nhóm `tiki_attribute`, ta chọn đều các dòng (kể cả dòng 1 token) cho đến
khi đủ 1,6M token. Không nhân thêm `t_i`, nên dòng attribute ngắn không bị loại.

Với B `balanced`, mỗi nhóm nhận 1.000.000 token; title 49 token và description
400 token đều được chọn theo tần suất dòng thật, không theo `t_i`.

### 4.4. Vì sao cố định 500 step?

Nếu chỉ cố định 4M token mà để số step thay đổi:

- cấu hình nhiều title/attribute ngắn sẽ có nhiều batch hơn;
- model được update nhiều hơn;
- kết quả có thể tăng vì **nhiều step**, không phải vì weight.

Vì vậy mọi cấu hình dùng:

```text
500 optimizer step
8.000 token/step
cùng batch/accum
```

---

## 5. Trọng số tập A — title + attribute

| Tên | 1688 title | 1688 attr | Tiki title | Tiki attr | Câu hỏi |
|---|---:|---:|---:|---:|---|
| natural | token-proportional | token-proportional | token-proportional | token-proportional | baseline |
| balanced | 25 | 25 | 25 | 25 | cân bằng 4 nhóm |
| title_heavy | 40 | 20 | 25 | 15 | title có quan trọng hơn? |
| attr_heavy | 10 | 60 | 5 | 25 | attribute có quan trọng hơn? |
| tiki_heavy | 10 | 20 | 30 | 40 | Tiki có cần ưu tiên? |

### Cách đọc từng cấu hình A

**natural**
- Không nhân hệ số.
- 1688 attribute chiếm 65% token; title chỉ 16%.
- Dùng để trả lời: “phân bố tự nhiên đã đủ tốt chưa?”

**balanced**
- Mỗi nhóm 25%.
- So với natural: 1688 attr bị giảm hệ số ~0,38; Tiki title tăng ~2,41; Tiki attr tăng ~2,92.
- Dùng để kiểm tra: cân bằng 4 nhóm có tốt hơn để model học đều không?

**title_heavy**
- 1688 title 40%, Tiki title 25%, attr còn 20% và 15%.
- So với natural: 1688 title tăng ~2,5×; Tiki title tăng ~2,4×; 1688 attr giảm ~0,31×.
- Dùng để kiểm tra: tăng title có cải thiện title 1688/Tiki mà không phá attribute không?

**attr_heavy**
- 1688 attr 60%, Tiki attr 25%; title chỉ còn 10% và 5%.
- So với natural: 1688 attr ~0,92× (natural đã 65%), Tiki attr ~2,92×; tổng attribute tăng từ ~73,7% lên 85%.
- Dùng để kiểm tra: attribute có cần được bảo vệ/ưu tiên riêng không?

**tiki_heavy**
- Tiki title 30%, Tiki attr 40%; 1688 title 10%, attr 20%.
- So với natural: Tiki attr tăng ~4,67×; 1688 attr giảm ~0,31×.
- Dùng để kiểm tra: ưu tiên domain Tiki có tăng Tiki mà vẫn giữ 1688 không?

---

## 6. Trọng số tập B — title + description

Tập B chờ raw BT 4B đủ; bảng dưới là cấu hình đã chốt trước khi chạy.

| Tên | 1688 title | 1688 desc | Tiki title | Tiki desc | Câu hỏi |
|---|---:|---:|---:|---:|---|
| natural | token-proportional | token-proportional | token-proportional | token-proportional | baseline |
| balanced | 25 | 25 | 25 | 25 | cân bằng title/desc |
| title_heavy | 40 | 10 | 40 | 10 | title có quan trọng hơn? |
| desc_heavy | 2 | 55 | 3 | 40 | description có quan trọng hơn? |
| tiki_heavy | 10 | 15 | 35 | 40 | Tiki có cần ưu tiên? |

### Cách đọc từng cấu hình B

**natural**
- Description chiếm ~90% token; title rất ít.
- Dùng để kiểm tra: model có bị “chìm” trong description không?

**balanced**
- Mỗi nhóm 25%.
- Vì natural title rất ít, balanced thực tế **tăng title rất mạnh** (hệ số ~6–14× tuỳ nhóm).
- Dùng để kiểm tra: tăng title có làm model dịch title tốt hơn mà description không giảm quá 0,01 không?

**title_heavy**
- 1688 title 40%, Tiki title 40%; desc còn 10% mỗi bên.
- Dùng để kiểm tra: title là yếu tố quyết định chất lượng hay chỉ là phần phụ?

**desc_heavy**
- 1688 desc 55%, Tiki desc 40%; title chỉ còn 2% và 3% — tổng description 95%.
- Dùng để kiểm tra: đẩy gần như toàn bộ token vào description có cải thiện description thật không, và title có bị phá không?

**tiki_heavy**
- Tiki title 35%, Tiki desc 40%; 1688 title 10%, desc 15%.
- Dùng để kiểm tra: ưu tiên Tiki có tốt hơn cho domain đích không?

### Ví dụ B

Với `1688_title` 49,2 token, `1688_desc` 237,9 token, `tiki_desc` 168,3 token:

- natural: desc 1688 có trọng số lớn nhất (237,9), title nhỏ (49,2).
- balanced: title được nhân hệ số lớn (~6×) vì natural title share rất thấp; desc bị giảm hệ số.
- title_heavy: title được nhân mạnh hơn nữa; desc chỉ còn 10%.
- desc_heavy: desc 1688/Tiki được đẩy lên 95% tổng token; title gần như bị bỏ.

Đây là lý do không thể suy kết quả A sang B: **cùng tên “title_heavy” nhưng hệ số thực tế khác nhau** vì natural share của A và B khác nhau.

---

## 7. Huấn luyện

```text
model       : Qwen3-4B + LoRA
LoRA        : r=16, alpha=32, dropout=0.05, all-linear
precision   : bf16
max_seq     : 960 token
budget      : 4M token/run
steps       : 500 optimizer step
step tokens : ~8.000 token
batch       : theo VRAM, chia đều 8.000 token/step
seed        : 42 trước, 123 cho top cấu hình nếu còn quota
```

Điểm quan trọng:

- Mọi cấu hình A dùng **cùng 4 nhóm A**; không trộn description.
- Mọi cấu hình B dùng **cùng 4 nhóm B**; không trộn attribute.
- Không đổi prompt, eval set, tokenizer, LR, số step giữa các cấu hình.

---

## 8. Đánh giá

### Tập A

```text
1688 title, 1688 attribute : COMET-da
Tiki title, Tiki attribute : COMET-QE
macro A = trung bình 4 stratum
```

### Tập B

```text
1688 title, 1688 description : COMET-da
Tiki title, Tiki description : COMET-QE
macro B = trung bình 4 stratum
```

Tiki description **không có reference người**; chỉ dùng QE.

### Gemma

```text
model  : gemma-4-31b-it
scale  : AAA, AA, A, BBB, BB, B, CCC, CC, C
id     : blind, không có config/seed trong prompt
```

Gemma là tín hiệu phụ; ưu tiên COMET paired.

### Kiểm định

```text
paired bootstrap 2.000 lần theo eval_id
so sánh từng cấu hình với natural
CI 95% và P(win)
```

Quy tắc:

```text
1. primary = macro của tập A hoặc B
2. guardrail = không stratum nào giảm quá 0,01
3. nếu best - natural < 0,005 hoặc CI chứa 0 -> natural
4. nếu 2 seed lệch dấu -> chưa kết luận
```

---

## 9. Kiểm soát công bằng

Các điểm phải giữ cố định:

- cùng `product_id` split: train/dev/test không đổi;
- cùng eval `eval_id`;
- cùng prompt;
- cùng token budget;
- cùng số optimizer step;
- cùng seed;
- không dùng eval để chọn dòng train;
- không lọc quality theo kết quả eval (tránh leakage).

Khác biệt duy nhất giữa các run: **phân bố token theo 4 nhóm**.

---

## 10. Artefacts

```text
data/cleaned/weight_splits_v2/A_title_attribute.parquet
data/cleaned/weight_splits_v2/eval_A.parquet
data/cleaned/weight_splits_v2/meta_A.json

# B sẽ sinh sau khi raw BT 4B đủ:
data/cleaned/weight_splits_v2/B_title_description.parquet
data/cleaned/weight_splits_v2/eval_B.parquet
data/cleaned/weight_splits_v2/meta_B.json
```

Builder:

```bash
python tools/build_weight_splits_v2.py \
  --tiki-desc /path/to/qwen3-4b/predictions.parquet \
  --out data/cleaned/weight_splits_v2
```

---

## 11. Điểm cần nhóm thống nhất

1. Tập A dùng 4M token vì chỉ có 4,707M token khả dụng; 5M sẽ lặp gần như toàn bộ.
2. Tập B dùng raw BT 4B, **không** dùng `bt_tiki_desc` đã QC.
3. Chấp nhận rằng raw B còn boilerplate/rỗng/truncated; nếu muốn sạch, phải định nghĩa lại “pre-EDA” và ghi rõ.
4. Có chạy thêm seed 123 cho top mỗi tập hay chỉ 1 seed?
5. Có bổ sung `no_attr` (A) / `no_desc` (B) như biến thể “bỏ hẳn” không?
6. Có dùng Gemma credit làm tiêu chí phụ chính thức hay chỉ tham khảo?

---

## 12. Tóm tắt một câu

```text
Mỗi cấu hình chia quota token cho 4 nhóm; trong nhóm sample đều có hoàn lại;
mọi cấu hình cùng 4M token, 500 step, cùng eval — để chênh lệch chỉ còn do weight.
```
