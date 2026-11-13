# Rà soát kỹ thuật khảo sát trọng số A/B

**Người rà soát:** trợ lý của Nhật Anh  
**File được rà:** `WEIGHT_SURVEY_TECHNIQUE_A_B.md`  
**Ngày:** 08/11/2026  
**Mục đích:** chỉ ra chỗ chưa đúng, đề xuất bản sửa, để Quang review trước khi chạy thí nghiệm.

---

## 0. Tóm tắt

Kỹ thuật tổng thể đúng hướng: chia 4 nhóm theo nguồn × loại, lấy mẫu theo token, cố định budget, số step, seed, eval, có bootstrap và guardrail. Phần khung này giữ được.

Có **một lỗi toán nghiêm trọng** ở công thức tính xác suất chọn dòng: file dùng `M_g = tổng token` làm mẫu số, nhưng khi đơn vị lấy mẫu là **cả dòng** thì phải dùng `S_g = tổng bình phương token`. Nếu giữ `M_g`, tỉ lệ token thực tế lệch rất mạnh so với tỉ lệ đặt ra. Ví dụ trên dữ liệu thật: đặt 10,09% title nhưng chỉ ra 0,90%; đặt 50% nhưng chỉ ra 9,86%. Đổi sang `S_g` thì ra 10,10% và 49,48%.

Ngoài lỗi chính, có 6 điểm cần làm rõ để kết quả không bị confound hoặc bị hiểu sai: đơn vị lấy mẫu, `eff_tokens` và prompt, token cặp so với token đích, thiết kế 4 nhóm bị confound, cách đánh giá, và phạm vi dữ liệu.

---

## 1. Lỗi chính: mẫu số phải là `S_g = Σ e_i²`, không phải `M_g = Σ e_i`

### 1.1. Công thức trong file

Ký hiệu:

- `e_i`: số token của dòng `i` (nguồn + đích, đã cắt ở `max_len`).
- `s_g`: tỉ lệ token mục tiêu của nhóm `g`, tổng các `s_g` bằng 1.
- `w_g`: hệ số của nhóm `g`.
- `p_i`: xác suất chọn dòng `i`.

File đang viết:

```text
M_g = Σ e_i                     (tổng token của nhóm g)
w_g = s_g / M_g
p_i = (w_g · e_i) / Σ_j (w_g(j) · e_j)
```

Sau đó file kết luận: làm vậy thì token share của nhóm `g` đúng bằng `s_g`.

### 1.2. Vì sao sai

Khi bốc một dòng, mình lấy **toàn bộ** `e_i` token của dòng đó, không phải chỉ một token.

- Xác suất bốc dòng `i` tỉ lệ với `e_i`.
- Khi bốc được, dòng đóng góp `e_i` token.

Vì vậy token đóng góp của dòng tỉ lệ với:

```text
e_i × e_i = e_i²
```

không phải `e_i`.

Token share thực tế của nhóm `g` là:

```text
share_g = Σ_{i∈g} p_i · e_i / Σ_j p_j · e_j
```

Thay `p_i = w_g · e_i / Z` vào, với `Z = Σ_j w_g(j) · e_j`:

```text
share_g = w_g · S_g / Σ_h w_h · S_h
```

trong đó:

```text
S_g = Σ_{i∈g} e_i²
```

Nếu tiếp tục thay `w_g = s_g / M_g` như file:

```text
share_g = (s_g · S_g / M_g) / Σ_h (s_h · S_h / M_h)
```

Biểu thức này chỉ bằng `s_g` khi `S_g / M_g` giống nhau ở mọi nhóm. Thực tế không giống.

`S_g / M_g` chính là **độ dài trung bình có trọng số token** của nhóm. Nhóm nào có dòng dài hơn, độ dài chênh lệch nhiều hơn thì `S_g / M_g` lớn hơn, nên bị lấy quá nhiều token. Đó là lý do title bị mất phần token đáng lẽ thuộc về nó.

### 1.3. Công thức đúng

```text
S_g = Σ e_i²                    (tổng bình phương token của nhóm g)
w_g = s_g / S_g
p_i = (w_g · e_i) / Σ_j (w_g(j) · e_j)
```

Khi đó:

```text
share_g = (s_g / S_g) · S_g / Σ_h (s_h / S_h) · S_h
        = s_g / Σ_h s_h
        = s_g
```

Đúng cho mọi phân bố độ dài, không cần điều kiện gì thêm.

Chỉ có **một chỗ đổi** so với file: mẫu số từ `M_g = Σ e_i` thành `S_g = Σ e_i²`. Các bước còn lại giữ nguyên.

### 1.4. Ví dụ 1: hai nhóm, mỗi nhóm một dòng

Giả sử:

- Nhóm A: 1 dòng 10 token.
- Nhóm B: 1 dòng 20 token.
- Muốn mỗi nhóm 50% token.

**Theo công thức trong file:**

```text
M_A = 10, M_B = 20
w_A = 0,5 / 10  = 0,05
w_B = 0,5 / 20  = 0,025
p_A ∝ 0,05 × 10 = 0,5
p_B ∝ 0,025 × 20 = 0,5
p_A = 0,5, p_B = 0,5
```

Bốc một dòng, token share A:

```text
0,5 × 10 / (0,5 × 10 + 0,5 × 20) = 5 / 15 = 33,3%
```

Không phải 50%.

**Theo công thức sửa:**

```text
S_A = 10² = 100
S_B = 20² = 400
w_A = 0,5 / 100  = 0,005
w_B = 0,5 / 400  = 0,00125
p_A ∝ 0,005 × 10  = 0,05
p_B ∝ 0,00125 × 20 = 0,025
p_A = 2/3, p_B = 1/3
```

Token share A:

```text
(2/3 × 10) / ((2/3 × 10) + (1/3 × 20))
= 6,667 / (6,667 + 6,667)
= 50%
```

Đúng.

Điểm cần hiểu: dòng A ngắn hơn nên phải được bốc **nhiều lần hơn** (2/3 so với 1/3) mới đủ 50% token. Công thức cũ bốc đều 50/50 nên nhóm A bị thiếu token.

### 1.5. Ví dụ 2: mỗi nhóm hai dòng, có dòng ngắn lẫn dòng dài

Giả sử:

- Nhóm A: dòng 5 token và dòng 15 token.
- Nhóm B: hai dòng 10 token.
- Muốn mỗi nhóm 50% token.

**Theo công thức trong file:**

```text
M_A = 20, M_B = 20
w_A = w_B = 0,5 / 20 = 0,025
p ∝ [0,125; 0,375; 0,25; 0,25]
```

Token share A:

```text
(0,125 × 5 + 0,375 × 15) /
[(0,125 × 5 + 0,375 × 15) + (0,25 × 10 + 0,25 × 10)]
= 6,25 / 11,25
= 55,6%
```

Sai 5,6 điểm.

**Theo công thức sửa:**

```text
S_A = 5² + 15² = 25 + 225 = 250
S_B = 10² + 10² = 100 + 100 = 200
w_A = 0,5 / 250 = 0,002
w_B = 0,5 / 200 = 0,0025
p ∝ [0,01; 0,03; 0,025; 0,025]
```

Token share A:

```text
(0,1111 × 5 + 0,3333 × 15) / ...
= 5,556 / (5,556 + 5,556)
= 50%
```

Đúng.

### 1.6. Ví dụ 3: dùng chính bảng tính tay ở §4.3 của file

§4.3 của file tính bốn trọng số tương đối:

| Dòng | `e_i` | Hệ số nhóm | `w_g × e_i` |
|---|---:|---:|---:|
| 1688 title | 49,2 | 0,63 | 31,0 |
| 1688 attribute | 4,2 | 0,31 | 1,3 |
| Tiki title | 41,7 | 2,90 | 120,9 |
| Tiki attribute | 14,7 | 4,67 | 68,6 |

Bốn số 31,0; 1,3; 120,9; 68,6 là **trọng số chọn dòng**, không phải token share.

Nếu bốc theo bốn trọng số này, token đóng góp phải nhân thêm `e_i` một lần nữa:

| Dòng | Trọng số chọn dòng | `e_i` | Token đóng góp |
|---|---:|---:|---:|
| 1688 title | 31,0 | 49,2 | 1.525 |
| 1688 attribute | 1,3 | 4,2 | 5,5 |
| Tiki title | 120,9 | 41,7 | 5.043 |
| Tiki attribute | 68,6 | 14,7 | 1.009 |

Token share thực tế của bốn dòng này:

```text
1688 title      : 20,1%
1688 attribute  : 0,07%
Tiki title      : 66,5%
Tiki attribute  : 13,3%
```

Trong khi cấu hình `tiki_heavy` đặt:

```text
1688 title 10%, 1688 attribute 20%, Tiki title 30%, Tiki attribute 40%
```

Hai kết quả khác hẳn nhau. Đây là bằng chứng ngay trong ví dụ của file: bảng §4.3 đang là trọng số chọn dòng, nhưng được đọc như token share.

### 1.7. Kiểm trên dữ liệu thật

Để chắc không phải lỗi lý thuyết, em chạy cả hai công thức trên bộ pool của nhóm (23.017 sản phẩm, 46.120 mẫu cụm, 46.110 mẫu pair, 4M token cặp, seed 42). Kết quả:

| Bộ | Cấu hình | Title đặt | Công thức trong file | Công thức sửa |
|---|---|---:|---:|---:|
| Cụm | natural | 10,09% | **0,90%** | **10,10%** |
| Pair | natural | 11,20% | **4,93%** | **11,37%** |
| Cụm | balanced | 50% | 9,86% | 49,48% |
| Pair | balanced | 50% | 31,31% | 50,42% |
| Cụm | title_heavy | 80% | 29,24% | 78,85% |
| Pair | title_heavy | 65% | 45,70% | 65,30% |

Cột "Title đặt" là tỉ lệ token cặp của hai nhóm title. Cột "Công thức trong file" dùng `M_g`. Cột "Công thức sửa" dùng `S_g`.

Lệch lớn nhất ở bộ cụm vì khối mô tả dài và độ dài chênh nhau nhiều. Bộ pair cũng lệch vì mỗi sản phẩm có nhiều dòng attribute độ dài khác nhau.

Số thật của bộ dữ liệu:

- Pool: 23.017 sản phẩm (13.352 của 1688, 9.665 của Tiki).
- Cụm: 46.120 mẫu; pair: 46.110 mẫu.
- Pair có 414.243 dòng attribute (1688: 324.949 dòng, 24,2 dòng/mẫu mô tả, tức khoảng 24,4 dòng/sản phẩm; Tiki: 66.277 dòng, 6,9 dòng/mẫu).
- Chia fold 0: 18.533 sản phẩm train, 4.484 sản phẩm dev. Test 2.558 sản phẩm không đụng.

Các số này chỉ để kiểm chứng; lỗi công thức đúng cả trên dữ liệu khác.

### 1.8. Vì sao nhìn qua khó thấy

Công thức `p_i ∝ w_g · e_i` rất giống công thức chuẩn của **lấy mẫu theo token**:

- Nếu đơn vị là **một token**, chọn token theo trọng số `w_g`, rồi lấy dòng chứa token đó, thì `p_i ∝ w_g · e_i` là đúng.
- Nhưng khi huấn luyện, mình lấy **cả dòng**, nên các token còn lại của dòng cũng được tính. Lúc đó phải chia thêm một lần `e_i`, tức mẫu số thành `Σ e_i²`.

Nói cách khác: công thức trong file đúng cho "chọn token", sai cho "chọn dòng". File §3 nói mỗi dòng là một cặp, §4.2 lấy mẫu dòng, nên trường hợp đúng là "chọn dòng".

### 1.9. Bản sửa tối thiểu để dán vào file

Thay đoạn §4.2 bằng:

```text
S_g = Σ e_i²                    (tổng bình phương eff_tokens của nhóm g)
w_g = s_g / S_g
p_i = (w_g · e_i) / Σ_j (w_g(j) · e_j)
```

Giữ nguyên: 4 nhóm, lấy mẫu có hoàn lại, budget token, số optimizer step, token mỗi step, seed, eval, prompt, tokenizer.

Thêm một dòng kiểm tra:

```text
Nếu code cho hai dòng 10 và 20 token, muốn 50/50, mà p = [0,5; 0,5]
và token share ra 33,3% thì đang dùng M_g. Phải ra p = [2/3; 1/3]
và token share 50%.
```

### 1.10. Đoạn code kiểm nhanh

```python
def kiem_tra(e, s):
    # e: list token từng dòng, s: list tỉ lệ mục tiêu (tổng = 1)
    # Công thức trong file
    M = [sum(e[i] for i in nhom) for nhom in range(len(s))]
    w = [s[g] / M[g] for g in range(len(s))]
    # Công thức sửa
    S = [sum(e[i] ** 2 for i in nhom) for nhom in range(len(s))]
    w_sua = [s[g] / S[g] for g in range(len(s))]
    # ... tính p và token share ...
```

Với `e = [[10], [20]]`, `s = [0.5, 0.5]`:

- Công thức file: `p = [0.5, 0.5]`, token share `[0.333, 0.667]`.
- Công thức sửa: `p = [0.667, 0.333]`, token share `[0.5, 0.5]`.

---

## 2. Các điểm cần làm rõ (không hẳn là lỗi, nhưng ảnh hưởng kết luận)

### 2.1. Định nghĩa A/B chưa khớp với pair/cụm

File định nghĩa:

- A = title + attribute (mỗi attribute là một dòng).
- B = title + description (cả khối).

Trong khi nhóm đang dùng:

- **Pair**: mỗi dòng `name: value` là một mẫu.
- **Cụm**: cả khối `name: value; ...` là một mẫu.

Nếu A/B không định nghĩa giống pair/cụm thì kết quả không so được với nhau. Cụ thể, bản pair của nhóm có 414.243 dòng attribute, còn nếu để cả khối như B thì chỉ có 46.110 mẫu. Đơn vị mẫu khác nhau sẽ làm công thức, budget và cách đánh giá khác nhau.

Đề nghị: thống nhất trước khi chạy, và ghi rõ trong file là "A = pair", "B = cụm" hay "A/B là bài toán khác".

### 2.2. `eff_tokens` và prompt

`eff_tokens` trong file chỉ tính cặp nguồn + đích. Khi huấn luyện, mỗi mẫu còn có prompt và title context. Với attribute ngắn, prompt có thể lớn hơn nội dung 3 đến 4 lần.

Ví dụ trên bộ pair của nhóm:

- Nội dung cặp của một dòng attribute: khoảng 30 token.
- Prompt và title context lặp lại: khoảng 88 đến 95 token.

Nếu "8.000 token/step" là **token tổng** thì mỗi step chỉ có khoảng 2.000 token cặp, và 500 step chỉ đạt khoảng 1M token cặp, không phải 4M. Nếu "8.000 token/step" là **token cặp** thì compute thực lớn hơn nhiều.

Đề nghị: định nghĩa rõ 8k là token cặp hay token tổng, và log cả hai.

### 2.3. Token cặp và token đích

`s_g` trong file là tỉ lệ trên `eff_tokens` (nguồn + đích). Nếu mục tiêu là tỉ trọng loss hoặc token đích thì phải:

- dùng `t_i` (token đích) thay `e_i`, và `S_g = Σ t_i²`; hoặc
- hiệu chuẩn `s_g` bằng chia đôi để ra đúng tỉ lệ đích.

Ví dụ trên bộ cụm của nhóm, cấu hình `balanced` đặt 25% token cặp cho mỗi nhóm, nhưng token đích của 1688 title ra 29,5%, của 1688 description ra 26,6%. Nếu đọc như "25% token đích" thì sai.

Đề nghị: ghi rõ đang kiểm soát token cặp hay token đích, và báo cả hai trong kết quả.

### 2.4. Bốn nhóm là tích của nguồn × loại, nên các cấu hình bị confound

Bốn nhóm là:

```text
1688_title, 1688_attribute/description, tiki_title, tiki_attribute/description
```

Vì vậy mỗi cấu hình đổi **hai trục cùng lúc**: tỉ lệ nguồn và tỉ lệ loại.

Ví dụ ở tập A:

| Cấu hình | % nguồn 1688 | % title |
|---|---:|---:|
| natural | 81,1 | 26,3 |
| balanced | 50 | 50 |
| title_heavy | 60 | 65 |
| attr_heavy | 70 | 15 |
| tiki_heavy | 30 | 40 |

`title_heavy` vừa tăng title lên 65% vừa tăng Tiki lên 40%. `tiki_heavy` cũng tăng title lên 40%. Nếu kết quả tốt hơn, không biết là do title hay do Tiki.

Ở tập B, `natural` khoảng 50/50 nguồn và 7% title; `balanced` 50/50 nguồn và 50% title; `title_heavy` 80% title nhưng vẫn 50/50 nguồn; `tiki_heavy` 75% Tiki và 45% title. Cùng tên "balanced" hay "title_heavy" nhưng ở A và B là hai thí nghiệm khác nhau.

Đề nghị:

- Thiết kế 2×2 (nguồn × loại), hoặc
- giữ cố định một trục, chỉ đổi trục kia. Ví dụ giữ nguồn natural, đổi title 5/11/20/30/50%; giữ title natural, đổi nguồn 30/50/70/90%.

Nếu vẫn giữ 5 cấu hình hiện tại, khi viết kết luận chỉ nên nói "mix này tốt hơn", không nên nói "title quan trọng hơn" hay "Tiki quan trọng hơn".

### 2.5. Cách đánh giá

Một số điểm nên sửa trước khi chạy:

1. **Đơn vị eval:** 1.200 dòng chia 4 stratum = 300 dòng/stratum. Với attribute 4 token, COMET rất nhiễu. Nên gộp theo sản phẩm, hoặc báo riêng theo nhóm độ dài.
2. **Bootstrap:** đang theo `eval_id`. Nhiều dòng cùng một sản phẩm tương quan với nhau, nên CI sẽ hẹp giả. Nên cluster bootstrap theo `product_id`.
3. **Macro trộn hai thang:** COMET-da cho 1688 và COMET-QE cho Tiki là hai thang khác nhau. Trung bình cộng 4 stratum sẽ trộn hai thang. Nên báo từng stratum, hoặc chuẩn hóa trước khi macro.
4. **Đa bội:** 4 cấu hình so với natural, không hiệu chỉnh. Với 95% CI, xác suất có ít nhất một dương tính giả mỗi tập khoảng 18,5%; hai tập khoảng 33%. Nên khóa một primary endpoint hoặc Bonferroni.
5. **Ngưỡng và seed:** `best - natural < 0,005` có thể mịn hơn nhiễu huấn luyện lại. Với 1 seed, không đo được dao động. Nên lặp natural 2 lần cùng cấu hình khác seed, hoặc 2 seed cho natural và best.
6. **Thiếu baseline tuyệt đối:** natural chỉ là baseline phân bố. Nên thêm zero-shot cùng eval để biết mọi cấu hình đang tốt hay xấu.
7. **Gemma:** thang AAA đến C 9 mức chưa có rubric và neo. Chỉ nên để tín hiệu phụ, ghi rõ là "rà soát bằng LLM".

### 2.6. Phạm vi dữ liệu và hướng dịch

- A/B dùng dữ liệu **trước EDA**, còn Tiki description lấy từ **raw BT 4B**. Dữ liệu còn boilerplate, dòng rỗng, dòng truncated. File đã ghi rõ, nhưng nếu mục tiêu là mô hình cuối thì nên chạy lại trên dữ liệu sau Judge, hoặc nói rõ đây là nghiên cứu riêng.
- Tiki target là bản dịch của **Qwen3-4B**, cùng họ với mô hình đang fine-tune. Có rủi ro self-distillation: mô hình học lại đầu ra của chính nó, và COMET-QE có thể ưu ái văn phong đó. Nên nêu trong giới hạn.
- Theo các ví dụ trong file, cả A và B đều có nguồn Việt, đích Trung, tức **vi→zh**. Nếu mục tiêu chính là **zh→vi**, kết quả có thể không chuyển được. Cần nói rõ phạm vi.
- Cắt `max_len` khi train phải giống khi eval. Nếu eval dùng dòng dài hơn phần train đã cắt thì điểm không phản ánh đúng.
- Sampler phải chỉ lấy từ **train**, không lấy từ eval/dev/test. File nói "cùng product_id split" nhưng chưa nói rõ eval có bị loại khỏi pool lấy mẫu không.
- Chưa thấy nói chống gần trùng sản phẩm khi chia tập. Nên gom nhóm sản phẩm gần trùng trước khi chia, tránh rò rỉ.

### 2.7. Ví dụ 4,2 token ở §4.3 không đại diện

File ghi:

```text
rows: 118.688
effective token: 4,707M
1688_attribute: 65,10% token
```

Nếu mỗi dòng 1688 attribute chỉ 4,2 token thì cần khoảng:

```text
4.707.000 × 0,651 / 4,2 ≈ 729.000 dòng
```

nhiều hơn tổng 118.688 dòng. Vậy 4,2 token chỉ là **một dòng ví dụ rất ngắn**, không phải trung bình. Trung bình thật của 1688 attribute có thể cao hơn nhiều, có thể 30 đến 50 token.

Hệ quả: câu "Tiki attribute 14,7 token nặng hơn 1688 attribute 4,2 token" chỉ đúng cho hai dòng đó, không đại diện cho cả nhóm. Khi lập luận "dòng ngắn so với dòng dài", cần dùng trung bình và phân bố thật, hoặc ghi rõ đây là ví dụ minh họa.

---

## 3. Đề xuất bản sửa tối thiểu

### 3.1. Sửa công thức

Đổi:

```text
M_g = Σ e_i
w_g = s_g / M_g
```

thành:

```text
S_g = Σ e_i²
w_g = s_g / S_g
```

Giữ nguyên `p_i`:

```text
p_i = (w_g · e_i) / Σ_j (w_g(j) · e_j)
```

### 3.2. Thêm kiểm tra tự động

Trước khi chạy, thêm một test nhỏ:

```text
e = [[10], [20]], s = [0.5, 0.5]
- Công thức đúng: p = [2/3, 1/3], token share = [0.5, 0.5]
- Công thức sai: p = [0.5, 0.5], token share = [0.333, 0.667]
```

Test này bắt được lỗi ngay, không cần dữ liệu thật.

### 3.3. Ghi rõ trong file

- `s_g` là tỉ lệ token cặp hay token đích.
- `8k token/step` là token cặp hay token tổng.
- A/B là pair/cụm hay bài toán khác.
- Có lấy eval ra khỏi pool sampling không.
- Có áp dụng chống gần trùng khi chia tập không.

### 3.4. Nếu muốn tách hiệu ứng nguồn và loại

Thay 5 cấu hình hiện tại bằng lưới giữ cố định một trục:

- Trục title: các mức quanh natural của từng bộ (ví dụ natural/2, natural, 2×natural, 50%, 80%), giữ nguồn theo natural.
- Trục nguồn: 30%, 50%, 70%, 90%, giữ title theo natural.
- Thêm 2×2 nếu muốn xem tương tác.

Như vậy mới trả lời được "title có quan trọng hơn không" và "Tiki có cần ưu tiên không" một cách tách bạch.

---

## 4. Câu hỏi cho Quang

1. `s_g` trong file là tỉ lệ token cặp hay token đích?
2. `8.000 token/step` là token cặp hay token tổng?
3. A/B có chủ ý là pair/cụm không, hay là bài toán khác?
4. Có chủ ý chỉ chạy vi→zh không?
5. Có muốn giữ 5 cấu hình hiện tại hay chuyển sang lưới 2×2 / giữ cố định một trục?
6. Eval dùng COMET hay metric khác? Có thêm zero-shot không?
7. Có lấy eval/dev ra khỏi pool lấy mẫu không?
8. Dữ liệu pre-EDA và BT 4B có phải chủ ý cho nghiên cứu riêng không?

---

## 5. Kết luận

Kỹ thuật tổng thể của file là hướng đúng: kiểm soát token, cố định budget/step/seed, có bootstrap và guardrail. Chỉ có một lỗi toán ở mẫu số, nhưng lỗi này làm kết quả lệch rất mạnh, nên cần sửa trước khi chạy.

Bản sửa gọn nhất:

```text
S_g = Σ e_i²
w_g = s_g / S_g
p_i = (w_g · e_i) / Σ_j (w_g(j) · e_j)
```

Các điểm còn lại là cần làm rõ định nghĩa và cách đánh giá, không phải lỗi toán. Mong Quang xem lại, chỗ nào chưa đúng thì phản hồi để nhóm thống nhất trước khi tốn GPU.

---

**Phụ lục: file và số liệu dùng để kiểm**

- Nguồn: pool 23.017 sản phẩm của nhóm, 46.120 mẫu cụm, 46.110 mẫu pair.
- Pair: 414.243 dòng attribute.
- Chia fold 0: 18.533 sản phẩm train, 4.484 sản phẩm dev.
- Tokenizer: Qwen3, vocab 151.669.
- Lấy mẫu: 4M token cặp mỗi cấu hình, seed 42.
- Các bảng kết quả: `data/eda/khao_sat_trong_so/`.
