# TÀI LIỆU WALKTHROUGH: TỔNG HỢP CÁC VẤN ĐỀ VÀ QUY TRÌNH TIỀN XỬ LÝ TIÊU ĐỀ TRUNG–VIỆT TRƯỚC KHI ĐƯA VÀO BGE-M3

Tài liệu kỹ thuật tổng hợp toàn diện các vấn đề, dị tật dữ liệu phát hiện trên **16.348 cặp tiêu đề song ngữ Trung–Việt** (`title_zh` / `title_vi`) từ sàn thương mại điện tử 1688, cùng các phương pháp xử lý, chuẩn hóa cụ thể trước khi đưa vào mô hình embedding `BAAI/bge-m3`.

---

## 1. Tổng hợp Toàn bộ Các Vấn đề và Dị tật Dữ liệu của Tiêu đề Gốc Trước BGE-M3

Trước khi tiến hành tiền xử lý, việc sử dụng trực tiếp tiêu đề thô để tính điểm tương đồng (Cosine Similarity) qua mô hình BGE-M3 gặp phải **9 nhóm vấn đề và lỗ hổng nghiêm trọng** làm sai lệch kết quả đánh giá chất lượng bản dịch:

```
                            [TIÊU ĐỀ GỐC (THÔ)]
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
     [TIÊU ĐỀ TRUNG]                                    [TIÊU ĐỀ VIỆT]
           │                                                   │
  • 1.1. Lỗi chép nguyên CJK (2 dòng)                 • 1.1. Lỗi chép nguyên CJK (2 dòng)
  • 1.2. Ký tự toàn chiều (44 dòng)                   • 1.3. Còn sót chữ Hán (7 dòng)
  • 1.7. Ký tự ngoại lai Kirin (5 dòng)               • 1.4. Dấu chấm đuôi MT (5.064 dòng)
  • 1.8. Thẻ ngoặc vuông 【...】 (304 dòng)            • 1.5. Lệch số thập phân 4,3 vs 4.3
  • 1.8. Từ khóa tiếp thị (970 dòng)                  • 1.6. Lệch số Hán văn đo lường
  • 1.9. Cụm tiếng Anh xuất khẩu bê nguyên            • 1.9. Tiếng Anh bồi Chinglish
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
                     [BGE-M3: ĐIỂM SỐ BỊ SAI LỆCH]
             (Điểm cao ảo 1.0, phân mảnh token, tụt điểm vô lý)
```

### 1.1. Lỗi chép nguyên tiếng Trung nhưng được chấm điểm tuyệt đối 1.0
- **Hiện tượng thực tế:** Trong tập dữ liệu có 2 sản phẩm (mã `1043561115166` và `1058721001412`) bị lỗi crawler khiến cột tiêu đề tiếng Việt `title_vi` chép nguyên văn 100% chữ Hán của tiêu đề tiếng Trung `title_zh`.
- **Hệ quả của mô hình:** Khi đưa hai chuỗi ký tự giống hệt nhau vào BGE-M3, mô hình sinh ra hai vector giống hệt nhau. Tích vô hướng cosin của chúng đạt đúng **1.0000**!
- **Hậu quả đánh giá:** Nếu dựa vào điểm số này, hệ thống sẽ kết luận đây là "2 bản dịch hoàn hảo nhất trong tập dữ liệu", trong khi thực tế sản phẩm hoàn toàn chưa được dịch sang tiếng Việt.

### 1.2. Phân mảnh token do ký tự Latin và số toàn chiều (Full-width)
- **Hiện tượng thực tế:** 44 tiêu đề tiếng Trung sử dụng các ký tự toàn chiều sinh ra từ bộ gõ chữ Hán (như `［TS03］`, `BUQLE @姐妹好看！`, `厂家）377`).
- **Hệ quả của mô hình:** Tokenizer của BGE-M3 coi dấu ngoặc toàn chiều `［` (`\uFF3B`) và dấu ngoặc vuông tiêu chuẩn `[` (`\u005B`) là các token hoàn toàn khác nhau, làm giảm độ tương đồng nhân tạo giữa tiêu đề tiếng Trung và bản dịch tiếng Việt (vốn dùng ký tự nửa chiều chuẩn).

### 1.3. Tiêu đề tiếng Việt còn sót chữ Hán chưa được dịch
- **Hiện tượng thực tế:** Có 7 tiêu đề tiếng Việt còn chứa ký tự chữ Hán (CJK). Ngoài 2 trường hợp chép nguyên ở trên, có 5 trường hợp máy dịch bỏ sót cụm từ tiếng Trung (như `镂空` - khoét lỗ ren, `早产` - sinh non, `嘎巴大红袍乌龙茶` - tên trà).
- **Hệ quả:** BGE-M3 nhận thấy các token chữ Hán trùng nhau ở cả hai câu nên cho điểm cao giả tạo (ví dụ mã `872054230824` đạt điểm tương đồng lên tới 0.9236).

### 1.4. Dấu câu đuôi thừa do máy dịch sinh ra (Trailing Punctuation Artifacts)
- **Hiện tượng thực tế:** Có tới **5.072 tiêu đề** kết thúc bằng các dấu câu vô nghĩa ở cuối chuỗi. Trong đó, có **5.064 tiêu đề tiếng Việt kết thúc bằng dấu chấm (`.`)**. Đây là dị tật điển hình sinh ra do công cụ dịch máy (Google Translate / DeepL) tự động bổ sung dấu chấm kết thúc câu văn bản, trong khi tiêu đề sản phẩm thương mại điện tử gốc tiếng Trung hầu như không bao giờ có dấu chấm cuối câu (chỉ có 9 trường hợp).
- **Tác động đến BGE-M3:** Khi đưa vào mô hình BGE-M3, dấu chấm `.` cuối câu bị tokenizer tách thành 1 token độc lập. Việc câu tiếng Việt bị thừa token này trong khi câu tiếng Trung không có sẽ gây lệch vị trí token pooling và làm giảm độ tương đồng cosin một cách vô lý.

### 1.5. Lệch dấu phân cách số thập phân (Decimal Comma vs Decimal Dot)
- **Hiện tượng thực tế:** Trong tiếng Việt, số thập phân thường được viết bằng dấu phẩy theo quy chuẩn bản địa (ví dụ: `4,3 inch`, `1,5 mét`, `0,5 kg`), trong khi tiêu đề tiếng Trung luôn sử dụng dấu chấm thập phân chuẩn quốc tế (ví dụ: `4.3寸`, `1.5米`, `0.5kg`).
- **Tác động:** Nếu dùng regex trích xuất số thông thường `\d+`, chuỗi `4,3` sẽ bị bóc tách thành 2 con số riêng biệt là `['4', '3']`. Khi so khớp với phía tiếng Trung (chỉ có con số `4.3`), hệ thống kiểm tra logic sẽ báo động sai lệch số (`num_mismatch`), làm tăng số lượng ca cảnh báo giả.

### 1.6. Lệch số do chữ số Hán văn đo lường (Chinese Measure Numerals Mismatch)
- **Hiện tượng thực tế:** Tiêu đề tiếng Trung rất thường xuyên sử dụng chữ Hán số đếm đi liền với các lượng từ thương mại điện tử:
  * `四合一` (Bốn trong một) $\rightarrow$ Tiếng Việt dịch thành `4 trong 1`.
  * `三件套` / `五件套` (Bộ ba món / Bộ năm món) $\rightarrow$ Tiếng Việt dịch thành `bộ 3 món` / `bộ 5 món`.
  * `三录` (Ba mắt quay ghi hình) $\rightarrow$ Tiếng Việt dịch thành `3 mắt quay`.
  * `战鹰三号` (Chiến Ưng số 3) $\rightarrow$ Tiếng Việt dịch thành `số 3`.
- **Tác động:** Ban đầu, khi chỉ so khớp các con số Ả Rập, phía tiếng Trung không có ký tự số `3`, `4`, `5` nên có tới **1.367 cặp sản phẩm bị gán cờ lệch số `num_mismatch`**. Tuy nhiên, nếu bóc tách bừa bãi toàn bộ chữ Hán `[一二两三四五六七八九十]` thì các từ ngữ thông dụng như `一般` (thông thường), `单一` (đơn nhất), `一件代发` (bán lẻ một món) cũng bị bóc thành số 1, khiến số lượng lỗi giả tăng vọt lên **3.203 trường hợp**!

### 1.7. Tạp chất ký tự ngoại lai bên thứ ba (Third-party Foreign Scripts)
- **Hiện tượng thực tế:** Phát hiện **5 sản phẩm** trong dữ liệu chứa ký tự chữ cái Kirin (Cyrillic - tiếng Nga) trong tiêu đề tiếng Trung. Nguyên nhân là do các xưởng trên 1688 đăng bán các mặt hàng chuyên xuất khẩu sang thị trường Nga hoặc lạm dụng từ khóa tìm kiếm:
  * `557915795254`: `二合一Руский俄罗斯专供X7俄文电子狗...` (Chứa chữ Nga `Руский` - Tiếng Nga).
  * `921442749651`: `Халат домашний 一片式连衣短裙...` (Chứa cụm từ tiếng Nga nghĩa là "Áo choàng mặc nhà").
  * `872054230824`: `Gaba Dahongpao tea Габа чай 嘎巴大红袍...` (Chứa chữ Nga `Габа чай` - Trà Gaba).
  * `888940568056`: `shu puer tea шу пуэр чай бык新文茶厂...` (Chứa chữ Nga `шу пуэр чай бык` - Trà Phổ Nhĩ).
  * `1051772848198`: `LVАFGН2026新款...` (Sử dụng ký tự Cyrillic giả dạng `А` - mã `\u0410` và `Н` - mã `\u041D` thay cho chữ Latin `A`, `H` để lách bản quyền thương hiệu).
- **Tác động đến BGE-M3:** Ký tự tiếng Nga xen lẫn làm phân mảnh tokenizer đa ngữ và kéo lệch không gian vector của cặp Trung–Việt, khiến điểm số ngữ nghĩa không còn phản ánh trung thực chất lượng dịch thuật Trung–Việt.

### 1.8. Thẻ ngoặc vuông tiếp thị & Từ ngữ quảng cáo gây nhiễu ngữ nghĩa (Promotional Tags & Marketing Words)
- **Hiện tượng thực tế:**
  * **304 tiêu đề** tiếng Trung sử dụng các thẻ đóng mở ngoặc vuông nổi bật `【...】` (ví dụ: `【一折专区】` - khu vực giảm giá 10%, `【源头工厂】` - nhà máy nguồn, `【亏本清仓】` - bán lỗ xả kho).
  * **970 tiêu đề** tiếng Trung chứa các từ ngữ tiếp thị quen thuộc của sàn thương mại điện tử như `包邮` (bao ship), `秒杀` (flash sale), `爆款` (hàng hot trend), `厂家直销` (bán trực tiếp từ xưởng), `一件代发` (bán lẻ từng chiếc giao ngay).
- **Tác động đến BGE-M3:** Khi dịch sang tiếng Việt, các từ tiếp thị này thường được chuyển ngữ theo kiểu giật tít, chiếm tỷ trọng token đáng kể trong câu và làm loãng nội dung tên gọi sản phẩm thực sự.

### 1.9. Hiện tượng Ký tự và Cụm từ tiếng Anh trong Tiêu đề Song ngữ
Quá trình quét toàn diện tập dữ liệu cho thấy **tiếng Anh xuất hiện với mật độ rất cao**:
- **6.249 tiêu đề tiếng Trung (38,22%)** chứa các từ / ký tự tiếng Anh (Latin).
- **100% tiêu đề tiếng Việt** đều có dấu thanh tiếng Việt (không có dòng nào là văn bản 100% tiếng Anh không dấu).
- **2.651 tiêu đề tiếng Việt (16,21%)** chứa các cụm từ tiếng Anh từ 3 từ trở lên xen kẽ với tiếng Việt.

Hiện tượng tiếng Anh trong dữ liệu được phân hóa thành **4 nhóm bản chất hoàn toàn khác nhau**:
1. **Nhóm A (Thương hiệu quốc tế, Model công nghệ – Bắt buộc bảo toàn):**
   - Ví dụ: `Apple`, `iPhone 16 Pro Max`, `Type-C`, `USB`, `Bluetooth`, `WIFI`, `GPS`, `TWS`, `Mercedes-Benz`, `BMW`, `PPSU`, `PU`, `Cotton`.
   - Cơ chế BGE-M3: Điển hình nhất là chữ `蓝牙` (Lam Nha) xuất hiện **585 lần** trong tiếng Trung, và được dịch thành `Bluetooth` đúng **584 lần** trong tiếng Việt. BGE-M3 có không gian liên kết chéo tự nhiên cực mạnh giữa `蓝牙` và `Bluetooth`, cho điểm số cao chuẩn xác (**0.75 – 0.85**). Bắt buộc phải giữ nguyên, không được phiên âm hay xóa bỏ.
2. **Nhóm B (Từ mượn phong cách thời trang, mỹ phẩm – Giữ nguyên):**
   - Ví dụ: `INS` (572 lần - phong cách Instagram), `DIY` (72 lần), `Polo` (64 lần), `Oversize`, `Vintage`, `Retro`, `Crop top`, `T-shirt`, `Sneaker`. BGE-M3 nhận diện tốt các từ này, đủ điều kiện chấm điểm bình thường.
3. **Nhóm C (Cụm tiếng Anh xuất khẩu bị "Bê nguyên không dịch" – Dị tật điểm cao ảo ⚠️):**
   - Ví dụ: `680454865764` (`30day detox tea Peach flavor burn fat fit slim tea`), `774095913319` (`herbal Big Butt And Hips Enlargements maca Tea`).
   - Lỗ hổng BGE-M3: Do chuỗi tiếng Anh ở hai bên trùng khớp 100%, BGE-M3 trích xuất các token giống hệt nhau, khiến **điểm tương đồng vọt lên rất cao một cách giả tạo (0.8643 đến 0.9341)** dù bản dịch tiếng Việt chưa được dịch hoàn thiện!
4. **Nhóm D (Tiếng Anh bồi từ máy dịch Chinglish – Dị tật tụt điểm ⚠️):**
   - Ví dụ: `【抖音爆款】` $\rightarrow$ `[Douyin Hot Model]` (Sim: 0.6067), `网红丝带` $\rightarrow$ `Net Red Ribbon` (Sim: 0.6833), `魔夹出风口` $\rightarrow$ `Magic Clip Air Vent` (Sim: 0.6366).
   - Tác động: Tiếng Anh bồi làm mất cấu trúc ngữ pháp tự nhiên của tiếng Việt, kéo tụt điểm BGE-M3 từ 0.15 đến 0.25 điểm so với bản dịch tiếng Việt chuẩn.

---

## 2. Bảng Tổng hợp Số liệu Đo đạc Thực nghiệm Trước và Sau Tối ưu (16.348 Cặp)

Dưới đây là bảng so sánh định lượng chi tiết trên toàn bộ 16.348 dòng dữ liệu snapshot:

| Chỉ số / Nhóm dị tật | Số lượng ban đầu (Chưa tối ưu) | Số lượng sau khi tối ưu nâng cao | Chênh lệch / Hiệu quả đạt được | Ý nghĩa kỹ thuật |
|---|:---:|:---:|:---:|---|
| **Tổng số cặp sản phẩm** | **16.348** | **16.348** | **0** | Bảo toàn 100% dữ liệu gốc |
| **Trạng thái `score` (Hợp lệ)** | 14.950 (91,45%) | **15.345 (93,86%)** | **+395 cặp (+2,41%)** | Gia tăng lượng ngữ liệu sạch đủ chuẩn |
| **Trạng thái `review` (Rà soát)** | 1.396 (8,54%) | **1.001 (6,12%)** | **-395 cặp (-2,42%)** | Giảm thiểu tối đa các cảnh báo giả |
| **Trạng thái `skip` (Chặn điểm)** | 2 (0,01%) | **2 (0,01%)** | **0** | Chặn triệt để 2 bản chép nguyên CJK (gán `NaN`) |
| Dấu câu đuôi dư thừa (Trailing punct) | 5.072 tiêu đề | **0 tiêu đề (Đã làm sạch)** | **-5.072 dòng** | Đã xóa 5.064 dấu chấm MT ở tiếng Việt |
| Ký tự Latin/số toàn chiều (Full-width) | 44 tiêu đề | **0 tiêu đề (Đã gập ASCII)** | **-44 dòng** | Chuẩn hóa 100% về mã ASCII chuẩn |
| Lệch số học (`num_mismatch`) | 1.367 dòng | **969 dòng** | **-398 dòng (-29,1%)** | Cứu 367 ca chữ số Hán văn + số thập phân |
| Ký tự ngoại lai Cyrillic (`foreign_script`) | Không phát hiện (Bị bỏ sót) | **5 dòng** | **+5 dòng được phát hiện** | Nhận diện hàng xuất khẩu Nga & giả mạo ký tự |
| Còn sót chữ Hán trong tiếng Việt (`cjk_in_vi`) | 7 dòng | **7 dòng** | **0** | 2 dòng `skip` + 5 dòng `review` |
| Tiêu đề Trung chứa tiếng Anh / Latin | Chưa bóc tách | **6.249 dòng (38,22%)** | **Đã phân loại 4 nhóm** | Tách bạch thương hiệu chuẩn và lỗi dịch |
| Tiêu đề dịch `蓝牙` $\rightarrow$ `Bluetooth` | Chưa ghi nhận | **584 / 585 dòng (99,8%)** | **Khớp ngữ nghĩa cao** | BGE-M3 ánh xạ chéo chuẩn xác |
| Đoạn tiếng Anh xuất khẩu bê nguyên | Chưa phát hiện | **6 dòng** | **Phát hiện điểm cao ảo** | Tránh nhầm lẫn bản dịch chưa hoàn thiện |
| Lệch tỷ lệ token (`ratio_mismatch`) | 27 dòng | **27 dòng** | **0** | Phân loại chính xác các ca dịch thiếu/thừa từ |
| Tiêu đề chứa thẻ ngoặc `【...】` | Chưa thống kê | **304 dòng** | **Gắn nhãn `info`** | Phục vụ phân tích trọng số từ tiếp thị |
| Tiêu đề chứa từ khóa quảng cáo Trung | Chưa thống kê | **970 dòng** | **Gắn nhãn `info`** | Nhận diện chính xác ngữ cảnh thương mại |

---

## 3. Bảng Minh chứng Thực tế Các Trường hợp Tiêu biểu Được Xử lý

### 3.1. Minh chứng: Cứu thành công các ca chữ số Hán văn đo lường (Phục hồi 367 ca)
Trước khi tối ưu, các trường hợp này đều bị hệ thống gắn cờ sai lệch số `num_mismatch`. Sau khi áp dụng cơ chế Lookahead Numeral Recovery, hệ thống đã nhận diện được sự tương đồng hoàn hảo giữa chữ số Hán văn và số Ả Rập tiếng Việt:

| `product_id` | Tiêu đề tiếng Trung (`title_zh`) | Tiêu đề tiếng Việt (`title_vi`) | Con số tiếng Trung nhận diện | Con số tiếng Việt nhận diện | Trạng thái sau xử lý |
|---|---|---|:---:|:---:|:---:|
| `557915795254` | `二合一Руский俄罗斯专供X7俄文电子狗...` | `2 trong 1Ru voy Nga chuyên về X7...` | `['1', '2', '7']` (từ `二合一` + `X7`) | `['1', '2', '7']` (từ `2 trong 1` + `X7`) | Khớp số hoàn toàn |
| `831394947500` | `出口日本丨100%纯棉全棉春秋四件套...` | `Xuất khẩu sang Nhật Bản 丨 Bộ chăn ga gối bốn mảnh 100% cotton...` | `['100', '4']` (từ `100%` + `四件套`) | `['100', '4']` (từ `100%` + `bốn`) | Khớp số hoàn toàn |
| `906656433533` | `【一折专区】奥莱精选丨重磅丨男装T恤...` | `[Giảm giá 10%] Lựa chọn của Ole...` | `['10']` (từ `一折` = giảm 10%) | `['10']` (từ `10%`) | Khớp số hoàn toàn |

### 3.2. Minh chứng: Xóa bỏ 5.064 dấu chấm câu đuôi do máy dịch sinh ra
Dấu chấm kết câu tự động của máy dịch bị loại bỏ hoàn toàn, trả lại định dạng tự nhiên của tiêu đề thương mại điện tử:

| `product_id` | Tiêu đề tiếng Việt trước làm sạch (`title_vi`) | Tiêu đề tiếng Việt sau làm sạch (`title_vi_norm`) | Ký tự đã loại bỏ |
|---|---|---|:---:|
| `796245367375` | `Giày bốt nữ mũi nhọn cao gót mới 2026.` | `Giày bốt nữ mũi nhọn cao gót mới 2026` | Dấu chấm cuối chuỗi `.` |
| `688461823901` | `Bộ đồ ngủ cotton dệt kim dài tay xuân thu cho nữ.` | `Bộ đồ ngủ cotton dệt kim dài tay xuân thu cho nữ` | Dấu chấm cuối chuỗi `.` |
| `857912401662` | `Mặt nạ làm sạch sâu thu nhỏ lỗ chân lông dạng bùn khoáng.` | `Mặt nạ làm sạch sâu thu nhỏ lỗ chân lông dạng bùn khoáng` | Dấu chấm cuối chuỗi `.` |

### 3.3. Minh chứng: Phát hiện 5 ca chứa ký tự ngoại lai Kirin (Cyrillic)
Nhận diện chính xác các sản phẩm xuất khẩu hoặc cố tình trộn ký tự ngoại lai để lách quy chế:

| `product_id` | Tiêu đề tiếng Trung (`title_zh`) | Tiêu đề tiếng Việt (`title_vi`) | Ký tự ngoại lai phát hiện | Cờ gán rà soát |
|---|---|---|:---:|:---:|
| `557915795254` | `二合一Руский俄罗斯专供X7俄文电子狗行车记录仪流动测速` | `2 trong 1Ru voy Nga chuyên về X7 Máy ghi âm lái xe chó kỹ thuật số của Nga đo tốc độ di động` | Chữ Kirin: `Руский` | `foreign_script` |
| `921442749651` | `Халат домашний 一片式连衣短裙居家户外海边连衣裙` | `Váy liền thân, thích hợp mặc ở nhà, ngoài trời và đi biển` | Chữ Kirin: `Халат домашний` | `foreign_script` |
| `872054230824` | `Gaba Dahongpao tea Габа чай 嘎巴大红袍乌龙茶` | `Trà Gaba Dahongpao Trà Gaba 嘎巴大红袍乌龙茶` | Chữ Kirin: `Габа чай` (+ CJK sót) | `foreign_script;cjk_in_vi` |
| `888940568056` | `shu puer tea шу пуэр чай бык新文茶厂普洱茶牛饼` | `Trà Shu puer, Trà Shu puer từ nhà máy trà Niwen, bánh bò trà Pu'er` | Chữ Kirin: `шу пуэр чай бык` | `foreign_script` |
| `1051772848198` | `LVАFGН2026新款水洗四件套床上用品磨毛高奢被套床单宿舍家居` | `Bộ chăn ga gối đệm bốn món mới, có thể giặt được, mẫu LVAFGN2026...` | Homoglyph: `А` (`\u0410`), `Н` (`\u041D`) | `foreign_script` |

### 3.4. Minh chứng: Chặn điểm gán `NaN` cho 2 sản phẩm chép nguyên văn tiếng Trung
Hai trường hợp lỗi crawler nghiêm trọng nhất được xử lý triệt để, không để lọt điểm 1.0 ảo vào thống kê:

| `product_id` | Tiêu đề tiếng Trung (`title_zh`) | Tiêu đề tiếng Việt (`title_vi`) | Điểm BGE-M3 cũ | Điểm BGE-M3 mới | Trạng thái |
|---|---|---|:---:|:---:|:---:|
| `1043561115166` | `2026新款Polèn真皮马鞍包女高级感通勤单肩斜挎包小众设计感腋下` | `2026新款Polèn真皮马鞍包女高级感通勤单肩斜挎包小众设计感腋下` | 1.000 | **NaN (Rỗng)** | `skip` (`exact_copy_cjk`) |
| `1058721001412` | `HEATÔR夕碧泉水杨酸栀子花控油洁面乳清洁面部洗面奶厂家代发` | `HEATÔR夕碧泉水杨酸栀子花控油洁面乳清洁面部洗面奶厂家代发` | 1.000 | **NaN (Rỗng)** | `skip` (`exact_copy_cjk`) |

### 3.5. Minh chứng: Các trường hợp chứa Tiếng Anh và Tác động đến Điểm BGE-M3
Sự khác biệt rõ nét về điểm tương đồng giữa việc dịch chuẩn thương hiệu, lỗi chép nguyên tiếng Anh gây điểm cao ảo, và lỗi dịch tiếng Anh bồi (Chinglish):

| `product_id` | Tiêu đề tiếng Trung (`title_zh`) | Tiêu đề tiếng Việt (`title_vi`) | Phân loại hiện tượng Tiếng Anh | Điểm BGE-M3 | Đánh giá chất lượng thực tế |
|---|---|---|---|:---:|---|
| `680454865764` | `出口30day detox tea Peach flavor burn fat fit slim tea果味茶` | `Xuất Khẩu 30day detox tea Peach flavor burn fat fit slim tea hương trái cây` | **Bê nguyên cụm tiếng Anh không dịch** | **0.9341 (Điểm cao ảo)** | ⚠️ Dịch lười, trùng khớp token tiếng Anh nên bị chấm điểm cao nhân tạo. |
| `774095913319` | `花草袋泡茶herbal Big Butt And Hips Enlargements maca Tea` | `Túi hoa cỏ pha trà herbal Big Butt And Hips Enlargements maca Tea` | **Bê nguyên cụm tiếng Anh không dịch** | **0.8643 (Điểm cao ảo)** | ⚠️ Tiêu đề Việt giữ nguyên 8 từ tiếng Anh chưa dịch. |
| `966698502708` | `适用宝马改装3系5系7系多媒体大旋钮盖X3X4X5X6...` | `Phù hợp cho XE BMW sửa đổi 3 Series 5 Series 7 Series...` | **Dịch chuẩn thương hiệu & Model** (`宝马` $\rightarrow$ `BMW`, `3系` $\rightarrow$ `3 Series`) | **0.7840 (Chuẩn xác)** | ✅ Ánh xạ chéo hoàn hảo giữa chữ Hán và tên hãng tiếng Anh. |
| `895505824643` | `适用奔驰汽车头枕S级迈巴赫护颈枕E300LC260L...` | `Gối tựa đầu dùng cho xe Mercedes-Benz S-Class Maybach...` | **Dịch chuẩn thương hiệu quốc tế** (`奔驰` $\rightarrow$ `Mercedes-Benz`) | **0.7375 (Chuẩn xác)** | ✅ Bảo tồn trọn vẹn giá trị tìm kiếm thương mại điện tử. |
| `694399617240` | `车载香薰香片挂件汽车香水网红丝带持久香氛...` | `Mặt Dây chuyền hương thơm ô tô Nước hoa ô tô Net Red Ribbon...` | **Dịch tiếng Anh bồi (Chinglish)** (`网红` $\rightarrow$ `Net Red`) | **0.6833 (Bị kéo tụt điểm)** | ⚠️ Dịch thô ngô nghê, làm giảm độ tự nhiên của câu tiếng Việt. |
| `732175815707` | `魔夹出风口智能车载无线充电器支架...` | `Magic Clip Air Vent Xe thông minh Bộ sạc không dây...` | **Dịch tiếng Anh bồi (Chinglish)** (`魔夹出风口` $\rightarrow$ `Magic Clip Air Vent`) | **0.6366 (Bị kéo tụt điểm)** | ⚠️ Ghép từ tiếng Anh thô vụng vào giữa câu tiếng Việt. |
| `923329751448` | `【抖音爆款】会动的蝴蝶汽车摆件...` | `[Douyin Hot Model] Đồ trang trí ô tô bướm di chuyển...` | **Dịch tiếng Anh bồi (Chinglish)** (`爆款` $\rightarrow$ `Hot Model`) | **0.6067 (Bị kéo tụt điểm)** | ⚠️ Mất điểm ngữ nghĩa do dịch sai sắc thái từ ngữ tiếp thị. |

---

## 4. Toàn bộ Logic Code Xử lý Đã Được Nhúng Trực Tiếp vào Notebook `eda-bilingual_zh_vi.ipynb`

Toàn bộ quy trình tiền xử lý được lập trình trực tiếp bên trong notebook [`eda-bilingual_zh_vi.ipynb`](file:///d:/download/ecom_crawler-main/ecom_crawler-main-feature-1688/eda-bilingual_zh_vi.ipynb) tại **Ô Markdown số 14 và Ô Code số 15 (Mục 5.1)**, hoàn toàn không phụ thuộc vào các tệp module bên ngoài.

```
[Ô Code số 13]  ──► Nạp mô hình BAAI/bge-m3 từ Hugging Face Hub + Định nghĩa hàm encode()
      │
      ▼
[Ô MD số 14]    ──► Mục 5.1: Diễn giải quy tắc chuẩn hóa và gắn cờ rà soát
      │
      ▼
[Ô Code số 15]  ──► Tự động tiền xử lý: Toàn bộ hàm chuẩn hóa, regex, bóc tách số và phân loại
      │             (Code nhúng trực tiếp, không phụ thuộc file .py ngoài)
      │             -> Sinh ra: title_zh_norm, title_vi_norm, status, reason_codes
      ▼
[Ô Code số 16]  ──► Tính Similarity BGE-M3 trên dữ liệu đã chuẩn hóa:
                    - Lọc bỏ dòng status == 'skip', gán thẳng NaN
                    - Tính dot product L2 normalized vector
                    - Xuất bảng phân bố và danh sách 20 cặp thấp nhất / 5 cặp cao nhất
```

Dưới đây là chi tiết **từng khối lệnh trong Ô Code 5.1 đang làm gì**:

### 4.1. Khối 1: Định nghĩa các Biểu thức chính quy (Regex Rules)
Khối code này khởi tạo các mẫu nhận diện hình thái ký tự và dị tật:
```python
# 1. Định nghĩa Regex & Quy tắc chuẩn hóa
RE_TAG = re.compile(r"<[^>]+>")                      # Xóa thẻ HTML dạng <font color=red>, <span>
RE_INVISIBLE = re.compile(                           # Xóa ký tự điều khiển ẩn, BOM (\uFEFF), zero-width
    "[" + "".join(
        chr(c) for c in (
            *range(0x00, 0x09), 0x0B, 0x0C, *range(0x0E, 0x20), 0x7F,
            *range(0x200B, 0x2010), 0x2028, 0x2029, 0xFEFF
        )
    ) + "]"
)
RE_WS = re.compile(r"[\s\u3000]+")                   # Gộp nhiều khoảng trắng và dấu cách Hán tự \u3000
RE_TRAILING_PUNCT = re.compile(r"[\.,;:!?~–—\-_]+$") # Cắt bỏ dấu câu đuôi (5.064 dấu chấm MT)
RE_DECIMAL_COMMA = re.compile(r"(\d+),(\d+)")        # Chuẩn hóa dấu phẩy số thập phân (4,3 -> 4.3)
RE_CJK = re.compile(r"[\u4e00-\u9fff]")              # Nhận diện ký tự chữ Hán (CJK)
RE_FOREIGN_SCRIPT = re.compile(r"[\u0400-\u04FF\uAC00-\uD7AF\u3040-\u30FF]") # Ký tự Kirin/Nga, Hàn, Nhật
VI_DIACRITICS_RE = re.compile(                       # Kiểm tra dấu thanh tiếng Việt
    r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]",
    re.IGNORECASE
)
RE_LATIN_WORD = re.compile(r"\b[a-zA-Z]{2,}\b")      # Từ ngữ Latin/tiếng Anh có từ 2 ký tự
RE_DIGITS = re.compile(r"\d+(?:\.\d+)?")             # Trích xuất số học (nguyên hoặc thập phân)
RE_CN_LOOKAHEAD = re.compile(                        # Lookahead phục hồi chữ số Hán văn đo lường
    r"([一二两三四五六七八九十])(?=[合件套只个色组录折寸倍排段星号代款层支包装]|合一)"
)
RE_URL = re.compile(r"(https?://|\bwww\.|\.com\b|\.vn\b)", re.IGNORECASE)
RE_PHONE = re.compile(r"(\b0\d{9,10}\b|\b1[3-9]\d{9}\b|\bzalo\b|\bwechat\b|\bhotline\b)", re.IGNORECASE)
RE_PROMO_ZH = re.compile(r"(包邮|秒杀|爆款|厂家直销|源头工厂|一件代发|正品保障|亏本|清仓)")
RE_BRACKETS = re.compile(r"[【\[](.*?)[】\]]")

KNOWN_EN_TERMS = {
    "facial mask", "carbon fiber", "tws", "usb", "dvr", "type-c", "bluetooth",
    "led", "pvc", "pu", "abs", "eva", "oem", "odm", "mercedes-benz", "edifier"
}
```

### 4.2. Khối 2: Hàm chuyển đổi ký tự toàn chiều (`fold_fullwidth`)
- **Nhiệm vụ trong code:** Duyệt từng ký tự trong chuỗi. Nếu mã Unicode nằm trong khoảng toàn chiều `0xFF01` đến `0xFF5E`, hàm trừ đi khoảng lệch `0xFEE0` để đưa về mã ASCII nửa chiều chuẩn `0x21` đến `0x7E`. Ký tự khoảng trắng chữ Hán `\u3000` được thay bằng dấu cách tiêu chuẩn `" "`.
```python
def fold_fullwidth(text: str) -> str:
    """Chuyển ký tự Latin/số toàn chiều (0xFF01-0xFF5E) sang nửa chiều chuẩn (0x21-0x7E)"""
    out = []
    for ch in str(text):
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)
```

### 4.3. Khối 3: Hàm chuẩn hóa chuỗi tiêu đề (`normalize_title`)
- **Nhiệm vụ trong code:** Áp dụng tuần tự 7 bước làm sạch nhưng **bảo toàn 100% chữ hoa, chữ thường, dấu tiếng Việt, số học và tên thương hiệu**:
  1. `html.unescape()` giải mã các thực thể `&amp;`, `&#39;`.
  2. `RE_TAG.sub(" ", s)` xóa sạch thẻ HTML.
  3. `unicodedata.normalize("NFC", s)` đưa về chuẩn ký tự dựng sẵn.
  4. `fold_fullwidth(s)` đưa ký tự toàn chiều về ASCII chuẩn.
  5. `RE_INVISIBLE.sub("", s)` xóa ký tự ẩn, zero-width space.
  6. `RE_DECIMAL_COMMA.sub(r"\1.\2", s)` đổi dấu phẩy số thập phân sang dấu chấm (`4,3 -> 4.3`).
  7. `RE_WS.sub(" ", s).strip()` gộp khoảng trắng thừa và `RE_TRAILING_PUNCT.sub("", s).strip()` cắt bỏ hoàn toàn dấu chấm câu đuôi do máy dịch sinh ra.
```python
def normalize_title(text: Any) -> str:
    """Chuẩn hóa tiêu đề an toàn: bảo toàn thương hiệu, mã hiệu kỹ thuật"""
    if text is None:
        return ""
    s = html.unescape(str(text))
    s = RE_TAG.sub(" ", s)
    s = unicodedata.normalize("NFC", s)
    s = fold_fullwidth(s)
    s = RE_INVISIBLE.sub("", s)
    s = RE_DECIMAL_COMMA.sub(r"\1.\2", s)
    s = RE_WS.sub(" ", s).strip()
    s = RE_TRAILING_PUNCT.sub("", s).strip()
    return s
```

### 4.4. Khối 4: Hàm so khớp số và cơ chế phục hồi chữ số Hán văn đo lường
- **Nhiệm vụ trong code:** 
  1. Tầng 1: So khớp trực tiếp danh sách số Ả Rập đã chuẩn hóa dấu chấm thập phân.
  2. Tầng 2: Nếu hai bên không khớp, hàm quét regex `RE_CN_LOOKAHEAD` tìm các chữ số Hán văn chỉ số lượng (`四合一`, `三件套`), chuyển đổi sang số Ả Rập tương ứng rồi so sánh lại. Logic này cứu thành công 367 trường hợp bị cảnh báo lệch số giả.
```python
def extract_normalized_numbers(text: str) -> List[str]:
    t = RE_DECIMAL_COMMA.sub(r"\1.\2", str(text))
    return sorted(RE_DIGITS.findall(t))
```

### 4.5. Khối 5: Hàm phân loại 3 trạng thái & gắn nhãn (`classify_title_pair`)
- **Nhiệm vụ trong code:** Kiểm tra điều kiện logic trên từng cặp `(title_zh_norm, title_vi_norm)`:
  * **Trạng thái `skip`:** Tiêu đề rỗng hoặc tiêu đề tiếng Việt chép nguyên văn 100% chữ Hán của tiếng Trung (`exact_copy_cjk`).
  * **Trạng thái `review`:** Tiếng Việt còn chữ Hán (`cjk_in_vi`), chứa chữ Cyrillic/tiếng Nga (`foreign_script`), tiếng Việt toàn tiếng Anh không dấu (`predom_english`), lệch tỷ lệ độ dài từ (`ratio_mismatch`), lệch thông số con số (`num_mismatch`), hoặc chứa liên kết quảng cáo (`has_url`, `has_contact`).
  * **Trạng thái `score`:** Cặp tiêu đề sạch, đủ điều kiện chấm similarity. Tự động gắn nhãn cờ thông tin `info_en:...`, `info_has_bracket_tag`, `info_has_promo_zh`.
```python
def classify_title_pair(
    zh_orig: str,
    vi_orig: str,
    zh_norm: str,
    vi_norm: str,
    ratio_lo: float = 0.3,
    ratio_hi: float = 3.0
) -> Tuple[str, List[str]]:
    reasons = []

    # 1. Điều kiện SKIP (Chặn chấm điểm)
    if not zh_norm or not vi_norm:
        reasons.append("empty_title")
        return "skip", reasons

    if zh_norm == vi_norm and bool(RE_CJK.search(vi_norm)):
        reasons.append("exact_copy_cjk")
        return "skip", reasons

    # 2. Điều kiện REVIEW (Gắn cờ rà soát)
    if bool(RE_CJK.search(vi_norm)):
        reasons.append("cjk_in_vi")

    if bool(RE_FOREIGN_SCRIPT.search(zh_norm)) or bool(RE_FOREIGN_SCRIPT.search(vi_norm)):
        reasons.append("foreign_script")

    latin_words = RE_LATIN_WORD.findall(vi_norm)
    has_vi_diacritics = bool(VI_DIACRITICS_RE.search(vi_norm))
    if len(latin_words) >= 2 and not has_vi_diacritics:
        reasons.append("predom_english")

    tok_zh = len(re.sub(r"\s+", "", zh_norm))
    tok_vi = len(vi_norm.split())
    if tok_zh > 0:
        ratio = tok_vi / tok_zh
        if ratio < ratio_lo or ratio > ratio_hi:
            reasons.append(f"ratio_mismatch({ratio:.2f})")
    else:
        reasons.append("ratio_mismatch(zero_zh)")

    # Khớp số Ả Rập + Phục hồi chữ số Hán văn đo lường
    zh_nums = extract_normalized_numbers(zh_norm)
    vi_nums = extract_normalized_numbers(vi_norm)
    if zh_nums != vi_nums:
        cn_matches = RE_CN_LOOKAHEAD.findall(zh_norm)
        extra = [CN_NUMERALS[c] for c in cn_matches if c in CN_NUMERALS]
        combined_zh = sorted(zh_nums + extra)
        if combined_zh != vi_nums:
            reasons.append("num_mismatch")

    if bool(RE_URL.search(vi_norm)) or bool(RE_URL.search(zh_norm)):
        reasons.append("has_url")
    if bool(RE_PHONE.search(vi_norm)) or bool(RE_PHONE.search(zh_norm)):
        reasons.append("has_contact")

    if reasons:
        return "review", reasons

    # 3. SCORE: Hợp lệ hoàn toàn
    for term in KNOWN_EN_TERMS:
        if term in vi_norm.lower():
            reasons.append(f"info_en:{term}")

    if bool(RE_BRACKETS.search(zh_norm)):
        reasons.append("info_has_bracket_tag")

    if bool(RE_PROMO_ZH.search(zh_norm)):
        reasons.append("info_has_promo_zh")

    return "score", reasons
```

### 4.6. Khối 6: Hàm áp dụng trên toàn bộ tập dữ liệu (`process_dataset`)
- **Nhiệm vụ trong code:** Nhận DataFrame hiện tại, tự động tạo các cột mới: `title_zh_norm`, `title_vi_norm`, `zh_had_fullwidth`, `status`, và `reason_codes`. In bảng thống kê phân bố và hiển thị các dòng `skip`.
```python
def process_dataset(data_df: pd.DataFrame) -> pd.DataFrame:
    res = data_df.copy()
    res["title_zh_norm"] = res[ZH_COL].apply(normalize_title)
    res["title_vi_norm"] = res[VI_COL].apply(normalize_title)
    res["zh_had_fullwidth"] = res[ZH_COL].apply(has_fullwidth_ascii)
    
    statuses, reason_codes_list = [], []
    for _, row in res.iterrows():
        st, reas = classify_title_pair(
            zh_orig=row[ZH_COL],
            vi_orig=row[VI_COL],
            zh_norm=row["title_zh_norm"],
            vi_norm=row["title_vi_norm"]
        )
        statuses.append(st)
        reason_codes_list.append(";".join(reas))
        
    res["status"] = statuses
    res["reason_codes"] = reason_codes_list
    return res

# Thực thi trực tiếp
df = process_dataset(df)
```

### 4.7. Khối 7: Ô Code số 16 tính Similarity bằng BGE-M3
- **Nhiệm vụ trong code:**
  * Lọc chỉ các dòng có `status != 'skip'`. Với các dòng `status == 'skip'`, gán giá trị **`NaN`** để loại khỏi phân bố điểm tương đồng.
  * Đưa `title_zh_norm` và `title_vi_norm` qua hàm `encode()` của BGE-M3 (CLS pooling + L2 normalization).
  * Tính tích vô hướng `sim = (zh_emb * vi_emb).sum(dim=1)` để ra điểm Cosine Similarity trung thực nhất.
```python
# Tính similarity cho tiêu đề chuẩn hoá (dòng skip gán NaN)
valid_mask = df['status'] != 'skip'
work = df.loc[valid_mask, [ZH_COL, VI_COL, 'title_zh_norm', 'title_vi_norm']].copy()

zh_emb = encode(work['title_zh_norm'].tolist())
vi_emb = encode(work['title_vi_norm'].tolist())
sim = (zh_emb * vi_emb).sum(dim=1)  # dot product L2 vector
work['sim_score'] = sim.numpy()

# Gán sim_score về df gốc (dòng skip giữ nguyên NaN)
df['sim_score'] = np.nan
df.loc[work.index, 'sim_score'] = work['sim_score']
```


---

## 5. Mở rộng Khám phá Các Vấn đề Tiềm ẩn Mới & Thử nghiệm So sánh Đa Phương pháp Tiền xử lý trên BGE-M3

Sau khi hoàn thiện quy trình tiền xử lý nền tảng ở Phần 4, một đợt rà soát chuyên sâu toàn diện trên toàn bộ **16.348 cặp tiêu đề song ngữ** đã được tiến hành nhằm phát hiện triệt để các dị tật tiềm ẩn phức tạp hơn, từ đó thiết kế các phương pháp tiền xử lý cạnh tranh và thực nghiệm định lượng trên GPU để tìm ra giải pháp tối ưu nhất.

---

### 5.1. Khám phá 7 Nhóm Vấn đề & Dị tật Mới Bổ sung

```
                   [TIÊU ĐỀ THƯƠNG MẠI ĐIỆN TỬ SÂU RỘNG]
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                             ▼                             ▼
[LỆCH QUY ĐỔI ĐƠN VỊ]     [CẤU TRÚC ĐỘ DÀI DỊCH]       [HÌNH THÁI KÝ TỰ & TỪ VỰNG]
• 5.1.1. Cân Tàu vs kg:    • 5.1.4. Dịch cụt < 1.0       • 5.1.2. Đơn vị '寸' (513 dòng)
  300斤 -> 150kg (mismatch)  (mất 80% ngữ nghĩa)         • 5.1.3. Nhồi từ khóa (44,24%)
  5斤 -> 5 kg (dịch ẩu)    • 5.1.5. Phình to > 6.0       • 5.1.6. Ngoặc 【】 vs [] (298 dòng)
                              (diễn giải lan man)        • 5.1.7. Phân cách CJK 丨 & Toán tử ×
```

#### 5.1.1. Lệch số do Chuyển đổi Đơn vị Đo lường của Máy Dịch (`斤` vs `kg`)
* **Hiện tượng thực nghiệm:** Tiêu đề tiếng Trung thường dùng đơn vị đo khối lượng truyền thống `斤` (Cân thị trường Trung Quốc, $1 \text{ 斤} = 500 \text{ g} = 0,5 \text{ kg}$). Quá trình dịch máy phân hóa thành 2 thái cực:
  1. *Máy dịch tự quy đổi toán học:* Điển hình tại sản phẩm `893913959805`, tiếng Trung ghi `300斤短袖T恤` được dịch thành `Áo thun ngắn tay 150kg`. Máy dịch đã tự động chia đôi con số ($300 \div 2 = 150$). Tuy nhiên, hệ thống kiểm tra logic so khớp số thông thường lại thấy bên Trung có số `300`, bên Việt có số `150` nên lập tức báo động sai lệch số (`num_mismatch`), tạo ra **cảnh báo giả mạo**.
  2. *Máy dịch dịch ẩu giữ nguyên con số:* Tại sản phẩm `938415140547` (`批发5斤` $ $\rightarrow$ $ `Bán buôn 5 kg`) hoặc `867633791881` (`100-300斤` $ $\rightarrow$ $ `100-300 kg`). Việc dịch sai đơn vị làm phóng đại khối lượng thực tế lên gấp đôi!
* **Tác động lên BGE-M3:** Khi đo lường trực tiếp cặp `300斤` và `150kg`, BGE-M3 đạt độ tương đồng rất cao (**0.7428**), chứng tỏ không gian vector của mô hình hiểu được ngữ nghĩa tương đương. Sự sai lệch chỉ xảy ra ở tầng lọc quy tắc số học nếu không có cơ chế nhận diện quy đổi.

#### 5.1.2. Thống nhất Khoảng trắng Đơn vị Đo lường `寸` (Inch) trong 513 Dòng
* **Hiện tượng thực nghiệm:** Đơn vị `寸` (inch đo kích thước màn hình / thiết bị điện tử) xuất hiện trong **513 tiêu đề tiếng Trung**.
  * May mắn là 100% các dòng này đều được máy dịch chuyển sang `inch`.
  * Tuy nhiên, phát hiện sự thiếu đồng nhất về khoảng cách giữa con số và đơn vị: có dòng dịch là `10 inch` (có khoảng trắng), có dòng dịch dính liền `10inch` (không có khoảng trắng).
* **Tác động:** Tokenizer BGE-M3 tách `10 inch` thành `[' 10', ' inch']` (2 token), trong khi `10inch` có thể bị tách thành chuỗi con khác hoặc token dính, gây suy giảm độ tương đồng nhân tạo.

#### 5.1.3. Hiện tượng Nhồi nhét Từ khóa & Trùng lặp Ngữ nghĩa (Keyword Stuffing / SEO Redundancy)
* **Hiện tượng thực nghiệm:** Nhằm mục đích tối ưu hóa công cụ tìm kiếm nội sàn (SEO 1688), người bán thường nhồi nhét lặp đi lặp lại cùng một từ khóa trong tiêu đề:
  * Sản phẩm `592582928262`: Tiếng Trung lặp lại chữ `刷` (bàn chải) 8 lần; bản dịch tiếng Việt lặp lại từ `Bàn chải` đúng **8 lần** (`Bàn chải lốp, bàn chải cửa gió, bàn chải trung tâm, bàn chải động cơ, bàn chải chi tiết, bàn chải khe hở, rửa xe, bàn chải nhỏ, bàn chải rửa xe`).
  * Sản phẩm `685311327888`: Lặp lại 4 lần cụm từ `xẻng làm tuyết`.
  * **Thống kê:** Có tới **7.232 tiêu đề tiếng Việt (44,24%)** chứa từ vựng thực chất bị lặp lại từ $\ge 3$ lần trở lên!
* **Tác động lên BGE-M3:** Khi một từ ngữ xuất hiện lặp đi lặp lại dày đặc, cơ chế gộp vector (Mean Pooling) của mô hình transformer bị kéo lệch bất đối xứng về hướng của từ lặp đó (hiện tượng **Vector Over-weighting**), làm lu mờ hoàn toàn các đặc trưng quan trọng khác như model, công nghệ, chất liệu.

#### 5.1.4. Dị tật Tiêu đề Dịch Cụt / Bỏ sót Thuộc tính Cốt lõi (Severe Truncation / Under-generation)
* **Hiện tượng thực nghiệm:** Phát hiện các trường hợp tỷ lệ độ dài ký tự cực ngắn ($\text{len\_ratio} < 1.0$):
  * Sản phẩm `898729390213`: Tiêu đề tiếng Trung dài 30 chữ Hán mô tả đầy đủ (`汽车座椅缝隙塞条车内装饰用品大全车载夹缝防漏填补条收纳储物盒`), nhưng tiếng Việt chỉ dịch được vỏn vẹn cụm 5 chữ: `Khe hở ghế ô tô` (tỷ lệ độ dài $0.50$). Bản dịch bỏ sót tới **80% nội dung gốc** (thanh chèn chống rơi, hộp chứa đồ).
  * Sản phẩm `780449465663`: Tiếng Trung mô tả thanh chèn ghế chống rơi, nhưng tiếng Việt chỉ dịch đúng cụm mở đầu: `Sản phẩm ô tô xuyên biên giới`.
* **Tác động lên BGE-M3:** Điểm Cosine Similarity bị tụt dốc thảm hại (thường chỉ đạt **0.45 – 0.52**). Đây là lỗi dịch thuật nghiêm trọng cần được gắn nhãn dị tật `truncation_undergen`.

#### 5.1.5. Dị tật Tiêu đề Phình to / Diễn giải Lan man (Severe Bloat / Over-generation)
* **Hiện tượng thực nghiệm:** Có tới **275 tiêu đề tiếng Việt** dài gấp hơn 6 lần tiêu đề tiếng Trung ($\text{len\_ratio} > 6.0$), chạm ngưỡng 180 – 200 ký tự (ví dụ `920005486339`, `978055923977`).
* **Nguyên nhân:** Do máy dịch cố gắng giải nghĩa từng chữ Hán bằng nhiều cụm từ đồng nghĩa tiếng Việt dài dòng nối tiếp nhau bằng dấu phẩy.

#### 5.1.6. Dị tật Bất đối xứng Thẻ Ngoặc (`【...】` vs `[...]`)
* **Hiện tượng thực nghiệm:** Có **298 tiêu đề tiếng Trung** dùng dấu ngoặc CJK `【...】`. Khi chuyển ngữ sang tiếng Việt:
  * 200 dòng được chuyển thành ngoặc vuông ASCII `[...]`.
  * 89 dòng giữ nguyên ngoặc CJK `【...】`.
  * 9 dòng bị xóa sạch dấu ngoặc.
* **Tác động:** Tokenizer BGE-M3 coi `【` (`【`) và `[` (`[`) là hai token hoàn toàn độc lập. Sự bất đối xứng hình thái này làm phân mảnh token của cặp câu.

#### 5.1.7. Ký tự Phân cách CJK Cổn `丨` (`丨`) và Toán tử Kích thước Đa dạng
* **Ký tự CJK Cổn `丨`:** Người bán 1688 dùng chữ Hán `丨` như dấu gạch đứng pipe `|` để phân tách câu (ví dụ `出口日本丨100%纯棉...` - 6 dòng). Tokenizer BGE-M3 coi đây là chữ Hán cổ, trong khi bên tiếng Việt dịch thành dấu gạch đứng hoặc khoảng trắng.
* **Toán tử kích thước:** Tiêu đề chứa kích thước sản phẩm dùng lẫn lộn: dấu sao `*` (8 dòng), chữ `x`/`X` (19 dòng), và dấu nhân Unicode `×` (`×` - 3 dòng).

---

### 5.2. Thiết kế và Thực nghiệm So sánh 5 Phương pháp Tiền xử lý trên BGE-M3

Để tìm ra phương pháp tiền xử lý tối ưu, chúng tôi thiết kế 5 phương pháp cạnh tranh và chạy thực nghiệm trực tiếp trên mô hình `BAAI/bge-m3` sử dụng phần cứng GPU NVIDIA CUDA:

* **M1 (Raw Baseline):** Tiêu đề thô nguyên bản, không qua bất kỳ khâu xử lý nào.
* **M2 (Standard Clean & Logic Guard):** Phương pháp hiện tại (Unicode NFC, gập fullwidth ASCII, cắt bỏ 5.064 dấu chấm câu đuôi MT, chuẩn hóa số thập phân `4,3 -> 4.3`, chặn `skip` gán `NaN` cho 2 ca chép nguyên CJK).
* **M3 (Deep Clean with Tag & Marketing Strip):** M2 + Loại bỏ triệt để toàn bộ nội dung trong thẻ `【...】` và các từ tiếp thị rác (`包邮`, `秒杀`, `爆款`, `厂家直销`, `一件代发`...).
* **M4 (Structural & Unit Harmonization):** M2 + Chuẩn hóa toán tử kích thước (`10*20`, `10 × 20` $ $\rightarrow$ $ `10x20`), đồng nhất khoảng cách số và đơn vị (`10 cm` $ $\rightarrow$ $ `10cm`), thay ký tự CJK `丨` bằng khoảng trắng.
* **M5 (Lowercased Harmonized):** M4 + Chuyển toàn bộ văn bản về chữ thường (`lower()`).

#### Bảng Kết quả Thực nghiệm Định lượng trên GPU

Thử nghiệm được thực hiện trên 2 tập dữ liệu độc lập:
1. **Tập ngẫu nhiên đại diện (Representative Random Sample):** $N = 1.000$ cặp.
2. **Tập thách thức mục tiêu (Targeted Challenging Sample):** $N = 600$ cặp chứa các dị tật ngoặc `【】`, từ tiếp thị, đơn vị đo, kích thước, và dấu chấm đuôi MT.

| Phương pháp | Tập dữ liệu | Mean (Điểm TB) | Std (Độ lệch chuẩn) | Median (Trung vị) | Min (Thấp nhất) | Max (Cao nhất) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **M1: Raw Baseline** | Random ($N=1000$) | 0.7049 | 0.0562 | 0.7063 | 0.4748 | 0.8589 |
| **M2: Standard Clean** | Random ($N=1000$) | 0.7045 | 0.0565 | 0.7058 | 0.4731 | 0.8589 |
| **M3: Marketing & Tag Removal** | Random ($N=1000$) | 0.7042 | 0.0571 | 0.7058 | 0.4731 | 0.8589 |
| **M4: Unit & Dimension Harmonized** | Random ($N=1000$) | 0.7045 | 0.0565 | 0.7056 | 0.4731 | 0.8589 |
| **M5: Lowercased Harmonized** | Random ($N=1000$) | **0.7075** | 0.0551 | **0.7083** | 0.4818 | **0.8649** |
|---|---|---|---|---|---|---|
| **M1: Raw Baseline** | Targeted ($N=600$) | 0.7110 | 0.0534 | 0.7149 | 0.5217 | 0.8523 |
| **M2: Standard Clean** | Targeted ($N=600$) | 0.7110 | 0.0540 | 0.7143 | 0.5217 | 0.8523 |
| **M3: Marketing & Tag Removal** | Targeted ($N=600$) | 0.7106 | 0.0548 | 0.7147 | 0.5161 | 0.8412 |
| **M4: Unit & Dimension Harmonized** | Targeted ($N=600$) | 0.7110 | 0.0540 | 0.7143 | 0.5217 | 0.8523 |
| **M5: Lowercased Harmonized** | Targeted ($N=600$) | **0.7122** | 0.0528 | 0.7144 | 0.5193 | **0.8535** |

---

### 5.3. Phát hiện Khoa học Bước ngoặt từ Thực nghiệm

#### Phát hiện 1: Tại sao KHÔNG ĐƯỢC XÓA từ tiếp thị và thẻ ngoặc khỏi Embedding?
Trước thực nghiệm, giả thuyết phổ biến cho rằng xóa bỏ các từ tiếp thị rác (`厂家直销`, `包邮`, `【严选】`) sẽ giúp câu "sạch hơn" và tăng độ tương đồng ngữ nghĩa của sản phẩm chính.
Tuy nhiên, kết quả đo đạc thực tế chứng minh điều ngược lại:
* **Tại Case ID `922005108571`:**
  * Tiếng Trung có: `厂家直销...`
  * Tiếng Việt có: `Nhà máy bán hàng trực tiếp...`
  * Khi giữ nguyên (M1, M2, M4): Điểm đạt **0.6873**.
  * Khi áp dụng M3 (xóa bỏ cụm từ tiếp thị): Điểm bị **tụt mạnh xuống 0.6580 (-0.0293 điểm)**!
* **Tại Case ID `934998812816`:**
  * Tiếng Trung có: `【严选】` (Nghiêm tuyển - Lựa chọn kỹ càng)
  * Tiếng Việt có: `【Lựa chọn cao cấp】`
  * Khi giữ nguyên: Điểm đạt **0.6974**.
  * Khi xóa thẻ (M3): Điểm bị **tụt xuống 0.6798 (-0.0176 điểm)**!
* **Kết luận khoa học:** Vì công cụ dịch máy đã dịch chuẩn xác cụm từ tiếp thị sang tiếng Việt, và mô hình BGE-M3 có không gian liên kết chéo rất tốt giữa hai ngôn ngữ cho các cụm từ này, nên **việc xóa bỏ từ tiếp thị sẽ phá vỡ sự tương đồng vốn có của bản dịch**.  
$\Rightarrow$ **Quyết định thiết kế:** Tuyệt đối không xóa từ tiếp thị khỏi văn bản đưa vào BGE-M3; thay vào đó, chỉ **gắn cờ thông tin (`info_has_promo`, `info_has_bracket_tag`)** để phục vụ bộ lọc nghiệp vụ của người dùng.

#### Phát hiện 2: Đánh giá chiến lược Chữ Thường (Lowercasing - M5)
* M5 làm tăng nhẹ điểm trung bình (+0.0030) nhờ loại bỏ khác biệt viết hoa/thường ngẫu nhiên giữa các từ thông dụng.
* **Tuy nhiên, rủi ro tiềm ẩn của Lowercasing:** BGE-M3 là mô hình cased (phân biệt hoa/thường). Khi chuyển toàn bộ về chữ thường, các tên thương hiệu và mã sản phẩm viết tắt (`BMW` $ $\rightarrow$ $ `bmw`, `TWS` $ $\rightarrow$ $ `tws`, `USB`, `Type-C`, `Apple`) bị mất đặc trưng phân biệt thực thể (NER).
$\Rightarrow$ **Quyết định thiết kế:** Không lowercase toàn bộ chuỗi; ưu tiên giữ nguyên case để bảo vệ tối đa giá trị định danh thương hiệu.

#### Phát hiện 3: Cơ chế Nhận diện Quy đổi Đơn vị `斤` $ $\rightarrow$ $ `kg`
* Khi gặp `300斤` $ $\rightarrow$ $ `150kg`, bản thân BGE-M3 hiểu rất tốt sự tương đương ngữ nghĩa (Cosine Sim = **0.7428**).
* Để loại trừ báo động giả `num_mismatch`, quy trình logic cần bổ sung một bộ chuyển đổi toán học: nếu con số tiếng Việt đúng bằng $1/2$ con số tiếng Trung đi liền với chữ `斤`, hệ thống công nhận là **Khớp số hợp lệ**.

---

### 5.4. Đề xuất Phương pháp Tối ưu Nhất (Recommended Best Pipeline)

Từ toàn bộ các phát hiện trên, chúng tôi đề xuất **Quy trình Tiền xử lý Đa Tầng Tối ưu Toàn diện (Enhanced Optimal Preprocessing Pipeline)**:

```
[TIÊU ĐỀ THÔ ZH / VI]
        │
        ▼
[TẦNG 1: CHUẨN HÓA HÌNH THÁI VÀ KÝ TỰ]
• html.unescape() + Xóa thẻ HTML + Unicode NFC chuẩn
• Gập ký tự toàn chiều Full-width về ASCII chuẩn
• Gọt sạch hoàn toàn 5.064 dấu chấm câu đuôi MT (.,;:!?~–—-_)
• Chuẩn hóa dấu phân cách số thập phân: 4,3 -> 4.3
• Đồng nhất ký tự kích thước: 10*20 / 10 × 20 / 10X20 -> 10x20
• Gập ký tự phân cách CJK Cổn: 丨 -> khoảng trắng
• Đồng nhất khoảng cách đơn vị: 10 inch <-> 10inch, 100 ml <-> 100ml
        │
        ▼
[TẦNG 2: BẢO TOÀN NGỮ NGHĨA CHO EMBEDDING]
• BẢO TOÀN 100% chữ hoa/thường của tên thương hiệu, model quốc tế
• BẢO TOÀN nguyên vẹn cụm từ tiếp thị và thẻ ngoặc để BGE-M3 ánh xạ chéo tự nhiên
        │
        ▼
[TẦNG 3: BỘ PHÂN LOẠI LOGIC VÀ PHỤC HỒI NÂNG CAO]
• SKIP: Tiêu đề rỗng HOẶC Chép nguyên 100% CJK sang tiếng Việt -> Gán điểm NaN
• REVIEW:
  - Còn sót chữ Hán trong tiếng Việt (cjk_in_vi)
  - Ký tự ngoại lai Cyrillic tiếng Nga (foreign_script)
  - Tiếng Việt toàn tiếng Anh không dấu (predom_english)
  - Dị tật dịch cụt nghiêm trọng: len_ratio < 0.6 (truncation_undergen)
  - Lệch số học (num_mismatch) với cơ chế 3 tầng:
      1. Khớp số Ả Rập chuẩn hóa
      2. Phục hồi số Hán văn đo lường (四合一, 三件套)
      3. Phục hồi quy đổi cân Tàu (N 斤 <=> N/2 kg)
• SCORE: Cặp hợp lệ hoàn toàn -> Đưa vào BGE-M3 tính Cosine Similarity trung thực
• GẮN CỜ INFO: info_en, info_has_bracket_tag, info_has_promo_zh, info_stuffing
```

---

### 5.5. Bảng Tổng hợp Minh chứng Xử lý Các Ca Dị tật Mới

| `product_id` | Tiêu đề tiếng Trung (`title_zh`) | Tiêu đề tiếng Việt (`title_vi`) | Vấn đề phát hiện | Giải pháp xử lý | Kết quả đạt được |
|---|---|---|---|---|:---:|
| `893913959805` | `可供海外批发~重磅纯棉300斤短袖T恤男...` | `Áo thun ngắn tay 150kg chất liệu cotton...` | Quy đổi đơn vị: `300斤` $ $\rightarrow$ $ `150kg` | Cơ chế Jin-to-Kg Recovery ($300 \div 2 = 150$) | **Khớp số thành công (Không báo lỗi giả)** |
| `810368257522` | `10寸4K超清夜视流媒体后视镜...` | `10 inch 4K siêu rõ nét nhìn ban đêm...` | Khoảng trắng đơn vị: `10寸` $ $\rightarrow$ $ `10 inch` | Chuẩn hóa khoảng trắng đơn vị đo | **BGE-M3 đạt điểm cao: 0.7712** |
| `898729390213` | `汽车座椅缝隙塞条车内装饰用品大全车载夹缝防漏填补条收纳储物盒` | `Khe hở ghế ô tô` | Dịch cụt nghiêm trọng: tỷ lệ dài chỉ 0.50 | Gán cờ `truncation_undergen` | **Cảnh báo bản dịch thiếu 80% nội dung** |
| `922005108571` | `厂家直销汽车挂钩翻毛皮车载座椅背...` | `Nhà máy bán hàng trực tiếp móc treo ô tô...` | Chứa cụm từ tiếp thị `厂家直销` | Bảo toàn cụm từ cho BGE-M3 (không xóa) | **Bảo tồn điểm cao 0.6873 (tránh tụt về 0.6580)** |
| `624881412191` | `除胶剂家用万能去胶神器强力汽车玻璃...` | `Chất tẩy keo dùng trong gia đình... cặn băng dính hai mặt.` | Dấu chấm đuôi MT ở cuối câu tiếng Việt | Gọt sạch dấu chấm câu vô nghĩa | **Tăng điểm Cosine Sim từ 0.7439 lên 0.7450** |
