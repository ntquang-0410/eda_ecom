# Description song ngữ Trung–Việt đã làm sạch (bản desc_v1, Nhật Anh, 03/10/2026)

*Cleaned Chinese–Vietnamese product attribute descriptions (1688 e-commerce), 16,342 products / 366,016 training pairs. Vietnamese documentation below.*

Bộ này là phần **description** (chuỗi "tên thuộc tính: giá trị") của dataset song ngữ thương mại điện tử 1688, sau khi chạy qua pipeline làm sạch `notebooks/desc-pipeline.ipynb` (repo EDA của nhóm). Đầu vào là snapshot `bilingual_zh_vi.parquet` (16.348 sản phẩm, 406.871 cặp thuộc tính). Bản dịch tiếng Việt gốc là bản dịch máy; pipeline không dịch lại từ đầu mà sửa hoặc loại từng cặp bằng luật, từ điển đã duyệt và phần đọc thủ công.

## Các file

| File | Mỗi dòng | Số dòng | Dung lượng |
|---|---|---|---|
| `bilingual_zh_vi_desc_cleaned(nhatanh).parquet` | 1 sản phẩm | 16.342 | 3,0 MB |
| `bilingual_zh_vi_desc_pairs(nhatanh).parquet` | 1 cặp thuộc tính | 405.275 | 7,2 MB |

Mã SHA256 nằm trong `SHA256SUMS_desc(nhatanh).txt`. Trên Hugging Face, bốn file của bản này nằm trong `data/cleaned/` với đuôi `(nhatanh)`, cạnh file title của Huy.

### File sản phẩm (`..._desc_cleaned(nhatanh).parquet`)

| Cột | Ý nghĩa |
|---|---|
| `product_id` | Mã sản phẩm, nối được với dataset gốc |
| `category` | 1 trong 9 nhóm: bags 2.878, mother_baby 2.495, home 2.493, shoes 2.310, food 1.420, fashion 1.350, beauty 1.317, electronics 1.108, auto 971 |
| `description_zh_sach`, `description_vi_sach` | Description dựng lại từ các cặp dùng được cho train, giữ thứ tự gốc, dạng `tên: giá trị; tên: giá trị` |
| `so_cap_giu`, `so_cap_loai` | Số cặp giữ lại và số cặp bị loại của sản phẩm (trung bình giữ 22,4 cặp) |
| `trung_y_het` | 1 nếu cả hai description trùng y hệt một sản phẩm đứng trước (165 sản phẩm). Nên bỏ khi train |
| `nhom_trung` | Mã nhóm sản phẩm giống nhau (cùng title, cùng shop, cùng description hoặc gần trùng MinHash, Jaccard ≥ 0,8). Có 6.566 nhóm, nhóm lớn nhất 629 sản phẩm. **Khi chia train/dev/test, mọi sản phẩm cùng nhóm phải nằm cùng một tập** |

6 sản phẩm không ghép được cặp zh–vi (mã `DS02`) đã bị bỏ khỏi file này.

#### Cột `nhom_trung` (chi tiết)

**Là gì.** Mã nhóm của những sản phẩm "họ hàng" với nhau: cùng một mẫu hàng do nhiều shop đăng, hoặc một shop đăng nhiều biến thể na ná nhau. Giá trị chỉ là mã (số nguyên), không mang ý nghĩa lớn nhỏ. Mọi sản phẩm cùng giá trị `nhom_trung` thuộc cùng một nhóm. Sản phẩm không giống ai thì đứng một mình một nhóm.

**Đánh giá trùng như thế nào.** Hai sản phẩm được nối với nhau khi có ít nhất **một** điểm chung:

| Tiêu chí | Cách so |
|---|---|
| Cùng title | `title_zh` giống hệt |
| Cùng shop | tên shop giống hệt (cột shop không có trong bản công bố, chỉ dùng nội bộ để gộp nhóm) |
| Cùng description | `description_zh_sach` giống hệt |
| Description gần giống | MinHash trên `description_zh_sach` (cắt thành các đoạn 5 ký tự, 128 hàm băm, tìm ứng viên bằng LSH), nối khi độ giống Jaccard ≥ 0,8, tức khoảng 80% các đoạn trùng nhau |

Việc nối có tính **bắc cầu** (union-find): A nối với B, B nối với C thì A, B, C chung một nhóm, kể cả khi A và C không giống nhau trực tiếp.

Cột `trung_y_het` là trường hợp hẹp hơn: description sạch giống hệt **cả bản Trung lẫn bản Việt** với một sản phẩm đứng trước. Một sản phẩm có `trung_y_het = 1` luôn nằm cùng nhóm với bản gốc của nó.

**Ví dụ** (một nhóm 3 sản phẩm, category bags; tên shop không công bố nên ghi shop 1, shop 2):

| product_id | Title (rút gọn) | Shop |
|---|---|---|
| 759616415977 | 双肩包男士简约大容量旅行背包女休闲日系… | shop 1 |
| 982784765319 | 双肩包男士简约大容量旅行背包女休闲日系… (giống hệt title trên) | shop 2 |
| 759430336752 | 书包女ins风初中生纯色背包2022新款… (title khác) | shop 1 |

- Sản phẩm 1 và 2 **cùng title** (hai shop đăng cùng một mẫu balo) → nối.
- Sản phẩm 1 và 3 **cùng shop** → nối.
- Kết quả: cả 3 chung một `nhom_trung`, dù sản phẩm 2 và 3 không giống nhau trực tiếp.

**Phân bố.** 16.342 sản phẩm gom thành **6.566 nhóm**:

| Cỡ nhóm | Số nhóm |
|---|---|
| 1 sản phẩm | 4.121 |
| 2–5 sản phẩm | 1.951 |
| 6–50 sản phẩm | 485 |
| trên 50 sản phẩm | 9 |

Nhóm lớn nhất có **629 sản phẩm** (71 shop, 568 title khác nhau; home 334, mother_baby 135, electronics 72). Nhóm này to vì nối bắc cầu: shop A có chung một mẫu với shop B, shop B có description gần giống shop C, cứ thế lan ra; nhiều sản phẩm còn có description ngắn na ná nhau (kiểu "Thương hiệu: Khác; Xuất xứ: …"). Cách gộp này **rộng có chủ đích**: thà gộp dư còn hơn để lọt sản phẩm gần giống sang hai tập khác nhau. Nếu sau này nhóm lớn gây khó khi chia tập, có thể nới tiêu chí (ví dụ bỏ điều kiện "cùng shop") rồi tính lại.

**Mục đích.**
1. **Chia train/dev/test không lộ đề (chính).** Nếu sản phẩm 1 ở train mà sản phẩm 2 (cùng title) ở test, mô hình đã gặp câu gần giống khi train nên điểm test cao giả. Chia theo `nhom_trung` (cả nhóm nằm trọn một tập) tránh được điều này. Bản này **chưa chia tập**; cột chỉ chuẩn bị sẵn.
2. **Giảm mẫu trội.** Giới hạn mỗi nhóm tối đa N sản phẩm, hoặc giảm trọng số nhóm lớn, để vài mẫu hàng được đăng lại nhiều lần không lấn át phần còn lại (charter §6).
3. **Đo độ đa dạng thật.** Số mẫu hàng khác nhau thật sự là khoảng 6.566 nhóm, không phải 16.342 sản phẩm; nên ghi con số này khi báo cáo quy mô dữ liệu.
4. **Chọn mẫu để người kiểm hoặc dịch lại.** Lấy mỗi nhóm một sản phẩm thì cùng số mẫu mà phủ được nhiều loại hàng hơn.
5. **Tìm trùng lặp còn sót.** Xem các nhóm lớn để biết shop nào đăng hàng loạt sản phẩm na ná nhau.

### File cặp (`..._desc_pairs(nhatanh).parquet`)

| Cột | Ý nghĩa |
|---|---|
| `pair_id`, `product_id`, `category`, `vi_tri` | Định danh cặp, sản phẩm và vị trí trong description |
| `name_zh`, `name_vi` | Tên thuộc tính (bản Việt đã chuẩn hoá) |
| `gia_tri_zh`, `gia_tri_vi` | Giá trị đã làm sạch (nhiều giá trị nối bằng ", ") |
| `vai_tro` | `mo_ta_hang` (mô tả hàng), `giao_dich_1688` (điều khoản giao dịch), `lua_chon_sku`, `ma_giay_to` (mã, giấy phép), `thuong_hieu` |
| `ly_do` | Mã lý do loại, nối bằng ";". Rỗng là dùng được |
| `nhan` | Nhãn chỉ để biết, không loại cặp |
| `dung_cho_train` | 1 khi `ly_do` rỗng (366.016 cặp) |

**Mã lý do** (một cặp có thể có nhiều mã):

| Mã | Số cặp | Ý nghĩa |
|---|---|---|
| `T2_CHUA_KIEM` | 36.184 | Có giá trị hiếm chưa kiểm được nghĩa (thử Qwen2.5-7B làm bộ lọc nhưng không đạt) |
| `DS07` | 2.762 | Bản Việt còn chữ Hán chưa dịch |
| `RONG` | 1.369 | Giá trị chỉ có ký hiệu, không có nội dung |
| `DS10` | 554 | Số trong bản Việt lệch bản Trung |
| `DS17` | 110 | Lỗi sai nghĩa đã biết nằm trong chuỗi dài, không sửa an toàn được |
| `DS09_dai` | 74 | Tên hãng quá dài hoặc là tên công ty |
| `DS03` | 62 | Danh sách lệch thứ tự không sắp lại được |
| `DS18` | 3 | Địa danh dịch sai |
| `DS25` | 2 | Cặp trùng trong cùng sản phẩm |

**Nhãn:**
- `chua_kiem_nghia` (36.200): chưa kiểm nghĩa. Không nên dùng làm tập test.
- `it_thong_tin` (19.106): giá trị ít thông tin như "Khác", "Không có".
- `chep_ma` (16.694): mã được chép nguyên từ bản Trung.
- `vai_tro_tam` (399.563): vai trò lấy theo luật/bảng và được kiểm bằng mẫu, không duyệt từng tên thuộc tính.

## Những gì đã làm

Pipeline có 13 bước, mỗi bước có cổng duyệt bằng mẫu ngẫu nhiên. Tất cả cổng đã đạt; mẫu cuối 200 sản phẩm không có lỗi.

1. Tách description thành cặp và ghép zh–vi theo thứ tự đoạn.
2. Làm sạch ký tự: 【】 → [ ], dấu câu Trung → dấu thường, khoảng trắng thừa.
3. Thông tin cá nhân:
   - bỏ các cặp liên hệ và cặp tên/địa chỉ nhà sản xuất (cả hai **không có** trong file công bố);
   - che số điện thoại, email, URL, WeChat/QQ bằng nhãn `[SĐT]`, `[EMAIL]`, `[URL]`, `[LIÊN HỆ]`;
   - bỏ cột `shop`.
4. Gắn vai trò thuộc tính; dịch chuẩn các giá trị rỗng nghĩa (其他 → "Khác", 无 → "Không có").
5. Bỏ số đuôi do máy dịch thêm ("Không 2" → "Không").
6. Từ điển 300 tên thuộc tính phổ biến.
7. Từ điển giá trị, luật ngữ cảnh, địa danh, tên hãng (chữ Latin giữ nguyên, còn lại chuyển pinyin), thuật ngữ thống nhất với title.
8. Số và đơn vị:
   - mã giấy phép chép nguyên; mã còn chữ Hán có chú thích tỉnh sau mã, ví dụ `粤妆20170110 (Quảng Đông)`;
   - ghép lại số thập phân bị tách, sửa năm, sắp lại danh sách lệch thứ tự.
9. Định dạng: bỏ dấu chấm cuối do máy thêm, bỏ IN HOA toàn bộ, đổi "7,5" → "7.5" khi bản Trung ghi 7.5.
10. Giá trị hiếm:
    - tầng 1: 4.006 cặp được Claude (Claude Code) đọc từng cặp, 1.151 cặp sai được sửa; người kiểm 30 mẫu SAI + 30 mẫu DUNG, 0 sai;
    - tầng 2: khoảng 130 nghìn cặp quá hiếm, gắn `T2_CHUA_KIEM`.
11. Loại các cặp còn lỗi bằng mã lý do.
12. Đánh dấu trùng lặp và nhóm sản phẩm giống nhau.

## Vì sao thử bge-m3 và CometKiwi, và vì sao không dùng để lọc

**CometKiwi và bge-m3 không được dùng để sửa hay lọc dữ liệu trong bản này.** Đây là thí nghiệm đo thử. Kết quả chưa đủ tốt nên dữ liệu giữ nguyên.

### Vấn đề

Sau khi làm sạch còn **36.184 cặp (8,9%)** chứa giá trị hiếm (mã `T2_CHUA_KIEM`): 130.229 giá trị khác nhau, 90% chỉ xuất hiện một lần, giá trị tiếng Trung dài trung vị 11 ký tự, 46% là tên màu/mẫu (颜色). Không đọc tay hết được. Claude đọc thử 300 cặp ngẫu nhiên trong nhóm này: **26,3% dịch sai**. Câu hỏi: có công cụ tự động nào tách cặp đúng khỏi cặp sai để cứu lại một phần không?

### Cách đo

Không chọn ngưỡng theo cảm giác. Dùng **4.291 cặp đã có nhãn đúng/sai**: 3.991 cặp tầng 1 (2.840 đúng, 1.151 sai) và 300 cặp hiếm (221 đúng, 79 sai). Chấm từng công cụ, rồi đo **AUC** = xác suất một cặp sai có điểm thấp hơn một cặp đúng (0,5 là đoán mò, 1 là tách hoàn hảo).

| Công cụ | AUC tầng 1 | AUC tầng 2 (cặp hiếm) | Kết luận |
|---|---|---|---|
| Qwen2.5-7B-Instruct làm giám khảo (4-bit, Colab) | — | precision 28–32% ở mọi ngưỡng | ngang đoán mò, bỏ |
| bge-m3 cosine, chỉ giá trị | 0,580 | 0,574 | gần đoán mò |
| bge-m3 cosine, "tên: giá trị" | 0,626 | 0,570 | gần đoán mò |
| CometKiwi (`Unbabel/wmt22-cometkiwi-da`), chỉ giá trị | 0,692 | 0,705 | khá hơn |
| **CometKiwi, "tên: giá trị"** | **0,733** | **0,724** | tốt nhất, nhưng chưa đủ |

bge-m3 chạy qua Ollama trên CPU (cắt 480 ký tự); CometKiwi chạy trên Colab T4.

### Vì sao bge-m3 hợp với title nhưng không hợp với description

bge-m3 đo hai đoạn **có cùng chủ đề** hay không, không đo **dịch có đúng** hay không. Title là câu dài nên cách này còn dùng được. Giá trị thuộc tính thì quá ngắn:

| Bản Trung → bản Việt | Đúng/sai | Cosine bge-m3 |
|---|---|---|
| 奶瓶 → "奶瓶" (chưa dịch) | sai | **1,00** |
| 33529米色 → "33529m màu" | sai | 0,89 |
| 寿眉 → "Thọ Mi" (tên trà) | đúng | **0,28** |
| 浮世绘 → "Ukiyo-e" | đúng | 0,29 |

Cosine trung bình: cặp đúng 0,71, cặp sai 0,67. Ở mọi ngưỡng từ 0,30 đến 0,90, trong số cặp bị gắn cờ chỉ 27–38% là sai thật, gần bằng tỉ lệ sai sẵn có 26%.

### Vì sao thử CometKiwi

CometKiwi là mô hình **ước lượng chất lượng bản dịch không cần bản tham chiếu** (quality estimation): được huấn luyện để dự đoán điểm người chấm cho một bản dịch, và là hệ thống đứng đầu phần QE của WMT 2022. Nó được làm ra đúng cho câu hỏi "bản dịch này đúng hay sai", là câu bge-m3 không trả lời.

Kết quả:
- AUC tăng từ khoảng 0,6 lên **0,72–0,73**.
- **Bắt tốt lỗi nặng:** trong 2% cặp tầng 1 có điểm thấp nhất (79 cặp), **87%** là sai thật; đáy 5%: 82%; đáy 10%: 71%.
- **Bỏ lọt lỗi số và đơn vị:** 3XL【建议190-215斤】 → "3XL [khuyến nghị 190-215 kg]" vẫn được 0,86 (đúng phải là 95–107,5 kg, vì 1 斤 = 0,5 kg); 苹果6S → "Apple 6S" được 0,86.
- **Chấm oan thuật ngữ ngắn đúng:** 灯芯绒 → "Nhung kẻ", 魔术贴 → "Băng dính gai" chỉ được 0,25–0,28.

### Vì sao cuối cùng không dùng

Thử giữ lại phần cặp hiếm có điểm CometKiwi ("tên: giá trị") cao nhất, trên 300 cặp có nhãn:

| Giữ lại | Số cặp | Phần giữ vẫn sai |
|---|---|---|
| 100% (không lọc) | 300 | 26,3% |
| 50% điểm cao nhất | 150 | 14,0% |
| 30% điểm cao nhất | 90 | 10,0% |
| 20% điểm cao nhất | 60 | 8,3% (5/60, mẫu nhỏ nên thực tế có thể khoảng 3–18%) |

Phần dữ liệu đang giữ có tỉ lệ lỗi ước tính **dưới khoảng 1,5%** (200 mẫu cuối, 0 sai). Cứu thêm các cặp còn sai 8–14% sẽ kéo chất lượng xuống, nên **giữ nguyên cách loại các cặp hiếm**. Các cặp này vẫn nằm trong file cặp kèm mã `T2_CHUA_KIEM`, để kiểm lại khi có cách tốt hơn.

**Tóm lại:** bge-m3 hợp với title nhưng không hợp với description; CometKiwi tốt hơn rõ nhưng chưa đủ an toàn để đưa thêm dữ liệu vào train. Số liệu và mã nguồn của cả hai lần đo nằm ở repo GitHub `ntquang-0410/eda_ecom`, nhánh `nhatanh_updating`: `data/eda/description/bge_do_truoc/`, `data/eda/description/kiwi_do_truoc/`, `scripts/eda/desc_bge_do_truoc.py`, `scripts/eda/desc_kiwi_phan_tich.py`, `notebooks/desc-cometkiwi-do-truoc-colab.ipynb`. CometKiwi có giấy phép CC BY-NC-SA 4.0 (nghiên cứu, không thương mại).

## Lưu ý khi dùng

- **Chưa chia train/dev/test.** Khi chia, nhóm theo `nhom_trung` và bỏ `trung_y_het = 1`.
- **Chữ tiếng Anh/pinyin trong bản Việt chưa xử lý.** Phần này sẽ làm ở bản sau.
- **Bản Việt vẫn là bản dịch máy đã sửa.** Phần giữ lại có tỉ lệ lỗi ước tính dưới khoảng 1,5% (95%, dựa trên 200 mẫu cuối, 0 sai; quy tắc 3/n).
- Bản Việt có thể ngắn hơn bản gốc vì các cặp bị loại không được ghép lại. Cặp bị loại vẫn có trong file cặp, kèm mã lý do.
