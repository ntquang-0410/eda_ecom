# 02. EDA và làm sạch mô tả (bản `desc_v1`)

[← 01](01_thu_thap_crawl.md) · [Tổng quan](00_TONG_QUAN.md) · [03 →](03_dich_theo_don_vi_bo_1_2.md)

Thời gian: 28/09 – 03/10/2026. Nguồn:
- WALKTHROUGH mục 4–6;
- README bản phát hành `data/release/desc_v1/README.md`, cũng có trên HF `data/cleaned/` với đuôi `(nhatanh)`;
- `CLAUDE.md` mục 5.3.

## 1. EDA title (28/09)

- Đo độ giống `title_zh`–`title_vi` bằng `BAAI/bge-m3` trên toàn bộ 16.348 sản phẩm 1688, có **negative baseline** (ghép cặp ngẫu nhiên và ghép cặp xáo trộn trong cùng category) để tự suy ra ngưỡng thay vì dùng 0,5 cố định. AUC 0,9769 (so với cặp xáo cùng category), 0,9964 (so với cặp ngẫu nhiên) [ĐO].
- Audit tiếng Anh/pinyin lẫn trong `title_vi`, lập bảng quyết định từng từ Latin (`title_latin_tokens_editted.csv`).
- Từ 01/10 **phía title giao cho Huy**. Nhật Anh làm phía mô tả.

## 2. Các vấn đề của mô tả 1688

- Nhật Anh và Claude rà ra **25 nhóm lỗi** (mã `DS..`), xếp từ nặng tới nhẹ. Mỗi lỗi kèm bằng chứng theo `product_id`, ở `data/eda/data_issues/`.
- Lỗi nặng tiêu biểu:
  - **Đánh số máy thêm vào giá trị** ("Không 2", "Có 3"), 9.286 sản phẩm (56,8%) [ĐO, D07].
  - Bản Việt còn chữ Hán.
  - Số trong bản Việt lệch bản Trung.
  - Danh sách giá trị lệch thứ tự.
  - Đơn vị 斤 dịch thành "kg" (đúng là 0,5 kg).
  - Tên hãng, địa danh dịch sai.
  - Thông tin liên hệ.

## 3. Pipeline làm sạch mô tả (13 bước, `notebooks/desc-pipeline.ipynb`)

Đầu vào: snapshot `bilingual_zh_vi.parquet`, 16.348 sản phẩm, 406.871 cặp thuộc tính. Pipeline **không dịch lại** mà sửa hoặc loại từng cặp bằng luật, từ điển đã duyệt và đọc thủ công. Mỗi bước có cổng duyệt bằng mẫu ngẫu nhiên.

1. Tách mô tả thành cặp, ghép zh–vi theo thứ tự đoạn.
2. Làm sạch ký tự: 【】 → [ ], dấu câu Trung → dấu thường.
3. Thông tin cá nhân:
   - bỏ cặp liên hệ và tên/địa chỉ nhà sản xuất;
   - che SĐT, email, URL, WeChat/QQ bằng nhãn;
   - bỏ cột `shop`.
4. Gắn vai trò thuộc tính (mô tả hàng, giao dịch 1688, lựa chọn SKU, mã giấy tờ, thương hiệu); dịch chuẩn giá trị rỗng nghĩa (其他 → "Khác", 无 → "Không có").
5. Bỏ số đuôi do máy thêm.
6. Từ điển 300 tên thuộc tính phổ biến.
7. Từ điển giá trị, luật ngữ cảnh, địa danh, tên hãng (chữ Latin giữ nguyên, còn lại chuyển pinyin).
8. Số và đơn vị: chép nguyên mã giấy phép, ghép số thập phân bị tách, sửa năm, sắp lại danh sách.
9. Định dạng: bỏ dấu chấm cuối, bỏ IN HOA toàn bộ, "7,5" → "7.5" khi bản Trung ghi 7.5.
10. Giá trị hiếm:
    - tầng 1: 4.006 cặp được Claude đọc từng cặp, sửa 1.151 cặp sai; Nhật Anh kiểm 60 mẫu;
    - tầng 2: khoảng 130 nghìn giá trị quá hiếm, gắn `T2_CHUA_KIEM` và **loại**.
11. Loại các cặp còn lỗi bằng mã lý do.
12. Đánh dấu trùng (`trung_y_het`) và nhóm sản phẩm giống nhau `nhom_trung`:
    - nối hai sản phẩm khi cùng title, cùng shop, cùng mô tả, hoặc MinHash Jaccard ≥ 0,8;
    - nối bắc cầu;
    - kết quả 6.566 nhóm, nhóm lớn nhất 629.

Mẫu cuối 200 sản phẩm: 0 lỗi, nên tỉ lệ lỗi phần giữ ước dưới khoảng 1,5% (quy tắc 3/n) [ĐO trên mẫu].

## 4. Kết quả `desc_v1` (03/10)

| File | Mỗi dòng | Số dòng |
|---|---|---|
| `bilingual_zh_vi_desc_cleaned(nhatanh).parquet` | 1 sản phẩm | **16.342** (bỏ 6 sản phẩm không ghép được cặp) |
| `bilingual_zh_vi_desc_pairs(nhatanh).parquet` | 1 cặp | **405.275**, trong đó `dung_cho_train = 1`: **366.016** |

Cặp bị loại theo mã lý do (một cặp có thể có nhiều mã):

| Mã | Ý nghĩa | Số cặp |
|---|---|---|
| `T2_CHUA_KIEM` | giá trị hiếm chưa kiểm nghĩa | 36.184 |
| `DS07` | còn chữ Hán | 2.762 |
| `RONG` | chỉ có ký hiệu | 1.369 |
| `DS10` | số lệch | 554 |
| `DS17` | lỗi nghĩa trong chuỗi dài | 110 |
| `DS09_dai` | tên hãng/công ty quá dài | 74 |
| `DS03` | danh sách lệch thứ tự | 62 |
| `DS18` | địa danh sai | 3 |
| `DS25` | cặp trùng trong sản phẩm | 2 |

## 5. Thử bộ lọc tự động (không dùng)

Có 36.184 cặp giá trị hiếm. Claude đọc 300 cặp ngẫu nhiên trong nhóm này thì thấy 26,3% sai. Thử các công cụ tự tách đúng/sai, đo AUC trên 4.291 cặp có nhãn:

| Công cụ | AUC tầng 1 | AUC cặp hiếm |
|---|---|---|
| Qwen2.5-7B làm giám khảo | – | precision 28–32%, ngang đoán mò |
| bge-m3 cosine | 0,58–0,63 | 0,57 |
| CometKiwi (`wmt22-cometkiwi-da`), "tên: giá trị" | **0,733** | **0,724** |

Giữ 20% cặp có điểm CometKiwi cao nhất thì phần giữ vẫn sai 8,3%, cao hơn nhiều so với khoảng 1,5% của phần đang giữ. Vì vậy **không cứu** cặp hiếm.

**Bài học dùng về sau:**
- bge-m3 hợp với title nhưng không hợp với giá trị thuộc tính ngắn (奶瓶 → "奶瓶" chưa dịch vẫn được cosine 1,00).
- CometKiwi tốt hơn, nhưng bỏ lọt lỗi số/đơn vị và chấm oan thuật ngữ ngắn.
- Ở bộ 1–2, CometKiwi được dùng để **chọn giữa hai bản dịch**, không dùng làm bộ lọc tuyệt đối.

## 6. Quyết định quan trọng ở cuối giai đoạn này

- Bản Việt Alibaba sau làm sạch vẫn còn nhiều lỗi khó sửa bằng luật. Nhật Anh quyết **bỏ hoàn toàn bản Việt của 1688** và **dịch lại bằng LLM** (Prompt 166).
- Tiki không có bản Trung nên phải dịch vi→zh.
- `desc_v1` vẫn được dùng làm **đầu vào cặp** (đã tách, đã làm sạch phía Trung, đã định tuyến luật) cho bộ 1.
