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
    - tầng 1: 4.006 cặp được đọc thủ công, 1.151 cặp sai được sửa;
    - tầng 2: khoảng 130 nghìn cặp quá hiếm, gắn `T2_CHUA_KIEM`.
11. Loại các cặp còn lỗi bằng mã lý do.
12. Đánh dấu trùng lặp và nhóm sản phẩm giống nhau.

## Lưu ý khi dùng

- **Chưa chia train/dev/test.** Khi chia, nhóm theo `nhom_trung` và bỏ `trung_y_het = 1`.
- **Chữ tiếng Anh/pinyin trong bản Việt chưa xử lý.** Phần này sẽ làm ở bản sau.
- **Bản Việt vẫn là bản dịch máy đã sửa.** Phần giữ lại có tỉ lệ lỗi ước tính dưới khoảng 2% (95%, dựa trên 200 mẫu cuối, 0 sai).
- Bản Việt có thể ngắn hơn bản gốc vì các cặp bị loại không được ghép lại. Cặp bị loại vẫn có trong file cặp, kèm mã lý do.
