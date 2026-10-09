# e-commerce - EDA

Bước EDA và Xử lý dữ liệu của tiểu luận "Tối ưu hoá mô hình dịch máy Việt–Trung
cho TMĐT xuyên biên giới". Dữ liệu crawl từ 1688.com đã xong (xem
`data/snapshot/SNAPSHOT.md`); folder này chỉ phân tích và làm sạch, không crawl.

**Đọc trước:** `CLAUDE.md` (quy tắc làm việc) và `WALKTHROUGH.md` (nhật ký,
các vấn đề của data kèm bằng chứng, giải pháp, việc còn nợ). Mục 0 của
WALKTHROUGH nói rõ đang ở đâu.

## Dựng môi trường (Windows, PowerShell)

```
python -m venv .venv                       # Python 3.13.x
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Không cần `.env` hay token. Snapshot có sẵn trong `data/snapshot/`.

## Chạy lại các bước đã có (từ gốc folder)

```
.venv\Scripts\python.exe scripts\eda\attr_split_check.py     # -> data\interim\attr_pairs.parquet
.venv\Scripts\python.exe scripts\eda\title_audit.py          # -> data\eda\title_audit\
.venv\Scripts\python.exe scripts\eda\build_tokens_edited.py  # -> title_latin_tokens_editted.csv
.venv\Scripts\python.exe scripts\eda\data_issues.py          # -> data\eda\data_issues\
```

Bốn bước này chạy lại ra file **giống hệt từng byte** bản gốc (kiểm 01/10/2026).
`scripts\eda\sim_full.py` (similarity toàn bộ, khoảng 20 phút) và
`scripts\notebook\run_nb_harness.py` cần Ollama chạy `bge-m3` ở
`127.0.0.1:11434`; chưa chạy lại sau khi chuyển folder (xem WALKTHROUGH).

## Cấu trúc

```
notebooks/   eda-bilingual_zh_vi.ipynb (+ archive/)
scripts/     eda/ · notebook/ · archive/
src/         textnorm.py (chuẩn hoá đã áp lên dữ liệu crawl)
docs/        PROJECT_CHARTER_team_CLAUDE.md, PROVENANCE.md, copy_manifest.csv, design/, ecom_crawler_README.md
data/        snapshot/ (bất biến) · eda/ (kết quả) · interim/ (cache, sinh lại được) · processed/ (đầu ra bước xử lý, đang rỗng)
```


---

## HƯỚNG DẪN CHẠY CÁC FILE CODE XỬ LÝ DESCRIPTION

*Viết cho người hoặc AI chạy lại pipeline làm sạch description trên máy khác (ví dụ máy Quang). Làm đúng thứ tự ở mục (2). **Không gồm xử lý tiếng Anh**: phần đó Quang làm riêng.*

**Pipeline làm gì.** Lấy cột description của snapshot `data/snapshot/bilingual_zh_vi.parquet` (chuỗi `tên thuộc tính: giá trị; ...`; bản Việt là bản máy dịch của Alibaba), tách thành từng cặp thuộc tính, sửa hoặc loại từng cặp bằng luật và từ điển đã duyệt, rồi dựng lại bảng cặp sạch và description sạch theo sản phẩm. Cặp không sửa được thì gắn **mã lý do** và `dung_cho_train = 0`, **không xoá**. Pipeline không dịch lại từ đầu.

**Ngoài phạm vi (đang tắt, đừng bật):**
- Xử lý tiếng Anh và pinyin (bước P7.6, cờ `XU_LY_TIENG_ANH = False` ở ô P0).
- Qwen kiểm nghĩa tầng 2 (cờ `DUNG_QWEN = False`; cặp hiếm gắn `T2_CHUA_KIEM`).
- Thí nghiệm bge-m3 và CometKiwi (notebook riêng, không nằm trong pipeline).
- Chia train, dev, test (hoãn; pipeline chỉ gắn `nhom_trung`).

### (1) Danh sách kỹ thuật xử lý

Mỗi bước ghi ra `data/processed/desc/pN_cap.parquet` (cặp) và `pN_gia_tri.parquet` (từng giá trị); mọi thay đổi ghi vào `log_thay_doi.parquet`.

1. **P1 Tách và ghép cặp (DS01, DS02, DS25).** Ghép zh–vi theo thứ tự đoạn trong `description_vi` (cột `attributes_vi` lệch thứ tự ở 99,8% dòng nên không dùng được). Cặp trùng y hệt trong cùng sản phẩm gắn `DS25`, giữ một bản. Sản phẩm không ghép được gắn `DS02` ở cấp sản phẩm.
2. **P2 Làm sạch ký tự (DS24), áp trên bản Việt.** Bỏ ký tự vùng riêng (PUA); 【】 thành [ ]; dấu câu Trung thành dấu ASCII (chỉ khi bản Việt đã dịch và khác bản Trung); `"; "` thành `", "` để không lẫn dấu phân cách khi dựng lại description; gộp khoảng trắng thừa. Năm giá trị có ký tự hỏng hoặc chữ Hàn đi qua file duyệt `P2_sua_tay_editted.csv`.
3. **P3 Lọc thông tin cá nhân (DS12).** Bỏ hẳn cặp liên hệ (tên thuộc tính chứa 联系方式, 电话, 微信, 联系人), mã `PII`. Cặp tên hoặc địa chỉ nhà sản xuất (生产厂家, 厂址, 公司, 制造商...) mã `PII_NSX`. Giá trị còn lại: che số di động, máy bàn, URL, email, WeChat/QQ bằng nhãn `[SĐT]`, `[URL]`, `[EMAIL]`, `[LIÊN HỆ]` ở cả hai phía. Bản công bố bỏ cột `shop`. Bước này phải chạy trước mọi bước gửi dữ liệu ra LLM.
4. **P4 Vai trò thuộc tính và giá trị rỗng nghĩa (DS21).** Vai trò (`mo_ta_hang`, `giao_dich_1688`, `lua_chon_sku`, `ma_giay_to`, `thuong_hieu`) lấy từ bảng `desc_vai_tro_thuoc_tinh_editted.csv`, ưu tiên cột `anh_duyet`, rồi `goi_y_claude`, rồi luật; tên ngoài bảng dùng luật (luật mã xét trước luật giao dịch). Giá trị rỗng nghĩa dịch về một cách chuẩn (其他 thành "Khác", 无 thành "Không có", 见包装 thành "Xem bao bì") và gắn nhãn `it_thong_tin`. Cặp mà mọi giá trị chỉ là ký hiệu gắn `RONG`.
5. **P5 Gỡ lỗi đánh số (DS06, DS06B).** Bỏ 1–2 chữ số cuối do máy dịch thêm ("Không 2" thành "Không") khi thoả cả ba điều kiện: bỏ đi thì tập số khớp bản Trung; số đó không phải chữ số Hán có trong bản Trung (二级 dịch "Cấp độ 2" là đúng); phần còn lại trùng một bản dịch khác trong cùng sản phẩm.
6. **P6 Từ điển tên thuộc tính (DS15, DS16, DS05, DS07 tên).** 300 tên phổ biến nhất cùng các tên sai nghĩa đã biết, file `P6_ten_thuoc_tinh_editted.csv`. Thứ tự ưu tiên khi áp: `anh_duyet`, rồi `goi_y_claude`, rồi `ten_chuan_tam` (cách dịch đa số hoặc nghĩa đúng). Tên ngoài bảng chỉ viết hoa chữ đầu.
7. **P7 Từ điển giá trị có điều kiện (DS17, DS18, DS19, DS13, DS09A/B, DS08, DS07 giá trị).**
   - Từ điển giá trị chữ Hán gặp từ 50 lần trở lên (`P7_gia_tri_editted.csv`), áp theo `gia_tri_zh`, trừ 品牌 và cặp mã giấy tờ.
   - Luật theo ngữ cảnh (苹果 ở thuộc tính dòng máy thành "Apple"; 中 ở kích cỡ thành "Vừa", ở độ cứng thành "Trung bình").
   - Lỗi sai nghĩa nằm trong chuỗi dài, không sửa an toàn được thì gắn `DS17`.
   - Địa danh: 87 địa danh Hán Việt, dựng lại "Thành phố, Tỉnh"; có địa danh nhưng bản Việt sai gắn `DS18`.
   - Tên hãng (品牌): từ chung dịch theo bảng 101 từ (`P7_tu_chung_ten_hang_v2_editted.csv`); tên toàn chữ Latin chép nguyên; có phần Latin thì giữ phần Latin; còn lại chuyển pinyin bằng `pypinyin`, giữ chữ số (70迈 thành 70mai); tên công ty hoặc phần chính dài hơn 7 chữ Hán gắn `DS09_dai`.
   - Thuật ngữ dùng chung với title (`P7_thuat_ngu_editted.csv`, 11 thuật ngữ).
   - Mục 7.6 (thay chữ Anh và pinyin bằng bảng title) đang **TẮT**.
8. **P8 Số, mã và đơn vị (DS10, DS03, DS20).**
   - Mã thật (vai trò mã/giấy tờ mà giá trị Trung không có chữ Hán, hoặc số giấy phép/đăng ký có từ 4 chữ số liền) chép nguyên bản Trung; mã còn chữ Hán thêm chú thích tỉnh hoặc "quốc gia", ví dụ `粤妆20170110 (Quảng Đông)`.
   - Ngày `2023年05月24日` thành `24/05/2023`.
   - Ghép lại số thập phân bị tách ("53, 5" thành "53.5"); sửa năm bị đổi (lấy năm bên Trung).
   - Danh sách lệch thứ tự: sắp lại theo số của bản Trung khi là hoán vị, không sắp được gắn `DS03`.
   - **Quy đổi 斤 thành kg (mới 06/10, `src/doi_chieu.py`, hàm `kiem_jin`):** 1 斤 = 0,5 kg; bản Việt viết kg mà giữ nguyên số của bản Trung ("190-215斤" thành "190-215 kg") thì chia đôi các số đó. Giá theo 斤 ("30元/斤") viết theo kg, hoặc số không bằng số gốc cũng không bằng một nửa, gắn `DS10`. Không đụng 公斤 (là kg) và 千斤顶 (cái kích xe).
   - **Cảnh báo đơn vị (mới 06/10, hàm `canh_bao_don_vi`):** chỉ gắn nhãn `CB_cun` (寸), `CB_luong` (两), `CB_van` (万), `CB_ty` (丝), không sửa, không loại; danh sách ghi ở `data/processed/desc/p8_canh_bao_don_vi.csv`.
   - Số còn lệch sau khi quy đổi gắn `DS10`.
9. **P9 Định dạng (DS22, DS23, DS14).** Bỏ dấu chấm cuối do máy thêm (trừ khi bản Trung cũng có hoặc giá trị kết thúc bằng "..."); chữ Việt IN HOA toàn bộ về viết hoa chữ đầu; "a,b" thành "a.b" khi bản Trung có đúng "a.b" với 1–2 chữ số sau dấu (không đụng "1.000" và danh sách "9,12,24").
10. **P10 Kiểm nghĩa phần đuôi dài (DS04).** Tầng 1: khoảng 4.000 cặp (zh, vi) phổ biến nhất và cặp có tỉ lệ độ dài bất thường; Claude Code đọc từng cặp và gắn `DUNG`, `SAI` hoặc `KHONG_CHAC` (file `P10_tang1_v7_editted.csv`); `SAI` có bản sửa thì thay, không có thì gắn `P10_SAI`. Tầng 2 (khoảng 130 nghìn giá trị hiếm): gắn `T2_CHUA_KIEM`, không vào train nhưng giữ trong file, vì Qwen2.5-7B chỉ đạt precision 28–32% khi thử và Claude chấm thấy khoảng 26% sai nghĩa.
11. **P11 Loại có mã lý do.** Bản Việt còn chữ Hán (trừ cặp mã giấy tờ) gắn `DS07`; sản phẩm `DS02` loại cả sản phẩm. `dung_cho_train = 1` khi cặp không có mã lý do. Mã lý do khác nhãn: nhãn (`it_thong_tin`, `chua_kiem_nghia`, `chep_ma`, `vai_tro_tam`) chỉ để biết, không loại cặp.
12. **P12 Khử trùng lặp và gắn nhóm (DS11).** `trung_y_het = 1` cho bản thứ hai trở đi của cặp description sạch trùng y hệt (cả zh lẫn vi). `nhom_trung`: gộp sản phẩm cùng `title_zh`, cùng shop, cùng description, hoặc gần trùng theo MinHash (đoạn 5 ký tự, 128 hàm băm, LSH, Jaccard từ 0,8), nối bắc cầu bằng union-find.
13. **P13 Dựng output và kiểm định.** Dựng `desc_pairs_clean.parquet` (mọi cặp, kèm mã lý do, nhãn, vai trò, `dung_cho_train`) và `desc_clean.parquet` (mỗi sản phẩm một dòng, description zh/vi dựng lại từ cặp giữ lại); đo lại các lỗi chính trước và sau; in bảng trạng thái các cổng duyệt; bốc 200 sản phẩm ngẫu nhiên (`P13_mau_cuoi_mau_v3.csv`) chỉ khi mọi cổng khác đã đạt.

**Cổng duyệt.** Các file trong `data/processed/desc/duyet/` là kết quả người đã duyệt (`*_editted.csv`, cột `anh_duyet`, `ket_qua`) và mẫu kiểm bốc bằng seed cố định (`*_mau_v*.csv`). Notebook **đọc lại, không ghi đè** phần đã điền; mục mới phát sinh được nối vào file `_bo_sung.csv` và áp tạm bằng giá trị gợi ý, chờ duyệt. Dữ liệu chỉ là **chính thức** khi bảng trạng thái cuối notebook báo mọi cổng đạt.

### (2) Thứ tự chạy các file code

Chạy từ **gốc repo**, PowerShell, Windows. Lỗi ở bước nào thì dừng ở bước đó, không nhảy sang bước sau.

| Bước | Việc | Lệnh hoặc file | Kết quả đúng |
|---|---|---|---|
| 0 | Dựng môi trường | `python -m venv .venv` (Python 3.13.x) rồi `.venv\Scripts\python.exe -m pip install -r requirements.txt` | Cài được pandas, numpy, pyarrow, pypinyin, datasketch, ipykernel (kèm jupyter_client). Không cần `.env` hay token |
| 1 | Tải snapshot (công khai, không cần token) | `curl.exe -L -o data\snapshot\bilingual_zh_vi.parquet https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/resolve/main/data/snapshot/bilingual_zh_vi.parquet` | `Get-FileHash data\snapshot\bilingual_zh_vi.parquet` ra `B4E5CE75D5AE8AF523235B5C23F5C7289E405A8230D879F11136AFC938135A29`. Ô P0 kiểm đúng mã này và **dừng nếu lệch**: không sửa mã trong notebook cho qua, báo người phụ trách |
| 2 | Chạy bộ thử đơn vị của luật 斤 và cảnh báo đơn vị | `.venv\Scripts\python.exe scripts\eda\test_doi_chieu.py` | Hai dòng cuối: `30/30 ca quy đổi 斤 đạt` và `26/26 ca cảnh báo đơn vị đạt`. Không đạt thì đừng chạy bước 3 |
| 3 | Chạy pipeline P0 đến P13 (14 ô code, theo đúng thứ tự trong notebook) | `.venv\Scripts\python.exe scripts\notebook\run_desc_pipeline.py` | In `ô N ok` cho từng ô rồi `ALL OK`. Ghi kết quả vào bản sao `notebooks\desc-pipeline.da_chay.ipynb`, notebook gốc không bị sửa. Hoặc mở `notebooks/desc-pipeline.ipynb` bằng kernel `.venv` và Run All |
| 4 | Đọc kết quả | Ô cuối của `desc-pipeline.da_chay.ipynb` | Có "Trạng thái cổng duyệt" và "KẾT LUẬN" (số cặp giữ, số sản phẩm, "Dữ liệu là CHÍNH THỨC" hay chưa) |

**Các file code theo thứ tự notebook gọi** (không cần chạy riêng, chỉ để biết dữ liệu vào từ đâu):
- ô P0: tìm gốc repo bằng sự có mặt của `data/snapshot/bilingual_zh_vi.parquet`, kiểm SHA-256, đặt cờ và hàm dùng chung.
- P2: đọc `duyet/P2_sua_tay_editted.csv`. P4: đọc `data/eda/description/desc_vai_tro_thuoc_tinh_editted.csv`.
- P6: `duyet/P6_ten_thuoc_tinh_editted.csv`. P7: `duyet/P7_gia_tri_editted.csv`, `P7_dia_danh_v2_editted.csv`, `P7_tu_chung_ten_hang_v2_editted.csv`, `P7_thuat_ngu_editted.csv`, cùng `data/eda/title_audit/title_term_consistency.csv` và `title_latin_tokens_editted.csv` (đã có sẵn trong repo; bảng tiếng Anh chỉ dùng khi bật cờ, đang tắt).
- P8: `from doi_chieu import canh_bao_don_vi, kiem_jin` (file `src/doi_chieu.py`).
- P10: `duyet/P10_tang1_v7_editted.csv` (kèm `_bo_sung.csv`). P13: `duyet/P13_mau_cuoi_mau_v3.csv`.

**Kết quả tham chiếu để so** (lần chạy kiểm 09/10/2026 trên một bản sao sạch của repo, máy 15,6 GB RAM, 286 giây, cả 14 ô code báo `ok`):
- P8, quy đổi 斤: `ok` 1.677, `da_sua` 1.612 (đã chia đôi số), `lech_khac` 13, `gia_theo_can` 34, `khong_ghep` 2; gắn thêm `DS10` cho 10 cặp; tổng `DS10` 560 cặp. Cảnh báo đơn vị: `CB_ty` 130, `CB_van` 20 (`CB_cun` và `CB_luong` không có trường hợp nào).
- P10: tầng 1 gồm 3.987 cặp khác nhau, cả hai cổng mẫu (30 mẫu `SAI`, 30 mẫu `DUNG`) đạt, 0 sai; tầng 2 gắn `T2_CHUA_KIEM` cho 36.187 cặp.
- P13, dòng KẾT LUẬN cuối: `Giữ 366.003/406.871 cặp (90,0%) cho train; 16.342 sản phẩm có description sạch.` và `Dữ liệu là CHÍNH THỨC` (mọi cổng đạt; chưa gồm xử lý tiếng Anh). Cổng `P13_mau_cuoi`: 200 mẫu, 0 sai.
- Bản đã công bố trước đó (desc_v1, chưa có luật 斤 mới) giữ 366.016 cặp. Chênh vài chục cặp so với số trên là bình thường; chênh hàng trăm cặp trở lên là có vấn đề, hãy dừng và báo lại.

**Có thể gặp khi chạy (không phải lỗi):**
- Ô P10 in `170 mục mới chưa có trong file duyệt → nối vào ..._bo_sung.csv (áp tạm, chờ duyệt)`. Lý do: luật 斤 mới ở P8 đổi bản Việt của một số cặp nên khoá (giá trị Trung, giá trị Việt) của nhãn tầng 1 cũ không còn khớp; các mục này được áp bằng giá trị gợi ý tạm và cổng vẫn đạt. Notebook ghi thêm vào file `P10_tang1_v7_editted_bo_sung.csv` trong repo: đừng commit thay đổi của thư mục `duyet/` nếu chưa có người duyệt (`git checkout -- data/processed/desc/duyet` để bỏ).
- Ô cuối từng một lần báo `MemoryError` khi máy gần hết RAM. Đóng bớt ứng dụng rồi chạy lại từ đầu (bước 3); lần chạy sau đã qua.
- Chạy lại nhiều lần là an toàn: notebook ghi đè các file `pN_*.parquet`, `desc_*_clean.parquet`, `log_thay_doi.parquet` (sinh lại được) nhưng không ghi đè file người đã duyệt.

**Không làm:**
- Không xoá hay ghi đè các file trong `data/processed/desc/duyet/` (đó là phần người đã duyệt).
- Không commit `*.parquet` hay `notebooks/*.da_chay.ipynb` (chứa dòng dữ liệu mẫu; `.gitignore` đã loại).
- Không bật `XU_LY_TIENG_ANH` hay `DUNG_QWEN`, không đổi mã SHA-256 trong ô P0 để "cho qua".
- Không sửa snapshot trong `data/snapshot/`.
