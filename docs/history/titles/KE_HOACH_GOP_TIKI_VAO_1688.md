# Kế hoạch gộp Tiki vào 1688

Ngày 28/10/2026 · Người làm: Huy (phía Title), Nhật Anh (phía Description) · Claude hỗ trợ viết script

Gộp Tiki vào 1688 cho ra **26.040 sản phẩm** (1688 14.996, Tiki 11.044) trong 2 file 9 cột, sau 5 quy tắc loại bỏ và trần 3.500 sản phẩm mỗi nhóm. Huy lo phía title, Nhật Anh lo phía description.

> **NOTE cho Huy (đọc trước):** các quy tắc loại bỏ ở mục 3 do Nhật Anh và Claude soạn dựa trên số liệu description và category. **Huy cần kiểm tra lại các quy tắc này sao cho phù hợp nhất với Title, và tìm xem Title còn lỗi nào chưa được quy tắc nào bắt.** Cách làm và cách báo kết quả nằm ở mục 5 ("Việc Huy phải kiểm tra trên Title"). Quy tắc nào Huy sửa hoặc thêm thì báo lại trước cổng G2, vì số sản phẩm giữ lại sẽ đổi theo.

## 1. Kết quả cuối

Hai file 9 cột (bản description dịch theo cụm và bản dịch theo pair) cùng 26.040 sản phẩm, cộng một file ghi các sản phẩm bị loại để khôi phục được. Số liệu tính thử trên dữ liệu ngày 28/10/2026; đổi nếu đầu vào đổi.

| File (đề xuất) | Nội dung | Dòng |
| --- | --- | ---: |
| `bilingual_official_context.parquet` | Description dịch theo cụm, 9 cột | 26.040 |
| `bilingual_official_pair.parquet` | Description dịch theo pair, 9 cột, đúng cùng danh sách sản phẩm | 26.040 |
| `excluded_products.parquet` | Sản phẩm bị loại: product_id, source, category, title_zh, title_vi, ly_do_loai | 3.508 |
| `selection_report.json` | Số sản phẩm sau từng quy tắc, theo nhóm và nguồn | – |
| `manifest.json`, `SHA256SUMS.txt`, `README.md` | Cột, số dòng, sha256, giới hạn dữ liệu | – |

Số sản phẩm giữ lại theo nhóm và nguồn:

| Nhóm | 1688 | Tiki | Cộng |
| --- | ---: | ---: | ---: |
| home | 1.779 | 1.721 | 3.500 |
| mother_baby | 1.957 | 1.543 | 3.500 |
| beauty | 1.292 | 2.096 | 3.388 |
| electronics | 1.085 | 2.094 | 3.179 |
| food | 1.406 | 1.710 | 3.116 |
| bags | 2.874 | 0 | 2.874 |
| auto | 960 | 1.880 | 2.840 |
| shoes | 2.309 | 0 | 2.309 |
| fashion | 1.334 | 0 | 1.334 |
| **Tổng** | **14.996** | **11.044** | **26.040** |

## 2. Đầu vào và điều kiện bắt đầu

Bốn file full 9 cột của Huy (HF `246ba7e482`, 28/10 08:41 UTC) đã đủ dòng và đúng nội dung; chỉ sai tên cột thứ 8 nên chưa gộp được.

| File | Dòng | Ghi chú |
| --- | ---: | --- |
| `data/1688/full_dataset/1688_full_attributes_context_zh_vi.parquet` | 16.348 | 6 sản phẩm không có description |
| `data/1688/full_dataset/1688_full_attributes_independent_zh_vi.parquet` | 16.348 | 6 sản phẩm không có description |
| `data/tiki/full_dataset/tiki_full_attributes_context_vi_zh.parquet` | 13.200 | 13 sản phẩm không có description |
| `data/tiki/full_dataset/tiki_full_attributes_independent_vi_zh.parquet` | 13.200 | 13 sản phẩm không có description |

Đã kiểm ngày 28/10: `product_id` duy nhất; `category` đủ (1688 có 9 nhóm, Tiki có 6 nhóm mỗi nhóm 2.200); `description_zh`, `description_vi`, `desc_nguon_dich` khớp 100% bảng sản phẩm Nhật Anh đã đẩy (16.342 và 13.187 sản phẩm).

**Cổng G0 (Huy):** cột 8 hiện tên `title_ngon_dich` ở cả 4 file và `manifest.json`; đổi thành `title_nguon_dich` rồi đẩy lại. Script gộp từ chối chạy nếu danh sách cột khác thứ tự chốt:

`product_id`, `source`, `category`, `title_zh`, `title_vi`, `description_zh`, `description_vi`, `title_nguon_dich`, `desc_nguon_dich`.

## 3. Quy tắc loại bỏ

Những quy tắc dưới đây áp theo đúng thứ tự bảng; mỗi sản phẩm bị loại ghi đúng một `ly_do_loai` (quy tắc đầu tiên nó vi phạm). Quy tắc dựa vào description và category do Nhật Anh duyệt (R1 đến R3), quy tắc dựa vào title do Huy duyệt (R4). Danh sách giữ/loại quyết định một lần trên bảng sản phẩm bộ cụm rồi áp giống hệt cho bản pair.

| Bước | Quy tắc | `ly_do_loai` | Loại | Còn lại |
| --- | --- | --- | ---: | ---: |
| R0 | Không có description (1688 6, Tiki 13), từ 29.548 sản phẩm đầu vào (1688 16.348 + Tiki 13.200) | `khong_co_description` | 19 | 29.529 |
| R1 | Sản phẩm nằm sai nhóm hàng | `sai_nhom` | 594 | 28.935 |
| R2 | Tỷ lệ cặp yếu từ 0,25 | `cap_yeu_cao` | 214 | 28.721 |
| R3 | Mô tả trùng hệt quá 3 bản | `mo_ta_trung` | 825 | 27.896 |
| R4 | Title 1688 còn chữ Hán | `title_con_chu_han` | 31 | 27.865 |
| R5 | Vượt trần 3.500 của nhóm | `vuot_tran_nhom` | 1.825 | 26.040 |

- **R1, sai nhóm (594).** Tiki food 377 (thú cưng, bể cá, dược phẩm), Tiki mother_baby 163 (tã, sữa, miếng lót cho người lớn, gậy chống), Tiki home 17 (nhạc cụ), 1688 37 (bát ăn thú cưng, đồ người lớn, USB và thẻ nhớ, túi y tế, băng gạc, bao bì). Xác định theo danh mục lá (`category_tiki`, `category_1688`); danh sách ở Phụ lục.
- **R2, chất lượng yếu.** Tỷ lệ cặp yếu = `so_cap_thap` chia `so_cap` của bảng sản phẩm bộ cụm; ngưỡng 0,25 cho cả hai nguồn. Không dùng nhãn `do_tin_cay` cấp sản phẩm vì hai nguồn tính khác nhau (nếu loại theo nhãn này thì fashion chỉ còn 13 sản phẩm).
- **R3, mô tả trùng.** Cùng `source`, `category` và `description_zh`: giữ tối đa 3 sản phẩm, chọn theo mã băm crc32 của `product_id`. Không chọn theo tỷ lệ cặp yếu vì trong cả 814 nhóm trùng của Tiki và 88 nhóm của 1688, mọi sản phẩm cùng nhóm có tỷ lệ cặp yếu giống hệt nhau. Phần lớn là Tiki (821 trong 825). Nhóm lớn nhất có 77 sản phẩm: các tấm lót sàn Macsim cho những đời xe khác nhau, title khác nhau nhưng cùng một bảng thuộc tính (thương hiệu, xuất xứ, bảo hành 2 năm). Những title bị loại vẫn nằm trong file loại và khôi phục được.
- **R4, title.** Loại title 1688 có ký tự Hán (U+4E00 đến U+9FFF) trong `title_vi`. Title Tiki có chữ Việt trong `title_zh` KHÔNG bị loại: 407 sản phẩm, phần lớn là tên hãng hoặc mẫu giữ nguyên (ví dụ "Hợp Trí Storm …"), chỉ ghi vào README. **Đây là phần Huy cần kiểm tra lại (xem mục 5).**

## 4. Trần 3.500 mỗi nhóm

Mỗi nhóm giữ tối đa 3.500 sản phẩm, tính trên cả hai nguồn cộng lại. Chỉ home và mother_baby vượt trần nên chỉ hai nhóm này bị bớt (944 và 881 sản phẩm), tổng mất khoảng 6,5% dữ liệu. Không tính fashion (1.334 sản phẩm, không tăng được vì Tiki không có hàng thời trang) thì tám nhóm còn lại lệch nhau 1,52 lần (3.500 chia 2.309).

Thứ tự bỏ trong một nhóm vượt trần:

1. Hàng lệch ngành (bậc B) trước.
2. Sản phẩm có tỷ lệ cặp yếu cao hơn.
3. Mã băm crc32 của `product_id`, để chạy lại ra đúng kết quả cũ.

Bậc B chỉ là ưu tiên bỏ, không phải quy tắc loại: nhóm nào không vượt trần thì giữ hết. Hàng bậc B trước và sau trần:

| Nhóm | Hàng lệch ngành | Trước trần | Sau trần |
| --- | --- | ---: | ---: |
| mother_baby | xe mô hình, mô hình nhân vật, boardgame, cờ | 323 | 0 |
| home | pin, ổ cắm, hạt giống, phân bón, máy khoan | 204 | 0 |
| auto | xe máy, xe đạp, xe hoàn chỉnh, mũ bảo hiểm | 904 | 904 |
| beauty | bao cao su, khẩu trang y tế, thiết bị y tế, thực phẩm chức năng | 496 | 496 |
| food | rượu, bia | 77 | 77 |

Hàng lệch ngành còn lại (1.477 sản phẩm) vẫn hợp lệ, khác loại hàng với 1688; ghi rõ trong README. Đã thử và bỏ cách san đều theo danh mục lá: cách này giảm tập trung vào vài lá lớn nhưng cắt mạnh 1688 và đẩy tỷ lệ Tiki lên 62 đến 78%.

Nếu đổi trần: 3.000 còn 24.357 sản phẩm, 2.700 còn 22.543 sản phẩm. Tỷ lệ Tiki sau trần 3.500: mother_baby 44%, home 49%, food 55%, beauty 62%, electronics 66%, auto 66%.

## 5. Phân công và việc Huy phải kiểm tra trên Title

Chia theo phía dữ liệu mà mỗi người sở hữu, không chia theo nửa tập: trần áp theo nhóm trên cả hai nguồn và danh sách giữ/loại phải là một cho cả hai bản. Một người duy nhất đẩy bản cuối để không hai commit đè nhau.

| Người | Việc | Đầu ra |
| --- | --- | --- |
| **Huy** (phía title) | Đổi tên cột `title_ngon_dich` thành `title_nguon_dich` ở 4 file full và manifest (cổng G0) | 4 file full đúng tên cột trên HF |
| **Huy** | **Kiểm tra lại các quy tắc loại bỏ của plan này sao cho phù hợp nhất với Title, và tìm các lỗi Title còn sót (chi tiết ngay dưới bảng)** | Báo cáo kiểm tra title gửi Nhật Anh trước cổng G2 |
| **Huy** | Duyệt R4: 31 title 1688 còn chữ Hán bị loại, title Tiki còn chữ Việt được giữ (hoặc đề xuất sửa quy tắc theo kết quả kiểm tra) | Xác nhận đồng ý quy tắc title |
| **Huy** | Chạy script gộp và đẩy bản cuối lên HF (người đẩy duy nhất) | Một commit, chỉ thêm file mới |
| **Nhật Anh** (phía description) | Duyệt R1 (danh sách lá), R2 (ngưỡng 0,25), R3 (tối đa 3 bản), ưu tiên bỏ bậc B và trần 3.500 | Xác nhận đồng ý quy tắc description và category |
| **Nhật Anh** | Xem mẫu file bị loại trước khi đẩy (30 dòng mỗi `ly_do_loai`) và duyệt phần giới hạn trong README | Chấp thuận kết quả chạy thử |
| **Claude** | Viết `scripts/release/gop_tiki_vao_1688.py` (đọc đầu vào từ HF theo revision cố định, để Huy chạy được trên máy mình), kiểm tự động, test đơn vị cho từng quy tắc | Script + bộ kiểm nghiệm thu |
| **Claude** | Sinh README, manifest, `selection_report.json`, file bị loại; viết script đẩy `kiem`/`day` theo mẫu các lần trước; đối chiếu sau khi đẩy | Gói thư mục sẵn để đẩy |

### Việc Huy phải kiểm tra trên Title (NOTE)

Mục tiêu: các quy tắc loại bỏ của plan phải khớp với tình trạng thật của Title, và không để lỗi Title nào lọt vào bộ chính thức. Huy kiểm tra trên 4 file full hiện tại (cùng một title ở cả bản cụm và bản pair). Mỗi mục dưới đây Huy tự quyết loại, giữ hay sửa; Claude sẽ đưa kết quả vào script.

1. **Chữ Hán sót trong `title_vi` của 1688 (R4).** Hiện 31 sản phẩm. Xác nhận loại hẳn, hay còn sửa được bằng ứng viên khác của hai mô hình.
2. **Chữ Việt trong `title_zh` của Tiki.** Hiện 407 sản phẩm: 403 lẫn chữ Hán và chữ Việt (phần lớn là tên hãng hoặc mẫu), 4 hoàn toàn không có chữ Hán (chưa dịch). Thêm 17 `title_zh` không có chữ Hán và cũng không có chữ Việt (giống hệt `title_vi`, thường là tên Latin thuần), nên cần xem từng cái. Tách hai loại: tên hãng hoặc mẫu hợp lệ (giữ) và phần chưa dịch hoặc dịch dở (loại hoặc sửa). Nếu có nhóm cần loại thì đề xuất cách nhận ra.
3. **Title dịch giống hệt title nguồn (không dịch).** Đếm ở cả hai nguồn.
4. **Title dịch quá ngắn, quá dài, bị cắt cụt hoặc lặp chuỗi** so với title nguồn (tỷ lệ độ dài bất thường, cùng một cụm từ lặp lại nhiều lần).
5. **Dấu vết lỗi sinh** trong title dịch: chữ như "Translation:", thẻ `<think>`, dấu ngoặc hoặc dấu nháy thừa ở đầu hoặc cuối.
6. **Title trùng nhau giữa nhiều sản phẩm** (`title_vi` hoặc `title_zh` giống hệt): xem có cần quy tắc giới hạn như R3 không.
7. **Title chọn bản có điểm COMETKiwi thấp hơn** (`title_ly_do_chon` hoặc `selection_reason` là `existing_alternative`): xem các bản đó có chấp nhận được không.
8. **Bất kỳ lỗi Title nào Huy nhìn thấy mà quy tắc chưa nói tới.**

Cách báo kết quả (gửi Nhật Anh trước cổng G2): với mỗi mục trên, ghi số sản phẩm, 5 ví dụ (`product_id`, title nguồn, title dịch), và đề xuất (loại / giữ / sửa). Nếu Huy đề xuất thêm quy tắc loại thì ghi tên `ly_do_loai` mới và ngưỡng; quy tắc title áp trước khi áp trần (cùng chỗ với R4) nên tổng số sản phẩm giữ lại sẽ đổi.

## 6. Các bước và cổng duyệt

Mỗi cổng là một điểm dừng: chưa qua cổng thì chưa sang bước sau.

1. **G0, đổi tên cột (Huy).** Đổi `title_ngon_dich` thành `title_nguon_dich` ở 4 file full và manifest, đẩy lại. Claude kiểm lại 4 file: cột, số dòng, khớp bảng description.
2. **G1, đóng băng đầu vào.** Ghi commit HF và sha256 của 4 file full cùng 4 bảng sản phẩm description vào `selection_report.json`. Từ đây đầu vào không đổi; đổi thì chạy lại từ G1.
3. **G2, duyệt quy tắc (cả hai).** Huy gửi báo cáo kiểm tra title (mục 5) và duyệt R4; Nhật Anh duyệt R1 đến R3, bậc B và trần. Mỗi người nhắn "đồng ý" cho Claude ghi vào WALKTHROUGH.
4. **G3, chạy thử.** Script in số sản phẩm sau từng quy tắc và ghi file loại mẫu; hai người so với bảng ở mục 1 (26.040 sản phẩm, hoặc số mới nếu có quy tắc mới) và đọc mẫu 30 dòng mỗi `ly_do_loai`.
5. **G4, chạy chính thức.** Ghi vào thư mục tạm, chạy bộ kiểm nghiệm thu, chạy lại lần hai và so sha256 (phải giống hệt).
6. **G5, đẩy HF (Huy).** Chạy `kiem` rồi `day`: một commit, chỉ thêm file mới, không ghi đè file nào. Claude đối chiếu từng file trên HF với bản trên máy sau khi đẩy.
7. **G6, tài liệu.** Cập nhật cây thư mục ở README gốc (người sở hữu repo), `CLAUDE.md` và WALKTHROUGH.

### Nếu một bên xóa hoặc thêm sản phẩm ở đầu vào

Quy ước: **không ai xóa dòng khỏi file full hoặc bảng description để bỏ một sản phẩm.** Muốn bỏ thì thêm quy tắc (title: vào `quy_tac_title`; description và category: vào file quy tắc) để sản phẩm nằm trong `excluded_products.parquet` kèm `ly_do_loai`, khôi phục được và tổng giữ + loại vẫn khớp đầu vào.

Script có chốt ở cổng G1: file full (gốc của phép gộp, do Huy làm) phải đúng số dòng `so_dong_dau_vao` trong file quy tắc (1688 16.348, Tiki 13.200), và mọi sản phẩm của bảng description phải có trong file full. Lệch thì script dừng, không để sản phẩm biến mất không dấu vết. Hai bản cụm và pair của Huy phải cùng sản phẩm, cùng title, cùng category (script cũng kiểm). Nếu thay đổi là cố ý thì cập nhật `so_dong_dau_vao` rồi chạy lại từ G1.

## 7. Nghiệm thu

Bộ kiểm này chạy tự động ở G4 và sau khi đẩy; bất kỳ mục nào sai thì không đẩy.

- [ ] Hai file đúng 9 cột theo thứ tự chốt, kiểu string.
- [ ] Mỗi file 26.040 dòng (hoặc số cuối sau G2), cùng tập `product_id` và cùng thứ tự ở cả hai bản.
- [ ] Không có ô null ở cột nào (đã loại sản phẩm không có description).
- [ ] Đầu vào = giữ + loại: 29.548 sản phẩm, mỗi sản phẩm xuất hiện đúng một lần (26.040 giữ + 3.508 loại).
- [ ] Mỗi nhóm không quá 3.500; số theo nhóm và nguồn khớp bảng ở mục 1.
- [ ] Không còn `title_vi` 1688 chứa chữ Hán.
- [ ] `source` chỉ là `1688` hoặc `tiki`; `category` thuộc 9 nhóm (Tiki chỉ 6 nhóm); `title_nguon_dich` chỉ là `gemma4` hoặc `qwen3_14b_awq`; `desc_nguon_dich` thuộc 4 giá trị đã định nghĩa.
- [ ] `description_zh`, `description_vi`, `desc_nguon_dich` khớp bảng sản phẩm tương ứng (cụm hoặc pair) 100%.
- [ ] `ly_do_loai` chỉ nhận các giá trị ở bảng quy tắc; tổng từng lý do khớp `selection_report.json`.
- [ ] Chạy lại script cho ra cùng sha256.
- [ ] Sau khi đẩy: mọi file khớp bản trên máy; các file khác trong repo không đổi.

## 8. Nội dung README bắt buộc

README viết bằng tiếng Anh như các bộ trước, số liệu lấy từ `selection_report.json`, không gõ tay. Phải có đủ các ý sau:

- Dữ liệu dịch máy, chưa ai xác minh nghĩa (`semantic_verified` là `False`); không dùng làm bộ test nếu chưa có người rà.
- Hai nguồn ngược chiều: 1688 là zh→vi (bản gốc tiếng Trung), Tiki là vi→zh (bản gốc tiếng Việt); lọc theo `source`. Giải thích `title_nguon_dich` và `desc_nguon_dich` (từ điển và quy tắc luôn đóng góp, cột chỉ nêu mô hình đã dịch ít nhất một cặp).
- Tiki chiếm 44 đến 66% ở sáu nhóm chung (mother_baby 44%, home 49%, food 55%, beauty 62%, electronics 66%, auto 66%); bags, shoes, fashion chỉ có 1688. So sánh nhóm theo tỷ lệ, không theo số tuyệt đối.
- Fashion có một danh mục lá chiếm 73% (áo thun nam), shoes 26,5%.
- Còn 1.477 sản phẩm lệch ngành (xe máy và xe đạp trong auto, sản phẩm y tế trong beauty, rượu bia trong food); cách nhận ra: nối `category_leaf` từ bảng `cado` trên HF.
- 407 title Tiki còn chữ Việt trong `title_zh` được giữ vì phần lớn là tên hãng hoặc mẫu (cập nhật theo kết quả kiểm tra của Huy).
- Một số title 1688 được chọn bản có điểm COMETKiwi thấp hơn khi sửa chữ Hán; chi tiết ở README title của Huy.
- Quy tắc loại bỏ, thứ tự, ngưỡng, số sản phẩm mỗi quy tắc, và cách khôi phục từ `excluded_products.parquet`.
- Bản theo cụm và bản theo pair dùng chung một danh sách sản phẩm: không huấn luyện bằng cả hai bản cho cùng một sản phẩm.
- Cách đọc dữ liệu, sha256 và commit HF của đầu vào.

## 9. Điểm chưa chốt

Nhật Anh và Huy đã cùng đồng ý toàn bộ đề xuất ngày 28/10/2026. Cột giữa là quyết định đã chốt. Việc chưa xong chỉ còn báo cáo kiểm tra Title của Huy (mục 5): nếu kết quả làm đổi quy tắc title thì cập nhật lại mục 3 và các con số ở mục 1 trước cổng G2.

| Câu hỏi | Quyết định (đã chốt, cả hai đồng ý) | Ai quyết |
| --- | --- | --- |
| Thư mục đích trên HF (repo hiện chia theo nguồn `data/1688`, `data/tiki`) | `data/bilingual_official/` | Huy |
| Ai đẩy bản cuối | Huy | Huy và Nhật Anh |
| Mức trần mỗi nhóm | 3.500 (trần 3.000 còn 24.357 sản phẩm; trần 2.700 còn 22.543) | Nhật Anh |
| Giữ 1.477 sản phẩm lệch ngành còn lại hay loại hẳn | Giữ, ghi README (loại hẳn thì còn 24.563 sản phẩm) | Nhật Anh |
| Loại title Tiki còn chữ Việt | Không loại hàng loạt; Huy duyệt từng cái ở 21 title (4 chưa dịch và 17 giống hệt nguồn), loại hoặc sửa nếu là lỗi thật | Huy |
| Thêm cột `category_leaf` vào file | Không (nối từ bảng `cado`) | Hai người |
| Một file có cột phương pháp hay hai file riêng | Hai file riêng (cụm và pair) | Hai người |

**Mới 28/10: cột `title_van_de`.** Lúc 09:19 UTC Huy thêm cột thứ 10 `title_van_de` (nhãn nghi vấn của title, nhiều nhãn cách nhau dấu `;`) vào 4 file full. Đề xuất: bộ chính thức vẫn đúng 9 cột đã chốt (script nhận file 10 cột nhưng không đưa cột này ra); Huy dùng các nhãn này để chọn quy tắc title rồi thêm vào file quy tắc. Chờ Nhật Anh và Huy xác nhận.

## 10. Script gộp và cách chạy

Script đã viết xong ngày 28/10 và qua 15 test đơn vị: `scripts/release/gop_tiki_vao_1688.py`; quy tắc ở `scripts/release/gop_tiki_vao_1688.quy_tac.json`; test ở `scripts/release/test_gop_tiki_vao_1688.py`. Chạy thử trên dữ liệu HF bản `874b521491` ra đúng kết quả của plan (26.040 sản phẩm giữ, 3.508 loại); hai lần chạy chính thức vào thư mục tạm cho sha256 giống hệt nhau. Lúc đó file full của Huy còn tên cột `title_ngon_dich` nên phải chạy với `--chap-nhan-ten-cu`.

Chạy bằng Python của `ecom_crawler/.venv`; máy ít RAM thì đặt `OPENBLAS_NUM_THREADS=1`:

- Chạy thử (không ghi bộ chính thức): `python scripts/release/gop_tiki_vao_1688.py --che-do thu --ra <thư mục tạm>`
- Chính thức: `python scripts/release/gop_tiki_vao_1688.py --che-do chinh_thuc --revision <mã commit HF> --ra <thư mục đích>`
- Thêm quy tắc title (Huy): thêm một mục vào `quy_tac_title` trong file quy tắc với `ly_do`, `nguon` (`1688` hoặc `tiki`), `cot` (`title_zh`, `title_vi` hoặc `title_van_de`), `regex`, `mo_ta`, `mo_ta_en`. Không sửa code.
- Test: `python scripts/release/test_gop_tiki_vao_1688.py`

Script từ chối chạy khi file full sai tên cột (cổng G0), tự chạy bộ nghiệm thu ở mục 7 trước và sau khi ghi. Script đẩy HF (`kiem`/`day`) chưa viết, sẽ viết ở cổng G5.

## Phụ lục: danh mục lá bị loại (bậc A)

Danh sách đúng 56 danh mục lá của R1; danh sách hàng lệch ngành (bậc B, 175 lá) được script ghi vào `selection_report.json` để hai người duyệt lúc chạy thử.

- **Tiki, food (33 lá, 377 sản phẩm):** Phụ kiện bể cá; Thức ăn cho mèo; Thức ăn cho chó; Tẩy giun & diệt bọ chét; Thức ăn cho chim; Dành cho chim; Đồ dùng vệ sinh cho chó; Thức ăn cho cá; Bể cá; Sữa tắm chó mèo; Bồn, cát, đồ dùng vệ sinh cho mèo; Dụng cụ chăm sóc chó; Phụ kiện cho chó khác; Dụng cụ ăn uống dành cho chó; Dụng cụ chăm sóc mèo; Bàn chải lông; Dụng cụ ăn uống dành cho mèo; Phụ kiện cho mèo khác; Dây thừng cho mèo; Xịt dưỡng lông; Các dụng cụ chăm sóc khác; Vòng cổ cho mèo; Đồ chơi phát tiếng động cho mèo; Thức ăn thú cưng khác; Các chất vệ sinh khác; Cát vệ sinh; Vòng cổ cho chó; Khay vệ sinh; Thức ăn cho chuột; Các đồ dùng vệ sinh khác; Các Loại Thú Nuôi Khác; Kìm cắt, bấm móng; Dược phẩm.
- **Tiki, mother_baby (6 lá, 163 sản phẩm):** Sữa bột chức năng cho người lớn; Tã quần cho người lớn; Tã dán cho người lớn; Miếng lót dành cho người lớn; Sữa nước chức năng cho người lớn; Gậy chống.
- **Tiki, home (6 lá, 17 sản phẩm):** Phụ kiện nhạc cụ khác; Phụ kiện đàn khác; Các loại đàn khác; Dây đàn; Dây đeo Guitar; Đàn organ - Keyboard.
- **1688 (11 lá, 37 sản phẩm):** 狗狗食具 và 猫猫食具 (bát thú cưng); 名器倒模 và 飞机杯 (đồ người lớn); U盘, 数码存储卡 và 测速仪 (lưu trữ số, thiết bị đo); 医用包 và 皮肤外用敷料/伤口敷料 (y tế); 数码3C包装 và 烘焙包装 (bao bì).
