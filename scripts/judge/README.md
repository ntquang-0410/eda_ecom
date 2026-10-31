# Judge bản dịch theo cụm (bilingual_official_context)

Mô hình judge **cố định**: `deepseek-v4.1-flash` qua API OpenCode (đổi trong `cau_hinh.py`, không đổi bằng tham số).
Khóa API **không nằm trong file nào**; chỉ nhập ẩn khi chạy.

## Chạy (PowerShell, từ thư mục này)
```powershell
.\chay_judge.ps1 --chi-luat              # miễn phí, không cần khóa: quy tắc cứng trên cả 26.038 dòng
.\chay_judge.ps1 --kiem-tra              # hỏi khóa (gõ ẩn), thử địa chỉ + tên mô hình + 1 lời gọi nhỏ
.\chay_judge.ps1 --thu 300               # thí điểm 300 sản phẩm chia tầng (nguồn x nhóm hàng x mô hình dịch)
.\chay_judge.ps1 --tat-ca --tran-usd 30   # toàn bộ; --tran-usd chặn theo chi phí thật (usage.cost), --tran-token chặn theo token
```
Chạy từ thư mục gốc repo thì gõ `.\scripts\judge\chay_judge.ps1 ...`.

**Gói Go (30/10/2026):** mặc định script gọi đường Go `https://opencode.ai/inference/go/openai/v1` (docs opencode.ai/docs/go: Go có đường riêng; đường không có `/go` tính theo số dư trả trước nên bị 402). Docs nêu: Go $10/tháng, `deepseek-v4.1-flash` có trong Go với hạn mức $80/tháng tính theo giá niêm yết (5 giờ = 20%, tuần = 50%, tháng = 100%); mặc định chỉ vùng Âu/Mỹ, ngoài đó chủ/quản trị phải chọn Global ở Settings > Privacy > Regions; hết hạn mức thì chặn trừ khi bật Extra Usage (dùng số dư). Chưa kiểm chứng bằng khóa thật.

Địa chỉ/xác thực đã đối chiếu với docs v2 của OpenCode (opencode.ai/v2/docs/console/api/inference/, đọc 30/10/2026) và `--kiem-tra` của anh (29/10): khóa hợp lệ, `deepseek-v4.1-flash` có trong danh sách. Mô hình `-free` KHÔNG gọi được từ script (docs: client ngoài OpenCode nhận HTTP 403), nên không dùng. HTTP 402 = tài khoản hết số dư (hoặc chưa có gói Go/ hết hạn mức Go): script dừng ngay. Mỗi dòng kết quả ghi `chi_phi_usd` và `nguon_tinh_phi` (free/balance/go/go-plus/byok) lấy từ phản hồi.
- Chạy tiếp được: kết quả ghi từng dòng vào `data/processed/judge/ket_qua_<ten>.jsonl`; chạy lại lệnh cũ sẽ chỉ chấm những dòng chưa xong (dòng lỗi được thử lại).
- Ctrl+C dừng an toàn. `--song-song N` đổi số luồng (mặc định 4, tăng dần nếu không dính giới hạn tốc độ).
- Khi xong có `tom_tat_<ten>.json` (số dòng, cờ, token vào/ra, thời gian).

## Trình tự nên chạy
1. `--chi-luat` (đã chạy: 122/26.038 dòng bị cờ cứng).
2. `--kiem-tra` → sửa `OPENCODE_BASE_URL` nếu 404.
3. `--thu 300` → xem nhanh ~50 ca, sửa `rubric.py` nếu chấm lệch (tăng `PHIEN_BAN_RUBRIC`).
4. `--tat-ca`.
5. Người chấm mù ~300 mẫu chia tầng theo dải điểm để đo độ đồng ý (chưa viết script xuất bảng chấm).

## QUY TRÌNH CHÍNH (chốt 30/10/2026 tối): DeepSeek CHỌN Gemma hay Qwen cho Title + Mô tả, ở cả bộ CỤM và bộ PAIR
Hai bộ cuối = (Title do judge chọn, dùng chung) + (Mô tả do judge chọn riêng cho từng bộ), rồi fine-tune hai bộ và so sánh.
```powershell
python -I -X utf8 scripts/judge/so_cap_du_lieu.py                  # dựng 3 bảng mục (đã chạy): title 25.383 | cum 7.562 | pair 34.608 => 67.553 mục, 135.106 lời gọi (bỏ các mục chỉ khác khoảng trắng/dấu câu)
.\scripts\judge\chay_judge.ps1 --so-cap --loai title --thu 60 --tat-suy-luan --ten t60       # thí điểm (chia tầng), tương tự c60, p60
.\scripts\judge\chay_judge.ps1 --so-cap --loai cum   --thu 60 --tat-suy-luan --ten c60
.\scripts\judge\chay_judge.ps1 --so-cap --loai pair  --thu 60 --tat-suy-luan --ten p60
.\scripts\judge\chay_judge.ps1 --so-cap --bao-cao t60                                          # thiên vị vị trí/độ dài, tỉ lệ có người thắng
# chạy cả đợt: --tat-ca --tran-usd N (chạy tiếp được), tên đợt dùng ở ghep_bo
python -I -X utf8 scripts/judge/ghep_bo.py --kiem-vong                                         # 0 quyết định phải ra đúng bộ gốc (đã đạt, 25.575 sản phẩm, cả hai bộ)
python -I -X utf8 scripts/judge/ghep_bo.py --title t_title --cum t_cum --pair t_pair           # ghép -> data/processed/bo_chon/bo_{cum,pair}_chon_deepseek.parquet
```
- Bộ CỤM: mục = cặp (tên: giá trị) mà Gemma (A) và Qwen (B) sau rà soát khác nhau (`song_ngu_4/cap_cum_*`, `song_ngu_3/so_sanh_cum_*`); bộ PAIR: mục = đơn vị (tên hoặc giá trị thuộc tính, dịch riêng; `song_ngu_1/2/don_vi_da_chon_*`), bỏ đơn vị khóa bằng từ điển (`tu = co_dinh`). A = Gemma, B = Qwen.
- Chỉ thay khi hai lượt (hoán đổi vị trí) cùng chọn một mô hình; chưa chấm = giữ hiện trạng; chấm nhưng không người thắng = `--khi-khong-quyet` (mặc định giữ hiện trạng, CÙNG chính sách cho hai bộ); chỉ thay vào chỗ hiện trạng đúng bằng A hoặc B (khác = từ điển/quy tắc khóa); chặn bản mới nếu gây Hán sót (1688) hoặc thêm chữ Việt có dấu (Tiki).
- Lời nhắc KHÔNG có ngữ cảnh sản phẩm (chỉ nguồn + hai bản + nhóm hàng), đối xứng cho cả hai bộ.

Chạy thử vài mẫu bằng MỘT lệnh (dán khóa một lần, tự chạy title + cụm + pair + báo cáo, tự xóa khóa): `.\scripts\judge\thu_so_cap.ps1` (mặc định 10 mục mỗi loại = 60 lời gọi, trần 0,5 USD mỗi loại; `-N 30` nhiều hơn; `-Gia` chạy máy móc không cần khóa).

### Bộ CỤM có ngữ cảnh (chốt 31/10/2026) và nhà cung cấp Alibaba
- **Giao thức đã thống nhất:** bộ CỤM do mô hình dịch khi thấy cả tiêu đề và danh sách thuộc tính, nên judge cũng xem ngữ cảnh đó (cách B: MỌI mục). Bộ PAIR (dịch từng đơn vị, không ngữ cảnh) và title dùng cách C (hai tầng: tầng 1 không ngữ cảnh, chỉ mục chưa có người thắng mới chạy lại có ngữ cảnh; tầng 2 của pair/title ĐÃ viết 31/10, xem mục dưới). Mục vẫn chưa quyết: giữ hiện trạng và đánh dấu riêng (file `bo_<cum|pair>_danh_dau.parquet`). Phải ghi rõ trong báo cáo rằng giao thức judge KHÁC nhau giữa hai bộ.
- Mục cụm có ngữ cảnh: `so_cap_du_lieu.py` dựng `muc_cum_ctx.parquet` (22.309 mục = (sản phẩm, cặp), id `cumctx:<nguồn>:<product_id>:<pair_id>`). Ngữ cảnh = tiêu đề NGUỒN + mọi thuộc tính NGUỒN của sản phẩm theo pair_id (không có bản dịch nào); cắt ở 3.000 ký tự (giữ cặp đang xét và các cặp kề; trung vị 549 ký tự, chỉ 1 mục bị cắt). Lời nhắc có khối `PRODUCT CONTEXT` đặt trước `SOURCE` và hệ thống thêm đúng một đoạn giải thích (`HE_THONG_NGU_CANH`); mục không ngữ cảnh giữ nguyên lời nhắc cũ. Bản ghi mang `phien_ban = so_cap_v1+ngu_canh`, `dai_ngu_canh`, `id_khong_ngu_canh`.
- Thí điểm so hai cách trên CÙNG cặp, một lệnh: `.\scripts\judge\thu_cum_ngu_canh.ps1 -NhaCungCap alibaba` (mặc định 40 cặp: 20 của 1688 + 20 của Tiki; chạy có ngữ cảnh, chạy lại không ngữ cảnh, rồi `so_sanh_ngu_canh.py`: ma trận chuyển, ca đảo ngược, tokens/giây mỗi mục, ước tính toàn bộ). `-Gia` chạy máy móc không cần khóa.
- Ghép: `ghep_bo.py --cum <tên kết quả cum_ctx>` đọc kết quả có ngữ cảnh theo (sản phẩm, cặp), mỗi cặp ở mỗi sản phẩm có quyết định riêng; `--cum-khong-ngu-canh` đọc kết quả loại `cum` cũ.
- `--nha-cung-cap alibaba`: Alibaba Cloud Model Studio, mô hình `deepseek-v4.1-flash`, khóa `DASHSCOPE_API_KEY`, địa chỉ lấy từ biến `ALIBABA_BASE_URL` (có WorkspaceId; gói Token Plan dùng `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` nhưng docs Alibaba ghi gói này chỉ dành cho công cụ lập trình tương tác); tắt suy luận bằng `enable_thinking=false`.

### TẦNG 2 của title và pair + file đánh dấu (31/10/2026)
- Vì judge gần như tất định (chạy lặp giống 97%), hỏi lại cùng lời nhắc không gỡ được mục `khong_nhat_quan`; chỉ thông tin mới (ngữ cảnh) mới làm đổi kết quả. Tầng 2 chạy lại CHỈ các mục `khong_nhat_quan` của tầng 1 (mục `hoa_that` giữ nguyên, không chạy lại).
- Bảng mục: `muc_title_ctx.parquet` (25.383 mục; ngữ cảnh = thuộc tính NGUỒN của chính sản phẩm, không có dòng Title vì văn bản đang xét là tiêu đề) và `muc_pair_ctx.parquet` (34.608 mục; ngữ cảnh = tiêu đề + thuộc tính NGUỒN của MỘT sản phẩm đại diện = product_id nhỏ nhất theo thứ tự chuỗi trong số sản phẩm dùng đơn vị đó; quyết định áp cho đơn vị ở mọi sản phẩm). CÙNG `item_id` với tầng 1 nên kết quả tầng 2 ghi đè đúng mục. Ngữ cảnh cắt ở 3.000 ký tự (trung vị 353 ký tự title, 518 ký tự pair).
- Chạy: `so_cap_chon.py --loai title_ctx --chi-khong-nhat-quan <đợt tầng 1> --tat-ca --tat-suy-luan ...` (tầng 2 BẮT BUỘC kèm `--chi-khong-nhat-quan`). Thí điểm một lệnh: `.\scripts\judge\thu_tang2.ps1 -NhaCungCap alibaba -SongSong 8` (dùng title_100 và pair_100; ghi `log_thu_tang2_*.txt`, `ket_qua_*_t2.jsonl`, `so_sanh_tang_*.json`). `so_sanh_tang.py` cho ma trận tầng 1 sang tầng 2, số mục gỡ được, token.
- Ghép: `ghep_bo.py --title <t1>,<t2> --pair <p1>,<p2> --cum <cum_ctx>`. Bản ghi tầng 2 mang `tang: 2` và THẮNG tầng 1 của cùng mục bất kể thứ tự tên đợt. Với mỗi mục TRANH CHẤP (hai ứng viên khác nhau thật sự, không tính đơn vị `co_dinh`) ghi nhãn nguồn quyết định: `judge_t1`, `judge_t2`, `hoa_that`, `khong_nhat_quan`, `chua_cham`; file `bo_<cum|pair>_danh_dau.parquet` có mỗi sản phẩm một dòng (`title_nhan`, `title_da_doi`, `mo_ta_so_manh_*`); `ghep_bo_thong_ke.json` có đếm theo nhãn. Bộ huấn luyện `bo_*_chon_deepseek.parquet` KHÔNG đổi cột.
- Đã thử đầu cuối bằng kết quả tầng 1 GIẢ trên dữ liệu thật (không gọi API): số kết quả tầng 2 = số mục `khong_nhat_quan`, tổng nhãn = số mục tranh chấp (title 25.575 sản phẩm gồm 192 không tranh chấp; mảnh mô tả cụm 22.309, pair 223.132), cột bộ huấn luyện giống bộ gốc, `--kiem-vong` vẫn giống hệt bộ gốc.
### Bộ quy tắc v2 và chế độ MỘT LƯỢT (31/10/2026, chờ đo)
- Yêu cầu mới của Nhật Anh: judge chỉ chấm theo tiêu chí và chọn MỘT lần (không hoán đổi hai thứ tự), bắt buộc chọn 1 trong 2. Số đo cũ: thiên vị vị trí 1 (~63%), một lượt cố định thứ tự làm Gemma 85% khi đứng trước và 59% khi đứng sau; trong 94 mục "không nhất quán", 30 mục cả hai lượt đều nêu `meaning` nhưng chọn khác nhau => tiêu chí chưa đủ chặt.
- `so_cap_rubric.py`: `HE_THONG_V2` (+ bản có khối ngữ cảnh) = 4 bước: (1) liệt kê lỗi cụ thể từng bản (WRONG, MISSING, ADDED, BROKEN, TERM; phải chỉ đúng chữ mới được nói thiếu/thêm), (2) ít lỗi hơn thắng, WRONG/BROKEN nặng hơn MISSING/ADDED nặng hơn TERM, (3) các khác biệt KHÔNG BAO GIỜ là lý do: từ đồng nghĩa ngang vị thế (kể cả dạng rút gọn và đầy đủ), định dạng đơn vị/số/dấu câu, thứ tự từ giữ nghĩa, độ trôi chảy trừ khi sai ngữ pháp, độ dài, vị trí, (4) nếu vẫn tương đương thì `equivalent: true` và chọn bản 1. JSON theo thứ tự: defects_1, defects_2, equivalent, winner, criterion, reason; không có "tie". Các dòng CHÍNH SÁCH (thương hiệu/tên Latin, từ ngoại ngữ thông thường, từ đồng nghĩa, định dạng đơn vị) nằm riêng trong `QUY_TAC_CHINH_SACH`, Nhật Anh ĐÃ CHỐT 31/10: trung lập cả bốn, kèm một ngoại lệ ở thương hiệu (đổi thành thương hiệu KHÁC, vd phiên âm bịa, là lỗi WRONG). Lời nhắc KHÔNG chứa ví dụ thật của các mục đã đo.
- `so_cap_chon.py --rubric v2 --che-do hai_luot|mot_luot`: `mot_luot` hỏi đúng một lời gọi, thứ tự ngẫu nhiên theo băm cố định của item_id (một nửa Gemma trước, một nửa Qwen trước; tái lập được; item_id không vào lời nhắc); kết quả = lượt duy nhất; bản ghi có `che_do`, `tuong_duong`, `so_loi_1/2`. Chạy tiếp phân biệt phiên bản và chế độ.
- Đo: `.\scripts\judge\thu_quy_tac.ps1 -NhaCungCap alibaba -SongSong 8` chạy v2 đủ hai thứ tự trên CÙNG 300 mục (đã kiểm: cùng mẫu với v1), rồi `so_sanh_quy_tac.py` cho (theo tập dev/test chia bằng băm item_id; số ở tập test đáng tin hơn): tỉ lệ không nhất quán v1 so v2, trong đó lệch CÓ khai báo equivalent (đồng xu công bằng) và lệch KHÔNG khai báo (judge không ổn định thật), thiên vị vị trí, và MÔ PHỎNG chế độ một lượt (trùng kết quả hai thứ tự ở mục đã quyết, tỉ lệ khai báo tương đương). Ghi `so_sanh_quy_tac.json` kèm các mục còn lệch mà không khai báo để đọc.
- Chỉ khi số đo cho thấy một lượt đủ ổn mới dùng chế độ một lượt cho lần chạy lớn; sau đó sửa `ghep_bo.py` để nhận bản ghi một lượt (mục `equivalent` có nhãn riêng). Chưa làm.
### SỬA LỖI BẢN RỖNG (31/10/2026) - bảng mục mới, số mục thay đổi
- Phát hiện: ở bộ CỤM 1688, 6.193 cặp `nguon_ban_dich = dich_moi` chỉ có bản của Gemma (`v_cuoi_B` rỗng: Qwen không dịch cặp đó). Trước đây các cặp này bị đưa vào bảng mục (5.139 mục = 23% mục cụm có ngữ cảnh; 3.116 = 41% cặp duy nhất) nên judge "thắng" cho Gemma một cách tầm thường (thí điểm: 18/100 mục `cum_100`, 4/40 mục `cumctx_40` có bản rỗng, hầu hết thành Gemma).
- Sửa: một cặp/đơn vị/tiêu đề có ứng viên RỖNG không phải "cuộc đấu", không đưa vào bảng mục (`co_chu`, `co_noi_dung` trong `so_cap_du_lieu.py`; `xay_title`, `xay_don_vi`, `xay_cum`, `doc_cap_cum`) và `ghep_bo.py` không đếm chúng là mục tranh chấp. Phần mô tả đó giữ bản của Gemma như bộ official (vì Qwen không có bản). Ghép với 0 quyết định vẫn giống hệt bộ official ở cả hai bộ.
- Số mục mới: title 25.374 (-9), cụm không ngữ cảnh 4.446 (-3.116; 1688 2.609, Tiki 1.837), cụm có ngữ cảnh 17.170 (-5.139; 1688 15.195, Tiki 1.975), pair 34.594 (-14) = 64.414 mục = 128.828 lời gọi hai thứ tự (trước 135.106).
- Hệ quả: các số đo cụm của thí điểm cũ bị thổi phồng về Gemma và nhẹ tỉ lệ lệch (cum_100 v1: 28/82 lệch = 34% trên cặp thật; v2: 35/82 = 43%). Muốn số mới phải chạy lại thí điểm (mẫu 100 mới khác mẫu cũ vì bảng mục đổi).
### CHẾ ĐỘ GOM + BẢNG CỤM LAI + PAIR ƯU TIÊN (31/10/2026, theo ngân sách ~15 triệu token)
- Ngân sách: tổng ~15 triệu token, đã dùng ~2,73 triệu (92% là token VÀO). 85% token vào của title/pair là phần quy tắc v2 (~727 token) lặp lại ở MỖI lời gọi. Quyết định của Nhật Anh: judge xem TẤT CẢ tiêu đề; bản cụm dùng cách LAI.
- `so_cap_gom.py`: gom K mục (mặc định 15; mục có ngữ cảnh gom ít hơn: max(3, K//2)) vào MỘT lời gọi, rubric v2, một lượt; quy tắc gửi một lần; thứ tự hai bản của TỪNG mục ngẫu nhiên theo băm; judge xét từng mục độc lập; đầu ra gọn `{"results":[{"i","d1","d2","eq","w","c"}]}`. Mục thiếu/hỏng thì ghi lỗi (chạy tiếp chỉ làm lại các mục đó). Chạy: `so_cap_chon.py --loai title --tat-ca --rubric v2 --che-do mot_luot --gom 15 --tat-suy-luan ...`.
- Bảng cụm LAI `muc_cum_lai.parquet` (`xay_cum_lai`): 4.446 mục = 3.838 cặp dùng bởi ĐÚNG 1 sản phẩm (CÓ ngữ cảnh, id `cumctx:`) + 608 cặp dùng bởi >= 2 sản phẩm (không ngữ cảnh, id `pair:`). Chạy `--loai cum_lai`. `ghep_bo.py --cum-che-do lai` (mặc định) tự chọn id theo số sản phẩm dùng cặp; đã kiểm trên dữ liệu thật: 17.170/17.170 mảnh tìm đúng quyết định.
- `--top N`: chỉ giữ N mục dùng nhiều nhất (`san_pham_n`), xếp giảm dần (ưu tiên). Pair: top 3.347 đơn vị phủ 95% lượt dùng.
- `ghep_bo.py --khi-tuong-duong giu_hien_tai|theo_judge` (mặc định giu_hien_tai, đề xuất của Claude, CHỜ Nhật Anh xác nhận): mục judge khai tương đương có nhãn `judge_tuong_duong`, giữ bản hiện tại hoặc lấy bản judge chọn (thứ tự đã ngẫu nhiên hóa).
- Thí điểm gom (cần làm trước khi chạy lớn): `.\scripts\judge\thu_gom.ps1 -NhaCungCap alibaba -SongSong 8` chạy gom 15 trên CÙNG 450 mục của `do_lai_quy_tac.ps1` và `so_sanh_gom.py` so với kết quả hai thứ tự không gom (tiêu chí: gom trùng mục bền vững >= 95%, tỉ lệ khai tương đương gần như không đổi, mục lỗi < 2%, token mỗi mục giảm rõ). Ước chi phí 0,15-0,3 triệu token.
### KẾT QUẢ THÍ ĐIỂM GOM + SỬA LỖI ĐƠN VỊ THEO NHÓM HÀNG + LAUNCHER CHẠY LỚN (31/10/2026)
- Thí điểm gom 15 (450 mục = 150 title + 150 cum_ctx + 150 pair, cùng mẫu với `m2_v2_*`): 0 mục lỗi/thiếu; ~0,16 triệu token; token mỗi mục 360 so với 1.060 khi một mục một lời gọi (title 239 so 958, cụm có ngữ cảnh 642 so 1.333, pair 200 so 889). Gemma/Qwen 314/136 so 313/137; chọn vị trí 1 73,8% so 73,1%; khai tương đương 39,8% so 36,9% (title 23% so 11%). Trên 247 mục bền vững (hai thứ tự đồng ý): gom đúng 202 (81,8%), khai tương đương 41 (16,6%; giữ hiện trạng nên không sai), ĐẢO NGƯỢC không khai chỉ 4 (1,6%). Mục lệch hai thứ tự mà gom không khai: 65/450 (14,4%; một mục một lời gọi: 12,7%). Số "trùng 90,3%" trong `so_sanh_gom.json` coi cả các lần gom khai tương đương là lệch; không phải tỉ lệ sai.
- LỖI ĐÃ SỬA: đơn vị pair là THEO NHÓM HÀNG (cùng chữ nguồn có nhiều `ma_don_vi`, vd "其他" có 5+ biến thể). `san_pham_n` trước đây đếm theo chữ không kể nhóm hàng (đếm lặp: tổng 769.654 so với 195.311 mảnh thật) và `ghep_bo.py` chỉ lấy biến thể đầu tiên. Nay `san_pham_n`, sản phẩm đại diện và ghép đều theo (chữ, nhóm hàng). Số đơn vị pair 34.534 (-60), mảnh tranh chấp bộ pair 195.311.
- ĐÍNH CHÍNH độ phủ `--top N` (đo lại theo MẢNH): top 1.000 đơn vị 72,6%, top 3.347 81,7%, top 5.000 83,9%, top 8.000 86,4%, top 12.000 88,5%, tất cả 34.534 = 100% (~6,9 triệu token). Con số "top 3.347 phủ 95%" cũ SAI vì đếm lặp. Phần chưa chấm giữ bản hiện tại.
- `scripts/judge/chay_lon.ps1`: chạy lớn theo giai đoạn pair (top N) -> cum LAI -> title, mỗi giai đoạn có trần token, chạy tiếp được, tự ghép hai bộ cuối khi xong (chỉ lưu trên máy), ghi log. `-Gia` chạy cả quy trình bằng judge giả (đã thử: ghép ra hai bộ 25.575 sản phẩm + file đánh dấu). Có thể chạy từng giai đoạn (`-ChiGiaiDoan pair|cum|title|ghep`) và tăng `-TopPair` để chạy tiếp phần pair còn lại khi còn ngân sách.
## CHẤM ĐIỂM ĐƠN CẢ HAI BỘ (phụ: kiểm toán chất lượng sau khi chọn)
Cùng 25.575 sản phẩm, cùng một judge, cùng cấu hình (PHẢI giống hệt nhau giữa các lượt chạy để so sánh công bằng: cùng `--tat-suy-luan` hay không, cùng rubric v3).
```powershell
python -I -X utf8 scripts/ft_clean/xuat_ban_pair.py                      # đã chạy: ft_buoc4_giu_pair.parquet (25.575 sản phẩm)
.\scripts\judge\chay_judge.ps1 --doi-tuong mo_ta       --tat-ca --tat-suy-luan --ten cum_toan_bo      # mô tả bản CỤM
.\scripts\judge\chay_judge.ps1 --doi-tuong mo_ta_pair  --tat-ca --tat-suy-luan --ten pair_toan_bo     # mô tả bản PAIR
.\scripts\judge\chay_judge.ps1 --doi-tuong title       --tat-ca --tat-suy-luan --ten title_toan_bo    # title (giống hệt ở hai bộ)
python -I -X utf8 scripts/judge/so_sanh_hai_bo.py cum_toan_bo pair_toan_bo --ten-title title_toan_bo
```
Thí điểm trước khi chạy cả đợt: thay `--tat-ca` bằng `--thu 300` (mẫu chia tầng giống nhau giữa các lượt vì hạt giống cố định; cụm và pair cùng product_id).
Chạy tiếp được; thêm `--tran-usd N` để chặn chi phí. Title có rubric riêng (cho phép dồn từ khóa, bắt rò lời nhắc); mô tả pair và cụm dùng chung một rubric (đoạn nối bằng "; " hoặc xuống dòng) và judge KHÔNG được báo đây là bản nào. `so_sanh_hai_bo.py` cho điểm trung bình từng chiều, chênh lệch A-B có khoảng tin cậy 95% (bootstrap có cặp theo sản phẩm), tỉ lệ `dat`/`can_xem`, loại lỗi; đây là chất lượng dữ liệu theo judge, chưa phải kết quả fine-tune và chưa đối chiếu người chấm.

## SO CẶP MÙ: chọn 1 trong 2 bản dịch Gemma / Qwen (30/10/2026; công cụ phụ, chưa dùng)
`so_cap_du_lieu.py` (dựng bảng mục, miễn phí) -> `so_cap_chon.py` (hỏi judge) -> `--bao-cao` (thống kê + kiểm tra thiên vị). Chạy qua launcher có nhập khóa ẩn:
```powershell
python -I -X utf8 scripts/judge/so_cap_du_lieu.py                                         # đã chạy: 25.517 title + 8.168 cặp
.\scripts\judge\chay_judge.ps1 --so-cap --loai title --thu 60 --tat-suy-luan --ten t60    # thí điểm chia đều tầng (nguồn x margin Kiwi)
.\scripts\judge\chay_judge.ps1 --so-cap --loai pair  --thu 60 --tat-suy-luan --ten p60
.\scripts\judge\chay_judge.ps1 --so-cap --bao-cao t60
```
Tiêu chí chọn (viết trong lời nhắc, theo thứ tự quan trọng): meaning, completeness, numbers_codes, terminology, fluency; judge trả `winner` 1/2/tie, `criterion` quyết định, `reason` <= 20 từ.
Chống thiên vị (đều có bài thử tự động trong `test_so_cap.py`, 12 bài):
1. MÙ: lời nhắc chỉ có nguồn + "Translation 1/2" + nhóm hàng + loại văn bản; không tên mô hình, không điểm CometKiwi, không id (quét cả 33.685 mục thật: 0 từ cấm trong khung).
2. ĐỐI XỨNG: luôn hỏi đủ hai thứ tự (Gemma ở vị trí 1, rồi Qwen ở vị trí 1); hai lời nhắc giống hệt trừ chỗ hoán đổi.
3. KHÔNG MẶC ĐỊNH: có người thắng chỉ khi hai lượt cùng chọn MỘT mô hình; còn lại là `hoa_that` (cả hai lượt "tie") hoặc `khong_nhat_quan` (lệch nhau, gồm cả thiên vị vị trí). Không có luật "hòa thì lấy Gemma".
4. Kiểm toán: `--bao-cao` in tỉ lệ chọn vị trí 1 (gần 0,5 là tốt), tỉ lệ Gemma thắng theo nguồn, thiên vị độ dài, tiêu chí quyết định, mức trùng với CometKiwi theo margin.
Bản giả (`--gia`) chọn bản dài hơn để chứng minh bộ kiểm toán bắt được thiên vị độ dài.

## Ghi chú
- Hướng dịch khác nhau theo nguồn: 1688 nguồn là tiếng Trung (bản Việt do máy dịch); Tiki nguồn là tiếng Việt (bản Trung do máy dịch). `rubric.py` truyền đúng hướng cho từng dòng.
- Chiều chấm: sai_nghia, thieu_y, them_y, thuat_ngu, so_don_vi, troi_chay (1-5), ket_luan (dat / can_xem / loi), danh sách lỗi.
- Cắm cờ: ket_luan khác `dat` hoặc có chiều <= 3 (`NGUONG_CO`, đặt tạm).
- Chưa kiểm chứng với khóa thật: địa chỉ `https://opencode.ai/inference/openai/v1`, tên mô hình, việc mô hình có nhận `response_format=json_object` (nếu máy chủ từ chối thì tự bỏ tham số đó), độ tốt của rubric trên cặp Việt-Trung.
- Kết quả là file riêng, không sửa bộ chính thức trên HF.
- Thử máy móc không gọi API: `python -I -X utf8 test_judge.py` (6 bài).
