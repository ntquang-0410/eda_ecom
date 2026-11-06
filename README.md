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

## BỘ SONG NGỮ THỨ NHẤT: HƯỚNG DẪN CHẠY (CPU VÀ GPU)

Pipeline ý 1 của hướng mới (thiết kế: `docs/design/pipeline_bo_song_ngu_1.html`): lấy chữ Trung của file cặp desc_v1, bỏ hẳn bản Việt cũ, cho mô hình A dịch (từ 16/10/2026 là Gemma 4 26B-A4B qua Google AI Studio bản miễn phí, notebook riêng; trước đó thử DeepSeek V4.1 Flash) và mô hình B dịch (từ 22/10/2026 là Qwen3-14B bản AWQ chính thức chạy trên Kaggle; HY-MT1.5-7B bỏ vì giấy phép), kiểm luật, chấm CometKiwi, chọn một bản Việt cho mỗi cặp. Laptop chạy các bước nhẹ (CPU), Kaggle chạy phần dịch và chấm điểm (GPU T4 x2).

| File | Vai trò |
|---|---|
| `src/song_ngu_1/` | Mã xử lý: `chuan_bi.py` (bước 1-3), `luat_kiem.py` (bước 5), `chon_ban.py` (bước 6b), `hoan_tat.py` (bước 5, 6b, 7), `cau_hinh.py` (mô hình, ngưỡng, đường dẫn) |
| `src/song_ngu_1/kaggle/` | `dich.py` (bước 4), `cham_kiwi.py` (bước 6a): chạy trên Kaggle |
| `scripts/song_ngu_1/` | `chay_chuan_bi.py`, `chay_hoan_tat.py`, `test_song_ngu_1.py` |
| `notebooks/song-ngu-1-kaggle.ipynb` | Notebook Kaggle (GPU): B (HY-MT, dùng cả 2 GPU) dịch, rồi chấm CometKiwi cho A và B song song; A cũ (DeepSeek) tắt mặc định |
| `notebooks/song-ngu-1-gemma-kaggle.ipynb` | Notebook Kaggle (không GPU): A (Gemma 4 26B) dịch qua API, nhiều khoá |

**Bước A (laptop): thử và chuẩn bị**
```
.venv\Scripts\python.exe scripts\song_ngu_1\test_song_ngu_1.py      # phải ra 50/50 ca đạt
.venv\Scripts\python.exe scripts\song_ngu_1\chay_chuan_bi.py         # khoảng 10 giây
```
Đầu vào mặc định: `data/release/desc_v1/bilingual_zh_vi_desc_pairs(nhatanh).parquet`. Ra `data/processed/song_ngu_1/bang_cap.parquet`, `the_chuan_bi.json` và thư mục `data/processed/song_ngu_1/vao_kaggle/` (4 file). Chạy lại bước này mỗi khi sửa mã trong `src/song_ngu_1/kaggle/`, để thư mục tải lên có bản mới nhất.

**Bước B (Kaggle): dịch và chấm điểm**
1. Kaggle > Datasets > New Dataset (Private): tải lên 4 file trong `vao_kaggle/`.
2. Kaggle > Code > Import notebook: chọn `notebooks/song-ngu-1-kaggle.ipynb`. Settings: Accelerator **GPU T4 x2**, Internet **On** (tài khoản cần xác minh số điện thoại).
3. Add-ons > Secrets: tạo secret `HF_TOKEN` (token đọc trên huggingface.co, đã chấp nhận điều khoản ở trang `Unbabel/wmt22-cometkiwi-da`), bật Attach. Từ 15/10/2026 notebook này KHÔNG dịch mô hình A (`CHAY_DICH_A = False`, không gọi API trả phí): A = Gemma 4 dịch ở notebook riêng `song-ngu-1-gemma-kaggle.ipynb`, notebook này chỉ dịch B (HY-MT) và chấm CometKiwi cho cả hai; muốn chấm bản Gemma thì Add Data > Notebook Output của notebook Gemma (thư mục `ra_gemma`). `DASHSCOPE_API_KEY` chỉ cần khi đặt `CHAY_DICH_A = True` (cách DeepSeek cũ; KHÔNG dùng khoá Token Plan `sk-sp-`). Chỉ dán giá trị vào ô Value của Kaggle, không dán vào notebook hay vào chat.
4. Add Data: gắn Dataset ở mục 1.
5. **Chạy thử trước (mốc M0):** để `CHE_DO = "thu"` ở ô Cấu hình, bấm **Save Version > Save & Run All (Commit)**. Xong thì mở tab Output: phải có `ra_thu/kiem_luu.txt`, `ra_thu/deepseek_v41_flash/`, `ra_thu/hy_mt15_7b/`, `ra_thu/kiwi_*`, `ra_thu/the_kaggle.json`. Đọc ô cuối: backend nào chạy được (vLLM hay hf) và ước thời gian cả bộ.
6. **Chạy thật:** sửa `CHE_DO = "that"`, Save & Run All lần nữa. Kết quả ở `ra/`.
7. **Nếu chưa xong** (hết `GIO_TOI_DA_DICH` hoặc phiên bị ngắt): tạo phiên bản mới, Add Data > Notebook Output của phiên bản trước, Save & Run All lại. Notebook tự nhận phần đã có và chỉ làm phần còn thiếu. Cứ lặp lại đến khi ô cuối báo đã dịch đủ.
8. Tải thư mục `ra/` của phiên bản cuối về laptop, giải nén vào `data/processed/song_ngu_1/tu_kaggle/`.

**Bước C (laptop): kiểm luật, chọn bản, xuất**
```
.venv\Scripts\python.exe scripts\song_ngu_1\chay_hoan_tat.py --kaggle data\processed\song_ngu_1\tu_kaggle
```
Khoảng 1,5 phút trên 43.687 đơn vị. Ra trong `data/processed/song_ngu_1/` (mỗi lần chạy có dấu thời gian, không ghi đè): `bo_song_ngu_1_day_du_*.parquet` (18 cột), `bo_song_ngu_1_cong_bo_*.parquet` (9 cột), `cat_rieng_*.parquet`, `the_nguon_goc_*.json`, `phan_bo_diem_*.csv`, `do_nhay_nguong_*.csv`, `don_vi_da_chon_*.parquet`. Ngưỡng mặc định là **tạm**: τ = 60 (chrF đủ giống), κ = 0,5 (điểm sàn CometKiwi), κ ngắn = 0,35 (sàn riêng cho chuỗi Trung từ 8 ký tự trở xuống), δ = 0,25 (chênh điểm CometKiwi để bỏ mô hình ưu tiên), ưu tiên mô hình A (DeepSeek). Xem `phan_bo_diem_*.csv` và `do_nhay_nguong_*.csv` rồi chạy lại với `--tau --kappa --kappa_ngan --delta --uu_tien` khác; không phải chạy lại Kaggle.

**Chạy mô hình A bằng Gemma 4 (Google AI Studio) trên Kaggle, không cần GPU:** dùng `notebooks/song-ngu-1-gemma-kaggle.ipynb` (hướng dẫn ở ô đầu). Settings: Accelerator None, Internet On; Secrets: `GEMINI_API_KEY` và tối đa 4 khoá nữa `GEMINI_API_KEY_2` đến `_5` (mỗi khoá một dự án Google thì hạn mức mới riêng; `dich.py` tự chọn khoá nhanh nhất, khoá lỗi hoặc chậm thì nghỉ và chuyển khoá khác, khoá hết hạn mức ngày hoặc sai thì loại); Dataset `vao_kaggle` bản mới (có `dich.py` 62.824 byte, bản cũ thiếu `--gop` và nhiều khoá; ô thứ hai của notebook báo nếu dùng bản cũ). Save & Run All (Commit) rồi đóng tab hoặc tắt máy được, tối đa 12 giờ mỗi phiên (script dừng êm ở 11 giờ), làm tiếp bằng cách gắn đầu ra cũ. Đo ngày 15/10/2026: `gemma-4-26b-a4b-it` khoảng 4 giây mỗi cuộc gọi (100 đơn vị trong 166 giây), `gemma-4-31b-it` 35 đến 180 giây mỗi cuộc gọi (100 đơn vị trong 39 phút). Đầu ra ghi ở `ra_gemma/api_tuy_chon/`; khi chạy `hoan_tat.py` cần tên thư mục mô hình khớp `C.MO_HINH["A"]`.

**Khi gặp lỗi**
- vLLM không chạy trên T4: notebook tự chuyển sang transformers 4-bit (chậm hơn); xem `loi_vllm` trong `the_*.json`. Có thể thử ghim bản: `GOI_VLLM = "vllm==<phiên bản>"`, hoặc ép `BACKEND = "hf"`.
- Không có `HF_TOKEN`: phần dịch vẫn chạy, phần CometKiwi bị bỏ qua; thêm secret rồi chạy phiên bản sau (gắn Output cũ) để chấm riêng.
- Không có `DASHSCOPE_API_KEY`: mô hình A bị bỏ qua, chỉ B dịch; thêm secret rồi chạy phiên bản sau (gắn Output cũ) thì A dịch phần còn thiếu. Gặp lỗi 429 (quá hạn mức) thì giảm `SONG_SONG_API` ở ô Cấu hình; đơn vị lỗi API không được ghi nên lần sau tự dịch lại.
- **API DeepSeek trả phí:** `dich.py` bắt buộc có `--ngan_sach_usd` cho mô hình này (notebook đặt qua `NGAN_SACH_USD_THU` và `NGAN_SACH_USD_THAT`), ước chi phí theo giá giờ cao điểm (dư), chạm ngân sách là dừng và không tốn thêm. Giá Alibaba Cloud (Singapore, mỗi triệu token): vào 0,30 USD và ra 1,20 USD giờ cao điểm, một nửa giờ thấp điểm; cao điểm là 8h đến 22h giờ UTC+8, tức 7h đến 21h giờ Việt Nam hằng ngày, nên chạy thật từ 21h đến 7h sáng. Địa chỉ gọi API lấy trong bảng điều khiển Model Studio và truyền bằng `--api_base_url` (hoặc `API_BASE_URL` ở ô Cấu hình notebook).
- **Chạy mô hình A ngay trên laptop** (không cần Kaggle, kết quả ghi xuống đĩa ngay từng lô nên không sợ mất): trong một cửa sổ PowerShell, đặt khoá bằng `$k = Read-Host "Khoa" -AsSecureString; $env:DASHSCOPE_API_KEY = [System.Net.NetworkCredential]::new("", $k).Password`, rồi chạy `.venv\Scripts\python.exe src\song_ngu_1\kaggle\dich.py --vao data\processed\song_ngu_1\vao_kaggle\don_vi_can_dich.parquet --ra data\processed\song_ngu_1\tu_api --mo_hinh deepseek_alibaba --api_base_url <địa chỉ trong bảng điều khiển> --thu 20 --lo 20 --ngan_sach_usd 0.2` (thử 20 đơn vị; bỏ `--thu 20` và nâng `--ngan_sach_usd` để chạy cả bộ). Đóng cửa sổ là mất khoá trong bộ nhớ, không để lại file.
- Không commit `data/processed/song_ngu_1/` (đã có trong `.gitignore`). Giấy phép HY-MT1.5 hạn chế việc dùng đầu ra để huấn luyện mô hình khác: đọc `License.txt` và hỏi giảng viên trước khi công bố.


