# Qwen3-14B-AWQ Kaggle v6 — hướng dẫn chuyển và chạy

Bản này là notebook riêng cho Kaggle, được tạo từ Colab v6. Model, prompt, sampling, QA và cách nhập cache v2 được giữ nguyên. Thay đổi chỉ ở lưu trữ/runtime. Notebook gốc Colab và dữ liệu Drive không bị sửa.

## A. Tạo private Dataset đầu vào

Notebook cần bundle v6, checkpoint v2 cũ và Parquet nguồn để dùng đầu vào Kaggle cục bộ. Tải các tệp sau:

- [Bundle v6](https://drive.google.com/file/d/10IxY4bGCF1QrRtMMTIGRIGAHr66G53o1/view)
- Parquet nguồn `bilingual_zh_vi_cado.parquet` đã có trong thư mục dự án và cần được đưa vào private Kaggle Dataset.
- Tải checkpoint SQLite [cache](https://drive.google.com/file/d/1EozHEpTyYmjhqZS9VFBK1WyV7T3kcCj5/view) và [environment lock](https://drive.google.com/file/d/15VuDB17N1kNbUdla-TQzFMXGNulbSpI-/view) từ v2. Status không bắt buộc; hãy ngắt phiên Colab cũ trước khi chuyển để tránh thay đổi checkpoint đồng thời.

Trên máy, đặt bundle, Parquet nguồn, cache và environment lock vào cấu trúc sau rồi nén thư mục `qwen3_14b_awq_kaggle_input` thành ZIP:

```text
qwen3_14b_awq_kaggle_input/
├── title_translation_qwen3_14b_awq_colab_v6_sampled_autoreload_bundle.zip
├── bilingual_zh_vi_cado.parquet
└── title_translation_cado_qwen3_14b_awq_full_v2/
    ├── cache_qwen3_14b_awq.sqlite3
    └── hybrid_environment_lock.json
```

Trong Kaggle, vào **Datasets → New Dataset**, tải ZIP lên, đặt quyền **Private**, rồi tạo Dataset. Không công khai Dataset này: SQLite có chứa title và bản dịch. Kaggle tự giải nén ZIP khi gắn làm Input.

Ngoài Dataset dữ liệu, gắn thêm private dependency Dataset đã chuẩn bị sẵn: `nadsa091/qwen3-14b-awq-offline-wheelhouse`. Dataset này chỉ chứa package và manifest SHA-256 để cài dependency offline; không chứa title hay bản dịch.

## B. Import notebook và bật GPU

1. Tải `title_translation_qwen3_14b_awq_kaggle_v6.ipynb` từ Drive hoặc file đính kèm cùng câu trả lời về máy.
2. Kaggle → **Create → New Notebook** → **File → Import Notebook** → chọn file tải về.
3. Ở **Add Input**, thêm cả private Dataset dữ liệu vừa tạo và `nadsa091/qwen3-14b-awq-offline-wheelhouse`.
4. Trong **Settings / Session options**, chọn **Accelerator = GPU**. Internet có thể tắt vì package, Parquet và model đều nằm trong Kaggle Input. Nếu có **Persistence**, chọn **Files only** trước khi khởi chạy.
5. Bấm **Start Session**, sau đó **Run All**. Cell đầu sẽ xác nhận bundle và cache v2 đã được tìm thấy; cell GPU sẽ dừng ngay nếu không có CUDA, VRAM dưới 14 GiB hoặc compute capability thấp hơn 7.5.

T4/T4x2 có thể phù hợp. Kaggle có thể cấp P100; P100 không chạy được kernel ExLlamaV2 được chọn bởi cấu hình v6, nên notebook sẽ dừng trước khi dịch. GPU/quota/khả dụng phụ thuộc tài khoản và thời điểm.

## C. Chạy tiếp nếu Kaggle ngắt

- Nếu **Files only** persistence đang bật, giữ nguyên notebook/session để `/kaggle/working/title_translation_cado_qwen3_14b_awq_full_v3_sampled` được mang sang lần kết nối lại; chạy lại từ đầu để nạp model và resume cache.
- Nếu persistence không bật, chạy cell cuối để tạo `qwen3_14b_awq_kaggle_checkpoint.zip` rồi tải ZIP từ Output trước khi đóng phiên. Đưa ZIP này vào một version mới của private Kaggle Dataset; giữ nguyên bundle v6 và thư mục v2 ban đầu. Gắn version Dataset mới vào notebook rồi chạy lại.
- Không bấm **Save Version → Save & Run All** để “resume” khi một phiên đang chạy: thao tác đó chạy notebook từ đầu trong một session riêng. Dùng **Run All** trong session tương tác hiện tại.
- Mỗi 100 title, SQLite cache được mirror vào `/kaggle/working`. Nếu dừng cưỡng bức, có thể mất tối đa phần chưa kịp mirror kể từ checkpoint cuối.

Kaggle không mount Google Drive và bản này không hot-reload `qwen_control.json`. Nếu cần cập nhật code, dừng ở checkpoint, cập nhật notebook/bundle đã review, rồi chạy lại. Mọi output và status Kaggle vẫn ở `/kaggle/working/title_translation_cado_qwen3_14b_awq_full_v3_sampled`; notebook không tự đồng bộ chúng ngược về Drive.

Dependency, Parquet dữ liệu và model đều được nạp từ Kaggle Input; bước chạy không cần tải file qua Hugging Face hoặc PyPI.

## D. Giới hạn hiện tại

- Kaggle Notebook có giới hạn thời lượng phiên/quota GPU; tài liệu hiện tại nêu tối đa 12 giờ cho phiên notebook và khoảng 30 giờ GPU mỗi tuần (có thể thay đổi theo nhu cầu).
- Trước khi rời phiên, xác nhận checkpoint ZIP đã xuất/tải hoặc Files persistence được bật.
- Model được nạp từ Kaggle Input ở runtime mới; Parquet cũng được đọc trực tiếp từ Kaggle Input.
- Python dependencies được cài offline từ wheelhouse; Internet có thể tắt.
- Mọi code cell có trạng thái bước bắt đầu/kết thúc và heartbeat 30 giây trong `qwen_health.json` cùng log `[qwen-health]`. Thời điểm tiến độ cuối tách riêng với heartbeat; quá ngưỡng sẽ báo `suspected_stall`, không tự dừng tiến trình. Bộ đọc local phải báo không quan sát được nếu API Kaggle không cung cấp log; không suy ra tiến độ từ `RUNNING`.

Tài liệu Kaggle: [Notebooks](https://www.kaggle.com/docs/notebooks), [Datasets](https://www.kaggle.com/docs/datasets), [GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage).
