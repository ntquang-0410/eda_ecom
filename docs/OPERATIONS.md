# Vận hành

Chạy các lệnh từ thư mục gốc checkout bằng Python ≥3.10. Cài `requirements.txt` trong virtual environment riêng. Dùng `python manage.py list` để xem lệnh, thêm `--help` sau tên lệnh có CLI argparse.

## Kiểm tra mã không gọi API

```powershell
python -m pytest -q
python scripts/maintenance/check_repository.py
python manage.py list
```

Test dùng dữ liệu tự dựng, fake judge/client và temporary directory; không cần token, model weights hoặc dữ liệu thật. Không chạy archive trong test discovery.

## Dịch title và notebook Qwen

`src/title_translation_hybrid.py` giữ adapter qua các giai đoạn, có cả Qwen3.5 cũ. Runner local ban đầu không phải cấu hình Qwen3-14B-AWQ đã được rewrite. Dùng builder theo đúng phiên bản khi tái tạo notebook AWQ:

```powershell
python manage.py build-qwen-v6
python manage.py build-kaggle-qwen
python manage.py build-qa-repair
```

Kết quả mặc định ở `outputs/generated/`. Thứ tự trên tạo sampled bundle trước QA-repair builder. Các notebook này vẫn cần source/checkpoint và model/wheelhouse inputs tương ứng trên Kaggle. Chúng không được tự push hoặc chạy chỉ vì builder hoàn tất. Hướng dẫn Colab/interactive sinh từ builder là tài liệu của phiên bản lịch sử; quy trình batch tự chủ dùng supervisor và private inputs có kiểm chứng.

Local runner nhận `ECOM_SOURCE`, `ECOM_WORKDIR` và `ECOM_WORKSPACE` qua môi trường. Không cung cấp khóa trong command-line hoặc file checked-in. Ví dụ chỉ đặt đường dẫn dữ liệu:

```powershell
$env:ECOM_SOURCE='D:/my-data/bilingual_zh_vi_cado.parquet'
$env:ECOM_WORKDIR='D:/my-runs/title-1688'
python manage.py translate-local-legacy --help
```

## COMETKiwi

```powershell
python manage.py compare-1688 prepare --source SOURCE.parquet --gemma GEMMA.parquet --qwen QWEN.parquet --workdir outputs/comparison_1688
python manage.py build-comet-1688
```

Scorer dùng model/revision đã pin trong source. HF model access cần tài khoản được cấp quyền và Kaggle Secret `HF_TOKEN`. Builder từ chối nhúng token vào notebook. Môi trường COMET: `unbabel-comet==2.2.7`, `protobuf==5.29.5`, Torch/CUDA phù hợp; tách khỏi môi trường test và tránh TensorFlow/Flax khi dùng scorer PyTorch.

`compare-1688 select` và adapter Tiki giữ **strict policy lịch sử**. Chính sách no-floor từng phát hành nằm ở archive/release_history; không đổi strict selector một cách âm thầm. Việc publish HF là thao tác riêng cần chọn đúng policy và revision.

## Judge và ghép

```powershell
python manage.py prepare-judge
python manage.py judge --help
python manage.py merge-judged --help
python manage.py prepare-ft --help
python manage.py merge-official --help
```

Judge sources là snapshot thực chạy có adapter Alibaba. Đặt `DASHSCOPE_API_KEY` và `ALIBABA_BASE_URL` trong môi trường nếu dùng provider đó; không có giá trị khóa hoặc endpoint workspace cá nhân trong repo. Các đường mặc định đọc private snapshots ở `data/snapshot` và `data/processed`; cần staging đúng đầu vào trước. Quy trình lịch sử/định nghĩa rubric đọc tại `docs/history/judge/`, không coi ghi chú giá/quota cũ là thông tin hiện hành.

Không chấm lại mục đã thành công; không đổi rubric/config của checkpoint giữa chừng, không né provider block, không gán blocked thành thành công. File đánh dấu và provenance phải được giữ khi rút gọn bảng chính.

`notebooks/templates/judge_full_resume_with_progress.ipynb` lưu bản orchestration batch đã dùng: resume, checkpoint, kiểm soát ngân sách và tiến độ. Notebook đã bỏ outputs và thay giá trị tài khoản bằng Kaggle Secrets; counts, tên input và ngân sách là của snapshot lịch sử, phải đối chiếu trước khi tái chạy. Không tự tạo lượt judge mới từ template này.

## Audit hai file cuối chỉ đọc

```powershell
python manage.py audit-release --cum data/inputs/bo_cum_chon_deepseek.parquet --pair data/inputs/bo_pair_chon_deepseek.parquet --revision INPUT_COMMIT --out outputs/audit_new_revision
```

Đầu ra: `audit_report.json` và `review_signals.parquet`; không chép title/description vào signal mặc định, không sửa file nguồn, không upload. Dùng thư mục đầu ra mới. Exit code 1 khi schema/khóa bắt buộc hỏng hoặc hai bộ không khớp thứ tự sản phẩm/title. Các signal về chữ còn sót, số hoặc ký tự lặp cần ngữ cảnh; không tự xóa dòng theo signal.

Audit này không thay thế rà số liệu theo từng thuộc tính hoặc đánh giá ngữ nghĩa. Các scripts rà thuộc tính/phát hành lịch sử ở archive chỉ chạy lại khi khôi phục đúng input snapshot/checkpoint, không import như thư viện.
