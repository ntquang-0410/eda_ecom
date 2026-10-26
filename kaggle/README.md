# Kaggle setup — Plan B 4-model Back-Translation

Thư mục này chuẩn bị connection, dataset package và config cho 4 Qwen3 model.

## 1. Credentials

Kaggle CLI 2.0.2 hiện dùng **access token**, không cần file `kaggle.json`:

```text
~/.kaggle/access_token
```

Kiểm tra:

```bash
kaggle config view
```

Kỳ vọng:

```text
- username: ntquang0410
- auth_method: ACCESS_TOKEN
```

Nếu dùng environment:

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_access_token"
```

Không commit token vào repo.

## 2. Kiểm tra connection

```bash
python kaggle/check_connection.py
```

Script chỉ in username và số dataset, không in key.

## 3. Đóng gói Plan A dataset

```bash
export KAGGLE_USERNAME="your_username"
python kaggle/prepare_assets.py \
  --owner "$KAGGLE_USERNAME" \
  --dataset-slug tiki-plan-a-bt-input \
  --staging-dir /tmp/opencode/kaggle_plan_a_flat
```

Output staging là **flat** để Kaggle CLI upload đúng:

```text
/tmp/opencode/kaggle_plan_a_flat/
  dataset-metadata.json
  manifest.json
  tiki_bt_input.parquet
  tiki_translation_cache.parquet
  ...
  tiki_key_map.csv
  qwen3_0_6b.json
  ...
  code.zip
  requirements.txt
```

Upload lần đầu nên dùng bản **flat** (không có thư mục con):

```bash
python kaggle/prepare_assets.py \
  --dataset-slug tiki-plan-a-bt-input \
  --staging-dir /tmp/opencode/kaggle_plan_a_flat \
  --force

kaggle datasets create -p /tmp/opencode/kaggle_plan_a_flat -r skip -t
```

Các lần sau:

```bash
kaggle datasets version \
  -p /tmp/opencode/kaggle_plan_a_flat \
  -m "Plan A update" -r skip -t
```

`-t` giữ parquet, không convert sang CSV. `-r skip` vì staging đã flat.

## 4. Model configs

```text
kaggle/configs/qwen3_0_6b.json
kaggle/configs/qwen3_1_7b.json
kaggle/configs/qwen3_4b.json
kaggle/configs/qwen3_8b.json
kaggle/configs/plan_b_models.json
```

4 dataset sẽ dùng cùng input Plan A, chỉ khác `model_id` và output directory.

## 5. Kaggle kernel

Template:

```text
kaggle/kernel/plan_b_env_check.py
kaggle/kernel/kernel-metadata.json.template
```

Sinh metadata theo owner:

```bash
python kaggle/prepare_kernel_metadata.py \
  --owner "$KAGGLE_USERNAME" \
  --dataset-slug tiki-plan-a-bt-input \
  --kernel-slug tiki-plan-b-env-check \
  --output kaggle/kernel/kernel-metadata.json
```

Push env-check:

```bash
cd kaggle/kernel
kaggle kernels push -p .
kaggle kernels status "$KAGGLE_USERNAME/tiki-plan-b-env-check"
```

Kernel env-check chỉ kiểm tra:

- dataset mount;
- shape parquet;
- cache route counts;
- TPU/GPU availability;
- không load model.

Sau khi pass env-check mới chạy BT thật.

Theo dõi queue/session:

```bash
python kaggle/watch_kaggle.py \
  --kernel "$KAGGLE_USERNAME/tiki-plan-b-env-check" \
  --interval 900
```

## 6. Queue strategy

- Không push 4 full job cùng lúc.
- Chạy theo thứ tự: `0.6B → 1.7B → 4B → 8B`.
- 8B cần compatibility spike trước (TPU sharding hoặc GPU quantized).
- Output/checkpoint phải ghi ra Kaggle Dataset, không tin `/kaggle/working`.
- Watcher chỉ nên giữ một job active.

## 7. Trạng thái input hiện tại

```text
120.884 unit
29.446 MT cache entry
11.200 non-MT cache
649/698 key mapped
49 key pending
13.140 product ready
60 product blocked key map
```

## 8. Lưu ý

- Không upload API key.
- Không upload output chưa QC vào dataset chính.
- Mỗi model có output riêng:
  ```text
  qwen3-0.6b/
  qwen3-1.7b/
  qwen3-4b/
  qwen3-8b/
  ```
- Không ghi đè dataset Plan A; tạo version mới.
