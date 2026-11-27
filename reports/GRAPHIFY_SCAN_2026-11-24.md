# Graphify — Quét toàn bộ source code `eda_ecom`

**Ngày:** 2026-11-24 · **Repo:** `/content/eda_ecom` · **Công cụ:** graphify 0.9.118 (Graphify-Labs)
**Graph build từ commit:** `c463e69` (HEAD hiện tại `736b22d` chỉ thêm `.graphifyignore`, không đổi mã nguồn)
**Artifact:** `graphify-out/graph.json`, `graphify-out/graph.html`, `graphify-out/GRAPH_REPORT.md`, `graphify-out/merged-graph.json`

---

## 1. Cài đặt

```bash
git clone --depth 1 https://github.com/Graphify-Labs/graphify.git /content/graphify
uv tool install --force --from '/content/graphify[mcp]' graphifyy
graphify --version   # graphify 0.9.118
```

`[mcp]` extra là bắt buộc: `graphify query/explain/path` import `anyio` + `starlette`; thiếu extra sẽ crash `ModuleNotFoundError: anyio`.

Quét không dùng LLM/API key: `graphify extract . --code-only` (AST tree-sitter, deterministic, 0 token).

---

## 2. Phạm vi quét

| Hạng mục | Số lượng |
|---|---|
| File code được index | **236** (148 `.py`, 68 `.json`, 9 `.yaml`, 11 `.sh`) |
| LOC quét được | **34.567** |
| Node / edge | **1.790 / 4.372** (98% EXTRACTED, 2% INFERRED) |
| Community phát hiện | **113** |
| File `.py` tracked được index | **148/201 = 100%** phần mã nguồn "hiện tại" |
| File `.py` bị loại có chủ đích | 53 (31 trong `.artifacts/`, 22 trong `archive/`) |
| Notebook mirror | 54 ipynb → 230 node / 740 edge (graph riêng, merge vào `merged-graph.json`: **1.998 node / 5.112 edge**) |

### Vì sao phải lọc `.artifacts/` và `archive/`

Lần quét đầu (không lọc) cho **3.035 node**, trong đó **1.035 node (34%)** là bản sao mã nguồn nằm trong `.artifacts/plan_a/code/cado_core/...` và `.artifacts/*/code/...`. Hệ quả:

- centrality bị chia đôi cho cùng một hàm (ví dụ `normalize_text()` xuất hiện 2 lần: bản thật degree 55, bản copy degree 37);
- top "god nodes" bị nhiễu bởi node trùng tên.

Đã thêm `.graphifyignore` (`.artifacts/`, `archive/`, `reports_v3/`, `reports_v5/`, `data/`, `csv/`, `eda/`, `dictionary/`, `hf_dataset/`, `graphify-out/`) và build lại. Số liệu trong báo cáo này là bản sạch.

---

## 3. Kiến trúc rút ra từ graph

### 3.1 Kích thước theo tầng

| Thư mục | File | LOC | Node |
|---|---|---|---|
| `tools/` | 69 | 10.324 | 435 |
| `cado_core/` | 16 | 8.521 | 366 |
| `kaggle/` | 99 | 7.331 | 198 |
| `tiki_mining/` | 11 | 2.676 | 66 |
| `tests/` | 9 | 1.460 | 108 |
| `run_pipeline.py` | 1 | 1.261 | 31 |
| `configs/` | 13 | 699 | 382 |
| `cado_layer/` | 6 | 454 | 19 |
| `bitext_mining/` | 2 | 114 | 6 |

### 3.2 God nodes (hub kiến trúc)

| Hàm / class | Degree | File |
|---|---|---|
| `normalize_text()` | 55 | `cado_core/text.py` |
| `resolve_path()` | 44 | `cado_core/config.py` |
| `extract_human_value_pairs()` | 34 | `cado_core/length_prior_v2.py` |
| `run_cado()` | 31 | `cado_core/cado.py` |
| `run_tiki_eda()` | 30 | `cado_core/tiki.py` |
| `build_tiki_units()` | 28 | `cado_core/tiki_units.py` |
| `normalize_key()` | 27 | `cado_core/tiki_units.py` |
| `BitextConfig` | 25 | `cado_core/bitext.py` |

**Nhận xét:** toàn bộ hệ thống dồn vào 4 module nền của `cado_core`: `text.py` (fan-in 102), `config.py` (68), `quantity_norm.py` (54), `dictionary.py` (52). Đây là blast radius lớn nhất — mọi thay đổi ở 4 file này lan ra toàn repo.

### 3.3 Luồng phụ thuộc giữa các tầng

```
run_pipeline.py  ──71 cạnh──> cado_core
tests/           ──155 cạnh──> cado_core
cado_layer/      ──46 cạnh──> cado_core
tiki_mining/     ──105 cạnh──> cado_core        ← tầng cũ vẫn phụ thuộc tầng mới
tools/           ──53 cạnh──> cado_core
```

- **0 vòng lặp import** ở mức file (SCC toàn bộ là singleton) → kiến trúc phân tầng sạch, không có circular import.
- `cado_core` là tầng duy nhất không phụ thuộc ngược lên trên.

### 3.4 Community tiêu biểu (graphify tự phát hiện, chưa đặt tên bằng LLM)

| C | Node | Thành phần chính | Ý nghĩa suy ra |
|---|---|---|---|
| C0 | 92 | `run_pipeline.py`, `length_prior.py`, tests | Orchestrator + length-prior v1 |
| C1 | 73 | `bitext.py`, `dictionary.py` | Lõi bitext & từ điển |
| C2 | 71 | `train_weight_t4.py`, `tpu_weight_train.py`, `run_queue.sh` | Launcher train T4/TPU/GPU |
| C3 | 68 | `gemma_judge*.py`, `watch_milestones.py` | Judge + giám sát milestone |
| C4 | 67 | `quota_sampling.py`, `aggregate_weight_*.py` | Sampling quota + tổng hợp điểm |
| C5 | 64 | `verify_review.py`, `prepare_assets.py` | Hash/verify/asset prep |
| C6 | 59 | `bt_eda_clean.py`, `eda_weight_splits.py` | EDA & chia split |
| C9 | 40 | `dedup_chunk_pairs.py`, `process_bt_desc.py` | Xử lý cặp & dedup |
| C7/C8 | 40+40 | `configs/matrix_4models|5models/ecomgpt_7b_lora.yaml` | Cấu hình EcomGPT (bản trùng) |
| C10 | 39 | `kaggle/comet_*/build.py`, `matrix_m4b/build.py` | Sinh kernel Kaggle |
| C11 | 37 | `tiki_bt_prepare.py` (core + legacy) | Chuẩn bị BT Tiki |
| C12–C15 | 37×4 | `opennmt_A/B.yaml` (2 bản × 2 thư mục) | Cấu hình OpenNMT (bản trùng) |
| C17 | 35 | `tiki_units.py`, `tiki.py` | Unit/key mapping Tiki |

---

## 4. Phát hiện

### F1 — `configs/matrix_4models` và `configs/matrix_5models` là bản sao

5 file **trùng byte-for-byte**: `ecomgpt_7b_lora.yaml`, `moses_pipeline.sh`, `opennmt_A.yaml`, `opennmt_B.yaml`, `qwen3_8b_lora.yaml`; chỉ `measured_runs.json` khác và `matrix_5models` thêm `qwen3_4b_lora.yaml`.
Ngoài ra **không file code nào đọc `configs/`** (grep 0 kết quả trong `.py`, `.sh`, `.ipynb`) → đây là tài liệu cấu hình tham chiếu, không phải config được nạp. Rủi ro: sửa 1 bản tưởng là đã đổi hành vi, nhưng thực tế không ai đọc; và 2 bản dễ lệch nhau.

### F2 — `tiki_mining/` là thế hệ cũ song song với `cado_core/`

- `tiki_mining/tiki_bt_prepare.py` 168 dòng vs `cado_core/tiki_bt_prepare.py` 815 dòng (cùng chức năng).
- `tiki_mining/` còn `tiki_length_prior{,2,3}.py`, `tiki_key_map_review.py`, `tiki_eda.py` — bản rút gọn của module tương ứng trong `cado_core/`.
- `tiki_mining` vẫn import `cado_core` (105 cạnh) nên không phải dead code hoàn toàn, nhưng là **nợ kỹ thuật**: 2 định nghĩa song song cho cùng khái niệm.

### F3 — Điểm mù lớn nhất: logic nằm trong string của notebook

- 82 notebook tracked; 54 ngoài `archive/` được mirror sang `.py` để quét.
- **25/54 notebook không sinh ra symbol nào**: chúng dùng `Path(...).write_text('...code...')` để sinh script rồi chạy bằng subprocess (`kaggle/comet_*/comet.py`, `kaggle/nllb_b_infer`, `rtx_pro_6000_*`…). AST không nhìn thấy code trong string.
- Hệ quả: pipeline chấm COMET, inference NLLB, các stage RTX đang **không có trong graph**. Muốn phủ, phải trích string ra file `.py` thật (hoặc chuyển các notebook này sang script).

### F4 — `tools/` quá lớn và trùng lặp chức năng

69 file / 10.324 LOC. Trùng lặp đã đo:

- `load_scores()` định nghĩa 4 lần (`aggregate_seed123`, `aggregate_weight_ab`, `aggregate_weight_eval`, `model_diagnostics`)
- `build_prompt()` 4 lần (`gemma_judge`, `gemma_judge_credit`, `train_weight_t4`, `tpu_weight_train`)
- `compact_len()` 6 lần, `sha256()` 4 lần, `log()` 7 lần
- 17 file tên `build.py` trong `kaggle/` (mỗi kernel 1 bản, gần như cùng khung)

Ứng viên: tách `tools/_common.py` (io, hash, prompt, log) và `kaggle/_kernel_template.py`.

### F5 — Test chỉ phủ `cado_core`

12/102 module nguồn được test tham chiếu trực tiếp; **toàn bộ 12 đều là `cado_core/`** (12/16 module). Chưa có test cho `tools/`, `tiki_mining/`, `cado_layer/`, `bitext_mining/`.
Trong `cado_core`, 4 module chưa được test: `bitext.py`, `cado_vi.py`, `config.py`, `__init__.py`.

### F6 — `run_pipeline.py` là monolith điều phối

1.261 dòng, fan-out 71 (cao nhất repo), fan-in 0. Mọi bước pipeline đi qua file này; là điểm nghẽn để test và để chạy từng phần.

### F7 — 93 file không có cạnh liên-file

Phần lớn **không phải dead code**: 13 file `configs/` (dữ liệu cấu hình), ~30 `kaggle/*/build.py` (mỗi kernel độc lập, chỉ chạy trên Kaggle), các `__init__.py`. Nhưng cũng có script một lần (`kaggle/check_connection.py`, `auto_tpu_pipeline.py`) không được tham chiếu — nên rà lại xem còn dùng.

### F8 — Không có isolated node

0 node degree 0 → không có symbol mồ côi hoàn toàn; mọi định nghĩa đều nối vào ít nhất 1 cạnh (chủ yếu qua `contains` của file).

---

## 5. Cách dùng graph

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /content/eda_ecom

# Hỏi đáp / duyệt
graphify query "làm sao length prior được fit và áp dụng?"
graphify explain "apply_length_prior_v2"
graphify path "run_cado" "build_tiki_units"
graphify god-nodes --top 15

# Blast radius trước khi sửa
graphify affected "normalize_text" --depth 3

# Cập nhật sau khi sửa code (AST-only, miễn phí)
graphify update .

# Xem
# mở graphify-out/graph.html trong browser (click node, filter, search)
```

Graph notebook (mirror): `graphify query "..." --graph graphify-out/merged-graph.json`.

---

## 6. Hạn chế của lần quét

| Hạn chế | Ảnh hưởng |
|---|---|
| Không đặt tên community bằng LLM (`--no-label`, không có API key) | Community là `Community N`; bảng ánh xạ ở §3.4 do tôi suy ra từ file thành viên |
| Không quét doc/PDF/image (dùng `--code-only`) | 67 tài liệu + 37 ảnh không vào graph; nội dung `reports/*.md` không được liên kết |
| Notebook không phải định dạng tree-sitter | 54 notebook mirror, 25 cái rỗng symbol (F3) |
| `configs/*.yaml|json` chỉ được parse cấu trúc | 382 node chủ yếu là key cấu hình, không phải logic |

---

## 7. Việc nên làm tiếp

1. **Chuyển logic string trong notebook thành script thật** (F3) — đây là điểm mù lớn nhất, và cũng là code đang chạy thật trên Kaggle.
2. Gộp `configs/matrix_4models` + `matrix_5models` thành 1 nguồn, hoặc đánh dấu rõ bản nào là canonical (F1).
3. Đánh dấu `tiki_mining/` là legacy trong README, hoặc xoá các module đã được `cado_core` thay thế (F2).
4. Tách `tools/_common.py` cho 4–7 hàm đang lặp (F4).
5. Cài hook `graphify hook install` để graph tự cập nhật sau mỗi commit.
