# Repair các dòng BT chưa dịch (per-model)

## Vì sao

Vòng BT chính (`plan_b_rtx_v3`) sinh đủ 29.446 dòng cho mỗi model, nhưng khi kiểm
tra target có chữ Hán thì phát hiện model nhỏ thường **copy nguyên tiếng Việt**:

| model | target có Hán | chưa dịch | tỉ lệ |
|---|---|---|---|
| 0.6B | 10.430 | 13.576 | 46,1% |
| 1.7B | 16.121 | 8.569 | 29,1% |
| 4B | 26.200 | 234 | 0,8% |
| 8B | 26.706 | 279 | 0,9% |

Ví dụ 0.6B:

```text
"1 Bát Tô"  ->  "1 Bát Tô"     (đáng lẽ 1个大碗)
"Số 4: 5L"  ->  "4: 5L"        (dịch thiếu "Số")
"Hổ"        ->  "Hổ"           (đáng lẽ 虎)
```

Nguyên nhân: prompt cũ đưa `key_vi`/`key_zh` vào payload nhưng không nói rõ
`source_text` là **value** phải dịch, nên model nhỏ dịch nhầm key hoặc bỏ qua.

## Cách repair

Giữ nguyên bản sắc từng model: **mỗi model tự dịch lại đúng các dòng của nó**,
không lấy kết quả của model khác.

Gate QC (áp dụng giống nhau cho cả 4 model):

```text
cần repair  =  json_ok = False
            OR target rỗng
            OR (target không có chữ Hán AND còn dấu tiếng Việt)
```

Ngoại lệ hợp lệ: brand/model Latin (`iPhone 13`, `Type C`) không có dấu tiếng Việt
nên không bị đưa vào repair.

## Prompt mới (`plan_b_repair_v4`)

Thêm 3 ràng buộc so với v3:

1. `source_text` là **VALUE** phải dịch; key chỉ là ngữ cảnh, không được dịch key.
2. Bắt buộc chữ Hán giản thể, cấm copy tiếng Việt (trừ brand/model code).
3. Dịch cả danh từ đơn (`Hổ` → `虎`).

Kernel tự probe **40 dòng/model** với 2 biến thể (`strict` và `strict + few-shot`)
rồi chọn biến thể có tỉ lệ đúng cao hơn, trước khi chạy toàn bộ.

## Khối lượng

```text
0.6B : 13.738 dòng
1.7B :  8.717 dòng
4B   :    253 dòng
8B   :    292 dòng
tổng : 23.000 dòng
```

Input: dataset `ntquang0410/tiki-bt-repair-input` (2MB).
Output: `repair_outputs/<model>/repair.parquet` + `run_state.json`.

## Cách chạy

```bash
cd kaggle/bt_repair
python build_notebook.py          # sinh plan_b_bt_repair.ipynb
kaggle kernels push -p .
kaggle kernels status ntquang0410/tiki-plan-b-bt-repair
kaggle kernels output ntquang0410/tiki-plan-b-bt-repair -p /tmp/repair_out
```

## Sau khi repair

Merge `repair.parquet` trở lại `predictions.parquet` theo `cache_key`, đánh dấu:

```text
repair_pass = 1
prompt_version = plan_b_repair_v4
qc_status = repaired | still_untranslated
```

Sau đó mới cắt/merge vào bộ 1688 để fine-tune.
