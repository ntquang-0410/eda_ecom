# Audit adapter provenance — 22 run canonical (4.4)

Điều kiện pass: `recorded=true` + path khớp token dataset của cell + có md5 (và md5 khớp lại nếu file còn).

## 1. Theo run

| Cell | Seed | Nguồn | dataset_ref | adapter | Path khớp | md5 | Pass | Lý do |
|---|---:|---|---|---|---:|---:|---|---|
| ecomgpt_a | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| ecomgpt_a_balanced | 42 | dataset | nguynthanhquang/ecomgpt-ckpt | adapter_balanced_seed42 | ✓ | ✓ | ✅ |  |
| ecomgpt_b | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| moses_a | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| moses_b | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| nllb_a | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| nllb_a | 7 | dataset | nguynthanhquang/tiki-matrix-checkpoints | adapters/nllb_a_s7_adapter | ✓ | ✓ | ✅ |  |
| nllb_b | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| nllb_b | 7 | dataset | nguynthanhquang/tiki-matrix-checkpoints | adapters/nllb_b_s7_adapter | ✓ | ✓ | ✅ |  |
| onmt_a | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| onmt_b | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| qwen4b_a | 123 | kernel_output | anhtamxinhdep/tiki-m4b-a-s123 | adapter_natural_seed123 | ✓ | ✓ | ✅ |  |
| qwen4b_a | 42 | kernel_output | nguynthanhquang/tiki-matrix-m4b-a | adapter_natural_seed42 | ✓ | ✓ | ✅ |  |
| qwen4b_a | 7 | dataset | nguynthanhquang/qwen4b-a-s7-ckpt | adapter_natural_seed7_ckpt | ✓ | ✓ | ✅ |  |
| qwen4b_a_balanced | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| qwen4b_b | 123 | kernel_output | anhtamxinhdep/tiki-m4b-b-s123 | adapter_balanced_seed123 | ✓ | ✓ | ✅ |  |
| qwen4b_b | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| qwen4b_b_natural | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| qwen8b_a | 42 | kernel_output | nguynthanhquang/tiki-matrix-m8b-a | adapter_natural_seed42 | ✓ | ✓ | ✅ |  |
| qwen8b_a_balanced | 42 | unknown |  |  | ✗ | ✗ | ❌ | chưa ghi provenance |
| qwen8b_b | 42 | kernel_output | nguynthanhquang/tiki-matrix-m8b-b | adapter_balanced_seed42 | ✓ | ✓ | ✅ |  |
| qwen8b_b_natural | 42 | dataset | nguynthanhquang/qwen8b-b-ckpt | adapter | ✓ | ✓ | ✅ |  |

## 2. Theo cell (pass = mọi seed đều pass)

| Cell | #seed | Pass | QE 858 hiển thị |
|---|---:|---|---|
| ecomgpt_a | 1 | ❌ | **ẩn** |
| ecomgpt_a_balanced | 1 | ✅ | có |
| ecomgpt_b | 1 | ❌ | **ẩn** |
| moses_a | 1 | ❌ | **ẩn** |
| moses_b | 1 | ❌ | **ẩn** |
| nllb_a | 2 | ❌ | **ẩn** |
| nllb_b | 2 | ❌ | **ẩn** |
| onmt_a | 1 | ❌ | **ẩn** |
| onmt_b | 1 | ❌ | **ẩn** |
| qwen4b_a | 3 | ✅ | có |
| qwen4b_a_balanced | 1 | ❌ | **ẩn** |
| qwen4b_b | 2 | ❌ | **ẩn** |
| qwen4b_b_natural | 1 | ❌ | **ẩn** |
| qwen8b_a | 1 | ✅ | có |
| qwen8b_a_balanced | 1 | ❌ | **ẩn** |
| qwen8b_b | 1 | ✅ | có |
| qwen8b_b_natural | 1 | ✅ | có |

**Tổng:** 10/22 run pass; 5/17 cell pass.

Cell fail bị **ẩn cột COMET-QE** trong `CANONICAL_EVAL.md` (QE là chẩn đoán); COMET-da và xếp hạng chính thức giữ nguyên.
