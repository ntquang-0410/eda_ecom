#!/usr/bin/env bash
# Queue thực nghiệm trên Colab A100 — chạy tuần tự, sống sót qua SSH disconnect.
# Mỗi job: train 500 step (4M token) + sinh hypothesis, có fallback micro-batch.
set -u
cd /content/eda_ecom || exit 1
DATA=.artifacts/matrix_out/m4b_a/matrix_data
LOGD=/content/logs
mkdir -p "$LOGD"

export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/content/hf

python - <<'PY'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B"):
    for attempt in range(3):
        try:
            p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
            print("ok", m, p, flush=True)
            break
        except Exception as e:
            print("retry", m, type(e).__name__, str(e)[:120], flush=True)
    else:
        raise SystemExit("khong tai duoc " + m)
PY

run_job () { # $1=name $2=model $3=dataset $4=config $5=seed
  local name="$1" model="$2" ds="$3" cfg="$4" seed="$5"
  local out="/content/runs/$name"
  local log="$LOGD/$name.log"
  : > "$log"
  echo "[$(date -u '+%F %T')] START $name ($model, $ds/$cfg, seed $seed)" | tee -a "$log"
  for MB in 4 2 1; do
    echo "[$(date -u '+%T')] thu micro-batch $MB" >> "$log"
    rm -rf "$out"
    python tools/train_weight_quota_gpu.py \
      --dataset "$ds" --data-root "$DATA" --out "$out" \
      --model "$model" --seed "$seed" \
      --tokens 4000000 --steps 500 --configs "$cfg" \
      --micro-batch "$MB" --dtype bf16 --max-seq 960 --save-every 100 >> "$log" 2>&1
    rc=$?
    echo "[$(date -u '+%T')] rc=$rc (mb=$MB)" >> "$log"
    if [ "$rc" -eq 0 ]; then
      echo "[$(date -u '+%F %T')] DONE $name" | tee -a "$log"
      return 0
    fi
  done
  echo "[$(date -u '+%F %T')] FAILED $name" | tee -a "$log"
  return 1
}

# 1-2: Qwen3-8B 2x2 (chạy lại từ step 0 — adapter step100 cũ chỉ còn trên Kaggle)
run_job qwen8b_a_balanced Qwen/Qwen3-8B A balanced 42
run_job qwen8b_b_natural  Qwen/Qwen3-8B B natural  42

# 3-4: seed thứ 3 cho Qwen3-4B (A-natural + B-balanced), seed 7
run_job qwen4b_a_seed7 Qwen/Qwen3-4B A natural  7
run_job qwen4b_b_seed7 Qwen/Qwen3-4B B balanced 7

echo "[$(date -u '+%F %T')] QUEUE DONE (EcomGPT-7B cần Kaggle token — chưa chạy)" | tee -a "$LOGD/queue.done"
