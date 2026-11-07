#!/bin/bash
# Upload hypothesis A/B lên Kaggle rồi chạy 2 kernel COMET offline.
set -euo pipefail
ROOT=/content/eda_ecom
STAGE=/content/kaggle_hyps

rm -rf "$STAGE"; mkdir -p "$STAGE"
# Đổi tên để A/B không ghi đè nhau và notebook lọc theo _A_/_B_.
for s in a b; do
  S=$(echo "$s" | tr 'a-z' 'A-Z')
  for f in "$ROOT"/.artifacts/weight_ab_$s/hyp_*.parquet; do
    [ -e "$f" ] || continue
    cp "$f" "$STAGE/hyp_${S}_$(basename "$f" | sed 's/^hyp_//')"
  done
  for f in "$ROOT"/.artifacts/weight_ab_$s/run_*.json; do
    [ -e "$f" ] || continue
    cp "$f" "$STAGE/run_${S}_$(basename "$f" | sed 's/^run_//')"
  done
done
ls -lh "$STAGE"

cat > "$STAGE/dataset-metadata.json" <<'EOF'
{
  "title": "Tiki Weight AB Hyps",
  "id": "nadsa091/tiki-weight-ab-hyps",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

cd "$ROOT"
if kaggle datasets files nadsa091/tiki-weight-ab-hyps >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "A/B quota survey hypotheses" -r skip
else
  kaggle datasets create -p "$STAGE" -r skip
fi

sleep 45
kaggle kernels push -p kaggle/comet_ab_a
kaggle kernels push -p kaggle/comet_ab_b
