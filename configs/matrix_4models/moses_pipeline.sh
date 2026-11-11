#!/bin/bash
# Moses SMT pipeline — chạy trong Kaggle CPU (không cần GPU)
set -e
DATASET=A            # A hoặc B
BUDGET=4000000

# 1. Sample quota 4M token (cùng quota_sampling.py)
python tools/moses_prepare.py --dataset "$DATASET" --budget "$BUDGET"

# 2. Tokenize
#    vi: tách punctuation; zh: tách punctuation + mỗi chữ Hán
#    (xem tools/moses_prepare.py)

# 3. Alignment: eflomal
python - <<'PY'
from eflomal import Aligner
al = Aligner(model=3, n_iterations=(4, 4, 4), n_samplers=4)
with open('train.vi') as s, open('train.zh') as t:
    al.align(s, t, links_filename_fwd='fwd', links_filename_rev='rev', quiet=True)
PY
# sym = grow-diag-final-and(fwd, rev)

# 4. Language model 3-gram
zcat -f train.zh | lmplz -o 3 > lm.arpa
gzip -f lm.arpa

# 5. Train phrase-based model
train-model.perl \
  -root-dir work \
  -corpus train -f vi -e zh \
  -alignment grow-diag-final-and -alignment-file sym \
  -do-steps 4-9 \
  -lm 0:3:$PWD/lm.arpa.gz \
  -max-phrase-length 7 \
  -reordering msd-bidirectional-fe

# 6. Decode eval_F
moses -f work/model/moses.ini -threads 4 < eval.vi > pred.zh
