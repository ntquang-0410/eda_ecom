#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh notebook Stage A offline cho account nadsa091 / RTX Pro 6000.

Giữ nguyên competition ARC trong metadata; thêm dataset riêng + model Qwen3-4B.
Không dùng internet trong notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kaggle" / "rtx_pro_6000_offline"
OUT.mkdir(parents=True, exist_ok=True)


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


cells = [
    md(
        "# Stage A — Qwen3-4B weight experiment (offline)\n\n"
        "- 5 cấu hình × seed 42\n"
        "- 5M token/run, 625 optimizer step\n"
        "- chỉ train + generate; COMET/Gemma chấm sau ở môi trường có internet\n"
        "- giữ nguyên dataset/competition có sẵn của notebook\n"),
    code(
        "import glob, json, os, subprocess, sys, time, zipfile\n"
        "from pathlib import Path\n"
        "\n"
        "import torch\n"
        "print('python', sys.version.split()[0])\n"
        "print('torch', torch.__version__, '| cuda', torch.cuda.is_available())\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('Không có GPU; dừng để tiết kiệm quota')\n"
        "p = torch.cuda.get_device_properties(0)\n"
        "vram = p.total_memory / 1e9\n"
        "print('gpu', p.name, round(vram, 1), 'GB | capability', torch.cuda.get_device_capability(0))\n"
        "if vram < 40:\n"
        "    raise SystemExit(f'GPU {p.name} chỉ {vram:.1f} GB, không phải RTX Pro 6000; dừng trước khi train')\n"
    ),
    code(
        "# Cài wheel offline (không internet), chỉ 6 package; bỏ qua nếu lỗi.\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],\n"
        "               capture_output=True)\n"
        "WHEELS = Path('/kaggle/input/tiki-offline-wheels')\n"
        "if WHEELS.exists():\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',\n"
        "                    '--find-links', str(WHEELS), '--no-deps',\n"
        "                    'transformers==5.22.0', 'peft==0.21.2',\n"
        "                    'accelerate==1.15.0', 'huggingface_hub==1.33.0',\n"
        "                    'tokenizers==0.23.3', 'safetensors==0.8.0'], check=False)\n"
        "import transformers, peft, accelerate\n"
        "print('transformers', transformers.__version__, '| peft', peft.__version__,\n"
        "      '| accelerate', accelerate.__version__)\n"
    ),
    code(
        "# Tìm dataset (Kaggle có thể mount folder hoặc file zip).\n"
        "DATA_ROOT = Path('/kaggle/input/tiki-4model-eval')\n"
        "if not (DATA_ROOT / 'data/weight_eval/eval_1688.parquet').exists():\n"
        "    zips = sorted(DATA_ROOT.glob('*.zip'))\n"
        "    if zips:\n"
        "        dest = Path('/kaggle/working/tiki-4model-eval')\n"
        "        dest.mkdir(parents=True, exist_ok=True)\n"
        "        with zipfile.ZipFile(zips[0]) as zf:\n"
        "            zf.extractall(dest)\n"
        "        DATA_ROOT = dest\n"
        "print('dataset', DATA_ROOT)\n"
        "assert (DATA_ROOT / 'data/weight_eval/eval_1688.parquet').exists()\n"
        "print('eval rows', len(__import__('pandas').read_parquet(DATA_ROOT/'data/weight_eval/eval_1688.parquet')))\n"
    ),
    code(
        "# Tìm Qwen3-4B đã attach làm model source (không tải mạng).\n"
        "MODEL = None\n"
        "for cfg_path in glob.glob('/kaggle/input/**/config.json', recursive=True):\n"
        "    try:\n"
        "        c = json.loads(Path(cfg_path).read_text(encoding='utf-8'))\n"
        "    except Exception:\n"
        "        continue\n"
        "    if (c.get('hidden_size'), c.get('intermediate_size'),\n"
        "            c.get('num_hidden_layers')) == (2560, 9728, 36):\n"
        "        MODEL = Path(cfg_path).parent\n"
        "        break\n"
        "print('model', MODEL)\n"
        "if MODEL is None:\n"
        "    raise SystemExit('Không thấy Qwen3-4B trong /kaggle/input; kiểm tra model_sources')\n"
        "print(sorted(x.name for x in MODEL.glob('*'))[:20])\n"
    ),
    code(
        "# Smoke 2 step: chỉ kiểm tra load + backward + generate, không đụng eval set.\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "from peft import LoraConfig, get_peft_model\n"
        "tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)\n"
        "if tok.pad_token is None: tok.pad_token = tok.eos_token\n"
        "try:\n"
        "    base = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.bfloat16,\n"
        "                                                trust_remote_code=True, low_cpu_mem_usage=True)\n"
        "except TypeError:\n"
        "    base = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16,\n"
        "                                                trust_remote_code=True, low_cpu_mem_usage=True)\n"
        "base.gradient_checkpointing_enable(); base.enable_input_require_grads()\n"
        "smoke = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',\n"
        "                                        task_type='CAUSAL_LM', target_modules='all-linear')).to('cuda')\n"
        "smoke.train()\n"
        "batch = tok(['Áo thun nam cotton', 'Bình nước 2 lít'], return_tensors='pt', padding=True).to('cuda')\n"
        "batch['labels'] = batch['input_ids'].clone()\n"
        "opt = torch.optim.AdamW([q for q in smoke.parameters() if q.requires_grad], lr=1e-4)\n"
        "for i in range(2):\n"
        "    loss = smoke(**batch).loss\n"
        "    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)\n"
        "    print('smoke step', i, 'loss', round(float(loss.item()), 4))\n"
        "with torch.inference_mode():\n"
        "    gen = smoke.generate(batch['input_ids'][:1], max_new_tokens=4, do_sample=False)\n"
        "print('smoke generate ok', gen.shape)\n"
        "del smoke, base, opt; torch.cuda.empty_cache()\n"
        "print('SMOKE OK')\n"
    ),
    md("## Stage A: 5 cấu hình × seed 42\n"),
    code(
        "OUT = Path('/kaggle/working/weight_runs')\n"
        "OUT.mkdir(parents=True, exist_ok=True)\n"
        "CONFIGS = ['natural', 'balanced', 'no_desc', 'tiki_heavy', 'title_heavy']\n"
        "MB = 8 if vram >= 60 else (4 if vram >= 30 else 2)\n"
        "env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',\n"
        "           TOKENIZERS_PARALLELISM='false')\n"
        "started = time.time()\n"
        "for cfg in CONFIGS:\n"
        "    hyp = OUT / f'hyp_{cfg}_seed42.parquet'\n"
        "    if hyp.exists():\n"
        "        print('skip existing', hyp, flush=True); continue\n"
        "    cmd = [sys.executable, str(DATA_ROOT / 'code/train_weight_gpu.py'),\n"
        "           '--model', str(MODEL),\n"
        "           '--train-view', str(DATA_ROOT / 'data/weight_eval/train_view_4b.parquet'),\n"
        "           '--eval-dir', str(DATA_ROOT / 'data/weight_eval'),\n"
        "           '--configs', cfg, '--seeds', '42',\n"
        "           '--tokens', '5000000', '--steps', '625',\n"
        "           '--out', str(OUT), '--dtype', 'bf16', '--micro-batch', str(MB)]\n"
        "    print('RUN', cfg, 'micro-batch', MB, flush=True)\n"
        "    subprocess.run(cmd, check=True, env=env)\n"
        "    if (time.time()-started)/60 >= 210:\n"
        "        print('TIME GUARD: dừng ở 210 phút để không bị timeout', flush=True)\n"
        "        break\n"
        "print('STAGE A DONE in', round((time.time()-started)/60, 1), 'min')\n"
    ),
    code(
        "for p in sorted(OUT.glob('hyp_*.parquet')):\n"
        "    print(p.name, p.stat().st_size)\n"
        "print('--- adapters ---')\n"
        "for p in sorted(OUT.glob('adapter_*')):\n"
        "    print(p.name)\n"
        "(OUT / 'stage_a_done.json').write_text(json.dumps({'configs': CONFIGS, 'seed': 42,\n"
        "    'vram_gb': round(vram, 1), 'gpu': p if False else '', 'finished_at': time.time()}), encoding='utf-8')\n"
    ),
]

# fix last cell accidentally referencing p; keep simple
cells[-1]["source"] = [
    "for p in sorted(OUT.glob('hyp_*.parquet')):\n",
    "    print(p.name, p.stat().st_size)\n",
    "print('--- adapters ---')\n",
    "for p in sorted(OUT.glob('adapter_*')):\n",
    "    print(p.name)\n",
    "(OUT / 'stage_a_done.json').write_text(json.dumps({'configs': CONFIGS, 'seed': 42,\n",
    "    'vram_gb': round(vram, 1), 'gpu': torch.cuda.get_device_name(0),\n",
    "    'finished_at': time.time()}), encoding='utf-8')\n",
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(OUT / "rtx-pro-6000.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

metadata = {
    "id": "nadsa091/rtx-pro-6000",
    "title": "RTX Pro 6000",
    "code_file": "rtx-pro-6000.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": ["nadsa091/tiki-4model-eval", "nadsa091/tiki-offline-wheels"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [],
    "model_sources": ["qwen-lm/qwen-3/Transformers/4b/1"],
    "machine_shape": "NvidiaRtxPro6000",
}
(OUT / "kernel-metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
