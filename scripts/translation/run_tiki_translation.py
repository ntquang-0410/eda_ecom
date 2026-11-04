# -*- coding: utf-8 -*-
"""Script điều khiển chạy dịch tiêu đề Tiki (Việt -> Trung) trực tiếp từ Agent hoặc Terminal.

Hỗ trợ 2 mô hình:
- gemma4: Gọi Google AI Studio API cục bộ (cần GEMINI_API_KEY).
- qwen35: Gọi Qwen3.5-9B trên Google Colab qua Cloudflare Tunnel URL (--endpoint).

Các chế độ chạy:
- smoke: Dịch thử nghiệm 10 mẫu để kiểm tra kết nối & QA.
- pilot: Dịch 50 mẫu đại diện đánh giá chất lượng.
- full: Dịch toàn bộ 4.131 tiêu đề Tiki duy nhất (hỗ trợ ngắt/resume tự động).
- status: Xem tiến độ hiện tại của cả 2 mô hình.
- export: Xuất file Parquet song ngữ tích hợp QA và đánh giá.
"""

from __future__ import annotations

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[2]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()

import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


from tiki_translation_pipeline import (
    DEFAULT_DATASET,
    load_tiki_unique_titles,
    qa_check_vi_to_zh,
    SQLiteTranslationCache,
    GemmaAPIClient,
    QwenRemoteClient,
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CACHE_GEMMA = DATA_DIR / "cache_tiki_gemma4.sqlite3"
CACHE_QWEN = DATA_DIR / "cache_tiki_qwen35.sqlite3"
OUTPUT_PARQUET = DATA_DIR / "tiki_bilingual_zh_vi_title_translated.parquet"


def print_status() -> None:
    unique_titles = load_tiki_unique_titles(DEFAULT_DATASET)
    total = len(unique_titles)

    cache_g = SQLiteTranslationCache(CACHE_GEMMA, "gemma4")
    cache_q = SQLiteTranslationCache(CACHE_QWEN, "qwen35")

    done_g = cache_g.count_completed()
    done_q = cache_q.count_completed()

    print("=" * 65)
    print(f"BÁO CÁO TIẾN ĐỘ DỊCH TIÊU ĐỀ TIKI (VIỆT -> TRUNG)")
    print(f"Tổng số tiêu đề duy nhất: {total:,} mẫu (từ 4.157 sản phẩm)")
    print("-" * 65)
    print(f"1. Gemma 4-26B (Google AI Studio API): {done_g:,}/{total:,} ({done_g/total*100:.1f}%)")
    print(f"   Cache SQLite: {CACHE_GEMMA}")
    print(f"2. Qwen3.5-9B (Colab GPU / Local):    {done_q:,}/{total:,} ({done_q/total*100:.1f}%)")
    print(f"   Cache SQLite: {CACHE_QWEN}")
    print("=" * 65)


def run_translation(
    model_key: str,
    mode: str,
    colab_endpoint: str | None = None,
    gemini_key: str | None = None,
    delay_seconds: float | None = None,
) -> None:
    unique_titles = load_tiki_unique_titles(DEFAULT_DATASET)
    total_unique = len(unique_titles)

    # Chọn cache và client phù hợp
    if model_key == "gemma4":
        cache = SQLiteTranslationCache(CACHE_GEMMA, "gemma4")
        client = GemmaAPIClient(api_key=gemini_key)
        delay = 4.0 if delay_seconds is None else delay_seconds
        print(f"[gemma4] Đã nạp Google AI Studio Client (Model: gemma-4-26b-a4b-it). Nhịp nghỉ: {delay}s")
    elif model_key == "qwen35":
        if not colab_endpoint:
            raise ValueError("Cần cung cấp --endpoint khi chạy Qwen3.5 (ví dụ: --endpoint https://xxxx.trycloudflare.com)")
        cache = SQLiteTranslationCache(CACHE_QWEN, "qwen35")
        client = QwenRemoteClient(endpoint_url=colab_endpoint)
        delay = 0.5 if delay_seconds is None else delay_seconds
        print(f"[qwen35] Đã kết nối Colab Server: {colab_endpoint}. Nhịp nghỉ: {delay}s")
    else:
        raise ValueError(f"Model không hợp lệ: {model_key}. Chọn 'gemma4' hoặc 'qwen35'.")

    # Xác định danh sách cần chạy theo mode
    if mode == "smoke":
        limit = 10
        target_titles = unique_titles[:limit]
    elif mode == "pilot":
        limit = 50
        target_titles = unique_titles[:limit]
    elif mode == "full":
        target_titles = unique_titles
    else:
        raise ValueError(f"Mode không hợp lệ: {mode}")

    print(f"\n--- BẮT ĐẦU CHẠY [{model_key.upper()}] - CHẾ ĐỘ: {mode.upper()} ({len(target_titles)} tiêu đề) ---")

    completed_count = 0
    passed_qa_count = 0
    start_time = time.time()

    for idx, title_vi in enumerate(target_titles, 1):
        # Kiểm tra cache
        cached = cache.get(title_vi)
        if cached and cached.get("status") == "ok":
            completed_count += 1
            if cached.get("qa_passed"):
                passed_qa_count += 1
            continue

        t0 = time.time()
        try:
            title_zh = client.translate_single(title_vi)
            status = "ok" if title_zh else "empty"
        except Exception as exc:
            title_zh = ""
            status = "error"
            print(f"\n[LỖI] Dòng {idx}: {exc}")

        elapsed = time.time() - t0
        qa_result = qa_check_vi_to_zh(title_vi, title_zh)
        cache.put(title_vi, title_zh, status, qa_result, elapsed=elapsed)

        completed_count += 1
        if qa_result.get("passed"):
            passed_qa_count += 1

        flag_str = f" [Flags: {','.join(qa_result['flags'])}]" if qa_result["flags"] else " [QA: PASS]"
        print(
            f"[{completed_count}/{len(target_titles)}] ({elapsed:.1f}s) "
            f"VI: {title_vi[:35]}... -> ZH: {title_zh[:25]}...{flag_str}"
        )

        if delay > 0:
            time.sleep(delay)

    total_time = time.time() - start_time
    pass_rate = (passed_qa_count / max(1, completed_count)) * 100
    print("\n" + "=" * 65)
    print(f"HOÀN TẤT ĐỢT CHẠY [{model_key.upper()} - {mode.upper()}]")
    print(f"Thời gian: {total_time:.1f}s | Đã xử lý: {completed_count}/{len(target_titles)}")
    print(f"Tỷ lệ QA đạt chuẩn: {passed_qa_count}/{completed_count} ({pass_rate:.1f}%)")
    print("=" * 65)


def export_bilingual_parquet() -> None:
    """Gộp bản dịch của 2 model vào file Parquet gốc Tiki."""
    if not DEFAULT_DATASET.exists():
        raise FileNotFoundError(f"Không tìm thấy: {DEFAULT_DATASET}")

    df = pd.read_parquet(DEFAULT_DATASET)
    col = "title_vi_cleaned" if "title_vi_cleaned" in df.columns else "title_vi"

    cache_g = SQLiteTranslationCache(CACHE_GEMMA, "gemma4")
    cache_q = SQLiteTranslationCache(CACHE_QWEN, "qwen35")

    print(f"Đang tổng hợp dữ liệu cho {len(df):,} sản phẩm...")

    zh_gemma = []
    qa_gemma_pass = []
    qa_gemma_flags = []

    zh_qwen = []
    qa_qwen_pass = []
    qa_qwen_flags = []

    for title_vi in df[col]:
        # Gemma
        cg = cache_g.get(str(title_vi))
        if cg and cg.get("status") == "ok":
            zh_gemma.append(cg.get("title_zh", ""))
            qa_gemma_pass.append(cg.get("qa_passed", False))
            qa_gemma_flags.append(",".join(cg.get("qa_flags", [])))
        else:
            zh_gemma.append("")
            qa_gemma_pass.append(False)
            qa_gemma_flags.append("not_translated")

        # Qwen
        cq = cache_q.get(str(title_vi))
        if cq and cq.get("status") == "ok":
            zh_qwen.append(cq.get("title_zh", ""))
            qa_qwen_pass.append(cq.get("qa_passed", False))
            qa_qwen_flags.append(",".join(cq.get("qa_flags", [])))
        else:
            zh_qwen.append("")
            qa_qwen_pass.append(False)
            qa_qwen_flags.append("not_translated")

    df["title_zh_gemma4"] = zh_gemma
    df["title_zh_gemma4_qa_passed"] = qa_gemma_pass
    df["title_zh_gemma4_qa_flags"] = qa_gemma_flags

    df["title_zh_qwen35"] = zh_qwen
    df["title_zh_qwen35_qa_passed"] = qa_qwen_pass
    df["title_zh_qwen35_qa_flags"] = qa_qwen_flags

    # Cập nhật cột title_zh chính bằng bản dịch ưu tiên (nếu cả 2 đạt QA thì lấy Gemma hoặc Qwen)
    consensus_zh = []
    for zg, zq, pg, pq in zip(zh_gemma, zh_qwen, qa_gemma_pass, qa_qwen_pass):
        if pg:
            consensus_zh.append(zg)
        elif pq:
            consensus_zh.append(zq)
        else:
            consensus_zh.append(zg or zq or "")
    df["title_zh"] = consensus_zh

    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Đã xuất bản tệp Parquet song ngữ Tiki thành công:")
    print(f"-> {OUTPUT_PARQUET} ({OUTPUT_PARQUET.stat().st_size / (1024*1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Runner dịch tiêu đề Tiki Việt -> Trung (Gemma4 & Qwen3.5)")
    parser.add_argument("--model", choices=["gemma4", "qwen35"], help="Chọn mô hình (gemma4 hoặc qwen35)")
    parser.add_argument("--mode", choices=["smoke", "pilot", "full", "status", "export"], default="status", help="Chế độ chạy")
    parser.add_argument("--endpoint", type=str, help="Cloudflare Tunnel URL của Colab (dành cho Qwen3.5)")
    parser.add_argument("--gemini-key", type=str, help="Google AI Studio Gemini API Key (nếu chưa set biến môi trường)")
    parser.add_argument("--delay", type=float, help="Thời gian nghỉ giữa các request (giây)")

    args = parser.parse_args()

    if args.mode == "status":
        print_status()
    elif args.mode == "export":
        export_bilingual_parquet()
    else:
        if not args.model:
            print("Vui lòng chỉ định --model gemma4 hoặc --model qwen35.")
            sys.exit(1)
        run_translation(
            model_key=args.model,
            mode=args.mode,
            colab_endpoint=args.endpoint,
            gemini_key=args.gemini_key,
            delay_seconds=args.delay,
        )


if __name__ == "__main__":
    main()
