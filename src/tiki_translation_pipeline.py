# -*- coding: utf-8 -*-
"""Pipeline dịch và kiểm định tiêu đề Tiếng Việt sang Tiếng Trung cho bộ dữ liệu Tiki.

Kiến trúc Dual-Model:
1. Google AI Studio API: gemma-4-26b-a4b-it
2. Remote Colab GPU / Local OpenAI-compatible API: Qwen/Qwen3.5-9B

Bảo toàn 100% mã kỹ thuật, số liệu, đơn vị và kiểm định QA theo tiêu chuẩn sàn TMĐT.
"""

from __future__ import annotations

# Repository layout bootstrap; translation policy is unchanged.
import sys as _repo_sys
from pathlib import Path as _RepoPath
_repo_root = _RepoPath(__file__).resolve().parents[1]
if (_repo_root / "project_bootstrap.py").is_file():
    _repo_sys.path.insert(0, str(_repo_root))
    from project_bootstrap import configure_paths as _configure_paths
    _configure_paths()

import re
import os
import sys
import json
import sqlite3
import hashlib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Generator
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT_DIR / "data" / "tiki_vi_cleaned.parquet"

# ==================== CẤU HÌNH PROMPTS VIỆT -> TRUNG ====================
GEMMA_SYSTEM_PROMPT_VI_ZH = (
    "Translate one Vietnamese e-commerce product title into natural, concise Simplified Chinese (zh-CN) suitable for e-commerce platforms (Taobao/1688/Tmall). "
    "Return ONLY the Chinese title. No explanations, no markdown formatting, no quotation marks. "
    "Preserve all brand names, English model codes (e.g., LED, UV, WRGB, UFO), alphanumeric identifiers, and numbers. "
    "Preserve exact numbers, dimensions, and standard units (e.g., W, V, ml, g, kg, cm, mm, h, mAh). "
    "Any ZXQCODETOKEN marker is a protected product-code placeholder: copy it exactly once, unchanged, in the matching position. "
    "Do not add promotional buzzwords (e.g., 包邮, 秒杀, 正品保证) unless explicitly present in the source title."
)

QWEN_PROMPT_TEMPLATE_VI_ZH = (
    "Translate this Vietnamese e-commerce product title into natural, concise Simplified Chinese (zh-CN) suitable for Taobao/1688. "
    "Return only the Chinese title. Preserve brand names, alphanumeric model codes, numbers, and units. Do not add explanations.\n\n{title}"
)

# Danh sách từ rác tiếp thị cấm kỵ tiếng Trung
BANNED_ZH_MARKETING_SPAM = [
    "包邮", "秒杀", "正品保证", "假一赔十", "亏本甩卖", "买一送一",
    "爆款推荐", "掌柜推荐", "全网最低", "顶级好物"
]

# Regex nhận diện ký tự tiếng Việt có dấu
VI_ACCENTED_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE
)

# Regex nhận diện chữ Hán CJK
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Regex số và đơn vị đo
NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
UNIT_RE = re.compile(r"\b(?:w|v|kw|ml|l|g|kg|cm|mm|m|inch|h|mah|gb|tb)\b", re.IGNORECASE)

# Regex mã hiệu sản phẩm (chữ in hoa kèm số hoặc ký tự phân cách)
MODEL_CODE_RE = re.compile(r"\b[A-Z0-9]{2,}(?:[-_/][A-Z0-9]+)+\b|\b[A-Z]+[0-9]+[A-Z0-9]*\b")


def protect_model_codes(title: str) -> tuple[str, list[tuple[str, str]]]:
    """Bảo vệ các mã hiệu model kỹ thuật bằng token an toàn để tránh LLM dịch méo mó."""
    matches = list(MODEL_CODE_RE.finditer(title))
    protected = title
    replacements: list[tuple[str, str]] = []
    for index, match in reversed(list(enumerate(matches))):
        val = match.group(0)
        # Bỏ qua các từ thông dụng viết hoa ngắn
        if val in {"LED", "UV", "RGB", "USB", "WRGB", "UFO", "NO", "PRO", "MAX", "MINI"}:
            continue
        marker = f"ZXQCODETOKEN{chr(65 + index % 26)}ZXQ"
        replacements.append((marker, val))
        protected = protected[: match.start()] + marker + protected[match.end() :]
    return protected, list(reversed(replacements))


def restore_model_codes(text: str, replacements: list[tuple[str, str]]) -> str:
    """Khôi phục lại mã hiệu model sau khi dịch."""
    restored = text
    for marker, original in replacements:
        restored = restored.replace(marker, original)
    return restored


# ==================== BỘ KIỂM ĐỊNH CHẤT LƯỢNG QA (VIỆT -> TRUNG) ====================
def qa_check_vi_to_zh(vi: str, zh: str) -> dict[str, Any]:
    """Kiểm tra tính hợp lệ và các lỗi kỹ thuật trong bản dịch từ Tiếng Việt sang Tiếng Trung.

    Kế thừa và điều chỉnh hệ thống luật kiểm định từ Báo cáo Đề án Pipeline:
    - rong: Bản dịch trống hoặc null
    - sot_tieng_viet: Còn ký tự có dấu tiếng Việt lọt lưới
    - ti_le_chu_han_thap: Ký tự CJK không chiếm ưu thế
    - bi_cat_cut: Độ dài tiếng Trung quá ngắn so với tiếng Việt (< 0.20)
    - ao_giac_lap_tu: Độ dài tiếng Trung phình to bất thường (> 1.40)
    - mat_so_lieu: Làm rơi rớt số nguyên / số thập phân gốc
    - jin_sai_spam: Nhồi nhét từ khóa quảng cáo cấm
    """
    vi = (vi or "").strip()
    zh = (zh or "").strip()
    flags: list[str] = []

    if not zh:
        return {"passed": False, "flags": ["rong"], "ratio": 0.0, "cjk_ratio": 0.0}

    # 1. Kiểm tra sót tiếng Việt có dấu
    vi_accent_match = VI_ACCENTED_RE.search(zh)
    if vi_accent_match:
        flags.append("sot_tieng_viet")

    # 2. Tỷ lệ ký tự chữ Hán (CJK)
    cjk_chars = len(CJK_RE.findall(zh))
    total_chars = max(1, len(zh.replace(" ", "")))
    cjk_ratio = cjk_chars / total_chars
    if cjk_chars == 0 or cjk_ratio < 0.25:
        # Nếu tiêu đề toàn tên tiếng Anh / mã số kỹ thuật thì cjk_ratio có thể thấp hơn một chút,
        # nhưng nếu 0 chữ Hán nào thì chắc chắn chưa dịch
        flags.append("ti_le_chu_han_thap")

    # 3. Tỷ lệ độ dài (Length Ratio)
    # Tiếng Trung 1 chữ = 1 từ (2-5 ký tự tiếng Việt), nên zh_len / vi_len thường từ 0.20 - 1.25
    ratio = len(zh) / max(1, len(vi))
    if ratio < 0.18:
        flags.append("bi_cat_cut")
    elif ratio > 1.40:
        flags.append("ao_giac_lap_tu")

    # 4. Bảo toàn số liệu (Numbers Preservation)
    nums_vi = set(NUM_RE.findall(vi))
    # Chuẩn hóa dấu phẩy trong số (ví dụ 1,6 -> 1.6)
    nums_vi_norm = {n.replace(",", ".") for n in nums_vi}
    zh_text_norm = zh.replace(",", ".")
    missing_nums = [n for n in nums_vi_norm if n not in zh_text_norm]
    if missing_nums:
        flags.append(f"mat_so_lieu:{','.join(missing_nums[:3])}")

    # 5. Lọc rác tiếp thị tiếng Trung
    zh_lower = zh.lower()
    for spam_word in BANNED_ZH_MARKETING_SPAM:
        if spam_word in zh_lower:
            flags.append(f"jin_sai_spam:{spam_word}")

    passed = len([f for f in flags if not f.startswith("warning:")]) == 0
    return {
        "passed": passed,
        "flags": flags,
        "ratio": round(ratio, 3),
        "cjk_ratio": round(cjk_ratio, 3),
    }


# ==================== SQLITE RESILIENT CACHE ====================
class SQLiteTranslationCache:
    """Quản lý bộ nhớ đệm SQLite bền bỉ tại Local, ghi nhận ngay lập tức sau mỗi bản dịch."""

    def __init__(self, db_path: Path, model_key: str):
        self.db_path = Path(db_path)
        self.model_key = model_key
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS translations (
                    title_vi TEXT PRIMARY KEY,
                    title_zh TEXT,
                    model_key TEXT,
                    status TEXT,
                    qa_passed INTEGER,
                    qa_flags TEXT,
                    cjk_ratio REAL,
                    length_ratio REAL,
                    elapsed_seconds REAL,
                    updated_at TEXT
                )"""
            )
            conn.commit()

    def get(self, title_vi: str) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT title_zh, status, qa_passed, qa_flags, cjk_ratio, length_ratio FROM translations WHERE title_vi = ?",
                (title_vi,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "title_zh": row[0],
                    "status": row[1],
                    "qa_passed": bool(row[2]),
                    "qa_flags": json.loads(row[3]) if row[3] else [],
                    "cjk_ratio": row[4],
                    "length_ratio": row[5],
                }
            return None

    def put(
        self,
        title_vi: str,
        title_zh: str,
        status: str,
        qa_result: dict[str, Any],
        elapsed: float = 0.0,
    ) -> None:
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """INSERT INTO translations (
                    title_vi, title_zh, model_key, status, qa_passed, qa_flags, cjk_ratio, length_ratio, elapsed_seconds, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(title_vi) DO UPDATE SET
                    title_zh=excluded.title_zh,
                    status=excluded.status,
                    qa_passed=excluded.qa_passed,
                    qa_flags=excluded.qa_flags,
                    cjk_ratio=excluded.cjk_ratio,
                    length_ratio=excluded.length_ratio,
                    elapsed_seconds=excluded.elapsed_seconds,
                    updated_at=excluded.updated_at""",
                (
                    title_vi,
                    title_zh,
                    self.model_key,
                    status,
                    int(qa_result.get("passed", False)),
                    json.dumps(qa_result.get("flags", [])),
                    qa_result.get("cjk_ratio", 0.0),
                    qa_result.get("ratio", 0.0),
                    elapsed,
                    now_utc,
                ),
            )
            conn.commit()

    def count_completed(self) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM translations WHERE status = 'ok'")
            return cursor.fetchone()[0]


# ==================== DATASET LOADER ====================
def load_tiki_unique_titles(parquet_path: Path = DEFAULT_DATASET) -> list[str]:
    """Tải và loại bỏ trùng lặp danh sách tiêu đề sạch của Tiki."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file parquet Tiki tại: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    col = "title_vi_cleaned" if "title_vi_cleaned" in df.columns else "title_vi"
    unique_titles = df[col].dropna().drop_duplicates().tolist()
    return unique_titles


# ==================== GEMMA CLIENT (GOOGLE AI STUDIO API) ====================
class GemmaAPIClient:
    """Gọi Gemma 4-26B-A4B-it qua Google AI Studio Gemini API."""

    def __init__(self, api_key: str | None = None, model_id: str = "gemma-4-26b-a4b-it", timeout_ms: int = 120000):
        from google import genai
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("Thiếu GEMINI_API_KEY. Vui lòng cấu hình key Google AI Studio.")
        self.model_id = model_id
        types = genai.types
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=timeout_ms, retry_options=types.HttpRetryOptions(attempts=2)),
        )

    def translate_single(self, title_vi: str) -> str:
        return self.translate_record(title_vi)['title_zh']

    def translate_record(self, title_vi: str) -> dict[str, Any]:
        from google.genai import types
        protected_title, replacements = protect_model_codes(title_vi)
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=protected_title,
            config=types.GenerateContentConfig(
                system_instruction=GEMMA_SYSTEM_PROMPT_VI_ZH,
                temperature=0.0,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        text = str(getattr(response, "text", "") or "").strip()
        # Loại bỏ ngoặc kép bao quanh nếu có
        text = text.strip("\"'「」『』")
        text = restore_model_codes(text, replacements)
        candidates = getattr(response, 'candidates', None) or []
        reason = getattr(candidates[0], 'finish_reason', None) if candidates else None
        reason = str(getattr(reason, 'value', reason) or '')
        truncated = reason.upper().endswith('MAX_TOKENS')
        usage = getattr(response, 'usage_metadata', None)
        return {
            'title_zh': text,
            'status': 'truncated' if truncated else ('ok' if text else 'empty'),
            'finish_reason': reason,
            'input_tokens': int(getattr(usage, 'prompt_token_count', 0) or 0),
            'output_tokens': int(getattr(usage, 'candidates_token_count', 0) or 0),
        }


# ==================== QWEN CLIENT (COLAB REMOTE API / LOCAL) ====================
class QwenRemoteClient:
    """Gọi Qwen3.5-9B đang chạy trên Colab qua Cloudflare Tunnel / OpenAI-compatible endpoint."""

    def __init__(self, endpoint_url: str, model_name: str = "Qwen/Qwen3.5-9B", timeout: float = 30.0):
        import requests
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()

    def translate_single(self, title_vi: str) -> str:
        # Hỗ trợ cả 2 chuẩn endpoint: /translate và /v1/chat/completions
        url_custom = f"{self.endpoint_url}/translate"
        url_openai = f"{self.endpoint_url}/v1/chat/completions"

        prompt = QWEN_PROMPT_TEMPLATE_VI_ZH.format(title=title_vi)

        try:
            # Thử gọi endpoint gọn nhẹ trước
            resp = self.session.post(
                url_custom,
                json={"title": title_vi},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("translation", "").strip("\"' ")
        except Exception:
            pass

        # Fallback sang chuẩn OpenAI-compatible
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a professional e-commerce translator. Translate Vietnamese titles to concise Simplified Chinese."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 256
        }
        resp = self.session.post(url_openai, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text.strip("\"'「」『』 ")
