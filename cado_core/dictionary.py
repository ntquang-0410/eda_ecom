# -*- coding: utf-8 -*-
"""Whitelist + CC-CEDICT + lemmatization.

Sửa lỗi quan trọng so với bản cũ: CC-CEDICT được index theo **headword giản
thể** (cột thứ 2), không phải phồn thể. Dữ liệu 1688 là giản thể, nếu dùng
phồn thể thì rất nhiều bản dịch không bao giờ khớp (VD ``书包`` vs ``書包``).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import CEDICT_FILE, CJK_CHAR_RE, WHITELIST_CSV

_CEDICT_ENTRY_RE = re.compile(r"^(\S+)\s+(\S+)\s+\[(.+?)\]\s+/(.+)/$")


# ═══════════════════════════════════════════════════════════════════
# WHITELIST
# ═══════════════════════════════════════════════════════════════════

class Whitelist:
    """Tập từ không bao giờ bị xoá (brand, tech term, đơn vị, từ mượn)."""

    def __init__(self, rows: list[dict[str, str]]):
        self.rows = tuple(rows)
        self.words = frozenset(
            row["word"].strip().lower()
            for row in rows
            if row.get("word", "").strip()
        )

    @classmethod
    def load(cls, path: str | Path = WHITELIST_CSV) -> "Whitelist":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy whitelist: {path}. "
                "Không nên chạy khi thiếu whitelist vì brand/model sẽ bị xoá nhầm."
            )
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = [row for row in csv.DictReader(f) if row.get("word", "").strip()]
        return cls(rows)

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and word.lower() in self.words

    def __len__(self) -> int:
        return len(self.words)

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            key = row.get("category", "unknown") or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts


# ═══════════════════════════════════════════════════════════════════
# LEMMATIZATION (heuristic, không cần spaCy)
# ═══════════════════════════════════════════════════════════════════

def lemmatize(word: str) -> list[str]:
    """Sinh các dạng lemma tiếng Anh thường gặp trong title TMĐT.

    Chỉ là heuristic: số nhiều, đuôi ``-ing``/``-ed``, phụ âm đôi.
    Luôn giữ nguyên dạng gốc ở vị trí đầu tiên.
    """
    w = word.lower().strip()
    if not w:
        return []
    forms: list[str] = [w]

    # ── Số nhiều ──
    if len(w) > 4 and w.endswith("sses"):
        forms.append(w[:-2])                     # dresses → dress
    elif len(w) > 4 and w.endswith("ies"):
        forms.append(w[:-3] + "y")               # ladies → lady
    elif len(w) > 3 and w.endswith("ves"):
        forms.extend([w[:-3] + "f", w[:-3] + "fe"])  # knives → knife
    elif len(w) > 4 and w.endswith("es"):
        base = w[:-2]
        if base.endswith(("sh", "ch", "x", "z", "s", "o")):
            forms.append(base)                   # boxes → box, shoes → sho
        forms.extend([w[:-1], base])             # shoes → shoe (ưu tiên)
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        forms.append(w[:-1])                     # bags → bag

    # ── Động từ ──
    if len(w) > 5 and w.endswith("ing"):
        stem = w[:-3]
        forms.extend([stem, stem + "e"])         # running → runn/runne
        if len(stem) >= 3 and stem[-1] == stem[-2]:
            forms.append(stem[:-1])              # running → run
    if len(w) > 4 and w.endswith("ed"):
        stem = w[:-2]
        forms.extend([stem, w[:-1]])             # stopped → stopp/stopped
        if len(stem) >= 3 and stem[-1] == stem[-2]:
            forms.append(stem[:-1])              # stopped → stop

    # Bỏ trùng, giữ thứ tự
    return list(dict.fromkeys(forms))


# ═══════════════════════════════════════════════════════════════════
# CEDICT
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Overlap:
    """Kết quả tra overlap giữa một từ Anh và văn bản Trung."""

    status: str          # "overlap" | "no_overlap" | "no_entry"
    match: str = ""

    @property
    def overlap(self) -> bool:
        return self.status == "overlap"


class Cedict:
    """Từ điển CC-CEDICT offline, index English → headword giản thể."""

    def __init__(
        self,
        eng_to_zh: dict[str, tuple[str, ...]],
        zh_to_eng: dict[str, frozenset[str]] | None = None,
        *,
        entries: int = 0,
        english_keys: int = 0,
    ):
        self.eng_to_zh = eng_to_zh
        self.zh_to_eng = zh_to_eng or {}
        self.entries = entries
        self.english_keys = english_keys

    # ── Load ────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        path: str | Path = CEDICT_FILE,
        *,
        reverse: bool = False,
        include_traditional: bool = False,
        use_cache: bool = True,
    ) -> "Cedict":
        if use_cache:
            return _load_cached(str(Path(path)), reverse, include_traditional)
        return _parse_cedict(Path(path), reverse, include_traditional)

    # ── Query ───────────────────────────────────────────────

    def translations(self, word: str) -> tuple[str, ...]:
        """Bản dịch của dạng lemma đầu tiên tìm thấy trong từ điển."""
        for lemma in lemmatize(word):
            if lemma in self.eng_to_zh:
                return self.eng_to_zh[lemma]
        return ()

    def overlap(self, word: str, text: str) -> Overlap:
        """Kiểm tra bản dịch tiếng Trung của ``word`` có nằm trong ``text``.

        Chỉ xét headword có ≥2 ký tự Hán (đã lọc lúc parse) để tránh
        false positive với ký tự đơn như ``包``.
        """
        translations: list[str] = []
        found_entry = False
        for lemma in lemmatize(word):
            trans = self.eng_to_zh.get(lemma)
            if not trans:
                continue
            found_entry = True
            translations.extend(trans)

        if not found_entry:
            return Overlap("no_entry")

        # Pass 1: exact substring (ưu tiên bản dịch dài).
        for trans in translations:
            if trans in text:
                return Overlap("overlap", trans)

        # Pass 2: fuzzy cho headword có khoảng trắng (hiếm).
        for trans in translations:
            if " " in trans:
                parts = [p for p in trans.split() if p]
                if parts and all(p in text for p in parts):
                    return Overlap("overlap", f"fuzzy:{trans}")

        return Overlap("no_overlap")

    def bigram_english(self, bigram: str) -> frozenset[str]:
        """Các từ tiếng Anh có headword đúng bằng bigram (cho signal S4)."""
        return self.zh_to_eng.get(bigram, frozenset())


# ═══════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════

def _parse_cedict(
    path: Path,
    reverse: bool,
    include_traditional: bool,
) -> Cedict:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy CC-CEDICT: {path}. "
            "Tải tại https://www.mdbg.net/chinese/export/cedict/"
            "cedict_1_0_ts_utf-8_mdbg.zip rồi đặt vào dictionary/."
        )

    eng_sets: dict[str, set[str]] = {}
    zh_to_eng: dict[str, set[str]] = {}
    entries = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = _CEDICT_ENTRY_RE.match(line)
            if not match:
                continue

            traditional, simplified, _pinyin, gloss = match.groups()
            headwords = [simplified]
            if include_traditional and traditional != simplified:
                headwords.append(traditional)

            gloss_words = {
                token
                for token in re.findall(r"[a-z]+", gloss.lower())
                if len(token) >= 2
            }
            if not gloss_words:
                continue

            entries += 1
            for head in headwords:
                if len(CJK_CHAR_RE.findall(head)) < 2:
                    continue
                for token in gloss_words:
                    eng_sets.setdefault(token, set()).add(head)
                if reverse and len(CJK_CHAR_RE.findall(head)) == 2:
                    zh_to_eng.setdefault(head, set()).update(gloss_words)

    eng_to_zh = {
        key: tuple(sorted(values, key=lambda s: (-len(s), s)))
        for key, values in eng_sets.items()
    }
    return Cedict(
        eng_to_zh,
        {k: frozenset(v) for k, v in zh_to_eng.items()},
        entries=entries,
        english_keys=len(eng_to_zh),
    )


@lru_cache(maxsize=8)
def _load_cached(path: str, reverse: bool, include_traditional: bool) -> Cedict:
    return _parse_cedict(Path(path), reverse, include_traditional)
