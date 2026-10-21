"""CADO core — thư viện dùng chung cho pipeline Trung–Việt.

Các module:
    config      Đường dẫn, hằng số, regex dùng chung
    text        Chuẩn hoá Unicode, trích/xoá từ Latin an toàn với CJK
    dictionary  Whitelist + CC-CEDICT + lemmatization
    rules       Pattern detection, auto loanword, phân loại keep/remove
    cado        Pipeline làm sạch tiếng Anh trong văn bản Trung (T0–T5)
    cado_vi     Pipeline làm sạch tiếng Anh trong văn bản Việt (CADO-VI)
    bitext      Preprocess + chấm điểm cặp song ngữ (cosine + lexical)
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
