# -*- coding: utf-8 -*-
"""Plan A — chuẩn bị dữ liệu Tiki trước Back-Translation.

Không gọi MT. Module này:

* xây key map VI→ZH từ key pairs 1688 + key inventory Tiki;
* tạo BT input ở grain value/title (không dùng ``text_vi`` cho attribute);
* route ``text/title → MT``, ``admin → dictionary``, ``numeric → rule``,
  ``model → preserve``;
* tạo translation cache key và state machine ban đầu;
* đánh dấu unit/product đủ điều kiện merge sau BT/QC.

Artifact v1/v2/v3 không bị ghi đè.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .config import (
    TIKI_BT_INPUT_PARQUET,
    TIKI_BT_INPUT_SAMPLE_CSV,
    TIKI_BT_PLAN_A_STATS_JSON,
    TIKI_BT_ROUTING_CSV,
    TIKI_BT_UNITS_PLAN_A_PARQUET,
    TIKI_KEY_MAP_CSV,
    TIKI_KEY_MAP_REVIEW_CSV,
    TIKI_PRODUCTS_BT_STATE_PARQUET,
    TIKI_TRANSLATION_CACHE_PARQUET,
)
from .length_prior import measure_text
from .text import normalize_text
from .tiki import extract_protected_spans
from .tiki_units import classify_key, normalize_key

PLAN_A_VERSION = "plan_a_v1"
SOURCE_LANG = "vi"
TARGET_LANG = "zh"

# Seed cho các key Tiki phổ biến nhưng không khớp exact key ngắn của 1688.
# Các key còn lại vẫn vào review queue; không tự đoán key long-tail.
MANUAL_KEY_MAP: dict[str, str] = {
    "xuất xứ (made in)": "产地",
    "sản phẩm có được bảo hành không?": "是否保修",
    "xuất xứ thương hiệu": "品牌产地",
    "hạn sử dụng": "保质期",
    "hình thức bảo hành": "保修方式",
    "kích thước": "尺寸",
    "model": "型号",
    "trọng lượng sản phẩm": "产品重量",
    "hướng dẫn sử dụng": "使用说明",
    "hướng dẫn bảo quản": "保存说明",
    "dòng máy": "机型",
    "khối lượng (kg)": "重量(kg)",
    "tên đơn vị/tổ chức chịu trách nhiệm về hàng hóa": "责任单位名称",
    "địa chỉ tổ chức chịu trách nhiệm về hàng hóa": "责任单位地址",
    "kích thước (dài x rộng x cao)": "尺寸(长x宽x高)",
    "chọn màu": "选择颜色",
    "phương thức giao hàng seller delivery": "配送方式 Seller Delivery",
    "size": "尺寸",
    "chọn số": "选择号码",
    "dung tích": "容量",
    "đời xe": "车型年份",
    "lưu ý": "注意",
    "chọn số sim": "选择SIM号",
    "dòng iphone": "iPhone系列",
    "dòng xe": "车型",
    "khác": "其他",
    "số lượng": "数量",
    "trọng lượng": "重量",
    "tính năng": "功能",
    "chọn iphone": "选择iPhone",
    "iphone": "iPhone",
    "mẫu": "款式",
    "kết nối": "连接方式",
    "tải trọng tối đa": "最大承重",
    "mùi hương": "香味",
    "chọn mẫu": "选择款式",
    "phanh": "刹车",
    "chọn dòng máy": "选择机型",
    "chọn size": "选择尺寸",
    "variation": "选项",
    "color": "颜色",
    "chọn màu sắc": "选择颜色",
    "thời gian sạc đầy": "充满时间",
    "combo": "组合",
    "độ tuổi": "年龄",
    "dòng xe sử dụng": "适用车型",
    "loại tóc": "发质",
    "số km/mỗi lần sạc": "每次充电公里数",
    "loại chìa khóa": "钥匙类型",
    "năm ra mắt": "上市年份",
    "hỗ trợ sạc nhanh": "支持快充",
    "samsung": "三星",
    "chất liệu nồi & chảo": "锅具材质",
    "sử dụng cho bếp": "适用炉灶",
    "thời gian sử dụng": "使用时间",
    "bluetooth": "蓝牙",
    "chiều cao khuyến nghị": "建议身高",
    "loại nồi chảo": "锅具类型",
    "nắp nồi": "锅盖",
    "tốc độ tối đa": "最高速度",
    "hương vị": "口味",
    "thương hiệu thiết bị phù hợp": "适用设备品牌",
    "bộ": "套装",
    "có thuế vat": "含VAT",
    "màu da": "肤色",
    "đầu ra": "输出",
    "phụ kiện đi kèm": "附带配件",
    "độ phân giải quang học": "光学分辨率",
    "thời gian pin": "电池续航",
    "loại jack cắm": "接口类型",
    "bộ bài": "卡牌套装",
    "lựa chọn": "选项",
    "chọn kích thước ga": "选择床垫尺寸",
    "dòng máy / màu sắc": "机型/颜色",
    "chọn điện thoại": "选择手机",
    "hàm lượng dinh dưỡng": "营养成分",
    "loại pin": "电池类型",
    "tuổi thọ tối đa": "最长寿命",
    "mẫu tranh": "画款",
    "cổng usb": "USB接口",
    "hãng xe": "汽车品牌",
    "thời gian sạc": "充电时间",
    "độ tuổi phù hợp (dành cho thú cưng)": "适用年龄(宠物)",
    "tên máy": "机器名称",
    "dạng": "形态",
    "mầu sắc": "颜色",
    "đặc sản vùng miền": "地方特产",
    "chọn loại iphone": "选择iPhone型号",
    "phuộc": "前叉",
    "tần số": "频率",
    "dòng samsung": "三星系列",
    "họa tiết.": "图案",
    "logo": "LOGO",
    "tùy chọn": "选项",
    "khả năng giữ nhiệt": "保温能力",
    "mẫu pvc": "PVC款式",
    "độ nhạy": "灵敏度",
    "chọn logo": "选择LOGO",
    "tinh dầu": "精油",
    "màu rối": "发色",
    "chọn": "选择",
    "chọn khối lượng": "选择重量",
    "hỗ trợ lắp đặt": "支持安装",
    "chọn kích cỡ": "选择尺寸",
    "chọn loại": "选择类型",
    "dịch vụ nổi bật 2": "特色服务2",
    "dịch vụ nổi bật 3": "特色服务3",
    "giống nho": "葡萄品种",
    "lựa chọn ghế": "选择椅子",
    "thắng xe": "汽车刹车",
    "chọn vị": "选择口味",
    "chọn mùi hương": "选择香味",
    "hoạ tiết": "图案",
    "kich cỡ gạt mưa": "雨刷尺寸",
    "mã": "编号",
    "đối tượng": "适用对象",
    "phân loại hàng": "商品分类",
    "quà tặng": "礼品",
    "thành phẩm": "成品",
    "loại ipad": "iPad型号",
    "miếng lẻ": "单卖片",
    "màu son": "口红颜色",
    "chiều cao nệm": "床垫高度",
    "combo 10 miếng": "10片组合",
    "màu ánh sáng": "灯光颜色",
    "mệnh giá": "面值",
    "pl": "PL",
    "số miếng": "片数",
    "inbound policy (days)": "入境政策(天)",
    "chữ cái biến hình": "变形字母",
    "gia vị vipep": "Vipep调味料",
    "nhóm màu": "色系",
    "đầu vào": "输入",
    "hương": "香味",
    "hộp": "盒装",
    "trong full hd": "全高清",
    "độ dài": "长度",
    "chọn kiểu dáng và size": "选择款式和尺寸",
    "hệ điều hành": "操作系统",
    "hỗ trợ đàm thoại": "支持通话",
    "mùi vị": "口味",
    "nẹp gỗ composit": "复合木线条",
    "đường kính đầu chải": "刷头直径",
    "chiều dài dây phơi": "晾衣绳长度",
    "chọn kiểu dáng": "选择款式",
    "chọn mua": "选择购买",
    "chống chỉ định": "禁忌",
    "loại xe": "车型",
    "dạng bào chế": "剂型",
    "mầu": "颜色",
    "các vị": "口味",
    "dòng ipad": "iPad系列",
    "hương thơm": "香型",
    "lót ghế ô tô hạt gỗ": "汽车木珠坐垫",
    "mùi tinh dầu": "精油香味",
    "độ đạm": "蛋白质含量",
    "bao cao su": "避孕套",
    "cảm ứng": "触控",
    "dạng sản phẩm": "产品形态",
    "gạch màu": "彩色瓷砖",
    "độ chịu nước": "防水性",
    "chống nhìn trộm": "防窥",
    "kính cnt 360 độ": "CNT 360度镜片",
    "mẫu logo": "LOGO款式",
    "switch": "开关",
    "vạch báo đầy": "满电刻度",
    "chọn màu sơn": "选择油漆颜色",
    "cổng sạc": "充电接口",
    "dòng": "系列",
    "dòng macbook": "MacBook系列",
    "dòng máy - màu": "机型-颜色",
    "kích thước (ngang x sâu x cao)": "尺寸(宽x深x高)",
    "kích thước mặt": "表面尺寸",
    "số ngày sử dụng": "使用天数",
    "tông màu": "色调",
    "các màu sắc": "颜色选项",
    "gia vị": "调味料",
    "loại/ công nghệ màn hình": "屏幕类型/技术",
}


def _load_existing_manual(path: str | Path) -> dict[str, str]:
    """Đọc key map cũ nhưng chỉ giữ các dòng manual, tránh auto thành manual."""
    key_path = Path(path)
    if not key_path.exists():
        return {}
    frame = pd.read_csv(key_path, dtype=str, encoding="utf-8-sig").fillna("")
    if "key_zh" not in frame.columns:
        return {}
    if "key_map_status" in frame.columns:
        frame = frame[
            frame["key_map_status"].isin({
                "manual", "manual_seed", "mapped_existing",
                "qwen_high", "qwen_medium",
            })
        ]
    key_column = "key_norm" if "key_norm" in frame.columns else "key_vi"
    if key_column not in frame.columns:
        return {}
    return {
        normalize_key(row[key_column]): str(row["key_zh"]).strip()
        for _, row in frame.iterrows()
        if str(row["key_zh"]).strip()
    }


@dataclass
class KeyMapResult:
    table: pd.DataFrame
    review: pd.DataFrame
    mapping: dict[str, str]
    stats: dict[str, Any]


@dataclass
class PlanAResult:
    units: pd.DataFrame
    bt_input: pd.DataFrame
    cache: pd.DataFrame
    product_state: pd.DataFrame
    key_map: KeyMapResult
    stats: dict[str, Any]


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("key_zh_norm", "key_vi_norm"):
        if column in result.columns:
            result[column] = result[column].map(normalize_key)
    return result


def build_tiki_key_map(
    units: pd.DataFrame,
    *,
    alignment_map_path: str | Path,
    existing_map_path: str | Path = TIKI_KEY_MAP_CSV,
    min_key_freq: int = 20,
    min_key_share: float = 0.60,
) -> KeyMapResult:
    """Map ``key_norm`` của Tiki sang ``key_zh`` từ key pairs 1688.

    Ưu tiên map manual đã có. Nếu chưa có, lấy chiều ngược từ
    ``alignment_key_map.csv`` của Phase B; key long-tail để pending review.
    """
    if "key_norm" not in units.columns:
        raise KeyError("units cần có cột key_norm")
    alignment_path = Path(alignment_map_path)
    if not alignment_path.exists():
        raise FileNotFoundError(
            f"Thiếu alignment key map: {alignment_path}. "
            "Chạy tiki-length-prior-v3 trước."
        )

    alignment = pd.read_csv(
        alignment_path,
        dtype=str,
        encoding="utf-8-sig",
    ).fillna("")
    required = {"key_zh_norm", "key_vi_norm", "freq"}
    missing = sorted(required - set(alignment.columns))
    if missing:
        raise KeyError(f"alignment map thiếu cột: {missing}")
    alignment = _normalize_keys(alignment)
    alignment["freq"] = pd.to_numeric(alignment["freq"], errors="coerce").fillna(0)
    alignment = alignment[(alignment["key_zh_norm"] != "") & (alignment["key_vi_norm"] != "")]
    alignment = alignment.sort_values("freq", ascending=False)

    vi_total = alignment.groupby("key_vi_norm")["freq"].sum()
    vi_top = (
        alignment.drop_duplicates("key_vi_norm", keep="first")
        .set_index("key_vi_norm")[["key_zh_norm", "freq"]]
        .rename(columns={"key_zh_norm": "auto_key_zh", "freq": "auto_freq"})
    )
    vi_top["auto_share"] = (
        vi_top["auto_freq"] / vi_top.index.map(vi_total).astype(float)
    ).round(4)
    auto_map = {
        key_vi: row.auto_key_zh
        for key_vi, row in vi_top.iterrows()
        if row.auto_freq >= min_key_freq and row.auto_share >= min_key_share
    }

    existing = _load_existing_manual(existing_map_path)
    manual_seed = {
        normalize_key(key): value for key, value in MANUAL_KEY_MAP.items()
    }

    attribute_units = units[units["unit_type"] == "attribute"].copy()
    grouped = (
        attribute_units.groupby("key_norm", dropna=False)
        .agg(
            key_vi=("key_vi", "first"),
            key_class=("key_class", "first"),
            n_units=("pair_id", "size"),
            n_products=("product_id", "nunique"),
            n_train_eligible=("train_eligible", "sum"),
        )
        .reset_index()
    )
    grouped["key_norm"] = grouped["key_norm"].map(normalize_key)
    grouped = grouped[grouped["key_norm"] != ""]

    rows: list[dict[str, Any]] = []
    for row in grouped.to_dict("records"):
        key_norm = str(row["key_norm"])
        key_zh = ""
        status = "pending_review"
        source = "pending"
        if key_norm in existing and existing[key_norm]:
            key_zh = existing[key_norm]
            status = "manual"
            source = "existing_key_map"
        elif key_norm in manual_seed:
            key_zh = manual_seed[key_norm]
            status = "manual_seed"
            source = "manual_tiki_seed"
        elif key_norm in auto_map:
            key_zh = auto_map[key_norm]
            status = "auto_1688"
            source = "alignment_1688"
        rows.append({
            **row,
            "key_zh": key_zh,
            "key_map_status": status,
            "source": source,
            "auto_key_zh": vi_top.loc[key_norm, "auto_key_zh"] if key_norm in vi_top.index else "",
            "auto_freq": int(vi_top.loc[key_norm, "auto_freq"]) if key_norm in vi_top.index else 0,
            "auto_share": float(vi_top.loc[key_norm, "auto_share"]) if key_norm in vi_top.index else 0.0,
        })

    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=[
            "key_norm", "key_vi", "key_class", "n_units", "n_products",
            "n_train_eligible", "key_zh", "key_map_status", "source",
            "auto_key_zh", "auto_freq", "auto_share",
        ])
    table = table.sort_values(
        ["n_units", "key_norm"], ascending=[False, True]
    ).reset_index(drop=True)
    review = table[table["key_map_status"] == "pending_review"].copy()

    total_units = int(units[units["unit_type"] == "attribute"].shape[0])
    mapped_units = int(
        table.loc[table["key_zh"] != "", "n_units"].sum()
    ) if not table.empty else 0
    train_units = int(
        units.loc[
            (units["unit_type"] == "attribute") & units["train_eligible"],
            "pair_id",
        ].shape[0]
    )
    mapped_train_units = int(
        table.loc[
            (table["key_zh"] != "") & (table["n_train_eligible"] > 0),
            "n_train_eligible",
        ].sum()
    ) if not table.empty else 0
    stats = {
        "n_keys": int(len(table)),
        "n_mapped_keys": int((table["key_map_status"] != "pending_review").sum()) if not table.empty else 0,
        "n_manual_keys": int(
            table["key_map_status"].isin({"manual", "manual_seed"}).sum()
        ) if not table.empty else 0,
        "n_auto_keys": int((table["key_map_status"] == "auto_1688").sum()) if not table.empty else 0,
        "n_pending_keys": int(len(review)),
        "n_attribute_units": total_units,
        "n_mapped_attribute_units": mapped_units,
        "attribute_unit_coverage": round(mapped_units / total_units, 4) if total_units else 0.0,
        "n_train_attribute_units": train_units,
        "n_mapped_train_attribute_units": mapped_train_units,
        "train_attribute_unit_coverage": round(mapped_train_units / train_units, 4) if train_units else 0.0,
        "min_key_freq": int(min_key_freq),
        "min_key_share": float(min_key_share),
    }
    mapping = {
        str(row["key_norm"]): str(row["key_zh"])
        for row in table.to_dict("records")
        if row["key_zh"]
    }
    return KeyMapResult(
        table=table,
        review=review,
        mapping=mapping,
        stats=stats,
    )


def _route_for(unit_type: str, value_class: str) -> tuple[str, str]:
    if unit_type == "title":
        return "mt", ""
    if value_class == "text":
        return "mt", ""
    if value_class == "admin":
        return "dictionary", "mt"
    if value_class == "numeric":
        return "rule", ""
    if value_class == "model":
        return "preserve", ""
    return "mt", ""


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0, index=frame.index, dtype=int)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)


def _cache_key(
    source_text: str,
    key_norm: str,
    value_class: str,
    route: str,
) -> str:
    payload = "\x1f".join([
        normalize_text(source_text, collapse_whitespace=True).strip().lower(),
        normalize_key(key_norm),
        str(value_class or ""),
        str(route or ""),
        PLAN_A_VERSION,
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def prepare_bt_input(
    units: pd.DataFrame,
    products: pd.DataFrame,
    key_map: KeyMapResult,
    *,
    separator_chars: int = 2,
) -> PlanAResult:
    """Tạo unit-level BT input, cache và product state, chưa gọi MT."""
    required = {"pair_id", "product_id", "unit_type", "value_vi", "text_vi"}
    missing = sorted(required - set(units.columns))
    if missing:
        raise KeyError(f"units thiếu cột: {missing}")

    result = units.copy().reset_index(drop=True)
    result["key_zh_plan_a"] = result["key_norm"].map(key_map.mapping)
    result["key_zh_effective"] = result["key_zh"].where(
        result["key_zh"].notna() & (result["key_zh"].astype(str).str.strip() != ""),
        result["key_zh_plan_a"],
    )
    status_by_key = {
        str(row["key_norm"]): str(row["key_map_status"])
        for row in key_map.table.to_dict("records")
    }
    result["key_map_status_plan_a"] = result["key_norm"].map(status_by_key).fillna("pending_review")
    result.loc[
        result["key_zh"].notna() & (result["key_zh"].astype(str).str.strip() != ""),
        "key_map_status_plan_a",
    ] = "mapped_existing"

    result["source_text_bt"] = result["text_vi"]
    attribute_mask = result["unit_type"] == "attribute"
    result.loc[attribute_mask, "source_text_bt"] = result.loc[attribute_mask, "value_vi"]
    result["source_text_bt"] = result["source_text_bt"].map(
        lambda value: normalize_text(value, collapse_whitespace=True).strip()
    )
    result["source_lang"] = SOURCE_LANG
    result["target_lang"] = TARGET_LANG

    routes = [
        _route_for(str(unit_type), str(value_class))
        for unit_type, value_class in zip(result["unit_type"], result["value_class"])
    ]
    result["bt_route"] = [route for route, _ in routes]
    result["bt_route_fallback"] = [fallback for _, fallback in routes]
    result["needs_mt"] = result["bt_route"].isin({"mt", "dictionary"})
    result["is_preserve"] = result["bt_route"] == "preserve"

    result["protected_spans"] = result["source_text_bt"].map(
        lambda text: json.dumps(
            extract_protected_spans(
                str(text or ""),
                brand_tokens=set(),
                whitelist=set(),
                limit=80,
            ),
            ensure_ascii=False,
        )
    )
    result["source_text_chars"] = result["source_text_bt"].map(
        lambda text: measure_text(text)[0]
    )
    result["source_text_words"] = result["source_text_bt"].map(
        lambda text: measure_text(text)[1]
    )

    key_chars = result["key_zh_effective"].map(
        lambda key: measure_text(key)[0] if key else 0
    )
    result["key_zh_chars"] = key_chars.astype(int)
    result["expected_zh_key_chars"] = key_chars.astype(int)
    expected_value = _numeric_series(result, "expected_zh_value_chars")
    value_low = _numeric_series(result, "zh_value_chars_p05")
    value_high = _numeric_series(result, "zh_value_chars_p95")
    separator = result["key_zh_chars"].map(lambda count: separator_chars if count > 0 else 0)
    result["expected_zh_text_chars"] = expected_value + key_chars + separator
    result["zh_text_chars_p05"] = value_low + key_chars + separator
    result["zh_text_chars_p95"] = value_high + key_chars + separator
    result["key_len_source_plan_a"] = result["key_zh_chars"].map(
        lambda count: "actual" if count > 0 else "missing"
    )
    if "key_len_source" in result.columns:
        result.loc[
            (result["key_zh_chars"] == 0) & result["key_len_source"].notna(),
            "key_len_source_plan_a",
        ] = result["key_len_source"]
    result.loc[result["unit_type"] == "title", "key_len_source_plan_a"] = "title"

    if "train_eligible" in result.columns:
        result["train_pair_eligible"] = result["train_eligible"].fillna(False).astype(bool)
    else:
        result["train_pair_eligible"] = False
    result["merge_required"] = True
    result["priority"] = result["train_pair_eligible"].map(
        lambda value: 1 if value else 2
    )
    result["bt_ready"] = result["source_text_bt"].str.len() > 0
    result["merge_ready"] = (
        result["bt_ready"]
        & (
            (result["unit_type"] == "title")
            | (
                (result["unit_type"] == "attribute")
                & (result["key_zh_chars"] > 0)
            )
        )
    )
    result["merge_block_reason"] = ""
    result.loc[~result["bt_ready"], "merge_block_reason"] = "source_empty"
    result.loc[
        result["bt_ready"]
        & (result["unit_type"] == "attribute")
        & (result["key_zh_chars"] == 0),
        "merge_block_reason",
    ] = "key_unmapped"

    result["cache_key"] = [
        _cache_key(source, key, value_class, route)
        for source, key, value_class, route in zip(
            result["source_text_bt"],
            result["key_norm"].fillna(""),
            result["value_class"].fillna(""),
            result["bt_route"],
        )
    ]
    result["bt_status"] = result["bt_ready"].map(
        lambda ready: "queued" if ready else "blocked"
    )
    result["qc_status"] = "pending"
    result["bt_attempts"] = 0
    result["bt_model"] = "pending"
    result["bt_version"] = PLAN_A_VERSION
    result["plan_a_version"] = PLAN_A_VERSION
    result["target_text_bt"] = None
    result["target_text_zh"] = None

    bt_input = result[result["bt_ready"]].copy()
    cache = build_translation_cache(bt_input)
    product_state = build_product_state(products, result)

    route_counts = {
        str(key): int(value)
        for key, value in result["bt_route"].value_counts().items()
    }
    status_counts = {
        str(key): int(value)
        for key, value in result["bt_status"].value_counts().items()
    }
    stats = {
        "plan": "A",
        "version": PLAN_A_VERSION,
        "n_units": int(len(result)),
        "n_bt_ready": int(result["bt_ready"].sum()),
        "n_merge_ready": int(result["merge_ready"].sum()),
        "n_train_pair_eligible": int(result["train_pair_eligible"].sum()),
        "n_title_units": int((result["unit_type"] == "title").sum()),
        "n_attribute_units": int((result["unit_type"] == "attribute").sum()),
        "route_counts": route_counts,
        "bt_status_counts": status_counts,
        "key_map_status_counts": {
            str(key): int(value)
            for key, value in result["key_map_status_plan_a"].value_counts().items()
        },
        "merge_block_reason_counts": {
            str(key): int(value)
            for key, value in result["merge_block_reason"].value_counts().items()
        },
        "n_cache_keys": int(len(cache)),
        "cache_reduction": round(1 - len(cache) / max(1, len(bt_input)), 4),
        "n_product_ready_for_bt": int(
            (product_state["product_bt_status"] == "ready_for_bt").sum()
        ),
        "n_product_blocked_key_map": int(
            (product_state["product_bt_status"] == "blocked_key_map").sum()
        ),
        "n_product_blocked_source": int(
            (product_state["product_bt_status"] == "blocked_source").sum()
        ),
        "n_product_merge_eligible": int(product_state["merge_eligible"].sum()),
        "key_map": key_map.stats,
    }
    return PlanAResult(
        units=result,
        bt_input=bt_input,
        cache=cache,
        product_state=product_state,
        key_map=key_map,
        stats=stats,
    )


def build_translation_cache(bt_input: pd.DataFrame) -> pd.DataFrame:
    """Gom các unit trùng source/key/class/route thành một entry cache."""
    if bt_input.empty:
        return pd.DataFrame(columns=[
            "cache_key", "source_text_bt", "key_vi", "key_zh_effective",
            "value_class", "bt_route", "bt_route_fallback", "n_units",
            "n_products", "min_priority", "sample_pair_ids",
        ])
    records: list[dict[str, Any]] = []
    for cache_key, group in bt_input.groupby("cache_key", dropna=False):
        first = group.iloc[0]
        records.append({
            "cache_key": str(cache_key),
            "source_text_bt": first["source_text_bt"],
            "key_vi": first.get("key_vi", ""),
            "key_zh_effective": first.get("key_zh_effective", ""),
            "value_class": first.get("value_class", ""),
            "bt_route": first.get("bt_route", ""),
            "bt_route_fallback": first.get("bt_route_fallback", ""),
            "n_units": int(len(group)),
            "n_products": int(group["product_id"].nunique()),
            "min_priority": int(group["priority"].min()),
            "sample_pair_ids": json.dumps(
                group["pair_id"].head(5).tolist(), ensure_ascii=False
            ),
        })
    return pd.DataFrame(records).sort_values(
        ["min_priority", "n_units"], ascending=[True, False]
    ).reset_index(drop=True)


def build_product_state(
    products: pd.DataFrame,
    units: pd.DataFrame,
) -> pd.DataFrame:
    """Tổng hợp trạng thái pre-BT/merge theo product."""
    if units.empty:
        return products.copy()
    grouped = units.groupby("product_id").agg(
        n_units_plan_a=("pair_id", "size"),
        n_train_pair_eligible=("train_pair_eligible", "sum"),
        n_merge_required=("merge_required", "sum"),
        n_merge_ready=("merge_ready", "sum"),
        n_bt_ready=("bt_ready", "sum"),
        n_key_unmapped=("merge_block_reason", lambda values: int((values == "key_unmapped").sum())),
        n_source_empty=("merge_block_reason", lambda values: int((values == "source_empty").sum())),
        n_route_mt=("bt_route", lambda values: int((values == "mt").sum())),
        n_route_dictionary=("bt_route", lambda values: int((values == "dictionary").sum())),
        n_route_rule=("bt_route", lambda values: int((values == "rule").sum())),
        n_route_preserve=("bt_route", lambda values: int((values == "preserve").sum())),
    ).reset_index()
    state = grouped.copy()
    state["key_map_complete"] = state["n_key_unmapped"] == 0
    state["translation_ready"] = state["n_bt_ready"] == state["n_units_plan_a"]
    state["plan_a_ready"] = (
        (state["n_merge_ready"] == state["n_merge_required"])
        & state["translation_ready"]
    )
    state["product_bt_status"] = "ready_for_bt"
    state.loc[~state["key_map_complete"], "product_bt_status"] = "blocked_key_map"
    state.loc[~state["translation_ready"], "product_bt_status"] = "blocked_source"
    state["merge_eligible"] = False  # chỉ bật sau BT + QC + reassembly
    state["plan_a_version"] = PLAN_A_VERSION

    base_columns = [
        column for column in (
            "product_id", "title_vi", "category", "split",
            "n_units", "n_train_eligible_units", "bt_status", "zh_origin",
        ) if column in products.columns
    ]
    merged = products[base_columns].merge(
        state,
        on="product_id",
        how="left",
        suffixes=("", "_plan_a"),
    )
    for column in (
        "n_units_plan_a", "n_train_pair_eligible", "n_merge_required",
        "n_merge_ready", "n_bt_ready", "n_key_unmapped", "n_source_empty",
        "n_route_mt", "n_route_dictionary", "n_route_rule", "n_route_preserve",
    ):
        if column in merged.columns:
            merged[column] = merged[column].fillna(0).astype(int)
    for column in ("key_map_complete", "translation_ready", "plan_a_ready", "merge_eligible"):
        if column in merged.columns:
            merged[column] = merged[column].fillna(False).astype(bool)
    merged["product_bt_status"] = merged["product_bt_status"].fillna("empty")
    return merged.reset_index(drop=True)


def export_plan_a(
    result: PlanAResult,
    *,
    units_path: str | Path = TIKI_BT_UNITS_PLAN_A_PARQUET,
    input_path: str | Path = TIKI_BT_INPUT_PARQUET,
    cache_path: str | Path = TIKI_TRANSLATION_CACHE_PARQUET,
    product_state_path: str | Path = TIKI_PRODUCTS_BT_STATE_PARQUET,
    key_map_path: str | Path = TIKI_KEY_MAP_CSV,
    key_map_review_path: str | Path = TIKI_KEY_MAP_REVIEW_CSV,
    routing_path: str | Path = TIKI_BT_ROUTING_CSV,
    sample_path: str | Path = TIKI_BT_INPUT_SAMPLE_CSV,
    stats_path: str | Path = TIKI_BT_PLAN_A_STATS_JSON,
) -> dict[str, str]:
    """Ghi toàn bộ artifact Plan A."""
    paths = {
        "units": Path(units_path),
        "bt_input": Path(input_path),
        "cache": Path(cache_path),
        "product_state": Path(product_state_path),
        "key_map": Path(key_map_path),
        "key_map_review": Path(key_map_review_path),
        "routing": Path(routing_path),
        "sample": Path(sample_path),
        "stats": Path(stats_path),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    result.units.to_parquet(paths["units"], index=False, compression="snappy")
    result.bt_input.to_parquet(paths["bt_input"], index=False, compression="snappy")
    result.cache.to_parquet(paths["cache"], index=False, compression="snappy")
    result.product_state.to_parquet(
        paths["product_state"], index=False, compression="snappy"
    )
    result.key_map.table.to_csv(paths["key_map"], index=False, encoding="utf-8-sig")
    result.key_map.review.to_csv(
        paths["key_map_review"], index=False, encoding="utf-8-sig"
    )

    routing = (
        result.units.groupby(
            ["unit_type", "value_class", "bt_route", "bt_route_fallback"],
            dropna=False,
        )
        .agg(
            n_units=("pair_id", "size"),
            n_train_pair_eligible=("train_pair_eligible", "sum"),
            n_merge_ready=("merge_ready", "sum"),
            n_key_unmapped=("merge_block_reason", lambda values: int((values == "key_unmapped").sum())),
        )
        .reset_index()
        .sort_values(["unit_type", "value_class", "bt_route"])
    )
    routing.to_csv(paths["routing"], index=False, encoding="utf-8-sig")
    result.bt_input.head(200).to_csv(
        paths["sample"], index=False, encoding="utf-8-sig"
    )

    with paths["stats"].open("w", encoding="utf-8") as handle:
        json.dump(result.stats, handle, ensure_ascii=False, indent=2)
    return {key: str(value) for key, value in paths.items()}
