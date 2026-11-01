import pytest

from cado_core.quantity_norm import cn_to_number, compare, extract_quantities, signature


@pytest.mark.parametrize("src,tgt", [
    ("144,5 cm", "144.5 厘米"),
    ("1.234 kg", "1234 克"),
    ("10 gói x 16g", "10包×16克"),
    ("3-5 cm", "3-5厘米"),
    ("3-5 cm", "3-5 厘米"),
    ("3x5cm", "3x5厘米"),
    ("20W", "20瓦"),
    ("3000L", "3000升"),
    ("5V", "5伏"),
    ("2.4m", "2.4米"),
    ("500ml", "500毫升"),
    ("12 tháng", "12个月"),
    ("256GB", "256gb"),
    ("1/2 kg", "0.5公斤"),
    ("½ kg", "0.5公斤"),
    ("20寸", "20英寸"),
    ("3cm-5cm", "3-5厘米"),
])
def test_quantity_match(src, tgt):
    assert compare(src, tgt)["match"], (src, tgt, compare(src, tgt))


@pytest.mark.parametrize("src,tgt", [
    ("5cm", "8厘米"),
    ("1kg", "500克"),
    ("20W", "30瓦"),
    ("2L", "2ml"),
])
def test_quantity_mismatch(src, tgt):
    result = compare(src, tgt)
    assert not result["match"]
    assert result["strong_mismatch"]


@pytest.mark.parametrize("src,tgt", [
    ("Bộ bốn mảnh", "四件套"),
    ("2026新款", "2026年新款"),
    ("Móc treo 6 cái", "挂钩 6个"),
])
def test_weak_quantities_ignored(src, tgt):
    # số đếm / năm / bare không được làm fail strict compare
    assert compare(src, tgt)["match"]


@pytest.mark.parametrize("text,expected", [
    ("三百", 300),
    ("十五", 15),
    ("两", 2),
    ("半", 0.5),
    ("一千二百", 1200),
])
def test_chinese_numerals(text, expected):
    assert cn_to_number(text) == expected


def test_code_tokens_ignored():
    # số trong mã sản phẩm không phải đại lượng
    quantities = extract_quantities("Elmich EL0002D189 6件", "zh")
    values = [q.value for q in quantities]
    assert 2 not in values and 189 not in values


def test_range_unit_propagation():
    quantities = extract_quantities("3-5 cm", "vi")
    dims = [q.dimension for q in quantities]
    assert dims.count("length") == 2


def test_ideal_models_not_units():
    assert compare("iPhone 15", "iPhone 15")["match"]
