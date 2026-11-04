import random
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import title_translation_pipeline as pipeline


class TranslationQualityTests(unittest.TestCase):
    def test_repeated_dot_thousands_groups_do_not_crash_number_qa(self):
        self.assertEqual(pipeline._number_value('2.500.000'),2500000)
        self.assertEqual(pipeline._number_value('2.500.000,50'),2500000.5)
        self.assertEqual(pipeline._number_value('2.5'),2.5)
        self.assertEqual(pipeline._number_value('2,500,000.50'),2500000.5)

    def test_comparison_key_only_normalizes_requested_surface_details(self):
        self.assertEqual(
            pipeline.normalize_title_key("  Áo   thun, 4G TS03。 "),
            pipeline.normalize_title_key("áo thun, 4G TS03"),
        )
        self.assertNotEqual(
            pipeline.normalize_title_key("áo thun"),
            pipeline.normalize_title_key("ao thun"),
        )
        self.assertNotEqual(
            pipeline.normalize_title_key("Sản phẩm 10"),
            pipeline.normalize_title_key("Sản phẩm 11"),
        )

    def test_model_codes_and_decimal_numbers_are_preserved(self):
        source = "车载屏 4G TS03 10.26寸"
        good = "Màn hình ô tô 4G TS03 10,26 inch"
        bad = "Màn hình ô tô 5G TS04 10,27 inch"
        self.assertEqual(pipeline.check_translation(source, good)["status"], "pass")
        bad_result = pipeline.check_translation(source, bad)
        self.assertEqual(bad_result["status"], "fail")
        self.assertIn("number_mismatch", bad_result["hard_flags"])
        self.assertTrue(
            any(
                flag.startswith("model_code_missing:")
                and {"4g", "ts03"}.issubset(set(flag.partition(":")[2].split(",")))
                for flag in bad_result["hard_flags"]
            )
        )
        converted_unit = pipeline.check_translation("热销35克", "Bán chạy 35g")
        self.assertEqual(converted_unit["status"], "pass")
        quantity_unit = pipeline.check_translation("工具套装43PCS", "Bộ dụng cụ 43 món")
        self.assertEqual(quantity_unit["status"], "pass")
        self.assertNotIn("43pcs", pipeline._model_codes("工具套装43PCS"))

    def test_chinese_written_numbers_and_hours_are_not_false_failures(self):
        written_numbers = pipeline.check_translation(
            "汽车坐垫五座四合一",
            "Đệm ghế ô tô 5 chỗ 4 trong 1",
        )
        self.assertEqual(written_numbers["status"], "pass")

        repeated_shape_number = pipeline.check_translation(
            "高密度吸水海绵8字洗车海绵擦车八字海绵块",
            "Bọt biển thấm nước mật độ cao 8 chữ, mút rửa xe 8 chữ",
        )
        self.assertEqual(repeated_shape_number["status"], "pass")

        hours = pipeline.check_translation("现货48h", "Có sẵn trong 48 giờ")
        self.assertEqual(hours["status"], "pass")

        round_the_clock = pipeline.check_translation("远程监控24小时", "Giám sát từ xa 24/24")
        self.assertEqual(round_the_clock["status"], "pass")

        brand_number = pipeline.check_translation("70迈记录仪70mai", "Camera hành trình 70mai")
        self.assertEqual(brand_number["status"], "pass")

        model_split = pipeline.check_translation("适用modelY/3", "Phù hợp Model Y/3")
        self.assertEqual(model_split["status"], "pass")

        year = pipeline.check_translation("pu2026年新款", "Mẫu PU 2026 mới")
        self.assertEqual(year["status"], "pass")

        brand_name = pipeline.check_translation("三和变色龙除胶剂", "Sanhe tẩy keo cho ô tô")
        self.assertEqual(brand_name["status"], "pass")

        double_layer = pipeline.check_translation(
            "汽车座椅双层储物网兜",
            "Túi lưới đựng đồ hai lớp cho ghế ô tô",
        )
        self.assertEqual(double_layer["status"], "pass")
        self.assertFalse(
            any(flag.startswith("unmatched_latin_tokens:") for flag in double_layer["advisory_flags"])
        )

        four_seasons = pipeline.check_translation(
            "卡通汽车头枕四季通用",
            "Gối đầu ô tô dùng được quanh năm",
        )
        self.assertEqual(four_seasons["status"], "pass")

        four_seasons_literal = pipeline.check_translation(
            "卡通汽车头枕四季通用",
            "Gối đầu ô tô dùng được bốn mùa",
        )
        self.assertEqual(four_seasons_literal["status"], "pass")

        one_pair = pipeline.check_translation("一双鞋", "Một đôi giày")
        self.assertEqual(one_pair["status"], "pass")

        double_sided = pipeline.check_translation("双面胶", "Băng keo hai mặt")
        self.assertEqual(double_sided["status"], "pass")
        double_camera = pipeline.check_translation("双镜头", "Camera kép")
        self.assertEqual(double_camera["status"], "pass")
        double_shoulder = pipeline.check_translation("新款双肩包", "Ba lô hai quai mẫu mới")
        self.assertEqual(double_shoulder["status"], "pass")
        dual_recording = pipeline.check_translation(
            "前后双录行车记录仪双镜头",
            "Camera hành trình ghi hình trước sau, hai ống kính",
        )
        self.assertEqual(dual_recording["status"], "pass")
        double_bed = pipeline.check_translation("双人床", "Giường đôi")
        self.assertEqual(double_bed["status"], "pass")
        repeated_double_shoulder = pipeline.check_translation(
            "双肩包双肩背包", "Ba lô hai quai"
        )
        self.assertEqual(repeated_double_shoulder["status"], "pass")
        dual_recording_and_lens = pipeline.check_translation(
            "前后双录双镜头行车记录仪",
            "Camera hành trình ghi hình trước sau, hai kênh và ống kính kép",
        )
        self.assertEqual(dual_recording_and_lens["status"], "pass")
        generations = pipeline.check_translation(
            "耳机二三代运动四五六代", "Tai nghe thế hệ 2 3, 4 5 6"
        )
        self.assertEqual(generations["status"], "pass")
        variant_set = pipeline.check_translation(
            "外贸套三套四拉杆箱", "Vali kéo bộ 3 hoặc bộ 4"
        )
        self.assertEqual(variant_set["status"], "pass")
        two_use_sprayer = pipeline.check_translation(
            "喷油壶一体两用", "Bình xịt dầu một bình, hai công dụng"
        )
        self.assertEqual(two_use_sprayer["status"], "pass")
        neckline = pipeline.check_translation(
            "一字领复古礼服", "Váy cổ trễ vai phong cách cổ điển"
        )
        self.assertEqual(neckline["status"], "pass")
        tea_cake = pipeline.check_translation(
            "云南七子饼茶", "Trà bánh Phổ Nhĩ Vân Nam 7 viên"
        )
        self.assertEqual(tea_cake["status"], "pass")
        nine_grid = pipeline.check_translation(
            "手串收纳九宫格", "Hộp đựng vòng tay chia 9 ngăn"
        )
        self.assertEqual(nine_grid["status"], "pass")

        lily_tea = pipeline.check_translation("百合茶", "Trà hoa bách hợp")
        self.assertEqual(lily_tea["status"], "pass")
        spare_parts = pipeline.check_translation("零件盒", "Hộp đựng linh kiện")
        self.assertEqual(spare_parts["status"], "pass")
        vietnamese_supply = pipeline.check_translation(
            "冬季汽车用品", "Vật tư ô tô mùa đông"
        )
        self.assertEqual(vietnamese_supply["status"], "pass")

        omitted_size = pipeline.check_translation(
            "婚庆行李箱20寸旅行箱24寸",
            "Vali cưới 24 inch",
        )
        self.assertEqual(omitted_size["status"], "fail")
        self.assertIn("number_mismatch", omitted_size["hard_flags"])

        invented_rotation = pipeline.check_translation(
            "行李箱万向轮",
            "Vali có bánh xe xoay 360 độ",
        )
        self.assertEqual(invented_rotation["status"], "fail")
        self.assertIn("number_mismatch", invented_rotation["hard_flags"])

    def test_title_classifier_units_roman_types_and_semantic_count_equivalents(self):
        cases = [
            (
                "医用补水级修复冷敷贴二类械字号医美三型胶原蛋白敷料医用面膜",
                "Mặt nạ cấp ẩm phục hồi y tế loại 2, collagen type III dùng trong thẩm mỹ y khoa",
            ),
            (
                "2.0米蓓尔蓝绷带涂抹面膜补水保湿提亮修护舒缓华熙生物",
                "Mặt nạ bôi Belle Blue Bandage 2.0m cấp ẩm dưỡng ẩm làm sáng phục hồi làm dịu Huaxi Bio",
            ),
            (
                "虾青素双抗面膜抗氧化抗皱紧致抗初老熬夜提亮肤色",
                "Mặt nạ Astaxanthin chống oxy hóa, chống nhăn, săn chắc và chống lão hóa sớm",
            ),
            (
                "面膜双抗紧致提亮修复",
                "Mặt nạ hai tác dụng chống oxy hóa, làm săn chắc và sáng da",
            ),
            (
                "美中0添加桃花生物发酵面膜护肤定制保湿",
                "Mặt nạ lên men sinh học hoa đào không phụ gia, dưỡng ẩm tùy chỉnh",
            ),
            (
                "黄芪本草软膜粉专用七子粉提亮肤色",
                "Bột mặt nạ Hoàng Kỳ chuyên dụng, bột Thất Tử làm sáng da",
            ),
            (
                "红绷带提拉面膜咬合双下巴神器",
                "Mặt nạ băng đai nâng cơ và giảm nọng cằm",
            ),
            (
                "碧素堂八杯水莹润补水面膜盒装",
                "Mặt nạ dưỡng ẩm Bisu Tang 8 cốc nước dạng hộp",
            ),
            (
                "三型胶原蛋白猪皮免洗面膜补水保湿",
                "Mặt nạ collagen type III da heo cấp ẩm",
            ),
            (
                "十大名牌洗面奶氨基酸控油祛痘",
                "Sữa rửa mặt trong top 10 thương hiệu nổi tiếng, amino acid kiềm dầu trị mụn",
            ),
            (
                "洗卸合一洗面奶氨基酸温和清洁",
                "Sữa rửa mặt kết hợp làm sạch và tẩy trang dịu nhẹ với amino acid",
            ),
        ]
        for source, candidate in cases:
            with self.subTest(source=source):
                result = pipeline.check_translation(source, candidate)
                self.assertEqual(result["status"], "pass", result["hard_flags"])

    def test_camera_recording_counts_storage_and_resolution_equivalences(self):
        cases = [
            ("汽车头枕一对", "Gối tựa đầu bộ đôi"),
            (
                "1080高清行车记录仪10寸",
                "Camera hành trình 1080P HD 10 inch",
            ),
            ("70迈行车记录仪4K", "Camera hành trình 70mai 4K"),
            (
                "T头行车记录仪u盘用typec适用c6l/glc级s级EL新eqe车载优盘64128g",
                "Camera hành trình Type-C dùng cho c6l/glc, EQE, USB 64GB 128GB",
            ),
            (
                "前后双摄行车记录仪",
                "Camera hành trình với camera trước và sau",
            ),
            (
                "前后双录行车记录仪",
                "Camera hành trình ghi hình hai chiều trước sau",
            ),
            (
                "U3单录行车记录仪",
                "U3 camera hành trình ghi hình một kênh",
            ),
            (
                "跨境1080P三录行车记录仪",
                "1080P camera hành trình ghi hình ba kênh",
            ),
            ("四录行车记录仪", "Camera hành trình ghi hình bốn kênh"),
        ]
        for source, candidate in cases:
            with self.subTest(source=source):
                result = pipeline.check_translation(source, candidate)
                self.assertEqual(result["status"], "pass", result["hard_flags"])

        wrong_three_channel_count = pipeline.check_translation(
            "跨境1080P三录行车记录仪",
            "1080P camera hành trình ghi hình hai chiều",
        )
        self.assertIn("number_mismatch", wrong_three_channel_count["hard_flags"])

        omitted_supplier_number = pipeline.check_translation(
            "行车记录仪前后双摄高夜视倒车影像5新免走线停车监控智能夜视",
            "Camera hành trình hai camera trước sau, tầm nhìn ban đêm, hỗ trợ camera lùi",
        )
        self.assertIn("number_mismatch", omitted_supplier_number["hard_flags"])

    def test_voltage_resolution_and_shorthand_model_year_specs(self):
        cases = [
            ("汽车记录仪12伏24V", "Camera hành trình ô tô 12V 24V"),
            ("跨境1080P三录行车记录仪", "1080P camera hành trình ghi hình ba kênh"),
            ("1080高清行车记录仪10寸", "Camera hành trình 1080P HD 10 inch"),
            ("摄像头25款", "Camera mẫu 2025"),
            ("两芯款", "Bản 2 lõi"),
            (
                "一体机前后双摄",
                "Camera hành trình camera trước sau tất cả trong một",
            ),
            (
                "奇瑞瑞虎9x专用原厂瑞虎九2026新款",
                "Camera Chery Tiggo 9x chuyên dụng chính hãng, Tiggo 9 mẫu mới 2026",
            ),
            (
                "行车记录仪后视镜三镜头WiFi高清1080P倒车影像全屏触控三录同显",
                "Gương chiếu hậu camera hành trình 3 ống kính WiFi HD 1080P màn hình cảm ứng toàn màn hình hỗ trợ camera lùi hiển thị đồng thời 3 video",
            ),
        ]
        for source, candidate in cases:
            with self.subTest(source=source):
                result = pipeline.check_translation(source, candidate)
                self.assertEqual(result["status"], "pass", result["hard_flags"])

        duration_case = pipeline.check_translation(
            "批发新款四录高清1080P行车记录仪WIFI互联倒车影像GPS24小时监控",
            "Sỉ mẫu mới camera hành trình 1080P quay bốn kênh độ nét cao, kết nối WIFI, camera lùi GPS24 giám sát trong 24 giờ",
        )
        self.assertEqual(duration_case["status"], "pass", duration_case["hard_flags"])

    def test_school_year_backpack_and_large_number_equivalences_are_contextual(self):
        cases = [
            ("商务车记录仪系统四方位", "Camera hành trình xe thương mại bốn hướng"),
            ("严选行车记录仪九亿H06", "Camera hành trình cao cấp 900 triệu H06"),
            ("26年新款户外背包", "Ba lô dã ngoại mẫu mới 2026"),
            ("跨境轻奢女包双肩包25年新款", "Ba lô nữ cao cấp mẫu mới 2025"),
            ("五件套双肩背包", "Bộ 5 món ba lô"),
            ("双肩包双肩背包", "Ba lô"),
            (
                "双肩包初高中小学生书包",
                "Ba lô hai quai cho học sinh cấp 1, cấp 2, cấp 3",
            ),
        ]
        for source, candidate in cases:
            with self.subTest(source=source):
                result = pipeline.check_translation(source, candidate)
                self.assertEqual(result["status"], "pass", result["hard_flags"])

        wrong_school_level = pipeline.check_translation(
            "高中小学生书包", "Cặp sách cho học sinh cấp 2, cấp 3"
        )
        self.assertIn("number_mismatch", wrong_school_level["hard_flags"])

        missing_backpack = pipeline.check_translation("双肩包", "Túi xách")
        self.assertIn("number_mismatch", missing_backpack["hard_flags"])

        missing_generation = pipeline.check_translation(
            "三代哈弗H6二代大狗导航仪", "Thiết bị định vị Haval H6 thế hệ 2 Big Dog"
        )
        self.assertIn("number_mismatch", missing_generation["hard_flags"])

        missing_use_count = pipeline.check_translation("北极三用双肩包", "Ba lô hai quai đa năng")
        self.assertIn("number_mismatch", missing_use_count["hard_flags"])

        non_year_25_must_not_expand = pipeline.check_translation(
            "双肩包25爆款", "Ba lô nữ mẫu hot 2025"
        )
        self.assertIn("number_mismatch", non_year_25_must_not_expand["hard_flags"])

        one_piece_wholesale_still_requires_its_quantity = pipeline.check_translation(
            "三镜头行车记录仪一件批发", "Camera hành trình ba ống kính"
        )
        self.assertIn("number_mismatch", one_piece_wholesale_still_requires_its_quantity["hard_flags"])

    def test_gemma_normalizes_unambiguous_residual_recording_terms(self):
        import title_translation_hybrid as hybrid

        normalized = hybrid._normalize_gemma_measurements(
            "Auto投屏 USB单录/双录 song录"
        )
        self.assertEqual(
            normalized,
            "Auto chiếu màn hình USB ghi hình một kênh/ghi hình hai kênh ghi hình hai kênh",
        )
        self.assertFalse(pipeline.contains_cjk(normalized))

    def test_jin_conversion_is_checked_and_price_is_not_halved(self):
        converted = pipeline.check_translation("商品净重2斤", "Khối lượng tịnh 1 kg")
        self.assertEqual(converted["status"], "pass")
        self.assertTrue(converted["jin_conversion_detected"])
        self.assertIn("jin_mass_conversion_detected", converted["advisory_flags"])

        price_changed = pipeline.check_translation("售价20元/斤", "Giá bán 10元/kg")
        self.assertEqual(price_changed["status"], "fail")
        self.assertIn("currency_amount_mismatch", price_changed["hard_flags"])

        half_jin = pipeline.check_translation("净重半斤", "Khối lượng tịnh 250 g")
        self.assertEqual(half_jin["status"], "pass")
        one_and_half_jin = pipeline.check_translation("净重一斤半", "Khối lượng tịnh 750 g")
        self.assertEqual(one_and_half_jin["status"], "pass")
        repeated_metric_equivalent = pipeline.check_translation(
            "500g手提一斤", "Túi trà trọng lượng 500g"
        )
        self.assertEqual(repeated_metric_equivalent["status"], "pass")

        wrong_vietnamese_cate = pipeline.check_translation("净重一斤", "Khối lượng tịnh 1 cân")
        self.assertEqual(wrong_vietnamese_cate["status"], "fail")
        self.assertIn("jin_unit_unresolved", wrong_vietnamese_cate["hard_flags"])

    def test_real_world_gemma_pilot_number_and_code_regressions(self):
        cases = [
            (
                "PD120w快充充电宝20000毫安大容量typec双向数显移动电源超大容量",
                "Sạc dự phòng PD120w sạc nhanh dung lượng 20000mAh Type-C hai chiều hiển thị số",
            ),
            ("婴儿套装儿童春秋打底两件套", "Bộ quần áo trẻ em mùa xuân thu hai món"),
            ("新人可专享0.01元可爱笔筒", "Người mới nhận ưu đãi 0.01 yuan, ống cắm bút"),
            ("新人可专享百元内商品", "Sản phẩm giá dưới 100 tệ"),
            ("快充2万毫安充电宝", "Sạc dự phòng sạc nhanh 20000mAh"),
            ("LODI新生儿宝宝0-6个月一岁以上", "LODI cho bé sơ sinh 0-6 tháng, từ 1 tuổi trở lên"),
            ("茶叶【买一斤送半斤】", "Trà 【mua 500 g tặng 250 g】"),
            ("三只松鼠_蜀香牛肉100g", "Three Squirrels - thịt bò Shu Xiang 100g"),
            ("纯棉短袖T恤重磅220克", "Áo thun cotton nguyên chất dày dặn 220g"),
            ("纯棉短袖T恤重磅220克", "Áo thun 100% cotton dày dặn 220g"),
            ("【一折专区】男装T恤", "Khu vực ưu đãi 【giá còn 10%】 áo thun nam"),
            ("【一折专区】男装T恤", "Khu vực giảm giá 90% áo thun nam"),
            ("【大盒+54个方瓶】30ml小方药瓶", "Hộp lớn + 54 chai vuông, chai thuốc 30ml"),
            ("好吃点香脆饼干800g/盒", "Bánh quy giòn thơm 800g/hộp"),
            ("适用OPPOreno16/15/14手机壳", "Ốp điện thoại dùng cho OPPOreno16/15/14"),
            ("苦荞茶2500g5斤", "Trà kiều mạch đắng 2500g và 2.5 kg"),
        ]
        for source, candidate in cases:
            with self.subTest(source=source, candidate=candidate):
                result = pipeline.check_translation(source, candidate)
                self.assertEqual(result["status"], "pass", result["hard_flags"])

        omitted_code = pipeline.check_translation(
            "适用OPPOreno16/15/14手机壳", "Ốp điện thoại dùng cho OPPO rno16/15/14"
        )
        self.assertTrue(any(flag.startswith("model_code_missing:") for flag in omitted_code["hard_flags"]))

        duplicated_year = pipeline.check_translation(
            "LV2026新款四件套", "Bộ chăn ga 4 món LV2026 mẫu mới 2026"
        )
        self.assertIn("number_mismatch", duplicated_year["hard_flags"])

        dropshipping = pipeline.check_translation("一件代发", "Dịch vụ dropshipping")
        self.assertEqual(dropshipping["status"], "pass")

        one_piece = pipeline.check_translation("俄语 一片式连衣短裙", "Váy liền thân dáng ngắn")
        self.assertEqual(one_piece["status"], "pass")
        single_shoulder = pipeline.check_translation("单肩斜挎包", "Túi đeo chéo một vai")
        self.assertEqual(single_shoulder["status"], "pass")
        thousand_separator = pipeline.check_translation("快充2万毫安", "Sạc nhanh 20,000 mAh")
        self.assertEqual(thousand_separator["status"], "pass")
        textile_count = pipeline.check_translation(
            "【60支匹马棉】", "Cotton Supima 60s cao cấp"
        )
        self.assertEqual(textile_count["status"], "pass")
        double_twill = pipeline.check_translation("双绞丝", "Sợi xoắn double")
        self.assertEqual(double_twill["status"], "pass")
        pure_cotton_equivalent = pipeline.check_translation(
            "出口日本棉100%纯棉四件套", "Bộ chăn ga 4 món cotton nguyên chất xuất khẩu Nhật Bản"
        )
        self.assertEqual(pure_cotton_equivalent["status"], "pass")
        pure_cotton_without_percent_sign = pipeline.check_translation(
            "夏季白色100纯棉T恤", "Áo thun trắng cotton nguyên chất"
        )
        self.assertEqual(pure_cotton_without_percent_sign["status"], "pass")
        explicit_percent_stays_valid = pipeline.check_translation(
            "100%棉纯棉男士Polo", "Áo Polo nam cotton nguyên chất 100%"
        )
        self.assertEqual(explicit_percent_stays_valid["status"], "pass")
        repeated_source_spec_can_be_deduplicated = pipeline.check_translation(
            "四件套2026款四件套", "Bộ 4 món phiên bản 2026"
        )
        self.assertEqual(repeated_source_spec_can_be_deduplicated["status"], "pass")
        candidate_must_not_repeat_year = pipeline.check_translation(
            "2026新款", "Mẫu mới 2026, năm 2026"
        )
        self.assertIn("number_mismatch", candidate_must_not_repeat_year["hard_flags"])
        slashed_mass_unit = pipeline.check_translation(
            "鲜花饼240g/袋", "Bánh hoa hồng 240g/túi"
        )
        self.assertEqual(slashed_mass_unit["status"], "pass")
        slash_separated_mass_options = pipeline.check_translation(
            "零食50g/180g", "Đồ ăn vặt 50g/180g"
        )
        self.assertEqual(slash_separated_mass_options["status"], "pass")
        self.assertNotIn("50g/180g", pipeline._model_codes("零食50g/180g"))
        jin_range_conversion = pipeline.check_translation(
            "体重100-300斤适穿", "Phù hợp cân nặng 50-150kg"
        )
        self.assertEqual(jin_range_conversion["status"], "pass")
        omitted_jin_range_endpoint = pipeline.check_translation(
            "体重2-300斤适穿", "Phù hợp cân nặng 100-150kg"
        )
        self.assertIn("number_mismatch", omitted_jin_range_endpoint["hard_flags"])

    def test_gemma_protects_codes_and_normalizes_residual_chinese_units(self):
        import title_translation_hybrid as hybrid

        protected, mappings = hybrid._protect_model_codes(
            "适用OPPOreno16/15/14手机壳A6/A5"
        )
        self.assertIn("ZXQCODETOKEN", protected)
        self.assertEqual(len(mappings), 2)
        normalized = hybrid._normalize_gemma_measurements("Trà 5斤 và giá 0.01元")
        self.assertEqual(normalized, "Trà 2.5 kg và giá 0.01 yuan")

    def test_gemma_repairs_shortened_code_placeholders(self):
        import title_translation_hybrid as hybrid

        protected = hybrid._protect_model_codes("适用W895手机壳")[1]
        self.assertEqual(
            hybrid._restore_gemma_model_codes("Ốp lưng ZXQCODETOKEN", protected),
            "Ốp lưng W895",
        )
        self.assertEqual(
            hybrid._restore_gemma_model_codes("Ốp lưng W895ZXQ", protected),
            "Ốp lưng W895",
        )
        self.assertEqual(
            hybrid._restore_gemma_model_codes("Vali CODETOKEN", []),
            "Vali",
        )

    def test_truncated_candidate_cannot_pass(self):
        result = pipeline.check_translation(
            "简单商品标题",
            "Tiêu đề sản phẩm",
            generation_truncated=True,
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("generation_limit_reached", result["hard_flags"])

    def test_full_is_locked_to_a_passed_validation_and_rejected_after_stop(self):
        with self.assertRaisesRegex(RuntimeError, "validation passes"):
            pipeline._verify_stage_prerequisites(
                "full", {"config_id": "cfg", "stages": {}}, "cfg"
            )
        state = {
            "config_id": "cfg",
            "stages": {"validate": {"passed": True, "config_id": "cfg"}},
        }
        pipeline._verify_stage_prerequisites("full", state, "cfg")
        state["stages"]["full"] = {"status": "stopped_quality_gate"}
        with self.assertRaisesRegex(RuntimeError, "previously stopped"):
            pipeline._verify_stage_prerequisites("full", state, "cfg")

    def test_full_window_gate_enforces_completion_and_technical_qa_limits(self):
        rows = 200
        audit = pd.DataFrame(
            {
                "translategemma_complete": [True] * 198 + [False] * 2,
                "translategemma_status": ["pass"] * 190 + ["fail"] * 10,
                "qwen3_complete": [True] * rows,
                "qwen3_status": ["pass"] * rows,
            }
        )
        gate = pipeline._full_window_gate(audit, 0, rows)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["models"]["translategemma"]["incomplete_rate"], 0.01)
        self.assertEqual(gate["models"]["translategemma"]["technical_qa_failure_rate"], 0.05)

        audit.loc[0:10, "qwen3_status"] = "fail"
        self.assertFalse(pipeline._full_window_gate(audit, 0, rows)["passed"])

    def test_selection_keeps_existing_vi_or_holds_unresolved_cases(self):
        source = "车载屏 4G TS03"
        raw_vi = "Màn hình xe 4G TS03"
        cleaned_bad = "Màn hình xe 4G TS03 产品"
        selected = pipeline.select_translation(
            source,
            raw_vi,
            cleaned_bad,
            "Màn hình ô tô 4G TS03",
            "Màn hình xe hơi 4G TS03",
        )
        self.assertEqual(selected["source"], "title_vi")
        self.assertFalse(selected["semantic_verified"])

        hold = pipeline.select_translation(
            source,
            "车载屏 4G TS03",
            "车载屏 4G TS03",
            "Màn hình ô tô 4G TS03",
            "Màn hình 4G TS03",
        )
        self.assertIsNone(hold["selected"])
        self.assertEqual(hold["qa_status"], "hold")

    def test_sample_split_is_disjoint_stratified_and_covers_known_challenges(self):
        records = []
        categories = [f"category_{index}" for index in range(9)]
        for index in range(630):
            title = f"商品 {index:04d} 规格"
            if index < 5:
                title += f" А{index}"
            vi = f"Sản phẩm {index:04d}"
            if 5 <= index < 12:
                vi += " 商品"
            if index % 3 == 0:
                title += " 2斤"
            if index % 4 == 0:
                title += " TS03"
            records.append(
                {
                    "title_zh": title,
                    "category": categories[index % len(categories)],
                    "title_vi": vi,
                    "title_vi_cleaned": vi,
                }
            )
        groups = pipeline._group_table(pd.DataFrame(records))
        rng = random.Random(pipeline.SEED)
        pilot_challenge = pipeline._sample_challenge(groups, set(), 50, rng)
        pilot_rep = pipeline._sample_representative(groups, set(pilot_challenge), 100, rng)
        pilot = set(pilot_challenge) | set(pilot_rep)
        validate_challenge = pipeline._sample_challenge(groups, pilot, 40, rng)
        validate_rep = pipeline._sample_representative(
            groups, pilot | set(validate_challenge), 160, rng
        )
        validate = set(validate_challenge) | set(validate_rep)
        self.assertEqual(len(pilot), 150)
        self.assertEqual(len(validate), 200)
        self.assertFalse(pilot & validate)
        self.assertTrue(all(pipeline.contains_cyrillic(title) for title in groups.loc[groups["source_cyrillic"], "title_zh"]))
        self.assertTrue(set(groups.loc[groups["source_cyrillic"], "title_zh"]).issubset(pilot))
        self.assertTrue(set(groups.loc[groups["existing_vi_cjk"], "title_zh"]).issubset(pilot))
        self.assertEqual(set(groups["category"]), set(categories))

    def test_cache_survives_close_and_reopen_and_keeps_duplicate_source_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            cache = pipeline.TranslationCache(
                path,
                "qwen3",
                "Qwen/Qwen3-14B",
                "revision123",
                "config123",
            )
            cache.put_many(
                [
                    (
                        "相同 title",
                        {
                            "output_text": "Bản dịch dùng chung",
                            "status": "ok",
                            "truncated": False,
                            "output_tokens": 8,
                            "attempts": 1,
                            "elapsed_seconds": 0.2,
                        },
                    ),
                    (
                        "已截断",
                        {
                            "output_text": "Bản dịch chưa hoàn tất",
                            "status": "ok",
                            "truncated": True,
                            "output_tokens": 512,
                            "attempts": 2,
                            "elapsed_seconds": 1.0,
                        },
                    ),
                ]
            )
            reopened = pipeline.TranslationCache(
                path,
                "qwen3",
                "Qwen/Qwen3-14B",
                "revision123",
                "config123",
            )
            records = reopened.get_many(["相同 title", "已截断", "không có"])
            self.assertEqual(records["相同 title"]["output_text"], "Bản dịch dùng chung")
            self.assertTrue(pipeline._cached_complete(records["相同 title"]))
            self.assertFalse(pipeline._cached_complete(records["已截断"]))
            self.assertNotIn("không có", records)
            self.assertEqual(reopened.get_complete("已截断"), None)


if __name__ == "__main__":
    unittest.main()
