#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script truy vấn file parquet (bilingual_zh_vi.parquet) hỗ trợ:
  1. Truy vấn bằng cú pháp SQL hoàn chỉnh (SELECT ... FROM data WHERE ...)
  2. Lọc bằng điều kiện WHERE giống SQL (--where "price < 15 AND category = 'bags'")
  3. Tìm kiếm chuỗi ký tự trên tất cả các cột
  4. Hiển thị toàn bộ các cột (expanded key-value hoặc bảng ASCII grid)
  5. Xem cấu trúc các cột (schema) để dễ viết câu truy vấn

Cách sử dụng:
  # 1. Truy vấn SQL trực tiếp:
  python query_parquet.py "SELECT product_id, title_vi, price, category FROM data WHERE price < 15 LIMIT 5"
  python query_parquet.py "SELECT category, count(*) AS total, round(avg(price), 2) AS avg_price FROM data GROUP BY category"

  # 2. Sàng lọc theo điều kiện WHERE (giống SQL):
  python query_parquet.py -w "product_id = '944607859983'"
  python query_parquet.py -s "product_id, title_vi, price" -w "category = 'bags' AND price < 20"

  # 3. Tìm chuỗi trên tất cả các cột (hiện đầy đủ cả 30 cột):
  python query_parquet.py "944607859983"
  python query_parquet.py "bàn chải" -n 3

  # 4. Xem danh sách các cột và kiểu dữ liệu:
  python query_parquet.py --schema

  # 5. Chế độ console SQL tương tác:
  python query_parquet.py
"""

import argparse
import os
import re
import sys
import time

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import pandas as pd
except ImportError:
    pd = None


def get_db_connection(parquet_path):
    """Khởi tạo kết nối DuckDB và đăng ký bảng 'data', 'products', 'bilingual_zh_vi'."""
    if duckdb is None:
        return None
    con = duckdb.connect()
    # Đăng ký các alias thông dụng cho bảng
    con.execute(f"CREATE VIEW data AS SELECT * FROM '{parquet_path}'")
    con.execute(f"CREATE VIEW products AS SELECT * FROM '{parquet_path}'")
    con.execute(f"CREATE VIEW bilingual_zh_vi AS SELECT * FROM '{parquet_path}'")
    return con


def get_schema_df(con, parquet_path):
    """Lấy danh sách các cột và kiểu dữ liệu."""
    if con:
        return con.sql("DESCRIBE data").df()
    elif pd:
        df = pd.read_parquet(parquet_path)
        schema_data = [{"column_name": col, "column_type": str(dtype)} for col, dtype in df.dtypes.items()]
        return pd.DataFrame(schema_data)
    return None


def show_schema(con, parquet_path):
    """In danh sách 30 cột để người dùng nắm rõ cấu trúc khi viết SQL."""
    print("=" * 80)
    print(f"📋 CẤU TRÚC BẢNG (SCHEMA) - {parquet_path}")
    print("=" * 80)
    schema_df = get_schema_df(con, parquet_path)
    if schema_df is not None:
        if con:
            con.sql("DESCRIBE data").show(max_rows=50, max_width=100)
        else:
            print(schema_df.to_string(index=False))
    print(f"\n💡 Bạn có thể dùng tên bảng là: 'data', 'products' hoặc 'bilingual_zh_vi'")
    print(f"Ví dụ: SELECT product_id, title_vi, price FROM data WHERE price < 20 LIMIT 5;\n")


def print_expanded_records(df, keyword=None, limit=10):
    """
    In từng bản ghi với TOÀN BỘ các cột (mỗi cột 1 dòng giống `\\x` của PostgreSQL hoặc `\\G` của MySQL).
    Giúp hiển thị đầy đủ cả 30 cột mà không bị cắt xén như xem dạng bảng ngang.
    """
    total = len(df)
    display_count = total if limit <= 0 else min(limit, total)
    cols = list(df.columns)

    print(f"\n--- [ Hiển thị {display_count}/{total} bản ghi (Đầy đủ {len(cols)} cột) ] ---\n")

    for i in range(display_count):
        row = df.iloc[i]
        banner = f"─── [ Bản ghi #{i + 1}/{total} | ID: {row.get('product_id', 'N/A')} ] "
        banner += "─" * max(0, 80 - len(banner))
        print(banner)

        for col in cols:
            val = row[col]
            val_str = "" if val is None or (isinstance(val, float) and pd and pd.isna(val)) else str(val)

            # Rút gọn nhẹ đối với các trường văn bản quá dài (> 500 ký tự) nhưng vẫn đủ chi tiết
            if len(val_str) > 400:
                val_display = val_str[:397] + "..."
            else:
                val_display = val_str

            # Đánh dấu cột có chứa từ khóa nếu đang tìm kiếm
            mark = ""
            if keyword and keyword.lower() in val_str.lower():
                mark = " 🎯 [KHỚP]"

            print(f"  {col:<24} : {val_display}{mark}")

        print("─" * 80)

    if total > display_count:
        print(f"ℹ️ Còn {total - display_count} bản ghi chưa hiển thị. Dùng `-n {total}` hoặc `-o ket_qua.csv` để xem toàn bộ.\n")


def display_results(con, rel, df, format_mode="auto", keyword=None, limit=10):
    """
    Hiển thị kết quả:
      - 'table': Bảng ASCII grid chuẩn SQL
      - 'expanded': Hiển thị toàn bộ các cột từng dòng
      - 'auto': <= 6 cột thì hiện bảng, > 6 cột thì hiện expanded
    """
    total = len(df)
    if total == 0:
        print("❌ Không tìm thấy bản ghi nào khớp.")
        return

    cols = list(df.columns)

    if format_mode == "table" or (format_mode == "auto" and len(cols) <= 6):
        if rel is not None:
            # Dùng bảng ASCII của DuckDB
            show_rows = total if limit <= 0 else limit
            rel.limit(show_rows).show(max_rows=show_rows + 5, max_width=200)
            if total > show_rows:
                print(f"ℹ️ Đang hiển thị {show_rows}/{total} dòng. Dùng `-n {total}` để hiện tất cả hoặc `-x` để xem dạng mở rộng.")
        else:
            show_df = df if limit <= 0 else df.head(limit)
            print(show_df.to_string(index=False))
    else:
        # Format expanded: hiện toàn bộ 30 cột chi tiết
        print_expanded_records(df, keyword=keyword, limit=limit)


def execute_sql(con, sql_query, format_mode="auto", limit=10, output_file=None):
    """Thực thi câu lệnh SQL trực tiếp trên file parquet."""
    if con is None:
        print("❌ Cần cài đặt duckdb để thực thi truy vấn SQL: pip install duckdb")
        return None

    print(f"\n⚡ Thực thi SQL: {sql_query.strip()}")
    t0 = time.time()
    try:
        rel = con.sql(sql_query)
        df = rel.df()
        elapsed = time.time() - t0
        print(f"⏱️ Thời gian: {elapsed:.2f}s | Trả về: {len(df)} dòng, {len(df.columns)} cột")

        if output_file:
            export_data(df, output_file)

        display_results(con, rel, df, format_mode=format_mode, limit=limit)
        return df
    except Exception as e:
        print(f"❌ Lỗi SQL: {e}")
        return None


def execute_where(con, parquet_path, select_cols="*", where_clause="", format_mode="auto", limit=10, output_file=None):
    """Lọc dữ liệu bằng mệnh đề SELECT ... WHERE ..."""
    cols_clause = select_cols if select_cols else "*"
    sql = f"SELECT {cols_clause} FROM data"
    if where_clause and where_clause.strip():
        sql += f" WHERE {where_clause}"
    return execute_sql(con, sql, format_mode=format_mode, limit=limit, output_file=output_file)


def search_all_columns(con, parquet_path, keyword, columns=None, case_sensitive=False, format_mode="auto", limit=10, output_file=None):
    """Tìm chuỗi ký tự trên tất cả các cột của bảng."""
    if not keyword or not keyword.strip():
        print("⚠️ Từ khóa tìm kiếm không được để trống.")
        return None

    keyword = keyword.strip()
    schema_df = get_schema_df(con, parquet_path)
    all_cols = list(schema_df["column_name"]) if schema_df is not None else []
    search_cols = [c for c in columns if c in all_cols] if columns else all_cols

    print(f"\n🔍 Đang tìm kiếm: '{keyword}'")
    print(f"📊 Tìm trong {len(search_cols)} cột (Phân biệt hoa/thường: {'Có' if case_sensitive else 'Không'})")

    if con:
        if case_sensitive:
            conds = " OR ".join([f'contains(CAST("{c}" AS VARCHAR), ?)' for c in search_cols])
        else:
            conds = " OR ".join([f'contains(lower(CAST("{c}" AS VARCHAR)), lower(?))' for c in search_cols])
        sql = f'SELECT * FROM data WHERE {conds}'
        params = [keyword] * len(search_cols)

        t0 = time.time()
        try:
            rel = con.execute(sql, params)
            df = rel.df()
            elapsed = time.time() - t0
            print(f"⏱️ Thời gian: {elapsed:.2f}s | Tìm thấy: {len(df)} kết quả")

            if output_file:
                export_data(df, output_file)

            display_results(con, con.sql(f"SELECT * FROM df"), df, format_mode=format_mode, keyword=keyword, limit=limit)
            return df
        except Exception as e:
            print(f"⚠️ DuckDB gặp lỗi ({e}), chuyển sang đọc bằng Pandas...")

    # Fallback Pandas nếu DuckDB không dùng được
    if pd:
        t0 = time.time()
        df_full = pd.read_parquet(parquet_path)
        mask = False
        for c in search_cols:
            col_series = df_full[c].astype(str)
            m = col_series.str.contains(keyword, case=case_sensitive, na=False, regex=False)
            mask = mask | m
        df = df_full[mask].copy()
        elapsed = time.time() - t0
        print(f"⏱️ Thời gian (Pandas): {elapsed:.2f}s | Tìm thấy: {len(df)} kết quả")

        if output_file:
            export_data(df, output_file)

        display_results(None, None, df, format_mode=format_mode, keyword=keyword, limit=limit)
        return df

    return None


def export_data(df, output_file):
    """Lưu kết quả ra file."""
    if output_file.endswith(".csv"):
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
    elif output_file.endswith(".parquet"):
        df.to_parquet(output_file, index=False)
    elif output_file.endswith(".json"):
        df.to_json(output_file, orient="records", force_ascii=False, indent=2)
    else:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"💾 Đã xuất {len(df)} dòng ra file: {output_file}")


def interactive_console(con, parquet_path):
    """Giao diện dòng lệnh tương tác hỗ trợ cả SQL lẫn tìm kiếm từ khóa."""
    print("=" * 75)
    print("🗄️  CONSOLE TRUY VẤN FILE BILINGUAL PARQUET")
    print("=" * 75)
    print("👉 Nhập câu lệnh SQL bắt đầu bằng SELECT, DESCRIBE, SHOW...")
    print("   Ví dụ: SELECT product_id, title_vi, price FROM data WHERE price < 15 LIMIT 5;")
    print("👉 Hoặc gõ điều kiện: WHERE category = 'bags' AND price > 50")
    print("👉 Hoặc gõ từ khóa trực tiếp để tìm kiếm trong tất cả các cột: 944607859983")
    print("👉 Gõ 'schema' để xem 30 cột | Gõ 'exit' hoặc 'quit' để thoát.")
    print("=" * 75)

    mode = "auto"

    while True:
        try:
            line = input("\nsql> ").strip()
            if not line:
                continue

            cmd_lower = line.lower()
            if cmd_lower in ("exit", "quit", "q"):
                print("Tạm biệt! 👋")
                break
            elif cmd_lower in ("schema", "desc", "describe"):
                show_schema(con, parquet_path)
            elif cmd_lower.startswith("set format "):
                mode = line.split()[-1]
                print(f"Đã đặt chế độ hiển thị: {mode}")
            elif re.match(r"^(select|with|describe|show|explain)\b", cmd_lower):
                execute_sql(con, line.rstrip(";"), format_mode=mode, limit=10)
            elif cmd_lower.startswith("where "):
                where_cond = line[6:].rstrip(";")
                execute_where(con, parquet_path, "*", where_cond, format_mode=mode, limit=10)
            else:
                # Tìm kiếm chuỗi ký tự trên tất cả các cột
                search_all_columns(con, parquet_path, keyword=line, format_mode=mode, limit=5)

        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát. 👋")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Truy vấn dữ liệu file parquet bằng SQL hoặc tìm kiếm chuỗi trong tất cả các cột."
    )
    parser.add_argument(
        "query_or_keyword",
        nargs="?",
        default=None,
        help="Câu truy vấn SQL (nếu bắt đầu bằng SELECT) HOẶC đoạn chuỗi từ khóa cần tìm trong tất cả các cột.",
    )
    parser.add_argument(
        "-f", "--file",
        default="bilingual_zh_vi.parquet",
        help="Đường dẫn file parquet (mặc định: bilingual_zh_vi.parquet)",
    )
    parser.add_argument(
        "--sql",
        type=str,
        default=None,
        help="Thực thi trực tiếp câu truy vấn SQL (bảng: 'data')",
    )
    parser.add_argument(
        "-w", "--where",
        type=str,
        default=None,
        help="Mệnh đề WHERE trong SQL (ví dụ: \"price < 20 AND category = 'bags'\")",
    )
    parser.add_argument(
        "--select",
        type=str,
        default="*",
        help="Danh sách cột cần SELECT (khi dùng với -w, mặc định: '*')",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=10,
        help="Số lượng bản ghi tối đa hiển thị (mặc định: 10, đặt 0 để xem tất cả)",
    )
    parser.add_argument(
        "-x", "--expanded",
        action="store_true",
        help="Hiển thị toàn bộ các cột theo dạng mở rộng (key-value từng dòng)",
    )
    parser.add_argument(
        "-t", "--table",
        action="store_true",
        help="Bắt buộc hiển thị dạng bảng ASCII grid",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Hiển thị danh sách 30 cột và kiểu dữ liệu của bảng",
    )
    parser.add_argument(
        "-s", "--case-sensitive",
        action="store_true",
        help="Phân biệt chữ hoa / chữ thường khi tìm từ khóa (mặc định: không)",
    )
    parser.add_argument(
        "-c", "--columns",
        type=str,
        default=None,
        help="Chỉ định cột cần tìm (cách nhau bởi dấu phẩy)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Xuất kết quả ra file (.csv, .json, .parquet)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ Không tìm thấy file '{args.file}'")
        sys.exit(1)

    con = get_db_connection(args.file)

    format_mode = "auto"
    if args.expanded:
        format_mode = "expanded"
    elif args.table:
        format_mode = "table"

    # 1. Xem schema
    if args.schema:
        show_schema(con, args.file)
        return

    # 2. Thực thi --sql
    if args.sql:
        execute_sql(con, args.sql, format_mode=format_mode, limit=args.limit, output_file=args.output)
        return

    # 3. Thực thi --where
    if args.where:
        execute_where(
            con, args.file,
            select_cols=args.select,
            where_clause=args.where,
            format_mode=format_mode,
            limit=args.limit,
            output_file=args.output,
        )
        return

    # 4. Kiểm tra tham số vị trí query_or_keyword
    arg_query = args.query_or_keyword
    if arg_query:
        arg_stripped = arg_query.strip()
        # Nếu bắt đầu bằng cú pháp SQL
        if re.match(r"^(select|with|describe|show|explain)\b", arg_stripped.lower()):
            execute_sql(con, arg_stripped, format_mode=format_mode, limit=args.limit, output_file=args.output)
        else:
            cols = [c.strip() for c in args.columns.split(",")] if args.columns else None
            search_all_columns(
                con, args.file,
                keyword=arg_stripped,
                columns=cols,
                case_sensitive=args.case_sensitive,
                format_mode=format_mode,
                limit=args.limit,
                output_file=args.output,
            )
        return

    # 5. Nếu không truyền gì, vào console tương tác
    interactive_console(con, args.file)


if __name__ == "__main__":
    main()
