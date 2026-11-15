# -*- coding: utf-8 -*-
"""Lưu vài trang chi tiết 1688 thật (HTML đã dựng + ảnh chụp) để viết bộ chọn DOM cho extension.
Dùng profile Chrome đã đăng nhập của crawler (.pw_profile_zh), mở có cửa sổ, nghỉ 4-8 giây giữa các trang.
Gặp tường CAPTCHA/đăng nhập thì DỪNG (không tự giải).

  ecom_crawler\\.venv\\Scripts\\python.exe -X utf8 demo/lay_mau_trang.py URL [URL ...]
Ra: demo/mau_trang/<offer_id>.html, <offer_id>.png"""
import asyncio
import random
import re
import sys
from pathlib import Path

from playwright.async_api import async_playwright

PROFILE = Path(r"d:\nam4_ky1\e-commerce - crawl\ecom_crawler\.pw_profile_zh")
RA = Path(__file__).resolve().parent / "mau_trang"
TUONG = ("login.taobao.com", "login.1688.com", "/punish", "_____tmd_____")


async def main(urls):
    RA.mkdir(exist_ok=True)
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(PROFILE), channel="chrome", headless=False, viewport={"width": 1366, "height": 800}, locale="zh-CN",
            extra_http_headers={"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5"}, args=["--disable-blink-features=AutomationControlled"])
        await ctx.add_cookies([{"name": "oversealanguage", "value": "zh", "domain": ".1688.com", "path": "/"}])
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        for u in urls:
            oid = re.search(r"/offer/(\d+)", u).group(1)
            await page.goto(u, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(4)
            html = await page.content()
            if any(m in page.url for m in TUONG) or "滑块" in html or "sd/punish" in html[:20000]:
                await page.screenshot(path=str(RA / f"tuong_{oid}.png"))
                print("TUONG", oid, len(html), page.url[:100], "- chờ người xử lý tối đa 240 giây trong cửa sổ Chrome", flush=True)
                for _ in range(80):
                    await asyncio.sleep(3)
                    html = await page.content()
                    if not (any(m in page.url for m in TUONG) or "滑块" in html or "sd/punish" in html[:20000]):
                        break
                else:
                    print("KHONG_QUA_TUONG", oid, flush=True)
                    break
                await asyncio.sleep(2)
                await page.goto(u, wait_until="domcontentloaded", timeout=60000)
                await asyncio.sleep(4)
            for _ in range(6):          # cuộn để các khối tải chậm hiện ra
                await page.mouse.wheel(0, 900)
                await asyncio.sleep(0.8)
            html = await page.content()
            (RA / f"{oid}.html").write_text(html, encoding="utf-8")
            await page.screenshot(path=str(RA / f"{oid}.png"), full_page=False)
            print("LUU", oid, len(html), page.url[:80], flush=True)
            await asyncio.sleep(random.uniform(4, 8))
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
