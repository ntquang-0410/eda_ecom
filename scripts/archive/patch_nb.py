# -*- coding: utf-8 -*-
"""Rewrite section 5 (cells 12-14) and section 7 (cells 17-18) of the EDA notebook.
Sections 0-4 and 6 are kept byte-for-byte, outputs included."""
import json, sys
sys.path.insert(0, r"C:\Users\ADMIN\AppData\Local\Temp\claude\d--nam4-ky1-AI-4-IOT-D4\dee1c3c9-eb48-4d7f-8a4a-9dcbb11cc25f\scratchpad")
import new_cells as N

path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8"))
cells = nb["cells"]

def src(c):
    return "".join(c["source"])

# Guard: make sure the layout is the one this patch was written against.
assert src(cells[12]).startswith("## 5. Similarity"), src(cells[12])[:40]
assert src(cells[13]).startswith("# --- Nạp bge-m3"), src(cells[13])[:40]
assert src(cells[14]).startswith("# --- Tính similarity cho tiêu đề"), src(cells[14])[:40]
assert src(cells[15]).startswith("## 6."), src(cells[15])[:40]
assert src(cells[17]).startswith("## 7."), src(cells[17])[:40]
assert len(cells) == 19

def lines(text):
    parts = text.splitlines(keepends=True)
    if parts and parts[-1].endswith("\n"):
        parts[-1] = parts[-1][:-1]
    return parts

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}

new = (
    cells[:12]
    + [md(N.MD_5), code(N.CODE_ENCODE),
       md(N.MD_51), code(N.CODE_51),
       md(N.MD_52), code(N.CODE_52),
       md(N.MD_53), code(N.CODE_53),
       md(N.MD_54), code(N.CODE_54),
       md(N.MD_55), code(N.CODE_55)]
    + cells[15:17]                      # section 6 unchanged
    + [md(N.MD_7), code(N.CODE_7)]
)
nb["cells"] = new
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"cells: 19 -> {len(new)}")
