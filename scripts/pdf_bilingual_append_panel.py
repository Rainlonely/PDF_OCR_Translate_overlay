#!/usr/bin/env python3
"""
Create bilingual PDF pages by keeping original page on top and adding
a Chinese translation panel below it on the same page.

This is resilient for dense drawings/specs where in-place overlay has no room.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict

import fitz


def parse_translation_md(md_path: Path) -> Dict[int, str]:
    text = md_path.read_text(encoding="utf-8")
    page_map: Dict[int, str] = {}

    page_re = re.compile(r"^## 第 (\d+) 页\s*$", re.M)
    matches = list(page_re.finditer(text))
    for idx, m in enumerate(matches):
        page_no = int(m.group(1))
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end]
        m_zh = re.search(r"### 中文翻译\s*```text\s*(.*?)\s*```", chunk, re.S)
        if not m_zh:
            continue
        page_map[page_no] = clean_translation_text(m_zh.group(1).strip())
    return page_map


def clean_translation_text(text: str) -> str:
    """Drop obvious OCR-garbage lines (many '?', replacement chars, etc.)."""
    cleaned = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            cleaned.append("")
            continue
        q_ratio = s.count("?") / max(1, len(s))
        bad_ratio = (s.count("�") + s.count("□")) / max(1, len(s))
        if q_ratio >= 0.35 or bad_ratio >= 0.2:
            continue
        cleaned.append(ln)
    out = "\n".join(cleaned).strip()
    return out if out else text


def estimate_panel_height(text: str, width: float, font_size: float, margin: float) -> float:
    usable_w = max(20.0, width - margin * 2)
    chars_per_line = max(12, int(usable_w / max(1.0, font_size * 0.55)))
    lines = max(1, math.ceil(len(text) / chars_per_line))
    explicit_breaks = text.count("\n")
    line_count = lines + explicit_breaks
    return margin * 2 + line_count * (font_size * 1.3) + 12


def render_page_with_panel(
    src_doc: fitz.Document,
    page_index: int,
    zh_text: str,
    out_path: Path,
    font_size: float = 7.0,
):
    src_page = src_doc[page_index]
    w = src_page.rect.width
    h = src_page.rect.height
    margin = 18.0
    zh_font = "/System/Library/Fonts/Hiragino Sans GB.ttc"

    panel_h = estimate_panel_height(zh_text, w, font_size, margin)
    total_h = h + panel_h

    out_doc = fitz.open()
    page = out_doc.new_page(width=w, height=total_h)

    # Keep original page untouched at top.
    page.show_pdf_page(fitz.Rect(0, 0, w, h), src_doc, page_index)

    # Translation panel.
    panel_top = h
    page.draw_rect(
        fitz.Rect(0, panel_top, w, total_h),
        color=None,
        fill=(0.97, 0.97, 0.97),
        overlay=False,
    )
    page.draw_line((0, panel_top), (w, panel_top), color=(0.35, 0.35, 0.35), width=0.8)
    page.insert_text(
        fitz.Point(margin, panel_top + 12),
        "中文翻译（本页）",
        fontsize=8.2,
        color=(0.1, 0.1, 0.1),
        fontfile=zh_font,
        fontname="ZHLabel",
    )
    textbox = fitz.Rect(margin, panel_top + 24, w - margin, total_h - margin)
    ret = page.insert_textbox(
        textbox,
        zh_text,
        fontsize=font_size,
        color=(0.12, 0.12, 0.12),
        lineheight=1.22,
        align=fitz.TEXT_ALIGN_LEFT,
        fontfile=zh_font,
        fontname="ZHBody",
    )

    # Retry with slightly smaller font if overflow occurs.
    if ret < 0:
        page = out_doc[-1]
        page.draw_rect(textbox, color=None, fill=(1, 1, 1), overlay=True)
        page.insert_textbox(
            textbox,
            zh_text,
            fontsize=5.6,
            color=(0.12, 0.12, 0.12),
            lineheight=1.2,
            align=fitz.TEXT_ALIGN_LEFT,
            fontfile=zh_font,
            fontname="ZHBodySmall",
        )

    out_doc.save(str(out_path), deflate=True)
    out_doc.close()


def merge_pages(page_dir: Path, total_pages: int, output_pdf: Path):
    out = fitz.open()
    for i in range(1, total_pages + 1):
        part = page_dir / f"page_{i:03d}.pdf"
        if not part.exists():
            raise FileNotFoundError(f"missing page file: {part}")
        src = fitz.open(str(part))
        out.insert_pdf(src)
        src.close()
    out.save(str(output_pdf), garbage=4, deflate=True, use_objstms=1)
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-pdf", required=True)
    ap.add_argument("--translation-md", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--output-pdf", required=True)
    args = ap.parse_args()

    input_pdf = Path(args.input_pdf).expanduser().resolve()
    md_path = Path(args.translation_md).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    output_pdf = Path(args.output_pdf).expanduser().resolve()
    page_dir = work_dir / "pages"
    state_dir = work_dir / "state"
    page_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    progress_file = state_dir / "progress.json"
    progress = {"done_pages": []}
    if progress_file.exists():
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
    done = set(progress.get("done_pages", []))

    zh_map = parse_translation_md(md_path)
    doc = fitz.open(str(input_pdf))
    total = doc.page_count

    for i in range(1, total + 1):
        out_page = page_dir / f"page_{i:03d}.pdf"
        if out_page.exists() and i in done:
            print(f"skip page {i}/{total}")
            continue
        zh = zh_map.get(i, "[本页未找到翻译文本]")
        print(f"processing page {i}/{total} ...")
        render_page_with_panel(doc, i - 1, zh, out_page)
        done.add(i)
        progress_file.write_text(
            json.dumps({"done_pages": sorted(done)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"done page {i}/{total}")

    doc.close()
    print("merging pages ...")
    merge_pages(page_dir, total, output_pdf)
    print(f"done merged pdf: {output_pdf}")


if __name__ == "__main__":
    main()
