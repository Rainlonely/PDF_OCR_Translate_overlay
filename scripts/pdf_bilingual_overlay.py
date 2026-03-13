#!/usr/bin/env python3
"""
Overlay Chinese translation under original text blocks in a PDF.

Design goals:
- Page-by-page generation to avoid losing progress on interruption.
- Resume-safe: skips already generated page PDFs.
- Translation cache persisted to disk for repeated runs/documents.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from deep_translator import GoogleTranslator


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_chunked(text: str, translator: GoogleTranslator, cache: Dict[str, str]) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    if text in cache:
        return cache[text]

    paras = re.split(r"\n\n+", text)
    chunks: List[str] = []
    cur = ""
    for p in paras:
        p = p.strip("\n")
        if not p:
            continue
        cand = (cur + "\n\n" + p).strip() if cur else p
        if len(cand) <= 1800:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= 1800:
                cur = p
            else:
                lines = p.split("\n")
                buf = ""
                for line in lines:
                    c2 = (buf + "\n" + line).strip() if buf else line
                    if len(c2) <= 1200:
                        buf = c2
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = line
                cur = buf
    if cur:
        chunks.append(cur)

    out_parts: List[str] = []
    for ch in chunks:
        translated = None
        for _ in range(4):
            try:
                translated = translator.translate(ch)
                break
            except Exception:
                time.sleep(1.0)
        if translated is None:
            translated = "[翻译失败]\n" + ch
        out_parts.append(translated)

    result = clean_translation_text("\n\n".join(out_parts))
    cache[text] = result
    return result


def clean_translation_text(text: str) -> str:
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


def repair_for_translation(text: str) -> str:
    # Normalize hard wraps and fix glued punctuation boundaries, e.g. "specification.The".
    t = normalize_text(text)
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"([a-z0-9])([.?!:;])([A-Z])", r"\1\2 \3", t)
    return t.strip()


def is_meta_only(text: str) -> bool:
    s = text.strip()
    low = s.lower()
    if any(k in low for k in ["doc. no", "doc no", "document no", "rev.", "rev ", "page ", "date "]):
        return True
    if re.fullmatch(r"page\s+\d+\s+of\s+\d+", low):
        return True
    if re.fullmatch(r"date\s+\d{1,2}/\d{1,2}/\d{2,4}", low):
        return True
    if re.fullmatch(r"[A-Z0-9_\-.]{8,}", s):
        return True
    return False


def intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
    return not (a.x1 <= b.x0 or a.x0 >= b.x1 or a.y1 <= b.y0 or a.y0 >= b.y1)


def extract_text_blocks(page: fitz.Page) -> List[Tuple[fitz.Rect, str, float]]:
    """Return list of (bbox, block_text, min_font_size)."""
    data = page.get_text("dict")
    blocks: List[Tuple[fitz.Rect, str, float]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = fitz.Rect(block["bbox"])
        lines = block.get("lines", [])
        text_lines: List[str] = []
        min_font = 9.0
        for ln in lines:
            spans = ln.get("spans", [])
            if not spans:
                continue
            ln_text = "".join(sp.get("text", "") for sp in spans).strip()
            if ln_text:
                text_lines.append(ln_text)
            for sp in spans:
                size = float(sp.get("size", 9.0))
                if size > 0:
                    min_font = min(min_font, size)
        block_text = normalize_text("\n".join(text_lines))
        if not block_text:
            continue
        # Skip very short artifacts (single symbols / line numbers).
        if len(block_text) < 3:
            continue
        blocks.append((bbox, block_text, min_font))
    return blocks


def estimate_box_height(text: str, width: float, font_size: float) -> float:
    chars_per_line = max(8, int(width / max(1.0, font_size * 0.55)))
    approx_lines = max(1, math.ceil(len(text) / chars_per_line))
    return approx_lines * (font_size * 1.35) + 4


def overlay_page(
    src_doc: fitz.Document,
    page_index: int,
    out_page_pdf: Path,
    translator: GoogleTranslator,
    cache: Dict[str, str],
    color: Tuple[float, float, float],
    dx: float,
    dy: float,
):
    single = fitz.open()
    single.insert_pdf(src_doc, from_page=page_index, to_page=page_index)
    page = single[0]
    page_h = page.rect.height
    zh_font = "/System/Library/Fonts/Hiragino Sans GB.ttc"

    blocks = extract_text_blocks(page)
    blocks.sort(key=lambda x: x[0].y0)
    placed_rects: List[fitz.Rect] = []
    overflow_entries: List[str] = []

    def valid_box(box: fitz.Rect) -> bool:
        vals = [box.x0, box.y0, box.x1, box.y1]
        if not all(math.isfinite(v) for v in vals):
            return False
        return box.width > 1.0 and box.height > 1.0

    # Move header-title translations to the top blank area.
    header_lines: List[str] = []
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = fitz.Rect(block["bbox"])
        if bbox.y0 >= 170:
            continue
        for ln in block.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            txt = normalize_text("".join(s.get("text", "") for s in spans))
            if len(txt) < 8 or is_meta_only(txt):
                continue
            if any(
                k in txt.lower()
                for k in [
                    "specification",
                    "e&c onshore",
                    "urea plant",
                    "laboratory examination",
                    "material certification",
                ]
            ):
                header_lines.append(txt)
    # De-dup preserving order
    header_lines = list(dict.fromkeys(header_lines))
    top_y = 8.0
    for txt in header_lines:
        zh = translate_chunked(txt, translator, cache)
        if not zh.strip():
            continue
        text_use = zh if len(zh) <= 95 else (zh[:95] + "…")
        box = fitz.Rect(20, top_y, page.rect.width - 20, top_y + 10)
        if not valid_box(box):
            continue
        ret = page.insert_textbox(
            box,
            text_use,
            fontsize=6.0,
            color=color,
            lineheight=1.02,
            align=fitz.TEXT_ALIGN_LEFT,
            fontfile=zh_font,
            fontname="ZHTop",
        )
        if ret >= 0:
            placed_rects.append(box)
            top_y += 10

    def next_para_y(idx: int, y1: float) -> float:
        for j in range(idx + 1, len(blocks)):
            b2 = blocks[j][0]
            # Adjacent fragments should be treated as same paragraph.
            if b2.y0 <= y1 + 10 and abs(b2.x0 - blocks[idx][0].x0) < 24:
                continue
            return b2.y0
        return page_h - 4

    for i, (bbox, original, orig_font) in enumerate(blocks):
        # Only skip true meta/header blocks, do not skip whole top region.
        lower = original.lower()
        header_like = (
            bbox.y0 < 170
            and (
                is_meta_only(original)
                or any(
                    k in lower
                    for k in [
                        "specification",
                        "e&c onshore",
                        "urea plant",
                        "laboratory examination",
                        "material certification",
                    ]
                )
            )
        )
        if header_like:
            continue
        zh = translate_chunked(repair_for_translation(original), translator, cache)
        if not zh.strip():
            continue

        text_long = zh
        text_short = zh
        x0, y0, x1, y1 = bbox

        fs_max = max(5.0, min(10.0, orig_font))
        fs_min = 4.0
        font_sizes: List[float] = []
        s = fs_max
        while s >= fs_min - 1e-6:
            font_sizes.append(round(s, 2))
            s -= 0.35

        placed = False

        # Priority 1: below paragraph, full-width, searching deeper downward.
        ny = next_para_y(i, y1)
        bx0, bx1 = 24.0, page.rect.width - 24.0
        base_y0 = y1 + 2.5
        max_y0 = min(ny - 6, page_h - 24)
        if max_y0 > base_y0:
            for fs in font_sizes:
                ty0 = base_y0
                while ty0 <= max_y0:
                    box = fitz.Rect(bx0, ty0, bx1, min(page_h - 4, ty0 + 24))
                    if not valid_box(box):
                        ty0 += 3
                        continue
                    if not any(intersects(box, r) for r in placed_rects):
                        ret = page.insert_textbox(
                            box,
                            text_long,
                            fontsize=fs,
                            color=color,
                            lineheight=1.04,
                            align=fitz.TEXT_ALIGN_LEFT,
                            fontfile=zh_font,
                            fontname="ZHBelow",
                        )
                        if ret >= 0:
                            placed_rects.append(box)
                            placed = True
                            break
                    ty0 += 3
                if placed:
                    break

        # Priority 2: left blank column (wrapped text).
        if not placed:
            lx0, lx1 = 18.0, 150.0
            ly_base = max(y0 - 2, 170.0)
            ly_max = min(ny + 30, page_h - 30)
            left_sizes = [min(6.0, fs_max), 5.6, 5.2, 4.8, 4.4, 4.0]
            for fs in left_sizes:
                ly0 = ly_base
                while ly0 < ly_max:
                    box = fitz.Rect(lx0, ly0, lx1, min(page_h - 4, ly0 + 34))
                    if not valid_box(box):
                        ly0 += 4
                        continue
                    if not any(intersects(box, r) for r in placed_rects):
                        ret = page.insert_textbox(
                            box,
                            text_short,
                            fontsize=fs,
                            color=color,
                            lineheight=1.03,
                            align=fitz.TEXT_ALIGN_LEFT,
                            fontfile=zh_font,
                            fontname="ZHLeft",
                        )
                        if ret >= 0:
                            placed_rects.append(box)
                            placed = True
                            break
                    ly0 += 4
                if placed:
                    break

        # Priority 3: overlay as last resort (no background), still non-overlap.
        if not placed:
            oy0 = max(1, y0 + 1)
            oy1 = min(page_h - 1, y1 + 16)
            for fs in font_sizes:
                for ddx, ddy in [(dx, dy), (4, 2), (6, 3), (-2, 2), (8, 0)]:
                    box = fitz.Rect(
                        max(1, x0 + ddx),
                        max(1, oy0 + ddy),
                        min(page.rect.width - 2, x1 + 12 + ddx),
                        min(page_h - 1, oy1 + ddy),
                    )
                    if not valid_box(box):
                        continue
                    if any(intersects(box, r) for r in placed_rects):
                        continue
                    ret = page.insert_textbox(
                        box,
                        text_short,
                        fontsize=fs,
                        color=color,
                        lineheight=1.02,
                        align=fitz.TEXT_ALIGN_LEFT,
                        fontfile=zh_font,
                        fontname="ZHOverlay",
                    )
                    if ret >= 0:
                        placed_rects.append(box)
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            # Keep full translation for bottom overflow panel.
            overflow_entries.append(zh)
    if not overflow_entries:
        single.save(str(out_page_pdf), deflate=True)
        single.close()
        return

    # Build an overflow panel at page bottom (or append page height if needed).
    panel_lines = [f"[{i+1}] {t}" for i, t in enumerate(overflow_entries)]
    panel_text = "\n".join(panel_lines)
    panel_font = 5.2
    margin = 14.0
    panel_h = min(220.0, max(70.0, 18.0 + 10.0 * len(panel_lines)))
    bottom_box = fitz.Rect(margin, page_h - panel_h, page.rect.width - margin, page_h - 6)

    # If bottom panel intersects many existing inserts, append extra canvas below.
    conflict = any(intersects(bottom_box, r) for r in placed_rects)
    if conflict:
        ext_doc = fitz.open()
        new_page = ext_doc.new_page(width=page.rect.width, height=page_h + panel_h)
        new_page.show_pdf_page(fitz.Rect(0, 0, page.rect.width, page_h), single, 0)
        panel_box = fitz.Rect(margin, page_h + 4, page.rect.width - margin, page_h + panel_h - 6)
        new_page.draw_line((0, page_h + 2), (page.rect.width, page_h + 2), color=(0.75, 0.75, 0.75), width=0.6)
        new_page.insert_textbox(
            panel_box,
            "以下为本页补充翻译（完整）\n" + panel_text,
            fontsize=panel_font,
            color=color,
            lineheight=1.12,
            align=fitz.TEXT_ALIGN_LEFT,
            fontfile=zh_font,
            fontname="ZHOverflow",
        )
        ext_doc.save(str(out_page_pdf), garbage=4, deflate=True, use_objstms=1)
        ext_doc.close()
    else:
        page.draw_line((margin, page_h - panel_h - 2), (page.rect.width - margin, page_h - panel_h - 2), color=(0.75, 0.75, 0.75), width=0.6)
        page.insert_textbox(
            bottom_box,
            "补充翻译：\n" + panel_text,
            fontsize=panel_font,
            color=color,
            lineheight=1.12,
            align=fitz.TEXT_ALIGN_LEFT,
            fontfile=zh_font,
            fontname="ZHOverflowInline",
        )
        single.save(str(out_page_pdf), garbage=4, deflate=True, use_objstms=1)
    single.close()


def merge_pages(page_dir: Path, total_pages: int, output_pdf: Path):
    out = fitz.open()
    for i in range(1, total_pages + 1):
        part = page_dir / f"page_{i:03d}.pdf"
        if not part.exists():
            raise FileNotFoundError(f"Missing page PDF: {part}")
        src = fitz.open(str(part))
        out.insert_pdf(src)
        src.close()
    out.save(str(output_pdf), garbage=4, deflate=True, use_objstms=1)
    out.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pdf", required=True)
    parser.add_argument("--work-dir", required=True, help="Folder for per-page outputs and state")
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--start-page", type=int, default=1, help="1-based")
    parser.add_argument("--end-page", type=int, default=0, help="1-based inclusive; 0 means last page")
    parser.add_argument("--color-r", type=float, default=0.86)
    parser.add_argument("--color-g", type=float, default=0.18)
    parser.add_argument("--color-b", type=float, default=0.18)
    parser.add_argument("--offset-x", type=float, default=4.0)
    parser.add_argument("--offset-y", type=float, default=2.0)
    args = parser.parse_args()

    input_pdf = Path(args.input_pdf).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    output_pdf = Path(args.output_pdf).expanduser().resolve()
    page_dir = work_dir / "pages"
    state_dir = work_dir / "state"
    page_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    cache_file = state_dir / "translation_cache.json"
    progress_file = state_dir / "progress.json"

    cache: Dict[str, str] = load_json(cache_file, {})
    progress = load_json(progress_file, {"done_pages": []})
    done_pages = set(progress.get("done_pages", []))

    doc = fitz.open(str(input_pdf))
    total_pages = doc.page_count
    start = max(1, args.start_page)
    end = total_pages if args.end_page <= 0 else min(total_pages, args.end_page)

    translator = GoogleTranslator(source="auto", target="zh-CN")
    color = (args.color_r, args.color_g, args.color_b)

    for page_no in range(start, end + 1):
        out_page = page_dir / f"page_{page_no:03d}.pdf"
        if out_page.exists() and page_no in done_pages:
            print(f"skip page {page_no}/{total_pages}")
            continue

        print(f"processing page {page_no}/{total_pages} ...")
        overlay_page(
            doc,
            page_no - 1,
            out_page,
            translator,
            cache,
            color=color,
            dx=args.offset_x,
            dy=args.offset_y,
        )
        done_pages.add(page_no)
        save_json(cache_file, cache)
        save_json(progress_file, {"done_pages": sorted(done_pages)})
        print(f"done page {page_no}/{total_pages}")

    doc.close()

    if end == total_pages and start == 1:
        print("merging pages ...")
        merge_pages(page_dir, total_pages, output_pdf)
        print(f"done merged pdf: {output_pdf}")
    else:
        print("partial run finished (no merge for sub-range).")


if __name__ == "__main__":
    main()
