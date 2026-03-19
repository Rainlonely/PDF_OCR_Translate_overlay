#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List

import fitz
from deep_translator import GoogleTranslator
from rapidocr_onnxruntime import RapidOCR


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_list_like(line: str) -> bool:
    s = line.strip()
    return bool(
        re.match(r"^([0-9]+\.)\s+", s)
        or re.match(r"^[a-zA-Z]\)\s+", s)
        or re.match(r"^[•\-]\s+", s)
    )


def should_join(prev: str, cur: str) -> bool:
    if not prev or not cur:
        return False
    p = prev.rstrip()
    c = cur.lstrip()
    if is_list_like(c):
        return False
    if re.search(r"[:;.]$", p):
        return False
    if re.match(r"^[A-Z0-9][A-Z0-9 .()/_-]{0,20}:?$", c):
        return False
    # PDF hard-wrap: next line often starts lowercase/continuation token.
    if re.match(r"^[a-z0-9(]", c):
        return True
    # Split inside sentence where previous line has no end punctuation.
    if not re.search(r"[.!?)]$", p):
        return True
    return False


def repair_line_breaks(text: str) -> str:
    text = normalize_newlines(text)
    paras = re.split(r"\n\s*\n", text)
    fixed_paras: List[str] = []
    for para in paras:
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        merged: List[str] = [lines[0]]
        for ln in lines[1:]:
            if should_join(merged[-1], ln):
                merged[-1] = merged[-1].rstrip() + " " + ln.lstrip()
            else:
                merged.append(ln)
        fixed_paras.append("\n".join(merged))
    merged_text = "\n\n".join(fixed_paras).strip()
    # Fix OCR/PDF glued sentence boundaries like "specification.The".
    merged_text = re.sub(r"([a-z0-9])([.?!:;])([A-Z])", r"\1\2 \3", merged_text)
    return merged_text


def split_translation_chunks(text: str, max_len: int = 1500) -> List[str]:
    paras = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    cur = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        candidate = (cur + "\n\n" + para).strip() if cur else para
        if len(candidate) <= max_len:
            cur = candidate
            continue
        if cur:
            chunks.append(cur)
            cur = ""
        if len(para) <= max_len:
            cur = para
            continue
        # Extra-long paragraph: split by sentence punctuation first.
        parts = re.split(r"(?<=[.!?;:])\s+", para)
        buf = ""
        for part in parts:
            cand = (buf + " " + part).strip() if buf else part
            if len(cand) <= max_len:
                buf = cand
            else:
                if buf:
                    chunks.append(buf)
                buf = part
        if buf:
            cur = buf
    if cur:
        chunks.append(cur)
    return chunks


def clean_zh(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        if s.count("?") / max(1, len(s)) >= 0.4:
            continue
        lines.append(ln)
    return "\n".join(lines).strip() or text


def is_tableish_row(cells: List[Dict[str, float]]) -> bool:
    if len(cells) >= 3:
        return True
    if len(cells) == 2:
        left = cells[0]["text"].strip()
        right = cells[1]["text"].strip()
        if len(left) <= 40 and len(right) <= 60:
            return True
    return False


def merge_ocr_rows(items: List[Dict[str, float]], y_tol: float = 18) -> List[List[Dict[str, float]]]:
    rows: List[List[Dict[str, float]]] = []
    for item in sorted(items, key=lambda it: (it["y0"], it["x0"])):
        if not rows:
            rows.append([item])
            continue
        last = rows[-1]
        last_y = sum(x["y0"] for x in last) / len(last)
        if abs(item["y0"] - last_y) <= y_tol:
            last.append(item)
        else:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda it: it["x0"])
    return rows


def rows_to_rect(rows: List[List[Dict[str, float]]], pad: float = 6.0) -> fitz.Rect:
    cells = [cell for row in rows for cell in row]
    x0 = min(c["x0"] for c in cells) - pad
    y0 = min(c["y0"] for c in cells) - pad
    x1 = max(c["x1"] for c in cells) + pad
    y1 = max(c["y1"] for c in cells) + pad
    return fitz.Rect(x0, y0, x1, y1)


def rects_overlap(a: fitz.Rect, b: fitz.Rect) -> bool:
    return not (a.x1 <= b.x0 or a.x0 >= b.x1 or a.y1 <= b.y0 or a.y0 >= b.y1)


def classify_row(row: List[Dict[str, float]], page_h: float) -> str:
    y0 = min(c["y0"] for c in row)
    if y0 <= page_h * 0.10:
        return "header"
    if y0 >= page_h * 0.90:
        return "footer"
    if is_tableish_row(row):
        return "table"
    return "body"


def merge_block_ocr_rows(result) -> List[List[Dict[str, float]]]:
    items: List[Dict[str, float]] = []
    for box, txt, score in (result or []):
        txt = txt.strip()
        if not txt:
            continue
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        items.append(
            {
                "text": txt,
                "score": score,
                "x0": min(xs),
                "y0": min(ys),
                "x1": max(xs),
                "y1": max(ys),
            }
        )
    return merge_ocr_rows(items, y_tol=16)


def ocr_block_text(
    page: fitz.Page,
    rect: fitz.Rect,
    kind: str,
    ocr: RapidOCR,
    state_dir: Path,
    page_num: int,
    block_num: int,
) -> str:
    clip = fitz.Rect(
        max(0, rect.x0),
        max(0, rect.y0),
        min(page.rect.width, rect.x1),
        min(page.rect.height, rect.y1),
    )
    if clip.width <= 2 or clip.height <= 2:
        return ""
    img_path = state_dir / f"page_{page_num:03d}_block_{block_num:03d}_{kind}.png"
    page.get_pixmap(clip=clip, dpi=260).save(str(img_path))
    result, _ = ocr(str(img_path))
    rows = merge_block_ocr_rows(result)
    if not rows:
        return ""
    if kind == "table":
        return "\n".join(" | ".join(cell["text"] for cell in row) for row in rows)
    return "\n".join(" ".join(cell["text"] for cell in row) for row in rows)


def build_structured_text_from_ocr(
    page: fitz.Page,
    img_path: Path,
    ocr: RapidOCR,
    state_dir: Path,
    page_num: int,
) -> str:
    result, _ = ocr(str(img_path))
    if not result:
        return ""

    page_h = page.rect.height
    page_w = page.rect.width
    items: List[Dict[str, float]] = []
    scale_x = page_w / max(1.0, page.rect.width * 220 / 72)
    scale_y = page_h / max(1.0, page.rect.height * 220 / 72)

    for box, txt, score in result:
        txt = txt.strip()
        if not txt:
            continue
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        items.append(
            {
                "text": txt,
                "score": score,
                "x0": min(xs) * scale_x,
                "y0": min(ys) * scale_y,
                "x1": max(xs) * scale_x,
                "y1": max(ys) * scale_y,
            }
        )

    rows = merge_ocr_rows(items)

    blocks: List[Dict[str, object]] = []
    body_current: List[List[Dict[str, float]]] = []
    body_indent = None
    body_last_y1 = None
    last_kind = None

    def flush_body() -> None:
        nonlocal body_current, body_indent, body_last_y1
        if body_current:
            blocks.append({"kind": "body", "rows": body_current[:]})
        body_current = []
        body_indent = None
        body_last_y1 = None

    for row in rows:
        kind = classify_row(row, page_h)
        row_x0 = min(c["x0"] for c in row)
        row_y0 = min(c["y0"] for c in row)
        row_y1 = max(c["y1"] for c in row)
        row_text = " ".join(c["text"] for c in row).strip()

        if kind != "body":
            flush_body()
            if blocks and blocks[-1]["kind"] == kind:
                prev_rows = blocks[-1]["rows"]  # type: ignore[index]
                prev_last_y1 = max(c["y1"] for c in prev_rows[-1])  # type: ignore[index]
                if row_y0 - prev_last_y1 <= 24:
                    prev_rows.append(row)  # type: ignore[attr-defined]
                else:
                    blocks.append({"kind": kind, "rows": [row]})
            else:
                blocks.append({"kind": kind, "rows": [row]})
            last_kind = kind
            continue

        if not body_current:
            body_current = [row]
            body_indent = row_x0
            body_last_y1 = row_y1
            last_kind = kind
            continue

        gap = row_y0 - (body_last_y1 or row_y0)
        indent_shift = abs(row_x0 - (body_indent or row_x0))
        if gap > 20 or indent_shift > 40 or is_list_like(row_text):
            flush_body()
            body_current = [row]
            body_indent = row_x0
        else:
            body_current.append(row)
        body_last_y1 = row_y1
        last_kind = kind

    flush_body()

    sections: List[str] = []
    for idx, block in enumerate(blocks, 1):
        kind = str(block["kind"])
        rows_in_block = block["rows"]  # type: ignore[index]
        rect = rows_to_rect(rows_in_block)  # type: ignore[arg-type]
        text = ocr_block_text(page, rect, kind, ocr, state_dir, page_num, idx)
        text = normalize_newlines(text)
        if not text:
            continue
        if kind == "header":
            sections.append("[HEADER]\n" + text)
        elif kind == "footer":
            sections.append("[FOOTER]\n" + text)
        elif kind == "table":
            sections.append("[TABLE]\n" + text)
        else:
            sections.append(text)
    return "\n\n".join(sec.strip() for sec in sections if sec.strip())


def build_structured_text_from_native(page: fitz.Page) -> str:
    sections: List[tuple[float, str]] = []
    page_h = page.rect.height
    table_rects: List[fitz.Rect] = []

    if hasattr(page, "find_tables"):
        try:
            tables = page.find_tables().tables
            for table in tables:
                table_rects.append(fitz.Rect(table.bbox))
                rows = table.extract() or []
                table_lines: List[str] = []
                for row in rows:
                    cells = []
                    for cell in row:
                        txt = normalize_newlines(str(cell or ""))
                        cells.append(txt)
                    table_lines.append(" | ".join(cells))
                if table_lines:
                    sections.append((table.bbox[1], "[TABLE]\n" + "\n".join(table_lines)))
        except Exception:
            pass

    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = fitz.Rect(block["bbox"])
        if any(rects_overlap(bbox, t) for t in table_rects):
            continue
        lines = []
        for ln in block.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            txt = "".join(s.get("text", "") for s in spans).strip()
            if txt:
                lines.append(txt)
        text = normalize_newlines("\n".join(lines))
        if not text:
            continue
        if bbox.y0 <= page_h * 0.10:
            sections.append((bbox.y0, "[HEADER]\n" + text))
        elif bbox.y0 >= page_h * 0.90:
            sections.append((bbox.y0, "[FOOTER]\n" + text))
        else:
            sections.append((bbox.y0, text))

    sections.sort(key=lambda item: item[0])
    return "\n\n".join(text for _, text in sections)


def translate_text(text: str, tr: GoogleTranslator, cache: Dict[str, str]) -> str:
    key = "MDv2::" + text
    if key in cache:
        return cache[key]
    chunks = split_translation_chunks(text)
    out: List[str] = []
    for ch in chunks:
        translated = None
        for _ in range(4):
            try:
                translated = tr.translate(ch)
                break
            except Exception:
                time.sleep(0.8)
        out.append(translated if translated else "[翻译失败]\n" + ch)
    zh = clean_zh("\n\n".join(out))
    cache[key] = zh
    return zh


def parse_md_sections(md_text: str) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []
    page_matches = list(re.finditer(r"^## 第 (\d+) 页\s*$", md_text, re.M))
    for i, m in enumerate(page_matches):
        start = m.end()
        end = page_matches[i + 1].start() if i + 1 < len(page_matches) else len(md_text)
        chunk = md_text[start:end]
        en_m = re.search(r"### 原文（修复断句后）\s*```text\s*(.*?)\s*```", chunk, re.S)
        zh_m = re.search(r"### 中文翻译\s*```text\s*(.*?)\s*```", chunk, re.S)
        sections.append(
            {
                "page": m.group(1),
                "en": (en_m.group(1).strip() if en_m else ""),
                "zh": (zh_m.group(1).strip() if zh_m else ""),
            }
        )
    return sections


def split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paras:
        return paras
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines


def render_html(md_text: str) -> str:
    title = "双语对照"
    m_title = re.search(r"^# (.+)$", md_text, re.M)
    if m_title:
        title = m_title.group(1).strip()
    sections = parse_md_sections(md_text)

    out: List[str] = []
    out.append("<!doctype html>")
    out.append("<html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>")
    out.append(f"<title>{html.escape(title)}</title>")
    out.append(
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,PingFang SC,Hiragino Sans GB,Microsoft YaHei,sans-serif;max-width:1080px;margin:28px auto;padding:0 16px;line-height:1.55;color:#1f2937;background:#fff}"
        "h1{margin:0 0 14px} h2{margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid #e5e7eb}"
        ".pair{border:1px solid #e5e7eb;border-radius:10px;margin:10px 0;overflow:hidden}"
        ".en{background:#f8fafc;padding:10px 12px;border-bottom:1px dashed #e5e7eb}"
        ".zh{background:#fff7ed;padding:10px 12px;position:relative}"
        ".tag{display:inline-block;font-size:12px;padding:1px 6px;border-radius:999px;margin-bottom:6px}"
        ".tag-en{background:#e2e8f0;color:#334155}.tag-zh{background:#fee2e2;color:#991b1b}"
        ".copy-btn{position:absolute;right:10px;top:10px;border:1px solid #fca5a5;background:#fff;color:#b91c1c;border-radius:8px;padding:4px 8px;cursor:pointer;font-size:12px}"
        ".copy-btn:hover{background:#fef2f2}"
        ".txt{white-space:pre-wrap;word-break:break-word;padding-right:72px}"
        ".ok{color:#166534;font-size:12px;margin-left:6px;display:none}"
        "</style>"
    )
    out.append("</head><body>")
    out.append(f"<h1>{html.escape(title)}</h1>")

    for sec in sections:
        out.append(f"<h2>第 {html.escape(sec['page'])} 页</h2>")
        en_paras = split_paragraphs(sec["en"])
        zh_paras = split_paragraphs(sec["zh"])
        n = max(len(en_paras), len(zh_paras))
        for i in range(n):
            en = en_paras[i] if i < len(en_paras) else ""
            zh = zh_paras[i] if i < len(zh_paras) else ""
            zh_id = f"zh_{sec['page']}_{i}"
            ok_id = f"ok_{sec['page']}_{i}"
            out.append("<div class='pair'>")
            out.append("<div class='en'>")
            out.append("<span class='tag tag-en'>EN</span>")
            out.append(f"<div class='txt'>{html.escape(en)}</div>")
            out.append("</div>")
            out.append("<div class='zh'>")
            out.append("<span class='tag tag-zh'>中文</span>")
            out.append(f"<button class='copy-btn' onclick=\"copyText('{zh_id}','{ok_id}')\">📋 复制</button><span id='{ok_id}' class='ok'>已复制</span>")
            out.append(f"<div id='{zh_id}' class='txt'>{html.escape(zh)}</div>")
            out.append("</div>")
            out.append("</div>")

    out.append(
        "<script>"
        "function copyText(id,okId){"
        "const el=document.getElementById(id); if(!el)return;"
        "const t=el.innerText||el.textContent||'';"
        "navigator.clipboard.writeText(t).then(()=>{"
        "const ok=document.getElementById(okId); if(ok){ok.style.display='inline'; setTimeout(()=>ok.style.display='none',1200);}"
        "});"
        "}"
        "</script>"
    )
    out.append("</body></html>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-pdf", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-html", required=True)
    ap.add_argument("--state-dir", required=True)
    args = ap.parse_args()

    input_pdf = Path(args.input_pdf).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve()
    state_dir = Path(args.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    cache_file = state_dir / "translation_cache.json"
    cache: Dict[str, str] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))

    tr = GoogleTranslator(source="auto", target="zh-CN")
    ocr = RapidOCR()
    doc = fitz.open(str(input_pdf))

    parts: List[str] = []
    parts.append(f"# {input_pdf.stem} 双语对照\n")
    parts.append("说明：先进行断句/断行修复，再给出中文翻译。\n")

    for i, page in enumerate(doc, 1):
        raw = page.get_text("text") or ""
        raw = normalize_newlines(raw)
        if raw:
            raw = build_structured_text_from_native(page)
            raw = normalize_newlines(raw)
        else:
            # OCR fallback for scanned PDFs with layout reconstruction.
            pix = page.get_pixmap(dpi=220)
            img_path = state_dir / f"page_{i:03d}.png"
            pix.save(str(img_path))
            raw = build_structured_text_from_ocr(page, img_path, ocr, state_dir, i)
            raw = normalize_newlines(raw)
        fixed = repair_line_breaks(raw) if raw else "[本页未提取到可用文本]"
        zh = translate_text(fixed, tr, cache) if fixed else "[无可翻译文本]"

        parts.append(f"## 第 {i} 页\n")
        parts.append("### 原文（修复断句后）\n")
        parts.append("```text\n" + fixed + "\n```\n")
        parts.append("### 中文翻译\n")
        parts.append("```text\n" + zh + "\n```\n")

    md_text = "\n".join(parts)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md_text, encoding="utf-8")
    output_html.write_text(render_html(md_text), encoding="utf-8")
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", output_md)
    print("written", output_html)


if __name__ == "__main__":
    main()
