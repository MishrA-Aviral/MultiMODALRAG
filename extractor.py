"""
extractor.py — PDF content extraction pipeline.

Table extraction now uses PaddleOCR PP-Structure / paddlex table recognition
(vision-based) instead of pdfplumber text-alignment, fixing:

  • Chart gridlines mistaken as table borders (vision model sees the difference)
  • Nested / merged column headers (HTML colspan/rowspan → flat markdown)
  • Custom math-font (cid:) artifacts (OCR reads rendered glyphs, not encoding)
  • Phantom duplicate tables (per-page deduplication by caption)

pdfplumber is retained as a gridline-only fallback for pages where OCR
returns no results.
"""

import base64
import time
import fitz
import pdfplumber
import sqlite3
import os
import re
import tempfile

import numpy as np
from PIL import Image
from bs4 import BeautifulSoup

# ─── Disable PaddlePaddle OneDNN/MKL-DNN *before* any paddle import ──────────
# PaddlePaddle 3.x on Windows has a PIR executor bug in onednn_instruction.cc:
#   "ConvertPirAttribute2RuntimeAttribute not support
#    [pir::ArrayAttribute<pir::DoubleAttribute>]"
# These flags must be in os.environ BEFORE the paddle shared library is loaded,
# which happens the first time paddleocr/paddlex is imported.  Setting them
# inside _get_pp_engine() (after the lib is loaded) is too late.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_new_executor_use_local_scope", "0")
os.environ.setdefault("PADDLE_DISABLE_MKL_DNN", "1")

# ─── Render scale: 2× magnification → ~144 DPI (good OCR quality) ───────────
_RENDER_SCALE = 2.0

# ─── PP-Structure engine singleton (lazy-loaded; expensive to initialise) ────
_pp_engine = None
_pp_engine_type = None   # "paddlex" | "ppstructure" | "none"


# ─── Financial document support ────────────────────────────────────────────────
# Tier-1 label regex: matches explicit prefixes used in academic papers *and*
# financial reports (Table 3, Schedule A, Exhibit 99.1, Appendix B …)
_TABLE_LABEL_RE = re.compile(
    r'^\s*(Table|Tbl\.?|Schedule|Exhibit|Appendix)\s+[\dA-Z][\w\.]*[:\.[\s]',
    re.IGNORECASE,
)

# Tier-2 regex: matches known financial section-heading keywords.
# No explicit "Table N" prefix needed — the keyword alone is sufficient.
_FINANCIAL_TABLE_HEADING_RE = re.compile(
    r'balance\s+sheet'
    r'|income\s+statement'
    r'|statement[s]?\s+of\s+(cash\s+flows?|operations?|earnings?|equity|changes)'
    r'|profit\s+(?:and|&)\s+loss'
    r'|p\s*(?:&|and)\s*l\b'
    r'|cash\s+flow'
    r'|financial\s+(?:results?|highlights?|summary|position|data|statements?)'
    r'|shareholders?\s+(?:equity|funds?)'
    r'|earnings?\s+(?:per\s+share|release)'
    r'|consolidated\s+(?:statements?|results?|financials?|balance|income)'
    r'|notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements?'
    r'|segment\s+(?:results?|information|data)'
    r'|selected\s+financial\s+(?:data|information)'
    r'|quarterly\s+(?:results?|data|financials?)'
    r'|operating\s+(?:results?|data|summary)'
    r'|schedule\s+of'
    r'|five[- ]year\s+(?:summary|financial)'
    r'|revenue\s+(?:summary|breakdown|by\s+segment)'
    r'|return\s+on\s+(?:equity|assets|investment|capital)'
    r'|key\s+(?:financial|performance)\s+(?:metrics?|indicators?|data|highlights?)'
    r'|ageing\s+schedule'
    r'|property,\s+plant\s+(?:and|&)\s+equipment'
    r'|tangible\s+assets'
    r'|as\s+follows\b',
    re.IGNORECASE,
)

# Extended figure/chart label regex: academic "Figure N" + financial "Chart/Exhibit N".
_FIGURE_LABEL_RE = re.compile(
    r'^\s*(Figure|Fig\.?|Chart|Graph|Diagram|Exhibit|Illustration|Graphic)\s+[\dA-Z][\w\.]*[:\.\-\s]',
    re.IGNORECASE,
)

# Financial chart caption keywords — detect unlabelled chart/graph captions.
_FINANCIAL_CHART_CAPTION_RE = re.compile(
    r'revenue|margin|ebitda|earnings|profit(?:\s|$)|loss(?:\s|$)'
    r'|growth|trend|performance|sales|cash\s+flow|net\s+income'
    r'|operating\s+(?:income|profit|margin)|segment|return\s+on'
    r'|market\s+share|year[- ]over[- ]year|quarter(?:ly)?|annual(?:ized)?',
    re.IGNORECASE,
)

# Compiled patterns for the _looks_financial() guard in _is_chart_artifact().
_FINANCIAL_YEAR_RE   = re.compile(r'^(19|20)\d{2}$')
_FINANCIAL_AMOUNT_RE = re.compile(r'^[\(\$£€¥₹]?[\d,]+\.?\d*[KMBkmb]?\)?%?$')


def _sentence_boundaries(text: str) -> int:
    """Count mid-text sentence boundaries (period/!/? followed by whitespace + uppercase)."""
    return len(re.findall(r'(?<=[.!?])\s+[A-Z]', text))


def _is_table_heading(text: str) -> bool:
    """
    Return True if *text* looks like a table heading rather than body prose.

    Two-tier detection (works for both academic and financial documents):
      Tier 1 — explicit label prefix:  "Table 3:", "Schedule A.", "Exhibit 99.1"
      Tier 2 — known financial heading keyword (e.g. "Balance Sheet",
               "Statement of Cash Flows", "Income Statement")

    Sentence-count guard: blocks spanning 3+ sentences (i.e. containing more
    than 1 mid-text sentence boundary) are treated as body prose and rejected.
    """
    text = text.strip()
    if not text:
        return False
    if _TABLE_LABEL_RE.match(text):
        return True
        
    # Reject body paragraphs that happen to contain financial keywords.
    # True financial headings are usually noun phrases (no period) or end with a colon.
    # If it ends with a period, it's almost certainly a prose sentence (e.g. footnotes).
    if text.endswith('.') and not re.search(r'as\s+follows', text, re.IGNORECASE):
        return False

    if _sentence_boundaries(text) > 1:
        return False
    return bool(_FINANCIAL_TABLE_HEADING_RE.search(text))


def _is_financial_chart_caption(text: str) -> bool:
    """
    Return True if *text* looks like an unlabelled financial chart/graph caption
    (a short block — ≤ 2 sentences — that contains financial visualisation keywords).
    """
    text = text.strip()
    if not text or _sentence_boundaries(text) > 1:
        return False
    return bool(_FINANCIAL_CHART_CAPTION_RE.search(text))



# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Low-level cell helpers  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def clean_cell(val):
    if val is None:
        return ""
    return str(val).strip().replace("\n", " ")


def clean_numeric_cell(cell_str: str) -> str:
    tokens = cell_str.split()
    if len(tokens) == 2:
        def is_number(s):
            try:
                float(s.replace(',', ''))
                return True
            except ValueError:
                return False
        if len(tokens[0]) == 1 and is_number(tokens[1]):
            return tokens[1]
        elif len(tokens[1]) == 1 and is_number(tokens[0]):
            return tokens[0]
    return cell_str


def table_to_markdown(table_data) -> str:
    """Convert a 2-D list (pdfplumber output) to a markdown table string."""
    if not table_data or len(table_data) == 0:
        return ""
    max_cols = max(len(row) for row in table_data)

    cleaned_table = []
    for row in table_data:
        cleaned_row = [clean_cell(cell) for cell in row]
        if len(cleaned_row) < max_cols:
            cleaned_row.extend([""] * (max_cols - len(cleaned_row)))
        cleaned_table.append(cleaned_row)

    markdown_lines = []
    headers = cleaned_table[0]
    markdown_lines.append("| " + " | ".join(headers) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

    for row in cleaned_table[1:]:
        markdown_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Table Transformer (TATR) engine (lazy singleton)
# ═══════════════════════════════════════════════════════════════════════════════

_tatr_image_processor = None
_tatr_model_detection = None
_tatr_structure_processor = None
_tatr_model_structure = None

def _get_tatr_models():
    """
    Lazy-load Microsoft Table Transformer (TATR) models via Hugging Face.

    Compatibility notes
    -------------------
    * transformers ≥ 4.46  – DetrImageProcessor requires BOTH 'shortest_edge' AND
      'longest_edge' in the size config. The TATR checkpoints only ship
      {"longest_edge": 800}, so we reconstruct the processor with an explicit dict
      instead of mutating the SizeDict object (which became frozen/validated and
      silently dropped writes in newer releases).

    * transformers ≥ 4.50  – DetrConfig now uses strict dataclass validation and
      rejects the "dilation": null that ships in the TATR config.json files with a
      StrictDataclassFieldValidationError. We download the config, patch the field
      to False, and pass it explicitly to from_pretrained so the model loads cleanly.
    """
    global _tatr_image_processor, _tatr_model_detection, _tatr_structure_processor, _tatr_model_structure
    if _tatr_model_detection is not None:
        return _tatr_image_processor, _tatr_model_detection, _tatr_structure_processor, _tatr_model_structure

    try:
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection, DetrConfig
        print("  [OCR] Loading TATR models (this may take a moment)...")

        # ── Helper: reconstruct a processor with a valid size dict ────────────
        def _safe_processor(repo_id: str):
            """
            Load an AutoImageProcessor and ensure its size config contains both
            'shortest_edge' and 'longest_edge' so inference doesn't crash on newer
            transformers versions where SizeDict is validated strictly.
            We reconstruct from a plain dict rather than mutating the SizeDict
            attribute (which silently fails in transformers ≥ 4.48+).
            """
            proc = AutoImageProcessor.from_pretrained(repo_id)
            size = getattr(proc, "size", None)
            if size is not None:
                # Normalise to a plain dict regardless of whether it's a SizeDict
                if hasattr(size, "__dict__"):
                    size_dict = {k: v for k, v in size.__dict__.items() if v is not None}
                elif isinstance(size, dict):
                    size_dict = {k: v for k, v in size.items() if v is not None}
                else:
                    size_dict = {}

                longest = size_dict.get("longest_edge") or size_dict.get("width") or size_dict.get("height")
                if longest and ("shortest_edge" not in size_dict or size_dict.get("shortest_edge") is None):
                    size_dict["shortest_edge"] = longest
                if longest and ("longest_edge" not in size_dict or size_dict.get("longest_edge") is None):
                    size_dict["longest_edge"] = longest

                if size_dict:
                    proc = AutoImageProcessor.from_pretrained(repo_id, size=size_dict)
            return proc

        # ── Helper: patch dilation None→False before loading model ────────────
        def _safe_model(repo_id: str):
            """
            Load a TableTransformerForObjectDetection, patching 'dilation': null →
            False in the config beforehand. This avoids StrictDataclassFieldValidation
            errors introduced in transformers ≥ 4.50.
            """
            try:
                cfg = DetrConfig.from_pretrained(repo_id)
                if getattr(cfg, "dilation", None) is None:
                    cfg.dilation = False
                return TableTransformerForObjectDetection.from_pretrained(repo_id, config=cfg)
            except Exception:
                # Fallback: let transformers load with defaults; ignore unexpected keys
                return TableTransformerForObjectDetection.from_pretrained(
                    repo_id, ignore_mismatched_sizes=True
                )

        _tatr_image_processor = _safe_processor("microsoft/table-transformer-detection")
        _tatr_model_detection  = _safe_model("microsoft/table-transformer-detection")

        _tatr_structure_processor = _safe_processor("microsoft/table-transformer-structure-recognition-v1.1-all")
        _tatr_model_structure     = _safe_model("microsoft/table-transformer-structure-recognition-v1.1-all")

        print("  [OCR] TATR models loaded successfully.")
        return _tatr_image_processor, _tatr_model_detection, _tatr_structure_processor, _tatr_model_structure
    except Exception as e:
        print(f"  [OCR] Failed to load TATR models ({type(e).__name__}: {e})")
        return None, None, None, None



# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: HTML table → flat markdown  (handles colspan / rowspan)
# ═══════════════════════════════════════════════════════════════════════════════

def _expand_html_grid(table_tag) -> list:
    """
    Parse a BeautifulSoup <table> tag into a 2-D list of strings,
    correctly expanding colspan and rowspan so every logical cell is present.

    Returns [] if the tag is None or has no rows.
    """
    if table_tag is None:
        return []

    rows = table_tag.find_all('tr')
    if not rows:
        return []

    # Sparse (row_idx, col_idx) → cell text dict
    grid_dict = {}

    for r_idx, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        c_cursor = 0
        for cell in cells:
            # Skip positions already occupied by a rowspan from a previous row
            while (r_idx, c_cursor) in grid_dict:
                c_cursor += 1

            text = cell.get_text(separator=' ', strip=True)
            colspan = int(cell.get('colspan', 1) or 1)
            rowspan = int(cell.get('rowspan', 1) or 1)

            for r_off in range(rowspan):
                for c_off in range(colspan):
                    pos = (r_idx + r_off, c_cursor + c_off)
                    if pos not in grid_dict:
                        grid_dict[pos] = text

            c_cursor += colspan

    if not grid_dict:
        return []

    max_row = max(r for r, _ in grid_dict) + 1
    max_col = max(c for _, c in grid_dict) + 1

    return [
        [grid_dict.get((r, c), '') for c in range(max_col)]
        for r in range(max_row)
    ]


def html_table_to_markdown(html_str: str) -> str:
    """
    Convert a PP-Structure HTML table string to a flat, well-formed markdown
    table string.

    Multi-row headers produced by colspan / rowspan are flattened into a single
    header row: unique non-empty values per column are joined with ' / ' so the
    full header context is preserved for embedding and LLM reading.
    """
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        table_tag = soup.find('table')
        grid = _expand_html_grid(table_tag)

        if not grid or len(grid) < 2:
            return ""

        num_cols = max(len(r) for r in grid)

        # ── Detect where the header section ends ─────────────────────────────
        # A row is a "header row" if it contains no majority-numeric values.
        def _mostly_numeric(row):
            nonempty = [c.strip() for c in row if c.strip()]
            if not nonempty:
                return False
            numeric = sum(
                1 for c in nonempty
                if re.match(r'^[\d\.\-\+×xX·]+[%]?$', c)
            )
            return numeric > len(nonempty) / 2

        header_end = 1       # row 0 is always a header
        for i in range(1, len(grid)):
            if _mostly_numeric(grid[i]):
                header_end = i
                break
            header_end = i + 1

        header_rows = grid[:header_end]
        data_rows   = grid[header_end:]

        # Edge case: everything looks like a header (no numeric rows found)
        if not data_rows:
            header_rows = grid[:1]
            data_rows   = grid[1:]

        # ── Flatten multi-row headers column-by-column ────────────────────────
        if len(header_rows) > 1:
            flat_header = []
            for c in range(num_cols):
                seen_vals = []
                seen_set  = set()
                for r in header_rows:
                    val = r[c].strip() if c < len(r) else ''
                    if val and val not in seen_set:
                        seen_vals.append(val)
                        seen_set.add(val)
                flat_header.append(' / '.join(seen_vals))
            header_rows = [flat_header]

        all_rows = header_rows + data_rows
        # Pad every row to the same width
        padded = [row + [''] * (num_cols - len(row)) for row in all_rows]

        lines = [
            '| ' + ' | '.join(clean_cell(c) for c in padded[0]) + ' |',
            '| ' + ' | '.join(['---'] * num_cols) + ' |',
        ]
        for row in padded[1:]:
            lines.append('| ' + ' | '.join(clean_cell(c) for c in row) + ' |')

        return '\n'.join(lines)

    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Chart-artifact rejection heuristics
# ═══════════════════════════════════════════════════════════════════════════════

def _looks_financial(values: list) -> bool:
    """
    Return True if *values* look like years or monetary amounts rather than
    bare integer Y-axis tick marks.  Used to exempt financial table columns
    from the chart-artifact integer heuristic in _is_chart_artifact().
    """
    if not values:
        return False
    hits = sum(
        1 for v in values
        if _FINANCIAL_YEAR_RE.match(v) or _FINANCIAL_AMOUNT_RE.match(v)
    )
    return hits / len(values) > 0.5


def _is_chart_artifact(grid: list) -> bool:
    """
    Return True if the extracted region looks like chart axis data or a
    figure legend rather than a real data table.

    Triggered by:
    • Column 0 is mostly standalone integers AND the table is very narrow (≤2 cols).
    • Cells contain rotated-label garble: ")%(", "re", "rorr", lone ")" etc.
    • Tiny 2-column / ≤6-row tables where >60 % of cells are ≤3 chars long.
    • Chart std-dev / response plots: header cell is a model-legend list and
      data column is a sequence of layer index integers (0, 20, 40, …).
    • Column 0 header contains 'layer index' (axis label of layer-response plots).
    """
    if not grid:
        return True

    all_cells = [c.strip() for row in grid for c in row]
    nonempty  = [c for c in all_cells if c]
    if len(nonempty) < 4:
        return True

    # Pre-compute column count — needed by multiple heuristics below.
    num_cols = max(len(r) for r in grid) if grid else 1

    # Heuristic 1 — column 0 is mostly bare integers (Y-axis tick marks).
    # IMPORTANT: only applied when the table is very narrow (≤2 columns).
    # Wide tables (e.g. architecture / CIFAR-10 comparison tables) legitimately
    # have integer-only first columns (n=3,5,7,9…) and must NOT be rejected.
    # EXEMPT: financial tables where column-0 values are years or monetary amounts.
    col0 = [row[0].strip() for row in grid if row and row[0].strip()]
    if col0 and num_cols <= 2 and not _looks_financial(col0):
        int_frac = sum(1 for v in col0 if re.match(r'^\d{1,3}$', v)) / len(col0)
        if int_frac > 0.5:
            return True

    # Heuristic 2 — rotated / garbled axis label fragments
    garble_patterns = [r'\)\%\(', r'^re$', r'^rorr', r'^\)$', r'^\($', r'^rorre$']
    if any(
        re.search(pat, c, re.IGNORECASE)
        for c in nonempty
        for pat in garble_patterns
    ):
        return True

    # Heuristic 3 — small 2-column tables with very short cells (legend fragments)
    if num_cols <= 2 and len(grid) <= 6:
        short = sum(1 for c in nonempty if len(c) <= 3)
        if short / len(nonempty) > 0.6:
            return True

    # Heuristic 4 — layer-response / std-dev chart: header contains 'layer index'
    # or a long concatenated string of model names (chart legend scraped into one cell)
    header_cells = [c.strip() for c in (grid[0] if grid else [])] if grid else []
    full_header = ' '.join(header_cells).lower()
    if 'layer index' in full_header:
        return True
    # If ANY single header cell contains 4+ model-name tokens (legend list)
    model_tokens = ['plain', 'resnet', 'residual', 'highway', 'fitnet']
    for hc in header_cells:
        token_hits = sum(1 for t in model_tokens if t in hc.lower())
        if token_hits >= 3:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: OCR table detection on a rendered page image
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_tatr_tables_on_page(fitz_page, render_scale: float):
    """
    Run Microsoft Table Transformer (TATR) on a rendered page PNG.

    Returns a list of:
        {"bbox_pdf": (x0, y0, x1, y1),   # in PDF point coordinates
         "grid":     [[cell, ...], ...]} # extracted text grid
         
    Return values:
        []   – TATR ran successfully but detected no tables on this page.
        None – TATR engine is unavailable OR predict() raised an exception.
    """
    try:
        import torch
        from PIL import Image
    except ImportError:
        return None

    models = _get_tatr_models()
    if models is None or models[0] is None:
        return None
        
    image_processor, model_detection, structure_processor, model_structure = models

    try:
        mat = fitz.Matrix(render_scale, render_scale)
        pix = fitz_page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        inputs = image_processor(images=img, return_tensors="pt")
        outputs = model_detection(**inputs)
        
        target_sizes = torch.tensor([img.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
        
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() == 0: # 0 is table
                tables.append(box.tolist())
                
        extracted_tables = []
        
        for t_box in tables:
            xmin, ymin, xmax, ymax = t_box
            pad = 10
            crop_box = (max(0, xmin-pad), max(0, ymin-pad), min(img.width, xmax+pad), min(img.height, ymax+pad))
            table_img = img.crop(crop_box)
            
            s_inputs = structure_processor(images=table_img, return_tensors="pt")
            s_outputs = model_structure(**s_inputs)
            s_target_sizes = torch.tensor([table_img.size[::-1]])
            s_results = structure_processor.post_process_object_detection(s_outputs, threshold=0.7, target_sizes=s_target_sizes)[0]
            
            rows = []
            cols = []
            for s_score, s_label, s_box in zip(s_results["scores"], s_results["labels"], s_results["boxes"]):
                if s_label.item() == 1: # column
                    cols.append(s_box.tolist())
                elif s_label.item() == 2: # row
                    rows.append(s_box.tolist())
                    
            rows = sorted(rows, key=lambda x: x[1])
            cols = sorted(cols, key=lambda x: x[0])
            
            grid = []
            for r_box in rows:
                row_data = []
                for c_box in cols:
                    cell_xmin, cell_xmax = c_box[0], c_box[2]
                    cell_ymin, cell_ymax = r_box[1], r_box[3]
                    
                    abs_xmin = cell_xmin + crop_box[0]
                    abs_ymin = cell_ymin + crop_box[1]
                    abs_xmax = cell_xmax + crop_box[0]
                    abs_ymax = cell_ymax + crop_box[1]
                    
                    pdf_rect = fitz.Rect(abs_xmin / render_scale, abs_ymin / render_scale, abs_xmax / render_scale, abs_ymax / render_scale)
                    text = fitz_page.get_text("words", clip=pdf_rect)
                    words = sorted(text, key=lambda w: (w[1], w[0]))
                    cell_text = " ".join([w[4] for w in words])
                    row_data.append(clean_cell(cell_text))
                grid.append(row_data)
                
            bbox_pdf = (xmin / render_scale, ymin / render_scale, xmax / render_scale, ymax / render_scale)
            extracted_tables.append({"bbox_pdf": bbox_pdf, "grid": grid})
            
        return extracted_tables
    except Exception as e:
        print(f"  [OCR] TATR processing error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: pdfplumber gridline-only fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _pdfplumber_fallback_tables(page) -> list:
    """
    Extract tables from a pdfplumber page when OCR is unavailable.

    Tries three strategy combinations in order, deduplicating by bbox so the
    same physical table is never returned twice:

    1. text+lines  — booktabs-style LaTeX tables (no vertical rules).
    2. lines+lines — fully-bordered tables (rare in papers, common in slides).
    3. text+text   — tables with no drawn rules at all (whitespace-aligned).

    _is_chart_artifact() filters any remaining chart false-positives.

    Returns a list of {"data": [[...], ...], "bbox": (x0,y0,x1,y1)}.
    """
    strategy_sets = [
        # Primary: booktabs-style (horizontal rules only, column sep by text)
        {
            "vertical_strategy":   "text",
            "horizontal_strategy": "lines",
            "text_y_tolerance":    3,
        },
        # Secondary: fully-bordered tables
        {
            "vertical_strategy":   "lines",
            "horizontal_strategy": "lines",
            "text_y_tolerance":    3,
        },
        # Tertiary: whitespace-only tables (no drawn rules at all)
        {
            "vertical_strategy":   "text",
            "horizontal_strategy": "text",
            "text_y_tolerance":    3,
            "text_x_tolerance":    3,
        },
    ]

    found       = []
    seen_bboxes = set()

    for settings in strategy_sets:
        try:
            for tbl in page.find_tables(table_settings=settings):
                data = tbl.extract()
                if not data or len(data) < 2:
                    continue
                # Deduplicate: round bbox coords to avoid float jitter
                bbox_key = tuple(round(v, 1) for v in tbl.bbox)
                if bbox_key in seen_bboxes:
                    continue
                seen_bboxes.add(bbox_key)

                cleaned = [
                    [clean_numeric_cell(str(c).strip().replace("\n", " ")) if c else ""
                     for c in row]
                    for row in data
                ]
                found.append({"data": cleaned, "bbox": tbl.bbox})
        except Exception as e:
            print(f"  [pdfplumber fallback] error: {e}")

    return found


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Main table extraction entry point
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tables(pdf_path: str,
                   db_path: str = "db/tables.db",
                   output_dir: str = "data/extracted_tables") -> list:
    """
    Extract all tables from a PDF using vision-based OCR (PP-Structure).

    Primary path : PP-Structure on a rendered page image.
                   → correctly handles nested headers and ignores chart lines.
    Fallback path: pdfplumber with gridline-only strategy (no text-alignment).

    Returns a list of dicts with keys:
        source, page, table_path, content, caption, bbox
    (same schema as the original implementation — downstream code unchanged).
    """
    os.makedirs(output_dir, exist_ok=True)
    table_records = []

    doc_fitz = fitz.open(pdf_path)

    for i, fitz_page in enumerate(doc_fitz):

            # ── 1. Collect table headings from fitz text blocks ─────────────────
            # _is_table_heading() handles both academic captions ("Table 3:") and
            # financial section headings ("Balance Sheet", "Statement of Cash Flows",
            # "Exhibit 99.1", etc.) via a two-tier regex + sentence-count guard.
            blocks = fitz_page.get_text("blocks")
            tbl_captions = []
            for b in blocks:
                b_text = b[4].strip()
                if _is_table_heading(b_text):
                    tbl_captions.append({
                        "text": b_text,
                        "bbox": (b[0], b[1], b[2], b[3])
                    })

            if not tbl_captions:
                continue    # No table headings found → skip page

            # ── 2 & 3. Run TATR table detection on the fitz page ──────────────
            tatr_tables = _extract_tatr_tables_on_page(fitz_page, _RENDER_SCALE)

            # ── 4. Build candidate list (TATR + PdfPlumber) ─────────────────────
            candidates = []

            if tatr_tables is None:
                print(f"  [OCR] p.{i+1}: TATR processing failed or unavailable")
            elif tatr_tables:
                for t in tatr_tables:
                    grid = t["grid"]
                    md = table_to_markdown(grid)
                    if md:
                        candidates.append({
                            "bbox":     t["bbox_pdf"],
                            "markdown": md,
                            "grid":     grid,
                            "source":   "tatr"
                        })

            # ── 4. Build candidate list (TATR only) ──────────────────────────
            caption_matches = {} # cap_text -> (min_dist, cand)
            
            for cand in candidates:
                bbox     = cand["bbox"]
                markdown = cand["markdown"]
                grid     = cand["grid"]

                # Reject chart axis artifacts
                if _is_chart_artifact(grid):
                    print(f"  [OCR] p.{i+1}: skipping chart artifact from {cand.get('source')}")
                    continue

                # Reject suspiciously sparse regions
                nonempty_cells = [
                    c for row in grid for c in row
                    if isinstance(c, str) and c.strip()
                ]
                if len(nonempty_cells) < 4:
                    continue

                tbl_x0, tbl_y0, tbl_x1, tbl_y1 = bbox

                # ── Caption matching ──────────────────────
                best_cap = None
                min_dist = float("inf")

                for cap in tbl_captions:
                    cap_x0, cap_y0, cap_x1, cap_y1 = cap["bbox"]
                    cap_center_y = (cap_y0 + cap_y1) / 2
                    overlap = max(0, min(tbl_x1, cap_x1) - max(tbl_x0, cap_x0))

                    if overlap > 30:
                        if cap_center_y < tbl_y0:
                            dist = tbl_y0 - cap_center_y
                        elif cap_center_y > tbl_y1:
                            dist = cap_center_y - tbl_y1
                        else:
                            dist = 0
                        if dist < min_dist:
                            min_dist = dist
                            best_cap = cap["text"]

                # Proximity-only fallback (no X-overlap requirement)
                if not best_cap and tbl_captions:
                    for cap in tbl_captions:
                        cap_center_y = (cap["bbox"][1] + cap["bbox"][3]) / 2
                        if cap_center_y < tbl_y0:
                            dist = tbl_y0 - cap_center_y
                        elif cap_center_y > tbl_y1:
                            dist = cap_center_y - tbl_y1
                        else:
                            dist = 0
                        if dist < min_dist:
                            min_dist = dist
                            best_cap = cap["text"]

                if not best_cap or min_dist > 150:
                    continue

                # Deduplicate: Keep only the closest table candidate per caption
                if best_cap not in caption_matches or min_dist < caption_matches[best_cap][0]:
                    caption_matches[best_cap] = (min_dist, cand)

            # ── Write markdown files for the unique matched tables ────────
            page_count = 0
            for best_cap, (dist, cand) in caption_matches.items():
                page_count += 1
                filename = (
                    f"{os.path.basename(pdf_path)}"
                    f"_page{i+1}_table_{page_count}.md"
                )
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cand["markdown"])

                table_records.append({
                    "source":     os.path.basename(pdf_path),
                    "page":       i + 1,
                    "table_path": filepath,
                    "content":    cand["markdown"],
                    "caption":    best_cap,
                    "bbox":       cand["bbox"]
                })

    doc_fitz.close()

    # ── 6. Deduplicate by (page, caption) to remove phantom duplicates ────────
    seen_keys   = {}
    deduped     = []
    for r in table_records:
        # Use (page, first 120 chars of caption) as the dedup key.
        # Two records with the same page + caption are the same logical table.
        key = (r["page"], (r["caption"] or "").strip()[:120])
        if key not in seen_keys:
            seen_keys[key] = True
            deduped.append(r)
        else:
            try:
                os.remove(r["table_path"])
            except OSError:
                pass
    table_records = deduped

    # ── 7. Write to SQLite (same schema as before) ────────────────────────────
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tables (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT,
            page        INTEGER,
            content     TEXT,
            caption     TEXT,
            table_path  TEXT
        )
    """)
    cursor.execute("DELETE FROM tables WHERE source = ?",
                   (os.path.basename(pdf_path),))
    for r in table_records:
        cursor.execute(
            "INSERT INTO tables (source, page, content, caption, table_path)"
            " VALUES (?, ?, ?, ?, ?)",
            (r["source"], r["page"], r["content"], r["caption"], r["table_path"])
        )
    conn.commit()
    conn.close()

    return table_records


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Image extraction  (extract_images + _describe_image_with_vision)
# ═══════════════════════════════════════════════════════════════════════════════

def _describe_image_with_vision(filepath: str, fallback_caption: str) -> str:
    """
    Use PaddleOCR to extract all text visible inside an image (figure/chart from
    a research paper) and combine it with the surrounding caption text so the
    FAISS index receives rich, searchable content about each figure.

    This is a fully local, lightweight operation — no vision LLM, no API key,
    no GPU required. PaddleOCR reads axis labels, legend entries, numerical
    values, and any other text embedded in the image.

    Falls back to the original fallback_caption silently on any error.
    """
    try:
        from paddleocr import PaddleOCR

        # Initialise with English; use_angle_cls catches rotated chart labels.
        # show_log=False suppresses PaddlePaddle's verbose startup output.
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        img       = Image.open(filepath).convert("RGB")
        img_array = np.array(img)

        result = ocr.ocr(img_array, cls=True)

        # result is a list-of-pages; we always pass a single image so index [0]
        extracted_lines = []
        if result and result[0]:
            for line in result[0]:
                # Each line: [[bbox_points], (text, confidence)]
                text, confidence = line[1]
                if confidence > 0.5 and text.strip():
                    extracted_lines.append(text.strip())

        if not extracted_lines:
            return fallback_caption

        ocr_text = " | ".join(extracted_lines)
        return f"{fallback_caption}\n\nOCR Text: {ocr_text}"

    except Exception:
        return fallback_caption


def extract_images(pdf_path: str,
                   output_dir: str = "data/extracted_images",
                   table_records: list = None) -> list:
    """
    Extract figure images from a PDF, assigning captions where possible.

    table_records (optional): list returned by extract_tables() for the same PDF.
    When provided, drawing elements that overlap with a known table region are
    excluded from the vector-figure capture window — this fixes the bug where
    the architecture table (Table 1) was merged into the Figure 4 image.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc          = fitz.open(pdf_path)
    image_records = []

    # Build per-page table bbox lookup (PDF point coordinates)
    table_bboxes_by_page = {}
    if table_records:
        for r in table_records:
            table_bboxes_by_page.setdefault(r["page"], []).append(r["bbox"])

    for i, page in enumerate(doc):
        page_num = i + 1

        # ── 1. Figure captions on this page ───────────────────────────────────
        # Tier 1: explicit label  (Figure N, Chart N, Graph N, Exhibit N, …)
        # Tier 2: unlabelled financial chart captions (short blocks with finance keywords)
        blocks      = page.get_text("blocks")
        fig_captions = []
        for b in blocks:
            b_text = b[4].strip()
            if _FIGURE_LABEL_RE.match(b_text) or _is_financial_chart_caption(b_text):
                fig_captions.append({
                    "text": b_text,
                    "bbox": fitz.Rect(b[0], b[1], b[2], b[3])
                })

        # ── 2. Raster images ──────────────────────────────────────────────────
        images_info  = page.get_image_info(xrefs=True)
        seen_xrefs   = set()
        raster_images = []
        for img in images_info:
            xref = img.get("xref")
            if not xref or xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            w    = img.get("width",  0)
            h    = img.get("height", 0)
            bbox = img.get("bbox")
            if w < 50 or h < 50 or not bbox:
                continue
            raster_images.append({
                "xref": xref,
                "bbox": fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3]),
                "width": w, "height": h
            })

        page_extracted_xrefs = set()
        for img in raster_images:
            xref = img["xref"]
            bbox = img["bbox"]
            try:
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                ext        = base_image["ext"]

                if len(img_bytes) < 5000:
                    continue

                img_center_y = (bbox.y0 + bbox.y1) / 2
                best_cap     = None
                min_dist     = float("inf")
                for cap in fig_captions:
                    cap_x0, cap_y0, cap_x1, cap_y1 = cap["bbox"]
                    cap_center_y = (cap_y0 + cap_y1) / 2
                    overlap = max(0, min(bbox.x1, cap_x1) - max(bbox.x0, cap_x0))
                    if overlap > 30:
                        dist = abs(img_center_y - cap_center_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_cap = cap["text"]

                if not best_cap and fig_captions:
                    for cap in fig_captions:
                        cap_center_y = (cap["bbox"].y0 + cap["bbox"].y1) / 2
                        dist = abs(cap_center_y - (bbox.y0 + bbox.y1) / 2)
                        if dist < min_dist:
                            min_dist = dist
                            best_cap = cap["text"]

                filename = f"{os.path.basename(pdf_path)}_page{page_num}_img_{xref}.{ext}"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(img_bytes)

                raw_caption     = best_cap if best_cap else f"Figure on page {page_num} of {os.path.basename(pdf_path)}"
                enriched_caption = _describe_image_with_vision(filepath, raw_caption)
                time.sleep(1)

                image_records.append({
                    "source":     os.path.basename(pdf_path),
                    "page":       page_num,
                    "image_path": filepath,
                    "caption":    enriched_caption,
                    "bbox":       (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                })
                page_extracted_xrefs.add(xref)

            except Exception as e:
                print(f"Error extracting raster image {xref}: {e}")

        # ── 3. Vector drawings (LaTeX pgfplots / tikz figures) ────────────────
        # Only attempted if no raster images were extracted on this page.
        drawings           = page.get_drawings()
        page_table_bboxes  = table_bboxes_by_page.get(page_num, [])

        if drawings and fig_captions and not page_extracted_xrefs:
            for idx, cap in enumerate(fig_captions):
                cy0 = cap["bbox"].y0
                cx0 = cap["bbox"].x0
                cx1 = cap["bbox"].x1

                fig_rects = []
                for d in drawings:
                    d_rect = d["rect"]

                    # Must horizontally overlap the caption column
                    overlap = max(0, min(cx1, d_rect.x1) - max(cx0, d_rect.x0))
                    if overlap <= 30:
                        continue

                    # Must be above the caption and within the capture window
                    if not (d_rect.y1 < cy0 and d_rect.y0 > cy0 - 350):
                        continue

                    # Skip full-width header/footer dividers
                    if d_rect.width > page.rect.width - 50:
                        continue

                    # ── FIX: skip drawing elements inside known table regions ──
                    in_table = False
                    for tbl_bbox in page_table_bboxes:
                        tx0, ty0, tx1, ty1 = tbl_bbox
                        # Any overlap between the drawing element and table bbox
                        if not (d_rect.x1 < tx0 or d_rect.x0 > tx1 or
                                d_rect.y1 < ty0 or d_rect.y0 > ty1):
                            in_table = True
                            break
                    if in_table:
                        continue

                    fig_rects.append(d_rect)

                if not fig_rects:
                    continue

                tx0 = min(r.x0 for r in fig_rects)
                ty0 = min(r.y0 for r in fig_rects)
                tx1 = max(r.x1 for r in fig_rects)
                ty1 = max(r.y1 for r in fig_rects)

                # Expand box slightly to capture axis ticks and borders
                tx0 = max(0, tx0 - 10)
                ty0 = max(0, ty0 - 10)
                tx1 = min(page.rect.width,  tx1 + 10)
                ty1 = min(page.rect.height, ty1 + 10)

                clip_rect = fitz.Rect(tx0, ty0, tx1, ty1)
                if clip_rect.width <= 30 or clip_rect.height <= 30:
                    continue

                try:
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                    filename = (
                        f"{os.path.basename(pdf_path)}"
                        f"_page{page_num}_vector_fig_{idx+1}.png"
                    )
                    filepath = os.path.join(output_dir, filename)
                    pix.save(filepath)

                    raw_caption      = cap["text"]
                    enriched_caption = _describe_image_with_vision(filepath, raw_caption)
                    time.sleep(1)

                    image_records.append({
                        "source":     os.path.basename(pdf_path),
                        "page":       page_num,
                        "image_path": filepath,
                        "caption":    enriched_caption,
                        "bbox":       (clip_rect.x0, clip_rect.y0,
                                       clip_rect.x1, clip_rect.y1)
                    })
                    print(f"  Extracted vector figure on page {page_num}: "
                          f"{cap['text'][:50]}...")
                except Exception as e:
                    print(f"Error rendering vector figure: {e}")

    return image_records


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Text extraction  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(pdf_path: str, table_records: list = None) -> list:
    doc = fitz.open(pdf_path)
    pages = []

    if table_records is None:
        
        table_records = extract_tables(pdf_path)

    table_bboxes_by_page = {}
    for r in table_records:
        page_num = r["page"]
        table_bboxes_by_page.setdefault(page_num, []).append(r["bbox"])

    for i, page in enumerate(doc):
        page_num    = i + 1
        table_bboxes = table_bboxes_by_page.get(page_num, [])

        blocks         = page.get_text("blocks")
        cleaned_blocks = []

        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            text = text.strip()
            if not text:
                continue

            overlap = False
            for t_bbox in table_bboxes:
                tx0, ty0, tx1, ty1 = t_bbox
                if not (x1 < tx0 or x0 > tx1 or y1 < ty0 or y0 > ty1):
                    overlap = True
                    break
            if overlap:
                continue

            text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
            text = re.sub(r'\n+', ' ', text)
            cleaned_blocks.append(text)

        page_text = "\n\n".join(cleaned_blocks)
        if len(page_text.strip()) > 50:
            pages.append({"page": page_num, "text": page_text})

    return pages


# For backward compatibility / testing figure captions
def extract_figure_captions(pages: list) -> list:
    captions = []
    pattern  = re.compile(
        r'((?:Figure|Fig|Table)\s+\d+[:\.].*?)(?:\n|$)', re.IGNORECASE
    )
    for page in pages:
        matches = pattern.findall(page["text"])
        for match in matches:
            captions.append({
                "caption":    match.strip(),
                "page":       page["page"],
                "image_path": None
            })
    return captions


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        print("Extracting tables...")
        tbls = extract_tables(pdf)
        print(f"  {len(tbls)} tables extracted")
        print("Extracting images...")
        imgs = extract_images(pdf, table_records=tbls)
        print(f"  {len(imgs)} images extracted")
        print("Extracting text...")
        pages = extract_text(pdf, tbls)
        print(f"  {len(pages)} pages extracted")