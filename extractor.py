import base64
import time
import fitz
import pdfplumber
import sqlite3
import os
import re

def clean_cell(val):
    if val is None:
        return ""
    return str(val).strip().replace("\n", " ")

def table_to_markdown(table_data) -> str:
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

def find_table_bboxes_by_alignment(page):
    h_lines = [l for l in page.lines if abs(l["y0"] - l["y1"]) < 1]
    if not h_lines:
        return []
    
    groups = []
    for line in h_lines:
        placed = False
        for g in groups:
            ref = g[0]
            if abs(line["x0"] - ref["x0"]) < 15 and abs(line["x1"] - ref["x1"]) < 15:
                if any(abs(line["top"] - item["top"]) < 300 for item in g):
                    g.append(line)
                    placed = True
                    break
        if not placed:
            groups.append([line])
            
    bboxes = []
    for g in groups:
        if len(g) >= 2:
            top = min(l["top"] for l in g)
            bottom = max(l["bottom"] for l in g)
            x0 = min(l["x0"] for l in g)
            x1 = max(l["x1"] for l in g)
            
            # Tight bounding box to prevent overlapping into adjacent columns during text masking
            x0_tight = max(0, x0 - 5)
            x1_tight = min(page.width, x1 + 5)
            y0_tight = max(0, top - 2)
            y1_tight = min(page.height, bottom + 2)
            
            bboxes.append((x0_tight, y0_tight, x1_tight, y1_tight))
    return bboxes


def _describe_image_with_vision(filepath: str, fallback_caption: str) -> str:
    """
    Call the Groq vision API (llama-3.2-11b-vision-preview) to generate a rich
    description of an extracted figure from the PDF.

    The returned string combines the original caption with the vision model's
    description so the FAISS index receives maximum textual context about the
    image — enabling meaningful retrieval for figure-related queries.

    Fails silently in ALL error cases (missing API key, rate limit, bad image,
    network error) and returns the original fallback_caption unchanged, so this
    function can never break the extraction pipeline.
    """
    try:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return fallback_caption

        with open(filepath, "rb") as f:
            img_bytes = f.read()

        ext = os.path.splitext(filepath)[1].lower().lstrip(".")
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"This is a figure from an academic research paper. "
                                f'The figure caption reads: "{fallback_caption}". '
                                f"Describe what you see in detail: chart type, axis labels, "
                                f"numerical values, trends, structural elements, arrows, and "
                                f"any text visible in the image. Be thorough so a reader can "
                                f"understand this figure without seeing it."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=400,
        )

        description = response.choices[0].message.content.strip()
        # Prepend original caption so it is always present in the indexed text
        return f"{fallback_caption}\n\nVision Description: {description}"

    except Exception:
        # Never let a vision API failure interrupt the extraction pipeline
        return fallback_caption


def extract_tables(pdf_path: str, db_path: str = "db/tables.db", output_dir: str = "data/extracted_tables") -> list:
    os.makedirs(output_dir, exist_ok=True)
    table_records = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Get table captions on this page using fitz
            doc_fitz = fitz.open(pdf_path)
            page_fitz = doc_fitz[i]
            blocks = page_fitz.get_text("blocks")
            tbl_captions = []
            for b in blocks:
                b_text = b[4].strip()
                if re.match(r'^\s*(Table)\s+\d+[:\.\s]', b_text, re.IGNORECASE):
                    tbl_captions.append({
                        "text": b_text,
                        "bbox": (b[0], b[1], b[2], b[3])
                    })
            doc_fitz.close()
            
            # Find table bboxes
            # 1. Alignment-based clustering
            bboxes = find_table_bboxes_by_alignment(page)
            
            # 2. Gridline-based tables
            default_tables = page.find_tables()
            
            extracted_tables = []
            
            # Process line clusters first
            for bbox in bboxes:
                try:
                    # Crop wider to catch text starting to the left of line rules (e.g. "ELMo", "CVT")
                    crop_bbox = (
                        max(0, bbox[0] - 25),
                        bbox[1],
                        min(page.width, bbox[2] + 15),
                        bbox[3]
                    )
                    cropped = page.crop(crop_bbox)
                    
                    # Peek at column structure to decide extraction strategy.
                    # If col 0's left boundary is significantly inside the crop
                    # (i.e. there is a margin to the left of the first column),
                    # use per-cell extraction with first-column expansion to fix
                    # truncation of numbers like "12"→"2" or words like "System"→"stem".
                    # If col 0 is already at the crop left edge, use extract_table()
                    # directly (it handles multi-header tables like Table 7 better).
                    table_data = None
                    found_tables = cropped.find_tables(table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                    })
                    if found_tables:
                        t = found_tables[0]
                        col0_x0 = t.columns[0].bbox[0] if t.columns else cropped.bbox[0]
                        first_col_inside_crop = col0_x0 > cropped.bbox[0] + 5

                        if first_col_inside_crop:
                            # Cell-expansion path: re-extract each cell, pulling
                            # the first column leftward to the crop edge so that
                            # characters at col0_x0 - N are captured.
                            table_data = []
                            for row in t.rows:
                                row_text = []
                                for col_idx, cell in enumerate(row.cells):
                                    if cell is None:
                                        row_text.append("")
                                        continue
                                    x0, y0, x1, y1 = cell
                                    if col_idx == 0:
                                        x0 = max(cropped.bbox[0], x0 - 20)
                                    try:
                                        cell_text = cropped.crop((x0, y0, x1, y1)).extract_text() or ""
                                        cell_text = cell_text.strip().replace("\n", " ")
                                    except Exception:
                                        cell_text = ""
                                    row_text.append(cell_text)
                                if any(c for c in row_text):
                                    table_data.append(row_text)
                        else:
                            # Standard path: col 0 is already at crop left edge;
                            # extract_table() handles this correctly.
                            table_data = cropped.extract_table(table_settings={
                                "vertical_strategy": "text",
                                "horizontal_strategy": "text",
                            })

                    if table_data and len(table_data) > 1 and len(table_data[0]) > 1:
                        extracted_tables.append({
                            "data": table_data,
                            "bbox": bbox,
                            "method": "line_cluster"
                        })
                except Exception as e:
                    pass
                    
            # Process default tables (if no overlap)
            for tbl in default_tables:
                overlap = False
                for c_bbox in bboxes:
                    if not (tbl.bbox[2] < c_bbox[0] or tbl.bbox[0] > c_bbox[2] or tbl.bbox[3] < c_bbox[1] or tbl.bbox[1] > c_bbox[3]):
                        overlap = True
                        break
                if not overlap:
                    table_data = tbl.extract()
                    if table_data and len(table_data) > 1:
                        extracted_tables.append({
                            "data": table_data,
                            "bbox": tbl.bbox,
                            "method": "gridline"
                        })
            
            # For each extracted table, find the closest column-aligned caption on this page
            for idx, tbl_item in enumerate(extracted_tables):
                bbox = tbl_item["bbox"]
                table_data = tbl_item["data"]
                
                tbl_x0, tbl_y0, tbl_x1, tbl_y1 = bbox
                tbl_center_y = (tbl_y0 + tbl_y1) / 2
                
                best_cap = None
                min_dist = float("inf")
                
                for cap in tbl_captions:
                    cap_x0, cap_y0, cap_x1, cap_y1 = cap["bbox"]
                    cap_center_y = (cap_y0 + cap_y1) / 2
                    
                    # Ensure the caption aligns horizontally in the same column
                    overlap = max(0, min(tbl_x1, cap_x1) - max(tbl_x0, cap_x0))
                    if overlap > 30: # 30pt overlap threshold
                        dist = abs(tbl_center_y - cap_center_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_cap = cap["text"]
                
                # If no caption found above, find absolute closest (backup fallback)
                if not best_cap and tbl_captions:
                    for cap in tbl_captions:
                        cap_center_y = (cap["bbox"][1] + cap["bbox"][3]) / 2
                        tbl_center_y = (bbox[1] + bbox[3]) / 2
                        dist = abs(cap_center_y - tbl_center_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_cap = cap["text"]
                
                markdown_content = table_to_markdown(table_data)
                filename = f"{os.path.basename(pdf_path)}_page{i+1}_table_{idx+1}.md"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                    
                table_records.append({
                    "source": os.path.basename(pdf_path),
                    "page": i + 1,
                    "table_path": filepath,
                    "content": markdown_content,
                    "caption": best_cap if best_cap else f"Table on page {i+1} of {os.path.basename(pdf_path)}",
                    "bbox": bbox
                })
                
    # Also write to tables.db SQLite for compatibility
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            page INTEGER,
            content TEXT,
            caption TEXT,
            table_path TEXT
        )
    """)
    # Clear old entries for this PDF
    cursor.execute("DELETE FROM tables WHERE source = ?", (os.path.basename(pdf_path),))
    for r in table_records:
        cursor.execute(
            "INSERT INTO tables (source, page, content, caption, table_path) VALUES (?, ?, ?, ?, ?)",
            (r["source"], r["page"], r["content"], r["caption"], r["table_path"])
        )
    conn.commit()
    conn.close()
    
    return table_records

def extract_images(pdf_path: str, output_dir: str = "data/extracted_images") -> list:
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_records = []
    
    for i, page in enumerate(doc):
        # 1. Get all figure captions on this page (restrictive pattern to avoid false matching regular sentences)
        blocks = page.get_text("blocks")
        fig_captions = []
        for b in blocks:
            b_text = b[4].strip()
            if re.match(r'^\s*(Figure|Fig)\s+\d+[:\.]', b_text, re.IGNORECASE):
                fig_captions.append({
                    "text": b_text,
                    "bbox": fitz.Rect(b[0], b[1], b[2], b[3])
                })
        
        # 2. Get all embedded raster images on this page
        images_info = page.get_image_info(xrefs=True)
        seen_xrefs = set()
        raster_images = []
        for img in images_info:
            xref = img.get("xref")
            if not xref or xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            
            w = img.get("width", 0)
            h = img.get("height", 0)
            bbox = img.get("bbox")
            if w < 50 or h < 50 or not bbox:
                continue
                
            raster_images.append({
                "xref": xref,
                "bbox": fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3]),
                "width": w,
                "height": h
            })
            
        # 3. Extract raster images if present
        page_extracted_xrefs = set()
        for img in raster_images:
            xref = img["xref"]
            bbox = img["bbox"]
            
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]
                
                if len(img_bytes) < 5000:
                    continue
                    
                # Find closest column-aligned caption
                img_center_y = (bbox.y0 + bbox.y1) / 2
                best_cap = None
                min_dist = float("inf")
                for cap in fig_captions:
                    cap_x0, cap_y0, cap_x1, cap_y1 = cap["bbox"]
                    cap_center_y = (cap_y0 + cap_y1) / 2
                    
                    # Ensure caption lies in the same column horizontally
                    overlap = max(0, min(bbox.x1, cap_x1) - max(bbox.x0, cap_x0))
                    if overlap > 30:
                        dist = abs(img_center_y - cap_center_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_cap = cap["text"]
                
                if not best_cap and fig_captions:
                    for cap in fig_captions:
                        cap_center_y = (cap["bbox"].y0 + cap["bbox"].y1) / 2
                        img_center_y = (bbox.y0 + bbox.y1) / 2
                        dist = abs(cap_center_y - img_center_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_cap = cap["text"]
                
                filename = f"{os.path.basename(pdf_path)}_page{i+1}_img_{xref}.{ext}"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(img_bytes)

                raw_caption = best_cap if best_cap else f"Figure on page {i+1} of {os.path.basename(pdf_path)}"
                enriched_caption = _describe_image_with_vision(filepath, raw_caption)
                time.sleep(1)  # Respect Groq vision API rate limits between calls

                image_records.append({
                    "source": os.path.basename(pdf_path),
                    "page": i + 1,
                    "image_path": filepath,
                    "caption": enriched_caption,
                    "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                })
                page_extracted_xrefs.add(xref)
            except Exception as e:
                print(f"Error extracting raster image {xref}: {e}")
                
        # 4. Extract vector drawings if no raster images were extracted on this page
        # This handles diagrams drawn dynamically in LaTeX (e.g. pgfplots, tikz)
        drawings = page.get_drawings()
        if drawings and fig_captions and not page_extracted_xrefs:
            for idx, cap in enumerate(fig_captions):
                cy0 = cap["bbox"].y0
                cx0 = cap["bbox"].x0
                cx1 = cap["bbox"].x1
                
                # Find drawing elements above the caption in the same column
                fig_rects = []
                for d in drawings:
                    d_rect = d["rect"]
                    # Ensure drawings fall into the same column horizontally
                    overlap = max(0, min(cx1, d_rect.x1) - max(cx0, d_rect.x0))
                    if overlap > 30:
                        if d_rect.y1 < cy0 and d_rect.y0 > cy0 - 350:
                            # Skip full-width header/footer divider lines
                            if d_rect.width > page.rect.width - 50:
                                continue
                            fig_rects.append(d_rect)
                        
                if fig_rects:
                    tx0 = min(r.x0 for r in fig_rects)
                    ty0 = min(r.y0 for r in fig_rects)
                    tx1 = max(r.x1 for r in fig_rects)
                    ty1 = max(r.y1 for r in fig_rects)
                    
                    # Expand box slightly to capture axis ticks and borders
                    tx0 = max(0, tx0 - 10)
                    ty0 = max(0, ty0 - 10)
                    tx1 = min(page.rect.width, tx1 + 10)
                    ty1 = min(page.rect.height, ty1 + 10)
                    
                    clip_rect = fitz.Rect(tx0, ty0, tx1, ty1)
                    if clip_rect.width > 30 and clip_rect.height > 30:
                        try:
                            # Render the vector area to a high-quality PNG using 2x resolution matrix
                            mat = fitz.Matrix(2, 2)
                            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                            filename = f"{os.path.basename(pdf_path)}_page{i+1}_vector_fig_{idx+1}.png"
                            filepath = os.path.join(output_dir, filename)
                            pix.save(filepath)
                            
                            raw_caption = cap["text"]
                            enriched_caption = _describe_image_with_vision(filepath, raw_caption)
                            time.sleep(1)  # Respect Groq vision API rate limits between calls

                            image_records.append({
                                "source": os.path.basename(pdf_path),
                                "page": i + 1,
                                "image_path": filepath,
                                "caption": enriched_caption,
                                "bbox": (clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1)
                            })
                            print(f"  Extracted vector figure on page {i+1}: {cap['text'][:50]}...")
                        except Exception as e:
                            print(f"Error rendering vector figure: {e}")
                            
    return image_records


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
        page_num = i + 1
        table_bboxes = table_bboxes_by_page.get(page_num, [])
        
        blocks = page.get_text("blocks")
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
                
            # Clean up hyphenation and newlines inside block
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
    pattern = re.compile(r'((?:Figure|Fig|Table)\s+\d+[:\.].*?)(?:\n|$)', re.IGNORECASE)
    for page in pages:
        matches = pattern.findall(page["text"])
        for match in matches:
            captions.append({
                "caption": match.strip(),
                "page": page["page"],
                "image_path": None
            })
    return captions

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        print("Extracting tables...")
        tbls = extract_tables(pdf)
        print(f"  {len(tbls)} tables extracted")
        print("Extracting images...")
        imgs = extract_images(pdf)
        print(f"  {len(imgs)} images extracted")
        print("Extracting text...")
        pages = extract_text(pdf, tbls)
        print(f"  {len(pages)} pages extracted")