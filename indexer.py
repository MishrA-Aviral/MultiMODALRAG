from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import sqlite3
import os
import pickle
import json
import hashlib
import re
import time
from langchain_community.retrievers import BM25Retriever

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

def index_text(pages: list, source_name: str, faiss_path: str = "db/faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"page": page["page"], "source": source_name, "type": "text"}
            ))
    
    if not docs:
        print("  No text chunks to index")
        return
        
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_path)
    print(f"  {len(docs)} text chunks indexed")


def index_bm25_text(pages: list, source_name: str,
                    bm25_path: str = "db/bm25_text.pkl"):
    """
    Build (or incrementally update) a BM25 sparse index over plain-text chunks.

    Uses the same chunking parameters as index_text() so the two indexes stay
    aligned.  Accumulated across multiple PDFs: if db/bm25_text.pkl already
    exists, the new docs are merged into the existing retriever rather than
    overwriting it.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    new_docs = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            new_docs.append(Document(
                page_content=chunk,
                metadata={"page": page["page"], "source": source_name, "type": "text"}
            ))

    if not new_docs:
        print("  No text chunks for BM25 text index")
        return

    # Accumulate: load existing docs from the persisted retriever, merge.
    all_docs = []
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            existing = pickle.load(f)
        # BM25Retriever stores its docs in .docs
        all_docs = list(getattr(existing, "docs", []))

    all_docs.extend(new_docs)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"  {len(new_docs)} text chunks added to BM25 text index ({len(all_docs)} total)")

def index_image_captions(image_records: list, faiss_path: str = "db/faiss_index"):
    docs = []
    for img in image_records:
        cap = img.get('caption', '')
        cap_lower = cap.lower()
        img_type = "Image"
        if any(w in cap_lower for w in ["chart", "graph", "trend", "plot"]):
            img_type = "Financial chart"
        elif "table" in cap_lower:
            img_type = "Data table"
        elif "diagram" in cap_lower or "architecture" in cap_lower:
            img_type = "Diagram"
            
        vis_desc = f"A {img_type.lower()} displaying information related to: {cap.split(chr(10))[0]}"
        
        page_content = (
            f"Image type:\n{img_type}\n\n"
            f"Caption:\n{cap}\n\n"
            f"Visual description:\n{vis_desc}\n\n"
            f"Metadata:\n"
            f"- image_path: {img['image_path']}\n"
            f"- source PDF: {img['source']}\n"
            f"- page number: {img['page']}\n"
            f"- caption: {cap}\n"
            f"- image type: {img_type}"
        )
        
        docs.append(Document(
            page_content=page_content,
            metadata={
                "image_path": img["image_path"],
                "source": img["source"],
                "page": img["page"],
                "type": "image"
            }
        ))
    if not docs:
        print("  No image captions to index")
        return
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_path)
    print(f"  {len(docs)} image captions indexed")

SUMMARY_CACHE_PATH = "db/summary_cache.json"
_BATCH_SIZE = 5

def _load_summary_cache() -> dict:
    if os.path.exists(SUMMARY_CACHE_PATH):
        try:
            with open(SUMMARY_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {}

def _save_summary_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(SUMMARY_CACHE_PATH), exist_ok=True)
    with open(SUMMARY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

def _table_hash(content: str, caption: str) -> str:
    raw = f"{caption}||{content}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:20]

def _summarize_batch(batch: list) -> list | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return None

    tables_block = ""
    for idx, item in enumerate(batch):
        cap = (item.get("caption") or "").strip()
        caption_line = f'Caption: "{cap}"\n' if cap else ""
        tables_block += f"TABLE {idx + 1}:\n{caption_line}{item['content']}\n\n---\n\n"

    prompt = (
        f"You are a precise financial data analyst. "
        f"Summarize each of the {len(batch)} markdown tables below.\n"
        f"For each table write ONE dense prose paragraph (max 4 sentences) that captures:\n"
        f"- The main topic/subject\n"
        f"- Every row label, entity name, metric, or line item listed\n"
        f"- The exact column headers\n"
        f"- Any notably significant values\n\n"
        f"CRITICAL RULE: Copy every company name, subsidiary name, person name, product name, "
        f"plan name, and proper noun VERBATIM from the table into your summary. "
        f"Do NOT paraphrase, generalise, or omit them (e.g. write 'Fluido Slovakia s.r.o.' not 'a Slovak subsidiary').\n\n"
        f"Return ONLY a valid JSON array of exactly {len(batch)} strings, "
        f"one summary per table, in the same order.\n"
        f'Example: ["The income statement table...", "The balance sheet table..."]\n\n'
        f"{tables_block}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            summarizer = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=min(500 * len(batch), 2000),
                api_key=api_key,
            )
            response = summarizer.invoke(prompt)
            raw = response.content.strip()

            # Strip markdown code fences (```json ... ``` or ``` ... ```)
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

            # Robustly locate the outermost JSON array by bracket-matching
            start = raw.find('[')
            if start != -1:
                depth = 0
                end = start
                for i, ch in enumerate(raw[start:], start=start):
                    if ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                raw = raw[start:end + 1]

            result = json.loads(raw)
            if isinstance(result, list) and len(result) == len(batch):
                return [str(s).strip() for s in result]
            # Length mismatch — retry rather than silently giving up
            print(f"    [batch summarize] length mismatch: got {len(result)}, expected {len(batch)}. Retrying...")
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "rate limit" in err_str:
                sleep_time = 15 * (attempt + 1)
                print(f"    [batch summarize] Rate limit hit. Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"    [batch summarize] failed ({type(exc).__name__}): {exc}")
                break
    return None

def generate_table_summaries_batch(tables: list, cache: dict) -> list:
    results = [""] * len(tables)
    uncached_indices = []

    for i, item in enumerate(tables):
        h = _table_hash(item.get("content", ""), item.get("caption", ""))
        item["_hash"] = h
        if h in cache:
            results[i] = cache[h]
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        print(f"    [summarize] all {len(tables)} summaries served from cache")
        return results

    n_cached = len(tables) - len(uncached_indices)
    n_calls  = -(-len(uncached_indices) // _BATCH_SIZE)
    print(f"    [summarize] {len(uncached_indices)} table(s) need summarization ({n_cached} from cache) => {n_calls} API call(s)")

    uncached_items = [tables[i] for i in uncached_indices]
    for batch_start in range(0, len(uncached_items), _BATCH_SIZE):
        batch = uncached_items[batch_start: batch_start + _BATCH_SIZE]
        batch_summaries = _summarize_batch(batch)

        if batch_summaries and len(batch_summaries) == len(batch):
            for j, summary in enumerate(batch_summaries):
                global_idx = uncached_indices[batch_start + j]
                results[global_idx] = summary
                cache[batch[j]["_hash"]] = summary
        else:
            for j, item in enumerate(batch):
                global_idx = uncached_indices[batch_start + j]
                summary = generate_table_summary(item.get("content", ""), item.get("caption", ""))
                results[global_idx] = summary
                cache[item["_hash"]] = summary

    return results

def generate_table_summary(markdown_content: str, caption: str = "") -> str:
    """
    Call a lightweight LLM to generate a dense prose summary of a markdown table.

    The summary is what gets embedded into FAISS. It is semantically richer than
    raw markdown for a text embedding model, which dramatically improves retrieval
    recall for table-related queries (vector blinding fix).

    Falls back to the original caption + markdown string on any failure so the
    indexing pipeline is never interrupted by an API error.
    """
    fallback = f"{caption}\n{markdown_content}" if caption else markdown_content
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return fallback

        summarizer = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.0,
            max_tokens=300,
            api_key=api_key,
        )

        caption_line = f'The table caption or heading is: "{caption}".\n' if caption else ""
        prompt = (
            f"You are a precise data analyst. {caption_line}"
            f"Below is a markdown table.\n\n"
            f"{markdown_content}\n\n"
            f"Write a single dense prose paragraph (max 5 sentences) that captures:\n"
            f"- The main topic or subject of the table (e.g. financial results, model performance, schedule).\n"
            f"- Every row label, entity name, metric, or line item listed.\n"
            f"- The exact column headers (e.g. years, quarters, model names, metric names).\n"
            f"- Any notably high, low, or significant values worth calling out.\n"
            f"Do NOT use bullet points. Output only the prose paragraph."
        )

        response = summarizer.invoke(prompt)
        summary = response.content.strip()
        return summary if summary else fallback

    except Exception:
        # Never let a summarization failure interrupt the indexing pipeline
        return fallback


def index_tables(db_path: str = "db/tables.db", faiss_path: str = "db/faiss_index"):
    if not os.path.exists(db_path):
        print("  No tables database found to index")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(tables)")
    columns = [col[1] for col in cursor.fetchall()]
    
    has_caption_path = "caption" in columns and "table_path" in columns
    has_group_id = "table_group_id" in columns

    if has_caption_path and has_group_id:
        cursor.execute("SELECT source, page, content, caption, table_path, table_group_id FROM tables")
    elif has_caption_path:
        cursor.execute("SELECT source, page, content, caption, table_path, NULL FROM tables")
    else:
        cursor.execute("SELECT source, page, content, NULL, NULL, NULL FROM tables")
        
    rows = cursor.fetchall()
    conn.close()

    cache = _load_summary_cache()

    # Process everything in chunks to keep memory usage flat
    chunk_size = 20
    total_indexed = 0
    all_docs = []      # FAISS gets prose summaries (semantic search)
    bm25_raw_docs = [] # BM25 gets raw markdown (exact keyword search)

    for i in range(0, len(rows), chunk_size):
        chunk_rows = rows[i:i + chunk_size]
        
        table_inputs = []
        for r in chunk_rows:
            table_inputs.append({
                "source": r[0], "page": r[1], "content": r[2], 
                "caption": r[3] if r[3] else "", 
                "table_path": r[4] if r[4] else "",
                "table_group_id": r[5]
            })

        summarize_inputs = [{"content": t["content"], "caption": t["caption"]} for t in table_inputs]
        
        print(f"  Chunk {i//chunk_size + 1}/{(len(rows) + chunk_size - 1)//chunk_size}:")
        summaries = generate_table_summaries_batch(summarize_inputs, cache)
        _save_summary_cache(cache)

        chunk_docs = []
        for item, summary in zip(table_inputs, summaries):
            if not summary:
                summary = item["content"]
            
            raw_md = item["content"]
            if has_caption_path:
                raw_md = f"Table Caption: {item['caption']}\nPage: {item['page']}\nTable Data:\n{item['content']}"
                
            metadata = {
                "source": item["source"],
                "page": item["page"],
                "type": "table",
                "raw_table_markdown": raw_md,
                "table_path": item["table_path"],
                "table_group_id": item["table_group_id"]
            }
                
            chunk_docs.append(Document(page_content=summary, metadata=metadata))
            all_docs.append(Document(page_content=summary, metadata=metadata))
            # BM25 indexes the raw markdown so exact entity-name keyword matches
            # always work even when the prose summary omits a proper noun.
            bm25_raw_docs.append(Document(page_content=raw_md, metadata=metadata))

        if chunk_docs:
            if os.path.exists(faiss_path):
                vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
                vectorstore.add_documents(chunk_docs)
            else:
                vectorstore = FAISS.from_documents(chunk_docs, embeddings)
            vectorstore.save_local(faiss_path)
            total_indexed += len(chunk_docs)

    if not all_docs:
        print("  No tables to index")
        return
    
    # --- BM25 SPARSE INDEX ---
    # BM25 intentionally indexes raw table markdown (not prose summaries) so that
    # exact entity-name keyword queries (e.g. "Fluido Slovakia", "Panaya GmbH")
    # are always findable via BM25 even when the FAISS summary omits the name.
    bm25_retriever = BM25Retriever.from_documents(bm25_raw_docs)
    bm25_path = "db/bm25_tables.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    print(f"  {total_indexed} table chunks indexed (FAISS + BM25)")