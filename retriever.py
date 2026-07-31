from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import re
import os
import time
import pickle
import json


bm25_retriever = None
bm25_text_retriever = None

# Registry built at index-load time: source_basename → list of sample text chunks.
# Populated by load_index(); used by _auto_detect_source() to route queries
# to the right document without any manual source-filter configuration.
_source_registry: dict = {}

# Phrases that indicate the LLM did not find the requested data.
# Citation tags must not be appended when the answer signals failure.
_FAILURE_PHRASES = (
    "not found in context",
    "not directly provided",
    "cannot",
    "unable to",
    "no explicit",
)

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=2048,
    api_key=os.getenv("GROQ_API_KEY")
)

def load_index(faiss_path: str = "db/faiss_index"):
    global bm25_retriever, bm25_text_retriever, _source_registry
    bm25_path = "db/bm25_tables.pkl"
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)

    bm25_text_path = "db/bm25_text.pkl"
    if os.path.exists(bm25_text_path):
        with open(bm25_text_path, "rb") as f:
            bm25_text_retriever = pickle.load(f)

    vs = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

    # Build source registry: collect up to 3 representative text chunks per
    # indexed source so the LLM can recognise which document a query refers to.
    _source_registry = {}
    for doc_id, doc in vs.docstore._dict.items():
        src = doc.metadata.get("source", "")
        if not src:
            continue
        if src not in _source_registry:
            _source_registry[src] = []
        if len(_source_registry[src]) < 3:
            chunk = doc.page_content.strip()[:300]
            if chunk:
                _source_registry[src].append(chunk)

    print(f"  [load_index] {len(_source_registry)} source(s) registered: "
          f"{list(_source_registry.keys())}")
    return vs


def _normalize_row(row: str) -> str:
    """
    Normalize a markdown table row for deduplication.

    Old logic deduped by numeric tokens only, which could delete valid rows that
    happened to share the same numbers. This version dedupes by the full row
    text after removing citation markers and collapsing whitespace.
    """
    row_no_refs = re.sub(r"\[\d+\]", "", row)
    row_no_refs = re.sub(r"\s+", " ", row_no_refs).strip()
    return row_no_refs



def clean_table_markdown(content: str) -> str:
    """
    Remove exact duplicate table rows while preserving headers and separators.
    """
    lines = content.splitlines()
    cleaned = []
    seen_rows = set()

    for line in lines:
        stripped = line.strip()

        if not stripped.startswith("|"):
            cleaned.append(line)
            continue

        if re.match(r'^\|\s*[-:]+', stripped):
            cleaned.append(line)
            continue

        key = _normalize_row(stripped)

        if not key:
            cleaned.append(line)
            continue

        if key in seen_rows:
            continue

        seen_rows.add(key)
        cleaned.append(line)

    return "\n".join(cleaned)






def _llm_classify_query(query: str) -> dict:
    """Use the LLM to classify the query routing intent."""
    prompt = (
        f"Analyze the following user query and return a valid JSON object with EXACTLY these four boolean keys:\n"
        f'  "wants_table_data": true if the query asks for numbers, metrics, comparisons, benchmarks, '
        f'financial figures (revenue, profit, EPS, ratios, balance sheet items), or any data typically '
        f'presented in tabular form.\n'
        f'  "wants_image_data": true if the query asks about charts, graphs, figures, diagrams, '
        f'or any visual content.\n'
        f'  "needs_analysis": true if the query asks for explanations, reasons, trends, tradeoffs, '
        f'or interpretation (why/how) rather than just a fact lookup. '
        f'CRITICAL EXCEPTION: YOU MUST ALSO SET THIS TO TRUE when the query asks for a list, identities, or headcount of personnel/board members, or a count/quantity/percentage of a named business activity. '
        f'Such facts typically appear in prose sections rather than structured tables and require more text slots to retrieve.\n'
        f'  "format_as_table": true if the query asks for a ranked list, comparison table, '
        f'or explicitly requests tabular output.\n\n'
        f'Query: "{query}"\n\n'
        f"Return ONLY JSON. No explanations."
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  [LLM Router Error] {e}")
        # Safe fallback
        return {
            "wants_table_data": True,
            "wants_image_data": False,
            "needs_analysis": False,
            "format_as_table": False
        }


def _matches_source(doc, source_filter: str) -> bool:
    """
    Return True if the doc's source metadata matches the filter.
    Tries normalized full-path comparison first; falls back to basename
    comparison so that queries work whether the index stores full paths
    or bare filenames.
    """
    doc_src = doc.metadata.get("source", "")
    # Primary: normalized path equality
    if os.path.normpath(doc_src) == os.path.normpath(source_filter):
        return True
    # Fallback: basename equality (handles mixed absolute/relative storage)
    return os.path.basename(doc_src) == os.path.basename(source_filter)


def _unique_docs(docs: list) -> list:
    """
    Deduplicate docs by source + page + type + content.
    This prevents repeated identical chunks from inflating the prompt.
    """
    seen = set()
    unique = []
    for doc in docs:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.metadata.get("type", ""),
            doc.page_content.strip()
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique

def rerank_docs(query: str, docs: list, top_n: int = 5) -> list:
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in ranked[:top_n]]

def retrieve(query: str, vectorstore, intent: dict, k: int = 6, source_filter: str = None) -> list:
    if intent.get("wants_image_data"):
        mode = "image"
    elif intent.get("wants_table_data"):
        mode = "hybrid" if intent.get("needs_analysis") else "table"
    else:
        mode = "text"

    fetch_k = 40 if source_filter is None else 30
    raw_docs = vectorstore.similarity_search(query, k=fetch_k)

    if source_filter:
        filtered = [d for d in raw_docs if _matches_source(d, source_filter)]
        if filtered:
            raw_docs = filtered
        else:
            # If the filter wiped out everything, don't fall back to returning wrong papers!
            raw_docs = []

    raw_docs = _unique_docs(raw_docs)

    if mode in ("table", "hybrid"):
        table_docs = []
        seen_content = set()

        # --- HYBRID SEARCH: FAISS + BM25 ---
        table_pool_k = 60 if source_filter is None else 80
        faiss_raw = vectorstore.similarity_search(query, k=table_pool_k)
        if source_filter:
            faiss_table_docs = [d for d in faiss_raw if _matches_source(d, source_filter)]
        else:
            faiss_table_docs = faiss_raw

        bm25_docs = []
        if bm25_retriever is not None:
            # BM25 is excellent for exact keyword matches (e.g. "FUNSD F1").
            # Increase k to prevent post-filtering starvation
            old_k = getattr(bm25_retriever, "k", 4)
            bm25_retriever.k = 30
            bm25_raw = bm25_retriever.invoke(query)
            bm25_retriever.k = old_k
            
            if source_filter:
                bm25_docs = [d for d in bm25_raw if _matches_source(d, source_filter)]
            else:
                bm25_docs = bm25_raw

        # Interleave Dense (FAISS) and Sparse (BM25) results — BM25 gets 2:1 priority.
        # For exact-metric queries ("mAP", "FUNSD F1"), BM25 keyword hits surface the
        # correct table more reliably than semantic vectors, so we emit two BM25 docs
        # for every one FAISS doc before moving to the next FAISS candidate.
        combined_raw = []
        faiss_idx = 0
        bm25_idx = 0
        while bm25_idx < len(bm25_docs) or faiss_idx < len(faiss_table_docs):
            # Emit up to 2 BM25 candidates first
            for _ in range(2):
                if bm25_idx < len(bm25_docs):
                    combined_raw.append(bm25_docs[bm25_idx])
                    bm25_idx += 1
            # Then emit 1 FAISS candidate
            if faiss_idx < len(faiss_table_docs):
                combined_raw.append(faiss_table_docs[faiss_idx])
                faiss_idx += 1

        # Secondary dedup key on raw_table_markdown: after the table summarization
        # refactor, page_content holds a prose summary. Two docs with different
        # summaries but identical underlying grids must not both consume a slot.
        seen_raw_markdown: set = set()
        for doc in combined_raw:
            if doc.metadata.get("type") != "table":
                continue
            if source_filter and not _matches_source(doc, source_filter):
                continue

            content = doc.page_content.strip()
            raw_md = doc.metadata.get("raw_table_markdown", content)
            if content in seen_content or raw_md in seen_raw_markdown:
                continue

            table_docs.append(doc)
            seen_content.add(content)
            seen_raw_markdown.add(raw_md)

            if len(table_docs) >= 15:
                break

        if not table_docs:
            table_docs = [d for d in raw_docs if d.metadata.get("type") == "table"]

        # ── GENERAL MULTI-PAGE TABLE CONTINUATION RETRIEVAL ─────────────────────
        # If a retrieved table is part of a multi-page table group, fetch all other
        # tables in that exact same group so the LLM gets the entire schedule.
        table_group_ids_in_results = {
            doc.metadata.get("table_group_id")
            for doc in table_docs
            if doc.metadata.get("table_group_id")
        }
        if table_group_ids_in_results:
            for doc_id, doc in vectorstore.docstore._dict.items():
                if doc.metadata.get("type") != "table":
                    continue
                grp = doc.metadata.get("table_group_id")
                if not grp or grp not in table_group_ids_in_results:
                    continue
                if source_filter and not _matches_source(doc, source_filter):
                    continue
                content = doc.page_content.strip()
                raw_md = doc.metadata.get("raw_table_markdown", content)
                if content not in seen_content and raw_md not in seen_raw_markdown:
                    table_docs.insert(0, doc)
                    seen_content.add(content)
                    seen_raw_markdown.add(raw_md)

        # ── TEXT CANDIDATES: FAISS + BM25 TEXT ────────────────────────────────
        # Pull text docs from FAISS results first, then supplement with BM25
        # text retriever. BM25 keyword matching surfaces narrative pages that semantic
        # search often misses in favor of financial tables.
        text_candidates = [
            d for d in raw_docs
            if d.metadata.get("type") == "text"
            and d.page_content.strip() not in seen_content
        ]
        if bm25_text_retriever is not None:
            try:
                old_k = getattr(bm25_text_retriever, "k", 4)
                bm25_text_retriever.k = 12
                bm25_text_extra = bm25_text_retriever.invoke(query)
                bm25_text_retriever.k = old_k
                if source_filter:
                    bm25_text_extra = [d for d in bm25_text_extra
                                       if _matches_source(d, source_filter)]
                for d in bm25_text_extra:
                    if d.page_content.strip() not in seen_content:
                        text_candidates.append(d)
            except Exception:
                pass

        if mode == "hybrid":
            text_slots = 3
        elif intent.get("needs_analysis"):
            text_slots = 2
        else:
            text_slots = 1

        top_text = (
            rerank_docs(query, _unique_docs(text_candidates), top_n=text_slots)
            if text_slots > 0
            else []
        )
        for doc in top_text:
            seen_content.add(doc.page_content.strip())

        # ── SLOT-PRIORITY ASSEMBLY ─────────────────────────────────────────────
        # Critical: text docs go FIRST so [:limit] never cuts them off.
        # Table slots are trimmed to make room: max_table_slots = limit - text - image.
        _TOTAL_LIMIT = 12
        ordered_tables = _unique_docs(table_docs)

        max_table_slots = _TOTAL_LIMIT - len(top_text) - 1  # reserve 1 for image
        trimmed_tables = ordered_tables[:max_table_slots]

        # Include the top-1 most relevant image doc.
        image_candidates = [
            d for d in raw_docs
            if d.metadata.get("type") == "image"
            and d.page_content.strip() not in seen_content
        ]
        top_image = []
        if image_candidates:
            _img_ranked = rerank_docs(query, _unique_docs(image_candidates), top_n=1)
            top_image = [d for d in _img_ranked if d.page_content.strip() not in seen_content]

        return top_text + trimmed_tables + top_image

    if mode == "image":
        image_docs = [d for d in raw_docs if d.metadata.get("type") == "image"]
        if not image_docs:
            image_docs = raw_docs
        return rerank_docs(query, _unique_docs(image_docs), top_n=3)

    # ── Text / fallback branch ────────────────────────────────────────────────
    text_docs = [d for d in raw_docs if d.metadata.get("type") == "text"]
    if not text_docs:
        text_docs = raw_docs

    # Merge BM25 text results (2:1 ratio vs FAISS) for better exact-keyword recall
    if bm25_text_retriever is not None:
        bm25_text_retriever.k = 6
        try:
            bm25_text_results = bm25_text_retriever.invoke(query)
            # Apply source filter if active
            if source_filter:
                bm25_text_results = [d for d in bm25_text_results
                                     if _matches_source(d, source_filter)]
        except Exception:
            bm25_text_results = []

        # 2:1 interleave: BM25 first, then FAISS
        merged = []
        bm25_iter = iter(bm25_text_results)
        faiss_iter = iter(text_docs)
        while True:
            b1 = next(bm25_iter, None)
            b2 = next(bm25_iter, None)
            f1 = next(faiss_iter, None)
            if b1 is None and b2 is None and f1 is None:
                break
            if b1 is not None:
                merged.append(b1)
            if b2 is not None:
                merged.append(b2)
            if f1 is not None:
                merged.append(f1)
        text_docs = _unique_docs(merged)

    return rerank_docs(query, _unique_docs(text_docs), top_n=5)


def get_page_docs(vectorstore, page_num: int, source_filter: str = None) -> list:
    """
    Return all docs for a given page number, optionally filtered by source.
    """
    all_docs = []
    docstore = vectorstore.docstore._dict
    for doc_id, doc in docstore.items():
        if doc.metadata.get("page") == page_num:
            if source_filter is None or _matches_source(doc, source_filter):
                all_docs.append(doc)
    return _unique_docs(all_docs)


_TPD_WAIT_RE = re.compile(r"try again in (\d+)m([\d.]+)s", re.IGNORECASE)
_TPD_CAP_SECONDS = 20 * 60  # 20-minute maximum auto-sleep for daily quota errors


def _invoke_llm_with_retry(prompt: str, retries: int = 3, base_sleep: int = 6):
    """
    Retry on Groq rate-limit errors, distinguishing two error classes:

    Per-minute (TPM) limits  → short exponential back-off (6 s, 12 s, 18 s).
    Per-day    (TPD) limits  → parse the stated wait from the error message and
                               sleep exactly that long (once), then retry once.
                               If the stated wait exceeds 20 minutes, raise
                               immediately with a clear user-facing message
                               rather than burning retries on useless short sleeps.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            msg = str(exc)
            msg_lower = msg.lower()

            if "429" not in msg_lower and "rate limit" not in msg_lower and "tokens per" not in msg_lower:
                raise  # Not a rate-limit error — propagate immediately

            last_exc = exc
            is_tpd = "tokens per day" in msg_lower or "tpd" in msg_lower or "daily" in msg_lower

            if is_tpd:
                # Try to parse the exact wait time Groq states in the message
                m = _TPD_WAIT_RE.search(msg)
                if m:
                    wait_seconds = int(m.group(1)) * 60 + float(m.group(2))
                    if wait_seconds > _TPD_CAP_SECONDS:
                        raise RuntimeError(
                            f"Groq daily token quota exhausted. "
                            f"Groq says to wait {wait_seconds/60:.1f} minutes, "
                            f"which exceeds the 20-minute auto-sleep cap. "
                            f"Please wait or upgrade your Groq tier, then retry."
                        ) from exc
                    print(f"  [LLM] Daily quota (TPD) limit hit. "
                          f"Sleeping {wait_seconds:.0f}s as instructed by Groq...")
                    time.sleep(wait_seconds)
                    continue  # one retry after the exact sleep
                else:
                    # TPD error but no parseable wait — fall back to long sleep
                    print("  [LLM] Daily quota (TPD) limit hit (no wait time in message). "
                          "Sleeping 60s and retrying once...")
                    time.sleep(60)
                    continue
            else:
                # Per-minute (TPM) limit — exponential back-off
                time.sleep(base_sleep * (attempt + 1))
                continue

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM invocation failed.")


def _auto_detect_source(query: str) -> str | None:
    """
    Use the LLM to determine which indexed source the query is about.

    The _source_registry (built at load_index() time) provides a name and
    sample text for every indexed document. The LLM reads these and picks
    the best match — or returns None for cross-document / ambiguous queries.

    Short-circuit cases (no LLM call):
      - 0 sources registered        → None (nothing to filter by)
      - Exactly 1 source registered → always that source
    """
    if not _source_registry:
        return None
    if len(_source_registry) == 1:
        return list(_source_registry.keys())[0]

    # Build one description block per source
    blocks = []
    for src, samples in _source_registry.items():
        preview = " | ".join(s[:200] for s in samples[:2])
        blocks.append(f'- "{src}"\n  Content preview: {preview}')
    sources_section = "\n".join(blocks)

    prompt = (
        f"You are a document routing assistant. The following documents are indexed:\n\n"
        f"{sources_section}\n\n"
        f'User query: "{query}"\n\n'
        f"Which document should be searched to answer this query?\n"
        f"Rules:\n"
        f'- If the query clearly refers to ONE document, return: {{"source": "exact_filename.pdf"}}\n'
        f'- If the query spans multiple documents or is ambiguous, return: {{"source": null}}\n'
        f"Return ONLY valid JSON. No explanation."
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        detected = result.get("source")
        # Only trust the result if it exactly matches a known source
        if detected and detected in _source_registry:
            return detected
        return None
    except Exception as e:
        print(f"  [_auto_detect_source error] {e}")
        return None

def answer_query(query: str, vectorstore, source_filter: str = None) -> str:
    # Auto-detect which document(s) this query is about when the caller
    # does not provide an explicit source filter (the normal case).
    # For cross-document queries the LLM returns null → source_filter stays None
    # and retrieval searches across all indexed documents.
    if source_filter is None:
        source_filter = _auto_detect_source(query)
        if source_filter:
            print(f"  [auto-detect] query routed to: {source_filter!r}")
        else:
            print(f"  [auto-detect] cross-document or ambiguous — searching all sources")

    page_match = re.search(r'page\s+(\d+)', query, re.IGNORECASE)
    
    intent = _llm_classify_query(query)
    
    if intent.get("wants_image_data"):
        mode = "image"
    elif intent.get("wants_table_data"):
        mode = "hybrid" if intent.get("needs_analysis") else "table"
    else:
        mode = "text"
    if page_match:
        page_num = int(page_match.group(1))
        docs = get_page_docs(vectorstore, page_num, source_filter=source_filter)
        if not docs:
            docs = retrieve(query, vectorstore, intent, source_filter=source_filter)
    else:
        docs = retrieve(query, vectorstore, intent, source_filter=source_filter)
    
    # If the user is just asking for an image, bypass the blind LLM and return the image tags directly
    if intent.get("wants_image_data"):
        image_tags = []
        for r in docs:
            if r.metadata.get("type") == "image" and "image_path" in r.metadata:
                image_tags.append(f"[Source Image: {r.metadata['image_path']}]")
        if image_tags:
            return {"answer": "\n\n".join(image_tags), "mode": mode, "intent": intent}
        return {"answer": "No relevant image found in the document.", "mode": mode, "intent": intent}

    mode = "hybrid" if intent.get("wants_table_data") and intent.get("needs_analysis") else ("table" if intent.get("wants_table_data") else "text")

    formatted_contexts = []
    referenced_tables = []
    referenced_images = []


    for idx, doc in enumerate(docs):
        doc_type = doc.metadata.get("type", "text")
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        if doc_type == "table":
            path = doc.metadata.get("table_path", "")
            if path:
                referenced_tables.append(path)

            # Use the pristine raw markdown matrix for LLM context.
            # FAISS indexes a prose summary (better embedding), but the LLM
            # must receive the original grid so it can read exact cell values.
            # Fall back to page_content for records indexed before this change.
            raw_markdown = doc.metadata.get("raw_table_markdown", doc.page_content)
            formatted_contexts.append(
                f"[TABLE | p.{page} | {source}]\n{raw_markdown}"
            )


        elif doc_type == "image":
            path = doc.metadata.get("image_path", "")
            if path:
                referenced_images.append(path)

            formatted_contexts.append(
                f"[IMAGE | p.{page} | {source}]\n{doc.page_content}"
            )

        else:
            formatted_contexts.append(
                f"[TEXT | p.{page} | {source}]\n{doc.page_content}"
            )

    context = "\n\n".join(formatted_contexts)

    # FIX #5: Truncate between document blocks, not mid-string.
    # This prevents the LLM from receiving a half-cut markdown table row.
    MAX_CONTEXT_CHARS = 18000 if mode in ("table", "hybrid") else 8000
    if len(context) > MAX_CONTEXT_CHARS:
        # Walk backwards from the char limit to find the nearest block boundary
        cutoff = context.rfind("\n\n", 0, MAX_CONTEXT_CHARS)
        if cutoff == -1:
            cutoff = MAX_CONTEXT_CHARS
        context = context[:cutoff] + "\n\n[... context truncated ...]"

    # FIX #11: Choose prompt based on whether the question is analytical.
    # 'hybrid' mode queries explicitly need discussion, not just cell lookups.
    _needs_discussion = (mode == "hybrid") or intent.get("needs_analysis")
    
    wants_table = intent.get("format_as_table")

    if wants_table:
        prompt = f"""You are a precise data analyst. Answer using ONLY the provided context.
First, provide a short natural language explanation or discussion answering the query.
Then, provide a clean MARKDOWN TABLE containing the requested structured data.

Context:
{context}

Question: {query}

Instructions:
- Check ALL context sections: TABLE data, TEXT passages, and IMAGE captions.
- Quote exact numbers from tables where relevant.
- Output BOTH a natural language explanation and a markdown table.
- If the value is not in the context, say "Not found in context".

Answer:"""
    elif mode in ("table", "hybrid") and not _needs_discussion:
        prompt = f"""You are a precise data analyst. Answer using ONLY the provided context.
Give a concise, direct answer. Do not explain or add commentary.

Context:
{context}

Question: {query}

Important:
- Check ALL context sections: TABLE data, TEXT passages, and IMAGE captions.
- Some tables were extracted from a PDF and may have garbled column headers or
  merged cells (e.g. "S ca hange ti" = "Share capital / Exchange rate"). Do your
  best to read the values even when formatting is imperfect.
- For exchange rates, look for patterns like "1 EUR = `105.47" or "EUR `105.47".
- Quote the exact value or figure from whichever section contains it.
- If the answer genuinely cannot be found anywhere in the context, say "Not found in context".

Answer:"""
    elif mode in ("table", "hybrid") and _needs_discussion:
        # Analytical: compare, explain, discuss using both tables AND text
        prompt = f"""You are an expert research analyst. Answer using ONLY the provided context.

        The context contains both TABLE data and TEXT passages from the paper.
        Use both to give a complete, accurate answer.

        Context:
        {context}

        Question: {query}

        Instructions:
        - Quote exact numbers from tables where relevant.
        - Use the text passages to explain tradeoffs, reasons, and implications.
        - Structure your answer clearly: state the finding, then the comparison/discussion.
        - Do NOT invent any numbers or claims not present in the context.

        Answer:"""
    elif mode == "image":
        prompt = f"""You are an analyst reading figures from an annual/research report.

        The context below contains each figure's caption and any text that OCR was able to
        read from inside the image (axis labels, legend entries, numerical annotations, etc.).
        This does NOT include descriptions of colors, icon shapes, chart types, or spatial
        layout — only text that was legible inside the image.

        Context:
        {context}

        Question: {query}

        Instructions:
        - Answer using ONLY what is stated in the caption or OCR text above.
        - If the question asks about visual design elements (icons, colors, shapes, layout)
          that are not captured as readable text, say plainly:
          "The available context only includes the figure's caption and embedded text (via OCR),
          not a visual description, so this detail is not available."
        - Do NOT guess, infer, or invent any visual details.

        Answer:"""
    else:
        prompt = f"""Answer the question using ONLY the context.
        Do not invent facts. If the context is insufficient, say so.
        Be specific — mention exact names, numbers, and benchmarks where available.

        Context:
        {context}

        Question: {query}

        Answer:"""  

    response = _invoke_llm_with_retry(prompt)
    answer = response.content.strip()

    if not any(p in answer.lower() for p in _FAILURE_PHRASES):
        if "[Source Table:" not in answer and referenced_tables:
            answer += f"\n\n[Source Table: {referenced_tables[0]}]"
        if "[Source Image:" not in answer and referenced_images:
            answer += f"\n\n[Source Image: {referenced_images[0]}]"

    return {"answer": answer, "mode": mode, "intent": intent}

# --- TEST 3: BM25 Tokenizer Check ---
if __name__ == "__main__":
    print("Loading indices for Test 3...")
    vs = load_index()  # This initializes both FAISS and BM25
    
    test_query = "mAP@[.5, .95]"
    print(f"Executing sparse keyword search for: {test_query}")
    
    if bm25_retriever is not None:
        results = bm25_retriever.invoke(test_query)
        print(f"Test 3 Passed: BM25 found {len(results)} results!")
        if results:
            print(f"Top Match Snippet: {results[0].page_content[:80]}...")
    else:
        print("BM25 Retriever is not initialized. Make sure db/bm25_tables.pkl exists.")
# ------------------------------------