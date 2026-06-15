from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import re
import os
import time
import pickle

bm25_retriever = None

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=2048,
    api_key=os.getenv("GROQ_API_KEY")
)

def load_index(faiss_path: str = "db/faiss_index"):
    global bm25_retriever
    bm25_path = "db/bm25_tables.pkl"
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
            
    return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)


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


def is_image_query(query: str) -> bool:
    q = query.lower()
    image_keywords = [
        "figure", "fig", "diagram", "chart", "graph", "plot",
        "architecture", "workflow", "flow", "visual", "illustration"
    ]
    return any(term in q for term in image_keywords)


def is_table_query(query: str) -> bool:
    q = query.lower()
    table_keywords = [
        "table", "benchmark", "comparison", "compare", "compared",
        "highest", "lowest", "best", "worst", "gain", "improvement",
        "difference", "increase", "decrease", "rank", "ranking",
        "average", "score", "accuracy", "f1", "map", "parameter",
        "ablation", "which category", "which model", "which approach"
    ]
    return any(term in q for term in table_keywords)


def _query_mode(query: str) -> str:
    """
    Return one of: 'table', 'image', 'text'
    """
    if is_table_query(query):
        return "table"
    if is_image_query(query):
        return "image"
    return "text"


def _matches_source(doc, source_filter: str) -> bool:
    """
    Return True if the doc's source metadata matches the filter filename.
    """
    import os
    doc_source = os.path.basename(doc.metadata.get("source", ""))
    return doc_source == os.path.basename(source_filter)


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

def retrieve(query: str, vectorstore, k: int = 6, source_filter: str = None) -> list:
    mode = _query_mode(query)

    fetch_k = 8 if source_filter is None else 12
    raw_docs = vectorstore.similarity_search(query, k=fetch_k)

    if source_filter:
        filtered = [d for d in raw_docs if _matches_source(d, source_filter)]
        if filtered:
            raw_docs = filtered

    raw_docs = _unique_docs(raw_docs)

    if mode == "table":
        table_docs = []
        seen_content = set()

        # --- HYBRID SEARCH: FAISS + BM25 ---
        table_pool_k = 12 if source_filter is None else 16
        faiss_table_docs = vectorstore.similarity_search(query, k=table_pool_k)
        
        bm25_docs = []
        if bm25_retriever is not None:
            # BM25 is excellent for exact keyword matches (e.g. "FUNSD F1")
            bm25_docs = bm25_retriever.invoke(query)

        # Interleave Dense (FAISS) and Sparse (BM25) results
        combined_raw = []
        for d1, d2 in zip(bm25_docs, faiss_table_docs):
            combined_raw.extend([d1, d2])
        combined_raw.extend(faiss_table_docs[len(bm25_docs):])
        combined_raw.extend(bm25_docs[len(faiss_table_docs):])

        for doc in combined_raw:
            if doc.metadata.get("type") != "table":
                continue
            if source_filter and not _matches_source(doc, source_filter):
                continue

            content = doc.page_content.strip()
            if content in seen_content:
                continue

            table_docs.append(doc)
            seen_content.add(content)

            if len(table_docs) >= 4:
                break

        if not table_docs:
            table_docs = [d for d in raw_docs if d.metadata.get("type") == "table"]

        if len(table_docs) < 4:
            for doc in raw_docs:
                if doc.metadata.get("type") == "text" and doc.page_content.strip() not in seen_content:
                    table_docs.append(doc)
                    seen_content.add(doc.page_content.strip())
                if len(table_docs) >= 6:
                    break

        # BYPASS CROSS-ENCODER RERANKER FOR TABLES (It destroys markdown)
        return _unique_docs(table_docs)[:6]

    if mode == "image":
        image_docs = [d for d in raw_docs if d.metadata.get("type") == "image"]
        if not image_docs:
            image_docs = raw_docs
        return rerank_docs(query, _unique_docs(image_docs), top_n=3)

    text_docs = [d for d in raw_docs if d.metadata.get("type") == "text"]
    if not text_docs:
        text_docs = raw_docs
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


def _invoke_llm_with_retry(prompt: str, retries: int = 3, base_sleep: int = 6):
    """
    Retry on Groq rate-limit errors.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            msg = str(exc).lower()
            if "rate limit" in msg or "tokens per minute" in msg or "429" in msg:
                last_exc = exc
                time.sleep(base_sleep * (attempt + 1))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM invocation failed.")


def answer_query(query: str, vectorstore, source_filter: str = None) -> str:
    page_match = re.search(r'page\s+(\d+)', query, re.IGNORECASE)
    mode = _query_mode(query)

    if page_match:
        page_num = int(page_match.group(1))
        docs = get_page_docs(vectorstore, page_num, source_filter=source_filter)
        if not docs:
            docs = retrieve(query, vectorstore, source_filter=source_filter)
    else:
        docs = retrieve(query, vectorstore, source_filter=source_filter)

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
            
            # BYPASS DEDUPLICATION: Pass raw Markdown directly to LLM
            formatted_contexts.append(
                f"[TABLE | p.{page} | {source}]\n{doc.page_content}"
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

    MAX_CONTEXT_CHARS = 9000 if mode == "table" else 6500
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"

    if mode == "table":
        prompt = f"""You are an expert data analyst. Use ONLY the provided context to answer.

Context:
{context}

Question: {query}

Instructions:
1. Identify the exact table, row, and column needed.
2. Trace the intersection carefully.
3. If the value is missing, explicitly state "Not provided in context".
4. Write your step-by-step reasoning inside <thinking> tags.
5. Provide the final, exact answer outside the tags.

Answer:"""
    elif mode == "image":
        prompt = f"""Answer the question using ONLY the context.

For figure/image questions:
- Use the caption and nearby text only.
- If the visual content is not available, say that clearly.
- Do not invent details.

Context:
{context}

Question: {query}

Answer in a short, direct form:"""
    else:
        prompt = f"""Answer the question using ONLY the context.
Do not invent facts. If the context is insufficient, say so.

Context:
{context}

Question: {query}

Answer in a short, direct form:"""

    response = _invoke_llm_with_retry(prompt)
    answer = response.content.strip()

    if "[Source Table:" not in answer and referenced_tables:
        answer += f"\n\n[Source Table: {referenced_tables[0]}]"
    if "[Source Image:" not in answer and referenced_images:
        answer += f"\n\n[Source Image: {referenced_images[0]}]"

    return answer