from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import re
import os
import time
import pickle
import pandas as pd
import io
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

bm25_retriever = None

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


def markdown_to_dataframe(md_string: str) -> pd.DataFrame:
    """
    Safely parse a markdown table string into a pandas DataFrame.
    """
    lines = [line.strip() for line in md_string.split('\n') if line.strip().startswith('|')]
    if not lines:
        return pd.DataFrame()
        
    # Find the separator row
    sep_idx = -1
    for i, line in enumerate(lines):
        if set(line.replace('|', '').replace(' ', '')) == {'-'}:
            sep_idx = i
            break
            
    if sep_idx > 1:
        # Multi-row header detected. Skip pandas-agent by returning empty df.
        return pd.DataFrame()
        
    # Remove the separator row
    if sep_idx != -1:
        lines = [line for i, line in enumerate(lines) if i != sep_idx]
        
    # Parse as CSV using | separator
    csv_str = '\n'.join(lines)
    # Remove leading/trailing |
    csv_str = '\n'.join([line.strip('|') for line in csv_str.split('\n')])
    try:
        df = pd.read_csv(io.StringIO(csv_str), sep='|')
        # Strip whitespace from column names and string cells
        df.columns = df.columns.str.strip()
        
        # Detect unusable headers (lots of Unnamed)
        unnamed = sum(1 for c in df.columns if str(c).startswith("Unnamed:"))
        if unnamed > len(df.columns) / 2:
            return pd.DataFrame()
            
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return df
    except Exception:
        return pd.DataFrame()


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


# Keywords that strongly signal the user wants a textual discussion/explanation
_ANALYTICAL_KEYWORDS = [
    "discuss", "explain", "why", "how does", "tradeoff", "trade-off",
    "reason", "suggest", "imply", "what does this", "elaborate",
    "describe", "what is the", "what are the", "relationship"
]


def _is_analytical_query(query: str) -> bool:
    """Return True if the query asks for explanation/discussion, not just a lookup."""
    q = query.lower()
    return any(term in q for term in _ANALYTICAL_KEYWORDS)


def _query_mode(query: str) -> str:
    """
    Return one of: 'hybrid', 'table', 'image', 'text'.

    'hybrid' is used when the query has table-related keywords but ALSO asks
    for discussion/explanation — meaning the LLM needs both table data AND
    surrounding prose to give a complete answer.
    """
    if is_image_query(query):
        return "image"
    if is_table_query(query):
        # If the query also wants analysis/discussion, use hybrid mode so
        # text context is always included alongside the tables.
        if _is_analytical_query(query):
            return "hybrid"
        return "table"
    return "text"


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

def retrieve(query: str, vectorstore, k: int = 6, source_filter: str = None) -> list:
    mode = _query_mode(query)

    fetch_k = 8 if source_filter is None else 12
    raw_docs = vectorstore.similarity_search(query, k=fetch_k)

    if source_filter:
        filtered = [d for d in raw_docs if _matches_source(d, source_filter)]
        if filtered:
            raw_docs = filtered

    raw_docs = _unique_docs(raw_docs)

    if mode in ("table", "hybrid"):
        table_docs = []
        seen_content = set()

        # --- HYBRID SEARCH: FAISS + BM25 ---
        table_pool_k = 12 if source_filter is None else 16
        faiss_table_docs = vectorstore.similarity_search(query, k=table_pool_k)

        bm25_docs = []
        if bm25_retriever is not None:
            # BM25 is excellent for exact keyword matches (e.g. "FUNSD F1").
            # FIX #4: Apply source filter to BM25 results BEFORE interleaving
            # so that irrelevant-paper tables don't consume context slots.
            bm25_raw = bm25_retriever.invoke(query)
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

            if len(table_docs) >= 4:
                break

        if not table_docs:
            table_docs = [d for d in raw_docs if d.metadata.get("type") == "table"]

        # FIX #2: Always reserve 2 slots for top text docs so the LLM has
        # prose context for analytical questions. Previously text was only
        # added when table_docs < 4, which almost never triggered.
        text_candidates = [
            d for d in raw_docs
            if d.metadata.get("type") == "text"
            and d.page_content.strip() not in seen_content
        ]
        # Dynamic context allocation slider.
        # Pure table lookups ("what is the highest F1?") get zero prose slots so
        # all 6 return positions are available for dense table data.
        # Analytical and hybrid queries still reserve prose for explanation.
        if mode == "hybrid":
            text_slots = 2
        elif _is_analytical_query(query):
            text_slots = 1
        else:
            text_slots = 0  # pure lookup — give every slot to tables

        top_text = (
            rerank_docs(query, _unique_docs(text_candidates), top_n=text_slots)
            if text_slots > 0
            else []
        )
        for doc in top_text:
            seen_content.add(doc.page_content.strip())
        table_docs = _unique_docs(table_docs) + top_text

        # BYPASS CROSS-ENCODER RERANKER FOR TABLES (It destroys markdown)
        return table_docs[:6]

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
    MAX_CONTEXT_CHARS = 9000 if mode in ("table", "hybrid") else 6500
    if len(context) > MAX_CONTEXT_CHARS:
        # Walk backwards from the char limit to find the nearest block boundary
        cutoff = context.rfind("\n\n", 0, MAX_CONTEXT_CHARS)
        if cutoff == -1:
            cutoff = MAX_CONTEXT_CHARS
        context = context[:cutoff] + "\n\n[... context truncated ...]"

    # FIX #11: Choose prompt based on whether the question is analytical.
    # 'hybrid' mode queries explicitly need discussion, not just cell lookups.
    _needs_discussion = (mode == "hybrid") or _is_analytical_query(query)

    if mode in ("table", "hybrid") and not _needs_discussion:
        # PANDAS CODE-GEN AGENT (Program-Aided Language)
        # Instead of asking the LLM to 'read' the text matrix, we formally parse
        # the retrieved tables into DataFrames and let the LLM generate/execute Python.
        
        dfs = []
        for doc in docs:
            if doc.metadata.get("type") == "table":
                raw_md = doc.metadata.get("raw_table_markdown", doc.page_content)
                df = markdown_to_dataframe(raw_md)
                if not df.empty:
                    dfs.append(df)
                    
        if dfs:
            try:
                # Use zero-shot-react-description since tool-calling is less stable on Groq
                agent = create_pandas_dataframe_agent(
                    llm, 
                    dfs, 
                    verbose=False,
                    allow_dangerous_code=True,
                    agent_type="zero-shot-react-description",
                    max_iterations=4
                )
                answer = agent.invoke({"input": f"Answer in MAXIMUM ONE SENTENCE with the exact value or metric. Do not include any explanations or code. Query: {query}"})
                answer_text = answer.get("output", str(answer))
                
                # BUG 2: Sanity check agent output
                rejected = False
                if "|" in answer_text or "(#:" in answer_text or "Agent stopped" in answer_text:
                    rejected = True
                else:
                    # Extract words and numbers from answer
                    ans_tokens = set(re.findall(r'\b[a-zA-Z0-9.\-]+\b', answer_text))
                    # Combine raw markdown
                    raw_md = " ".join([doc.metadata.get("raw_table_markdown", doc.page_content) for doc in docs if doc.metadata.get("type") == "table"])
                    # Check if at least one meaningful token appears in the raw table markdown
                    if "not found" not in answer_text.lower():
                        has_overlap = False
                        for t in ans_tokens:
                            if len(t) > 1 or t.isdigit():
                                if t in raw_md:
                                    has_overlap = True
                                    break
                        if not has_overlap and ans_tokens:
                            rejected = True
                
                if not rejected:
                    if "[Source Table:" not in answer_text and referenced_tables:
                        answer_text += f"\n\n[Source Table: {referenced_tables[0]}]"
                    return answer_text
                else:
                    print("Pandas Agent returned invalid output or stopped, falling back to standard LLM table lookup...")
                    
            except Exception as e:
                # Fallback to standard prompt if the agent hits a parsing error or iteration limit
                print(f"Pandas Agent failed ({e}), falling back to standard LLM table lookup...")
                
        # Pure lookup fallback prompt if agent failed or no valid DFS parsed
        prompt = f"""You are a precise data analyst. Answer using ONLY the provided context in MAXIMUM ONE SENTENCE. Do not explain.

        Context:
        {context}

        Question: {query}

        - Identify the correct table, row and column.
        - Quote the exact value from the table.
        - If the value is not in the context, say "Not found in context".

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
        prompt = f"""You are an expert at reading research paper figures.
        The context below contains figure captions AND detailed Vision Descriptions
        generated by analysing the actual image. Use ALL of this information to answer.

        Context:
        {context}

        Question: {query}

        Instructions:
        - Use the Vision Description to answer questions about what the figure shows.
        - Quote specific details: axis labels, values, trends, structural elements.
        - If a specific detail is genuinely not mentioned anywhere in the context, say so.
        - Do NOT say "visual content is not available" — use the Vision Description instead.

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

    if "[Source Table:" not in answer and referenced_tables:
        answer += f"\n\n[Source Table: {referenced_tables[0]}]"
    if "[Source Image:" not in answer and referenced_images:
        answer += f"\n\n[Source Image: {referenced_images[0]}]"

    return answer

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