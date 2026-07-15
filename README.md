## MultiModal Research Paper RAG Pipeline

An AI-powered **Retrieval-Augmented Generation (RAG)** pipeline designed for deep analysis of academic PDF papers. It extracts and indexes **text, tables, and figures** from PDFs into a unified vector database, then answers structured queries from an Excel file — writing answers back automatically.

---

## Features

- **Multi-Modal Extraction** — Text, tables, and figures are cleanly extracted and processed natively.
- **Table-Aware Text Extraction** — Table and figure regions are geometrically masked out from body text to prevent corrupt, misaligned data from poisoning the context.
- **Table Extraction (TATR)** — Uses Microsoft's **Table Transformer v1.1-all** (TATR) computer vision model to detect and perfectly parse tables into strict Markdown grids.
- **Table Summarization (Vector Blinding Fix)** — Raw markdown grids are notoriously hard for Dense embeddings to retrieve. Each table is summarized into a dense prose description by **Llama 3 8B** before embedding, dramatically improving semantic retrieval recall.
- **Vision Descriptions** — Extracted vector figures and charts are described natively, making visuals fully searchable.
- **Hybrid Retrieval** — FAISS (Dense) + BM25 (Sparse) are interleaved for lookups, with BM25 getting priority for exact keyword/metric matches (e.g., `mAP@[.5,.95]`).
- **Cross-Encoder Reranking** — Retrieved text and image chunks are reranked using `ms-marco-MiniLM-L-6-v2` before being passed to the LLM.
- **Query-Mode Routing** — Queries are automatically classified as `text`, `table`, `image`, or `hybrid` (analytical) and routed to the optimal retrieval strategy.
- **Statistical Computation Layer** — A deterministic Python math layer. If a query asks for statistics ("mean", "median", "delta", "percentile"), the LLM extracts the raw cell values, and **NumPy/SciPy computes the exact answer**, eliminating LLM math hallucinations.
- **Pandas Code-Gen Agent** — For pure table lookups, the LLM generates and executes Python/pandas code against parsed DataFrames (Program-Aided Language) for exact cell-level answers.
- **Structured Q&A from Excel** — Reads queries from `Queries.xlsx`, answers them via the RAG pipeline, and writes answers back to the same file iteratively.
- **Cross-Document or Source Filtered Retrieval** — Natively supports searching across multiple PDFs or filtering to a specific paper.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **PDF Parsing** | PyMuPDF (`fitz`) |
| **Embeddings** | `BAAI/bge-base-en-v1.5` (HuggingFace) |
| **Vector Store** | FAISS (persisted locally) |
| **Sparse Retrieval** | BM25 (persisted as `db/bm25_tables.pkl`) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Table Storage** | SQLite (`db/tables.db`) |
| **Table Extraction OCR** | Microsoft Table Transformer (`microsoft/table-transformer-structure-recognition-v1.1-all`) |
| **LLM — Answering** | Llama-3.3-70B Versatile via Groq API |
| **LLM — Table Summarization** | Llama 3 8B (8192 context) via Groq API |
| **Statistical Math** | NumPy, SciPy |
| **Pandas Agent** | LangChain `create_pandas_dataframe_agent` |
| **Excel I/O** | pandas, openpyxl |

---

## Project Structure

```
MultiModalRAG/
├── main.py           # Entry point: index PDFs or run Q&A pipeline
├── extractor.py      # Multi-modal PDF extraction (text, TATR tables, images)
├── indexer.py        # FAISS + BM25 indexing for text, tables (with LLM summaries), image captions
├── retriever.py      # Hybrid retrieval, reranking, Pandas Agent, and Statistical Computation
├── excel_io.py       # Read queries (with optional Source filter) / write answers to Queries.xlsx
├── Queries.xlsx      # Input: Query + optional Source columns; Output: answers written to column B
├── data/
│   ├── papers/             # Place your input PDF files here
│   ├── extracted_tables/   # Auto-generated markdown table files
│   └── extracted_images/   # Auto-generated figure images
├── db/               # Auto-generated (gitignored)
│   ├── faiss_index/        # Persisted FAISS vector index
│   ├── tables.db           # SQLite table store
│   └── bm25_tables.pkl     # Persisted BM25 retriever for tables
└── requirements.txt
```

---

## How It Works

### Indexing Phase (`python main.py index`)

1. For each PDF in `data/papers/`:
   - **Tables** are extracted using the Microsoft Table Transformer (TATR) `v1.1-all` model. Each table is converted to a pristine Markdown file and stored in SQLite. It is then summarized into a dense prose paragraph by **Llama 3 8B** before embedding.
   - **Images** are extracted (vector figures and raster maps). Each image is mapped to its nearest figure caption, and a rich Vision Description is created.
   - **Text** is extracted page-by-page. Table and figure bounding boxes are strictly masked out to prevent corrupted extraction.
2. Text chunks, Vision Descriptions, and table prose summaries (with the raw markdown grid stored silently as metadata) are all embedded using `BAAI/bge-base-en-v1.5` and indexed into a single FAISS store.
3. A **BM25** sparse index is built globally from the table summaries and persisted to disk.

### Query Phase (`python main.py`)

1. Loads the FAISS index and BM25 retriever.
2. Reads queries from `Queries.xlsx` (Optional `Source` column to filter by PDF filename; otherwise it searches globally).
3. The query is **routed** to one of four modes: `text`, `table`, `image`, or `hybrid` (analytical).
4. If a statistical math operation is detected ("average", "delta", "percentile"), the pipeline extracts raw numerical cell values from the markdown table and uses Python (`numpy`) to output a deterministic math answer.
5. For pure **table lookups**, a Pandas Code-Gen Agent parses retrieved tables into DataFrames and generates Python to retrieve exact cells.
6. For **analytical hybrid queries**, context is truncated perfectly at block boundaries, combined with surrounding text chunks, and passed to **Llama 3.3 70B** to generate a highly cohesive, sourced response.
7. Answers are written back iteratively to `Queries.xlsx`.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/MishrA-Aviral/MultiModalRAG.git
cd MultiModalRAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com).

### 5. Add your PDFs

Place PDF files inside `data/papers/`.

### 6. Run

**Index the PDFs first:**
```bash
python main.py index
```

**Then run the Q&A pipeline against `Queries.xlsx`:**
```bash
python main.py
```

---

## Queries.xlsx Format

Each sheet should have a `Query` column. Answers will be written to column B automatically.
An optional `Source` column can restrict a query to a specific PDF (by filename).

| Query | Source | Answer |
|---|---|---|
| What is the main contribution of this paper? | 1905.11946v5.pdf | *(written by the pipeline)* |
| What datasets were used for evaluation? | | *(written by the pipeline)* |

---

## Notes

- The first run downloads the `BAAI/bge-base-en-v1.5` embedding model, `ms-marco-MiniLM-L-6-v2` reranker, and `microsoft/table-transformer` models. Subsequent runs use the cache.
- Re-running `python main.py index` resets and rebuilds the entire index from scratch (FAISS, SQLite, and BM25).
- The FAISS index, BM25 pickle, and SQLite DB are gitignored — they are runtime artifacts and must be generated locally.
- A delay is inserted between queries during the Q&A phase to respect Groq rate limits.
