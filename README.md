## Multi-Modal Research Paper RAG Pipeline

An AI-powered **Retrieval-Augmented Generation (RAG)** pipeline designed for deep analysis of academic PDF papers. It extracts and indexes **text, tables, and figures** from PDFs into a unified vector database, then answers structured queries from an Excel file — writing answers back automatically.

---

## Features

- **Multi-modal extraction** — text, tables (gridline + alignment-based), and figures (raster + vector/LaTeX diagrams)
- **Table-aware text extraction** — table regions are masked out from text to avoid duplication
- **Caption matching** — figures and tables are automatically paired with their nearest `Figure N:` / `Table N:` captions
- **Unified FAISS vector index** — text chunks, table markdown, and image captions are all stored in one searchable index
- **Structured Q&A from Excel** — reads queries from `Queries.xlsx`, answers them via the RAG pipeline, and writes answers back to the same file
- **Page-specific queries** — queries containing `"page N"` retrieve documents from that specific page
- **LLM: Llama 3.1 8B via Groq** — fast inference with a constrained, context-faithful prompt

---

## Tech Stack

| Layer | Technology |
|---|---|
| **PDF Parsing** | PyMuPDF (`fitz`), pdfplumber |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Store** | FAISS (persisted locally) |
| **Table Storage** | SQLite (`db/tables.db`) |
| **LLM** | Llama 3.1 8B Instant via Groq API |
| **Orchestration** | LangChain |
| **Excel I/O** | pandas, openpyxl |

---

## Project Structure

```
ChatwithPDF-RAG/
├── main.py           # Entry point: index PDFs or run Q&A pipeline
├── extractor.py      # Multi-modal PDF extraction (text, tables, images)
├── indexer.py        # FAISS indexing for text, tables, image captions
├── retriever.py      # Semantic retrieval + Groq LLM answer generation
├── excel_io.py       # Read queries from / write answers to Queries.xlsx
├── Queries.xlsx      # Input: queries per sheet; Output: answers written back
├── data/
│   └── papers/       # Place your input PDF files here
├── db/               # Auto-generated (gitignored)
│   ├── faiss_index/  # Persisted FAISS vector index
│   └── tables.db     # SQLite table store
└── requirements.txt
```

---

## How It Works

### Indexing Phase (`python main.py index`)

1. For each PDF in `data/papers/`:
   - **Tables** are extracted using two strategies: horizontal-line clustering and pdfplumber gridline detection. Duplicate regions are de-duplicated. Each table is saved as a Markdown file and stored in SQLite.
   - **Images** are extracted: raster images by xref, and vector figures (tikz/pgfplots) by rendering drawing bounding boxes. Each image is matched to its nearest figure caption.
   - **Text** is extracted page-by-page with table regions masked out, and hyphenation artifacts cleaned.
2. Text chunks, image captions, and table Markdown are all embedded and indexed into a single FAISS store.

### Query Phase (`python main.py`)

1. Loads the FAISS index from `db/faiss_index/`.
2. Reads queries from each sheet of `Queries.xlsx` (expects a `Query` column).
3. For each query, retrieves the top-k relevant chunks (with a boost for table chunks).
4. Passes context + query to Llama 3.1 8B via Groq with a strict, hallucination-resistant prompt.
5. Writes answers back to column B of the corresponding sheet in `Queries.xlsx`.

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

| Query | Answer |
|---|---|
| What is the main contribution of this paper? | *(written by the pipeline)* |
| What datasets were used for evaluation? | *(written by the pipeline)* |

---

## Notes

- The first run downloads the `all-MiniLM-L6-v2` embedding model (~90MB). Subsequent runs use the cache.
- Re-running `python main.py index` resets and rebuilds the entire index from scratch.
- The FAISS index and SQLite DB are gitignored — they are runtime artifacts and must be generated locally.

---

## Contact

Feel free to connect or reach out for feedback!
