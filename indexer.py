from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import sqlite3
import os
import pickle
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

def index_image_captions(image_records: list, faiss_path: str = "db/faiss_index"):
    docs = []
    for img in image_records:
        page_content = f"Image Caption: {img['caption']}"
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

        caption_line = f'The table caption is: "{caption}".\n' if caption else ""
        prompt = (
            f"You are a research paper analyst. {caption_line}"
            f"Below is a markdown table extracted from an academic paper.\n\n"
            f"{markdown_content}\n\n"
            f"Write a single dense prose paragraph (max 5 sentences) that captures:\n"
            f"- The main benchmark, task, or topic being evaluated.\n"
            f"- Every model name, architecture, or approach listed in the rows or columns.\n"
            f"- The exact names of all evaluation metrics (e.g. F1, accuracy, mAP, BLEU).\n"
            f"- Any notably high or low values worth calling out.\n"
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
    
    docs = []
    if "caption" in columns and "table_path" in columns:
        cursor.execute("SELECT source, page, content, caption, table_path FROM tables")
        rows = cursor.fetchall()
        for source, page, content, caption, table_path in rows:
            # The original markdown string — preserved verbatim for LLM context injection
            raw_table_markdown = f"Table Caption: {caption}\nPage: {page}\nTable Data:\n{content}"
            # Generate a semantically rich prose summary for FAISS embedding
            print(f"    Summarizing table from {source} p.{page}…")
            summary = generate_table_summary(content, caption)
            docs.append(Document(
                page_content=summary,
                metadata={
                    "source": source,
                    "page": page,
                    "table_path": table_path,
                    "type": "table",
                    "raw_table_markdown": raw_table_markdown,
                }
            ))
    else:
        cursor.execute("SELECT source, page, content FROM tables")
        rows = cursor.fetchall()
        for source, page, content in rows:
            # Legacy schema without caption/table_path — summarize with no caption
            print(f"    Summarizing table from {source} p.{page}…")
            summary = generate_table_summary(content)
            docs.append(Document(
                page_content=summary,
                metadata={
                    "source": source,
                    "page": page,
                    "type": "table",
                    "raw_table_markdown": content,
                }
            ))
    conn.close()
    
    if not docs:
        print("  No tables to index")
        return
        
    # --- FAISS DENSE INDEX ---
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_path)
    
    # --- BM25 SPARSE INDEX (NEW) ---
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_path = "db/bm25_tables.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    print(f"  {len(docs)} table chunks indexed (FAISS + BM25)")