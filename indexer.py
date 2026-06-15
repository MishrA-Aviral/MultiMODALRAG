from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import sqlite3
import os

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def index_text(pages: list, source_name: str, faiss_path: str = "db/faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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

def index_tables(db_path: str = "db/tables.db", faiss_path: str = "db/faiss_index"):
    if not os.path.exists(db_path):
        print("  No tables database found to index")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check table structure
    cursor.execute("PRAGMA table_info(tables)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "caption" in columns and "table_path" in columns:
        cursor.execute("SELECT source, page, content, caption, table_path FROM tables")
        rows = cursor.fetchall()
        docs = []
        for source, page, content, caption, table_path in rows:
            page_content = f"Table Caption: {caption}\n\nTable Data:\n{content}"
            docs.append(Document(
                page_content=page_content,
                metadata={
                    "source": source,
                    "page": page,
                    "table_path": table_path,
                    "type": "table"
                }
            ))
    else:
        cursor.execute("SELECT source, page, content FROM tables")
        rows = cursor.fetchall()
        docs = []
        for source, page, content in rows:
            docs.append(Document(
                page_content=content,
                metadata={"source": source, "page": page, "type": "table"}
            ))
    conn.close()
    
    if not docs:
        print("  No tables to index")
        return
        
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_path)
    print(f"  {len(docs)} table chunks indexed")

if __name__ == "__main__":
    # Test script for indexing a single PDF directly
    import sys
    from extractor import extract_text, extract_tables, extract_images
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        print(f"Processing test PDF: {pdf}")
        tbl_recs = extract_tables(pdf)
        img_recs = extract_images(pdf)
        text_recs = extract_text(pdf, tbl_recs)
        
        # Reset index
        import shutil
        if os.path.exists("db/faiss_index"):
            shutil.rmtree("db/faiss_index")
            
        index_text(text_recs, os.path.basename(pdf))
        index_image_captions(img_recs)
        index_tables()