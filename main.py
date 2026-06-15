from excel_io import read_queries, write_answers
from extractor import extract_text, extract_tables, extract_images
from indexer import index_text, index_tables, index_image_captions
from retriever import load_index, answer_query
import os
import shutil
import time

def index_papers(papers_dir: str = "data/papers", reset_index: bool = True):
    pdfs = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    if not pdfs:
        print("No PDFs found in data/papers/")
        return
        
    if reset_index:
        print("Resetting vector database index...")
        if os.path.exists("db/faiss_index"):
            shutil.rmtree("db/faiss_index")
        if os.path.exists("db/tables.db"):
            os.remove("db/tables.db")
        # Also clean up previously extracted artifacts
        if os.path.exists("data/extracted_tables"):
            shutil.rmtree("data/extracted_tables")
        if os.path.exists("data/extracted_images"):
            shutil.rmtree("data/extracted_images")
            
    for pdf in pdfs:
        pdf_path = os.path.join(papers_dir, pdf)
        print(f"\nProcessing: {pdf}")
        
        print("  Step 1: Extracting tables...")
        tables = extract_tables(pdf_path)
        print(f"    {len(tables)} tables extracted")
        
        print("  Step 2: Extracting images and matching captions...")
        images = extract_images(pdf_path)
        print(f"    {len(images)} images extracted")
        
        print("  Step 3: Extracting clean text (filtering layout tables)...")
        pages = extract_text(pdf_path, tables)
        print(f"    {len(pages)} clean text pages extracted")
        
        print("  Step 4: Indexing text...")
        index_text(pages, pdf)
        
        print("  Step 5: Indexing image captions...")
        index_image_captions(images)
        
    # Index tables at the end (from the sqlite DB which accumulates all extracted tables)
    print("\nStep 6: Indexing all accumulated tables...")
    index_tables()
    print("Indexing Complete.")

def run_pipeline(excel_path: str = "Queries.xlsx"):
    print("Loading index...")
    vectorstore = load_index()
    print("Reading queries from Excel...")
    all_queries = read_queries(excel_path)
    answers = {}
    for sheet, sheet_data in all_queries.items():
        queries = sheet_data["queries"]
        source_filter = sheet_data["source"]
        if source_filter:
            print(f"\nProcessing {sheet} ({len(queries)} queries) "
                  f"[source filter: {source_filter}]...")
        else:
            print(f"\nProcessing {sheet} ({len(queries)} queries) "
                  f"[no source filter – searching all papers]...")
        sheet_answers = []
        for q in queries:
            print(f"  Q: {q}")
            answer = answer_query(q, vectorstore, source_filter=source_filter)
            print(f"  A: {answer[:150]}...")
            sheet_answers.append(answer)
            time.sleep(3)

        answers[sheet] = sheet_answers
    print("\nWriting answers back to Excel...")
    write_answers(excel_path, answers)
    print("Done! Open Queries.xlsx to see answers.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "index":
        index_papers()
    else:
        run_pipeline()