from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=2048,
    api_key=os.getenv("GROQ_API_KEY")
)

def load_index(faiss_path: str = "db/faiss_index"):
    return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

def retrieve(query: str, vectorstore, k: int = 6) -> list:
    # Standard semantic search
    docs = vectorstore.similarity_search(query, k=k)
    
    # Also grab top 2 table chunks specifically
    all_docs = vectorstore.docstore._dict
    table_docs = [doc for doc_id, doc in all_docs.items() 
                  if doc.metadata.get("type") == "table"]
    
    if table_docs:
        table_vs = vectorstore.similarity_search_with_score(query, k=20)
        table_hits = [doc for doc, score in table_vs 
                      if doc.metadata.get("type") == "table"][:2]
        
        seen = {d.page_content for d in docs}
        for doc in table_hits:
            if doc.page_content not in seen:
                docs.append(doc)
                seen.add(doc.page_content)
    
    return docs

def get_page_docs(vectorstore, page_num: int) -> list:
    all_docs = []
    docstore = vectorstore.docstore._dict
    for doc_id, doc in docstore.items():
        if doc.metadata.get("page") == page_num:
            all_docs.append(doc)
    return all_docs

def answer_query(query: str, vectorstore) -> str:
    # Handle page-specific queries
    page_match = re.search(r'page\s+(\d+)', query, re.IGNORECASE)
    if page_match:
        page_num = int(page_match.group(1))
        docs = get_page_docs(vectorstore, page_num)
        if not docs:
            docs = retrieve(query, vectorstore)
    else:
        docs = retrieve(query, vectorstore)

    # Format context with source metadata
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
            formatted_contexts.append(
                f"### Context {idx+1} [Type: Table, Source: {source}, Page: {page}]\n{doc.page_content}"
            )
        elif doc_type == "image":
            path = doc.metadata.get("image_path", "")
            if path:
                referenced_images.append(path)
            formatted_contexts.append(
                f"### Context {idx+1} [Type: Image, Source: {source}, Page: {page}]\n{doc.page_content}"
            )
        else:
            formatted_contexts.append(
                f"### Context {idx+1} [Type: Text, Source: {source}, Page: {page}]\n{doc.page_content}"
            )

    context = "\n\n".join(formatted_contexts)

    prompt = f"""You are a research assistant analyzing academic papers. Answer the question based ONLY on the provided context.
Be specific and extract key information directly from the context.
Do NOT create, invent, or summarize data into tables that are not present in the context.
Only reference tables or figures that are explicitly present in the context below.

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    # Append source references if not already mentioned
    if "[Source Table:" not in answer and referenced_tables:
        answer += f"\n\n[Source Table: {referenced_tables[0]}]"
    if "[Source Image:" not in answer and referenced_images:
        answer += f"\n\n[Source Image: {referenced_images[0]}]"

    return answer

if __name__ == "__main__":
    print("Loading index...")
    vs = load_index()
    print("Index loaded. Testing queries...\n")
    test_queries = [
        "What is the main contribution of this paper?",
        "What datasets were used?",
        "Summarize the results"
    ]
    for q in test_queries:
        print(f"Q: {q}")
        print(f"A:\n{answer_query(q, vs)}")
        print("-" * 60)