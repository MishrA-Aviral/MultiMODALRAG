import pandas as pd
from openpyxl import load_workbook

def read_queries(filepath: str) -> dict:
    xl = pd.ExcelFile(filepath)
    all_queries = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        if "Query" in df.columns:
            queries = df["Query"].dropna().tolist()
            all_queries[sheet] = queries
    return all_queries

def write_answers(filepath: str, answers: dict):
    wb = load_workbook(filepath)
    for sheet_name, qa_pairs in answers.items():
        ws = wb[sheet_name]
        if ws["B1"].value != "Answer":
            ws["B1"] = "Answer"
        for i, answer in enumerate(qa_pairs, start=2):
            ws.cell(row=i, column=2, value=answer)
    wb.save(filepath)
    print(f"Answers written to {filepath}")
    
if __name__ == "__main__":
    queries = read_queries("Queries.xlsx")
    print(queries)