import pandas as pd
from openpyxl import load_workbook

def read_queries(filepath: str) -> dict:
    """Read queries from all sheets in the Excel file.

    Returns:
        A dict mapping sheet name → list of dicts: [{"query": str, "source": str | None}, ...]
    """
    xl = pd.ExcelFile(filepath)
    all_queries = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        if "Query" not in df.columns:
            continue

        sheet_queries = []
        has_source = "Source" in df.columns
        for _, row in df.iterrows():
            q = row.get("Query")
            if pd.isna(q):
                continue
            src = None
            if has_source and not pd.isna(row.get("Source")):
                src = str(row.get("Source")).strip()
            sheet_queries.append({"query": str(q), "source": src})

        all_queries[sheet] = sheet_queries
    return all_queries


def write_answers(filepath: str, answers: dict):
    """Write answers back to the Excel file.

    ``answers`` maps sheet name → list of answer strings (positionally aligned
    with the query rows starting at row 2).
    """
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
    for sheet, qs in queries.items():
        print(f"Sheet: {sheet}")
        for item in qs:
            print(f"  - {item['query'][:80]}... [Source: {item['source']}]")