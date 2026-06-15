import pandas as pd
from openpyxl import load_workbook

def read_queries(filepath: str) -> dict:
    """Read queries from all sheets in the Excel file.

    Returns:
        A dict mapping sheet name → {"queries": [str, ...], "source": str | None}.

        "source" is the PDF filename (e.g. "2204.08387v3.pdf") used to restrict
        retrieval to a single paper.  It is read from column C, whose header must
        be "Source".  The first non-null value in that column is used for the whole
        sheet.  If the column is absent or empty, "source" is None (no filtering).
    """
    xl = pd.ExcelFile(filepath)
    all_queries = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        if "Query" not in df.columns:
            continue

        queries = df["Query"].dropna().tolist()

        # Read optional Source column (first non-null value applies to the sheet)
        source = None
        if "Source" in df.columns:
            source_vals = df["Source"].dropna().tolist()
            if source_vals:
                source = str(source_vals[0]).strip()

        all_queries[sheet] = {"queries": queries, "source": source}
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
    for sheet, data in queries.items():
        print(f"Sheet: {sheet}  |  Source filter: {data['source']}")
        for q in data["queries"]:
            print(f"  - {q[:80]}...")