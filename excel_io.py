import re
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

    Plain-text answers are written to column B of the corresponding query row.
    Markdown table answers are written starting below all query rows, with the
    query number noted in column A so the reader can match them up.
    Column B of the query row will say "→ See table below (row N)".
    """
    wb = load_workbook(filepath)
    for sheet_name, qa_pairs in answers.items():
        ws = wb[sheet_name]
        if ws["B1"].value != "Answer":
            ws["B1"] = "Answer"

        # First pass: write plain answers and collect tables
        # Tables are appended after all query rows to avoid collisions.
        total_queries = len(qa_pairs)
        # Tables start well below the last query row (+ 3 blank rows gap)
        current_table_row = total_queries + 4

        for i, answer in enumerate(qa_pairs, start=2):
            lines = [line for line in answer.split('\n')]
            table_lines = [line.strip() for line in lines if line.strip().startswith('|')]

            if len(table_lines) >= 3:
                # Mark the query row, then dump the table below
                ws.cell(row=i, column=2, value=f"→ See table below (row {current_table_row})")

                # Write a label in col A so the reader knows which query owns this table
                ws.cell(row=current_table_row, column=1, value=f"[Query {i - 1}]")

                for line in table_lines:
                    # Skip markdown separator lines like |---|---|
                    if re.match(r'^\|[\s\-:|]+\|[\s\-:|]*$', line):
                        continue

                    # Parse pipe-delimited cells cleanly
                    raw_cols = line.split('|')
                    # Strip leading/trailing empty elements from surrounding '|'
                    if raw_cols and not raw_cols[0].strip():
                        raw_cols = raw_cols[1:]
                    if raw_cols and not raw_cols[-1].strip():
                        raw_cols = raw_cols[:-1]

                    # Write starting at column B (col 2) to leave col A free for labels
                    for col_idx, col_val in enumerate(raw_cols, start=2):
                        ws.cell(row=current_table_row, column=col_idx, value=col_val.strip())
                    current_table_row += 1

                current_table_row += 1  # blank row between tables
            else:
                ws.cell(row=i, column=2, value=answer)

    wb.save(filepath)
    print(f"Answers written to {filepath}")


if __name__ == "__main__":
    queries = read_queries("Queries.xlsx")
    for sheet, qs in queries.items():
        print(f"Sheet: {sheet}")
        for item in qs:
            print(f"  - {item['query'][:80]}... [Source: {item['source']}]")