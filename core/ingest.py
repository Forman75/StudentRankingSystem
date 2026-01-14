from __future__ import annotations
import pandas as pd
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Optional
from openpyxl import load_workbook

def _sheet_to_matrix_with_merged(wb_bytes: bytes, sheet_name: str, max_rows: Optional[int]=None) -> List[List[Any]]:
    wb = load_workbook(BytesIO(wb_bytes), read_only=False, data_only=True)
    ws = wb[sheet_name]

    merged_map = {}
    for r in ws.merged_cells.ranges:
        min_col, min_row, max_col, max_row = r.bounds
        top_val = ws.cell(min_row, min_col).value
        for rr in range(min_row, max_row + 1):
            for cc in range(min_col, max_col + 1):
                merged_map[(rr, cc)] = top_val

    rows = []
    max_r = ws.max_row
    max_c = ws.max_column
    if max_rows is not None:
        max_r = min(max_r, max_rows)

    for r in range(1, max_r + 1):
        row_vals = []
        for c in range(1, max_c + 1):
            v = ws.cell(r, c).value
            if (r, c) in merged_map and (v is None or str(v).strip() == ""):
                v = merged_map[(r, c)]
            row_vals.append(v)
        rows.append(row_vals)

    return rows

def load_tables_from_uploads(uploads) -> List[Dict[str, Any]]:
    tables = []
    for up in uploads:
        name = up.name
        data = up.getvalue()

        if name.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(data), sep=None, engine="python")
            df.insert(0, "_origin_row", range(1, len(df) + 1))
            tables.append({"source_name": name, "sheet_name": "CSV", "df_raw": df})
            continue

        xls = pd.ExcelFile(BytesIO(data))
        for sheet in xls.sheet_names:
            try:
                matrix = _sheet_to_matrix_with_merged(data, sheet_name=sheet, max_rows=None)
                df_raw = pd.DataFrame(matrix)
            except Exception:
                df_raw = pd.read_excel(BytesIO(data), sheet_name=sheet, header=None)

            df_raw.insert(0, "_origin_row", range(1, len(df_raw) + 1))
            tables.append({"source_name": name, "sheet_name": sheet, "df_raw": df_raw})

    return tables
