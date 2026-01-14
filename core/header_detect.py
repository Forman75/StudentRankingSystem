from __future__ import annotations
import pandas as pd
import re
from typing import Tuple
from .utils import try_parse_date, norm_text

DATE_LIKE_RE = re.compile(
    r"^\s*\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\s*$|^\s*\d{4}[./-]\d{1,2}[./-]\d{1,2}\s*$"
)

def _row_date_score(row: pd.Series) -> int:
    score = 0
    for v in row.tolist():
        if v is None:
            continue
        if try_parse_date(v) is not None:
            score += 2
        else:
            txt = norm_text(v)
            if DATE_LIKE_RE.match(txt):
                score += 2
    return score

def _row_headerish_score(row: pd.Series) -> float:
    nonnull = row.notna().sum()
    as_str = row.astype(str)
    lens = as_str.str.len().fillna(0)
    shortish = lens.between(2, 60).sum()
    not_numeric = (~as_str.str.match(r"^\s*[-+]?\d+([.,]\d+)?\s*$", na=False)).sum()
    return float(nonnull) + 0.7 * float(shortish) + 0.2 * float(not_numeric)

def detect_header_block(df_raw: pd.DataFrame, max_scan_rows: int = 80) -> Tuple[int, int]:
    # (start_row_0based, header_height): Приоритет ранних строк (обычно заголовок с 1 строки). Высота заголовка авто-сканируется до 12.
    n = min(max_scan_rows, len(df_raw))
    best = (0, 1)
    best_score = -1e9

    for start in range(0, n):
        max_h = min(12, n - start)
        for h in range(1, max_h + 1):
            end = start + h
            block = df_raw.iloc[start:end, 1:]  # без _origin_row

            date_score = sum(_row_date_score(block.iloc[i]) for i in range(len(block)))
            header_score = sum(_row_headerish_score(block.iloc[i]) for i in range(len(block)))

            after = df_raw.iloc[end:min(end + 3, n), 1:]
            after_headerish = sum(_row_headerish_score(after.iloc[i]) for i in range(len(after))) if len(after) else 0

            score = 2.2 * date_score + 0.8 * header_score - 0.3 * after_headerish

            # приоритет ранних строк
            score -= 0.55 * start

            if score > best_score:
                best_score = score
                best = (start, h)

    return best
