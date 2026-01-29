from __future__ import annotations
import pandas as pd
from io import BytesIO
from typing import Optional

ID_COL = "Номер студенческого билета"

def export_to_excel_bytes(
    ranking_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    group_df: Optional[pd.DataFrame] = None,
    *,
    quality_df: Optional[pd.DataFrame] = None,
    duplicates_df: Optional[pd.DataFrame] = None,
) -> bytes:
    bio = BytesIO()

    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        ranking_df.to_excel(writer, index=False, sheet_name="Рейтинг")
        summary_df.to_excel(writer, index=False, sheet_name="Свод начислений")

        if group_df is not None and not group_df.empty:
            group_df.to_excel(writer, index=False, sheet_name="Свод по группам")

        if quality_df is not None and not quality_df.empty:
            quality_df.to_excel(writer, index=False, sheet_name="Качество данных")

        if duplicates_df is not None and not duplicates_df.empty:
            duplicates_df.to_excel(writer, index=False, sheet_name="Дубликаты активностей")

        wb = writer.book

        fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1, "valign": "vcenter"})
        fmt_text = wb.add_format({"border": 1, "valign": "top"})
        fmt_num = wb.add_format({"border": 1, "valign": "top", "num_format": "0.00"})
        fmt_title = wb.add_format({"bold": True, "bg_color": "#E8F0FE", "border": 1, "valign": "vcenter"})
        fmt_small = wb.add_format({"border": 1, "valign": "top", "font_color": "#555555"})
        fmt_lvl_err = wb.add_format({"border": 1, "valign": "top", "bg_color": "#FCE8E6"})
        fmt_lvl_warn = wb.add_format({"border": 1, "valign": "top", "bg_color": "#FEF7E0"})
        fmt_lvl_info = wb.add_format({"border": 1, "valign": "top", "bg_color": "#E8F0FE"})

        def format_df_sheet(sheet_name: str, df: pd.DataFrame, default_width: int = 22, max_width: int = 60):
            ws = writer.sheets.get(sheet_name)
            if ws is None:
                return
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(1, len(df)), max(0, len(df.columns) - 1))
            for col, name in enumerate(df.columns):
                ws.write(0, col, name, fmt_header)
                w = max(10, min(max_width, int(len(str(name)) * 1.2) + 10))
                ws.set_column(col, col, max(default_width, w))

        format_df_sheet("Рейтинг", ranking_df, default_width=22, max_width=40)
        format_df_sheet("Свод начислений", summary_df, default_width=24, max_width=48)

        if group_df is not None and not group_df.empty:
            format_df_sheet("Свод по группам", group_df, default_width=20, max_width=40)

        ws3 = wb.add_worksheet("Детализация начислений")
        writer.sheets["Детализация начислений"] = ws3

        cols = ["ФИО", "Группа", ID_COL, "Раздел", "Основание", "Значение", "Начислено (баллы)", "Источник", "Строка (Excel)"]
        for c, n in enumerate(cols):
            ws3.write(0, c, n, fmt_header)

        ws3.freeze_panes(1, 0)
        ws3.set_column(0, 0, 28)
        ws3.set_column(1, 1, 16)
        ws3.set_column(2, 2, 26)
        ws3.set_column(3, 3, 22)
        ws3.set_column(4, 4, 55)
        ws3.set_column(5, 5, 26)
        ws3.set_column(6, 6, 18)
        ws3.set_column(7, 7, 30)
        ws3.set_column(8, 8, 14)

        sum_map = {}
        if not summary_df.empty:
            for _, r in summary_df.iterrows():
                key = (str(r.get("ФИО", "")), str(r.get("Группа", "")), str(r.get(ID_COL, "")))
                try:
                    sum_map[key] = float(r.get("ИТОГО (баллы)", 0.0))
                except Exception:
                    sum_map[key] = 0.0

        r = 1
        if detail_df is not None and not detail_df.empty:
            df = detail_df.copy()
            df["__key"] = list(zip(df["ФИО"].astype(str), df["Группа"].astype(str), df[ID_COL].astype(str)))

            for key, block in df.groupby("__key", sort=True):
                fio, grp, sid = key
                total = float(sum_map.get(key, 0.0))

                title = f"{fio} | {grp} | {ID_COL}: {sid} | Итог: {total:.2f} баллов"
                ws3.merge_range(r, 0, r, len(cols) - 1, title, fmt_title)
                ws3.set_row(r, None, None, {"level": 0, "collapsed": True})
                r += 1

                for _, rr in block.drop(columns=["__key"]).iterrows():
                    ws3.write(r, 0, "", fmt_text)
                    ws3.write(r, 1, "", fmt_text)
                    ws3.write(r, 2, "", fmt_text)
                    ws3.write(r, 3, rr.get("Раздел", ""), fmt_text)
                    ws3.write(r, 4, rr.get("Основание", ""), fmt_small)
                    ws3.write(r, 5, rr.get("Значение", ""), fmt_text)
                    val = rr.get("Начислено (баллы)", "")
                    try:
                        ws3.write_number(r, 6, float(val), fmt_num)
                    except Exception:
                        ws3.write(r, 6, val, fmt_text)
                    ws3.write(r, 7, rr.get("Источник", ""), fmt_small)
                    ws3.write(r, 8, rr.get("Строка (Excel)", ""), fmt_text)

                    ws3.set_row(r, None, None, {"level": 1, "hidden": True})
                    r += 1

        ws3.autofilter(0, 0, max(1, r - 1), len(cols) - 1)

        if quality_df is not None and not quality_df.empty:
            format_df_sheet("Качество данных", quality_df, default_width=22, max_width=60)
            wsq = writer.sheets.get("Качество данных")
            if wsq is not None:
                cols_q = list(quality_df.columns)

                def _col(name: str):
                    try:
                        return cols_q.index(name)
                    except Exception:
                        return None

                for nm, w in [
                    ("Уровень", 10),
                    ("Код", 20),
                    ("Сообщение", 60),
                    ("ФИО", 28),
                    ("Группа", 14),
                    ("Тип", 18),
                    ("Роль", 14),
                    ("Название/описание", 60),
                    ("Доказательство", 35),
                    ("Источник", 30),
                    ("Строка (Excel)", 14),
                    ("Confidence", 12),
                ]:
                    j = _col(nm)
                    if j is not None:
                        wsq.set_column(j, j, w)

                jlvl = _col("Уровень")
                if jlvl is not None:
                    last_row = len(quality_df)
                    wsq.conditional_format(1, jlvl, last_row, jlvl, {
                        "type": "text",
                        "criteria": "containing",
                        "value": "error",
                        "format": fmt_lvl_err
                    })
                    wsq.conditional_format(1, jlvl, last_row, jlvl, {
                        "type": "text",
                        "criteria": "containing",
                        "value": "warn",
                        "format": fmt_lvl_warn
                    })
                    wsq.conditional_format(1, jlvl, last_row, jlvl, {
                        "type": "text",
                        "criteria": "containing",
                        "value": "info",
                        "format": fmt_lvl_info
                    })

        if duplicates_df is not None and not duplicates_df.empty:
            format_df_sheet("Дубликаты активностей", duplicates_df, default_width=22, max_width=60)

    return bio.getvalue()
