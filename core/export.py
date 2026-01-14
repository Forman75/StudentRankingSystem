from __future__ import annotations
import pandas as pd
from io import BytesIO

def export_to_excel_bytes(ranking_df: pd.DataFrame, summary_df: pd.DataFrame, detail_df: pd.DataFrame, group_df: pd.DataFrame | None = None) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        ranking_df.to_excel(writer, index=False, sheet_name="Рейтинг")
        summary_df.to_excel(writer, index=False, sheet_name="Свод начислений")
        if group_df is not None and not group_df.empty:
            group_df.to_excel(writer, index=False, sheet_name="Свод по группам")

        wb = writer.book
        fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1, "valign": "vcenter"})
        fmt_text = wb.add_format({"border": 1, "valign": "top"})
        fmt_num = wb.add_format({"border": 1, "valign": "top", "num_format": "0.00"})
        fmt_title = wb.add_format({"bold": True, "bg_color": "#E8F0FE", "border": 1, "valign": "vcenter"})
        fmt_small = wb.add_format({"border": 1, "valign": "top", "font_color": "#555555"})

        ws1 = writer.sheets["Рейтинг"]
        ws1.freeze_panes(1, 0)
        ws1.autofilter(0, 0, len(ranking_df), len(ranking_df.columns) - 1)
        for col, name in enumerate(ranking_df.columns):
            ws1.write(0, col, name, fmt_header)
            ws1.set_column(col, col, 22)

        ws2 = writer.sheets["Свод начислений"]
        ws2.freeze_panes(1, 0)
        ws2.autofilter(0, 0, len(summary_df), len(summary_df.columns) - 1)
        for col, name in enumerate(summary_df.columns):
            ws2.write(0, col, name, fmt_header)
            ws2.set_column(col, col, 24)

        if group_df is not None and not group_df.empty:
            ws4 = writer.sheets["Свод по группам"]
            ws4.freeze_panes(1, 0)
            ws4.autofilter(0, 0, len(group_df), len(group_df.columns) - 1)
            for col, name in enumerate(group_df.columns):
                ws4.write(0, col, name, fmt_header)
                ws4.set_column(col, col, 20)

        ws3 = wb.add_worksheet("Детализация начислений")
        writer.sheets["Детализация начислений"] = ws3

        cols = ["ФИО", "Группа", "ID", "Раздел", "Основание", "Значение", "Начислено (баллы)", "Источник", "Строка (Excel)"]
        for c, n in enumerate(cols):
            ws3.write(0, c, n, fmt_header)

        ws3.freeze_panes(1, 0)
        ws3.set_column(0, 0, 28)
        ws3.set_column(1, 1, 16)
        ws3.set_column(2, 2, 12)
        ws3.set_column(3, 3, 22)
        ws3.set_column(4, 4, 50)
        ws3.set_column(5, 5, 22)
        ws3.set_column(6, 6, 18)
        ws3.set_column(7, 7, 28)
        ws3.set_column(8, 8, 14)

        sum_map = {}
        if not summary_df.empty:
            for _, r in summary_df.iterrows():
                key = (str(r.get("ФИО","")), str(r.get("Группа","")), str(r.get("ID","")))
                sum_map[key] = float(r.get("ИТОГО (баллы)", 0.0))

        r = 1
        if not detail_df.empty:
            df = detail_df.copy()
            df["__key"] = list(zip(df["ФИО"].astype(str), df["Группа"].astype(str), df["ID"].astype(str)))

            for key, block in df.groupby("__key", sort=True):
                fio, grp, sid = key
                total = sum_map.get(key, 0.0)

                title = f"{fio} | {grp} | Итог: {total:.2f} баллов"
                ws3.merge_range(r, 0, r, len(cols)-1, title, fmt_title)
                ws3.set_row(r, None, None, {"level": 0, "collapsed": True})
                r += 1

                for _, rr in block.drop(columns=["__key"]).iterrows():
                    ws3.write(r, 0, "", fmt_text)
                    ws3.write(r, 1, "", fmt_text)
                    ws3.write(r, 2, "", fmt_text)
                    ws3.write(r, 3, rr.get("Раздел",""), fmt_text)
                    ws3.write(r, 4, rr.get("Основание",""), fmt_small)
                    ws3.write(r, 5, rr.get("Значение",""), fmt_text)
                    val = rr.get("Начислено (баллы)", "")
                    try:
                        ws3.write_number(r, 6, float(val), fmt_num)
                    except Exception:
                        ws3.write(r, 6, val, fmt_text)
                    ws3.write(r, 7, rr.get("Источник",""), fmt_small)
                    ws3.write(r, 8, rr.get("Строка (Excel)",""), fmt_text)

                    ws3.set_row(r, None, None, {"level": 1, "hidden": True})
                    r += 1

        ws3.autofilter(0, 0, max(1, r-1), len(cols)-1)

    return bio.getvalue()
