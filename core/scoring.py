from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from .entity import unify_students

def _pick_mode(series: pd.Series) -> str:
    s = series.astype(str).fillna("").map(lambda x: x.strip())
    s = s[s != ""]
    if s.empty:
        return ""
    try:
        return s.mode().iloc[0]
    except Exception:
        return s.iloc[0]

def _section(category: str) -> str:
    return {
        "A": "Успеваемость",
        "B": "Посещаемость",
        "C": "Активность",
        "X": "Ручные доп. баллы",
    }.get(category, "Другое")

ID_COL = "Номер студенческого билета"

def compute_scores(
    evidence_rows: List[Dict[str, Any]],
    params: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evid = pd.DataFrame(evidence_rows)
    if evid.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    mapping = unify_students(evid, enable_fuzzy=True)
    evid["student_key_canon"] = evid["student_key"].map(mapping).fillna(evid["student_key"])

    display = evid.groupby("student_key_canon").agg({
        "student_name": _pick_mode,
        "group": _pick_mode,
        "student_id": _pick_mode
    }).reset_index()

    A_norm = evid[evid["category"] == "A"].groupby("student_key_canon")["points"].mean()
    B_rate = evid[evid["category"] == "B"].groupby("student_key_canon")["points"].mean()
    C_sum  = evid[evid["category"] == "C"].groupby("student_key_canon")["points"].sum()
    X_sum  = evid[evid["category"] == "X"].groupby("student_key_canon")["points"].sum()

    A_cnt = evid[evid["category"] == "A"].groupby("student_key_canon").size()
    B_cnt = evid[evid["category"] == "B"].groupby("student_key_canon").size()

    students = sorted(evid["student_key_canon"].unique().tolist())

    mpA = float(params.get("acad_max", 50))
    mpB = float(params.get("att_max", 20))
    thrB = float(params.get("att_threshold", 0.60))

    summary_rows = []
    for sk in students:
        a = float(A_norm.get(sk, np.nan))
        b = float(B_rate.get(sk, np.nan))
        c = float(C_sum.get(sk, 0.0))
        x = float(X_sum.get(sk, 0.0))

        A_pts = 0.0 if np.isnan(a) else float(np.clip(a, 0, 1) * mpA)

        if np.isnan(b) or b < thrB:
            B_pts = 0.0
            b_note = f"Посещаемость ниже {int(thrB*100)}% → баллы не начислены"
        else:
            B_pts = float(np.clip(b, 0, 1) * mpB)
            b_note = ""

        C_pts = float(c)
        total = float(A_pts + B_pts + C_pts + x)

        drow = display[display["student_key_canon"] == sk]
        fio = drow["student_name"].iloc[0] if not drow.empty else ""
        grp = drow["group"].iloc[0] if not drow.empty else ""
        sid = drow["student_id"].iloc[0] if not drow.empty else ""

        summary_rows.append({
            "ФИО": fio,
            "Группа": grp,
            ID_COL: sid,
            "Успеваемость (норма 0..1)": "" if np.isnan(a) else round(a, 4),
            "Успеваемость (баллы)": round(A_pts, 2),
            "Посещаемость (%)": "" if np.isnan(b) else round(b * 100, 2),
            "Посещаемость (баллы)": round(B_pts, 2),
            "Активность (баллы)": round(C_pts, 2),
            "Ручные доп. баллы": round(x, 2),
            "ИТОГО (баллы)": round(total, 2),
            "Примечание": b_note
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("ИТОГО (баллы)", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "Место", summary_df.index + 1)

    ranking_df = summary_df[[
        "Место", "ФИО", "Группа", ID_COL,
        "Успеваемость (баллы)",
        "Посещаемость (баллы)",
        "Активность (баллы)",
        "Ручные доп. баллы",
        "ИТОГО (баллы)"
    ]].copy()

    disp_map = display.set_index("student_key_canon")[["student_name", "group", "student_id"]]

    def _fio(sk: str) -> str:
        return disp_map.loc[sk, "student_name"] if sk in disp_map.index else ""

    def _grp(sk: str) -> str:
        return disp_map.loc[sk, "group"] if sk in disp_map.index else ""

    def _sid(sk: str) -> str:
        return disp_map.loc[sk, "student_id"] if sk in disp_map.index else ""

    detail_rows: List[Dict[str, Any]] = []

    for _, row in evid.iterrows():
        sk = row["student_key_canon"]
        cat = str(row.get("category", ""))
        p = float(row.get("points", 0.0) or 0.0)

        if cat in ("C", "X"):
            awarded = float(p)
        else:
            awarded = 0.0

        if cat == "A":
            val = f"{p:.4f} (норма)"
            basis_prefix = "[сырые данные, учтено в среднем] "
        elif cat == "B":
            val = f"{p*100:.2f}%"
            basis_prefix = "[сырые данные, учтено в среднем] "
        elif cat in ("C", "X"):
            val = f"{p:.2f} (баллы)"
            basis_prefix = ""
        else:
            val = str(row.get("points", ""))
            basis_prefix = ""

        detail_rows.append({
            "ФИО": _fio(sk),
            "Группа": _grp(sk),
            ID_COL: _sid(sk),
            "Раздел": _section(cat),
            "Основание": basis_prefix + str(row.get("detail", "")),
            "Значение": val,
            "Начислено (баллы)": round(awarded, 2),
            "Источник": str(row.get("source", "")),
            "Строка (Excel)": str(row.get("origin_row", "")),
        })

    for sk in students:
        a = float(A_norm.get(sk, np.nan))
        nA = int(A_cnt.get(sk, 0))
        if not np.isnan(a) and nA > 0:
            A_pts = float(np.clip(a, 0, 1) * mpA)
            detail_rows.append({
                "ФИО": _fio(sk),
                "Группа": _grp(sk),
                ID_COL: _sid(sk),
                "Раздел": "Успеваемость",
                "Основание": f"Итог по успеваемости: среднее по {nA} источникам",
                "Значение": f"{a:.4f} (норма)",
                "Начислено (баллы)": round(A_pts, 2),
                "Источник": "Агрегировано системой",
                "Строка (Excel)": "",
            })

        b = float(B_rate.get(sk, np.nan))
        nB = int(B_cnt.get(sk, 0))
        if not np.isnan(b) and nB > 0:
            if b < thrB:
                B_pts = 0.0
                note = f"ниже порога {int(thrB*100)}%"
            else:
                B_pts = float(np.clip(b, 0, 1) * mpB)
                note = ""
            detail_rows.append({
                "ФИО": _fio(sk),
                "Группа": _grp(sk),
                ID_COL: _sid(sk),
                "Раздел": "Посещаемость",
                "Основание": f"Итог по посещаемости: среднее по {nB} источникам" + (f" ({note})" if note else ""),
                "Значение": f"{b*100:.2f}%",
                "Начислено (баллы)": round(B_pts, 2),
                "Источник": "Агрегировано системой",
                "Строка (Excel)": "",
            })

    detail_df = pd.DataFrame(detail_rows)

    sec_order = {"Успеваемость": 1, "Посещаемость": 2, "Активность": 3, "Ручные доп. баллы": 4, "Другое": 99}
    detail_df["__sec"] = detail_df["Раздел"].map(sec_order).fillna(99)
    detail_df = detail_df.sort_values(
        ["ФИО", "Группа", "__sec", "Начислено (баллы)"],
        ascending=[True, True, True, False]
    ).drop(columns=["__sec"]).reset_index(drop=True)

    grp_df = summary_df.copy()
    grp_df["ИТОГО (баллы)"] = pd.to_numeric(grp_df["ИТОГО (баллы)"], errors="coerce").fillna(0.0)
    group_df = grp_df.groupby("Группа", dropna=False).agg(
        Количество=("ФИО", "count"),
        Средний=("ИТОГО (баллы)", "mean"),
        Медиана=("ИТОГО (баллы)", "median"),
        Максимум=("ИТОГО (баллы)", "max")
    ).reset_index()
    group_df["Средний"] = group_df["Средний"].round(2)
    group_df["Медиана"] = group_df["Медиана"].round(2)
    group_df["Максимум"] = group_df["Максимум"].round(2)
    group_df = group_df.sort_values(["Средний"], ascending=False).reset_index(drop=True)
    return ranking_df, summary_df, detail_df, group_df
