from __future__ import annotations
import pandas as pd
import numpy as np
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
        "X": "Ручные доп. баллы"
    }.get(category, "Другое")


def compute_scores(
    evidence_rows: List[Dict[str, Any]],
    params: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evid = pd.DataFrame(evidence_rows)
    if evid.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # учитываем ФИО/группу/ID для сшивки
    mapping = unify_students(evid, enable_fuzzy=True)
    evid["student_key_canon"] = evid["student_key"].map(mapping).fillna(evid["student_key"])

    display = evid.groupby("student_key_canon").agg({
        "student_name": _pick_mode,
        "group": _pick_mode,
        "student_id": _pick_mode
    }).reset_index()

    A_norm = evid[evid["category"] == "A"].groupby("student_key_canon")["points"].mean()  # 0..1
    B_rate = evid[evid["category"] == "B"].groupby("student_key_canon")["points"].mean()  # 0..1
    C_sum  = evid[evid["category"] == "C"].groupby("student_key_canon")["points"].sum()   # points
    X_sum  = evid[evid["category"] == "X"].groupby("student_key_canon")["points"].sum()   # points

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
            "ID": sid,
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
        "Место", "ФИО", "Группа", "ID",
        "Успеваемость (баллы)",
        "Посещаемость (баллы)",
        "Активность (баллы)",
        "Ручные доп. баллы",
        "ИТОГО (баллы)"
    ]].copy()

    disp_map = display.set_index("student_key_canon")[["student_name", "group", "student_id"]]
    evid2 = evid.copy()
    evid2["Раздел"] = evid2["category"].map(_section).fillna("Другое")
    evid2["ФИО"] = evid2["student_key_canon"].map(lambda k: disp_map.loc[k, "student_name"] if k in disp_map.index else "")
    evid2["Группа"] = evid2["student_key_canon"].map(lambda k: disp_map.loc[k, "group"] if k in disp_map.index else "")
    evid2["ID"] = evid2["student_key_canon"].map(lambda k: disp_map.loc[k, "student_id"] if k in disp_map.index else "")

    def awarded(row) -> float:
        cat = row["category"]
        p = float(row["points"])
        if cat == "A":
            return float(np.clip(p, 0, 1) * mpA)
        if cat == "B":
            return float(np.clip(p, 0, 1) * mpB)
        if cat in ("C", "X"):
            return float(p)
        return 0.0

    def value_col(row):
        cat = row["category"]
        p = float(row["points"])
        if cat == "A":
            return f"{p:.4f} (норма)"
        if cat == "B":
            return f"{p*100:.2f}%"
        if cat in ("C", "X"):
            return f"{p:.2f} (баллы)"
        return str(row["points"])

    detail_df = pd.DataFrame({
        "ФИО": evid2["ФИО"],
        "Группа": evid2["Группа"],
        "ID": evid2["ID"],
        "Раздел": evid2["Раздел"],
        "Основание": evid2.get("detail", ""),
        "Значение": evid2.apply(value_col, axis=1),
        "Начислено (баллы)": evid2.apply(awarded, axis=1).round(2),
        "Источник": evid2.get("source", ""),
        "Строка (Excel)": evid2.get("origin_row", "")
    })

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
