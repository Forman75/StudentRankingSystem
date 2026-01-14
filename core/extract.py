from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from .utils import norm_name, norm_text, rules_path, load_json

RULES = load_json(rules_path(), {})

# Общие утилиты
# =========================
def _normalize_att_code(x: str) -> str:
    s = norm_text(x)
    # латиница на всякий
    if s == "h":
        s = "н"
    if s == "u":
        s = "у"
    return s


def make_student_key(student_id: str, name: str, group: str) -> str:
    sid = norm_text(student_id)
    if sid and sid not in ("nan", "none", "0"):
        return f"id:{sid}"
    return f"name:{norm_name(name)}|grp:{norm_text(group)}"


def _get_col_as_series(df: pd.DataFrame, col: str) -> pd.Series:
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x


def _pack_person_fields(df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    idx = df.index
    name_col = schema.get("student_name")
    id_col = schema.get("student_id")
    group_col = schema.get("group")

    names = _get_col_as_series(df, name_col).astype(str).reindex(idx) if name_col and name_col in df.columns else pd.Series([""] * len(df), index=idx)
    ids   = _get_col_as_series(df, id_col).astype(str).reindex(idx) if id_col and id_col in df.columns else pd.Series([""] * len(df), index=idx)
    grps  = _get_col_as_series(df, group_col).astype(str).reindex(idx) if group_col and group_col in df.columns else pd.Series([""] * len(df), index=idx)
    return names, ids, grps
# =========================

# 1) Посещаемость (+ / Н / У)
# =========================
def extract_attendance_wide(df: pd.DataFrame, schema: Dict[str, Any], source: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    name_col = schema.get("student_name")
    att_cols = [c for c in schema.get("attendance_cols", []) if c in df.columns]
    if not name_col or name_col not in df.columns or not att_cols:
        return pd.DataFrame(), []

    idx = df.index

    present_set = set(map(norm_text, RULES.get("attendance", {}).get("present", ["+"])))
    absent_set = set(map(norm_text, RULES.get("attendance", {}).get("absent", ["н"])))
    excused_set = set(map(norm_text, RULES.get("attendance", {}).get("excused", ["у"])))

    sub = df.loc[idx, att_cols].astype(str).apply(lambda col: col.map(_normalize_att_code))
    present = sub.apply(lambda col: col.isin(present_set))
    absent = sub.apply(lambda col: col.isin(absent_set))
    excused = sub.apply(lambda col: col.isin(excused_set))

    denom = (present | absent).sum(axis=1).replace(0, np.nan)
    rate = (present.sum(axis=1) / denom).clip(0, 1)

    names, ids, grps = _pack_person_fields(df, schema)
    origin = _get_col_as_series(df, "_origin_row").reindex(idx).to_numpy()

    out = pd.DataFrame({
        "student_name": names.to_numpy(),
        "student_id": ids.to_numpy(),
        "group": grps.to_numpy(),
        "attendance_rate": rate.to_numpy(),
        "attendance_present": present.sum(axis=1).to_numpy(dtype=float),
        "attendance_absent": absent.sum(axis=1).to_numpy(dtype=float),
        "attendance_excused": excused.sum(axis=1).to_numpy(dtype=float),
        "_origin_row": origin
    })

    evidence = []
    for _, r in out.iterrows():
        if pd.isna(r["attendance_rate"]):
            continue
        evidence.append({
            "student_key": make_student_key(r["student_id"], r["student_name"], r["group"]),
            "student_name": str(r["student_name"]),
            "student_id": str(r["student_id"]),
            "group": str(r["group"]),
            "category": "B",
            "points": float(r["attendance_rate"]),
            "source": source,
            "detail": f"Посещаемость: +{int(r['attendance_present'])} / Н{int(r['attendance_absent'])} / У{int(r['attendance_excused'])}",
            "origin_row": int(r["_origin_row"]) if str(r["_origin_row"]) not in ("nan", "None", "") else ""
        })

    return out, evidence
# =========================

# 2) Успеваемость:
# Успеваемость строится только по оценкам, а не по величине баллов
# =========================
_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_GRADE_BR_RE = re.compile(r"[\(\[\{]\s*([2-5])\s*[\)\]\}]")

def _points_to_grade(p: float) -> int:
    # включительно по условиям + разумные пределы
    if p >= 80:
        return 5
    if p >= 60:
        return 4
    if p >= 40:
        return 3
    return 2

def _parse_grade_cell_to_grade(x: Any) -> int | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None

    t = norm_text(s)

    # зачёт / незачёт
    if "незач" in t or "не зач" in t or t in ("нз", "н/з", "fail"):
        return 2
    if "зач" in t or t in ("з", "з/ч", "pass"):
        return 5

    # оценка в скобках имеет приоритет
    m = _GRADE_BR_RE.search(s)
    if m:
        try:
            g = int(m.group(1))
            if 2 <= g <= 5:
                return g
        except Exception:
            pass

    # числа
    s2 = s.replace(",", ".")
    nums = _NUM_RE.findall(s2)
    vals = []
    for q in nums:
        try:
            vals.append(float(q))
        except Exception:
            pass

    if not vals:
        return None

    # если единственное число <= 5.5 -> оценка
    if len(vals) == 1 and vals[0] <= 5.5:
        g = int(round(vals[0]))
        return g if 2 <= g <= 5 else None

    # иначе считаем, что это баллы
    p = float(vals[0])
    # допустим 0..120+, конвертируем в оценку
    if p < 0:
        p = 0.0
    if p > 9999:
        p = 9999.0
    return _points_to_grade(p)


def extract_grades_wide(df: pd.DataFrame, schema: Dict[str, Any], source: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    name_col = schema.get("student_name")
    grade_cols = [c for c in schema.get("grade_cols", []) if c in df.columns]
    if not name_col or name_col not in df.columns or not grade_cols:
        return pd.DataFrame(), []

    # если в grade_cols в основном +/Н/У, то это не оценки.
    sample = df[grade_cols].head(220).astype(str)
    flat = sample.values.ravel().tolist()
    flat = [norm_text(x) for x in flat if x and str(x).lower() not in ("nan", "none")]
    if flat:
        att_marks = sum(1 for x in flat if x in {"+", "н", "у", "-", "h", "n", "u"})
        if att_marks / max(1, len(flat)) >= 0.35:
            return pd.DataFrame(), []

    idx = df.index
    names, ids, grps = _pack_person_fields(df, schema)
    origin = _get_col_as_series(df, "_origin_row").reindex(idx).to_numpy()

    # Парсим в оценки, затем нормируем
    per_col_grade = []
    used_cells = 0
    used_cols = 0

    for c in grade_cols:
        s = _get_col_as_series(df, c).astype(str)
        grades = []
        ok = 0
        for v in s.tolist():
            g = _parse_grade_cell_to_grade(v)
            if g is None:
                grades.append(np.nan)
            else:
                ok += 1
                used_cells += 1
                grades.append(float(g))
        if ok > 0:
            used_cols += 1
        per_col_grade.append(pd.Series(grades, index=df.index))

    # если почти ничего не распознали — не считаем это оценками
    if used_cells < max(10, int(0.02 * max(1, len(df) * max(1, len(grade_cols))))):
        return pd.DataFrame(), []

    mat = pd.concat(per_col_grade, axis=1)

    # средняя оценка по предметам
    avg_grade = mat.mean(axis=1, skipna=True)

    # нормировка 2..5 -> 0..1
    grade_norm = ((avg_grade - 2.0) / 3.0).clip(0, 1)

    out_df = pd.DataFrame({
        "student_name": names.to_numpy(),
        "student_id": ids.to_numpy(),
        "group": grps.to_numpy(),
        "grade_norm": grade_norm.to_numpy(),
        "grade_cols_used": float(used_cols),
        "_origin_row": origin
    })

    evidence = []
    for _, r in out_df.iterrows():
        if pd.isna(r["grade_norm"]):
            continue
        evidence.append({
            "student_key": make_student_key(r["student_id"], r["student_name"], r["group"]),
            "student_name": str(r["student_name"]),
            "student_id": str(r["student_id"]),
            "group": str(r["group"]),
            "category": "A",
            "points": float(r["grade_norm"]),
            "source": source,
            "detail": f"Успеваемость: средняя оценка по {int(r['grade_cols_used'])} предметам (баллы при наличии конвертированы в 2–5)",
            "origin_row": int(r["_origin_row"]) if str(r["_origin_row"]) not in ("nan", "None", "") else ""
        })

    return out_df, evidence
# =========================

# 3) Активности (авто тип + роль участник/победитель и т.д.)
# =========================
KIND_KEYWORDS = {
    "Акселератор": ["акселератор", "startup", "стартап", "аксел"],
    "CTF": ["ctf", "security", "кибер", "pwn", "revers", "crypto"],
    "Публикация": ["публикац", "статья", "scopus", "wos", "ринц", "journal", "doi"],
    "Олимпиада/конкурс": ["олимпиад", "конкурс"],
    "Конференция": ["конференц", "семинар", "форум", "доклад", "тезис"],
    "Спорт": ["спорт", "турнир", "матч", "соревн"],
    "Волонтёрство": ["волонтер", "добровол"]
}

ROLE_REGEX = {
    "Победитель": re.compile(r"(побед|1\s*место|i\s*место|\bwinner\b)", re.I),
    "Призёр": re.compile(r"(приз|2\s*место|3\s*место|ii\s*место|iii\s*место|\bprize\b)", re.I),
    "Организатор": re.compile(r"(организ|оргком|куратор|жюри)", re.I),
    "Участник": re.compile(r"(участ(ник|ие)|participant|финалист|finalist)", re.I)
}
ROLE_PRIORITY = ["Победитель", "Призёр", "Организатор", "Участник"]


def _infer_kind(text: str, context_text: str = "") -> str:
    t = norm_text((context_text or "") + " " + (text or ""))
    best = "Прочее"
    best_hits = 0
    for k, kws in KIND_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in t)
        if hits > best_hits:
            best_hits = hits
            best = k
    return best


def _detect_role(text: str) -> str:
    s = text or ""
    for role in ROLE_PRIORITY:
        if ROLE_REGEX[role].search(s):
            return role
    return "Участник"


def _split_events(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[;\n\r]+|\s\|\s", text)
    parts = [p.strip() for p in parts if p.strip() and p.strip().lower() not in ("nan", "none")]
    return parts[:10]


def extract_activities_text(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    source: str,
    activity_points: Dict[str, Dict[str, float]],
    context_text: str = ""
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    name_col = schema.get("student_name")
    if not name_col or name_col not in df.columns:
        return pd.DataFrame(), []

    act_cols = [c for c in schema.get("activity_cols", []) if c in df.columns]
    if not act_cols:
        return pd.DataFrame(), []

    idx = df.index
    names, ids, grps = _pack_person_fields(df, schema)
    origin = _get_col_as_series(df, "_origin_row").reindex(idx).to_numpy()
    row_text = df.loc[idx, act_cols].astype(str).agg(" | ".join, axis=1)
    forced_kind = (schema.get("activity_kind") or schema.get("activity_kind_auto") or "Прочее").strip()
    forced_kind = forced_kind if forced_kind else "Прочее"
    evidence: List[Dict[str, Any]] = []
    totals: List[float] = []

    for i, ridx in enumerate(df.index):
        txt = str(row_text.loc[ridx])
        tnorm = norm_text(txt)
        if not tnorm or tnorm in ("nan", "none"):
            totals.append(0.0)
            continue

        kind = forced_kind if forced_kind not in ("Авто", "") else _infer_kind(tnorm, context_text=context_text)
        if kind not in activity_points:
            kind = "Прочее"

        events = _split_events(txt)
        pts_sum = 0.0

        for ev in events:
            role = _detect_role(ev)
            p = float(activity_points.get(kind, {}).get(role, activity_points.get("Прочее", {}).get(role, 0.0)))
            pts_sum += p

            evidence.append({
                "student_key": make_student_key(ids.iloc[i], names.iloc[i], grps.iloc[i]),
                "student_name": str(names.iloc[i]),
                "student_id": str(ids.iloc[i]),
                "group": str(grps.iloc[i]),
                "category": "C",
                "points": float(p),
                "source": source,
                "detail": f"{kind} — {role}: {ev}",
                "origin_row": int(origin[i]) if str(origin[i]) not in ("nan", "None", "") else ""
            })

        totals.append(float(pts_sum))

    out = pd.DataFrame({
        "student_name": names.to_numpy(),
        "student_id": ids.to_numpy(),
        "group": grps.to_numpy(),
        "activity_points_total": np.array(totals, dtype=float),
        "_origin_row": origin
    })

    return out, evidence