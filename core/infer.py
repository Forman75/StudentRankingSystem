from __future__ import annotations
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from .utils import (norm_text, is_mostly_unique, try_parse_date, column_signature, profiles_path, load_json, save_json, rules_path)
from .header_detect import detect_header_block

try:
    from rapidfuzz import fuzz  # type: ignore
    def _partial_ratio(a: str, b: str) -> int:
        return int(fuzz.partial_ratio(a, b))
except Exception:
    from difflib import SequenceMatcher
    def _partial_ratio(a: str, b: str) -> int:
        a, b = a.lower(), b.lower()
        if not a or not b:
            return 0
        if a in b or b in a:
            return 100
        return int(100 * SequenceMatcher(None, a, b).ratio())


RULES = load_json(rules_path(), {})

SYN = {
    "student_name": ["фио", "студент", "обучающийся", "слушатель", "фамилия", "имя", "отчество", "student", "name"],
    "student_id":   ["id", "зачетка", "зачётка", "номер зачетки", "табельный", "student id", "record book"],
    "group":        ["группа", "учебная группа", "group", "групп"],
    "attendance":   ["посещаемость", "посещения", "пропуски", "явка", "неявка", "attendance"],
    "grade":        ["оценка", "оценки", "балл", "баллы", "итог", "экзамен", "зачет", "зачёт", "grade", "score", "mark", "брс", "рейтинг"],
    "activity":     ["мероприят", "конкурс", "олимпиад", "хакатон", "акселератор", "волонтер", "спорт", "конференц", "соревн", "ctf", "публикац", "статья", "доклад", "тезис"],
}

ACTIVITY_KIND_KEYWORDS = {
    "Акселератор": ["акселератор", "startup", "стартап", "аксел"],
    "CTF": ["ctf", "security", "кибер", "pwn", "revers", "crypto"],
    "Публикация": ["публикац", "статья", "scopus", "wos", "ринц", "doi", "journal"],
    "Олимпиада/конкурс": ["олимпиад", "конкурс"],
    "Конференция": ["конференц", "семинар", "форум", "доклад", "тезис"],
    "Спорт": ["спорт", "турнир", "матч", "соревн"],
    "Волонтёрство": ["волонтер", "добровол"],
}

PUB_COL_HINTS = ["название", "тема", "доклад", "статья", "публикац", "журнал", "конференц", "scopus", "wos", "ринц", "doi"]

_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_GRADE_BR_RE = re.compile(r"[\(\[\{]\s*([2-5])\s*[\)\]\}]")

def _make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        base = str(c).strip()
        if base == "" or base.lower() == "nan":
            base = "col"
        n = seen.get(base, 0) + 1
        seen[base] = n
        out.append(base if n == 1 else f"{base}__{n}")
    return out

def _best_match(col: str, targets: List[str]) -> int:
    n = norm_text(col)
    return max((_partial_ratio(n, t) for t in targets), default=0)

def build_dataframe_with_headers(
    df_raw: pd.DataFrame,
    override_start: Optional[int] = None,
    override_h: Optional[int] = None,
    max_scan_rows: int = 80
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if override_start is None or override_h is None:
        start, h = detect_header_block(df_raw, max_scan_rows=max_scan_rows)
    else:
        start, h = int(override_start), int(override_h)

    h = max(1, h)
    end = min(len(df_raw), start + h)

    header_block = df_raw.iloc[start:end, 1:].copy()
    data_block = df_raw.iloc[end:, :].copy()

    headers = []
    for c in range(header_block.shape[1]):
        parts = []
        for r in range(header_block.shape[0]):
            v = header_block.iat[r, c]
            s = str(v).strip() if v is not None else ""
            if s and s.lower() != "nan":
                parts.append(s)
        headers.append(" | ".join(parts) if parts else f"col_{c+1}")

    headers = _make_unique(headers)
    df = data_block.copy()
    df.columns = ["_origin_row"] + headers

    meta = {"header_start_row": int(start) + 1, "header_rows": int(h), "columns": headers}
    return df, meta

def load_profiles() -> Dict[str, Any]:
    return load_json(profiles_path(), {})

def save_profiles(profiles: Dict[str, Any]) -> None:
    save_json(profiles_path(), profiles)

def _infer_kind_from_context(context_text: str) -> str:
    t = norm_text(context_text)
    best = "Прочее"
    best_hits = 0
    for kind, kws in ACTIVITY_KIND_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in t)
        if hits > best_hits:
            best_hits = hits
            best = kind
    return best

def _infer_activity_kind(df: pd.DataFrame, activity_cols: List[str], context_text: str = "") -> str:
    k0 = _infer_kind_from_context(context_text)
    if k0 != "Прочее":
        return k0
    headers = " ".join([norm_text(c) for c in activity_cols])[:8000]
    k1 = _infer_kind_from_context(headers)
    if k1 != "Прочее":
        return k1
    if not activity_cols:
        return "Прочее"
    sample = df[activity_cols].head(250).astype(str).agg(" | ".join, axis=1)
    return _infer_kind_from_context(" ".join(sample.tolist())[:20000])

def _parse_cell_points_grade(x: Any) -> tuple[float | None, int | None]:
    if x is None:
        return None, None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None, None
    s2 = s.replace(",", ".")
    g = None
    m = _GRADE_BR_RE.search(s2)
    if m:
        try:
            g = int(m.group(1))
        except Exception:
            g = None
    nums = _NUM_RE.findall(s2)
    vals = []
    for t in nums:
        try:
            vals.append(float(t))
        except Exception:
            pass
    if not vals:
        return None, g
    if len(vals) == 1 and vals[0] <= 5.5:
        return None, int(round(vals[0]))
    return vals[0], g

def _grade_like_ratio(series: pd.Series) -> float:
    # Доля ячеек, которые похожи на оценку/баллы/зачёт.
    s = series.head(220).astype(str).tolist()
    ok = 0
    tot = 0
    for v in s:
        t = norm_text(v)
        if t in ("", "nan", "none"):
            continue
        tot += 1
        # зачёт / незачёт
        if "незач" in t or "не зач" in t or t in ("нз", "н/з", "fail"):
            ok += 1
            continue
        if "зач" in t or t in ("з", "з/ч", "pass"):
            ok += 1
            continue
        p, g = _parse_cell_points_grade(v)
        if p is not None or g is not None:
            ok += 1
    return ok / max(1, tot)

def _suggest_activity_cols(df: pd.DataFrame, cols: List[str], exclude: set[str], prefer_keywords: List[str]) -> List[str]:
    candidates = []
    for c in cols:
        if c in exclude:
            continue
        s = df[c]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        samp = s.head(120).astype(str)
        avglen = float(samp.str.len().replace(0, np.nan).mean() or 0.0)
        header = norm_text(c)
        kw_hit = sum(1 for kw in prefer_keywords if kw in header)
        candidates.append((c, kw_hit, avglen))
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [c[0] for c in candidates[:3] if c[2] > 2 or c[1] > 0]

def infer_schema(df: pd.DataFrame, context_text: str = "") -> Dict[str, Any]:
    cols = [c for c in df.columns if c != "_origin_row"]
    sig = column_signature(cols)

    profiles = load_profiles()
    if sig in profiles:
        prof = profiles[sig]
        prof.setdefault("activity_kind", prof.get("activity_kind_auto", "Прочее"))
        prof.setdefault("activity_kind_auto", prof.get("activity_kind", "Прочее"))
        return prof | {"_sig": sig, "_from_profile": True}

    scored = []
    for c in cols:
        scores = {k: _best_match(c, v) for k, v in SYN.items()}
        scored.append((c, scores))

    sample = df.head(350)

    col_uniqueness = {}
    for c in cols:
        x = sample[c]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        vals = x.astype(str).str.strip().tolist()
        col_uniqueness[c] = is_mostly_unique(vals, 0.6)

    def pick_one(key, thr=55):
        candidates = [(c, sc[key], col_uniqueness.get(c, False)) for c, sc in scored]
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        if candidates and candidates[0][1] >= thr:
            return candidates[0][0]
        return None

    student_name = pick_one("student_name", 55)
    student_id = pick_one("student_id", 55)
    group = pick_one("group", 55)

    # date-like
    date_cols = []
    for c in cols:
        parts = [p.strip() for p in str(c).split("|")]
        for p in parts:
            if try_parse_date(p):
                date_cols.append(c)
                break

    def _att_like_values(c: str) -> bool:
        if c in (student_name, student_id, group):
            return False
        s = sample[c]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        vals = s.head(220).astype(str).map(norm_text).tolist()
        vals = [v for v in vals if v not in ("", "nan", "none")]
        if not vals:
            return False
        hits = sum(1 for v in vals if v in {"+", "н", "у", "-", "h", "n", "u"})
        return hits / max(1, len(vals)) >= 0.35

    numeric_ratio: Dict[str, float] = {}
    att_value_like: Dict[str, bool] = {}
    grade_like: Dict[str, float] = {}

    for c in cols:
        if c in (student_name, student_id, group) or c in date_cols:
            continue

        s = sample[c]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        ss = s.head(250).astype(str).str.replace(",", ".", regex=False)

        # numeric ratio
        numhits = 0
        total = 0
        for v in ss.tolist():
            p, g = _parse_cell_points_grade(v)
            if p is not None or g is not None:
                numhits += 1
            total += 1
        numeric_ratio[c] = numhits / max(1, total)

        # attendance-like by values
        att_value_like[c] = _att_like_values(c)

        # grade-like ratio (включая зачёт/незачёт)
        try:
            grade_like[c] = _grade_like_ratio(s)
        except Exception:
            grade_like[c] = 0.0

    attendance_cols: List[str] = []
    grade_cols: List[str] = []
    activity_cols: List[str] = []

    grade_keyword_cols = []
    activity_keyword_cols = []
    attendance_keyword_cols = []

    for c, sc in scored:
        if c in (student_name, student_id, group):
            continue
        if sc["grade"] >= 60:
            grade_keyword_cols.append(c)
        if sc["activity"] >= 60:
            activity_keyword_cols.append(c)
        if sc["attendance"] >= 60:
            attendance_keyword_cols.append(c)

    # attendance cols
    for c, sc in scored:
        if c in (student_name, student_id, group):
            continue
        if c in date_cols:
            attendance_cols.append(c)
            continue
        if sc["attendance"] >= 70:
            attendance_cols.append(c)
            continue
        if att_value_like.get(c, False):
            attendance_cols.append(c)

    # grade cols: берём по похожести на оценки, включая зачёт
    for c, sc in scored:
        if c in (student_name, student_id, group) or c in attendance_cols or c in date_cols:
            continue
        if att_value_like.get(c, False):
            continue

        gl = grade_like.get(c, 0.0)
        nr = numeric_ratio.get(c, 0.0)

        if sc["grade"] >= 60 and gl >= 0.20:
            grade_cols.append(c)
            continue
        if gl >= 0.55:
            grade_cols.append(c)
            continue
        if nr >= 0.70 and sc["activity"] < 70:
            grade_cols.append(c)
            continue

    # activity cols
    for c, sc in scored:
        if c in (student_name, student_id, group) or c in attendance_cols or c in date_cols:
            continue
        if sc["activity"] >= 65 and numeric_ratio.get(c, 0.0) < 0.85:
            activity_cols.append(c)

    # type decision
    ctx = norm_text(context_text)
    ctx_act_hits = sum(1 for kws in ACTIVITY_KIND_KEYWORDS.values() for kw in kws if kw in ctx)
    ctx_grade_hits = sum(1 for kw in SYN["grade"] if kw in ctx)
    att_value_cols = [c for c in cols if att_value_like.get(c, False)]
    att_strong = (len(att_value_cols) >= 3) or (len(attendance_keyword_cols) >= 1 and len(attendance_cols) >= 3)

    # grades strong: хотя бы 2 колонки, которые действительно похожи на оценки
    grade_real_cols = [c for c in grade_cols if grade_like.get(c, 0.0) >= 0.20 or numeric_ratio.get(c, 0.0) >= 0.50]
    grades_strong = len(grade_real_cols) >= 2 or (ctx_grade_hits >= 1 and len(grade_cols) >= 1)

    activity_strong = (ctx_act_hits >= 1 and len(activity_cols) >= 1) or (len(activity_cols) >= 2)

    # посещаемость приоритетнее, если действительно видно +/Н/У в нескольких колонках
    if att_strong:
        type_hint = "attendance"
    elif activity_strong and (ctx_act_hits >= ctx_grade_hits):
        type_hint = "activity"
    elif grades_strong:
        type_hint = "grades"
    elif len(activity_cols) >= 1:
        type_hint = "activity"
    elif len(grade_cols) >= 1:
        type_hint = "grades"
    else:
        type_hint = "unknown"

    activity_kind_auto = _infer_activity_kind(df, activity_cols, context_text=context_text) if type_hint == "activity" else "Прочее"
    exclude = {student_name, student_id, group, "_origin_row"}
    exclude = {x for x in exclude if x}
    if type_hint == "activity":
        if not activity_cols or (activity_kind_auto in ("Публикация", "Конференция") and len(activity_cols) < 1):
            prefer = PUB_COL_HINTS if activity_kind_auto in ("Публикация", "Конференция") else ["мероприят", "конкурс", "акселератор", "ctf", "конференц", "доклад", "статья"]
            activity_cols = _suggest_activity_cols(df, cols, exclude, prefer) or activity_cols

    schema = {
        "_sig": sig,
        "_from_profile": False,
        "type_hint": type_hint,

        "student_name": student_name,
        "student_id": student_id,
        "group": group,

        "attendance_cols": attendance_cols,
        "grade_cols": grade_cols,
        "activity_cols": activity_cols,

        "activity_kind_auto": activity_kind_auto,
        "activity_kind": activity_kind_auto
    }
    return schema

def persist_profile_for_schema(schema: Dict[str, Any]) -> None:
    sig = schema.get("_sig")
    if not sig:
        return
    profiles = load_profiles()
    to_save = {k: v for k, v in schema.items() if not k.startswith("_")}
    profiles[sig] = to_save
    save_profiles(profiles)
