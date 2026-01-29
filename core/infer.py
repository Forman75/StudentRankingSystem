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
    "student_name": ["фио", "студент", "обучающийся", "слушатель", "фамилия", "отчество", "full name", "student name"],
    # без общего "id", чтобы не ловить "ID пользователя"
    "student_id": [
        "номер студенческого билета",
        "студенческий билет",
        "номер студенческого",
        "зачетка",
        "зачётка",
        "номер зачетки",
        "номер зачётки",
        "record book",
        "student card",
        "student id",
        "студенческ",
        "билет",
        "id/зачетка",
        "id/зачётка",
    ],
    "group": ["группа", "учебная группа", "group", "групп"],
    "attendance": ["посещаемость", "посещения", "пропуски", "явка", "неявка", "attendance"],
    "grade": ["оценка", "оценки", "балл", "баллы", "итог", "экзамен", "зачет", "зачёт", "grade", "score", "mark", "брс", "рейтинг"],
    # "научн", чтобы колонка типа "Укажите научную..." попадала в activity
    "activity": ["мероприят", "конкурс", "олимпиад", "хакатон", "акселератор", "волонтер", "спорт", "конференц", "соревн", "ctf",
                 "публикац", "статья", "доклад", "тезис", "научн", "достижен", "активност"],
}

NAME_PREFER_KWS = ["фио", "фамилия", "отчество"]
NAME_AVOID_KWS = [
    "пользовател",
    "username",
    "user name",
    "display name",
    "отображаем",
    "email",
    "e-mail",
    "почт",
    "mail",
]

STUDENT_ID_PREFER_KWS = [
    "студенчес",
    "студенческий билет",
    "номер студенческого",
    "билет",
    "зачетк",
    "зачёт",
    "record book",
    "student card",
]
STUDENT_ID_AVOID_KWS = [
    "пользовател",
    "id пользователя",
    "user id",
    "userid",
    "username",
    "account",
    "аккаунт",
    "google",
    "email",
    "почт",
]

ACTIVITY_KIND_KEYWORDS = {
    "Акселератор": ["акселератор", "startup", "стартап", "аксел"],
    "CTF": ["ctf", "security", "кибер", "pwn", "revers", "crypto"],
    "Публикация": ["публикац", "статья", "scopus", "wos", "ринц", "doi", "journal"],
    "Олимпиада/конкурс": ["олимпиад", "конкурс", "хакатон", "hackathon"],
    "Конференция": ["конференц", "семинар", "форум", "доклад", "тезис"],
    "Спорт": ["спорт", "турнир", "матч", "соревн"],
    "Волонтёрство": ["волонтер", "добровол"],
}

PUB_COL_HINTS = ["название", "тема", "доклад", "статья", "публикац", "журнал", "конференц", "scopus", "wos", "ринц", "doi"]

_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_GRADE_BR_RE = re.compile(r"[\(\[\{]\s*([2-5])\s*[\)\]\}]")
# =========================

# Helpers
# =========================
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

def _clean_header_cell(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).replace("\ufeff", "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s

def _best_match(col: str, targets: List[str]) -> int:
    n = norm_text(col)
    return max((_partial_ratio(n, t) for t in targets), default=0)

def build_dataframe_with_headers(
    df_raw: pd.DataFrame,
    override_start: Optional[int] = None,
    override_h: Optional[int] = None,
    max_scan_rows: int = 80,
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
            s = _clean_header_cell(v)
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
    s = series.head(220).astype(str).tolist()
    ok = 0
    tot = 0
    for v in s:
        t = norm_text(v)
        if t in ("", "nan", "none"):
            continue
        tot += 1
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

def _pick_student_id_column(cols: List[str], email_like_ratio: Dict[str, float]) -> Optional[str]:
    candidates = []
    prefer_candidates = []

    for c in cols:
        h = norm_text(c)
        base = _best_match(c, SYN["student_id"])
        score = base

        if any(k in h for k in STUDENT_ID_PREFER_KWS):
            score += 120
        if any(k in h for k in STUDENT_ID_AVOID_KWS):
            score -= 260
        if h in ("id", "ид") and not any(k in h for k in STUDENT_ID_PREFER_KWS):
            score -= 140

        if email_like_ratio.get(c, 0.0) >= 0.10:
            score -= 320

        candidates.append((c, score))
        if any(k in h for k in STUDENT_ID_PREFER_KWS) and not any(k in h for k in STUDENT_ID_AVOID_KWS):
            prefer_candidates.append((c, score))

    if prefer_candidates:
        prefer_candidates.sort(key=lambda x: x[1], reverse=True)
        return prefer_candidates[0][0]

    candidates.sort(key=lambda x: x[1], reverse=True)
    if candidates and candidates[0][1] >= 55:
        return candidates[0][0]
    return None

def _auto_activity_trusted(cols: List[str], type_hint: str) -> bool:
    if type_hint != "activity":
        return False
    all_headers = " ".join([norm_text(c) for c in cols])

    form_markers = any(k in all_headers for k in [
        "отметка времени", "метка времени", "timestamp",
        "адрес электронной почты", "email", "e-mail",
        "id пользователя", "отображаемое имя пользователя",
    ])

    org_markers = any(k in all_headers for k in [
        "№", "п/п", "секция",
        "название доклада", "тема доклада", "доклад",
        "список участников", "участник", "участники"
    ])

    return bool(org_markers and not form_markers)

def _headers_look_like_google_forms(cols: List[str]) -> bool:
    all_headers = " ".join([norm_text(c) for c in cols])
    return any(k in all_headers for k in [
        "отметка времени", "метка времени", "timestamp",
        "адрес электронной почты", "email", "e-mail",
        "id пользователя", "отображаемое имя пользователя",
    ])

def _is_index_like(col_name: str, series: pd.Series) -> bool:
    # Индексные колонки типа "№ п/п", "No", "Номер" - не должны считаться успеваемостью
    h = norm_text(col_name)
    if any(k in h for k in ["№", "п/п", "no", "номер", "нумерац", "index", "идентификатор записи"]):
        # проверим что значения похожи на последовательность 1..N
        vals = series.head(80).astype(str).map(norm_text).tolist()
        vals = [v for v in vals if v not in ("", "nan", "none")]
        nums = []
        for v in vals[:40]:
            if v.isdigit():
                nums.append(int(v))
            else:
                return False
        if len(nums) >= 10:
            diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
            ok = sum(1 for d in diffs if d in (0, 1, 2))
            return ok / max(1, len(diffs)) >= 0.8
    return False

def _attendance_marker_ratio(series: pd.Series) -> float:
    """
    Насколько значения похожи на посещаемость
    Поддерживается + / Н / У и "да/нет", "present/absent", "1/0"
    """
    tokens = {"+", "н", "у", "-", "h", "n", "u", "да", "нет", "yes", "no", "y", "1", "0", "present", "absent"}
    s = series.head(220).astype(str).map(norm_text).tolist()
    s = [v for v in s if v not in ("", "nan", "none")]
    if not s:
        return 0.0
    hits = 0
    for v in s:
        if v in tokens:
            hits += 1
        elif v.startswith("присутств"):
            hits += 1
        elif v.startswith("отсут"):
            hits += 1
    return hits / max(1, len(s))


def _activity_value_ratio(series: pd.Series) -> float:
    # Доля строк, где встречаются ключевые слова активностей (в значениях)
    kws = [
        "ctf", "хакат", "hackathon", "публикац", "статья", "doi", "scopus", "wos", "ринц",
        "конференц", "доклад", "тезис", "форум", "семинар",
        "олимпиад", "конкурс",
        "волонтер", "волонтёр",
        "спорт", "турнир",
        "акселератор", "стартап",
    ]
    s = series.head(220).astype(str).map(norm_text).tolist()
    s = [v for v in s if v not in ("", "nan", "none")]
    if not s:
        return 0.0
    hits = 0
    for v in s:
        if any(k in v for k in kws):
            hits += 1
    return hits / max(1, len(s))

def infer_schema(df: pd.DataFrame, context_text: str = "") -> Dict[str, Any]:
    cols = [c for c in df.columns if c != "_origin_row"]
    sig = column_signature(cols)
    profiles = load_profiles()
    if sig in profiles:
        prof = profiles[sig]
        # миграция: если student_id указывает на "ID пользователя", а есть колонка студ.билета
        prof_sid = prof.get("student_id")
        if prof_sid and prof_sid in cols:
            h = norm_text(prof_sid)
            if any(k in h for k in STUDENT_ID_AVOID_KWS):
                sample = df.head(250)
                email_like_ratio = {}
                for c in cols:
                    s = sample[c].astype(str).tolist()
                    s = [x for x in s if x and str(x).lower() not in ("nan", "none")]
                    hits = sum(1 for v in s if ("@" in v and "." in v))
                    email_like_ratio[c] = hits / max(1, len(s))
                better = _pick_student_id_column(cols, email_like_ratio)
                if better and better != prof_sid:
                    prof["student_id"] = better

        if "activity_kind_auto" not in prof:
            prof["activity_kind_auto"] = prof.get("activity_kind", "Прочее")

        if "activity_kind" not in prof:
            if prof.get("type_hint") == "activity":
                prof["activity_kind"] = "Авто"
            else:
                prof["activity_kind"] = prof.get("activity_kind_auto", "Авто")

        if "activity_trusted" not in prof:
            prof["activity_trusted"] = _auto_activity_trusted(cols, prof.get("type_hint", "unknown"))
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

    email_like_ratio = {}
    for c in cols:
        x = sample[c]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        vals = [str(v).strip() for v in x.head(220).tolist()]
        vals = [v for v in vals if v and v.lower() not in ("nan", "none")]
        hits = sum(1 for v in vals if ("@" in v and "." in v))
        email_like_ratio[c] = hits / max(1, len(vals))

    def pick_one(key, thr=55):
        candidates = []
        for c, sc in scored:
            score = int(sc.get(key, 0))

            if key == "student_name":
                h = norm_text(c)
                if any(k in h for k in NAME_PREFER_KWS):
                    score += 18
                if any(k in h for k in NAME_AVOID_KWS):
                    score -= 28
                if email_like_ratio.get(c, 0.0) >= 0.30:
                    score -= 35

            candidates.append((c, score, col_uniqueness.get(c, False)))
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        if candidates and candidates[0][1] >= thr:
            return candidates[0][0]
        return None

    student_name = pick_one("student_name", 55)
    student_id = _pick_student_id_column(cols, email_like_ratio)
    group = pick_one("group", 55)

    # date-like columns
    date_cols = []
    for c in cols:
        parts = [p.strip() for p in str(c).split("|")]
        for p in parts:
            if try_parse_date(p):
                date_cols.append(c)
                break

    # индексные колонки типа № п/п
    index_like: Dict[str, bool] = {}
    for c in cols:
        s = sample[c]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        index_like[c] = _is_index_like(c, s)

    # attendance marker ratio + activity value ratio
    att_marker: Dict[str, float] = {}
    act_val_ratio: Dict[str, float] = {}
    avg_len: Dict[str, float] = {}

    for c in cols:
        if c in (student_name, student_id, group):
            continue
        s = sample[c]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        att_marker[c] = _attendance_marker_ratio(s)
        act_val_ratio[c] = _activity_value_ratio(s)
        avg_len[c] = float(s.head(120).astype(str).str.len().replace(0, np.nan).mean() or 0.0)

    # считаем att_value_like и для date_cols тоже
    att_value_like: Dict[str, bool] = {}
    for c in cols:
        if c in (student_name, student_id, group):
            continue
        if index_like.get(c, False):
            att_value_like[c] = False
            continue
        att_value_like[c] = (att_marker.get(c, 0.0) >= 0.35)

    # numeric/grade features, но пропускаем date_cols
    numeric_ratio: Dict[str, float] = {}
    grade_like: Dict[str, float] = {}

    for c in cols:
        if c in (student_name, student_id, group) or c in date_cols:
            continue
        if index_like.get(c, False):
            continue

        s = sample[c]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        ss = s.head(250).astype(str).str.replace(",", ".", regex=False)

        numhits = 0
        total = 0
        for v in ss.tolist():
            p, g = _parse_cell_points_grade(v)
            if p is not None or g is not None:
                numhits += 1
            total += 1
        numeric_ratio[c] = numhits / max(1, total)

        try:
            grade_like[c] = _grade_like_ratio(s)
        except Exception:
            grade_like[c] = 0.0

    # сбор колонки по типам
    attendance_cols: List[str] = []
    grade_cols: List[str] = []
    activity_cols: List[str] = []
    attendance_keyword_cols = []
    grade_keyword_cols = []
    activity_keyword_cols = []

    for c, sc in scored:
        if c in (student_name, student_id, group):
            continue
        if sc["attendance"] >= 60:
            attendance_keyword_cols.append(c)
        if sc["grade"] >= 60:
            grade_keyword_cols.append(c)
        if sc["activity"] >= 60:
            activity_keyword_cols.append(c)

    # attendance cols
    for c, sc in scored:
        if c in (student_name, student_id, group):
            continue
        if index_like.get(c, False):
            continue
        if c in date_cols:
            attendance_cols.append(c)
            continue
        if sc["attendance"] >= 70:
            attendance_cols.append(c)
            continue
        if att_value_like.get(c, False):
            attendance_cols.append(c)

    # grade cols (исключаем: индексные/уникальные/attendance-like)
    for c, sc in scored:
        if c in (student_name, student_id, group) or c in attendance_cols or c in date_cols:
            continue
        if index_like.get(c, False):
            continue
        if col_uniqueness.get(c, False):  # защита от ID/уникальных колонок
            continue
        if att_value_like.get(c, False):  # защита от посещаемости
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
        if index_like.get(c, False):
            continue

        # прежнее правило по заголовку
        if sc["activity"] >= 65 and numeric_ratio.get(c, 0.0) < 0.85:
            activity_cols.append(c)
            continue

        # по значениям
        if act_val_ratio.get(c, 0.0) >= 0.05 and avg_len.get(c, 0.0) >= 10:
            activity_cols.append(c)
            continue

        # "укажите/перечислите" + длинный текст
        h = norm_text(c)
        if ("укажите" in h or "перечисл" in h or "опишите" in h) and avg_len.get(c, 0.0) >= 12:
            activity_cols.append(c)

    # сила сигналов
    att_value_cols = [c for c in cols if att_value_like.get(c, False)]
    # важная правка: если много date_cols - это очень сильный сигнал посещаемости
    att_strong = (len(date_cols) >= 3 and len(attendance_cols) >= 3) or (len(att_value_cols) >= 3) or (len(attendance_cols) >= 6)
    grade_real_cols = [c for c in grade_cols if (grade_like.get(c, 0.0) >= 0.20 or numeric_ratio.get(c, 0.0) >= 0.50)]
    grades_strong = (len(grade_real_cols) >= 2) or (len(grade_keyword_cols) >= 1 and len(grade_cols) >= 1)
    activity_strong = (len(activity_cols) >= 2) or (len(activity_cols) >= 1 and (len(activity_keyword_cols) >= 1 or "ctf" in norm_text(context_text)))

    # финальный выбор типа
    form_markers = _headers_look_like_google_forms(cols)
    if att_strong:
        type_hint = "attendance"
    elif activity_strong:
        type_hint = "activity"
    elif grades_strong:
        type_hint = "grades"
    else:
        # CSV из формы с одной активностью должен становиться activity,
        # если видно form_markers и есть хотя бы 1 activity_col, а оценок/посещаемости не нашли
        if form_markers and len(activity_cols) >= 1 and len(grade_cols) == 0 and len(attendance_cols) == 0:
            type_hint = "activity"
        else:
            type_hint = "unknown"

    # activity kind + trusted flag
    activity_kind_auto = _infer_activity_kind(df, activity_cols, context_text=context_text) if type_hint == "activity" else "Прочее"
    exclude = {student_name, student_id, group, "_origin_row"}
    exclude = {x for x in exclude if x}
    if type_hint == "activity":
        if not activity_cols or (activity_kind_auto in ("Публикация", "Конференция") and len(activity_cols) < 1):
            prefer = PUB_COL_HINTS if activity_kind_auto in ("Публикация", "Конференция") else [
                "мероприят", "конкурс", "акселератор", "ctf", "конференц", "доклад", "статья", "публикац", "научн"
            ]
            activity_cols = _suggest_activity_cols(df, cols, exclude, prefer) or activity_cols
    activity_kind = "Авто" if type_hint == "activity" else activity_kind_auto
    activity_trusted = _auto_activity_trusted(cols, type_hint)

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
        "activity_kind": activity_kind,
        "activity_trusted": activity_trusted,
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