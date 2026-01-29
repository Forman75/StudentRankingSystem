from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from .utils import norm_name, norm_text, rules_path, load_json

RULES = load_json(rules_path(), {})
# =========================

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

# 2) Успеваемость (оценки / баллы -> 2..5 -> 0..1)
# =========================
_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_GRADE_BR_RE = re.compile(r"[\(\[\{]\s*([2-5])\s*[\)\]\}]")

def _points_to_grade(p: float) -> int:
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

    if "незач" in t or "не зач" in t or t in ("нз", "н/з", "fail"):
        return 2
    if "зач" in t or t in ("з", "з/ч", "pass"):
        return 5

    m = _GRADE_BR_RE.search(s)
    if m:
        try:
            g = int(m.group(1))
            if 2 <= g <= 5:
                return g
        except Exception:
            pass

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

    if len(vals) == 1 and vals[0] <= 5.5:
        g = int(round(vals[0]))
        return g if 2 <= g <= 5 else None

    p = float(vals[0])
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

    # если в grade_cols в основном +/Н/У, то это не оценки
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

    if used_cells < max(10, int(0.02 * max(1, len(df) * max(1, len(grade_cols))))):
        return pd.DataFrame(), []

    mat = pd.concat(per_col_grade, axis=1)
    avg_grade = mat.mean(axis=1, skipna=True)

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

# 3) Активности (несколько событий в одной ячейке)
# =========================
KIND_KEYWORDS = {
    "Акселератор": ["акселератор", "startup", "стартап", "аксел"],
    "CTF": ["ctf", "security", "кибер", "pwn", "revers", "crypto"],
    "Публикация": ["публикац", "статья", "scopus", "wos", "ринц", "journal", "doi"],
    "Олимпиада/конкурс": ["олимпиад", "конкурс", "хакатон", "hackathon"],
    "Конференция": ["конференц", "семинар", "форум", "доклад", "тезис"],
    "Спорт": ["спорт", "турнир", "матч", "соревн"],
    "Волонтёрство": ["волонтер", "добровол"],
}

ROLE_REGEX = {
    "Победитель": re.compile(r"(побед|1\s*место|i\s*место|\bwinner\b)", re.I),
    "Призёр": re.compile(r"(приз|2\s*место|3\s*место|ii\s*место|iii\s*место|\bprize\b)", re.I),
    "Организатор": re.compile(r"(организ|оргком|куратор|жюри)", re.I),
    "Участник": re.compile(r"(участ(ник|ие)|participant|финалист|finalist)", re.I),
}
ROLE_PRIORITY = ["Победитель", "Призёр", "Организатор", "Участник"]

URL_RE = re.compile(r"https?://\S+", re.I)

_EMPTY_EVENT_MARKERS = {"", "nan", "none", "-", "—"}

_PROOF_ABSENT_MARKERS = {
    "нет", "отсутствует", "не приложено", "не приложен", "не приложила", "не приложил",
    "не предоставлено", "не предоставлен", "без доказательства", "без доказ", "нет доказ",
    "n/a", "na", "-", "0"
}


def _proof_is_absent_token(s: str) -> bool:
    t = norm_text(s)
    if not t:
        return True
    if t in _PROOF_ABSENT_MARKERS:
        return True
    if "отсутств" in t and len(t) <= 25:
        return True
    if t.startswith("нет") and len(t) <= 30:
        return True
    if "не прилож" in t or "не предостав" in t:
        return True
    return False


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


def _detect_role_or_none(text: str) -> str | None:
    s = text or ""
    for role in ROLE_PRIORITY:
        if ROLE_REGEX[role].search(s):
            return role
    return None

def _detect_role(text: str) -> str:
    r = _detect_role_or_none(text)
    return r if r else "Участник"

def _extract_first_url(s: str) -> str:
    m = URL_RE.search(s or "")
    return m.group(0).strip() if m else ""

def _strip_urls(s: str) -> str:
    return URL_RE.sub(" ", s or "")

def _split_events(text: str) -> List[str]:
    if not text:
        return []
    # Разделяем по переносу строк / ; / буллетам
    parts = re.split(r"(?:\r\n|\n|\r|;)+|(?:^\s*[\-\•\*]\s+)|(?:\s\|\s)", text)
    parts = [p.strip() for p in parts if p and norm_text(p) not in _EMPTY_EVENT_MARKERS]
    return parts[:30]

def _kind_from_token(tok: str) -> str | None:
    t = norm_text(tok)
    if not t:
        return None
    for k in KIND_KEYWORDS.keys():
        if t == norm_text(k):
            return k
    # частые варианты
    if "олимпиад" in t or "конкурс" in t:
        return "Олимпиада/конкурс"
    if "конференц" in t:
        return "Конференция"
    if "публикац" in t or "статья" in t:
        return "Публикация"
    if "волонтер" in t or "волонтёр" in t:
        return "Волонтёрство"
    if "спорт" in t:
        return "Спорт"
    if "акселератор" in t:
        return "Акселератор"
    if "ctf" in t:
        return "CTF"
    return None

def _parse_structured_event(ev: str) -> tuple[str | None, str | None, str, str, bool]:
    # Пытаемся распознать: ТИП - РОЛЬ - ОПИСАНИЕ - URL, возвращает: (kind_exp, role_exp, title, proof_url, fixed)
    s0 = (ev or "").strip()
    if not s0:
        return None, None, "", "", False

    proof = _extract_first_url(s0)
    fixed = False

    # нормализуем тире/разделители
    s = s0.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s*:\s*", " : ", s)

    # режем по " - "
    parts = [p.strip() for p in s.split(" - ") if p.strip()]
    if len(parts) < 2:
        return None, None, s0, proof, False

    kind = _kind_from_token(parts[0])
    role = _detect_role_or_none(parts[1])

    title = " - ".join(parts[2:]).strip() if len(parts) >= 3 else ""

    # если URL нет, но третья часть — "отсутствует/нет" => это отсутствие доказательства
    if not proof and title and _proof_is_absent_token(title):
        fixed = True

    if not title:
        title = s0.strip()

    return kind, role, title, proof, fixed


def _event_key(kind: str, role: str, title: str) -> str:
    t = norm_text(_strip_urls(title))
    t = re.sub(r"\s+", " ", t).strip()
    return norm_text(f"{kind}|{role}|{t}")


def extract_activities_text(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    source: str,
    activity_points: Dict[str, Dict[str, float]],
    context_text: str = "",
    strict_validation: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    strict_validation=True  -> режим "форма": без URL баллы НЕ начисляются + issues
    strict_validation=False -> доверенная таблица организатора: URL и формат не обязательны
    """
    name_col = schema.get("student_name")
    if not name_col or name_col not in df.columns:
        return pd.DataFrame(), []

    act_cols = [c for c in schema.get("activity_cols", []) if c in df.columns]
    if not act_cols:
        return pd.DataFrame(), []

    idx = df.index
    names, ids, grps = _pack_person_fields(df, schema)
    origin = _get_col_as_series(df, "_origin_row").reindex(idx).to_numpy()

    row_text = df.loc[idx, act_cols].astype(str).agg("\n".join, axis=1)

    forced_kind = (schema.get("activity_kind") or schema.get("activity_kind_auto") or "Авто").strip()
    if not forced_kind:
        forced_kind = "Авто"

    evidence: List[Dict[str, Any]] = []
    totals: List[float] = []
    events_count: List[int] = []
    suppressed_count: List[int] = []
    issues_count: List[int] = []

    seen_global: set[tuple[str, str]] = set()  # (student_key, event_key)

    for i, ridx in enumerate(df.index):
        txt = str(row_text.loc[ridx] or "")
        tnorm = norm_text(txt)
        if not tnorm or tnorm in _EMPTY_EVENT_MARKERS:
            totals.append(0.0)
            events_count.append(0)
            suppressed_count.append(0)
            issues_count.append(0)
            continue

        events = _split_events(txt)
        pts_sum = 0.0
        ec = 0
        sc = 0
        ic = 0

        student_key = make_student_key(ids.iloc[i], names.iloc[i], grps.iloc[i])

        for ev in events:
            if norm_text(ev) in _EMPTY_EVENT_MARKERS:
                continue

            ec += 1
            kind_exp, role_exp, title, proof, fixed = _parse_structured_event(ev)

            # kind
            if forced_kind not in ("Авто", ""):
                kind = forced_kind
                kind_exp = kind_exp or kind
                conf = 1.0
            else:
                if kind_exp:
                    kind = kind_exp
                    conf = 1.0
                else:
                    kind = _infer_kind(ev, context_text=context_text)
                    conf = 0.85 if kind != "Прочее" else 0.70

            if kind not in activity_points:
                kind = "Прочее"

            # role
            role = role_exp if role_exp else _detect_role(ev)
            if not role_exp:
                conf = min(conf, 0.9)

            proof_missing = (not proof)

            issues: List[str] = []
            if kind == "Прочее" and not kind_exp and forced_kind in ("Авто", ""):
                issues.append("unknown_kind")
            if role == "Участник" and role_exp is None:
                issues.append("role_inferred_default")

            # строгий режим (форма): URL обязателен для любого события
            if strict_validation and proof_missing:
                # явно написал "отсутствует/нет"
                if _proof_is_absent_token(title) or any(x in norm_text(ev) for x in ["отсутств", "нет доказ", "без доказ", "не прилож", "не предостав"]):
                    issues.append("proof_marked_absent")
                issues.append("missing_proof_url")
                issues.append("points_suppressed_no_proof")

            # строгая форма: формат желательно структурированный
            if strict_validation and (kind_exp is None and role_exp is None):
                issues.append("unstructured_format")

            # key для дублей
            ek = _event_key(kind, role, title)
            full_key = (student_key, ek)

            # начисление
            p_raw = float(activity_points.get(kind, {}).get(role, activity_points.get("Прочее", {}).get(role, 0.0)))
            p = 0.0 if (strict_validation and proof_missing) else p_raw

            # дубль: не начисляем второй раз, но пишем запись и issue
            if full_key in seen_global:
                issues = list(issues) + ["duplicate_event_skipped"]
                p = 0.0
                sc += 1  # считаем как "пропущено"

            else:
                seen_global.add(full_key)
                if strict_validation and proof_missing:
                    sc += 1

            if issues:
                ic += 1

            pts_sum += p

            evidence.append({
                "student_key": student_key,
                "student_name": str(names.iloc[i]),
                "student_id": str(ids.iloc[i]),
                "group": str(grps.iloc[i]),
                "category": "C",
                "points": float(p),
                "source": source,
                "detail": f"{kind} — {role}: {title}" + (f" ({proof})" if proof else "") + (" [ДУБЛЬ]" if "duplicate_event_skipped" in issues else ""),
                "origin_row": int(origin[i]) if str(origin[i]) not in ("nan", "None", "") else "",

                # дополнительные поля для quality/дедупа
                "event_key": ek,
                "kind": kind,
                "role": role,
                "title": title,
                "proof": proof,
                "fixed": bool(fixed),
                "confidence": float(conf),
                "issues": issues,
            })

        totals.append(float(pts_sum))
        events_count.append(int(ec))
        suppressed_count.append(int(sc))
        issues_count.append(int(ic))

    out = pd.DataFrame({
        "student_name": names.to_numpy(),
        "student_id": ids.to_numpy(),
        "group": grps.to_numpy(),
        "activity_points_total": np.array(totals, dtype=float),
        "activity_events": np.array(events_count, dtype=int),
        "activity_suppressed": np.array(suppressed_count, dtype=int),
        "activity_issues": np.array(issues_count, dtype=int),
        "_origin_row": origin
    })

    return out, evidence