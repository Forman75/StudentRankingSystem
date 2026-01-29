from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple, Optional, Set

_URL_RE = re.compile(r"https?://\S+", re.I)

def _norm(s: Any) -> str:
    # Лёгкая нормализация для ключей (без зависимости от utils.norm_text)
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = t.replace("\ufeff", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _strip_urls(s: str) -> str:
    return _URL_RE.sub(" ", s or "")


def _event_base_key_from_row(r: Dict[str, Any]) -> str:
    """
    Строит ключ события без student_key. Используется если event_key отсутствует/пустой.
    Приоритет:
      1) kind + role + title
      2) kind + role + detail (без ссылок)
      3) detail (без ссылок)
    """
    kind = _norm(r.get("kind", ""))
    role = _norm(r.get("role", ""))
    title = _norm(_strip_urls(str(r.get("title", ""))))
    detail = _norm(_strip_urls(str(r.get("detail", ""))))

    if kind or role:
        if title:
            base = f"{kind}|{role}|{title}"
        elif detail:
            base = f"{kind}|{role}|{detail}"
        else:
            base = f"{kind}|{role}"
    else:
        base = detail

    base = re.sub(r"\s+", " ", base).strip()
    return base


def _normalized_event_key(r: Dict[str, Any]) -> str:
    """
    Возвращает "базовый" event_key (без student_key-префикса, если он туда встроен).
    В некоторых реализациях event_key делает как:
      "{student_key}|{kind}|{role}|{title}"
    """
    sk = _norm(r.get("student_key", ""))
    ek = str(r.get("event_key", "") or "").strip()

    if ek and sk and _norm(ek).startswith(sk + "|"):
        ek = ek[len(sk) + 1 :].strip()

    if not ek:
        ek = _event_base_key_from_row(r)

    return _norm(ek)


def _issues_count(r: Dict[str, Any]) -> int:
    issues = r.get("issues", [])
    if issues is None:
        return 0
    if isinstance(issues, str):
        return 1
    if isinstance(issues, list):
        return len([x for x in issues if x])
    return 0


def _confidence(r: Dict[str, Any]) -> float:
    # поддержка разных имён поля
    for k in ("activity_confidence", "confidence", "conf"):
        if k in r and r[k] is not None:
            try:
                return float(r[k])
            except Exception:
                pass
    return 1.0


def _has_proof(r: Dict[str, Any]) -> int:
    proof = str(r.get("proof", "") or "").strip()
    if proof:
        return 1
    # иногда proof может быть только в detail
    d = str(r.get("detail", "") or "")
    return 1 if "http://" in d.lower() or "https://" in d.lower() else 0


def _title_len(r: Dict[str, Any]) -> int:
    t = str(r.get("title", "") or "")
    t = _strip_urls(t)
    t = re.sub(r"\s+", " ", t).strip()
    return len(t)


def _rank_row_for_keep(r: Dict[str, Any]) -> Tuple[float, float, int, int, int]:
    """
    Чем больше - тем предпочтительнее запись оставить
    Приоритеты:
      1) points (больше лучше)
      2) confidence (больше лучше)
      3) наличие proof (лучше)
      4) длина title (длиннее обычно информативнее)
      5) меньше issues (лучше) => поэтому ставим отрицание
    """
    try:
        pts = float(r.get("points", 0.0) or 0.0)
    except Exception:
        pts = 0.0

    conf = _confidence(r)
    proof = _has_proof(r)
    tlen = _title_len(r)
    iss = _issues_count(r)

    return (pts, conf, proof, tlen, -iss)


def _dedupe_key(r: Dict[str, Any]) -> Optional[str]:
    # Полный ключ дедупа: student_key + category + normalized_event_key
    sk = _norm(r.get("student_key", ""))
    cat = _norm(r.get("category", ""))
    if not sk or not cat:
        return None

    ek = _normalized_event_key(r)
    if not ek:
        return None

    return f"{sk}|{cat}|{ek}"


def dedupe_evidence_rows(
    evidence_rows: List[Dict[str, Any]],
    categories: Set[str] | None = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Удаляет дубликаты evidence-строк по событиям, внутри одного студента
    Возвращает:
      - filtered_rows: список evidence без дублей
      - duplicates_log: список удалённых записей (с указанием что оставили)
    categories (по умолчанию {"C"})
    """
    if categories is None:
        categories = {"C"}

    kept: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []

    # key -> index в kept
    seen: Dict[str, int] = {}
    for r in evidence_rows:
        cat = str(r.get("category", "") or "")
        if cat not in categories:
            kept.append(r)
            continue

        key = _dedupe_key(r)
        if not key:
            kept.append(r)
            continue

        if key not in seen:
            seen[key] = len(kept)
            kept.append(r)
            continue

        # найден дубль - решаем, что оставить
        kept_idx = seen[key]
        current_kept = kept[kept_idx]
        rank_new = _rank_row_for_keep(r)
        rank_old = _rank_row_for_keep(current_kept)
        # keep the better one
        if rank_new > rank_old:
            # заменяем kept, старое уходит в duplicates
            kept[kept_idx] = r
            duplicates.append({
                "reason": "duplicate_event_key",
                "student_key": current_kept.get("student_key", ""),
                "student_name": current_kept.get("student_name", ""),
                "group": current_kept.get("group", ""),
                "category": current_kept.get("category", ""),

                "event_key": _normalized_event_key(current_kept),

                "kept_source": r.get("source", ""),
                "kept_points": r.get("points", 0.0),
                "kept_detail": r.get("detail", ""),

                "dropped_source": current_kept.get("source", ""),
                "dropped_points": current_kept.get("points", 0.0),
                "dropped_detail": current_kept.get("detail", ""),

                "kept_rank": rank_new,
                "dropped_rank": rank_old,
            })
        else:
            # новое уходит в duplicates
            duplicates.append({
                "reason": "duplicate_event_key",
                "student_key": r.get("student_key", ""),
                "student_name": r.get("student_name", ""),
                "group": r.get("group", ""),
                "category": r.get("category", ""),

                "event_key": _normalized_event_key(r),

                "kept_source": current_kept.get("source", ""),
                "kept_points": current_kept.get("points", 0.0),
                "kept_detail": current_kept.get("detail", ""),

                "dropped_source": r.get("source", ""),
                "dropped_points": r.get("points", 0.0),
                "dropped_detail": r.get("detail", ""),

                "kept_rank": rank_old,
                "dropped_rank": rank_new,
            })

    return kept, duplicates
