from __future__ import annotations
from typing import Any, Dict, List
import json
from .utils import manual_entries_path, norm_text

def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace(",", ".").strip()
        return float(s) if s else default
    except Exception:
        return default

def _normalize_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    # Приводит запись ручного ввода к ожидаемому формату
    if not isinstance(e, dict):
        return {}

    # используются UI
    fio = str(e.get("fio", e.get("ФИО", "")) or "").strip()
    group = str(e.get("group", e.get("Группа", "")) or "").strip()
    sid = str(e.get("id", e.get("student_id", e.get("ID", ""))) or "").strip()
    cat = str(e.get("cat", e.get("category", "")) or "").strip()
    if cat not in ("Активность (баллы)", "Доп. баллы (сразу в итог)"):
        # миграция из старых значений
        if cat in ("C", "activity", "активность"):
            cat = "Активность (баллы)"
        elif cat in ("X", "extra", "доп"):
            cat = "Доп. баллы (сразу в итог)"
        else:
            cat = "Доп. баллы (сразу в итог)"

    points = _as_float(e.get("points", e.get("баллы", 0.0)), 0.0)
    desc = str(e.get("desc", e.get("detail", e.get("Описание", ""))) or "").strip()
    if not desc:
        desc = "Ручной ввод"
    source = str(e.get("source", e.get("Источник", "")) or "").strip()
    if not source:
        source = "Ручной ввод"
    # опциональные расширения для event-level активности
    kind = str(e.get("kind", "") or "").strip()
    role = str(e.get("role", "") or "").strip()
    title = str(e.get("title", "") or "").strip()
    proof = str(e.get("proof", "") or "").strip()
    event_key = str(e.get("event_key", "") or "").strip()
    fixed = bool(e.get("fixed", False))
    confidence = _as_float(e.get("confidence", e.get("activity_confidence", 1.0)), 1.0)
    issues = e.get("issues", [])
    if issues is None:
        issues = []
    if isinstance(issues, str):
        issues = [issues]
    if not isinstance(issues, list):
        issues = []
    issues = [str(x) for x in issues if str(x).strip()]

    out = dict(e)  # сохраняем любые дополнительные поля
    out.update({
        "fio": fio,
        "group": group,
        "id": sid,
        "cat": cat,
        "points": float(points),
        "desc": desc,
        "source": source,

        # optional
        "kind": kind,
        "role": role,
        "title": title,
        "proof": proof,
        "event_key": event_key,
        "fixed": fixed,
        "confidence": float(confidence),
        "issues": issues,
    })
    return out

def load_manual_entries() -> List[Dict[str, Any]]:
    # Читает список ручных начислений из manual_entries.json
    path = manual_entries_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return []

    if not isinstance(obj, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in obj:
        norm = _normalize_entry(item) if isinstance(item, dict) else {}
        if not norm:
            continue
        # удаляем совсем пустые
        if not norm.get("fio") and not norm.get("id") and not norm.get("group"):
            continue
        out.append(norm)
    return out

def save_manual_entries(entries: List[Dict[str, Any]]) -> None:
    # Сохраняет список ручных начислений
    path = manual_entries_path()
    normed: List[Dict[str, Any]] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        ne = _normalize_entry(e)
        if not ne:
            continue
        # базовая валидация: чтобы не накапливать мусор
        if not ne.get("fio") or not ne.get("group") or not ne.get("id"):
            # UI уже требует эти поля, но на всякий
            continue
        # нормализация ключевых строк (чтобы dedupe проще работал в будущем)
        ne["fio"] = ne["fio"].strip()
        ne["group"] = ne["group"].strip()
        ne["id"] = ne["id"].strip()
        ne["desc"] = ne["desc"].strip()
        ne["source"] = ne["source"].strip()

        # если event_key задан - нормализуем пробелы
        if ne.get("event_key"):
            ek = str(ne["event_key"])
            ne["event_key"] = " ".join(ek.split())

        normed.append(ne)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normed, f, ensure_ascii=False, indent=2)
