from __future__ import annotations
from typing import Dict
import re
import pandas as pd
from .utils import norm_text, norm_name

# fuzzy fallback
try:
    from rapidfuzz import fuzz  # type: ignore

    def _sim(a: str, b: str) -> int:
        return int(fuzz.ratio(a, b))
except Exception:
    from difflib import SequenceMatcher

    def _sim(a: str, b: str) -> int:
        if not a or not b:
            return 0
        return int(100 * SequenceMatcher(None, a, b).ratio())


_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_SYSTEM_NAME_HINTS = [
    "пользователь", "user", "username", "display", "отображаем", "аккаунт", "account"
]
def _looks_like_email(s: str) -> bool:
    return bool(_EMAIL_RE.search(s or ""))

def _looks_like_system_user(s: str) -> bool:
    t = norm_text(s)
    return any(h in t for h in _SYSTEM_NAME_HINTS)

def _canon_id(s: str) -> str:
    x = norm_text(s)
    if x in ("", "nan", "none", "0"):
        return ""
    x = re.sub(r"\s+", "", x)
    return x

def _canon_grp(s: str) -> str:
    # Нормализация группы: нижний регистр, ё->е (в norm_text); убираем пробелы вокруг дефиса; если есть цифры, убираем все пробелы (обычно это код группы)
    x = norm_text(s)
    if x in ("", "nan", "none"):
        return ""
    # "ивт - 21" -> "ивт-21"
    x = re.sub(r"\s*-\s*", "-", x)
    # убираем пробелы
    if re.search(r"\d", x):
        x = x.replace(" ", "")
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _canon_name(s: str) -> str:
    # Нормализация ФИО: чистим спецсимволы (norm_name); если выглядит как email/логин/системное имя - игнорируем; если очень коротко или есть цифры - тоже игнорируем
    raw = str(s or "").strip()
    if not raw:
        return ""

    # email/логин
    if _looks_like_email(raw) or _looks_like_system_user(raw):
        return ""
    x = norm_name(raw)

    if x in ("", "nan", "none"):
        return ""

    # Если в "имени" есть цифры - это не ФИО
    if re.search(r"\d", x):
        return ""

    # слишком короткие строки - шум
    if len(x) < 5:
        return ""

    # нормализуем точки/дефисы/множественные пробелы
    x = re.sub(r"\s+", " ", x).strip()
    return x


def unify_students(evid: pd.DataFrame, enable_fuzzy: bool = True) -> Dict[str, str]:
    """
    Возвращает mapping: исходный student_key -> канонический ключ
    Канонический ключ если есть student_id: "id:<id>", иначе: "name:<фио>|grp:<группа>"
    """
    if evid.empty:
        return {}

    df = evid.copy()
    df["__id"] = df.get("student_id", "").astype(str).map(_canon_id)
    df["__grp"] = df.get("group", "").astype(str).map(_canon_grp)
    df["__name"] = df.get("student_name", "").astype(str).map(_canon_name)
    df["__key"] = df.get("student_key", "").astype(str)

    # статистика уникальности ФИО внутри группы
    grp_name_counts = (
        df[(df["__grp"] != "") & (df["__name"] != "")]
        .groupby(["__grp", "__name"])["__key"]
        .count()
        .to_dict()
    )

    # для каждого ФИО какие группы встречались
    name_to_groups: Dict[str, set[str]] = {}
    for _, r in df.iterrows():
        n = r["__name"]
        g = r["__grp"]
        if not n:
            continue
        name_to_groups.setdefault(n, set())
        if g:
            name_to_groups[n].add(g)

    # связь (grp,name) -> set(ids)
    id_by_grpname: Dict[tuple[str, str], set[str]] = {}
    # связь name -> set(ids) (если у id-строки нет группы)
    id_by_name: Dict[str, set[str]] = {}

    for _, r in df.iterrows():
        sid = r["__id"]
        n = r["__name"]
        g = r["__grp"]
        if not sid or not n:
            continue
        if g:
            id_by_grpname.setdefault((g, n), set()).add(sid)
        else:
            id_by_name.setdefault(n, set()).add(sid)

    # fuzzy: внутри группы сопоставим похожее ФИО только если у кандидата есть ID, у исходного нет ID, и ФИО уникально в группе
    fuzzy_map: Dict[tuple[str, str], str] = {}
    if enable_fuzzy:
        # в каждой группе список уникальных ФИО -> единственный ID
        by_group: Dict[str, Dict[str, set[str]]] = {}
        for _, r in df.iterrows():
            g = r["__grp"]
            n = r["__name"]
            sid = r["__id"]
            if not g or not n or not sid:
                continue
            # если имя в группе не уникально - не используем для fuzzy
            if grp_name_counts.get((g, n), 0) != 1:
                continue
            by_group.setdefault(g, {})
            by_group[g].setdefault(n, set()).add(sid)

        def _ok_for_fuzzy(name: str) -> bool:
            # минимум 2 слова; без цифр (уже отфильтровано), но проверим
            if not name:
                return False
            if re.search(r"\d", name):
                return False
            parts = [p for p in name.split() if p]
            return len(parts) >= 2

        # сопоставляем "без id"
        for _, r in df.iterrows():
            if r["__id"]:
                continue
            g = r["__grp"]
            n = r["__name"]
            if not g or not n:
                continue
            if not _ok_for_fuzzy(n):
                continue
            # если ФИО не уникально в группе - не трогаем
            if grp_name_counts.get((g, n), 0) != 1:
                continue

            cand = by_group.get(g, {})
            if not cand:
                continue

            best_name = None
            best_score = 0
            best_ids = None

            for cn, ids_set in cand.items():
                if not _ok_for_fuzzy(cn):
                    continue
                sc = _sim(n, cn)
                if sc > best_score:
                    best_score = sc
                    best_name = cn
                    best_ids = ids_set

            # строгий порог (оставляем высокий, чтобы не склеивать разных людей)
            if best_score >= 96 and best_ids and len(best_ids) == 1:
                fuzzy_map[(g, n)] = list(best_ids)[0]

    mapping: Dict[str, str] = {}

    for _, r in df.iterrows():
        orig_key = r["__key"]
        sid = r["__id"]
        n = r["__name"]
        g = r["__grp"]

        if sid:
            mapping[orig_key] = f"id:{sid}"
            continue

        # нет ID → работаем по ФИО/группе
        if n and g:
            # если ФИО уникально в группе
            if grp_name_counts.get((g, n), 0) == 1:
                ids = id_by_grpname.get((g, n), set())
                if len(ids) == 1:
                    mapping[orig_key] = f"id:{list(ids)[0]}"
                    continue

                # если где-то был ID без группы, и ФИО встречается только в одной группе
                if len(name_to_groups.get(n, set()) or set()) <= 1:
                    ids2 = id_by_name.get(n, set())
                    if len(ids2) == 1:
                        mapping[orig_key] = f"id:{list(ids2)[0]}"
                        continue

                # fuzzy-подвязка
                if (g, n) in fuzzy_map:
                    mapping[orig_key] = f"id:{fuzzy_map[(g, n)]}"
                    continue

            mapping[orig_key] = f"name:{n}|grp:{g}"
            continue

        if n:
            # если ФИО встречается только в одной группе и есть id_by_name
            if len(name_to_groups.get(n, set()) or set()) == 1:
                ids2 = id_by_name.get(n, set())
                if len(ids2) == 1:
                    mapping[orig_key] = f"id:{list(ids2)[0]}"
                    continue
            mapping[orig_key] = f"name:{n}|grp:{g}"
            continue

        # совсем нет данных
        mapping[orig_key] = orig_key

    return mapping
