from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
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


def _canon_id(s: str) -> str:
    x = norm_text(s)
    if x in ("", "nan", "none", "0"):
        return ""
    return x


def _canon_grp(s: str) -> str:
    x = norm_text(s)
    return "" if x in ("nan", "none") else x


def _canon_name(s: str) -> str:
    x = norm_name(s)
    return "" if x in ("nan", "none") else x


def unify_students(evid: pd.DataFrame, enable_fuzzy: bool = True) -> Dict[str, str]:
    if evid.empty:
        return {}

    # нормализуем поля
    df = evid.copy()
    df["__id"] = df.get("student_id", "").astype(str).map(_canon_id)
    df["__grp"] = df.get("group", "").astype(str).map(_canon_grp)
    df["__name"] = df.get("student_name", "").astype(str).map(_canon_name)
    df["__key"] = df.get("student_key", "").astype(str)

    # статистика уникальности ФИО внутри группы
    grp_name_counts = (
        df[df["__grp"] != ""]
        .groupby(["__grp", "__name"])["__key"]
        .count()
        .to_dict()
    )

    # для каждого ФИО → какие группы встречались
    name_to_groups = {}
    for _, r in df.iterrows():
        n = r["__name"]
        g = r["__grp"]
        if not n:
            continue
        if n not in name_to_groups:
            name_to_groups[n] = set()
        if g:
            name_to_groups[n].add(g)

    # связь (grp,name) -> set(ids)
    id_by_grpname = {}
    # связь name -> set(ids) (если у id-строки нет группы)
    id_by_name = {}

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

    """
    fuzzy: внутри группы сопоставим "похожее ФИО" только если:
    у кандидата есть ID
    у исходного нет ID
    в группе нет дубликатов по ФИО
    """
    fuzzy_map = {}
    if enable_fuzzy:
        # соберём в каждой группе список
        by_group = {}
        for _, r in df.iterrows():
            g = r["__grp"]
            n = r["__name"]
            sid = r["__id"]
            if not g or not n or not sid:
                continue
            # если имя в группе не уникально — не используем для fuzzy
            if grp_name_counts.get((g, n), 0) != 1:
                continue
            by_group.setdefault(g, {})
            by_group[g].setdefault(n, set()).add(sid)

        # сопоставляем "без id"
        for _, r in df.iterrows():
            if r["__id"]:
                continue
            g = r["__grp"]
            n = r["__name"]
            if not g or not n:
                continue
            # если ФИО не уникально в группе — не трогаем
            if grp_name_counts.get((g, n), 0) != 1:
                continue
            cand = by_group.get(g, {})
            if not cand:
                continue

            best_name = None
            best_score = 0
            best_ids = None
            for cn, ids_set in cand.items():
                sc = _sim(n, cn)
                if sc > best_score:
                    best_score = sc
                    best_name = cn
                    best_ids = ids_set

            # строгий порог
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
