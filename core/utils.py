import os
import re
import json
import hashlib
from pathlib import Path
from typing import Any, Optional
from dateutil import parser as dtparser

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"

APPDATA = os.environ.get("APPDATA")
if APPDATA:
    USER_DATA_DIR = Path(APPDATA) / "StudentRanker" / "data"
else:
    USER_DATA_DIR = DEFAULT_DATA_DIR  # fallback

def load_json(path: Path, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

_DASH_CHARS_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_NBSP_RE = re.compile(r"[\u00A0\u2007\u202F]")  # NBSP варианты


def norm_text(s: Any) -> str:
    """
    Универсальная нормализация текста:
    - lower
    - ё -> е
    - BOM/неразрывные пробелы
    - внешние кавычки
    - все виды тире -> '-'
    - схлопывание пробелов
    """
    if s is None:
        return ""

    s = str(s)

    # частые "невидимые" символы CSV/Excel
    s = s.replace("\ufeff", "")
    s = _NBSP_RE.sub(" ", s)
    s = s.strip()
    # убрать внешние кавычки
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    s = s.lower().replace("ё", "е")
    # разные тире/дефисы в один стандарт
    s = _DASH_CHARS_RE.sub("-", s)
    # схлопнуть пробелы
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_name(s: Any) -> str:
    """
    Нормализация ФИО/имени:
    - основана на norm_text
    - сохраняет буквы/цифры/пробелы/дефис/точку
    - вычищает остальное
    """
    s = norm_text(s)
    if not s:
        return ""

    # оставляем буквы/цифры/пробел/дефис/точку
    s = re.sub(r"[^a-zа-я0-9\s\-\.\']", " ", s)
    # дефисы приводим к формату "без лишних пробелов"
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def try_parse_date(s: Any) -> Optional[str]:
    # парсинг даты из заголовков (для определения посещаемости по датам)
    if s is None:
        return None

    # pandas.Timestamp / datetime.date / datetime.datetime
    if hasattr(s, "year") and hasattr(s, "month") and hasattr(s, "day"):
        try:
            return f"{int(s.year):04d}-{int(s.month):02d}-{int(s.day):02d}"
        except Exception:
            pass

    txt = norm_text(s)
    if not txt:
        return None

    txt = txt.replace(",", ".")
    # dd.mm[.yyyy]
    if re.match(r"^\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?$", txt):
        try:
            dt = dtparser.parse(txt, dayfirst=True, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    # yyyy-mm-dd
    if re.match(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}$", txt):
        try:
            dt = dtparser.parse(txt, dayfirst=False, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    return None

def is_mostly_unique(values, min_unique_ratio=0.7) -> bool:
    # Проверяет, что в колонке достаточно уникальных значений, как эвристика (например, ID/временные метки)
    vals = []
    for v in values:
        t = norm_text(v)
        if t in ("", "nan", "none"):
            continue
        vals.append(t)

    if len(vals) < 20:
        return False

    uniq = len(set(vals))
    return (uniq / max(1, len(vals))) >= min_unique_ratio

def column_signature(columns) -> str:
    # Сигнатура структуры таблицы по заголовкам (для профилей распознавания)
    joined = "||".join([norm_text(c) for c in columns])
    return hashlib.md5(joined.encode("utf-8")).hexdigest()

def rules_path() -> Path:
    return DEFAULT_DATA_DIR / "rules.json"

def profiles_path() -> Path:
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = USER_DATA_DIR / "profiles.json"
    if not p.exists():
        save_json(p, {})
    return p

def manual_entries_path() -> Path:
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = USER_DATA_DIR / "manual_entries.json"
    if not p.exists():
        save_json(p, [])
    return p
