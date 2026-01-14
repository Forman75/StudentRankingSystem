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

def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower().replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s

def norm_name(s: Any) -> str:
    s = norm_text(s)
    s = re.sub(r"[^a-zа-я0-9\s\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def try_parse_date(s: Any) -> Optional[str]:
    if s is None:
        return None
    if hasattr(s, "year") and hasattr(s, "month") and hasattr(s, "day"):
        try:
            return f"{int(s.year):04d}-{int(s.month):02d}-{int(s.day):02d}"
        except Exception:
            pass

    txt = norm_text(s)
    if not txt:
        return None

    txt = txt.replace(",", ".")
    if re.match(r"^\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?$", txt):
        try:
            dt = dtparser.parse(txt, dayfirst=True, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    if re.match(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}$", txt):
        try:
            dt = dtparser.parse(txt, dayfirst=False, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    return None

def is_mostly_unique(values, min_unique_ratio=0.7) -> bool:
    vals = [v for v in values if v not in (None, "", "nan")]
    if len(vals) < 20:
        return False
    uniq = len(set(vals))
    return (uniq / max(1, len(vals))) >= min_unique_ratio

def column_signature(columns) -> str:
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
