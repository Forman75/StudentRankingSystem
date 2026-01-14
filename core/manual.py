from __future__ import annotations
from typing import List, Dict, Any
from .utils import manual_entries_path, load_json, save_json

def load_manual_entries() -> List[Dict[str, Any]]:
    return load_json(manual_entries_path(), [])

def save_manual_entries(entries: List[Dict[str, Any]]) -> None:
    save_json(manual_entries_path(), entries)