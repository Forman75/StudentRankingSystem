"""
Этот пакет содержит:
- загрузку таблиц (CSV/XLSX)
- распознавание структуры (schema inference)
- извлечение фактов/доказательств (evidence)
- дедупликацию активностей
- подсчёт итоговых баллов
- экспорт отчётов
"""
from .ingest import load_tables_from_uploads
from .infer import build_dataframe_with_headers, infer_schema, persist_profile_for_schema
from .extract import (extract_attendance_wide, extract_grades_wide, extract_activities_text, make_student_key)
from .dedupe import dedupe_evidence_rows
from .scoring import compute_scores
from .export import export_to_excel_bytes

__all__ = [
    "load_tables_from_uploads",
    "build_dataframe_with_headers",
    "infer_schema",
    "persist_profile_for_schema",
    "extract_attendance_wide",
    "extract_grades_wide",
    "extract_activities_text",
    "make_student_key",
    "dedupe_evidence_rows",
    "compute_scores",
    "export_to_excel_bytes",
]
