from __future__ import annotations
import re
import streamlit as st
import pandas as pd
import hashlib
from core.ingest import load_tables_from_uploads
from core.infer import build_dataframe_with_headers, infer_schema, persist_profile_for_schema
from core.extract import (extract_attendance_wide, extract_grades_wide, extract_activities_text, make_student_key)
from core.dedupe import dedupe_evidence_rows
from core.scoring import compute_scores
from core.export import export_to_excel_bytes
from core.manual import load_manual_entries, save_manual_entries
from core.utils import rules_path, load_json, norm_text, norm_name

RULES = load_json(rules_path(), {})
ID_COL = "Номер студенческого билета"
st.set_page_config(page_title="Рейтинг студентов", layout="wide")
st.title("Система анализа таблиц и формирования рейтинга студентов")
# =========================

# Helpers
# =========================
_URL_RE = re.compile(r"https?://\S+", re.I)

ISSUE_MAP = {
    "missing_proof_url": ("error", "Нет ссылки-доказательства. В режиме формы баллы не начисляются."),
    "points_suppressed_no_proof": ("error", "Баллы НЕ начислены, потому что доказательство отсутствует (нет URL)."),
    "proof_marked_absent": ("error", "Студент явно указал, что доказательство отсутствует (например: 'Отсутствует/Нет')."),
    "unstructured_format": ("warn", "Не соблюдён строгий формат 'ТИП - РОЛЬ - ОПИСАНИЕ/ДОКАЗАТЕЛЬСТВО' (важно в режиме формы)."),
    "unknown_kind": ("warn", "Не удалось распознать тип активности (засчитано как 'Прочее')."),
    "role_inferred_default": ("info", "Роль не указана — выставлено по умолчанию 'Участник'."),
    "duplicate_event_skipped": ("warn", "Найден дубль активности. Баллы начислены один раз, повтор пропущен."),
    "auto_fix_applied": ("info", "Система автоматически исправила формат (разделители/пробелы/двоеточия и т.п.)."),
}

def _safe_key_prefix(src_key: str) -> str:
    # Используем MD5-хеш для создания безопасного и уникального ключа, который поддерживает кириллицу и спецсимволы в именах файлов.
    return hashlib.md5(src_key.encode("utf-8")).hexdigest()

def _strip_urls(s: str) -> str:
    return _URL_RE.sub(" ", s or "")

def _event_base_key(kind: str, role: str, title: str) -> str:
    t = norm_text(_strip_urls(title))
    t = re.sub(r"\s+", " ", t).strip()
    return norm_text(f"{kind}|{role}|{t}")

def _normalize_evidence_for_dedupe(evidence_rows: list[dict]) -> None:
    """
    1) Приводим event_key к виду без student_key-префикса (если он туда включён)
    2) Проставляем activity_confidence (для dedupe_evidence_rows)
    """
    for r in evidence_rows:
        if str(r.get("category", "")) != "C":
            continue

        sk = str(r.get("student_key", "")).strip()
        ek = str(r.get("event_key", "")).strip()

        if ek and sk and ek.startswith(sk + "|"):
            r["event_key"] = ek[len(sk) + 1 :].strip()
        elif not ek:
            kind = str(r.get("kind", "Прочее"))
            role = str(r.get("role", "Участник"))
            title = str(r.get("title", r.get("detail", "")))
            r["event_key"] = _event_base_key(kind, role, title)

        if "activity_confidence" not in r:
            if "confidence" in r:
                r["activity_confidence"] = r.get("confidence")
            else:
                r["activity_confidence"] = 1.0

def _build_quality_rows_from_evidence(evidence_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ev in evidence_rows:
        if str(ev.get("category", "")) != "C":
            continue

        issues = ev.get("issues", [])
        if issues is None:
            issues = []
        if isinstance(issues, str):
            issues = [issues]
        if not isinstance(issues, list):
            issues = []

        fixed = bool(ev.get("fixed", False))
        if fixed and "auto_fix_applied" not in issues:
            issues = list(issues) + ["auto_fix_applied"]

        for code in issues:
            if not code:
                continue
            level, msg = ISSUE_MAP.get(code, ("warn", f"Проблема: {code}"))

            out.append({
                "Уровень": level,
                "Код": code,
                "Сообщение": msg,

                "ФИО": str(ev.get("student_name", "")),
                "Группа": str(ev.get("group", "")),
                ID_COL: str(ev.get("student_id", "")),
                "student_key": str(ev.get("student_key", "")),

                "Тип": str(ev.get("kind", "")),
                "Роль": str(ev.get("role", "")),
                "Название/описание": str(ev.get("title", "")),
                "Доказательство": str(ev.get("proof", "")),

                "Источник": str(ev.get("source", "")),
                "Строка (Excel)": str(ev.get("origin_row", "")),

                "Confidence": float(ev.get("confidence", ev.get("activity_confidence", 1.0)) or 1.0),
                "Начислено (баллы)": float(ev.get("points", 0.0) or 0.0),
            })
    return out

def _sanitize_schema_for_df(schema: dict, df: pd.DataFrame, context_text: str) -> dict:
    # Защита от ситуации, когда Streamlit подставил старые значения виджетов
    cols = set(df.columns)

    base = infer_schema(df, context_text=context_text)

    def fix_one(k: str):
        v = schema.get(k)
        if not v or v not in cols:
            schema[k] = base.get(k)

    fix_one("student_name")
    fix_one("student_id")
    fix_one("group")

    for lk in ["attendance_cols", "grade_cols", "activity_cols"]:
        lst = schema.get(lk, [])
        if not isinstance(lst, list):
            lst = []
        lst = [c for c in lst if c in cols]
        if not lst:
            lst = base.get(lk, []) or []
            lst = [c for c in lst if c in cols]
        schema[lk] = lst

    if schema.get("type_hint") not in ("attendance", "grades", "activity", "unknown"):
        schema["type_hint"] = base.get("type_hint", "unknown")

    # activity-specific defaults
    schema.setdefault("activity_trusted", bool(base.get("activity_trusted", False)))
    if schema.get("type_hint") == "activity":
        schema.setdefault("activity_kind", "Авто")

    return schema

def _context_from_raw(df_raw: pd.DataFrame, max_rows: int = 4, max_cols: int = 10) -> str:
    try:
        top = df_raw.iloc[:max_rows, :max_cols].astype(str).values.ravel().tolist()
        top = [t for t in top if t and t.lower() not in ("nan", "none")]
        return " ".join(top)[:4000]
    except Exception:
        return ""
# =========================

# Uploads
# =========================
uploads = st.file_uploader(
    "Загрузите Excel/CSV файлы (можно несколько)",
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

st.subheader("Фильтр по студентам кафедры (опциональный)")
roster_file = st.file_uploader(
    "Загрузите список студентов кафедры (CSV/XLSX), если нужно фильтровать рейтинг",
    type=["csv", "xlsx"],
    accept_multiple_files=False
)

def _load_roster(file) -> set[str]:
    if not file:
        return set()
    try:
        if file.name.lower().endswith(".csv"):
            r = pd.read_csv(file)
        else:
            r = pd.read_excel(file)
    except Exception:
        return set()

    cols = list(r.columns)
    low = {c: norm_text(c) for c in cols}
    name_col = None
    id_col = None
    group_col = None

    for c in cols:
        if any(k in low[c] for k in ["фио", "студент", "фамил", "имя", "отчество"]):
            name_col = c
            break

    for c in cols:
        if any(k in low[c] for k in ["студенчес", "билет", "зач", "зачет", "номер", "табельн"]):
            id_col = c
            break

    for c in cols:
        if any(k in low[c] for k in ["группа", "group"]):
            group_col = c
            break

    keys = set()
    for _, row in r.iterrows():
        fio = str(row.get(name_col, "")).strip() if name_col else ""
        sid = str(row.get(id_col, "")).strip() if id_col else ""
        grp = str(row.get(group_col, "")).strip() if group_col else ""

        # Добавляем ID-ключ, если есть ID
        if sid and norm_text(sid) not in ("nan", "none", "0"):
            keys.add(f"id:{norm_text(sid)}")

        # добавляем Name-ключ, если есть ФИО
        if fio:
            keys.add(f"name:{norm_name(fio)}|grp:{norm_text(grp)}")

    return keys

roster_keys = _load_roster(roster_file)
if roster_file and not roster_keys:
    st.warning("Список кафедры загружен, но не удалось распознать ФИО/номер студенческого. Проверьте заголовки колонок.")
elif roster_keys:
    st.success(f"Список кафедры загружен: {len(roster_keys)} записей. В рейтинге будут только эти студенты.")


# Настройки распознавания заголовков
st.subheader("Настройки распознавания заголовков")
use_default_header = st.checkbox("По умолчанию считать заголовок: строка 1, высота 1", value=True)

# Параметры начисления
st.subheader("Параметры начисления")
c1, c2, c3 = st.columns(3)
with c1:
    acad_max = st.number_input("Успеваемость: максимум баллов", min_value=0.0, value=50.0)
with c2:
    att_max = st.number_input("Посещаемость: максимум баллов", min_value=0.0, value=20.0)
with c3:
    att_threshold = st.number_input("Порог посещаемости (0..1)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
params = {"acad_max": float(acad_max), "att_max": float(att_max), "att_threshold": float(att_threshold)}

# Баллы активностей
st.subheader("Баллы за активности (настраиваемо)")
activity_kinds = RULES.get(
    "activity_kinds",
    ["Акселератор", "CTF", "Публикация", "Олимпиада/конкурс", "Конференция", "Спорт", "Волонтёрство", "Прочее"]
)
default_points = RULES.get("activity_points_default", {})
roles = ["Участник", "Призёр", "Победитель", "Организатор"]

rows = []
for k in activity_kinds:
    row = {"Тип активности": k}
    for r in roles:
        row[r] = float(default_points.get(k, {}).get(r, default_points.get("Прочее", {}).get(r, 0.0)))
    rows.append(row)

points_df = pd.DataFrame(rows)
edited = st.data_editor(points_df, width="stretch", hide_index=True)

activity_points = {}
for _, rr in edited.iterrows():
    k = str(rr["Тип активности"])
    activity_points[k] = {role: float(rr[role]) for role in roles}


# Ручные достижения
with st.expander("Ручные достижения / доп. начисления", expanded=False):
    manual = load_manual_entries()

    with st.form("manual_add_form", clear_on_submit=True):
        m1, m2, m3 = st.columns(3)
        with m1:
            fio = st.text_input("ФИО", value="")
        with m2:
            grp = st.text_input("Группа", value="")
        with m3:
            sid = st.text_input("Номер студенческого билета/зачётки", value="")

        cat = st.selectbox("Куда начислять", ["Активность (баллы)", "Доп. баллы (сразу в итог)"])
        pts = st.number_input("Сколько добавить", value=1.0)
        desc = st.text_input("Описание", value="Ручной ввод")
        src = st.text_input("Источник/примечание", value="Ручной ввод")
        add_btn = st.form_submit_button("Добавить запись")

        if add_btn:
            if not fio.strip():
                st.error("ФИО обязательно")
            if not sid.strip():
                st.error("Номер студенческого билета/зачётки обязателен")
            if not grp.strip():
                st.error("Группа обязательна")
            else:
                manual.append({
                    "fio": fio.strip(),
                    "group": grp.strip(),
                    "id": sid.strip(),
                    "cat": cat,
                    "points": float(pts),
                    "desc": desc.strip() if desc.strip() else "Ручной ввод",
                    "source": src.strip() if src.strip() else "Ручной ввод"
                })
                save_manual_entries(manual)
                st.success("Добавлено.")
                st.rerun()

    if manual:
        st.markdown("### Удаление ручных записей")
        mdf = pd.DataFrame(manual)
        mdf.insert(0, "Удалить", False)
        edited_m = st.data_editor(mdf, width="stretch", hide_index=True)
        if st.button("Удалить отмеченные записи"):
            keep = []
            for _, r in edited_m.iterrows():
                if bool(r.get("Удалить", False)):
                    continue
                item = {k: r[k] for k in edited_m.columns if k != "Удалить"}
                keep.append(item)
            save_manual_entries(keep)
            st.success("Удалено.")
            st.rerun()


if not uploads:
    st.warning("Загрузите таблицы.")
    st.stop()

if "header_overrides" not in st.session_state:
    st.session_state["header_overrides"] = {}

tables = load_tables_from_uploads(uploads)
st.info(f"Найдено листов/таблиц: {len(tables)}")

prepared = []
bad_tables = []

for t in tables:
    df_raw = t["df_raw"]
    src_key = f'{t["source_name"]}::{t["sheet_name"]}'
    context = f'{t["source_name"]} {t["sheet_name"]} {_context_from_raw(df_raw)}'

    try:
        ov = st.session_state["header_overrides"].get(src_key)
        if ov:
            df, meta = build_dataframe_with_headers(df_raw, override_start=ov["start"], override_h=ov["h"], max_scan_rows=120)
        else:
            if use_default_header:
                df, meta = build_dataframe_with_headers(df_raw, override_start=0, override_h=1, max_scan_rows=120)
            else:
                df, meta = build_dataframe_with_headers(df_raw, max_scan_rows=120)

        schema = infer_schema(df, context_text=context)

        prepared.append({
            "source_name": t["source_name"],
            "sheet_name": t["sheet_name"],
            "src_key": src_key,
            "context": context,
            "df_raw": df_raw,
            "df": df,
            "meta": meta,
            "schema": schema
        })
    except Exception as e:
        bad_tables.append({"Файл": t["source_name"], "Лист": t["sheet_name"], "Ошибка": f"{type(e).__name__}: {e}"})

if bad_tables:
    st.error("Часть таблиц пропущена из-за ошибок (остальные обработаны):")
    st.dataframe(pd.DataFrame(bad_tables), width="stretch")


# Контроль распознавания
st.subheader("Контроль распознавания")
max_show = len(prepared)
default_show = min(8, max_show) if max_show else 0
show_n = st.number_input("Сколько таблиц показать (контроль)", min_value=0, max_value=max_show, value=default_show)

type_map = {"unknown": "Не определено", "attendance": "Посещаемость", "grades": "Оценки/успеваемость", "activity": "Активности"}
type_options = list(type_map.keys())

for idx, item in enumerate(prepared[: int(show_n)]):
    src = f'{item["source_name"]} / {item["sheet_name"]}'
    schema = item["schema"]
    df = item["df"]
    df_raw = item["df_raw"]
    meta = item["meta"]
    src_key = item["src_key"]
    kp = _safe_key_prefix(src_key)

    with st.expander(f"#{idx+1} {src} — тип: {type_map.get(schema.get('type_hint','unknown'),'Не определено')}", expanded=False):
        st.write(f"Заголовок: строка {meta['header_start_row']} (высота {meta['header_rows']})")

        new_start = st.number_input("Строка начала заголовка (Excel)", 1, len(df_raw), int(meta["header_start_row"]), key=f"{kp}__hs")
        new_h = st.number_input("Высота заголовка", 1, 50, int(meta["header_rows"]), key=f"{kp}__hh")
        if st.button("Сохранить заголовок для этого листа", key=f"{kp}__save_hdr"):
            st.session_state["header_overrides"][src_key] = {"start": int(new_start) - 1, "h": int(new_h)}
            st.success("Сохранено.")
            st.rerun()

        cols = [c for c in df.columns if c != "_origin_row"]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            schema["student_name"] = st.selectbox(
                "Колонка ФИО",
                [""] + cols,
                index=([""] + cols).index(schema.get("student_name") or "") if (schema.get("student_name") or "") in ([""] + cols) else 0,
                key=f"{kp}__name"
            )
        with c2:
            schema["student_id"] = st.selectbox(
                f"Колонка: {ID_COL}",
                [""] + cols,
                index=([""] + cols).index(schema.get("student_id") or "") if (schema.get("student_id") or "") in ([""] + cols) else 0,
                key=f"{kp}__id"
            )
        with c3:
            schema["group"] = st.selectbox(
                "Колонка Группа",
                [""] + cols,
                index=([""] + cols).index(schema.get("group") or "") if (schema.get("group") or "") in ([""] + cols) else 0,
                key=f"{kp}__group"
            )
        with c4:
            schema["type_hint"] = st.selectbox(
                "Тип таблицы",
                type_options,
                index=type_options.index(schema.get("type_hint", "unknown")) if schema.get("type_hint", "unknown") in type_options else 0,
                format_func=lambda x: type_map[x],
                key=f"{kp}__type"
            )

        if schema["type_hint"] == "attendance":
            schema["attendance_cols"] = st.multiselect(
                "Колонки посещаемости",
                cols,
                default=[c for c in schema.get("attendance_cols", []) if c in cols],
                key=f"{kp}__att_cols"
            )

        elif schema["type_hint"] == "grades":
            schema["grade_cols"] = st.multiselect(
                "Колонки оценок/баллов",
                cols,
                default=[c for c in schema.get("grade_cols", []) if c in cols],
                key=f"{kp}__grade_cols"
            )

        elif schema["type_hint"] == "activity":
            schema["activity_cols"] = st.multiselect(
                "Колонки достижений",
                cols,
                default=[c for c in schema.get("activity_cols", []) if c in cols],
                key=f"{kp}__act_cols"
            )

            schema.setdefault("activity_trusted", False)
            schema["activity_trusted"] = st.checkbox(
                "Доверенная таблица организаторов (можно без ссылок и строгого формата)",
                value=bool(schema.get("activity_trusted", False)),
                key=f"{kp}__trusted",
                help="Включайте для таблиц преподавателей/организаторов (список участников/докладов). "
                     "Тогда система не требует ссылку-доказательство и не требует формат 'ТИП - РОЛЬ - ...'."
            )

            st.caption("Активности: можно перечислять несколько мероприятий в одной ячейке (через Enter/;). Тип/роль определяются по каждому событию отдельно.")
            kind_options = ["Авто"] + activity_kinds
            current_kind = schema.get("activity_kind") or "Авто"
            if current_kind not in kind_options:
                current_kind = "Авто"

            schema["activity_kind"] = st.selectbox(
                "Тип активности (обычно оставьте 'Авто')",
                kind_options,
                index=kind_options.index(current_kind),
                key=f"{kp}__act_kind"
            )

        st.dataframe(df.head(8), width="stretch")

        if st.button("Сохранить профиль распознавания", key=f"{kp}__save_prof"):
            persist_profile_for_schema(schema)
            st.success("Сохранено.")
            st.rerun()


# Кэш результата
for k in [
    "result_ready",
    "ranking_df",
    "summary_df",
    "detail_df",
    "group_df",
    "evidence_rows",
    "quality_df",
    "duplicates_df",
]:
    st.session_state.setdefault(k, None)
st.session_state.setdefault("result_ready", False)

st.divider()
st.subheader("Формирование результата")

cq1, cq2 = st.columns(2)
with cq1:
    dedupe_activities = st.checkbox("Удалять дубли активностей (между полями/файлами)", value=True)
with cq2:
    show_quality = st.checkbox("Сформировать отчёт о качестве данных (формат/ссылки/автофикс)", value=True)

if st.button("Сформировать итоговый отчёт (рейтинг + начисления)", type="primary"):
    evidence_rows: list[dict] = []
    bad_extract: list[dict] = []

    for item in prepared:
        df = item["df"]
        schema = item["schema"]
        src = f'{item["source_name"]} / {item["sheet_name"]}'
        context = item["context"]

        try:
            # ключевая защита: приводим schema к df чтобы не падать при скачке виджетов
            schema = _sanitize_schema_for_df(schema, df, context)

            ttype = schema.get("type_hint", "unknown")

            if ttype == "attendance":
                _, att_ev = extract_attendance_wide(df, schema, src)
                evidence_rows.extend(att_ev)

            elif ttype == "grades":
                _, grd_ev = extract_grades_wide(df, schema, src)
                evidence_rows.extend(grd_ev)

            elif ttype == "activity":
                _, act_ev = extract_activities_text(
                    df,
                    schema,
                    src,
                    activity_points,
                    context_text=context,
                    strict_validation=not bool(schema.get("activity_trusted", False)),
                )
                evidence_rows.extend(act_ev)

            else:
                pass

        except Exception as e:
            bad_extract.append({"Файл": item["source_name"], "Лист": item["sheet_name"], "Ошибка": f"{type(e).__name__}: {e}"})

    # ручные записи
    manual = load_manual_entries()
    for m in manual:
        fio = str(m.get("fio", "")).strip()
        grp = str(m.get("group", "")).strip()
        sid = str(m.get("id", "")).strip()
        desc = str(m.get("desc", "Ручной ввод")).strip()
        src = str(m.get("source", "Ручной ввод")).strip()
        pts = float(m.get("points", 0.0))
        if not fio:
            continue
        category = "C" if m.get("cat") == "Активность (баллы)" else "X"
        evidence_rows.append({
            "student_key": make_student_key(sid, fio, grp),
            "student_name": fio,
            "student_id": sid,
            "group": grp,
            "category": category,
            "points": pts,
            "source": "Ручной ввод",
            "detail": desc + (f" (источник: {src})" if src else ""),
            "origin_row": ""
        })

    if roster_keys:
        evidence_rows = [r for r in evidence_rows if r.get("student_key") in roster_keys]

    _normalize_evidence_for_dedupe(evidence_rows)

    quality_rows: list[dict] = _build_quality_rows_from_evidence(evidence_rows) if show_quality else []
    quality_df = pd.DataFrame(quality_rows) if quality_rows else pd.DataFrame()

    duplicates_rows: list[dict] = []
    if dedupe_activities:
        evidence_rows, duplicates_rows = dedupe_evidence_rows(evidence_rows, categories={"C"})
    duplicates_df = pd.DataFrame(duplicates_rows) if duplicates_rows else pd.DataFrame()

    st.session_state["quality_df"] = quality_df
    st.session_state["duplicates_df"] = duplicates_df

    if bad_extract:
        st.warning("Часть листов не удалось извлечь (они пропущены):")
        st.dataframe(pd.DataFrame(bad_extract), width="stretch")

    ranking_df, summary_df, detail_df, group_df = compute_scores(evidence_rows, params)
    st.session_state["result_ready"] = True
    st.session_state["ranking_df"] = ranking_df
    st.session_state["summary_df"] = summary_df
    st.session_state["detail_df"] = detail_df
    st.session_state["group_df"] = group_df
    st.session_state["evidence_rows"] = evidence_rows
    st.success("Результат сформирован.")


if st.session_state.get("result_ready"):
    ranking_df = st.session_state["ranking_df"]
    summary_df = st.session_state["summary_df"]
    detail_df = st.session_state["detail_df"]
    group_df = st.session_state["group_df"]

    quality_df = st.session_state.get("quality_df")
    if quality_df is None:
        quality_df = pd.DataFrame()

    duplicates_df = st.session_state.get("duplicates_df")
    if duplicates_df is None:
        duplicates_df = pd.DataFrame()

    if ranking_df is None or ranking_df.empty:
        st.error("Результат пустой.")
    else:
        with st.expander("Проверка качества данных", expanded=False):
            n_issues = 0 if quality_df.empty else len(quality_df)
            n_dups = 0 if duplicates_df.empty else len(duplicates_df)

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.metric("Проблем/замечаний", n_issues)
            with cc2:
                st.metric("Удалённых дублей активностей", n_dups)
            with cc3:
                if not quality_df.empty and "Уровень" in quality_df.columns:
                    err_cnt = int((quality_df["Уровень"] == "error").sum())
                    st.metric("Ошибок (error)", err_cnt)
                else:
                    st.metric("Ошибок (error)", 0)

            if not quality_df.empty:
                sev_order = ["error", "warn", "info"]
                sev = st.multiselect("Фильтр по уровню", sev_order, default=sev_order)
                view_q = quality_df.copy()
                view_q = view_q[view_q["Уровень"].astype(str).isin(sev)]
                st.dataframe(view_q.head(1000), width="stretch")
            else:
                st.success("Проблемы не обнаружены (или отчёт качества отключён).")

            if not duplicates_df.empty:
                st.write("Дубликаты (удалённые записи):")
                st.dataframe(duplicates_df.head(1000), width="stretch")

        st.subheader("Рейтинг")
        f1, f2 = st.columns(2)
        with f1:
            q = st.text_input("Поиск по ФИО", value="")
        with f2:
            groups = ["(все)"] + sorted([g for g in summary_df["Группа"].astype(str).unique().tolist() if g and g != "nan"])
            gsel = st.selectbox("Фильтр по группе", groups, index=0)

        view = ranking_df.copy()
        if q.strip():
            view = view[view["ФИО"].astype(str).str.contains(q.strip(), case=False, na=False)]
        if gsel != "(все)":
            view = view[view["Группа"].astype(str) == gsel]
        st.dataframe(view.head(500), width="stretch")

        st.subheader("Свод начислений")
        sv = summary_df.copy()
        if q.strip():
            sv = sv[sv["ФИО"].astype(str).str.contains(q.strip(), case=False, na=False)]
        if gsel != "(все)":
            sv = sv[sv["Группа"].astype(str) == gsel]
        st.dataframe(sv.head(500), width="stretch")

        st.subheader("Детализация начислений (фрагмент)")
        dv = detail_df.copy()
        if q.strip():
            dv = dv[dv["ФИО"].astype(str).str.contains(q.strip(), case=False, na=False)]
        if gsel != "(все)":
            dv = dv[dv["Группа"].astype(str) == gsel]
        st.dataframe(dv.head(800), width="stretch")

        st.subheader("Свод по группам")
        st.dataframe(group_df, width="stretch")

        xbytes = export_to_excel_bytes(
            ranking_df,
            summary_df,
            detail_df,
            group_df,
            quality_df=quality_df,
            duplicates_df=duplicates_df,
        )
        st.download_button(
            "Скачать Excel-отчёт",
            data=xbytes,
            file_name="Рейтинг_студентов.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )