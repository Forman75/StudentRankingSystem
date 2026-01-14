from __future__ import annotations
import streamlit as st
import pandas as pd
from core.ingest import load_tables_from_uploads
from core.infer import build_dataframe_with_headers, infer_schema, persist_profile_for_schema
from core.extract import (extract_attendance_wide, extract_grades_wide, extract_activities_text, make_student_key)
from core.scoring import compute_scores
from core.export import export_to_excel_bytes
from core.manual import load_manual_entries, save_manual_entries
from core.utils import rules_path, load_json, norm_text, norm_name

RULES = load_json(rules_path(), {})

st.set_page_config(page_title="Рейтинг студентов", layout="wide")
st.title("Система анализа таблиц и формирования рейтинга студентов")

uploads = st.file_uploader(
    "Загрузите Excel/CSV файлы (можно несколько)",
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

# Опциональный список студентов кафедры
# ============================================================
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
            # по умолчанию первый лист
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
        if any(k in low[c] for k in ["id", "зач", "зачет", "номер", "табельн"]):
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

        if sid and norm_text(sid) not in ("nan", "none", "0"):
            keys.add(f"id:{norm_text(sid)}")
        elif fio:
            # ключ как в make_student_key при отсутствии id
            keys.add(f"name:{norm_name(fio)}|grp:{norm_text(grp)}")

    return keys

roster_keys = _load_roster(roster_file)

if roster_file and not roster_keys:
    st.warning("Список кафедры загружен, но не удалось распознать ФИО/ID. Проверьте заголовки колонок.")
elif roster_keys:
    st.success(f"Список кафедры загружен: {len(roster_keys)} записей. В рейтинге будут только эти студенты.")
# ============================================================

# Настройки распознавания заголовков
# ============================================================
st.subheader("Настройки распознавания заголовков")
use_default_header = st.checkbox("По умолчанию считать заголовок: строка 1, высота 1", value=True)
# ============================================================

# Параметры начисления
# ============================================================
st.subheader("Параметры начисления")
c1, c2, c3 = st.columns(3)
with c1:
    acad_max = st.number_input("Успеваемость: максимум баллов", min_value=0.0, value=50.0)
with c2:
    att_max = st.number_input("Посещаемость: максимум баллов", min_value=0.0, value=20.0)
with c3:
    att_threshold = st.number_input("Порог посещаемости (0..1)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
params = {"acad_max": float(acad_max), "att_max": float(att_max), "att_threshold": float(att_threshold)}
# ============================================================

# Баллы активностей (настраиваемо)
# ============================================================
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
edited = st.data_editor(points_df, use_container_width=True, hide_index=True)

activity_points = {}
for _, rr in edited.iterrows():
    k = str(rr["Тип активности"])
    activity_points[k] = {role: float(rr[role]) for role in roles}
# ============================================================

# Ручные достижения / доп. начисления
# ============================================================
with st.expander("Ручные достижения / доп. начисления", expanded=False):
    manual = load_manual_entries()

    with st.form("manual_add_form", clear_on_submit=True):
        m1, m2, m3 = st.columns(3)
        with m1:
            fio = st.text_input("ФИО", value="")
        with m2:
            grp = st.text_input("Группа", value="")
        with m3:
            sid = st.text_input("ID/зачётка", value="")

        cat = st.selectbox("Куда начислять", ["Активность (баллы)", "Доп. баллы (сразу в итог)"])
        pts = st.number_input("Сколько добавить", value=1.0)
        desc = st.text_input("Описание", value="Ручной ввод")
        src = st.text_input("Источник/примечание", value="Ручной ввод")
        add_btn = st.form_submit_button("Добавить запись")

        if add_btn:
            if not fio.strip():
                st.error("ФИО обязательно")
            if not sid.strip():
                st.error("ID/зачетка обязательно")
            if not grp.strip():
                st.error("Группа обязательно")
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
        edited_m = st.data_editor(mdf, use_container_width=True, hide_index=True)
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
# ============================================================

# Если нет файлов — стоп
# ============================================================
if not uploads:
    st.warning("Загрузите таблицы.")
    st.stop()
# ============================================================

# Header overrides
# ============================================================
if "header_overrides" not in st.session_state:
    st.session_state["header_overrides"] = {}

tables = load_tables_from_uploads(uploads)
st.info(f"Найдено листов/таблиц: {len(tables)}")

prepared = []
bad_tables = []

def _context_from_raw(df_raw: pd.DataFrame, max_rows: int = 4, max_cols: int = 10) -> str:
    try:
        top = df_raw.iloc[:max_rows, :max_cols].astype(str).values.ravel().tolist()
        top = [t for t in top if t and t.lower() not in ("nan", "none")]
        return " ".join(top)[:4000]
    except Exception:
        return ""

for t in tables:
    df_raw = t["df_raw"]
    src_key = f'{t["source_name"]}::{t["sheet_name"]}'
    ctx_extra = _context_from_raw(df_raw)
    context = f'{t["source_name"]} {t["sheet_name"]} {ctx_extra}'

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
    st.dataframe(pd.DataFrame(bad_tables), use_container_width=True)
# ============================================================

# Контроль распознавания
# ============================================================
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

    with st.expander(f"#{idx+1} {src} — тип: {type_map.get(schema.get('type_hint','unknown'),'Не определено')}", expanded=False):
        st.write(f"Заголовок: строка {meta['header_start_row']} (высота {meta['header_rows']})")

        new_start = st.number_input("Строка начала заголовка (Excel)", 1, len(df_raw), int(meta["header_start_row"]), key=f"hs_{idx}")
        new_h = st.number_input("Высота заголовка", 1, 50, int(meta["header_rows"]), key=f"hh_{idx}")
        if st.button("Сохранить заголовок для этого листа", key=f"save_hdr_{idx}"):
            st.session_state["header_overrides"][src_key] = {"start": int(new_start) - 1, "h": int(new_h)}
            st.success("Сохранено.")
            st.rerun()

        cols = [c for c in df.columns if c != "_origin_row"]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            schema["student_name"] = st.selectbox(
                "Колонка ФИО",
                [""] + cols,
                index=([""]+cols).index(schema.get("student_name") or "") if (schema.get("student_name") or "") in ([""]+cols) else 0,
                key=f"n_{idx}"
            )
        with c2:
            schema["student_id"] = st.selectbox(
                "Колонка ID",
                [""] + cols,
                index=([""]+cols).index(schema.get("student_id") or "") if (schema.get("student_id") or "") in ([""]+cols) else 0,
                key=f"i_{idx}"
            )
        with c3:
            schema["group"] = st.selectbox(
                "Колонка Группа",
                [""] + cols,
                index=([""]+cols).index(schema.get("group") or "") if (schema.get("group") or "") in ([""]+cols) else 0,
                key=f"g_{idx}"
            )
        with c4:
            schema["type_hint"] = st.selectbox(
                "Тип таблицы",
                type_options,
                index=type_options.index(schema.get("type_hint","unknown")) if schema.get("type_hint","unknown") in type_options else 0,
                format_func=lambda x: type_map[x],
                key=f"t_{idx}"
            )

        if schema["type_hint"] == "attendance":
            schema["attendance_cols"] = st.multiselect("Колонки посещаемости", cols, default=[c for c in schema.get("attendance_cols", []) if c in cols], key=f"a_{idx}")
        elif schema["type_hint"] == "grades":
            schema["grade_cols"] = st.multiselect("Колонки оценок/баллов", cols, default=[c for c in schema.get("grade_cols", []) if c in cols], key=f"gr_{idx}")
        elif schema["type_hint"] == "activity":
            schema["activity_cols"] = st.multiselect("Колонки достижений", cols, default=[c for c in schema.get("activity_cols", []) if c in cols], key=f"ac_{idx}")
            st.info(f"Авто-тип активности: {schema.get('activity_kind_auto','Прочее')}")
            schema["activity_kind"] = st.selectbox(
                "Тип активности (если надо)",
                activity_kinds,
                index=activity_kinds.index(schema.get("activity_kind_auto","Прочее")) if schema.get("activity_kind_auto","Прочее") in activity_kinds else 0,
                key=f"ak_{idx}"
            )

        st.dataframe(df.head(8), use_container_width=True)

        if st.button("Сохранить профиль распознавания", key=f"prof_{idx}"):
            persist_profile_for_schema(schema)
            st.success("Сохранено.")
            st.rerun()
# ============================================================

# Кэш результата
# ============================================================
for k in ["result_ready", "ranking_df", "summary_df", "detail_df", "group_df", "evidence_rows"]:
    st.session_state.setdefault(k, None)
st.session_state.setdefault("result_ready", False)
st.divider()
st.subheader("Формирование результата")

if st.button("Сформировать итоговый отчёт (рейтинг + начисления)", type="primary"):
    evidence_rows = []
    bad_extract = []

    for item in prepared:
        df = item["df"]
        schema = item["schema"]
        src = f'{item["source_name"]} / {item["sheet_name"]}'
        context = item["context"]
        ttype = schema.get("type_hint", "unknown")

        try:
            # извлекаем только по типу
            if ttype == "attendance":
                _, att_ev = extract_attendance_wide(df, schema, src)
                evidence_rows.extend(att_ev)
            elif ttype == "grades":
                _, grd_ev = extract_grades_wide(df, schema, src)
                evidence_rows.extend(grd_ev)
            elif ttype == "activity":
                _, act_ev = extract_activities_text(df, schema, src, activity_points, context_text=context)
                evidence_rows.extend(act_ev)
            else:
                pass

        except Exception as e:
            bad_extract.append({"Файл": item["source_name"], "Лист": item["sheet_name"], "Ошибка": f"{type(e).__name__}: {e}"})

    # ручные записи
    manual = load_manual_entries()
    for m in manual:
        fio = str(m.get("fio","")).strip()
        grp = str(m.get("group","")).strip()
        sid = str(m.get("id","")).strip()
        desc = str(m.get("desc","Ручной ввод")).strip()
        src = str(m.get("source","Ручной ввод")).strip()
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

    # фильтрация по списку кафедры (если загружен)
    if roster_keys:
        evidence_rows = [r for r in evidence_rows if r.get("student_key") in roster_keys]

    if bad_extract:
        st.warning("Часть листов не удалось извлечь (они пропущены):")
        st.dataframe(pd.DataFrame(bad_extract), use_container_width=True)

    ranking_df, summary_df, detail_df, group_df = compute_scores(evidence_rows, params)
    st.session_state["result_ready"] = True
    st.session_state["ranking_df"] = ranking_df
    st.session_state["summary_df"] = summary_df
    st.session_state["detail_df"] = detail_df
    st.session_state["group_df"] = group_df
    st.session_state["evidence_rows"] = evidence_rows
    st.success("Результат сформирован.")
# ============================================================

# Показ результата
# ============================================================
if st.session_state.get("result_ready"):
    ranking_df = st.session_state["ranking_df"]
    summary_df = st.session_state["summary_df"]
    detail_df = st.session_state["detail_df"]
    group_df = st.session_state["group_df"]

    if ranking_df is None or ranking_df.empty:
        st.error("Результат пустой.")
    else:
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
        st.dataframe(view.head(500), use_container_width=True)

        st.subheader("Свод начислений")
        sv = summary_df.copy()
        if q.strip():
            sv = sv[sv["ФИО"].astype(str).str.contains(q.strip(), case=False, na=False)]
        if gsel != "(все)":
            sv = sv[sv["Группа"].astype(str) == gsel]
        st.dataframe(sv.head(500), use_container_width=True)

        st.subheader("Детализация начислений (фрагмент)")
        dv = detail_df.copy()
        if q.strip():
            dv = dv[dv["ФИО"].astype(str).str.contains(q.strip(), case=False, na=False)]
        if gsel != "(все)":
            dv = dv[dv["Группа"].astype(str) == gsel]
        st.dataframe(dv.head(800), use_container_width=True)
        st.subheader("Свод по группам")
        st.dataframe(group_df, use_container_width=True)
        xbytes = export_to_excel_bytes(ranking_df, summary_df, detail_df, group_df)
        st.download_button(
            "Скачать Excel-отчёт",
            data=xbytes,
            file_name="Рейтинг_студентов.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )