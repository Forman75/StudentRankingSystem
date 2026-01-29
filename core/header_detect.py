from __future__ import annotations
import re
from typing import Tuple
import pandas as pd
from .utils import try_parse_date, norm_text

DATE_LIKE_RE = re.compile(
    r"^\s*\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\s*$|^\s*\d{4}[./-]\d{1,2}[./-]\d{1,2}\s*$"
)

EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+", re.I)
NUMERIC_RE = re.compile(r"^\s*[-+]?\d+([.,]\d+)?\s*$")

# Ключевые слова, характерные для шапок таблиц (включая Forms)
HEADER_KWS = [
    "отметка времени", "timestamp",
    "адрес электронной почты", "электронной почты", "email", "e-mail", "почта",
    "фио", "фамилия", "имя", "отчество", "student name", "full name",
    "группа", "учебная группа", "group",
    "зачетка", "зачётка", "номер зачетки", "номер зачётки", "student id", "id",
    "мероприят", "активност", "достижен", "публикац", "конференц", "хакатон", "олимпиад", "ctf",
    "оценк", "балл", "итог", "успеваем", "grade", "score",
    "посещаем", "пропуск", "attendance",
    "выберите", "выбор", "вариант", "ответ", "question",
]

FORM_NOISE_KWS = [
    "short answer", "long answer", "краткий ответ", "развернутый ответ",
    "multiple choice", "checkboxes", "dropdown",
]

def _cell_text(v) -> str:
    if v is None:
        return ""
    s = str(v)
    if not s or s.lower() == "nan":
        return ""
    return s


def _row_date_score(row: pd.Series) -> int:
    score = 0
    for v in row.tolist():
        if v is None:
            continue
        if try_parse_date(v) is not None:
            score += 2
        else:
            txt = norm_text(v)
            if DATE_LIKE_RE.match(txt):
                score += 2
    return score


def _row_headerish_score(row: pd.Series) -> float:
    # похоже на заголовок: есть непустые значения, они не слишком длинные, и не все числовые
    as_str = row.astype(str)
    nonnull = row.notna().sum()
    lens = as_str.str.len().fillna(0)
    shortish = lens.between(2, 80).sum()
    not_numeric = (~as_str.str.match(NUMERIC_RE, na=False)).sum()
    return float(nonnull) + 0.7 * float(shortish) + 0.2 * float(not_numeric)

def _row_keyword_score(row: pd.Series) -> float:
    # Сколько “заголовочных” ключевых слов встречается в строке
    score = 0.0
    for v in row.tolist():
        s = norm_text(_cell_text(v))
        if not s:
            continue
        # вопросительные/формат форм - чаще заголовок
        if "?" in s:
            score += 0.6

        # шумовые токены формы - это обычно 2-я строка шапки
        if any(k in s for k in FORM_NOISE_KWS):
            score += 0.4

        # ключевые слова заголовка
        for k in HEADER_KWS:
            if k in s:
                score += 1.0
                break
    return score


def _row_dataish_score(row: pd.Series) -> float:
    # Похоже на данные: много непустых значений, в них меньше ключевых слов заголовка, но встречаются email/цифры/ФИО-подобные строки
    nonempty = 0
    email_hits = 0
    numeric_hits = 0
    name_like = 0
    kw_hits = 0

    for v in row.tolist():
        raw = _cell_text(v)
        s = raw.strip() if isinstance(raw, str) else str(raw).strip()
        t = norm_text(s)

        if not t or t in ("nan", "none"):
            continue

        nonempty += 1

        if EMAIL_RE.search(s):
            email_hits += 1

        if NUMERIC_RE.match(s):
            numeric_hits += 1

        # ФИО-подобно: 2+ слов, букв много, цифр нет
        if re.search(r"[a-zа-я]", s, re.I) and not re.search(r"\d", s):
            parts = [p for p in re.split(r"\s+", s.strip()) if p]
            if len(parts) >= 2 and len(s) >= 8:
                name_like += 1

        for k in HEADER_KWS:
            if k in t:
                kw_hits += 1
                break

    if nonempty == 0:
        return 0.0

    # данные: много непустых + признаки пользовательских значений - признаки шапки
    return float(nonempty) + 0.7 * float(email_hits + numeric_hits + name_like) - 0.8 * float(kw_hits)

def detect_header_block(df_raw: pd.DataFrame, max_scan_rows: int = 80) -> Tuple[int, int]:
    """
    Возвращает (start_row_0based, header_height)
    Идея:
      - Заголовок обычно содержит много ключевых слов (ФИО/группа/мероприятия/время/почта)
      - Строки ниже должны быть более “data-like”
      - Сильно штрафуем поздний start
    """
    n = min(max_scan_rows, len(df_raw))
    best = (0, 1)
    best_score = -1e18

    for start in range(0, n):
        max_h = min(12, n - start)
        for h in range(1, max_h + 1):
            end = start + h

            block = df_raw.iloc[start:end, 1:]  # без _origin_row
            if block.empty:
                continue

            # фильтр: если в блоке совсем мало непустых - это не заголовок
            nonempty_cells = 0
            for i in range(len(block)):
                row = block.iloc[i]
                for v in row.tolist():
                    if norm_text(_cell_text(v)) not in ("", "nan", "none"):
                        nonempty_cells += 1
            if nonempty_cells < 3:
                continue

            date_score = sum(_row_date_score(block.iloc[i]) for i in range(len(block)))
            header_score = sum(_row_headerish_score(block.iloc[i]) for i in range(len(block)))
            kw_score = sum(_row_keyword_score(block.iloc[i]) for i in range(len(block)))

            after = df_raw.iloc[end:min(end + 3, n), 1:]
            after_headerish = sum(_row_headerish_score(after.iloc[i]) for i in range(len(after))) if len(after) else 0.0
            after_kw = sum(_row_keyword_score(after.iloc[i]) for i in range(len(after))) if len(after) else 0.0
            after_dataish = sum(_row_dataish_score(after.iloc[i]) for i in range(len(after))) if len(after) else 0.0

            # Основной скоринг
            score = (
                2.0 * date_score +
                1.1 * header_score +
                2.2 * kw_score
                - 0.55 * after_headerish
                - 0.80 * after_kw
                + 0.85 * after_dataish
            )

            # Приоритет ранних строк и не слишком высоких шапок
            score -= 0.65 * start
            score -= 0.08 * max(0, h - 2)

            if score > best_score:
                best_score = score
                best = (start, h)

    return best
