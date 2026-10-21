"""Даты и время в аргументах инструментов — один разбор для всех: день («today», «friday», «2026-10-20»,
«15 января», «15.01», «in 3 days») и момент («18:00», «tomorrow 9:00», «friday 18:00», «2026-10-20 18:00»).
И обратно: даты из ответов инструментов словами (with_words) — день недели модель не считает, а переписывает."""
import datetime as dt, re

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
WD_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
ISO = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}( [0-9]{2}:[0-9]{2})?")
MOMENT_KEYS = ("fire_at", "class_at", "due", "at")  # момент: «в пт 23 октября в 09:00», «завтра», «сегодня в 10:25»


def fmt(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M")


def parse_day(now: dt.datetime, s, future=False) -> dt.date:
    """День. Название дня недели — ближайший такой день, сегодня тоже; future=True: «к пятнице», сказанное
    в пятницу, — это следующая пятница. Число без года — ближайшее будущее такое число. Иначе ValueError."""
    today = now.date()
    s = str("today" if s is None else s).strip().lower()
    if s == "today":
        return today
    if s == "tomorrow":
        return today + dt.timedelta(days=1)
    if s in WEEKDAYS:
        d = today + dt.timedelta(days=(WEEKDAYS.index(s) - today.weekday()) % 7)
        return d + dt.timedelta(days=7) if future and d == today else d
    m = re.fullmatch(r"in (\d{1,3}) days?", s)
    if m:
        return today + dt.timedelta(days=int(m.group(1)))
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})(?:\.(\d{2}|\d{4}))?", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
    else:
        m = re.fullmatch(r"(\d{1,2})\s+([а-яё]+)(?:\s+(\d{4}))?(?:\s*г\.?)?", s)
        if not m:
            return dt.date.fromisoformat(s)
        day, year = int(m.group(1)), m.group(3)
        month = next((i for i, x in enumerate(MONTHS_GEN, 1) if x.startswith(m.group(2)[:3])), None)
        if month is None:
            raise ValueError(s)
    if year:
        return dt.date(int(year) + (2000 if len(year) == 2 else 0), month, day)
    d = dt.date(now.year, month, day)
    return d if d >= today else d.replace(year=d.year + 1)


def parse_at(now: dt.datetime, s):
    """Момент: (datetime, указано ли время). Без дня — ближайшее такое время (сегодня или завтра); день недели
    со временем, которое уже прошло, — через неделю. Без времени — конец дня. Иначе ValueError."""
    parts = str(s).strip().lower().split()
    t = None
    if parts and re.fullmatch(r"\d{1,2}:\d{2}", parts[-1]):
        h, m = map(int, parts.pop().split(":"))
        t = dt.time(h, m)
    if not parts:
        if t is None:
            raise ValueError(s)
        cand = dt.datetime.combine(now.date(), t)
        return (cand if cand > now else cand + dt.timedelta(days=1)), True
    day = " ".join(parts)
    cand = dt.datetime.combine(parse_day(now, day), t or dt.time(23, 59))
    if day in WEEKDAYS and cand <= now:
        cand += dt.timedelta(days=7)
    return cand, t is not None


def remind_time(now, in_minutes=None, in_hours=None, in_days=None, at=None):
    """Когда напомнить: через сколько (in_*) или в какой момент (at, со временем). -> (datetime, None) или (None, ошибка)."""
    if in_minutes or in_hours or in_days:
        try:
            fire = now + dt.timedelta(minutes=in_minutes or 0, hours=in_hours or 0, days=in_days or 0)
        except OverflowError:
            return None, "bad_time"
    elif at:
        try:
            fire, timed = parse_at(now, at)
        except (ValueError, OverflowError):
            return None, "bad_time"
        if not timed:
            return None, "bad_time"  # «в пятницу» без часа: модель переспросит время
    else:
        return None, None
    return (fire, None) if fire > now else (None, "time_in_past")


def date_words(d: dt.date) -> str:
    """2026-10-23 -> '23 октября'."""
    return f"{d.day} {MONTHS_GEN[d.month - 1]}"


def when_words(now: dt.datetime, s) -> str:
    """'2026-10-23 09:00' или '2026-10-23' -> 'сегодня в 09:00' / 'завтра' / 'в пт 23 октября в 09:00'."""
    t = dt.datetime.fromisoformat(s if " " in s else s + " 00:00")
    delta = (t.date() - now.date()).days
    day = {0: "сегодня", 1: "завтра"}.get(delta, f"в {WD_SHORT[t.weekday()]} {date_words(t.date())}")
    return f"{day} в {t:%H:%M}" if " " in s else day


def with_words(obj, now):
    """Копия ответа инструмента, где рядом с каждой датой — она же словами: fire_at, class_at, due, at ->
    <ключ>_text ('в пт 23 октября в 09:00'), date -> date_text ('пт 23 октября'), date с time -> ещё when_text
    ('в пт 23 октября в 18:00', 'завтра в 18:00'), start, end -> '11 января'. Старые *_text пересчитываются от now:
    так после полуночи «завтра» в истории разговора становится «сегодня»."""
    if isinstance(obj, list):
        return [with_words(x, now) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if isinstance(k, str) and k.endswith("_text") and (k[:-5] in obj or k == "when_text" and "date" in obj):
            continue  # пересчитаем ниже, рядом с исходным ключом
        out[k] = with_words(v, now)
        try:
            words = _words(k, v, now)
            t = obj.get("time") if k == "date" and words else None
            when = when_words(now, f"{v} {t}") if isinstance(t, str) and re.fullmatch(r"[0-9]{2}:[0-9]{2}", t) else None
        except ValueError:  # «2026-11-31», «25:00»: такого нет — оставляем как есть
            continue
        if words:
            out[k + "_text"] = words
        if when:
            out["when_text"] = when
    if isinstance(out.get("days_left"), int) and "date_text" in out and isinstance(out.get("date"), str):
        out["days_left"] = (dt.date.fromisoformat(out["date"]) - now.date()).days  # мероприятие, экзамен: после полуночи на день меньше
    if out.get("status") in ("soon", "now", "over") and "start_text" in out and "end_text" in out:  # сессия (get_session)
        start, end, today = dt.date.fromisoformat(out["start"]), dt.date.fromisoformat(out["end"]), now.date()
        out["status"] = "over" if today > end else "now" if today >= start else "soon"
        if out["status"] == "soon":
            out["days_left"] = (start - today).days
        else:
            out.pop("days_left", None)
    return out


def _words(k, v, now):
    if not (isinstance(v, str) and ISO.fullmatch(v)):
        return None
    if k in MOMENT_KEYS:
        return when_words(now, v)
    if " " in v:
        return None
    d = dt.date.fromisoformat(v)
    if k == "date":
        return f"{WD_SHORT[d.weekday()]} {date_words(d)}"
    return date_words(d) if k in ("start", "end") else None
