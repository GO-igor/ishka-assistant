"""Даты и время в аргументах инструментов — один разбор для всех: день («today», «friday», «2026-10-20»,
«15 января», «15.01», «in 3 days») и момент («18:00», «tomorrow 9:00», «friday 18:00», «2026-10-20 18:00»)."""
import datetime as dt, re

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]


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
