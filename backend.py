"""Эталонная реализация инструментов поверх db/*.json.
Нужна генератору датасета (ответы инструментов — настоящие) и как основа будущего бэкенда.
Изменяемые данные (Д/З, заметки, напоминания, дедлайны, экзамены) — в памяти; если студент вошёл в аккаунт
(store + account_id), они сохраняются в db/ishka.sqlite (store.py). Там же рассылки старосты группе."""
import datetime as dt, json, pathlib, re, unicodedata
from collections import Counter
from guard import Guard

DB = pathlib.Path(__file__).parent / "db"
load = lambda n: json.loads((DB / n).read_text(encoding="utf-8"))
LESSONS, SUBJECTS, BUILDINGS = load("lessons.json"), load("subjects.json"), load("buildings.json")
ROUTES, WALK, FOOD, CAL = load("routes.json"), load("walk.json"), load("food.json"), load("calendar.json")
EVENTS = load("events.json")["events"]  # мероприятия и хакатоны; файл заполняется руками
ADAPT = {"rus_a", "math_a", "eng_a"}
GUARD = Guard()  # запретные фразы: такие мероприятия и рассылки старосты модели не показываем
HOLIDAYS = set(CAL.get("holidays", []))
ENG_TEACHERS = sorted({l["teacher"] for l in LESSONS if l["subject"] == "eng" and l["teacher"]})
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WD_RU = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
EAT_MINUTES = 15  # сколько нужно на сам перекус
EVENT_KINDS = ["хакатон", "конференция", "олимпиада", "конкурс", "мастер-класс"]
EVENT_PERIODS = {"week": 7, "month": 31, "all": None}  # сколько дней вперёд смотреть
PERSIST = ("homework", "notes", "reminders", "deadlines", "exams", "brief", "hidden")  # что хранится в user_data
LIMITS = {"note": 10, "reminder": 10}  # сколько своих заметок и активных напоминаний можно держать
GROUP_LIMITS = {"homework": 10, "event": 10, "reminder": 10}  # сколько действующих рассылок старосты каждого вида
DELETE_KINDS = ("note", "reminder", "homework", "deadline", "exam", "event")
POST_KINDS = ("homework", "event", "reminder")
DELETE_TOOLS = {"delete_items", "group_delete_post"}
TEXT_KEYS = ("text", "subject", "task", "title", "place", "kind")  # где искать слова из query при удалении
POST_SHOW = {"homework": ("subject", "task", "due"), "event": ("title", "date", "time", "place"), "reminder": ("text", "fire_at")}
OWN_KEYS = {"reminder": ("text", "fire_at"), "homework": ("subject", "task", "due"), "deadline": ("title", "subject", "due"),
            "exam": ("subject", "kind")}  # по этим полям своя запись узнаётся после удаления соседних (экзамен — и после переноса)
STOP = {"по", "в", "во", "на", "с", "со", "к", "ко", "и", "о", "об", "про", "для", "от", "до", "за", "из", "у", "мою",
        "мой", "мое", "моё", "мои", "мне", "все", "всё"}  # предлоги и местоимения в query не мешают поиску
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]


def week_parity(d: dt.date) -> str:
    anchor = dt.date.fromisoformat(CAL["odd_week_monday"])
    return "odd" if ((d - anchor).days // 7) % 2 == 0 else "even"


def resolve_subject(text: str):
    t = text.lower().strip()
    for code, s in SUBJECTS.items():
        if t == code or t == s["name"].lower() or t in (a.lower() for a in s["aliases"]):
            return code
    if len(t) < 3:
        return None
    if "(а)" in t:  # «английский (А)», «математика (А)»
        base = t.replace("(а)", "").strip()[:5]
        for code, s in SUBJECTS.items():
            if code.endswith("_a") and base and base in s["name"].lower():
                return code
    # частичное совпадение; курсы (А) проверяем первыми, короткие алиасы («ИЯ», «вит») — только целым словом
    for code in sorted(SUBJECTS, key=lambda c: "(а)" not in t or not c.endswith("_a")):
        s = SUBJECTS[code]
        if t in s["name"].lower():
            return code
        for a in (x.lower() for x in s["aliases"]):
            if len(a) < 4:
                if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", t):
                    return code
            elif t in a or a in t:
                return code
    return None


def resolve_building(text):
    if text is None:
        return None
    t = str(text).lower().replace("корпус", "").replace("№", "").strip()
    for code, b in BUILDINGS.items():
        if t == code.lower() or t in b["aliases"] or text.lower() in b["aliases"]:
            return code
    return None


def walk_minutes(a, b):
    if a == b:
        return 0
    for w in WALK:
        if {w["a"], w["b"]} == {a, b}:
            return w["minutes"]
    return None


def _words(text):
    # NFC: «й» и «ё», разложенные на букву и знак (NFD), иначе рвут слово надвое
    return re.findall(r"\w+", unicodedata.normalize("NFC", str(text or "")).lower().replace("ё", "е"))


def _stem(w):
    if len(w) == 4 and w[-1] in "аеиоуыэюяьй":  # «лаба» находит «лабу», «мама» — «маме»
        return w[:3]
    return w if len(w) <= 4 else w[:len(w) - (1 if len(w) <= 6 else 2)]  # «проект» → «проек», не «прое»


def _stems(query):
    """Основы слов запроса: «вайфай» находит «вайфая», «староста» — «старосты». Предлоги («про», «по») не считаем."""
    return [w if w.isdigit() else _stem(w) for w in _words(query) if w not in STOP]  # числа — целиком: «45123» не «4512»


def _hit(stems, text):
    words = _words(text)
    return all(any(w.startswith(s) for w in words) for s in stems)


def _nums(x):
    """Номера из аргумента numbers: [2, 4], 2, "2, 4", "[2, 4]", ["2"] → [2, 4] без повторов; что-то непонятное → None."""
    if isinstance(x, str):
        try:
            v = json.loads(x)  # массив, присланный строкой: "[2, 4]"
        except ValueError:
            v = None
        if isinstance(v, (list, int)) and not isinstance(v, bool):
            x = v if isinstance(v, list) else [v]
        else:  # только «2», «2, 4», «2 и 4», «№2»: в «1–3» или «все кроме 2» цифры значат не то
            t = re.sub(r"^\s*[№#]\s*|\.\s*$", "", x)
            x = re.findall(r"-?\d+", t) if re.fullmatch(r"\s*-?\d+(?:(?:\s*[,;и]\s*|\s+)-?\d+)*\s*", t) else None
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        x = [x]
    if not isinstance(x, list) or not x:
        return None
    out = []
    for v in x:
        if isinstance(v, str) and re.fullmatch(r"\s*-?\d{1,9}\s*", v):
            v = int(v)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
            return None  # is_integer: у inf и NaN — False
        if int(v) not in out:
            out.append(int(v))
    return out


def _clean(v):
    """Одиночные суррогаты (\\udc00 из кривого JSON) — на «�»: иначе база не сохранит ни одной записи до конца сеанса."""
    if isinstance(v, str):
        return v.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {_clean(k): _clean(x) for k, x in v.items()}
    return v


def _hidden_alive(h, today, live):
    """Скрытая студентом рассылка (post:id) нужна, пока рассылка есть в базе; мероприятие ТПУ (ev:дата|название) —
    пока не прошла дата. live — id рассылок в базе (None — не знаем, оставляем)."""
    if not isinstance(h, str):
        return False
    if h.startswith("ev:"):
        return h[3:13] >= today
    if h.startswith("post:"):
        return live is None or not h[5:].isdigit() or int(h[5:]) in live
    return True


def prune(data, now, live=None):
    """Убирает из личных данных то, что уже прошло: Д/З после срока хранения (expires, неделя или до дня после сдачи),
    сработавшие напоминания, прошедшие дедлайны и экзамены, скрытые рассылки, которых уже нет (live — id рассылок
    в базе). Заметки без даты не трогаем. Возвращает число удалённых."""
    today, now_s = now.date().isoformat(), now.strftime("%Y-%m-%d %H:%M")
    keep = {"homework": lambda h: h.get("expires", "9999") >= today,
            "reminders": lambda r: r.get("fire_at", "9999") > now_s,
            "deadlines": lambda x: x.get("due", "9999") >= today,
            "exams": lambda e: e.get("date", "9999") >= today,
            "hidden": lambda h: _hidden_alive(h, today, live)}
    removed = 0
    for k, ok in keep.items():
        old = data.get(k) or []
        data[k] = [x for x in old if ok(x)]
        removed += len(old) - len(data[k])
    return removed


class Backend:
    def __init__(self, now: dt.datetime, profile: dict, store=None, account_id=None):
        # profile: subgroup, english_teacher, adaptation; у вошедших в аккаунт ещё group и role (student|starosta)
        self.now, self.profile = now, profile
        self.store, self.account_id = store, account_id  # store.Store; без них всё живёт до конца сеанса
        self.homework, self.notes, self.reminders, self.deadlines, self.exams = [], [], [], [], []
        self.hidden = []  # что студент убрал у себя: рассылки старосты (post:id) и мероприятия ТПУ (ev:дата|название)
        self.brief = {"enabled": True, "time": "07:30"}
        self.events = EVENTS
        # Состояние разговора, в базу не пишется: какой список с номерами показан последним (номера в delete_items
        # относятся к нему), что ждёт подтверждения «удалить все» и номер сообщения студента (new_turn). Сайту нужен
        # один Backend на разговор или снимок conversation(), сохранённый вместе с историей (restore).
        self.shown, self.pending, self.turn = {}, {}, None
        self.limits, self.group_limits = dict(LIMITS), dict(GROUP_LIMITS)  # можно поменять для одного Backend
        if store is not None and account_id is not None:
            data = store.load_data(account_id)
            for k in ("homework", "notes", "reminders", "deadlines", "exams", "hidden"):
                setattr(self, k, data.get(k, []))
            self.brief = data.get("brief", self.brief)
        # записано до лимитов больше, чем можно: старое не обрезаем, предел — сколько было (удалил одну — запишешь новую)
        self.had = {"note": len(self.notes), "reminder": sum(1 for r in self.reminders if r.get("fire_at", "") > self._fmt(now))}
        self._group_had = {}
        if profile.get("role") == "starosta" and store is not None and account_id is not None and profile.get("group"):
            self._group_had = {k: len(self._group_records(k)) for k in POST_KINDS}

    def cleanup(self):
        """Удалить из своих данных прошедшее (prune) и сохранить. Возвращает число удалённых записей."""
        live = None
        if self.store is not None and self.profile.get("group"):
            live = {p["id"] for p in self.store.posts(self.profile["group"], after=self._fmt(self.now))}
        data = {k: getattr(self, k) for k in ("homework", "reminders", "deadlines", "exams", "hidden")}
        n = prune(data, self.now, live)
        if n:
            for k, v in data.items():
                setattr(self, k, v)
            self.save()
        return n

    def _data(self):
        return {k: getattr(self, k) for k in PERSIST}

    def save(self):
        if self.store is not None and self.account_id is not None:
            self.store.save_data(self.account_id, self._data())

    # ---------- даты ----------
    def parse_day(self, day: str) -> dt.date:
        today = self.now.date()
        if day in ("today", None):
            return today
        if day == "tomorrow":
            return today + dt.timedelta(days=1)
        if day in WEEKDAYS:
            delta = (WEEKDAYS.index(day) - today.weekday()) % 7
            return today + dt.timedelta(days=delta)
        return dt.date.fromisoformat(day)

    def _lesson_out(self, l):
        o = {"time": f"{l['start']}-{l['end']}", "subject": SUBJECTS[l["subject"]]["name"], "type": l["type"]}
        if l["building"]:
            o.update(building=l["building"], room=l["room"])
        if l["teacher"]:
            o["teacher"] = l["teacher"]
        if l["subgroup"]:
            o["subgroup"] = l["subgroup"]
        return o

    def lessons_on(self, d: dt.date):
        p = self.profile
        res = []
        if d.isoformat() in HOLIDAYS:
            return res
        for l in LESSONS:
            if l["week"] != week_parity(d) or l["weekday"] != d.isoweekday():
                continue
            if l["subgroup"] and p.get("subgroup") and l["subgroup"] != p["subgroup"]:
                continue
            if l["subject"] in ADAPT and not p.get("adaptation"):
                continue
            if l["subject"] == "eng" and p.get("english_teacher") and l["teacher"] != p["english_teacher"]:
                continue
            res.append(l)
        return sorted(res, key=lambda l: l["slot"])

    def _at(self, d, hhmm):
        return dt.datetime.combine(d, dt.time.fromisoformat(hhmm))

    # ---------- инструменты ----------
    def get_schedule(self, day, group=None):
        d = self.parse_day(day)
        out = {"date": d.isoformat(), "weekday": WD_RU[d.weekday()],
               "week": "нечётная" if week_parity(d) == "odd" else "чётная",
               "lessons": [self._lesson_out(l) for l in self.lessons_on(d)]}
        if d.isoformat() in HOLIDAYS:
            out["holiday"] = True
        p = self.profile
        if p.get("subgroup") and p.get("english_teacher"):  # профиль заполнен, а две пары в одно время — накладка
            starts = [l["time"][:5] for l in out["lessons"]]
            for l in out["lessons"]:
                if starts.count(l["time"][:5]) > 1:
                    l["overlap"] = True
        if not self.profile.get("english_teacher") and any(l["subject"] == "eng" for l in self.lessons_on(d)):
            out["note"] = "английский показан для всех групп: преподаватель студента не указан в профиле"
        if not self.profile.get("subgroup"):
            out["note_subgroup"] = "подгруппа не указана, лабы показаны для всех подгрупп"
        return out

    def find_room(self, room=None, building=None, from_building=None):
        b = resolve_building(building)
        if b is None and room:  # ищем аудиторию в расписании
            cands = {l["building"] for l in LESSONS if l["room"] == room}
            if len(cands) == 1:
                b = cands.pop()
            elif len(cands) > 1:
                return {"error": "ambiguous_room", "buildings": sorted(cands)}
        if b is None:
            return {"error": "room_not_found"}
        out = {"building": b, "building_name": BUILDINGS[b]["name"], "address": BUILDINGS[b]["address"]}
        if room:
            out["room"] = room
            exact = [r for r in ROUTES if r["building"] == b and r["room"] == room]
            rule = [r for r in ROUTES if r["building"] == b and r["room"] == "*"]
            if exact:
                out["route"] = exact[0]["route"]
            elif rule:
                out["route"] = rule[0]["route"]
            else:
                digit = next((c for c in room if c.isdigit()), None)
                out["route"] = None
                out["floor_guess"] = int(digit) if digit else None
        fb = resolve_building(from_building)
        if fb:
            out["walk_minutes"] = walk_minutes(fb, b)
        return out

    def get_teacher(self, subject, lesson_type="любой"):
        code = resolve_subject(subject)
        if not code:
            return {"error": "subject_not_found"}
        by_type = {}
        for l in LESSONS:
            if l["subject"] == code and l["teacher"] and lesson_type in ("любой", None, {"ЛК": "лекция", "ПР": "практика", "ЛБ": "лабораторная"}[l["type"]]):
                if code == "eng" and self.profile.get("english_teacher") and l["teacher"] != self.profile["english_teacher"]:
                    continue
                key = {"ЛК": "лекции", "ПР": "практики", "ЛБ": "лабораторные"}[l["type"]]
                by_type.setdefault(key, set()).add(l["teacher"])
        return {"subject": SUBJECTS[code]["name"], "teachers": {k: sorted(v) for k, v in by_type.items()}}

    def next_lesson(self):
        for add in range(0, 8):
            d = self.now.date() + dt.timedelta(days=add)
            for l in self.lessons_on(d):
                if self._at(d, l["start"]) > self.now:
                    return d, l
        return None, None

    def current_or_last(self):
        d = self.now.date()
        past = [l for l in self.lessons_on(d) if self._at(d, l["start"]) <= self.now]
        return past[-1] if past else None

    def check_travel_time(self, from_building, to_room=None, to_building=None):
        fb = resolve_building(from_building)
        d, l = self.next_lesson()
        if to_building or to_room:
            r = self.find_room(to_room, to_building)
            tb = r.get("building")
            target = {"room": to_room, "building": tb}
            start = None
        else:
            if l is None:
                return {"error": "no_next_lesson"}
            if d != self.now.date():
                return {"error": "no_more_lessons_today", "next_lesson": self._lesson_out(l) | {"date": d.isoformat()}}
            tb, target = l["building"], self._lesson_out(l)
            start = self._at(d, l["start"])
        w = walk_minutes(fb, tb) if fb and tb else None
        out = {"target": target, "walk_minutes": w}
        if start:
            left = int((start - self.now).total_seconds() // 60)
            out["minutes_left"] = left
            out["verdict"] = "неизвестно" if w is None else ("успеваешь" if w <= left - 2 else "впритык" if w <= left else "опоздаешь")
        return out

    def find_food(self, building=None):
        b = resolve_building(building)
        if b is None:
            cur = self.current_or_last()
            b = cur["building"] if cur else None
        if b is None:
            return {"error": "building_unknown"}
        places = FOOD.get(b, [])
        out = {"building": b, "places": places}
        d, nl = self.next_lesson()
        if nl and d == self.now.date():
            last = self.current_or_last()
            free_from = max(self.now, self._at(d, last["end"])) if last else self.now
            brk = int((self._at(d, nl["start"]) - free_from).total_seconds() // 60)
            wn = walk_minutes(b, nl["building"]) if nl["building"] else 0
            out["next_lesson"] = self._lesson_out(nl)
            out["free_minutes"] = brk
            if places and wn is not None:
                p = min(places, key=lambda x: x["walk_minutes"])
                need = p["walk_minutes"] + EAT_MINUTES + p["walk_minutes"] + wn
                out["enough_time"] = need <= brk
                out["need_minutes"] = need
            nb = nl["building"]
            if nb and nb != b and FOOD.get(nb) and wn is not None:  # поесть уже у корпуса следующей пары
                p = min(FOOD[nb], key=lambda x: x["walk_minutes"])
                need2 = wn + 2 * p["walk_minutes"] + EAT_MINUTES
                out["near_next_building"] = {"building": nb, "places": FOOD[nb],
                                             "need_minutes": need2, "enough_time": need2 <= brk}
        return out

    def _exams(self, subject=None):
        """Экзамены, которые ещё впереди, по дате: [(запись, что показать)]."""
        out = []
        for e in self.exams:
            if subject and not self._same_subject(subject, e):
                continue
            left = (dt.date.fromisoformat(e["date"]) - self.now.date()).days
            if left < 0:
                continue
            out.append((e, {k: v for k, v in e.items() if k not in ("code", "reminders")} | {"days_left": left}))
        return sorted(out, key=lambda x: (x[1]["date"], x[1].get("time", "")))

    def get_exams(self, subject=None):
        items = self._exams(subject)
        self._remember("exam", items)
        return {"items": [o for _, o in items]}

    def _same_subject(self, subject, item):
        code = resolve_subject(subject)
        return item.get("code") == code if code else (item.get("subject") or "").lower() == subject.strip().lower()

    def add_exam(self, subject, date, kind="экзамен", time=None, room=None, building=None):
        code = resolve_subject(subject)
        try:
            d = self.parse_date(date)
        except (ValueError, StopIteration):
            return {"error": "bad_date"}
        left = (d - self.now.date()).days
        if left < 0:
            return {"error": "date_in_past"}
        e = {"code": code, "subject": SUBJECTS[code]["name"] if code else subject, "kind": kind, "date": d.isoformat()}
        for k, v in (("time", time), ("room", room), ("building", building)):
            if v:
                e[k] = v
        old = [x for x in self.exams if self._same_subject(subject, x) and x["kind"] == kind]
        for x in old:  # перенос даты: старую запись заменяем
            self.exams.remove(x)
        e["reminders"] = [(d - dt.timedelta(days=k)).isoformat() for k in (7, 1) if left > k]
        self.exams.append(e)
        out = {"ok": True, "subject": e["subject"], "date": e["date"], "days_left": left, "reminders": e["reminders"]}
        if old:
            out["updated"] = True
        return out

    # ---------- даты для Д/З, напоминаний, дедлайнов ----------
    def _fmt(self, t):
        return t.strftime("%Y-%m-%d %H:%M")

    def parse_due(self, due: str, future=False) -> dt.date:
        """YYYY-MM-DD | today | tomorrow | monday..sunday | 'in N days'.
        future=True: «к пятнице», сказанное в пятницу, — это следующая пятница."""
        due = due.strip().lower()
        if due.startswith("in ") and due.endswith(("day", "days")):
            return self.now.date() + dt.timedelta(days=int(due.split()[1]))
        d = self.parse_day(due)
        if future and due in WEEKDAYS and d <= self.now.date():
            d += dt.timedelta(days=7)
        return d

    def parse_date(self, s: str) -> dt.date:
        """YYYY-MM-DD или «15 января» (ближайшее будущее такое число)."""
        s = s.strip().lower()
        m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?", s)
        if m:
            day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
        else:
            m = re.fullmatch(r"(\d{1,2})\s+([а-яё]+)(?:\s+(\d{4}))?(?:\s*г\.?)?", s)
            if not m:
                return dt.date.fromisoformat(s)
            day, year = int(m.group(1)), m.group(3)
            month = next(i for i, x in enumerate(MONTHS_GEN, 1) if x.startswith(m.group(2)[:3]))
        if year:
            return dt.date(int(year) + (2000 if len(year) == 2 else 0), month, day)
        d = dt.date(self.now.year, month, day)
        return d if d >= self.now.date() else d.replace(year=d.year + 1)

    def parse_at(self, at: str) -> dt.datetime:
        """'HH:MM' (ближайшее такое) | 'tomorrow HH:MM' | 'monday HH:MM' | 'YYYY-MM-DD HH:MM'."""
        parts = at.strip().lower().split()
        t = dt.time.fromisoformat(parts[-1])
        if len(parts) == 1:
            cand = dt.datetime.combine(self.now.date(), t)
            return cand if cand > self.now else cand + dt.timedelta(days=1)
        d = self.parse_day(parts[0])
        cand = dt.datetime.combine(d, t)
        if parts[0] in WEEKDAYS and cand <= self.now:
            cand += dt.timedelta(days=7)
        return cand

    def next_class_of(self, code, after=None):
        after = after or self.now
        for add in range(0, 15):
            d = after.date() + dt.timedelta(days=add)
            for l in self.lessons_on(d):
                if l["subject"] == code and self._at(d, l["start"]) > after:
                    return self._at(d, l["start"])
        return None

    # ---------- Д/З (хранится 7 дней) ----------
    def _homework_due(self, code, name, due):
        """Срок Д/З по расписанию: {"due", "remind_at", "expires", "no_class_that_day"} или {"error": ...}."""
        if due == "next_class":
            when = self.next_class_of(code) if code else None
            if when is None:
                return {"error": "next_class_not_found", "subject": name}
            due_s, remind = self._fmt(when), self._fmt(when - dt.timedelta(hours=1))
        else:
            try:
                d = self.parse_due(due, future=True)
            except ValueError:
                return {"error": "bad_date"}
            if d < self.now.date():
                return {"error": "due_in_past", "subject": name}
            when = self.next_class_of(code, max(self.now, dt.datetime.combine(d, dt.time(0, 0)))) if code else None
            if when and when.date() == d:
                due_s, remind = self._fmt(when), self._fmt(when - dt.timedelta(hours=1))
            else:
                due_s = d.isoformat()
                remind = self._fmt(dt.datetime.combine(d - dt.timedelta(days=1), dt.time(19, 0)))
        if remind <= self._fmt(self.now):  # до пары меньше часа или «на завтра» после 19:00 — напоминать поздно
            remind = None
        due_d = dt.date.fromisoformat(due_s[:10])
        expires = max(self.now.date() + dt.timedelta(days=7), due_d + dt.timedelta(days=1)).isoformat()
        # пары по предмету в этот день нет — срок просто по дате
        no_class = bool(code and " " not in due_s and not any(l["subject"] == code for l in self.lessons_on(due_d)))
        return {"due": due_s, "remind_at": remind, "expires": expires, "no_class_that_day": no_class}

    def add_homework(self, subject, task, due="next_class"):
        code = resolve_subject(subject)
        name = SUBJECTS[code]["name"] if code else subject
        r = self._homework_due(code, name, due)
        if "error" in r:
            return r
        self.homework.append({"code": code, "subject": name, "task": task, "due": r["due"], "expires": r["expires"]})
        out = {"ok": True, "subject": name, "due": r["due"], "remind_at": r["remind_at"], "expires": r["expires"]}
        if r["no_class_that_day"]:
            out["no_class_that_day"] = True
        return out

    def _homework(self, subject=None, due=None):
        """Д/З, срок которого не прошёл, по сроку: [(запись, что показать)] или {"error": ...}."""
        group = [dict(p, due=d, expires=d[:10], from_starosta=True)  # от старосты, срок по расписанию читающего
                 for p in self._posts("homework") for d in [self._post_due(p)]]
        if subject and not resolve_subject(subject) and not any(self._same_subject(subject, h) for h in self.homework + group):
            return {"error": "subject_not_found", "subject": subject}
        day = self.parse_due(due).isoformat() if due else None
        now_s, today = self._fmt(self.now), self.now.date().isoformat()
        items = [h for h in self.homework + group if h["expires"] >= today
                 and (h["due"] > now_s if " " in h["due"] else h["due"] >= today)  # срок ещё не прошёл
                 and (not subject or self._same_subject(subject, h))
                 and (not day or h["due"].startswith(day))]
        return [(h, {k: h[k] for k in ("subject", "task", "due", "from_starosta") if k in h})
                for h in sorted(items, key=lambda h: h["due"])]

    def list_homework(self, subject=None, due=None):
        items = self._homework(subject, due)
        if isinstance(items, dict):
            return items
        self._remember("homework", items)
        return {"items": [o for _, o in items]}

    # ---------- напоминания и заметки ----------
    def create_reminder(self, text, in_minutes=None, in_hours=None, in_days=None, at=None):
        if not isinstance(text, str) or not text.strip():
            return {"error": "empty_text"}
        if in_minutes or in_hours or in_days:
            fire = self.now + dt.timedelta(minutes=in_minutes or 0, hours=in_hours or 0, days=in_days or 0)
        elif at:
            try:
                fire = self.parse_at(at)
            except ValueError:
                return {"error": "bad_time"}
        else:
            return {"error": "time_required"}
        if fire <= self.now:
            return {"error": "time_in_past"}
        own = [r for r in self._records("reminder") if not r[0].get("from_starosta")]
        if len(own) >= max(self.limits["reminder"], self.had["reminder"]):  # напоминания старосты не в счёт: их лимит у старосты
            return {"error": "limit", "limit": max(self.limits["reminder"], self.had["reminder"]),
                    "items": self._numbered("reminder", own, ask=True)}
        self.reminders.append({"text": text, "fire_at": self._fmt(fire)})
        return {"ok": True, "fire_at": self._fmt(fire)}

    def add_note(self, text):
        if not isinstance(text, str) or not text.strip():
            return {"error": "empty_text"}
        if len(self.notes) >= max(self.limits["note"], self.had["note"]):
            return {"error": "limit", "limit": max(self.limits["note"], self.had["note"]),
                    "items": self._numbered("note", self._records("note"), ask=True)}
        self.notes.append(text)
        return {"ok": True, "id": len(self.notes)}

    def list_notes(self, query=None, kind=None):
        """Заметки и напоминания (свои и от старосты) с номерами: по ним их можно удалить (delete_items).
        kind — только заметки (note) или только напоминания (reminder): номера запоминаем только у показанного,
        иначе «удали напоминание 3» после «мои заметки» ушло бы по списку, которого студент не видел.
        Поиск по основам слов: «вайфай» находит «вайфая», «староста» — «старосты»."""
        stems, out = _stems(query), {}
        if kind != "reminder":
            out["notes"] = self._numbered("note", [r for r in self._records("note") if _hit(stems, r[0])], found=bool(stems))
        if kind != "note":
            out["reminders"] = self._numbered("reminder", [r for r in self._records("reminder") if _hit(stems, r[0]["text"])],
                                              found=bool(stems))
        return out

    # ---------- дедлайны лаб и курсовых ----------
    def add_deadline(self, title, due, subject=None, kind="другое"):
        code = resolve_subject(subject) if subject else None
        name = SUBJECTS[code]["name"] if code else subject
        if due == "next_class":
            when = self.next_class_of(code) if code else None
            if when is None:
                return {"error": "next_class_not_found"}
            d = when.date()
        else:
            try:
                d = self.parse_due(due, future=True)
            except ValueError:
                return {"error": "bad_date"}
        left = (d - self.now.date()).days
        if left < 0:
            return {"error": "due_in_past"}
        rem = [(d - dt.timedelta(days=k)).isoformat() for k in (3, 1) if left > k]
        item = {"code": code, "title": title, "subject": name, "kind": kind, "due": d.isoformat()}
        self.deadlines.append(item)
        return {"ok": True, "due": d.isoformat(), "days_left": left, "reminders": rem}

    def _deadlines(self, subject=None, kind=None):
        """Дедлайны, которые не прошли, по сроку: [(запись, что показать)] или {"error": ...}."""
        if subject and not resolve_subject(subject) and not any(self._same_subject(subject, x) for x in self.deadlines):
            return {"error": "subject_not_found", "subject": subject}
        out = []
        for x in self.deadlines:
            left = (dt.date.fromisoformat(x["due"]) - self.now.date()).days
            if left < 0 or (kind and x["kind"] != kind) or (subject and not self._same_subject(subject, x)):
                continue
            out.append((x, {"title": x["title"], "subject": x["subject"], "due": x["due"], "days_left": left}))
        return sorted(out, key=lambda x: x[1]["days_left"])

    def list_deadlines(self, subject=None, kind=None):
        items = self._deadlines(subject, kind)
        if isinstance(items, dict):
            return items
        self._remember("deadline", items)
        return {"items": [o for _, o in items]}

    # ---------- сессия ----------
    def make_study_plan(self, subject, hours_per_day=None, skip_days=None, tickets_total=None):
        today = self.now.date().isoformat()
        ex = sorted((e for e in self.exams if self._same_subject(subject, e) and e["date"] > today), key=lambda e: e["date"])
        if not ex:
            return {"error": "exam_not_found"}
        if not tickets_total:
            return {"error": "tickets_unknown"}
        exam = dt.date.fromisoformat(ex[0]["date"])
        skip = {WEEKDAYS.index(w.strip().lower()) for w in (skip_days or "").split(",") if w.strip().lower() in WEEKDAYS}
        days = [self.now.date() + dt.timedelta(days=k) for k in range(1, (exam - self.now.date()).days)]
        days = [d for d in days if d.weekday() not in skip]
        review = days[-2:] if len(days) >= 6 else []
        learn = days[:len(days) - len(review)]
        if not learn:
            return {"error": "too_late", "exam_date": exam.isoformat()}
        blocks = min(4, len(learn), tickets_total)
        plan, t0 = [], 1
        for b in range(blocks):
            seg = learn[b * len(learn) // blocks:(b + 1) * len(learn) // blocks]
            t1 = max(t0, (b + 1) * tickets_total // blocks)
            plan.append({"from": seg[0].isoformat(), "to": seg[-1].isoformat(), "tickets": f"{t0}–{t1}" if t1 > t0 else str(t0)})
            t0 = t1 + 1
        if review:
            plan.append({"from": review[0].isoformat(), "to": review[-1].isoformat(), "tickets": "повторение всех билетов"})
        return {"ok": True, "subject": ex[0]["subject"], "exam_date": exam.isoformat(),
                "tickets_total": tickets_total, "plan": plan, "reminders": "в 19:00 в дни плана"}

    # ---------- брифинг ----------
    def set_daily_brief(self, enabled, time=None):
        self.brief = {"enabled": enabled, "time": (time or "07:30") if enabled else None}
        return {"ok": True, "enabled": enabled, "time": self.brief["time"]}

    def get_daily_brief(self, day="today"):
        s = self.get_schedule(day)
        d = self.parse_day(day)
        ls = self.lessons_on(d)
        for i in range(1, len(ls)):
            a, b = ls[i - 1], ls[i]
            if a["start"] == b["start"]:  # две пары в одно время: переход не считаем
                continue
            if a["building"] and b["building"] and a["building"] != b["building"]:
                s["lessons"][i]["walk_from_prev_minutes"] = walk_minutes(a["building"], b["building"])
        # Д/З от старосты — только в строке from_starosta, чтобы не повторять его дважды
        hw = [x for x in self._homework(due=d.isoformat()) if not x[1].get("from_starosta")]
        dl = [x for x in self._deadlines() if x[1]["days_left"] <= 7]
        ex = [x for x in self._exams() if 0 <= x[1]["days_left"] <= 14]
        for ns, items in (("homework", hw), ("deadline", dl), ("exam", ex)):  # «удали первое» — из того, что в брифинге
            self._remember(ns, items)
        s["homework"], s["deadlines"], s["exams"] = [o for _, o in hw], [o for _, o in dl], [o for _, o in ex]
        fs = self.starosta_news()
        if fs:  # отдельной строкой в брифинге
            s["from_starosta"] = fs
        return s

    def set_profile(self, subgroup=None, english_teacher=None, adaptation=None):
        if subgroup is not None and subgroup not in (1, 2, 3):
            return {"error": "bad_subgroup"}
        if english_teacher:  # «Аксёнова», «аксеновой» → «Аксёнова Н. В.»
            key = english_teacher.lower().replace("ё", "е")[:5]
            hits = [t for t in ENG_TEACHERS if t.lower().replace("ё", "е").startswith(key)]
            if len(hits) != 1:
                return {"error": "teacher_not_found", "options": ENG_TEACHERS}
            english_teacher = hits[0]
        for k, v in (("subgroup", subgroup), ("english_teacher", english_teacher), ("adaptation", adaptation)):
            if v is not None:
                self.profile[k] = v
        if self.store is not None and self.account_id is not None:
            self.store.save_profile(self.account_id, self.profile)
        return {"ok": True, "profile": {k: v for k, v in self.profile.items() if k not in ("group", "role")}}

    # ---------- мероприятия и хакатоны ----------
    def _events(self, kind=None, limit=None):
        """Мероприятия ТПУ и от старосты, которые ещё не начались, по дате: [(запись, что показать)].
        limit — сколько дней вперёд (None — все)."""
        today = self.now.date()
        group = [{"id": p["id"], "title": p["title"], "date": p["date"], "time": p.get("time"), "place": p.get("place"),
                  "from_starosta": True} for p in self._posts("event")]
        items = []
        for e in self.events + group:
            if GUARD.check_obj(e):  # мероприятие с запретной фразой (например, список пополнили позже) не показываем
                continue
            if self._ref("event", e) in self.hidden:  # студент убрал его у себя
                continue
            left = (dt.date.fromisoformat(e["date"]) - today).days
            if left < 0 or (limit is not None and left > limit) or self._started(e, left):
                continue
            if kind and e.get("kind") != kind and kind.lower()[:6] not in e["title"].lower():
                continue
            o = {k: e[k] for k in ("title", "kind", "date", "time", "place", "description", "url") if e.get(k)}
            o["days_left"] = left
            if e.get("from_starosta"):
                o["from_starosta"] = True
            items.append((e, o))
        return sorted(items, key=lambda x: (x[1]["date"], x[1].get("time") or ""))

    def get_events(self, kind=None, period="month"):
        items = self._events(kind, EVENT_PERIODS.get(period, 31))
        self._remember("event", items[:10])
        out = {"items": [o for _, o in items[:10]]}
        if len(items) > 10:
            out["more"] = len(items) - 10
        return out

    def _started(self, e, left):
        """Мероприятие сегодня, и время начала уже прошло (как в group_add_event: такое не показываем)."""
        return left == 0 and bool(e.get("time")) and e["time"] <= self.now.strftime("%H:%M")

    # ---------- удаление ----------
    # Номера в delete_items и group_delete_post — из последнего списка этого вида, который видел студент (list_notes,
    # list_homework, брифинг, список «что удалить?»): запись узнаётся по ссылке (_ref), поэтому после удаления одной
    # («удали 2») номера остальных не сдвигаются («и 4 тоже» — та же четвёртая). Списка ещё не было — номера не угадываем,
    # а показываем список. «Удалить все» — только после показанного списка: сначала подтверждение (confirm), удаляем
    # следующим же вызовом. Рассылки старосты и мероприятия ТПУ студент не удаляет, а скрывает у себя (hidden).
    # Если чат сообщает о каждом сообщении студента (new_turn), номера относятся только к спискам из прошлых сообщений,
    # а «да» подтверждает только список из предыдущего: модель не подтвердит удаление сама себе вторым вызовом.
    def new_turn(self):
        """Пришло новое сообщение студента: chat.py и сайт вызывают это перед ответом модели."""
        self.turn = (self.turn or 0) + 1

    def conversation(self):
        """Снимок разговора: какие списки с номерами показаны и что ждёт «да». Если ход не удался и убран из истории,
        chat.py возвращает снимок (restore): студент тех списков не видел. Сайт может хранить его вместе с историей."""
        return {"shown": dict(self.shown), "pending": dict(self.pending), "turn": self.turn}

    def restore(self, state):
        self.shown, self.pending, self.turn = dict(state["shown"]), dict(state["pending"]), state["turn"]

    def _seen(self, entry, previous=False):
        """Список (из shown или pending) студент уже видел: показан в одном из прошлых сообщений, previous=True —
        ровно в предыдущем. Без new_turn (генератор датасета, тесты) сообщения не считаем."""
        if entry is None:
            return False
        if self.turn is None or entry.get("turn") is None:
            return True
        return entry["turn"] == self.turn - 1 if previous else entry["turn"] < self.turn

    def _ref(self, kind, x):
        if kind == "note":
            return json.dumps(["note", x], ensure_ascii=False)
        if x.get("from_starosta"):
            return f"post:{x['id']}"
        if kind == "event":  # со временем: два занятия мастер-класса в один день — разные мероприятия
            return f"ev:{x['date']}|{x.get('time') or ''}|{x['title']}"
        if kind == "exam":  # как add_exam решает, что это тот же экзамен: предмет без учёта регистра
            return json.dumps(["exam", x.get("code") or str(x.get("subject") or "").strip().lower(), x.get("kind")],
                              ensure_ascii=False)
        return json.dumps([kind] + [x.get(k) for k in OWN_KEYS[kind]], ensure_ascii=False)

    def _records(self, kind):
        """Всё, что студент может удалить или скрыть, в том же порядке, что и в списках: [(запись, что показать, ссылка)]."""
        if kind == "note":
            items = [(n, {"text": n}) for n in self.notes]
        elif kind == "reminder":
            now_s = self._fmt(self.now)
            group = [dict(p, from_starosta=True) for p in self._posts("reminder")]
            items = [(r, {k: r[k] for k in ("text", "fire_at", "from_starosta") if k in r})
                     for r in sorted((r for r in self.reminders + group if r["fire_at"] > now_s), key=lambda r: r["fire_at"])]
        else:
            items = {"homework": self._homework, "deadline": self._deadlines, "exam": self._exams, "event": self._events}[kind]()
        return [(x, o, self._ref(kind, x)) for x, o in items]

    def _remember(self, kind, items):
        self._show(kind, [self._ref(kind, x) for x, _ in items])

    def _show(self, ns, refs, found=False):
        """Запомнить показанный список. Пустой (номеров в нём нет) прежний не заменяет. У старосты все списки — из одних
        рассылок группы: номера относятся к последнему из них, старые забываем. found — список найденного по словам."""
        if not refs:
            return
        if ns.startswith("group:"):
            for k in [k for k in self.shown if k.startswith("group:")]:
                del self.shown[k]
        self.shown[ns] = {"refs": refs, "turn": self.turn} | ({"found": True} if found else {})

    def _numbered(self, ns, recs, ask=False, part=False, found=False):
        """Список с номерами n. Запоминаем его: номера в следующем удалении значат то же самое.
        ask=True — список показан, чтобы выбрать, что удалить: ответ «все» на него удаляет весь этот список.
        part=True — это подтверждение «удалить все про X»: просто «все» его не подтверждает (это уже про все записи)."""
        refs = [ref for _, _, ref in recs]
        self._show(ns, refs, found)
        if ask:
            self.pending[ns] = {"refs": list(refs), "turn": self.turn, "part": part}
        return [{"n": i} | o for i, (_, o, _) in enumerate(recs, 1)]

    @staticmethod
    def _still(refs, recs, done=()):
        """Какие номера списка refs ещё есть среди recs: [(номер, ссылка)] (одинаковые заметки считаем поштучно).
        done — номера, уже удалённые по этому списку: у двух одинаковых заметок вторая не займёт номер первой."""
        left, out = Counter(ref for _, _, ref in recs), []
        for i, ref in enumerate(refs, 1):
            if i in done:
                continue
            if left[ref] > 0:
                left[ref] -= 1
                out.append((i, ref))
        return out

    @staticmethod
    def _rec_words(x):
        """Слова записи для поиска: текст полей TEXT_KEYS и отдельно — название предмета с сокращениями («питон»)."""
        if not isinstance(x, dict):
            return _words(x), []
        code = x.get("code") if x.get("code") in SUBJECTS else None
        main = " ".join(str(x.get(k) or "") for k in TEXT_KEYS if not (k == "subject" and code))
        return _words(main), _words(" ".join([SUBJECTS[code]["name"]] + SUBJECTS[code]["aliases"]) if code else "")

    @classmethod
    def _matches(cls, stems, x):
        """Подходит ли запись под слова запроса. Предмет узнаём и по сокращениям («удали дз по питону» — а в записи
        «Основы программирования на Python»); число («лаба 3») — только целиком и не из названия предмета («1.3»)."""
        mw, sw = cls._rec_words(x)
        return all(s in mw if s.isdigit() else any(w.startswith(s) for w in mw + sw) for s in stems)

    @classmethod
    def _close(cls, query, x):
        """Каждое слово запроса совпало со словом записи почти целиком, а не с началом другого, длинного слова:
        «Коля» — не «коллоквиум», «Маша» — не «машину». Только тогда одну найденную запись удаляем без вопроса."""
        mw, sw = cls._rec_words(x)
        def ok(w):
            s = _stem(w)
            return any(r.startswith(s) and len(r) <= len(w) + (1 if len(w) <= 4 else 2) for r in mw + sw)
        return all(ok(w) for w in _words(query) if w not in STOP and not w.isdigit())

    @staticmethod
    def _plain(x):
        """Текст записи как есть: в нём ищем знаки из запроса без слов («🎂»)."""
        return str(x) if not isinstance(x, dict) else " ".join(str(x.get(k) or "") for k in TEXT_KEYS)

    def _delete(self, ns, recs, numbers, query, all_, drop, again):
        """Общая часть delete_items и group_delete_post. recs — что можно удалить сейчас, drop(выбранное) удаляет
        и возвращает (удалено, скрыто), again() — список после удаления."""
        all_ = all_ is True or str(all_).strip().lower() in ("true", "1", "yes", "да")
        pending = self.pending.get(ns)
        # список ждёт ответа до следующего удаления этого вида; показанный в этом же сообщении (модель удаляет несколько
        # записей подряд) — ждёт дальше, иначе «все» в ответ на него спросит про все записи
        if all_ or self.turn is None or (pending or {}).get("turn") != self.turn:
            self.pending.pop(ns, None)
        if not self._seen(pending, previous=True):
            pending = None  # «да» подтверждает только список из прошлого сообщения
        if isinstance(query, (list, tuple)):  # модель могла прислать не строку: ["ДР"], 15 — ищем по тексту
            query = " ".join(map(str, query))
        elif query is not None and not isinstance(query, str):
            query = str(query)
        stems = _stems(query) if query else []
        # запрос без слов, кроме предлогов («🎂», «с 🎂»), — тоже отбор: ищем эти знаки как есть, а не считаем его «все»
        marks = [] if stems else re.findall(r"[^\w\s]+", query or "")
        if not recs:
            return {"error": "nothing_to_delete"}
        shown, by_number = self.shown.get(ns), numbers not in (None, "", [])
        if by_number:
            nums = _nums(numbers)
            if not self._seen(shown) or not shown["refs"]:  # списка не показывали: какую запись студент считает второй, не знаем
                return {"choose": True, "items": self._numbered(ns, recs, ask=True)}
            refs, done = shown["refs"], set(shown.get("done", ()))
            bad = [n for n in nums if not 1 <= n <= len(refs)] if nums else [numbers]
            if bad:  # номера остальных не меняем: студент поправит опечатку по тому же списку
                still = self._still(refs, recs, done)
                if not still:
                    return {"error": "bad_number", "numbers": bad, "items": self._numbered(ns, recs, ask=True)}
                shows = {ref: o for _, o, ref in recs}
                self.pending[ns] = {"refs": [ref for _, ref in still], "turn": self.turn, "part": False}
                return {"error": "bad_number", "numbers": bad, "items": [{"n": i} | shows[ref] for i, ref in still]}
            picked = [(n, refs[n - 1]) for n in nums]
        elif stems or marks or all_:
            hits, guessed = recs, False
            if marks:
                hits = [r for r in recs if all(m in self._plain(r[0]) for m in marks)]
            elif stems:
                hits = [r for r in recs if self._matches(stems, r[0])]
                code = None if hits else resolve_subject(query)  # «матеша», «мат.логика» — по названию предмета
                if code:
                    hits = [r for r in recs if isinstance(r[0], dict) and r[0].get("code") == code]
                    guessed = True  # совпал только предмет, остальные слова — нет: сами не удаляем, спрашиваем
                elif len(hits) == 1 and not self._close(query, hits[0][0]):
                    guessed = True  # слово совпало только с началом другого слова: тоже спрашиваем
            if not hits:
                return {"error": "not_found", "query": query, "items": self._numbered(ns, recs, ask=True)}
            refs = [ref for _, _, ref in hits]
            if all_:
                part = bool(stems or marks)
                if not (pending and ((not part and not pending["part"]) or Counter(pending["refs"]) == Counter(refs))):
                    return {"confirm": True, "count": len(hits), "items": self._numbered(ns, hits, ask=True, part=part, found=part)}
                refs = pending["refs"]  # подтвердили: удаляем ровно то, что показали
            elif len(hits) > 1 or guessed:
                return {"choose": True, "items": self._numbered(ns, hits, ask=True, found=bool(stems or marks))}
            picked, done = [(None, ref) for ref in refs], set()
        else:
            return {"choose": True, "items": self._numbered(ns, recs, ask=True)}
        pool = {}
        for r in recs:
            pool.setdefault(r[2], []).append(r)
        chosen, gone, used = [], [], set()
        for n, ref in picked:  # одинаковые заметки — разные номера: берём по одной на номер
            if n in done:  # этот номер уже удалили: вторую такую же заметку под ним не берём
                gone.append(n)
            elif pool.get(ref):
                chosen.append(pool[ref].pop(0))
                used.add(n)
            elif n is not None:
                gone.append(n)
        if not chosen:
            return {"error": "already_deleted"}
        deleted, hidden = drop(chosen)
        if by_number:  # в самом списке: снимок разговора (conversation) видит пометку — удаление ведь не откатывается
            shown["done"] = sorted(done | used)
        out = {"ok": True}
        if deleted:
            out["deleted"] = deleted
        if hidden:
            out["hidden"] = hidden
        if gone:
            out["gone"] = gone  # эти номера уже удалены раньше
        after = again()
        # сколько осталось: по номеру из полного списка — в нём (мероприятий могли показать не все), иначе всего:
        # после «Нашёл только это» и «да» нельзя сказать «больше ничего нет», если есть другие записи
        if by_number and not shown.get("found"):
            return out | {"left": len(self._still(shown["refs"], after, shown["done"]))}
        return out | {"left": len(after)}

    def _drop(self, kind, chosen):
        deleted, hidden = [], []
        own = {"note": self.notes, "reminder": self.reminders, "homework": self.homework, "deadline": self.deadlines,
               "exam": self.exams}
        for x, o, ref in chosen:
            if ref.startswith(("post:", "ev:")):  # рассылка старосты или мероприятие ТПУ: скрываем только у себя
                if ref not in self.hidden:
                    self.hidden.append(ref)
                hidden.append(o)
            else:
                own[kind].remove(x)
                deleted.append(o)
        return deleted, hidden

    def delete_items(self, kind, numbers=None, query=None, all=None):
        if kind not in DELETE_KINDS:
            return {"error": "bad_kind", "kinds": list(DELETE_KINDS)}
        return self._delete(kind, self._records(kind), numbers, query, all, lambda ch: self._drop(kind, ch),
                            lambda: self._records(kind))

    # ---------- рассылки старосты ----------
    def _posts(self, kind=None):
        """Актуальные рассылки старосты для группы и подгруппы студента (если он вошёл в аккаунт)."""
        if self.store is None or not self.profile.get("group"):
            return []
        posts = self.store.posts(self.profile["group"], self.profile.get("subgroup"), after=self._fmt(self.now), kind=kind)
        # рассылку с запретной фразой (список могли пополнить уже после неё) студенту не показываем, убранную им — тоже
        return [p for p in posts if f"post:{p['id']}" not in self.hidden and not self._blocked(p)]

    @staticmethod
    def _blocked(p):
        return GUARD.check_obj({k: p.get(k) for k in ("subject", "task", "title", "place", "text")})

    def starosta_news(self):
        """Всё актуальное от старосты: Д/З, мероприятия на 2 недели вперёд, напоминания на неделю вперёд."""
        out, now_s = [], self._fmt(self.now)
        week = self._fmt(self.now + dt.timedelta(days=7))
        for p in self._posts():
            if p["kind"] == "homework":
                due = self._post_due(p)
                if (due > now_s if " " in due else due >= now_s[:10]):
                    out.append(({"kind": "homework", "subject": p["subject"], "task": p["task"], "due": due}, due))
            elif p["kind"] == "event":
                left = (dt.date.fromisoformat(p["date"]) - self.now.date()).days
                if 0 <= left <= 14 and not self._started(p, left):
                    o = {"kind": "event", "title": p["title"], "date": p["date"]}
                    o.update({k: p[k] for k in ("time", "place") if p.get(k)})
                    out.append((o | {"days_left": left}, f"{p['date']} {p.get('time') or ''}"))
            elif p["kind"] == "reminder" and now_s < p["fire_at"] <= week:
                out.append(({"kind": "reminder", "text": p["text"], "fire_at": p["fire_at"]}, p["fire_at"]))
        return [o for o, _ in sorted(out, key=lambda x: x[1])]

    def _starosta(self, subgroup):
        """None, если можно рассылать, иначе ошибка."""
        if self.profile.get("role") != "starosta":
            return {"error": "not_starosta"}
        if self.store is None or self.account_id is None or not self.profile.get("group"):
            return {"error": "not_logged_in"}
        if subgroup is not None and subgroup not in (1, 2, 3):
            return {"error": "bad_subgroup"}
        return None

    def _post(self, kind, data, until, subgroup):
        self.store.add_post(self.account_id, self.profile["group"], kind, data, until, subgroup=subgroup,
                            created=self._fmt(self.now))
        out = {"recipients": self.store.members(self.profile["group"], subgroup, exclude=self.account_id)}
        if subgroup:
            out["subgroup"] = subgroup
        return out

    def group_add_homework(self, subject, task, due="next_class", subgroup=None):
        err = self._starosta(subgroup)
        if err:
            return err
        code = resolve_subject(subject)
        name = SUBJECTS[code]["name"] if code else subject
        due = due or "next_class"
        # «к следующей паре» (и время пары в названный день) у каждого своё: лабы у подгрупп и английский у групп
        # преподавателей в разное время. Поэтому в рассылке хранится сам срок (spec), а время пары каждый студент
        # получает по своему расписанию (_post_due). Здесь — срок для того, кому задали, и самый поздний из всех.
        b = self if not subgroup or subgroup == self.profile.get("subgroup") else Backend(self.now, dict(self.profile, subgroup=subgroup))
        r = b._homework_due(code, name, due)
        if r.get("error") in ("bad_date", "due_in_past"):
            return r
        # срок у каждого возможного получателя; due_differs — только среди тех, у кого этот предмет вообще есть
        # (курсы (А) — не у всех, у самого старосты предмета может не быть)
        alls = [(has, x) for has, x in ((not code or Backend(self.now, p).next_class_of(code) is not None,
                                          Backend(self.now, p)._homework_due(code, name, due)) for p in self._readers(subgroup))
                if "error" not in x]
        outs = [x for has, x in alls if has]
        if "error" in r or (code and b.next_class_of(code) is None):
            if not outs:
                return r
            r = min(outs, key=lambda x: x["due"])
        dues = {x["due"] for x in outs} | {r["due"]}
        until = max(d if " " in d else d + " 23:59" for d in dues | {x["due"] for _, x in alls})
        data = {"code": code, "subject": name, "task": task, "due": r["due"], "spec": due}
        full = self._group_full("homework")
        if full:
            return full
        sent = self._post("homework", data, until, subgroup)
        out = {"ok": True, "subject": name, "due": r["due"]}
        if len(dues) > 1:
            out["due_differs"] = True  # у разных подгрупп (групп английского) пара в разное время
        elif r["no_class_that_day"]:
            out["no_class_that_day"] = True
        return out | sent

    def _readers(self, subgroup):
        """Возможные профили получателей рассылки: подгруппа × преподаватель английского × курсы (А)."""
        return [{"subgroup": s, "english_teacher": t, "adaptation": a}
                for s in ([subgroup] if subgroup else [1, 2, 3]) for t in ENG_TEACHERS for a in (False, True)]

    def _post_due(self, p):
        """Срок Д/З из рассылки по расписанию этого студента: «к следующей паре» считаем от момента рассылки."""
        if not p.get("spec") or not p.get("code"):
            return p["due"]
        prof = dict(self.profile, subgroup=p["subgroup"]) if p.get("subgroup") else self.profile  # рассылка подгруппе
        r = Backend(dt.datetime.fromisoformat(p["created"]), prof)._homework_due(p["code"], p["subject"], p["spec"])
        return r.get("due", p["due"])

    def group_add_event(self, title, date, time=None, place=None, subgroup=None):
        err = self._starosta(subgroup)
        if err:
            return err
        try:
            try:
                d = self.parse_date(date)
            except (ValueError, StopIteration):
                d = self.parse_due(date, future=True)  # tomorrow, friday
        except (ValueError, StopIteration, AttributeError):
            return {"error": "bad_date"}
        if time:
            try:
                time = dt.time.fromisoformat(time).strftime("%H:%M")
            except ValueError:
                return {"error": "bad_time"}
        left = (d - self.now.date()).days
        if left < 0 or (left == 0 and time and time <= self.now.strftime("%H:%M")):
            return {"error": "date_in_past"}
        data = {"title": title, "date": d.isoformat()}
        data.update({k: v for k, v in (("time", time), ("place", place)) if v})
        full = self._group_full("event")
        if full:
            return full
        sent = self._post("event", data, f"{d} {time or '23:59'}", subgroup)  # после начала не показываем
        return {"ok": True} | data | {"days_left": left} | sent

    def group_create_reminder(self, text, in_minutes=None, in_hours=None, in_days=None, at=None, subgroup=None):
        err = self._starosta(subgroup)
        if err:
            return err
        if not isinstance(text, str) or not text.strip():
            return {"error": "empty_text"}
        if in_minutes or in_hours or in_days:
            fire = self.now + dt.timedelta(minutes=in_minutes or 0, hours=in_hours or 0, days=in_days or 0)
        elif at:
            try:
                fire = self.parse_at(at)
            except ValueError:
                return {"error": "bad_time"}
        else:
            return {"error": "time_required"}
        if fire <= self.now:
            return {"error": "time_in_past"}
        full = self._group_full("reminder")
        if full:
            return full
        sent = self._post("reminder", {"text": text, "fire_at": self._fmt(fire)}, self._fmt(fire), subgroup)
        return {"ok": True, "fire_at": self._fmt(fire)} | sent

    def _group_records(self, kind=None):
        """Действующие рассылки группе (всем подгруппам) для старосты: [(рассылка, что показать, ссылка)]."""
        out = []
        for p in self.store.posts(self.profile["group"], after=self._fmt(self.now), kind=kind):
            if self._blocked(p):
                continue
            o = {} if kind else {"kind": p["kind"]}
            o.update({k: p[k] for k in POST_SHOW[p["kind"]] if p.get(k)})
            if p.get("subgroup"):
                o["subgroup"] = p["subgroup"]
            out.append((p, o, f"post:{p['id']}"))
        return out

    def _group_full(self, kind):
        """Ошибка limit со списком, если у группы уже столько действующих рассылок этого вида, сколько можно."""
        recs = self._group_records(kind)
        cap = max(self.group_limits[kind], self._group_had.get(kind, 0))  # разослано до лимитов больше — как с заметками
        if len(recs) >= cap:
            return {"error": "limit", "limit": cap, "items": self._numbered("group:" + kind, recs, ask=True)}
        return None

    def group_delete_post(self, kind=None, numbers=None, query=None, all=None):
        err = self._starosta(None)
        if err:
            return err
        if kind is not None and kind not in POST_KINDS:
            return {"error": "bad_kind", "kinds": list(POST_KINDS)}
        def drop(chosen):  # из базы: пропадёт у всей группы
            self.store.delete_posts(self.profile["group"], [p["id"] for p, _, _ in chosen])
            return [o for _, o, _ in chosen], []
        return self._delete("group:" + (kind or "all"), self._group_records(kind), numbers, query, all, drop,
                            lambda: self._group_records(kind))

    def call(self, name, args):
        args = _clean(args)
        if name not in DELETE_TOOLS:
            self.pending = {}  # «удалить все» подтверждается только следующим же вызовом удаления
        before = json.dumps(self._data(), ensure_ascii=False, sort_keys=True)
        res = getattr(self, name)(**args)
        if json.dumps(self._data(), ensure_ascii=False, sort_keys=True) != before:  # сохраняем, только если что-то поменялось
            self.save()
        return res
