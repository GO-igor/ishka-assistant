"""Эталонная реализация инструментов поверх db/*.json.
Нужна генератору датасета (ответы инструментов — настоящие) и как основа будущего бэкенда.
Изменяемые данные (Д/З, заметки, напоминания, дедлайны, экзамены) — в памяти; если студент вошёл в аккаунт
(store + account_id), они сохраняются в db/ishka.sqlite (store.py). Там же рассылки старосты группе."""
import datetime as dt, json, pathlib, re

DB = pathlib.Path(__file__).parent / "db"
load = lambda n: json.loads((DB / n).read_text(encoding="utf-8"))
LESSONS, SUBJECTS, BUILDINGS = load("lessons.json"), load("subjects.json"), load("buildings.json")
ROUTES, WALK, FOOD, CAL = load("routes.json"), load("walk.json"), load("food.json"), load("calendar.json")
EVENTS = load("events.json")["events"]  # мероприятия и хакатоны; файл заполняется руками
ADAPT = {"rus_a", "math_a", "eng_a"}
HOLIDAYS = set(CAL.get("holidays", []))
ENG_TEACHERS = sorted({l["teacher"] for l in LESSONS if l["subject"] == "eng" and l["teacher"]})
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WD_RU = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
EAT_MINUTES = 15  # сколько нужно на сам перекус
EVENT_KINDS = ["хакатон", "конференция", "олимпиада", "конкурс", "мастер-класс"]
EVENT_PERIODS = {"week": 7, "month": 31, "all": None}  # сколько дней вперёд смотреть
SAVE_AFTER = {"add_homework", "create_reminder", "add_note", "add_deadline", "add_exam", "set_daily_brief"}
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


class Backend:
    def __init__(self, now: dt.datetime, profile: dict, store=None, account_id=None):
        # profile: subgroup, english_teacher, adaptation; у вошедших в аккаунт ещё group и role (student|starosta)
        self.now, self.profile = now, profile
        self.store, self.account_id = store, account_id  # store.Store; без них всё живёт до конца сеанса
        self.homework, self.notes, self.reminders, self.deadlines, self.exams = [], [], [], [], []
        self.brief = {"enabled": True, "time": "07:30"}
        self.events = EVENTS
        if store is not None and account_id is not None:
            data = store.load_data(account_id)
            for k in ("homework", "notes", "reminders", "deadlines", "exams"):
                setattr(self, k, data.get(k, []))
            self.brief = data.get("brief", self.brief)

    def save(self):
        if self.store is not None and self.account_id is not None:
            self.store.save_data(self.account_id, {"homework": self.homework, "notes": self.notes,
                                                   "reminders": self.reminders, "deadlines": self.deadlines,
                                                   "exams": self.exams, "brief": self.brief})

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

    def get_exams(self, subject=None):
        items = []
        for e in self.exams:
            if subject and not self._same_subject(subject, e):
                continue
            left = (dt.date.fromisoformat(e["date"]) - self.now.date()).days
            if left < 0:
                continue
            items.append({k: v for k, v in e.items() if k not in ("code", "reminders")} | {"days_left": left})
        return {"items": sorted(items, key=lambda e: (e["date"], e.get("time", "")))}

    def _same_subject(self, subject, item):
        code = resolve_subject(subject)
        return item.get("code") == code if code else item["subject"].lower() == subject.strip().lower()

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

    def list_homework(self, subject=None, due=None):
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
        return {"items": [{k: h[k] for k in ("subject", "task", "due", "from_starosta") if k in h}
                          for h in sorted(items, key=lambda h: h["due"])]}

    # ---------- напоминания и заметки ----------
    def create_reminder(self, text, in_minutes=None, in_hours=None, in_days=None, at=None):
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
        self.reminders.append({"text": text, "fire_at": self._fmt(fire)})
        return {"ok": True, "fire_at": self._fmt(fire)}

    def add_note(self, text):
        self.notes.append(text)
        return {"ok": True, "id": len(self.notes)}

    def list_notes(self, query=None):
        """Поиск по основам слов: «вайфай» находит «вайфая», «староста» — «старосты»."""
        cut = lambda w: w if len(w) <= 4 else w[:len(w) - (1 if len(w) <= 6 else 2)]  # «проект» → «проек», не «прое»
        stems = [cut(w) for w in re.findall(r"\w+", (query or "").lower().replace("ё", "е"))]
        def hit(text):
            words = re.findall(r"\w+", text.lower().replace("ё", "е"))
            return all(any(w.startswith(s) for w in words) for s in stems)
        now_s = self._fmt(self.now)
        group = [{"text": p["text"], "fire_at": p["fire_at"], "from_starosta": True} for p in self._posts("reminder")]
        return {"notes": [n for n in self.notes if hit(n)],
                "reminders": sorted((r for r in self.reminders + group if r["fire_at"] > now_s and hit(r["text"])),
                                    key=lambda r: r["fire_at"])}

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

    def list_deadlines(self, subject=None, kind=None):
        if subject and not resolve_subject(subject) and not any(self._same_subject(subject, x) for x in self.deadlines):
            return {"error": "subject_not_found", "subject": subject}
        out = []
        for x in self.deadlines:
            left = (dt.date.fromisoformat(x["due"]) - self.now.date()).days
            if left < 0 or (kind and x["kind"] != kind) or (subject and not self._same_subject(subject, x)):
                continue
            out.append({"title": x["title"], "subject": x["subject"], "due": x["due"], "days_left": left})
        return {"items": sorted(out, key=lambda x: x["days_left"])}

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
        s["homework"] = [h for h in self.list_homework(due=d.isoformat())["items"] if not h.get("from_starosta")]
        s["deadlines"] = [x for x in self.list_deadlines()["items"] if x["days_left"] <= 7]
        s["exams"] = [x for x in self.get_exams()["items"] if 0 <= x["days_left"] <= 14]
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
    def get_events(self, kind=None, period="month"):
        today, limit = self.now.date(), EVENT_PERIODS.get(period, 31)
        group = [{"title": p["title"], "date": p["date"], "time": p.get("time"), "place": p.get("place"),
                  "from_starosta": True} for p in self._posts("event")]
        items = []
        for e in self.events + group:
            left = (dt.date.fromisoformat(e["date"]) - today).days
            if left < 0 or (limit is not None and left > limit) or self._started(e, left):
                continue
            if kind and e.get("kind") != kind and kind.lower()[:6] not in e["title"].lower():
                continue
            o = {k: e[k] for k in ("title", "kind", "date", "time", "place", "description", "url") if e.get(k)}
            o["days_left"] = left
            if e.get("from_starosta"):
                o["from_starosta"] = True
            items.append(o)
        items.sort(key=lambda o: (o["date"], o.get("time") or ""))
        out = {"items": items[:10]}
        if len(items) > 10:
            out["more"] = len(items) - 10
        return out

    def _started(self, e, left):
        """Мероприятие сегодня, и время начала уже прошло (как в group_add_event: такое не показываем)."""
        return left == 0 and bool(e.get("time")) and e["time"] <= self.now.strftime("%H:%M")

    # ---------- рассылки старосты ----------
    def _posts(self, kind=None):
        """Актуальные рассылки старосты для группы и подгруппы студента (если он вошёл в аккаунт)."""
        if self.store is None or not self.profile.get("group"):
            return []
        return self.store.posts(self.profile["group"], self.profile.get("subgroup"), after=self._fmt(self.now), kind=kind)

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
        sent = self._post("event", data, f"{d} {time or '23:59'}", subgroup)  # после начала не показываем
        return {"ok": True} | data | {"days_left": left} | sent

    def group_create_reminder(self, text, in_minutes=None, in_hours=None, in_days=None, at=None, subgroup=None):
        err = self._starosta(subgroup)
        if err:
            return err
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
        sent = self._post("reminder", {"text": text, "fire_at": self._fmt(fire)}, self._fmt(fire), subgroup)
        return {"ok": True, "fire_at": self._fmt(fire)} | sent

    def call(self, name, args):
        res = getattr(self, name)(**args)
        if name in SAVE_AFTER:
            self.save()
        return res
