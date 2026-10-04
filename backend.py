"""Эталонная реализация инструментов поверх db/*.json.
Нужна генератору датасета (ответы инструментов — настоящие) и как основа будущего бэкенда.
Изменяемые данные (Д/З, заметки, напоминания, дедлайны, экзамены) — в памяти; на проде — PostgreSQL."""
import datetime as dt, json, pathlib

DB = pathlib.Path(__file__).parent / "db"
load = lambda n: json.loads((DB / n).read_text(encoding="utf-8"))
LESSONS, SUBJECTS, BUILDINGS = load("lessons.json"), load("subjects.json"), load("buildings.json")
ROUTES, WALK, FOOD, CAL = load("routes.json"), load("walk.json"), load("food.json"), load("calendar.json")
ADAPT = {"rus_a", "math_a", "eng_a"}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WD_RU = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
EAT_MINUTES = 15  # сколько нужно на сам перекус


def week_parity(d: dt.date) -> str:
    anchor = dt.date.fromisoformat(CAL["odd_week_monday"])
    return "odd" if ((d - anchor).days // 7) % 2 == 0 else "even"


def resolve_subject(text: str):
    t = text.lower().strip()
    for code, s in SUBJECTS.items():
        if t == code or t == s["name"].lower() or t in (a.lower() for a in s["aliases"]):
            return code
    for code, s in SUBJECTS.items():  # частичное совпадение
        if t in s["name"].lower() or any(t in a.lower() or a.lower() in t for a in s["aliases"]):
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
    def __init__(self, now: dt.datetime, profile: dict):
        self.now, self.profile = now, profile  # profile: subgroup, english_teacher, adaptation
        self.homework, self.notes, self.reminders, self.deadlines, self.exams = [], [], [], [], []

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
            if subject and resolve_subject(subject) != e["code"]:
                continue
            items.append({k: v for k, v in e.items() if k != "code"} |
                         {"days_left": (dt.date.fromisoformat(e["date"]) - self.now.date()).days})
        return {"items": items}

    def add_exam(self, subject, date, kind="экзамен", time=None, room=None, building=None):
        code = resolve_subject(subject)
        e = {"code": code, "subject": SUBJECTS[code]["name"] if code else subject, "kind": kind, "date": date}
        for k, v in (("time", time), ("room", room), ("building", building)):
            if v:
                e[k] = v
        self.exams.append(e)
        return {"ok": True, "subject": e["subject"], "date": date,
                "days_left": (dt.date.fromisoformat(date) - self.now.date()).days}

    def get_daily_brief(self, day="today"):
        s = self.get_schedule(day)
        ls = self.lessons_on(self.parse_day(day))
        for i in range(1, len(ls)):
            a, b = ls[i - 1], ls[i]
            if a["building"] and b["building"] and a["building"] != b["building"]:
                s["lessons"][i]["walk_from_prev_minutes"] = walk_minutes(a["building"], b["building"])
        s["homework"], s["deadlines"], s["exams"] = [], [], []
        return s

    def call(self, name, args):
        return getattr(self, name)(**args)
