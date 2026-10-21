"""Справочные инструменты: расписание, аудитории, преподаватели, дорога, перекус, мероприятия ИШИТР, сессия.
Только читают db/*.json (их собирает db/build_db.py; events.json и session.json заполняются руками)."""
import datetime as dt, json, pathlib, re, unicodedata
from dates import WEEKDAYS, parse_day
from guard import Guard

DB = pathlib.Path(__file__).parent / "db"
load = lambda n: json.loads((DB / n).read_text(encoding="utf-8"))
LESSONS, SUBJECTS, BUILDINGS = load("lessons.json"), load("subjects.json"), load("buildings.json")
ROUTES, WALK, FOOD, CAL = load("routes.json"), load("walk.json"), load("food.json"), load("calendar.json")
EVENTS = load("events.json")["events"]  # мероприятия и хакатоны
SESSION = load("session.json")  # даты сессии и экзаменов
ADAPT = {"rus_a", "math_a", "eng_a"}
GUARD = Guard()  # запретные фразы: такие мероприятия модели не показываем
HOLIDAYS = set(CAL.get("holidays", []))
ENG_TEACHERS = sorted({l["teacher"] for l in LESSONS if l["subject"] == "eng" and l["teacher"]})
WD_RU = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
WD_SHORT_RU = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
EAT_MINUTES = 15  # сколько нужно на сам перекус
EVENT_KINDS = ["хакатон", "конференция", "олимпиада", "конкурс", "мастер-класс"]
EVENT_PERIODS = {"week": 7, "month": 31, "all": None}  # сколько дней вперёд смотреть


def week_parity(d: dt.date) -> str:
    anchor = dt.date.fromisoformat(CAL["odd_week_monday"])
    return "odd" if ((d - anchor).days // 7) % 2 == 0 else "even"


def resolve_subject(text):
    t = str(text or "").lower().strip()
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
    # частичное совпадение; курсы (А) — первыми, если в запросе есть «(А)», иначе последними («английский язык» —
    # не «Английский язык (А)»); короткие алиасы («ИЯ», «вит») — только целым словом
    for code in sorted(SUBJECTS, key=lambda c: c.endswith("_a") != ("(а)" in t)):
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
        if t == code.lower() or t in b["aliases"] or str(text).lower() in b["aliases"] + [b["name"].lower()]:
            return code
    # «в 10 корпусе», «10-й корпус», «Учебный корпус № 8»: номер — только если кроме него в тексте одни «в», «учебный»,
    # «корпус», «№», «-й» («общежитие 8», «остановка 10», «8 этаж» — не корпус)
    raw = str(text).lower().replace("ё", "е")
    m = re.fullmatch(r"\s*((в|во|из|от|с|со|до|к|ко|у|около|возле)\s+)?(учебн\w*\s+)?(корп\w*\.?\s*)?№?\s*([0-9]+)\s*"
                     r"(-?\s*(й|ый|ой|ий|м|ом|ым|го|ого|му|ому|к|ка|ке|ку|кой|ки))?\s*(корп\w*\.?)?\s*", raw)
    if m:
        return m.group(5) if m.group(5) in BUILDINGS else None
    if re.search(r"[0-9]", raw):
        return None
    # «в главном корпусе», «десятом», «кибернетический центр»: слово — та же основа, что у алиаса, с другим окончанием
    # («девятом» — не «девятнадцатый», «киберспорт» — не «кибер центр»)
    words = [w for w in re.findall(r"[а-яa-z]+", raw) if not w.startswith(("учебн", "корп"))]
    words += [a + b for a, b in zip(words, words[1:])]  # «киберцентр» — это «кибер центр»

    def same(w, a):
        a = a.replace(" ", "") if w.startswith(a.replace(" ", "")[:6]) and " " in a else a.split()[0]
        if len(a) <= 3:
            return w == a
        stem = a[:-2] if len(a) > 5 else a
        return w.startswith(stem) and len(w) <= len(a) + 2

    hits = {code for code, b in BUILDINGS.items() for a in b["aliases"] + [b["name"].lower()] for w in words if same(w, a)}
    return hits.pop() if len(hits) == 1 else None


def english_teacher(name):
    """«Аксёнова», «аксеновой» → «Аксёнова Н. В.»; не нашёлся или подходят несколько — None."""
    key = unicodedata.normalize("NFC", str(name or "")).strip().lower().replace("ё", "е")[:5]
    hits = [t for t in ENG_TEACHERS if t.lower().replace("ё", "е").startswith(key)] if key else []
    return hits[0] if len(hits) == 1 else None


def walk_minutes(a, b):
    if a == b:
        return 0
    for w in WALK:
        if {w["a"], w["b"]} == {a, b}:
            return w["minutes"]
    return None


def lesson_days(ls):
    """В какие дни пары: ['вт по чётным', 'пт'] — день без пометки, если пара в обе недели."""
    weeks = {}
    for l in ls:
        weeks.setdefault(l["weekday"], set()).add(l["week"])
    return [WD_SHORT_RU[d - 1] + ("" if len(w) == 2 else " по чётным" if "even" in w else " по нечётным")
            for d, w in sorted(weeks.items())]


def route_to(b, room):
    """Маршрут до аудитории в корпусе b: {"route"} или, если его нет, {"route": None, "floor_guess"} по номеру."""
    exact = [r for r in ROUTES if r["building"] == b and r["room"] == room]
    rule = [r for r in ROUTES if r["building"] == b and r["room"] == "*"]
    if exact or rule:
        return {"route": (exact or rule)[0]["route"]}
    digit = next((c for c in room if c.isdigit()), None)
    return {"route": None, "floor_guess": int(digit) if digit else None}


class Campus:
    """Инструменты, которым нужны только время (self.now) и профиль (self.profile: subgroup, english_teacher,
    adaptation). self.events и self.session можно подменить (генератор датасета, тесты)."""

    def lessons_on(self, d: dt.date):
        p = self.profile
        if d.isoformat() in HOLIDAYS:
            return []
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

    def _lesson_out(self, l):
        o = {"time": f"{l['start']}-{l['end']}", "subject": SUBJECTS[l["subject"]]["name"], "type": l["type"]}
        if l["building"]:
            o.update(building=l["building"], room=l["room"])
        if l["teacher"]:
            o["teacher"] = l["teacher"]
        if l["subgroup"]:
            o["subgroup"] = l["subgroup"]
        return o

    def _at(self, d, hhmm):
        return dt.datetime.combine(d, dt.time.fromisoformat(hhmm))

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

    def next_class_of(self, code, after=None):
        """Начало ближайшей пары по предмету после момента after (по умолчанию — сейчас) в пределах двух недель."""
        after = after or self.now
        for add in range(0, 15):
            d = after.date() + dt.timedelta(days=add)
            for l in self.lessons_on(d):
                if l["subject"] == code and self._at(d, l["start"]) > after:
                    return self._at(d, l["start"])
        return None

    # ---------- инструменты ----------
    def get_schedule(self, day="today"):
        try:
            d = parse_day(self.now, day)
        except ValueError:
            return {"error": "bad_date"}
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
        if not p.get("english_teacher") and any(l["subject"] == "eng" for l in self.lessons_on(d)):
            out["note"] = "английский показан для всех групп: преподаватель студента не указан в профиле"
        if not p.get("subgroup"):
            out["note_subgroup"] = "подгруппа не указана, лабы показаны для всех подгрупп"
        return out

    def find_room(self, room=None, building=None, from_building=None, subject=None):
        if room and not any(c.isdigit() for c in room):  # «кабинет английского» в room: аудиторий без номера нет
            if subject or resolve_subject(room):
                return self.subject_rooms(subject or room, building, from_building)
            if resolve_building(building) is None:
                return {"error": "room_not_found"}
        if subject and not room:
            return self.subject_rooms(subject, building, from_building)
        b = resolve_building(building)
        if b is None and room:  # ищем аудиторию в расписании
            cands = {l["building"] for l in LESSONS if l["room"] == room}
            if len(cands) == 1:
                b = cands.pop()
            elif len(cands) > 1:
                return {"error": "ambiguous_room", "buildings": sorted(cands)}
        if b is None:  # аудиторию не нашли: если назван предмет — ищем по нему
            return self.subject_rooms(subject, building, from_building) if subject else {"error": "room_not_found"}
        out = {"building": b, "building_name": BUILDINGS[b]["name"], "address": BUILDINGS[b]["address"]}
        if room:
            out["room"] = room
            out |= route_to(b, room)
        fb = resolve_building(from_building)
        if fb:
            out["walk_minutes"] = walk_minutes(fb, b)
        return out

    def subject_rooms(self, subject, building=None, from_building=None):
        """find_room с предметом вместо аудитории: где проходят пары этого предмета у студента (его подгруппа и
        преподаватель английского из профиля), как туда дойти и когда ближайшая пара. building — только в этом корпусе."""
        code = resolve_subject(subject)
        if not code:
            return {"error": "subject_not_found"}
        p, name = self.profile, SUBJECTS[code]["name"]
        mine = [l for l in LESSONS if l["subject"] == code and l["building"]
                and not (l["subgroup"] and p.get("subgroup") and l["subgroup"] != p["subgroup"])
                and not (code == "eng" and p.get("english_teacher") and l["teacher"] != p["english_teacher"])]
        if not mine:
            return {"subject": name, "error": "no_room"}  # физра: аудитории в расписании нет
        out = {"subject": name}
        b = resolve_building(building)
        if building and b is None:
            out["building_not_found"] = str(building)  # корпус не понял — показываем все аудитории
        elif b and not any(l["building"] == b for l in mine):
            out["not_in_building"] = BUILDINGS[b]["name"]  # в этом корпусе пар нет — показываем, где есть
        elif b:
            mine = [l for l in mine if l["building"] == b]
        by_room = {}
        for l in sorted(mine, key=lambda l: (("ЛК", "ПР", "ЛБ").index(l["type"]), l["building"], l["subgroup"] or 0, l["room"])):
            by_room.setdefault((l["building"], l["room"]), []).append(l)
        fb, rooms = resolve_building(from_building), []
        for (bc, room), ls in by_room.items():
            r = {"building": bc, "building_name": BUILDINGS[bc]["name"], "address": BUILDINGS[bc]["address"], "room": room,
                 "types": list(dict.fromkeys(l["type"] for l in ls)), "days": lesson_days(ls)}
            if code == "eng" and not p.get("english_teacher"):
                r["teacher"] = ", ".join(sorted({l["teacher"] for l in ls if l["teacher"]}))
            if not p.get("subgroup") and any(l["subgroup"] for l in ls):  # подгруппа не указана: чьи это лабы
                r["subgroups"] = sorted({l["subgroup"] for l in ls if l["subgroup"]})
            r |= route_to(bc, room)
            if fb:
                r["walk_minutes"] = walk_minutes(fb, bc)
            rooms.append(r)
        out["rooms"] = rooms
        nxt = self.next_of(code)
        if nxt:
            out["next"] = nxt
        if code in ADAPT and not p.get("adaptation"):
            out["note_adaptation"] = "курсы (А) не отмечены в профиле: показаны аудитории для тех, кто на них ходит"
        if code == "eng" and not p.get("english_teacher"):
            out["note"] = "преподаватель английского не указан в профиле: показаны аудитории всех групп"
        if any("subgroups" in r for r in rooms):
            out["note_subgroup"] = "подгруппа не указана, лабы показаны для всех подгрупп"
        return out

    def next_of(self, code):
        """Ближайшая пара предмета в расписании студента (две недели вперёд): {"date", "time", "type", "building", "room"}."""
        for add in range(0, 15):
            d = self.now.date() + dt.timedelta(days=add)
            for l in self.lessons_on(d):
                if l["subject"] == code and self._at(d, l["start"]) > self.now:
                    o = {"date": d.isoformat(), "time": l["start"], "type": l["type"]}
                    if l["building"]:
                        o |= {"building": l["building"], "room": l["room"]}
                    if l["subgroup"] and not self.profile.get("subgroup"):  # чья это пара, если профиль не знает
                        o["subgroup"] = l["subgroup"]
                    if code == "eng" and not self.profile.get("english_teacher"):
                        o["teacher"] = l["teacher"]
                    return o
        return None

    def get_teacher(self, subject, lesson_type="любой"):
        code = resolve_subject(subject)
        if not code:
            return {"error": "subject_not_found"}
        names = {"ЛК": "лекция", "ПР": "практика", "ЛБ": "лабораторная"}
        by_type = {}
        for l in LESSONS:
            if l["subject"] == code and l["teacher"] and lesson_type in ("любой", None, names[l["type"]]):
                if code == "eng" and self.profile.get("english_teacher") and l["teacher"] != self.profile["english_teacher"]:
                    continue
                key = {"ЛК": "лекции", "ПР": "практики", "ЛБ": "лабораторные"}[l["type"]]
                by_type.setdefault(key, set()).add(l["teacher"])
        return {"subject": SUBJECTS[code]["name"], "teachers": {k: sorted(v) for k, v in by_type.items()}}

    def check_travel_time(self, from_building, to_room=None, to_building=None):
        fb = resolve_building(from_building)
        d, l = self.next_lesson()
        if to_building or to_room:
            tb = self.find_room(to_room, to_building).get("building") or resolve_building(to_building)
            target, start = {"room": to_room, "building": tb}, None
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

    def get_events(self, kind=None, period="month"):
        """Мероприятия и хакатоны для студентов ИШИТР (db/events.json), которые ещё не начались, по дате; не больше 10."""
        limit = EVENT_PERIODS.get(period, 31)
        today, items = self.now.date(), []
        for e in self.events:
            if GUARD.check_obj(e):  # мероприятие с запретной фразой (список могли пополнить позже) не показываем
                continue
            left = (dt.date.fromisoformat(e["date"]) - today).days
            started = left == 0 and bool(e.get("time")) and e["time"] <= self.now.strftime("%H:%M")
            if left < 0 or started or (limit is not None and left > limit):
                continue
            if kind and e.get("kind") != kind and str(kind).lower()[:6] not in e["title"].lower():
                continue
            o = {k: e[k] for k in ("title", "kind", "date", "time", "place", "description", "url") if e.get(k)}
            items.append(o | {"days_left": left})
        items.sort(key=lambda o: (o["date"], o.get("time") or ""))
        out = {"items": items[:10]}
        if len(items) > 10:
            out["more"] = len(items) - 10
        return out

    def get_session(self, subject=None):
        """Сколько дней до сессии и экзамены (db/session.json). Дата экзамена null — ещё не объявили."""
        s, today = self.session, self.now.date()
        start, end = dt.date.fromisoformat(s["start"]), dt.date.fromisoformat(s["end"])
        out = {"start": s["start"], "end": s["end"]}
        if today > end:
            out["status"] = "over"
        elif today >= start:
            out["status"] = "now"
        else:
            out |= {"status": "soon", "days_left": (start - today).days}
        exams = s.get("exams", [])
        if subject:
            code = resolve_subject(subject)
            exams = [e for e in exams if e["subject"] == code]
            if not exams:
                return out | {"error": "no_exam", "subject": SUBJECTS[code]["name"] if code else subject}
        out["exams"] = []
        for e in sorted(exams, key=lambda e: (e.get("date") or "9999", e.get("time") or "")):
            o = {"subject": SUBJECTS[e["subject"]]["name"], "kind": e.get("kind", "экзамен"), "date": e.get("date")}
            o.update({k: e[k] for k in ("time", "building", "room") if e.get(k)})
            if e.get("date"):
                left = (dt.date.fromisoformat(e["date"]) - today).days
                if left < 0:
                    continue  # уже сдали
                o["days_left"] = left
            out["exams"].append(o)
        return out
