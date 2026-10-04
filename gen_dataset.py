"""Генерация train/eval из реальной базы: вопросы по шаблонам -> настоящий вызов backend -> ответ по шаблону.
python gen_dataset.py  ->  data/train.jsonl, data/eval.jsonl
Формулировки делятся на train и eval (последние 2 в каждом списке — только eval), чтобы eval проверял обобщение.
Перед обучением ответы стоит перефразировать большой LLM и выборочно проверить руками."""
import datetime as dt, json, random, pathlib
import paraphrase
from backend import Backend, SUBJECTS, BUILDINGS, WD_RU, WEEKDAYS

rnd = random.Random(42)
TOOLS = {t["function"]["name"] for t in json.load(open(pathlib.Path(__file__).parent / "tools.json", encoding="utf-8"))}
ENG = ["Аксёнова Н. В.", "Ануфриева Т. Н.", "Пичугова И. Л.", "Ростовцева В. М."]
SEM_START, SEM_END = dt.date(2026, 9, 28), dt.date(2026, 12, 26)
SHORT = {"Математическая логика и теория алгоритмов": "матлогика", "Информатика 1.2": "информатика",
         "Математика 1.3 ИТ": "математика", "Математика (А)": "математика (А)",
         "Основы программирования на Python": "питон", "Иностранный язык (английский)": "английский",
         "Английский язык (А)": "АЯ (А)", "Русский язык (А)": "русский (А)", "История России": "история",
         "Введение в ИТ": "введение в ИТ", "Введение в проектную деятельность": "проектная деятельность",
         "Элективные дисциплины по физической культуре и спорту": "физра", "«Код ТПУ»": "«Код ТПУ»"}
TYPE = {"ЛК": "лекция", "ПР": "практика", "ЛБ": "лаба"}
WD_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
WD_ACC = ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"]
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]


def pick(xs):
    return rnd.choice(xs)


def rand_profile():
    if rnd.random() < 0.12:
        return {}
    return {"subgroup": rnd.randint(1, 3), "english_teacher": pick(ENG), "adaptation": rnd.random() < 0.3}


def profile_text(p):
    if not p:
        return "не заполнен"
    return (f"подгруппа {p['subgroup']}, английский — {p['english_teacher']}, "
            f"курсы (А): {'да' if p['adaptation'] else 'нет'}")


def system(now, p):
    return {"role": "system", "content":
            "Ты — Ишка, помощник студентов ИШИТР ТПУ. Общайся на «ты», коротко, дружелюбно, можно лёгкий юмор и "
            "эмодзи, без канцелярита. Факты о расписании, корпусах, преподавателях, Д/З, экзаменах и напоминаниях бери "
            "ТОЛЬКО из инструментов, ничего не выдумывай. "
            f"Сейчас: {WD_RU[now.weekday()]}, {now:%Y-%m-%d %H:%M}. Профиль студента: {profile_text(p)}."}


def rand_now(hours=(7, 21)):
    d = SEM_START + dt.timedelta(days=rnd.randrange((SEM_END - SEM_START).days))
    if d.weekday() == 6 and rnd.random() < 0.7:
        d -= dt.timedelta(days=1)
    return dt.datetime.combine(d, dt.time(rnd.randint(*hours), rnd.choice([0, 5, 10, 15, 20, 30, 40, 45, 50])))


def tc(name, args):
    assert name in TOOLS, name
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


def tr(name, obj):
    return {"role": "tool", "name": name, "content": json.dumps(obj, ensure_ascii=False)}


def where(l):
    if not l.get("building"):
        return ""
    b = "ГК" if l["building"] == "ГК" else ("КЦ" if l["building"] == "КЦ" else f"{l['building']} корпус")
    return f", {b}, ауд. {l['room']}"


def lesson_line(i, l, with_teacher=True):
    t = f" ({l['teacher']})" if with_teacher and l.get("teacher") else ""
    return f"{i}. {l['time'][:5]} — {SHORT[l['subject']]}, {TYPE[l['type']]}{where(l)}{t}"


def date_ru(d):
    return f"{d.day} {MONTHS[d.month - 1]}"


# ---------------- категории ----------------
# Каждая функция: (split) -> list[messages]. phr[split] — формулировки.
def split_phr(lst, split):
    return lst[:-2] if split == "train" else lst[-2:]


def cat_schedule(split):
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)
    kind = pick(["today", "tomorrow", "weekday"])
    if kind == "weekday":
        wd = rnd.randrange(6)
        q = pick(split_phr(["что у меня в {d}?", "пары в {d} какие", "расписание на {d} скинь", "чё в {d} по парам",
                            "во сколько начинаются пары в {d}", "какие пары будут в {d}", "в {d} много пар?",
                            "покажи {d}", "есть пары в {d}?"], split)).format(d=WD_ACC[wd])
        day = WEEKDAYS[wd]
    else:
        q = pick(split_phr({"today": ["какие пары сегодня", "че сегодня по парам", "расписание на сегодня",
                                      "сегодня что за пары?", "скинь пары на сегодня пж", "у меня сегодня есть пары?",
                                      "что сегодня", "сегодня во сколько к первой?"],
                            "tomorrow": ["что завтра", "пары на завтра", "чё у нас завтра по парам",
                                         "завтра к какой паре?", "завтра много пар?", "какое расписание завтра",
                                         "завтра пары есть?", "во сколько завтра вставать на пары"]}[kind], split))
        day = kind
    res = b.get_schedule(day)
    d = dt.date.fromisoformat(res["date"])
    head = {"today": "Сегодня", "tomorrow": "Завтра"}.get(kind, f"В {WD_ACC[d.weekday()]}") + f" ({WD_SHORT[d.weekday()]} {date_ru(d)}, {res['week']} неделя)"
    ls = res["lessons"]
    if not ls:
        ans = f"{head} пар нет 🎉 " + pick(["Можно выспаться.", "Отдыхай!", "Свободный день, кайф."])
    else:
        lines = "\n".join(lesson_line(i + 1, l) for i, l in enumerate(ls))
        ans = f"{head}:\n{lines}"
        if "note" in res or "note_subgroup" in res:
            ans += "\n\nЯ показал варианты для всех групп: скажи свою подгруппу и преподавателя по английскому — буду показывать только твои пары."
        elif len({l.get('building') for l in ls if l.get('building')}) > 1:
            ans += "\n\n" + pick(["Будут переходы между корпусами, закладывай время 🚶", "Корпуса разные, не залипай на переменах 😉", ""])
        ans = ans.rstrip()
    return [system(now, p), {"role": "user", "content": q}, tc("get_schedule", {"day": day}), tr("get_schedule", res),
            {"role": "assistant", "content": ans}]


ROOMS = sorted({(l["building"], l["room"]) for l in __import__("backend").LESSONS if l["building"]})


def cat_room(split):
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)
    bld, room = pick(ROOMS)
    with_b = rnd.random() < 0.5
    from_b = pick([None, None, "ГК", "КЦ", "19", "10"])
    bname = {"ГК": "главном корпусе", "КЦ": "КЦ", "МКЦ": "МКЦ"}.get(bld, f"{bld} корпусе")
    q = pick(split_phr(["как дойти до {r}{b}", "где ауд {r}{b}?", "где находится {r}{b}", "как найти {r}{b}",
                        "{r}{b} это где вообще", "подскажи дорогу до {r}{b}", "не могу найти {r}{b} помоги",
                        "где кабинет {r}{b}"], split)).format(r=room, b=f" в {bname}" if with_b else "")
    if from_b and from_b != bld:
        q += pick([f", я сейчас в {from_b if from_b in ('ГК','КЦ') else from_b + ' корпусе'}", f" (я в {from_b})"])
    else:
        from_b = None
    args = {"room": room}
    if with_b:
        args["building"] = bld
    if from_b:
        args["from_building"] = from_b
    res = b.find_room(**args)
    if "error" in res:
        ans = "Такую аудиторию у себя не нашёл 🤔 Подскажи корпус — поищу ещё раз."
        if res["error"] == "ambiguous_room":
            ans = f"Аудитория {room} есть в нескольких корпусах ({', '.join(res['buildings'])}). Тебе в какой?"
        return [system(now, p), {"role": "user", "content": q}, tc("find_room", args), tr("find_room", res), {"role": "assistant", "content": ans}]
    parts = [f"Ауд. {room} — {res['building_name']}" + (f" ({res['address']})" if res.get("address") else "") + "."]
    if from_b:
        w = res.get("walk_minutes")
        parts.append(f"От {from_b} идти примерно {w} мин 🚶" if w is not None else
                     f"Сколько идти от {from_b}, у меня пока нет данных.")
    if res.get("route"):
        parts.append(res["route"])
    elif res.get("floor_guess"):
        parts.append(f"Точного маршрута у меня пока нет, но судя по номеру — {res['floor_guess']} этаж. Там спроси у вахты, если что.")
    else:
        parts.append("Точного маршрута внутри корпуса у меня пока нет 🙃")
    return [system(now, p), {"role": "user", "content": q}, tc("find_room", args), tr("find_room", res),
            {"role": "assistant", "content": "\n".join(parts)}]


def cat_teacher(split):
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)
    code = pick([c for c in SUBJECTS if c not in ("pe",)])
    alias = pick(SUBJECTS[code]["aliases"])
    lt = pick(["любой", "любой", "лекция", "практика", "лабораторная"])
    word = {"любой": "", "лекция": "лекции ", "практика": "практику ", "лабораторная": "лабы "}[lt]
    q = pick(split_phr(["кто ведет {w}{s}", "кто у нас по {s} {w}", "препод по {s}?", "как зовут препода {w}по {s}",
                        "{s} {w}кто ведёт", "кто преподаёт {s}", "напомни кто ведет {w}{s}", "фамилия преподавателя по {s}"],
                       split)).format(w=word, s=alias).replace("  ", " ")
    args = {"subject": alias} | ({"lesson_type": lt} if lt != "любой" else {})
    res = b.get_teacher(**args)
    t = res["teachers"]
    if not t:
        ans = f"По {alias} {word.strip() or 'занятий'} в твоём расписании не нашёл 🤔"
    elif len(t) == 1 and len(next(iter(t.values()))) == 1:
        k, v = next(iter(t.items()))
        ans = f"{res['subject']}: {k} ведёт {v[0]}."
    else:
        ans = f"{res['subject']}:\n" + "\n".join(f"• {k} — {', '.join(v)}" for k, v in t.items())
    return [system(now, p), {"role": "user", "content": q}, tc("get_teacher", args), tr("get_teacher", res),
            {"role": "assistant", "content": ans}]


def lesson_now_and_next(p):
    """Подбирает момент на перемене/паре, когда следующая пара сегодня."""
    for _ in range(200):
        now = rand_now((8, 18))
        b = Backend(now, p)
        d, nl = b.next_lesson()
        cur = b.current_or_last()
        if nl and d == now.date() and cur and cur.get("building") and nl.get("building"):
            end = dt.datetime.combine(d, dt.time.fromisoformat(cur["end"]))
            now = end + dt.timedelta(minutes=rnd.randint(-10, 15))
            b = Backend(now, p)
            d2, nl2 = b.next_lesson()
            if nl2 and d2 == now.date():
                return now, b
    return None, None


def cat_travel(split):
    p = rand_profile() or {"subgroup": 1, "english_teacher": ENG[0], "adaptation": False}
    now, b = lesson_now_and_next(p)
    if not now:
        return None
    cur = b.current_or_last()
    fb = cur["building"]
    fbt = {"ГК": "в ГК", "КЦ": "в КЦ", "МКЦ": "в МКЦ"}.get(fb, f"в {fb} корпусе")
    q = pick(split_phr(["я {f}, успею на следующую?", "успеваю на след пару? я {f}", "сколько идти до следующей пары, я {f}",
                        "я {f}, не опоздаю?", "успею дойти? сейчас {f}", "я {f} а пара где следующая, успею?",
                        "реально успеть на следующую если я {f}?", "я {f}, бежать или можно спокойно идти?"], split)).format(f=fbt)
    res = b.check_travel_time(fb)
    tg, w, left = res["target"], res["walk_minutes"], res.get("minutes_left")
    place = f"{SHORT[tg['subject']]} в {tg['time'][:5]}{where(tg)}"
    v = res.get("verdict")
    if w == 0:
        ans = f"Следующая — {place}, в этом же корпусе. До начала {left} мин, расслабься 😌"
    elif v == "неизвестно":
        ans = f"Следующая — {place}. До начала {left} мин, но сколько идти от {fb}, у меня пока нет данных 🙃"
    elif v == "успеваешь":
        ans = f"Успеваешь ✅ Следующая — {place}. Идти ~{w} мин, до начала {left}."
    elif v == "впритык":
        ans = f"Впритык 😬 Идти ~{w} мин, а до начала {left}. Выходи прямо сейчас: {place}."
    else:
        ans = f"Не успеваешь 😕 Идти ~{w} мин, до начала {left}. Опоздаешь минут на {w - left}. Следующая — {place}."
    return [system(now, p), {"role": "user", "content": q}, tc("check_travel_time", {"from_building": fb}),
            tr("check_travel_time", res), {"role": "assistant", "content": ans}]


def cat_food(split):
    p = rand_profile() or {"subgroup": 2, "english_teacher": ENG[1], "adaptation": False}
    now, b = lesson_now_and_next(p)
    if not now:
        return None
    explicit = rnd.random() < 0.3
    bld = b.current_or_last()["building"]
    q = pick(split_phr(["где пожрать после пары", "есть где перекусить рядом?", "хочу есть, куда сходить",
                        "где поесть на перемене", "успею перекусить до следующей?", "куда сбегать за едой",
                        "где тут можно быстро поесть", "голодный, что рядом есть?"], split))
    args = {}
    if explicit:
        q += f" около {bld if bld in ('ГК', 'КЦ', 'МКЦ') else bld + ' корпуса'}"
        args["building"] = bld
    res = b.find_food(**args)
    if "error" in res or not res["places"]:
        ans = "Рядом с этим корпусом у меня пока нет мест для перекуса 🙃"
    else:
        pl = ", ".join(f"{x['name']} (~{x['walk_minutes']} мин" + (f", {x['where']}" if x.get("where") else "") + ")" for x in res["places"])
        ans = f"Рядом: {pl}."
        if "free_minutes" in res:
            fm = res["free_minutes"]
            if res.get("enough_time"):
                ans += f" До следующей пары {fm} мин — успеешь спокойно поесть 🍔"
            elif res.get("near_next_building", {}).get("enough_time"):
                nn = res["near_next_building"]
                ans += (f" Но до следующей пары {fm} мин, туда-обратно не успеешь. Лучше сразу иди к "
                        f"{nn['building'] if nn['building'] in ('ГК','КЦ') else nn['building'] + ' корпусу'} и перекуси там: "
                        f"{', '.join(x['name'] for x in nn['places'])}.")
            elif "enough_time" in res:
                ans += f" Но до следующей пары всего {fm} мин, нормально поесть не успеешь 😕 Возьми что-нибудь с собой."
    return [system(now, p), {"role": "user", "content": q}, tc("find_food", args), tr("find_food", res),
            {"role": "assistant", "content": ans}]


def cat_add_exam(split):
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)
    code = pick(["python", "inf", "math", "matlog"])
    alias = pick(SUBJECTS[code]["aliases"])
    d = dt.date(2027, 1, rnd.randint(9, 28))
    time = pick([None, "09:00", "10:25", "12:40"])
    q = pick(split_phr(["экзамен по {s} {d}{t}, запиши", "запиши экз по {s} на {d}{t}", "у нас {s} {d}{t}, добавь экзамен",
                        "внеси экзамен: {s}, {d}{t}", "сказали экзамен по {s} будет {d}{t}", "добавь в сессию {s} {d}{t}"],
                       split)).format(s=alias, d=date_ru(d), t=f" в {time}" if time else "")
    args = {"subject": alias, "kind": "экзамен", "date": d.isoformat()} | ({"time": time} if time else {})
    res = b.add_exam(**args)
    ans = f"Записал 📌 {res['subject']} — экзамен {date_ru(d)}{' в ' + time if time else ''}. Осталось {res['days_left']} дн. Напомню заранее, а могу и план повторения составить."
    return [system(now, p), {"role": "user", "content": q}, tc("add_exam", args), tr("add_exam", res),
            {"role": "assistant", "content": ans}]


def cat_profile(split):
    now = rand_now()
    sg, eng, ad = rnd.randint(1, 3), pick(ENG), rnd.random() < 0.3
    fam = eng.split()[0]
    q = pick(split_phr(["я во {sg} подгруппе, английский у {f}", "моя подгруппа {sg}, по англу {f}",
                        "запомни: подгруппа {sg}, английский — {f}{a}", "у меня {sg} подгруппа и {f} по английскому{a}",
                        "я из {sg} подгруппы, англ ведёт {f}", "подгруппа {sg}, англичанка {f}{a}"],
                       split)).format(sg=sg, f=fam, a=", хожу на русский (А)" if ad else "")
    args = {"subgroup": sg, "english_teacher": eng} | ({"adaptation": True} if ad else {})
    res = {"ok": True}
    ans = f"Запомнил ✍️ Подгруппа {sg}, английский — {eng}{' Курсы (А) тоже покажу.' if ad else ''} Теперь в расписании будут только твои пары."
    return [system(now, {}), {"role": "user", "content": q}, tc("set_profile", args), tr("set_profile", res),
            {"role": "assistant", "content": ans}]


CATS = [(cat_schedule, 30), (cat_room, 18), (cat_teacher, 14), (cat_travel, 12), (cat_food, 12), (cat_add_exam, 7), (cat_profile, 7)]


def build(n, split):
    out, fns, w = [], [c for c, _ in CATS], [w for _, w in CATS]
    while len(out) < n:
        m = rnd.choices(fns, w)[0](split)
        if m:
            out.append(m)
    return out


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent / "data"
    seed = [json.loads(l)["messages"] for l in open(root / "sample.jsonl", encoding="utf-8")]
    overlay = paraphrase.load()  # перефразированные ответы (только для train)
    for split, n in (("train", 900), ("eval", 100)):
        rows = build(n, split)
        if split == "train":
            print("перефразировано:", sum(paraphrase.apply(m, overlay) for m in rows))
            rows += seed  # ручные эталоны — в train
        rnd.shuffle(rows)
        with open(root / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for m in rows:
                f.write(json.dumps({"messages": m}, ensure_ascii=False) + "\n")
        print(split, len(rows))
