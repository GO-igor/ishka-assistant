"""Генерация train/eval из реальной базы: вопросы по шаблонам -> настоящий вызов backend -> ответ по шаблону.
python gen_dataset.py  ->  data/train.jsonl, data/eval.jsonl
Формулировки делятся на train и eval (последние 2 в каждом списке — только eval), чтобы eval проверял обобщение.
Перед обучением ответы стоит перефразировать большой LLM и выборочно проверить руками."""
import datetime as dt, json, random, pathlib
import paraphrase
from backend import Backend, SUBJECTS, BUILDINGS, WD_RU, WEEKDAYS, HOLIDAYS

rnd = random.Random(42)
TOOLS = {t["function"]["name"] for t in json.load(open(pathlib.Path(__file__).parent / "tools.json", encoding="utf-8"))}
ENG = ["Аксёнова Н. В.", "Ануфриева Т. Н.", "Пичугова И. Л.", "Ростовцева В. М."]
SEM_START, SEM_END = dt.date(2026, 9, 28), dt.date(2026, 12, 26)
SHORT = {"Математическая логика и теория алгоритмов": "матлогика", "Информатика 1.2": "информатика",
         "Математика 1.3 ИТ": "математика", "Математика (А)": "математика (А)",
         "Основы программирования на Python": "питон", "Иностранный язык (английский)": "английский",
         "Английский язык (А)": "английский (А)", "Русский язык (А)": "русский (А)", "История России": "история",
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
    b = f"{l['building']} корпус" if l["building"].isdigit() else l["building"]  # ГК, КЦ, МКЦ — без «корпус»
    room = f"ауд. {l['room']}" if l["room"][:1].isdigit() else l["room"][:1].lower() + l["room"][1:]  # «зал 1»
    return f", {b}, {room}"


def lesson_line(i, l, with_teacher=True):
    t = f" ({l['teacher']})" if with_teacher and l.get("teacher") else ""
    ov = " ⚠️ накладка" if l.get("overlap") else ""
    return f"{i}. {l['time'][:5]} — {SHORT[l['subject']]}, {TYPE[l['type']]}{where(l)}{t}{ov}"


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
        q = pick(split_phr(["что у меня {vd}?", "пары {vd} какие", "расписание на {d} скинь", "чё {vd} по парам",
                            "во сколько начинаются пары {vd}", "какие пары будут {vd}", "{vd} много пар?",
                            "покажи {d}", "есть пары {vd}?"], split)).format(d=WD_ACC[wd], vd=WEEKDAYS_FUT[wd])
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
    head = {"today": "Сегодня", "tomorrow": "Завтра"}.get(kind, cap(WEEKDAYS_FUT[d.weekday()])) + f" ({WD_SHORT[d.weekday()]} {date_ru(d)}, {res['week']} неделя)"
    ls = res["lessons"]
    if not ls:
        ans = f"{head} пар нет 🎉 " + pick(["Можно выспаться.", "Отдыхай!", "Свободный день, кайф."])
        if res.get("holiday"):
            ans = f"{head} нерабочий день, пар нет 🎉"
    else:
        lines = "\n".join(lesson_line(i + 1, l) for i, l in enumerate(ls))
        ans = f"{head}:\n{lines}"
        if any(l.get("overlap") for l in ls):
            ans += "\n\nДве пары в одно время — уточни у старосты, на какую идти."
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
    parts = [f"{'Ауд. ' + room if room[:1].isdigit() else room} — {res['building_name']}" + (f" ({res['address']})" if res.get("address") else "") + "."]
    if from_b:
        w = res.get("walk_minutes")
        fname = from_b if from_b in ("ГК", "КЦ", "МКЦ") else f"{from_b} корпуса"
        parts.append(f"От {fname} идти примерно {w} мин 🚶" if w is not None else
                     f"Сколько идти от {fname}, у меня пока нет данных.")
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
    tpl = pick(split_phr(["кто ведет {ws}", "кто у нас ведёт {w}по {sd}", "препод по {sd}?", "как зовут препода по {sd}{wp}",
                          "{s} {w}кто ведёт", "кто преподаёт {s}", "напомни кто ведет {ws}", "фамилия преподавателя по {sd}"], split))
    if not any(k in tpl for k in ("{w}", "{ws}", "{wp}")):  # тип занятия студент не назвал
        lt = "любой"
    word = {"любой": "", "лекция": "лекции ", "практика": "практику ", "лабораторная": "лабы "}[lt]
    wp = {"любой": "", "лекция": " на лекциях", "практика": " на практике", "лабораторная": " на лабах"}[lt]
    q = tpl.format(w=word, s=alias, sd=dat(alias), wp=wp, ws=f"{word}по {dat(alias)}" if word else alias).replace("  ", " ")
    args = {"subject": alias} | ({"lesson_type": lt} if lt != "любой" else {})
    res = b.get_teacher(**args)
    t = res["teachers"]
    if not t:
        neg = {"любой": "занятий", "лекция": "лекций", "практика": "практик", "лабораторная": "лаб"}[lt]
        ans = f"По {dat(alias)} {neg} в твоём расписании не нашёл 🤔"
    elif len(t) == 1 and len(next(iter(t.values()))) == 1:
        k, v = next(iter(t.items()))
        ans = f"{res['subject']}: {k} ведёт {v[0]}" + ("" if v[0].endswith(".") else ".")
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
    q = pick(split_phr(["экзамен по {sd} {d}{t}, запиши", "запиши экз по {sd} на {d}{t}", "у нас {s} {d}{t}, добавь экзамен",
                        "внеси экзамен: {s}, {d}{t}", "сказали экзамен по {sd} будет {d}{t}", "добавь в сессию {s} {d}{t}"],
                       split)).format(s=alias, sd=dat(alias), d=date_ru(d), t=f" в {time}" if time else "")
    args = {"subject": alias, "kind": "экзамен", "date": d.isoformat()} | ({"time": time} if time else {})
    res = b.add_exam(**args)
    n = res["days_left"]
    ans = (f"Записал 📌 {res['subject']} — экзамен {date_ru(d)}{' в ' + time if time else ''}. "
           f"{plural(n, 'Остался', 'Осталось', 'Осталось')} {days_n(n)}. Напомню {join_dates(res['reminders'])}, а могу и план повторения составить.")
    return [system(now, p), {"role": "user", "content": q}, tc("add_exam", args), tr("add_exam", res),
            {"role": "assistant", "content": ans}]


def cat_profile(split):
    now = rand_now()
    sg, eng, ad = rnd.randint(1, 3), pick(ENG), rnd.random() < 0.3
    fam = eng.split()[0]
    tpl = pick(split_phr(["я в {sg}-й подгруппе, английский у {fg}", "моя подгруппа {sg}, по англу {f}",
                          "запомни: подгруппа {sg}, английский — {f}{a}", "у меня {sg} подгруппа и {f} по английскому{a}",
                          "я из {sg}-й подгруппы, англ ведёт {f}", "подгруппа {sg}, англичанка {f}{a}"], split))
    ad = ad and "{a}" in tpl  # про курсы (А) студент сказал только в этих шаблонах
    q = tpl.format(sg=sg, f=fam, fg=fam[:-1] + "ой", a=", хожу на русский (А)" if ad else "")
    args = {"subgroup": sg, "english_teacher": fam} | ({"adaptation": True} if ad else {})
    res = Backend(now, {}).set_profile(**args)  # фамилию бэкенд превращает в полное ФИО
    eng = res["profile"]["english_teacher"]
    ans = f"Запомнил ✍️ Подгруппа {sg}, английский — {eng}{' Курсы (А) тоже покажу.' if ad else ''} Теперь в расписании будут только твои пары."
    return [system(now, {}), {"role": "user", "content": q}, tc("set_profile", args), tr("set_profile", res),
            {"role": "assistant", "content": ans}]


# ================= Д/З, напоминания, заметки, дедлайны, сессия, брифинг, ответы без инструментов =================
HW_TASKS = {
    "math": ["задачи 3.14–3.20", "номера 112, 115 и 118 из задачника", "доделать типовой расчёт 1", "выучить определения пределов"],
    "matlog": ["таблицы истинности для 5 формул", "задачи 1–6 на КНФ и ДНФ", "доказать 3 тождества из листка"],
    "python": ["дописать лабу 2", "задачи на списки из листка", "оформить отчёт по лабе 1"],
    "inf": ["перевести 10 чисел в двоичную систему", "конспект про кодирование текста", "оформить лабу 3"],
    "eng": ["упражнения 4–7 на стр. 23", "выучить 20 слов по теме Technology", "эссе на 150 слов"],
    "hist": ["прочитать параграф 5", "подготовить доклад про реформы Петра I"],
    "intro": ["ответить на вопросы после лекции", "установить Python и VS Code"],
    "proj": ["придумать идею проекта", "заполнить паспорт проекта"],
    "rus_a": ["упражнения 3–5", "выучить слова урока 4"],
}
UNKNOWN_HW = [("физика", "задачи 1–5 из параграфа 2"), ("химия", "выучить таблицу растворимости"), ("черчение", "доделать чертёж 2")]
# Напоминания: text — слова студента с заглавной буквы, без переписывания
REM_TASKS = ["сдать отчёт старосте", "позвонить в деканат", "купить тетрадь", "взять пропуск", "скинуть лабу преподу",
             "оплатить общагу", "забрать справку", "записаться на пересдачу", "зарядить ноут", "сходить в библиотеку",
             "сдать лабу 3 по питону", "скинуть эссе Ростовцевой", "забрать зачётку в деканате", "продлить книгу в НТБ",
             "оплатить проездной", "записаться на физру", "скинуть презу для проектной", "отправить реферат на проверку",
             "выпить таблетку", "позвонить маме", "заказать справку для военкомата", "распечатать отчёт по инфе",
             "подготовиться к коллоквиуму по матану", "проверить почту ТПУ", "скачать задания по матлогике",
             "купить проездной на месяц", "отдать конспект Кате", "сходить на консультацию по питону"]
# Заметки и как студент ищет их: (что говорит студент, что передать в query)
NOTES = [("пароль от вайфая в КЦ спросить у лаборанта", [("вайфай", "вайфай"), ("пароль", "пароль")]),
         ("купить флешку на 32 ГБ", [("флешку", "флешка")]),
         ("идея для проекта: бот с расписанием", [("проект", "проект"), ("бота", "бот")]),
         ("в субботу собрание группы", [("собрание", "собрание")]),
         ("книжку по матану вернуть в библиотеку", [("книжку", "книжка"), ("библиотеку", "библиотека")]),
         ("спросить у старосты про физру", [("старосту", "староста"), ("физру", "физра")]),
         ("скачать Anaconda для питона", [("питон", "питон"), ("Anaconda", "Anaconda")]),
         ("сделать фото на студенческий", [("студенческий", "студенческий"), ("фото", "фото")]),
         ("размер кроссовок для физры — 42", [("кроссовки", "кроссовки")]),
         ("тема реферата по истории: реформы Петра I", [("реферат", "реферат"), ("Петра", "Петр")]),
         ("в НТБ можно бесплатно распечатать 20 страниц", [("НТБ", "НТБ"), ("распечатку", "распечатка")]),
         ("код от шкафчика в спортзале 4512", [("шкафчик", "шкафчик")]),
         ("день рождения у Димы 14 ноября", [("день рождения", "день рождения")]),
         ("взять у Кати конспект по матлогике", [("конспект", "конспект")]),
         ("общагу оплатить до 25 числа", [("общагу", "общага")])]
WEEKDAYS_FUT = ["в понедельник", "во вторник", "в среду", "в четверг", "в пятницу", "в субботу", "в воскресенье"]
WD_DAT = ["понедельнику", "вторнику", "среде", "четвергу", "пятнице", "субботе", "воскресенью"]
WD_GEN = ["понедельника", "вторника", "среды", "четверга", "пятницы", "субботы", "воскресенья"]


def plural(n, one, few, many):
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


def days_n(n):
    return f"{n} {plural(n, 'день', 'дня', 'дней')}"


def in_days(n):
    """0 → «сегодня», 1 → «завтра», иначе «через N дней»."""
    return {0: "сегодня", 1: "завтра"}.get(n) or (pick(["послезавтра", "через 2 дня"]) if n == 2 else f"через {days_n(n)}")


def cap(s):
    return s[:1].upper() + s[1:]


def low(s):
    return s[:1].lower() + s[1:]


def hm(t):
    """'07:30' → '7:30'."""
    return f"{int(t[:2])}:{t[3:5]}"


def when_ru(now, s):
    """'2026-10-08 10:25' или '2026-10-08' -> 'сегодня в 10:25' / 'завтра' / 'в чт 8 октября в 10:25'."""
    has_t = " " in s
    t = dt.datetime.fromisoformat(s) if has_t else dt.datetime.fromisoformat(s + " 00:00")
    delta = (t.date() - now.date()).days
    day = {0: "сегодня", 1: "завтра"}.get(delta, f"в {WD_SHORT[t.weekday()]} {date_ru(t.date())}")
    return f"{day} в {t:%H:%M}" if has_t else day


def date_span(a, z):
    if a == z:
        return date_ru(a)
    if (a.year, a.month) == (z.year, z.month):
        return f"{a.day}–{z.day} {MONTHS[a.month - 1]}"
    return f"{date_ru(a)} – {date_ru(z)}"


def join_dates(ds):
    ds = [dt.date.fromisoformat(x) for x in ds]
    if len(ds) == 2 and ds[0].month == ds[1].month:
        return f"{ds[0].day} и {date_ru(ds[1])}"
    return " и ".join(date_ru(d) for d in ds)


def student_subjects(b):
    """Предметы, которые реально есть у студента в ближайшие 2 недели."""
    codes = set()
    for add in range(14):
        for l in b.lessons_on(b.now.date() + dt.timedelta(days=add)):
            codes.add(l["subject"])
    return [c for c in HW_TASKS if c in codes]


def has_class(b, code, d):
    return any(l["subject"] == code for l in b.lessons_on(d))


def pick_day(b, code, days):
    """Чаще — день, когда есть пара по предмету (так обычно и задают)."""
    with_class = [d for d in days if has_class(b, code, d)]
    return pick(with_class) if with_class and rnd.random() < 0.75 else pick(days)


def sub_alias(code):
    return pick(SUBJECTS[code]["aliases"])


DAT = {"матлогика": "матлогике", "мат.логика": "мат.логике", "матлог": "матлогу", "теория алгоритмов": "теории алгоритмов",
       "информатика": "информатике", "инфа": "инфе", "информатика 1.2": "информатике 1.2", "математика": "математике",
       "матеша": "матеше", "матан": "матану", "вышмат": "вышмату", "математика (А)": "математике (А)",
       "адаптационная математика": "адаптационной математике", "питон": "питону", "пайтон": "пайтону", "прога": "проге",
       "программирование": "программированию", "английский": "английскому", "инглиш": "инглишу", "англ": "англу",
       "английский (А)": "английскому (А)", "адаптационный английский": "адаптационному английскому",
       "русский": "русскому", "русский язык": "русскому языку", "русский (А)": "русскому (А)", "история": "истории",
       "история России": "истории России", "введение в ИТ": "введению в ИТ", "введение в айти": "введению в айти",
       "проектная деятельность": "проектной деятельности", "проектка": "проектке", "физра": "физре",
       "физкультура": "физкультуре", "«Код ТПУ»": "«Коду ТПУ»", "код ТПУ": "коду ТПУ", "физика": "физике",
       "химия": "химии", "черчение": "черчению"}


def dat(x):
    """«по матану», «к информатике»: дательный падеж названия предмета."""
    return DAT.get(x, x)


SLANG = {"инфа", "матан", "матеша", "вышмат", "прога", "питон", "пайтон", "матлог", "англ", "инглиш", "физра", "проектка"}


def sdat(alias):
    """Как напишет студент: обычно в нужном падеже, сленг иногда как попало («по инфа»)."""
    return alias if alias in SLANG and rnd.random() < 0.25 else dat(alias)


def short(subject):
    return SHORT.get(subject, subject)


def seed_homework(b, k):
    subs = student_subjects(b)
    for code in rnd.sample(subs, min(k, len(subs))):
        b.add_homework(SUBJECTS[code]["name"], pick(HW_TASKS[code]))


def hw_answer(now, res, task):
    sh, d_name = cap(short(res["subject"])), dat(short(res["subject"]))
    w = when_ru(now, res["due"])
    pair = " на паре" if " " in res["due"] else ""
    if res["remind_at"] is None and res["due"][:10] == now.date().isoformat() and not pair:
        rem = "Срок уже сегодня, так что напоминать не буду."
    elif res["remind_at"] is None:
        rem = pick(["Пара уже совсем скоро, так что напоминание не ставлю — успей 😉", "До пары меньше часа, напоминать не буду — успевай 😉"]) \
            if pair else pick(["Срок уже завтра, так что держи в голове 😉", "Напоминать уже поздно, срок совсем близко."])
    elif pair and rnd.random() < 0.5:
        rem = f"Напомню за час до пары, в {res['remind_at'][11:]}."
    else:
        rem = f"Напомню {when_ru(now, res['remind_at'])}."
    note = ""
    if res.get("no_class_that_day"):
        note = " " + pick([f"Пары по {d_name} в этот день нет, так что срок просто по дате.",
                           f"В этот день пар по {d_name} нет — записал на дату."])
    ans = pick([f"Записал ✍️ {sh}: {task}. Сдавать {w}{pair}.{note} {rem}",
                f"Готово 📝 {sh} — {task}. Срок: {w}{pair}.{note} {rem}",
                f"Есть! {sh}: {task}, срок — {w}{pair}.{note} {rem}",
                f"Ок, {task} по {d_name} — {w}{pair}.{note} {rem}"])
    if rnd.random() < 0.25:
        week = (now.date() + dt.timedelta(days=7)).isoformat()
        ans += " Храню неделю." if res["expires"] == week else " Храню до срока сдачи."
    return ans


def cat_add_homework(split):
    now, p = rand_now(), rand_profile() or {"subgroup": 1, "english_teacher": ENG[0], "adaptation": False}
    b = Backend(now, p)
    subs = student_subjects(b)
    if rnd.random() < 0.08 or not subs:  # предмета нет в расписании: бэкенд не найдёт пару
        alias, task = pick(UNKNOWN_HW)
        q = pick(split_phr(["запиши дз по {sd}: {t}", "задали по {sd} {t}, запиши", "по {sd} на след пару: {t}",
                            "дз {s} — {t}", "сохрани дз по {sd}: {t}", "надо по {sd} {t} к следующей паре"],
                           split)).format(s=alias, sd=sdat(alias), t=task)
        args = {"subject": alias, "task": task, "due": "next_class"}
        res = b.add_homework(**args)
        msgs = [system(now, p), {"role": "user", "content": q}, tc("add_homework", args), tr("add_homework", res)]
        msgs.append({"role": "assistant", "content": pick([
            f"Не нашёл ближайших пар по предмету «{alias}» в твоём расписании 🤔 К какому числу сдавать? Запишу на дату.",
            f"Хм, пар по предмету «{alias}» в твоём расписании не видно 🤔 Скажи дату сдачи — запишу на неё."])})
        if rnd.random() < 0.6:
            d = pick(no_sunday([now.date() + dt.timedelta(days=k) for k in range(2, 8)]))
            msgs.append({"role": "user", "content": pick([date_ru(d), f"к {date_ru(d)}", f"до {date_ru(d)}", f"{date_ru(d)} сдать"])})
            args = args | {"due": d.isoformat()}
            res = b.add_homework(**args)
            msgs += [tc("add_homework", args), tr("add_homework", res), {"role": "assistant", "content": hw_answer(now, res, task)}]
        return msgs
    code = pick(subs)
    task, alias = pick(HW_TASKS[code]), sub_alias(code)
    sd = sdat(alias)
    today = now.date()
    tomorrow = today + dt.timedelta(days=1)
    form = pick(["next", "next", "next", "weekday", "weekday", "tomorrow", "date", "date"])
    if form == "tomorrow" and (not no_sunday([tomorrow]) or (not has_class(b, code, tomorrow) and rnd.random() < 0.75)):
        form = "next"
    if form == "next":
        q = pick(split_phr(["запиши дз по {sd}: {t}", "задали по {sd} {t}, запиши", "дз {s} — {t}",
                            "по {sd} на след пару: {t}", "запиши домашку: {s}, {t}", "к следующей паре по {sd} задали {t}",
                            "сохрани дз по {sd}: {t}", "надо по {sd} {t} к следующей паре"], split)).format(s=alias, sd=sd, t=task)
        due = "next_class"
    elif form == "weekday":
        d = pick_day(b, code, no_sunday([today + dt.timedelta(days=k) for k in range(1, 8)]))
        wd = d.weekday()
        q = pick(split_phr(["по {sd} к {dd} {t}, запиши", "запиши дз по {sd} на {da}: {t}", "к {dd} по {sd} надо {t}",
                            "по {sd} задали {t}, сдавать {di}", "дз по {sd} на {da} — {t}", "{s}: {t}, к {dd}",
                            "дз по {sd} {t}, сдать к {dd}", "{di} по {sd} надо: {t}, запиши"],
                           split)).format(s=alias, sd=sd, t=task, dd=WD_DAT[wd], da=WD_ACC[wd], di=WEEKDAYS_FUT[wd])
        due = WEEKDAYS[wd]
    elif form == "tomorrow":
        q = pick(split_phr(["на завтра по {sd} {t}, запиши", "завтра по {sd} надо: {t}", "запиши на завтра: {s}, {t}",
                            "к завтрашней паре по {sd} {t}", "на завтра задали по {sd} {t}", "по {sd} к завтра: {t}",
                            "запиши: завтра {s} — {t}", "к завтра по {sd} надо {t}"], split)).format(s=alias, sd=sd, t=task)
        due = "tomorrow"
    else:
        d = pick_day(b, code, no_sunday([today + dt.timedelta(days=k) for k in range(2, 10)]))
        q = pick(split_phr(["запиши дз по {sd} на {d}: {t}", "по {sd} к {d} надо {t}", "дз: {s}, {t}, срок {d}",
                            "запиши: {s} — {t}, сдать до {d}", "по {sd} задали {t}, сдать {d}", "к {d} по {sd} {t}, запиши",
                            "дз по {sd} на {d} — {t}", "до {d} по {sd}: {t}"], split)).format(s=alias, sd=sd, t=task, d=date_ru(d))
        due = d.isoformat()
    args = {"subject": alias, "task": task, "due": due}
    res = b.add_homework(**args)
    return [system(now, p), {"role": "user", "content": q}, tc("add_homework", args), tr("add_homework", res),
            {"role": "assistant", "content": hw_answer(now, res, task)}]


def cat_list_homework(split):
    now, p = rand_now(), rand_profile() or {"subgroup": 2, "english_teacher": ENG[2], "adaptation": False}
    b = Backend(now, p)
    seed_homework(b, rnd.choice([0, 1, 2, 2, 3]))
    form = pick(["all", "all", "tomorrow", "today", "subject"])
    if form in ("tomorrow", "today"):
        d = now.date() + dt.timedelta(days=form == "tomorrow")
        codes = [l["subject"] for l in b.lessons_on(d) if l["subject"] in HW_TASKS
                 and dt.datetime.combine(d, dt.time.fromisoformat(l["start"])) > now]
        if codes and rnd.random() < 0.7:  # Д/З к парам этого дня
            c = pick(codes)
            b.add_homework(SUBJECTS[c]["name"], pick(HW_TASKS[c]), due=d.isoformat())
    if form == "subject":
        subs = student_subjects(b)
        if b.homework and rnd.random() < 0.75:
            h = pick(b.homework)
            code = next(c for c in SUBJECTS if SUBJECTS[c]["name"] == h["subject"])
        else:
            code = pick(subs or ["math"])
        alias = sub_alias(code)
        q = pick(split_phr(["что задано по {s}?", "что по {s} задавали", "дз по {s} какое", "есть дз по {s}?",
                            "напомни домашку по {s}", "по {s} что надо сделать"], split)).format(s=sdat(alias))
        args = {"subject": alias}
    elif form in ("tomorrow", "today"):
        q = pick(split_phr({"tomorrow": ["что задано на завтра?", "дз на завтра есть?", "на завтра что-то задавали",
                                         "к завтра что сделать надо", "завтра что сдавать", "домашка на завтра"],
                            "today": ["что на сегодня задано", "дз на сегодня", "сегодня надо что-то сдавать?",
                                      "есть дз к сегодняшним парам?", "что сдать сегодня", "сегодня какая домашка"]}[form], split))
        args = {"due": form}
    else:
        q = pick(split_phr(["что у меня по дз", "покажи всю домашку", "какие дз висят?", "список домашки",
                            "что задано вообще", "чё по домашке", "какие у меня домашки", "напомни все дз"], split))
        args = {}
    res = b.list_homework(**args)
    items = res["items"]
    day_word = {"tomorrow": "завтра", "today": "сегодня"}.get(form)
    if not items:
        if form == "subject":
            n = dat(short(SUBJECTS[code]["name"]))
            ans = pick([f"По {n} ничего не записано 🎉", f"По {n} у меня Д/З нет. Задали что-то? Скажи, запишу ✍️"])
        elif day_word:
            ans = pick([f"На {day_word} ничего не записано 🎉", f"К {'завтрашним' if form == 'tomorrow' else 'сегодняшним'} парам Д/З у меня нет 😎"
                        if b.lessons_on(d) else f"На {day_word} Д/З у меня нет 😎",
                        f"На {day_word} по моим записям ничего. Если задавали — скажи, запишу ✍️"])
        else:
            ans = pick(["Ничего не записано 🎉 Если что-то задали — скажи, запишу.", "Пусто, никаких Д/З не сохранено 😎",
                        "По моим записям — ничего. Задали что-то? Скажи, запишу ✍️"])
    elif len(items) == 1:
        h = items[0]
        ans = pick([f"Есть одно: {short(h['subject'])} — {h['task']}, срок {when_ru(now, h['due'])}.",
                    f"Записано одно задание: {short(h['subject'])}, {h['task']} — {when_ru(now, h['due'])} 📌"])
    else:
        lines = "\n".join(f"• {short(h['subject'])}: {h['task']} — {when_ru(now, h['due'])}" for h in items)
        head = [f"Записано {len(items)} {plural(len(items), 'задание', 'задания', 'заданий')}:\n", "Вот что задано:\n"]
        if day_word:
            head.append(f"На {day_word}:\n")
        ans = pick(head) + lines
    return [system(now, p), {"role": "user", "content": q}, tc("list_homework", args), tr("list_homework", res),
            {"role": "assistant", "content": ans}]


def reminder_phrase(now):
    """-> (фраза времени, аргументы). Время срабатывания — с 8 до 22, «в 15» — всегда ближайшее будущее."""
    probe = Backend(now, {})
    for _ in range(100):
        form = pick(["min", "hours", "hours", "days", "at", "at", "tomorrow", "weekday", "half"])
        m = pick([0, 0, 15, 30, 45])
        if form == "min":
            n = pick([5, 10, 15, 20, 30, 40, 45])
            spoken, args = f"через {n} {plural(n, 'минуту', 'минуты', 'минут')}", {"in_minutes": n}
        elif form == "half":
            spoken, args = "через полчаса", {"in_minutes": 30}
        elif form == "hours":
            n = pick([1, 2, 3, 4, 5])
            spoken, args = ("через час" if n == 1 else f"через {n} {plural(n, 'час', 'часа', 'часов')}"), {"in_hours": n}
        elif form == "days":
            n = pick([1, 2, 3, 5, 7])
            spoken, args = ("через сутки" if n == 1 else f"через {days_n(n)}"), {"in_days": n}
        else:
            h = rnd.randint(now.hour + 1, 22) if form == "at" and now.hour < 22 else rnd.randint(8, 21)
            hhmm = f"{h:02d}:{m:02d}"
            if m:
                t = f"{h}:{m:02d}"
            else:
                t = pick([f"{h}:00", f"{h}", f"{h} {plural(h, 'час', 'часа', 'часов')}"]
                         + ([f"{h} утра"] if h < 12 else [f"{h - 12} вечера"] if h >= 17 else []))
            if form == "at":
                spoken, args = f"в {t}", {"at": hhmm}
            elif form == "tomorrow":
                spoken, args = f"завтра в {t}", {"at": f"tomorrow {hhmm}"}
            else:
                wd = pick([x for x in range(7) if x != now.weekday()])  # про сегодня говорят «сегодня в …»
                spoken, args = f"{WEEKDAYS_FUT[wd]} в {t}", {"at": f"{WEEKDAYS[wd]} {hhmm}"}
        res = probe.create_reminder("x", **args)
        if res.get("ok") and 8 <= int(res["fire_at"][11:13]) <= 22:
            return spoken, args
    raise RuntimeError("не нашлось подходящего времени")


def reminder_answer(now, res, what):
    w = when_ru(now, res["fire_at"])
    return pick([f"Ок, {w} напомню {what} ⏰", f"Готово: {w} — напоминание «{what}» 🔔",
                 f"Поставил ⏰ {cap(w)} напомню {what}.", f"Напомню {what} {w} 👌", f"Есть, напоминалка на {w.replace('сегодня в ', '').removeprefix('в ')}: {what} 🔔"])


def cat_reminder(split):
    now, p = rand_now((7, 21)), rand_profile()
    b = Backend(now, p)
    what = pick(REM_TASKS)
    wp, targs = reminder_phrase(now)
    q = pick(split_phr(["напомни {w} {x}", "{w} напомни {x}", "поставь напоминалку {w}: {x}", "напомни мне {x} {w}",
                        "кинь напоминание {w} — {x}", "не дай забыть {x}, напомни {w}", "напоминание {w}: {x}",
                        "можешь {w} напомнить {x}?"], split)).format(w=wp, x=what)
    args = {"text": cap(what)} | targs
    res = b.create_reminder(**args)
    return [system(now, p), {"role": "user", "content": q}, tc("create_reminder", args), tr("create_reminder", res),
            {"role": "assistant", "content": reminder_answer(now, res, what)}]


def cat_reminder_clarify(split):
    now, p = rand_now((7, 21)), rand_profile()
    b = Backend(now, p)
    what = pick(REM_TASKS)
    wp, targs = reminder_phrase(now)
    q = pick(split_phr(["напомни {x}", "поставь напоминание {x}", "не дай мне забыть {x}", "надо не забыть {x}",
                        "напоминалку сделай: {x}", "хочу напоминалку: {x}", "напомни мне {x} пж"], split)).format(x=what)
    ask = pick(["Когда напомнить? Например «через час», «завтра в 9:00» или «в пятницу в 18:00».",
                "Ок! А когда? Можно «через 30 минут», «в 19:00» или «завтра в 8».",
                "Во сколько напомнить? 🙂 Скажи время или «через сколько»."])
    args = {"text": cap(what)} | targs
    res = b.create_reminder(**args)
    return [system(now, p), {"role": "user", "content": q}, {"role": "assistant", "content": ask},
            {"role": "user", "content": wp}, tc("create_reminder", args), tr("create_reminder", res),
            {"role": "assistant", "content": reminder_answer(now, res, what)}]


def cat_note_add(split):
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)
    x = pick(NOTES)[0]
    q = pick(split_phr(["запиши в заметки: {x}", "сохрани заметку: {x}", "заметка: {x}", "запомни: {x}",
                        "добавь в заметки {x}", "занеси в заметки — {x}"], split)).format(x=x)
    args = {"text": cap(x)}
    res = b.add_note(**args)
    ans = pick(["Сохранил в заметки 📝", "Записал 📝 Найдёшь в заметках.", "Готово, заметка сохранена ✍️",
                f"Записал в заметки: «{cap(x)}» 📝", "Запомнил 👌 Спроси «что в заметках» — покажу."])
    return [system(now, p), {"role": "user", "content": q}, tc("add_note", args), tr("add_note", res),
            {"role": "assistant", "content": ans}]


def cat_note_list(split):
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)
    seeded = rnd.sample(NOTES, rnd.choice([0, 1, 2, 3]))
    for x, _ in seeded:
        b.add_note(cap(x))
    for what in rnd.sample(REM_TASKS, rnd.choice([0, 1, 2])):
        at = pick(["", "tomorrow ", pick(WEEKDAYS) + " "]) + f"{rnd.randint(8, 21):02d}:{pick(['00', '15', '30', '45'])}"
        b.create_reminder(cap(what), at=at)
    intent = pick(["notes", "notes", "reminders", "both"] + (["search", "search"] if seeded else []))
    if intent == "search":
        pool = seeded if rnd.random() < 0.8 else [n for n in NOTES if n not in seeded]
        k, query = pick(pick(pool)[1])
        q = pick(split_phr(["найди заметку про {k}", "что я записывал про {k}?", "есть заметка про {k}?",
                            "поищи в заметках {k}", "где заметка про {k}?", "заметки про {k} покажи"], split)).format(k=k)
        args = {"query": query}
    else:
        q = pick(split_phr({"notes": ["что у меня в заметках", "покажи заметки", "мои заметки", "список заметок",
                                      "открой заметки", "что я записывал в заметки?"],
                            "reminders": ["какие напоминалки стоят?", "какие напоминания есть", "что ты мне должен напомнить?",
                                          "покажи напоминания", "есть напоминалки?", "мои напоминания"],
                            "both": ["покажи заметки и напоминания", "что я просил запомнить?", "заметки и напоминалки покажи",
                                     "что у меня записано?", "всё, что я просил записать", "что я тебе говорил запомнить"]}[intent], split))
        args = {}
    res = b.list_notes(**args)
    notes = "📝 Заметки:\n" + "\n".join(f"• {n}" for n in res["notes"])
    rems = "⏰ Напоминания:\n" + "\n".join(f"• {when_ru(now, r['fire_at'])} — {r['text']}" for r in res["reminders"])
    if intent == "search" and not res["notes"] and res["reminders"]:
        nr = len(res["reminders"])
        ans = f"Заметок про «{k}» нет, но есть {plural(nr, 'напоминание', 'напоминания', 'напоминаний')}:\n" + \
            "\n".join(f"• {when_ru(now, r['fire_at'])} — {r['text']}" for r in res["reminders"])
    elif intent == "search":
        parts = ([notes] if res["notes"] else []) + ([rems] if res["reminders"] else [])
        ans = "\n\n".join(parts) if parts else pick([f"Про «{k}» ничего не нашёл 🤷", f"Заметок про «{k}» нет 🙃"])
    elif intent == "notes":
        if res["notes"]:
            ans = notes
        elif res["reminders"]:
            ans = "Заметок пока нет. Зато есть напоминания:\n" + "\n".join(f"• {when_ru(now, r['fire_at'])} — {r['text']}" for r in res["reminders"])
        else:
            ans = pick(["Заметок пока нет 📭 Скажи «запиши в заметки …» — сохраню.", "Пока пусто — ни одной заметки."])
    elif intent == "reminders":
        ans = rems if res["reminders"] else pick(["Напоминаний нет 🙂", "Сейчас ни одного напоминания не стоит. Поставить?"])
    else:
        parts = ([notes] if res["notes"] else []) + ([rems] if res["reminders"] else [])
        ans = "\n\n".join(parts) if parts else pick(["Пока пусто — ни заметок, ни напоминаний.", "Ничего не записано 🙃"])
    return [system(now, p), {"role": "user", "content": q}, tc("list_notes", args), tr("list_notes", res),
            {"role": "assistant", "content": ans}]


# (название, вид, как сказать после «сдать» (вин. п.), как сказать в «дедлайн: …» (им. п.), предметы, где такое бывает)
DL = [("Лаба {n}", "лабораторная", ["лабу {n}", "лабораторку {n}", "лабу №{n}"], ["лаба {n}", "лабораторка {n}", "лаба №{n}"], ["python", "inf", "intro"]),
      ("Курсовая", "курсовая", ["курсач", "курсовую"], ["курсач", "курсовая"], ["proj"]),
      ("Реферат", "реферат", ["реферат"], ["реферат"], ["hist", "intro", "proj"]),
      ("Доклад", "другое", ["доклад"], ["доклад"], ["hist", "intro", "proj", "matlog"]),
      ("Проект", "другое", ["проект"], ["проект"], ["proj", "python", "intro"]),
      ("Типовой расчёт {n}", "другое", ["типовой расчёт {n}", "ТР {n}", "типовик {n}"], ["типовой расчёт {n}", "ТР {n}", "типовик {n}"], ["math"]),
      ("Эссе", "другое", ["эссе"], ["эссе"], ["eng", "hist"])]


def rand_deadline(code=None, kind=None):
    opts = [x for x in DL if (code is None or code in x[4]) and (kind is None or x[1] == kind)]
    title, kind, acc, nom, codes = pick(opts)
    n = rnd.randint(1, 3 if "расчёт" in title else 6)
    i = rnd.randrange(len(acc))
    return title.format(n=n), kind, acc[i].format(n=n), nom[i].format(n=n), code or pick(codes)


def left_txt(n):
    return {0: "это уже сегодня", 1: "это уже завтра"}.get(n) or f"{plural(n, 'остался', 'осталось', 'осталось')} {days_n(n)}"


def cat_add_deadline(split):
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)
    title, kind, x, xn, code = rand_deadline()
    alias = sub_alias(code)
    form = pick(["date", "date", "date", "rel", "rel", "weekday", "weekday", "tomorrow", "next"])
    if form == "next" and code not in student_subjects(b):
        form = "date"
    if form == "date":
        d = pick(no_sunday([now.date() + dt.timedelta(days=k) for k in range(3, 70)]))
        due, w = d.isoformat(), pick([f"до {date_ru(d)}", f"к {date_ru(d)}"])
    elif form == "rel":
        k = pick([7, 10, 14, 21, 30])
        due, w = f"in {k} days", {7: "через неделю", 10: "через 10 дней", 14: "через 2 недели", 21: "через 3 недели", 30: "через месяц"}[k]
    elif form == "weekday":
        wd = rnd.randrange(6)
        due, w = WEEKDAYS[wd], pick([f"до {WD_GEN[wd]}", f"к {WD_DAT[wd]}"])
    elif form == "tomorrow":
        due, w = "tomorrow", pick(["до завтра", "к завтра", "завтра"])
    else:
        due, w = "next_class", "к следующей паре"
    sd = sdat(alias)
    q = pick(split_phr(["{x} по {sd} сдать {w}, запиши", "запиши дедлайн: {xn} по {sd} {w}", "надо сдать {x} по {sd} {w}",
                        "дедлайн {xn} {s} — {w}", "сдать {x} по {sd} {w}", "{xn} по {sd} — дедлайн {w}",
                        "запиши, что {x} по {sd} сдавать {w}", "добавь дедлайн: {xn}, {s}, {w}",
                        "{x} по {sd} {w} сдавать, не дай забыть"], split)).format(x=x, xn=xn, s=alias, sd=sd, w=w)
    args = {"title": title, "subject": alias, "kind": kind, "due": due}
    res = b.add_deadline(**args)
    d = dt.date.fromisoformat(res["due"])
    left, n = res["days_left"], dat(short(SUBJECTS[code]["name"]))
    paren = {0: "сегодня", 1: "завтра"}.get(left) or days_n(left)
    ans = pick([f"Записал 📌 {title} по {n} — до {WD_SHORT[d.weekday()]} {date_ru(d)}, {left_txt(left)}.",
                f"Есть: {low(title)} по {n}, срок {date_ru(d)} ({paren}) ⏳",
                f"Готово ✅ {title} ({short(SUBJECTS[code]['name'])}) — {WD_SHORT[d.weekday()]} {date_ru(d)}, {left_txt(left)}.",
                f"Окей, {low(title)} по {n} до {date_ru(d)} ({paren})."])
    if res["reminders"]:
        ans += pick([f" Напомню {join_dates(res['reminders'])}.", f" Напоминания будут {join_dates(res['reminders'])} ⏰"])
    elif left == 0:
        ans += pick([" Сдавать уже сегодня — удачи! 🍀", " Это уже сегодня, напоминать некогда — успевай 😉"])
    else:
        ans += pick([" Срок совсем близко, так что не откладывай 😉", " Времени мало — лучше начать прямо сегодня 💪"])
    return [system(now, p), {"role": "user", "content": q}, tc("add_deadline", args), tr("add_deadline", res),
            {"role": "assistant", "content": ans}]


def cat_list_deadlines(split):
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)

    def seed(**kw):
        for _ in range(20):  # не больше одной работы каждого вида на предмет
            title, kind, _, _, code = rand_deadline(**kw)
            if (title.split()[0], SUBJECTS[code]["name"]) not in {(x["title"].split()[0], x["subject"]) for x in b.deadlines}:
                break
        d = pick(no_sunday([now.date() + dt.timedelta(days=k) for k in range(1, 41)]))
        b.add_deadline(title, d.isoformat(), SUBJECTS[code]["name"], kind)
        return code

    for _ in range(rnd.choice([0, 1, 2, 2, 3, 3])):
        seed()
    r = rnd.random()
    if r < 0.25:
        if rnd.random() < 0.7:
            seed(kind="лабораторная")
        q = pick(split_phr(["какие лабы горят?", "что по лабам", "дедлайны по лабам покажи", "какие лабораторные висят",
                            "сколько лаб надо сдать", "лабы когда сдавать"], split))
        args, what = {"kind": "лабораторная"}, "лабам"
    elif r < 0.37:
        code = pick(["python", "inf", "intro", "proj", "hist", "math"])
        if rnd.random() < 0.75:
            seed(code=code)
        alias = sub_alias(code)
        q = pick(split_phr(["дедлайны по {s}", "что по {s} надо сдать", "какие сроки по {s}", "по {s} что горит",
                            "что сдавать по {s}", "по {s} дедлайны есть?"], split)).format(s=sdat(alias))
        args, what = {"subject": alias}, dat(short(SUBJECTS[code]["name"]))
    elif r < 0.45:
        kind = pick(["курсовая", "реферат"])
        if rnd.random() < 0.7:
            seed(kind=kind)
        q = pick(split_phr({"курсовая": ["когда курсач сдавать", "что по курсовой", "дедлайн по курсачу какой", "курсовая когда"],
                            "реферат": ["когда реферат сдавать", "рефераты какие висят", "дедлайн по реферату", "реферат когда сдать"]}[kind], split))
        args, what = {"kind": kind}, {"курсовая": "курсовой", "реферат": "рефератам"}[kind]
    else:
        q = pick(split_phr(["какие у меня дедлайны", "что горит?", "покажи дедлайны", "что надо сдать в ближайшее время",
                            "какие сроки сдачи", "дедлайны есть?", "что по срокам", "чё горит по учёбе"], split))
        args, what = {}, None
    res = b.list_deadlines(**args)
    items = res["items"]
    if not items:
        ans = pick([f"По {what} дедлайнов не записано 🎉", f"По {what} ничего не горит, по крайней мере у меня не записано 😌"]) if what \
            else pick(["Дедлайнов не записано 🎉", "Ничего не горит, по крайней мере у меня не записано 😌",
                       "Пусто, дедлайнов нет 😎 Как появятся — скажи, запишу."])
    else:
        lines = []
        for x in items:
            d = dt.date.fromisoformat(x["due"])
            mark = "🔥" if x["days_left"] <= 3 else "•"
            lines.append(f"{mark} {x['title']} ({short(x['subject'])}) — {WD_SHORT[d.weekday()]} {date_ru(d)}, {in_days(x['days_left'])}")
        ans = "\n".join(lines)
        if "сколько" in q:
            ans = f"Лаб к сдаче: {len(items)}\n" + ans
        first = items[0]
        if first["days_left"] <= 3 and len(items) > 1:
            ans += f"\n\nПервым делом — {low(first['title'])} по {dat(short(first['subject']))} 🔥"
    return [system(now, p), {"role": "user", "content": q}, tc("list_deadlines", args), tr("list_deadlines", res),
            {"role": "assistant", "content": ans}]


EXAM_CODES = ["python", "inf", "math", "matlog"]
EXAM_POOL = [("python", "экзамен"), ("inf", "экзамен"), ("math", "экзамен"), ("matlog", "экзамен"),
             ("hist", "зачёт"), ("intro", "дифзачёт"), ("eng", "зачёт"), ("proj", "зачёт")]


def no_sunday(days):
    """Только учебные дни: без воскресений и праздников."""
    return [d for d in days if d.weekday() != 6 and d.isoformat() not in HOLIDAYS]


def seed_exams(b, k):
    jan = [d for d in no_sunday([dt.date(2027, 1, d) for d in range(9, 29)]) if d > b.now.date()]
    dec = [d for d in no_sunday([dt.date(2026, 12, d) for d in range(21, 31)]) if d > b.now.date()]
    for code, kind in rnd.sample(EXAM_POOL, k):
        pool = jan if kind == "экзамен" or not dec else dec
        d = pick(pool)
        pool.remove(d)
        kw = {"time": pick(["09:00", "10:25", "12:40"])} if rnd.random() < 0.6 else {}
        if rnd.random() < 0.5:
            kw["building"], kw["room"] = pick([r for r in ROOMS if r[1][:1].isdigit()])  # не спортзал
        b.add_exam(SUBJECTS[code]["name"], d.isoformat(), kind=kind, **kw)


def exam_line(e):
    d = dt.date.fromisoformat(e["date"])
    t = f" в {e['time']}" if e.get("time") else ""
    return f"{WD_SHORT[d.weekday()]} {date_ru(d)}{t}{where(e)}"


def cat_get_exams(split):
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)
    seed_exams(b, rnd.choice([0, 1, 2, 3, 4, 5]))
    if rnd.random() < 0.35:
        code = pick(EXAM_CODES + ["hist", "intro"])
        alias = sub_alias(code)
        q = pick(split_phr(["когда экзамен по {sd}?", "сколько до экза по {sd}", "экзамен по {sd} когда", "{s} экзамен какого числа",
                            "через сколько дней экзамен по {sd}", "напомни дату экзамена по {sd}"], split)).format(s=alias, sd=sdat(alias))
        args = {"subject": alias}
    else:
        q = pick(split_phr(["сколько до экзаменов осталось", "когда сессия", "какие экзамены и когда", "расписание экзаменов",
                            "когда экзы", "сколько дней до сессии", "покажи экзамены", "что у меня в сессию"], split))
        args = {}
    res = b.get_exams(**args)
    items = res["items"]
    if not items and args:
        n = dat(short(SUBJECTS[code]["name"]))
        ans = pick([f"По {n} у меня пока ничего не внесено 🤷 Как узнаешь дату — скинь, запишу 📌",
                    f"Даты экзамена по {n} пока не знаю. Скажешь — запишу и буду напоминать."])
    elif not items:
        ans = pick(["Пока ни одного экзамена не внесено 🤷 Как узнаешь даты — скинь, запишу 📌",
                    "Дат экзаменов у меня ещё нет. Сообщи, когда объявят, — внесу и буду напоминать."])
    elif args and len(items) == 1:
        e = items[0]
        ans = f"{cap(e['kind'])} по {dat(short(e['subject']))} — {exam_line(e)}, {in_days(e['days_left'])} ⏳"
    else:
        lines = [f"• {exam_line(e)} — {short(e['subject'])}, {e['kind']} ({in_days(e['days_left'])})" for e in items]
        head = ["Сессия:\n"] if any(e["kind"] != "экзамен" for e in items) else ["Экзамены:\n", "Твои экзамены:\n", ""]
        ans = pick(head) + "\n".join(lines)
    if any(e["kind"] == "экзамен" and e["days_left"] >= 3 for e in items) and rnd.random() < 0.5:
        ans += "\n\nМогу составить план повторения — скажи, сколько билетов и сколько часов в день готов заниматься 📖"
    return [system(now, p), {"role": "user", "content": q}, tc("get_exams", args), tr("get_exams", res),
            {"role": "assistant", "content": ans}]


def plan_answer(res, hours, skip_txt):
    lines = []
    for x in res["plan"]:
        a, z = dt.date.fromisoformat(x["from"]), dt.date.fromisoformat(x["to"])
        t = x["tickets"]
        t = t if not t[0].isdigit() else (f"билеты {t}" if "–" in t else f"билет {t}")
        lines.append(f"• {date_span(a, z)}: {t}")
    ex = dt.date.fromisoformat(res["exam_date"])
    n, N = dat(short(res["subject"])), res["tickets_total"]
    extra = (f", по {hours} ч в день" if hours else "") + skip_txt
    bil = f"{N} {plural(N, 'билет', 'билета', 'билетов')}"
    head = pick([f"План по {n} ({bil}, экзамен {date_ru(ex)}){extra}:\n",
                 f"Расписал {bil} по {n} до экзамена {date_ru(ex)}{extra}:\n",
                 f"Вот план подготовки к экзамену по {n} ({date_ru(ex)}, {bil}){extra}:\n"])
    end = pick(["В дни занятий в 19:00 напомню, что по плану. Ты сможешь 💪",
                "В учебные дни в 19:00 буду напоминать, какие билеты на сегодня 📌"]) if skip_txt else \
        pick(["Каждый день в 19:00 напомню, что сегодня по плану. Ты сможешь 💪",
              "В 19:00 буду напоминать, какие билеты на сегодня 📌", "Каждый вечер в 19:00 пришлю, что повторять. Погнали 🚀"])
    return head + "\n".join(lines) + "\n\n" + end


def cat_study_plan(split):
    now = dt.datetime.combine(dt.date(2026, 12, rnd.randint(1, 26)), dt.time(rnd.randint(9, 22), pick([0, 15, 30, 45])))
    p = rand_profile()
    b = Backend(now, p)
    code = pick(EXAM_CODES)
    alias = sub_alias(code)
    exam_day = pick(no_sunday([dt.date(2027, 1, d) for d in range(9, 29)]))
    if rnd.random() < 0.8:
        b.add_exam(SUBJECTS[code]["name"], exam_day.isoformat())
    tickets, hours = pick([25, 30, 35, 40, 45, 50, 60]), pick([None, 1, 2, 2, 3, 4])
    skip, skip_txt, skip_spoken = pick([(None, "", ""), ("sunday", ", воскресенья свободны", ", по воскресеньям отдыхаю"),
                                        ("saturday,sunday", ", выходные свободны", ", выходные не трогаем")])
    give_tickets = rnd.random() < 0.7
    h_txt = f", {hours} {plural(hours, 'час', 'часа', 'часов')} в день" if hours else ""
    t_txt = f", билетов {tickets}" if give_tickets else ""
    q = pick(split_phr(["составь план подготовки к экзамену по {s}{t}{h}{k}", "сделай план повторения по {s}{t}{h}{k}",
                        "помоги распланировать подготовку к {s}{t}{h}{k}", "план по билетам по {s}{t}{h}{k}",
                        "как готовиться к экзу по {s}{t}{h}{k}? распиши", "разбей билеты по {s} по дням{t}{h}{k}"],
                       split)).format(s=sdat(alias), t=t_txt, h=h_txt, k=skip_spoken)
    args = {"subject": alias} | ({"hours_per_day": hours} if hours else {}) | ({"skip_days": skip} if skip else {})
    msgs = [system(now, p), {"role": "user", "content": q}]
    if give_tickets:
        args["tickets_total"] = tickets
    res = b.make_study_plan(**args)
    msgs += [tc("make_study_plan", args), tr("make_study_plan", res)]
    prefix = ""
    if res.get("error") == "exam_not_found":
        msgs.append({"role": "assistant", "content": pick([f"Не вижу даты экзамена по {dat(alias)} 🤔 Когда он? Запишу и сразу составлю план.",
                                                           f"Сначала нужна дата экзамена по {dat(alias)} — у меня её нет. Скажи, когда он?"])})
        if rnd.random() < 0.4:
            return msgs
        msgs.append({"role": "user", "content": pick([date_ru(exam_day), f"{date_ru(exam_day)} вроде", f"экз {date_ru(exam_day)}",
                                                      f"он {date_ru(exam_day)}"])})
        ea = {"subject": alias, "kind": "экзамен", "date": exam_day.isoformat()}
        msgs += [tc("add_exam", ea), tr("add_exam", b.add_exam(**ea))]
        res = b.make_study_plan(**args)
        msgs += [tc("make_study_plan", args), tr("make_study_plan", res)]
        prefix = f"Экзамен {date_ru(exam_day)} записал 📌 "
    if res.get("error") == "tickets_unknown":
        msgs.append({"role": "assistant", "content": prefix + pick(["Сколько всего билетов (или вопросов)? Тогда распределю по дням 📖",
                                                                    "А сколько билетов? Без этого план не посчитать 🙂"])})
        prefix = ""
        msgs.append({"role": "user", "content": pick([f"{tickets}", f"{tickets} билетов", f"их {tickets}", f"вроде {tickets}"])})
        args = args | {"tickets_total": tickets}
        res = b.make_study_plan(**args)
        msgs += [tc("make_study_plan", args), tr("make_study_plan", res)]
    if res.get("error") == "too_late":
        msgs.append({"role": "assistant", "content": prefix + "Экзамен уже совсем скоро — дней на план не осталось 😬 Повторяй главное и высыпайся!"})
        return msgs
    msgs.append({"role": "assistant", "content": (prefix.strip() + "\n\n" if prefix else "") + plan_answer(res, hours, skip_txt)})
    return msgs


def has_overlap(b, d):
    starts = [l["start"] for l in b.lessons_on(d)]
    return len(starts) != len(set(starts))


def brief_answer(res, now, auto):
    d = dt.date.fromisoformat(res["date"])
    delta = (d - now.date()).days
    dw = {0: "сегодня", 1: "завтра"}[delta]
    if auto:
        out = [pick(["Доброе утро ☀️", "Утро доброе! ☀️", "Привет, просыпаемся ☕"]) + f" Сегодня {res['weekday']}, {res['week']} неделя."]
    else:
        out = [f"{cap(dw)} {res['weekday']}, {date_ru(d)}, {res['week']} неделя."]
    if res["lessons"]:
        lines = []
        for i, l in enumerate(res["lessons"], 1):
            s = lesson_line(i, l, with_teacher=False)
            if l.get("walk_from_prev_minutes"):
                s += f" (идти ~{l['walk_from_prev_minutes']} мин)"
            lines.append(s)
        out.append("📚 Пары:\n" + "\n".join(lines))
    else:
        out.append(f"Пар {dw} нет 🎉")
    if res["homework"]:
        out.append(f"✍️ Д/З на {dw}: " + "; ".join(f"{short(h['subject'])} — {h['task']}" for h in res["homework"]))
    if res["deadlines"]:
        out.append("⏳ " + "; ".join(f"{x['title']} ({short(x['subject'])}) — {in_days(x['days_left'])}" for x in res["deadlines"]))
    if res["exams"]:
        out.append("🎓 " + cap("; ".join(f"{e['kind']} по {dat(short(e['subject']))} {in_days(e['days_left'])}" for e in res["exams"])))
    if not (res["lessons"] or res["homework"] or res["deadlines"] or res["exams"]):
        out.append(pick(["Отдыхай 😌", "Можно выдохнуть 🙂"] + (["Хорошего выходного!"] if d.weekday() >= 5 else [])))
    else:
        late = not auto and now.hour >= 19
        out.append(pick(["Удачи!", "Ты справишься!"] + (["Высыпайся 😴", "Удачи завтра!"] if late else ["Хорошего дня 💪"])))
    return "\n\n".join(out)


def cat_brief(split):
    auto = rnd.random() < 0.65
    day = "today"
    if auto:
        now = dt.datetime.combine(SEM_START + dt.timedelta(days=rnd.randrange((SEM_END - SEM_START).days)), dt.time(7, pick([0, 30, 30])))
    else:
        now = rand_now((7, 23))
        day = "tomorrow" if now.hour >= 20 else pick(["today", "tomorrow"])
        if day == "today":  # сводку на сегодня просят утром, пока пары впереди
            now = now.replace(hour=rnd.randint(7, 11))
    d = now.date() + dt.timedelta(days=day == "tomorrow")
    for _ in range(20):  # профиль без накладок в этот день
        p = rand_profile() or {"subgroup": 1, "english_teacher": ENG[3], "adaptation": False}
        b = Backend(now, p)
        if not has_overlap(b, d):
            break
    for l in rnd.sample(b.lessons_on(d), min(len(b.lessons_on(d)), rnd.choice([0, 0, 1, 2]))):
        if l["subject"] in HW_TASKS and dt.datetime.combine(d, dt.time.fromisoformat(l["start"])) > now:  # Д/З к парам этого дня
            b.homework.append({"subject": SUBJECTS[l["subject"]]["name"], "task": pick(HW_TASKS[l["subject"]]),
                               "due": f"{d} {l['start']}", "expires": (d + dt.timedelta(days=3)).isoformat()})
    for _ in range(rnd.choice([0, 0, 1, 2])):
        title, kind, _, _, code = rand_deadline()
        if all(x["title"] != title or x["subject"] != SUBJECTS[code]["name"] for x in b.deadlines):
            b.add_deadline(title, pick(no_sunday([now.date() + dt.timedelta(days=k) for k in range(1, 13)])).isoformat(), SUBJECTS[code]["name"], kind)
    if (SEM_END - now.date()).days <= 30 and rnd.random() < 0.5:  # конец семестра: зачёты на ближайшие 2 недели
        for code, kind in rnd.sample([x for x in EXAM_POOL if x[1] != "экзамен"], rnd.choice([1, 2])):
            ed = pick(no_sunday([now.date() + dt.timedelta(days=k) for k in range(1, 15)]))
            b.add_exam(SUBJECTS[code]["name"], ed.isoformat(), kind=kind)
    if auto:
        q = "[авто] утренний брифинг"
    else:
        q = pick(split_phr({"today": ["брифинг на сегодня", "что у меня сегодня: пары, дз, дедлайны?", "дай сводку на день",
                                      "сводка на сегодня", "что сегодня по учёбе?", "кинь сводку на сегодня"],
                            "tomorrow": ["дай сводку на завтра", "что у меня завтра: пары, дз, дедлайны?", "брифинг на завтра",
                                         "что завтра по учёбе", "сводка на завтра", "кинь сводку на завтра пж"]}[day], split))
    res = b.get_daily_brief(day)
    return [system(now, p), {"role": "user", "content": q}, tc("get_daily_brief", {"day": day}),
            tr("get_daily_brief", res), {"role": "assistant", "content": brief_answer(res, now, auto)}]


def cat_set_brief(split):
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)
    form = pick(["time", "time", "off", "on"])
    if form == "time":
        h, m = pick([6, 7, 7, 8, 9]), pick([0, 0, 15, 30])
        hhmm = f"{h:02d}:{m:02d}"
        half = {(6, 30): "полседьмого", (7, 30): "полвосьмого", (8, 30): "полдевятого"}.get((h, m))
        spoken = pick([f"в {h}:{m:02d}", f"в {h}:{m:02d}"] + ([f"в {h}", f"в {h} утра"] if m == 0 else []) + ([half] if half else []))
        q = pick(split_phr(["присылай брифинг {t}", "можно брифинг {t}?", "поменяй время брифинга, пусть будет {t}",
                            "утреннюю сводку {t} давай", "брифинг пусть приходит {t}", "хочу брифинг {t}"], split)).format(t=spoken)
        args = {"enabled": True, "time": hhmm}
    elif form == "off":
        q = pick(split_phr(["отключи утренний брифинг", "не присылай больше брифинг", "выключи сводку по утрам",
                            "брифинг не нужен", "хватит слать брифинг по утрам", "убери утреннюю сводку"], split))
        args = {"enabled": False}
    else:
        q = pick(split_phr(["включи утренний брифинг", "хочу получать сводку по утрам", "давай брифинг каждое утро",
                            "верни утренний брифинг", "включи сводку на день", "присылай мне брифинг"], split))
        args = {"enabled": True}
    res = b.set_daily_brief(**args)
    if not res["enabled"]:
        ans = pick(["Отключил брифинг 👌 Захочешь вернуть — просто скажи.", "Готово, по утрам больше не пишу. Включить обратно можно в любой момент.",
                    "Ок, утренней сводки больше не будет 🙂"])
    else:
        t = hm(res["time"])
        ans = pick([f"Готово, брифинг будет приходить в {t} 🌅", f"Ок! Каждое утро в {t} пришлю пары, Д/З и дедлайны ☀️",
                    f"Договорились: сводка на день — в {t} ⏰"])
    return [system(now, p), {"role": "user", "content": q}, tc("set_daily_brief", args), tr("set_daily_brief", res),
            {"role": "assistant", "content": ans}]


EXPLAIN_BANK = None


def cat_explain(split):
    global EXPLAIN_BANK
    if EXPLAIN_BANK is None:
        path = pathlib.Path(__file__).parent / "data" / "explain_bank.json"
        EXPLAIN_BANK = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not EXPLAIN_BANK:
        return None
    it = pick(EXPLAIN_BANK)
    qs = it["questions"]
    q = pick(qs[:-1] if split == "train" and len(qs) > 1 else qs[-1:])
    now, p = rand_now(), rand_profile()
    return [system(now, p), {"role": "user", "content": q}, {"role": "assistant", "content": it["answer"]}]


CHAT = [
    (["привет", "прив", "здарова", "привет, ты кто?", "хай"],
     ["Привет! 👋 Я Ишка, помощник студентов ИШИТР. Подскажу пары, аудитории и как дойти, запишу Д/З, дедлайны и напоминалки, объясню тему. Что нужно?",
      "Привет-привет 🙂 Я Ишка. Могу показать расписание, найти аудиторию, напомнить о Д/З или дедлайне. С чего начнём?"]),
    (["спасибо", "спс", "благодарю", "спасибо большое!", "пасиб"],
     ["Обращайся! 🙌", "Всегда рад помочь 😊", "Не за что, удачи с учёбой! 💪"]),
    (["что ты умеешь?", "что ты можешь", "чем можешь помочь", "какие у тебя функции", "а ты что вообще умеешь"],
     ["Вот что я умею:\n• расписание на любой день и кто ведёт предмет\n• где аудитория, как дойти и успеешь ли до следующей пары\n• где перекусить рядом\n• Д/З (храню неделю и напоминаю перед парой), дедлайны лаб и курсовых\n• напоминалки и заметки\n• экзамены и план повторения\n• объяснить тему простыми словами\n\nСпрашивай 🙂"]),
    (["какой проходной балл был на 09.03.04?", "сколько баллов надо чтобы поступить в ИШИТР", "проходные баллы в тпу какие"],
     ["Проходных баллов у меня нет, а гадать не буду 🙃 Точные цифры — на сайте приёмной комиссии ТПУ. С расписанием, аудиториями и Д/З помогу сразу."]),
    (["какая погода завтра", "будет дождь?", "сколько градусов на улице"],
     ["Погоду я не знаю 🌦️ Глянь прогноз в приложении. А вот во сколько завтра пары и в каких корпусах — подскажу!"]),
    (["сделай за меня лабу по питону", "напиши за меня курсовую", "реши мне контрольную"],
     ["Сделать за тебя не могу 🙂 Но могу объяснить тему, разобрать пример или помочь найти ошибку в твоём коде. С чего начнём?",
      "Тут без читов 😅 Давай лучше разберём, что непонятно, — объясню по шагам."]),
    (["дай номер телефона Ворониной", "какой телефон у преподавателя по матану", "почта Шефера какая"],
     ["Контактов преподавателей у меня нет 🙃 Обычно их пишут на сайте ТПУ или знает староста. Могу подсказать, когда у вас с ним ближайшая пара."]),
    (["когда стипендия", "когда придёт стипуха?", "какая стипендия в тпу"],
     ["Про стипендию у меня данных нет 🤷 Точно ответят в деканате или студенческом офисе. А с расписанием и Д/З помогу!"]),
    (["ты бот?", "ты нейросеть?", "ты человек?"],
     ["Я Ишка — ИИ-помощник студентов ИШИТР 🤖 Не человек, но расписание знаю лучше многих 😉"]),
    (["мне лень учиться", "я устал", "всё бесит, сессия скоро"],
     ["Понимаю, бывает 😮‍💨 Давай маленькими шагами: скажи, что горит сильнее всего, — разложу по дням и поставлю напоминания. А ещё не забывай спать и есть 🍔",
      "Держись 💪 Хочешь, покажу, что реально надо сделать на этой неделе, а остальное подождёт?"]),
]


def cat_chat(split):
    qs, answers = pick(CHAT)
    q = pick(qs[:-1] if split == "train" and len(qs) > 1 else qs[-1:])
    now, p = rand_now(), rand_profile()
    return [system(now, p), {"role": "user", "content": q}, {"role": "assistant", "content": pick(answers)}]


NEW_CATS = [(cat_add_homework, 10), (cat_list_homework, 7), (cat_reminder, 9), (cat_reminder_clarify, 4),
            (cat_note_add, 4), (cat_note_list, 5), (cat_add_deadline, 7), (cat_list_deadlines, 6), (cat_get_exams, 5),
            (cat_study_plan, 6), (cat_brief, 7), (cat_set_brief, 3), (cat_explain, 9), (cat_chat, 5)]


CATS = [(cat_schedule, 30), (cat_room, 18), (cat_teacher, 14), (cat_travel, 12), (cat_food, 12), (cat_add_exam, 7), (cat_profile, 7)]


def build(n, split, cats=CATS):
    out, fns, w = [], [c for c, _ in cats], [w for _, w in cats]
    while len(out) < n:
        m = rnd.choices(fns, w)[0](split)
        if m:
            out.append(m)
    return out


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent / "data"
    # data/sample.jsonl в train не идёт: это образец формата с выдуманными данными (build_samples.py)
    overlay = paraphrase.load()  # перефразированные ответы (только для train)
    # 1) Расписание, аудитории, преподаватели, дорога, еда, экзамены, профиль. Порядок вызовов rnd не меняем:
    #    иначе разъедутся ключи перефразировок в data/paraphrases.json.
    data = {}
    for split, n in (("train", 900), ("eval", 100)):
        rows = build(n, split)
        if split == "train":
            print("перефразировано:", sum(paraphrase.apply(m, overlay) for m in rows))
        rnd.shuffle(rows)
        data[split] = rows
    # 2) Д/З, напоминания, заметки, дедлайны, сессия, брифинг, «объясни тему», болтовня — отдельный генератор.
    rnd = random.Random(2026)
    for split, n in (("train", 650), ("eval", 5)):
        # eval — поровну на каждую категорию (5 × 14), чтобы оценка покрывала все инструменты
        rows = build(n, split, NEW_CATS) if split == "train" else [r for c in NEW_CATS for r in build(n, split, [c])]
        if split == "train":
            print("перефразировано (новые):", sum(paraphrase.apply(m, overlay) for m in rows))
        data[split] += rows
        rnd.shuffle(data[split])
        with open(root / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for m in data[split]:
                f.write(json.dumps({"messages": m}, ensure_ascii=False) + "\n")
        print(split, len(data[split]))
