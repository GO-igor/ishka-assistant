"""Генерация train/eval из реальной базы: вопросы по шаблонам -> настоящий вызов backend -> ответ по шаблону.
python gen_dataset.py  ->  data/train.jsonl, data/eval.jsonl
Формулировки делятся на train и eval (последние 2 в каждом списке — только eval), чтобы eval проверял обобщение.
Перед обучением ответы стоит перефразировать большой LLM и выборочно проверить руками."""
import datetime as dt, json, random, pathlib
import paraphrase, prompt_format
from backend import Backend, SUBJECTS, WD_RU, WEEKDAYS, HOLIDAYS, EVENT_KINDS, resolve_subject
from conversation import _matches, _stems
from dates import remind_time
from store import Store

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


STAROSTA = False  # rand_profile() даёт профиль старосты (примеры, где староста просит что-то для себя)


def rand_profile():
    if rnd.random() < 0.12:
        return {"role": "starosta"} if STAROSTA else {}
    p = {"subgroup": rnd.randint(1, 3), "english_teacher": pick(ENG), "adaptation": rnd.random() < 0.3}
    return p | {"role": "starosta"} if STAROSTA else p


def profile_text(p):
    if not any(k in p for k in ("subgroup", "english_teacher", "adaptation")):
        s = "не заполнен"
    else:
        s = (f"подгруппа {p.get('subgroup') or 'не указана'}, английский — {p.get('english_teacher') or 'не указан'}, "
             f"курсы (А): {'да' if p.get('adaptation') else 'нет'}")
    return s + "; староста группы" if p.get("role") == "starosta" else s


def system(now, p):
    return {"role": "system", "content":
            "Ты — Ишка, помощник студентов ИШИТР ТПУ. Общайся на «ты», коротко, дружелюбно, можно лёгкий юмор и "
            "эмодзи, без канцелярита. Факты о расписании, корпусах, преподавателях, сессии, мероприятиях, заметках и "
            "напоминаниях бери ТОЛЬКО из инструментов, ничего не выдумывай. "
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
            ans += ("\n\nЯ показал пары для всех групп: в твоём аккаунте не указаны подгруппа или преподаватель английского. "
                    "Попроси того, кто выдавал аккаунт, дописать их — тогда буду показывать только твои пары.")
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


def skip_add_exam(split):
    """Было «запиши экзамен» (add_exam). Функции больше нет, но случайные числа тратим ровно как раньше: иначе
    у остальных примеров этого раздела сдвинутся формулировки и пропадут их перефразировки (data/paraphrases.json)."""
    rand_now(), rand_profile()
    code = pick(["python", "inf", "math", "matlog"])
    pick(SUBJECTS[code]["aliases"]), rnd.randint(9, 28), pick([None] * 4), pick(split_phr([None] * 6, split))
    return None


def skip_profile(split):
    """Было «запомни мою подгруппу» (set_profile): профиль теперь берётся из аккаунта. Случайные числа — как раньше."""
    rand_now(), rnd.randint(1, 3), pick(ENG), rnd.random(), pick(split_phr([None] * 6, split))
    return None


# ================= заметки и напоминания, напоминания о Д/З, сессия, брифинг, ответы без инструментов =================
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
# Заметки и как студент их называет: (что говорит студент, что передать в query)
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


def no_sunday(days):
    """Только учебные дни: без воскресений и праздников."""
    return [d for d in days if d.weekday() != 6 and d.isoformat() not in HOLIDAYS]


def account_backend(now, p, role=None):
    """Студент вошёл в аккаунт: база в памяти с одним аккаунтом (заметки и напоминания хранятся в ней)."""
    role = role or p.get("role") or "student"
    st = Store(":memory:")
    me = st.add_account("me", None, "8К51", role=role, subgroup=p.get("subgroup"),
                        english_teacher=p.get("english_teacher"), adaptation=p.get("adaptation"))
    return Backend(now, {k: v for k, v in p.items() if k != "role"} | {"group": "8К51", "role": role}, st, me)


def reminder_phrase(now):
    """-> (фраза времени, аргументы). Время срабатывания — с 8 до 22, «в 15» — всегда ближайшее будущее."""
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
        fire, _ = remind_time(now, **args)
        if fire and 8 <= fire.hour <= 22:
            return spoken, args
    raise RuntimeError("не нашлось подходящего времени")


def reminder_answer(now, res, what):
    w = when_ru(now, res["fire_at"])
    return pick([f"Ок, {w} напомню {what} ⏰", f"Готово: {w} — напоминание «{what}» 🔔",
                 f"Поставил ⏰ {cap(w)} напомню {what}.", f"Напомню {what} {w} 👌", f"Есть, напоминалка на {w.replace('сегодня в ', '').removeprefix('в ')}: {what} 🔔"])


def note_answer(text):
    return pick(["Сохранил в заметки 📝", "Записал 📝 Найдёшь в заметках.", "Готово, заметка сохранена ✍️",
                 f"Записал в заметки: «{text}» 📝", "Запомнил 👌 Спроси «что у меня записано» — покажу."])


REMIND_PHR = ["напомни {w} {x}", "{w} напомни {x}", "поставь напоминалку {w}: {x}", "напомни мне {x} {w}",
              "кинь напоминание {w} — {x}", "не дай забыть {x}, напомни {w}", "запиши {x} и напомни {w}",
              "напоминание {w}: {x}", "можешь {w} напомнить {x}?", "запиши в заметки: {x}, напомни {w}"]


def cat_note_timed(split):
    """Время названо — сразу напоминание."""
    now, p = rand_now((7, 21)), rand_profile()
    b = account_backend(now, p)
    what = pick(REM_TASKS)
    wp, targs = reminder_phrase(now)
    q = pick(split_phr(REMIND_PHR, split)).format(w=wp, x=what)
    args = {"text": cap(what)} | targs
    res = b.add_note(**args)
    return [system(now, p), {"role": "user", "content": q}, tc("add_note", args), tr("add_note", res),
            {"role": "assistant", "content": reminder_answer(now, res, what)}]


ASK_TIME = {"remind": ["Когда напомнить? Например «через час», «завтра в 9:00» или «в пятницу в 18:00». Или просто записать в заметки?",
                       "Ок! А когда? Можно «через 30 минут», «в 19:00» или «завтра в 8». Если напоминать не надо — просто сохраню.",
                       "Во сколько напомнить? 🙂 Скажи время или «через сколько» — или «не надо», и я просто запишу."],
            "note": ["Записываю 📝 Напомнить в какое-то время? Скажи когда — или «не надо», и просто сохраню заметку.",
                     "Ок! Напомнить об этом? Скажи время (например «завтра в 9») или «просто запиши».",
                     "Сохраню 🙂 Нужно напоминание? Назови время — или скажи «без времени»."]}
NO_TIME = ["просто запиши", "без времени", "не, просто сохрани", "не напоминай, просто запиши", "напоминать не надо",
           "просто в заметки", "не, без напоминания", "просто сохрани"]


def cat_note_clarify(split):
    """Время не названо: Ишка обязательно переспрашивает. Назвал — напоминание, отказался — просто заметка."""
    now, p = rand_now((7, 21)), rand_profile()
    b = account_backend(now, p)
    if rnd.random() < 0.5:  # просит напомнить, а когда — не сказал
        intent, what = "remind", pick(REM_TASKS)
        q = pick(split_phr(["напомни {x}", "поставь напоминание {x}", "не дай мне забыть {x}", "надо не забыть {x}",
                            "напоминалку сделай: {x}", "хочу напоминалку: {x}", "напомни мне {x} пж", "поставь напоминалку: {x}"],
                           split)).format(x=what)
    else:  # просит записать
        intent, what = "note", pick(NOTES)[0]
        q = pick(split_phr(["запиши в заметки: {x}", "сохрани заметку: {x}", "заметка: {x}", "запомни: {x}",
                            "добавь в заметки {x}", "запиши: {x}", "занеси в заметки — {x}", "сохрани: {x}"], split)).format(x=what)
    msgs = [system(now, p), {"role": "user", "content": q}, {"role": "assistant", "content": pick(ASK_TIME[intent])}]
    if rnd.random() < (0.75 if intent == "remind" else 0.35):
        wp, targs = reminder_phrase(now)
        msgs.append({"role": "user", "content": wp})
        args = {"text": cap(what)} | targs
        res = b.add_note(**args)
        return msgs + [tc("add_note", args), tr("add_note", res), {"role": "assistant", "content": reminder_answer(now, res, what)}]
    msgs.append({"role": "user", "content": say(split, ["не надо", "нет"], NO_TIME)})
    args = {"text": cap(what)}
    res = b.add_note(**args)
    return msgs + [tc("add_note", args), tr("add_note", res), {"role": "assistant", "content": note_answer(cap(what))}]


DAY_ONLY_PHR = ["напомни {d} {x}", "{d} напомни {x}", "напомни мне {x} {d}", "не дай забыть {x} {d}", "можешь {d} напомнить {x}?",
                "поставь напоминание {d}: {x}", "кинь напоминалку {d} — {x}", "{d} надо {x}, напомни", "напомни {x} {d} пж",
                "запиши и напомни {d}: {x}"]
ASK_HOUR = ["Во сколько напомнить {d}? 🙂", "Ок! А во сколько {d}? Например «в 9» или «в 18:30».", "{D} — понял. Во сколько напомнить?",
            "Напомню {d} 👌 Скажи только время: «в 10», «в 19:00»…"]


def cat_note_day_only(split):
    """День назван, а час нет («напомни в пятницу сдать отчёт»): Ишка спрашивает время. Иногда модель сначала зовёт
    add_note без часа, получает bad_time — и тоже спрашивает, а не сохраняет просто заметку."""
    now, p = rand_now((7, 21)), rand_profile()
    b = account_backend(now, p)
    what = pick(REM_TASKS)
    form = pick(["tomorrow", "weekday", "weekday", "date"])
    if form == "tomorrow":
        spoken, day = "завтра", "tomorrow"
    elif form == "weekday":
        wd = pick([x for x in range(7) if x != now.weekday()])
        spoken, day = WEEKDAYS_FUT[wd], WEEKDAYS[wd]
    else:
        d = now.date() + dt.timedelta(days=rnd.randint(3, 40))
        spoken, day = date_ru(d), d.isoformat()
    h, m = rnd.randint(8, 21), pick([0, 0, 0, 30, 15])
    if m:
        t = pick([f"в {h}:{m:02d}", f"{h}:{m:02d}"])
    else:
        t = pick([f"в {h}", f"в {h}:00", f"{h}:00", f"часов в {h}"] + ([f"в {h} утра"] if h < 12 else [f"в {h - 12} вечера"] if h >= 17 else []))
    msgs = [system(now, p), {"role": "user", "content": pick(split_phr(DAY_ONLY_PHR, split)).format(d=spoken, x=what)}]
    if rnd.random() < 0.2:
        bad = {"text": cap(what), "at": day}
        msgs += [tc("add_note", bad), tr("add_note", b.add_note(**bad))]
    msgs += [{"role": "assistant", "content": pick(ASK_HOUR).format(d=spoken, D=cap(spoken))}, {"role": "user", "content": t}]
    args = {"text": cap(what), "at": f"{day} {h:02d}:{m:02d}"}
    res = b.add_note(**args)
    if not res.get("ok"):
        return None
    return msgs + [tc("add_note", args), tr("add_note", res), {"role": "assistant", "content": reminder_answer(now, res, what)}]


def hw_rem_answer(now, res, task):
    cls, f = when_ru(now, res["class_at"]), when_ru(now, res["fire_at"])
    d = dat(short(res["subject"]))
    t = f": {task}" if task else ""
    day_before = res["fire_at"][11:] == "19:00" and res["fire_at"][:10] < res["class_at"][:10]
    if not day_before:  # накануне вечером уже поздно
        return pick([f"Накануне уже не успею, так что напомню {f} — за час до пары по {d} ({cls}) ⏰ Д/З{t}",
                     f"Пара по {d} {cls}, вечер перед ней уже прошёл — напомню про Д/З{t} {f}, за час до начала 🔔"])
    return pick([f"Готово 📚 Пара по {d} {cls}, напомню про Д/З{t} накануне — {f}.",
                 f"Ок! Напомню {f} — накануне пары по {d} ({cls}). Д/З{t} 🔔",
                 f"Поставил ⏰ {cap(f)} напомню про Д/З по {d}{t}. Пара — {cls}.",
                 f"Есть! Д/З по {d}{t} — напомню {f}, пара {cls} 📚"])


HW_REM_PHR = ["сделай напоминание о дз по {sd}: {t}", "напомни про дз по {sd} — {t}", "напомни сделать дз по {sd}: {t}",
              "поставь напоминалку про домашку по {sd}: {t}", "задали по {sd} {t}, напомни перед парой",
              "дз по {sd}: {t}, напомни накануне", "напомни о домашке по {sd}: {t}", "не дай забыть дз по {sd}: {t}",
              "сделай напоминание о ДЗ по {sd}: {t}", "напомни накануне пары по {sd}: {t}"]
HW_REM_PHR_NO_TASK = ["сделай напоминание о дз по {sd}", "напомни про дз по {sd}", "напомни сделать домашку по {sd}",
                      "поставь напоминание о дз по {sd}", "напомни накануне пары по {sd} про дз", "хочу напоминание о домашке по {sd}",
                      "сделай напоминание о ДЗ по {sd}", "напоминалку про дз по {sd} сделай"]


def cat_hw_reminder(split):
    """«Сделай напоминание о ДЗ по матану»: накануне ближайшей пары по предмету."""
    now, p = rand_now((7, 22)), rand_profile() or {"subgroup": 1, "english_teacher": ENG[0], "adaptation": False}
    b = account_backend(now, p)
    subs = student_subjects(b)
    if rnd.random() < 0.08 or not subs:  # предмета нет в расписании
        alias, task = pick(UNKNOWN_HW)
        q = pick(split_phr(HW_REM_PHR, split)).format(sd=sdat(alias), t=task)
        args = {"subject": alias, "task": task}
        res = b.add_homework_reminder(**args)
        msgs = [system(now, p), {"role": "user", "content": q}, tc("add_homework_reminder", args), tr("add_homework_reminder", res),
                {"role": "assistant", "content": pick([
                    f"Пар по предмету «{alias}» в твоём расписании не нашёл 🤔 Могу поставить обычное напоминание — скажи, когда напомнить.",
                    f"Хм, предмета «{alias}» в твоём расписании нет 🤔 Давай поставлю простое напоминание: когда напомнить?"])}]
        if rnd.random() < 0.6:
            wp, targs = reminder_phrase(now)
            msgs.append({"role": "user", "content": wp})
            text = f"Д/З по {dat(alias)}: {task}"
            args = {"text": text} | targs
            res = b.add_note(**args)
            w = when_ru(now, res["fire_at"])
            msgs += [tc("add_note", args), tr("add_note", res), {"role": "assistant", "content": pick([
                f"Поставил ⏰ {cap(w)} напомню про Д/З по {dat(alias)}: {task}.",
                f"Готово 🔔 Напомню {w}: Д/З по {dat(alias)} — {task}."])}]
        return msgs
    code = pick(subs)
    alias, task = sub_alias(code), pick(HW_TASKS[code]) if rnd.random() < 0.7 else None
    q = pick(split_phr(HW_REM_PHR if task else HW_REM_PHR_NO_TASK, split)).format(sd=sdat(alias), t=task)
    args = {"subject": alias} | ({"task": task} if task else {})
    res = b.add_homework_reminder(**args)
    if not res.get("ok"):
        return None
    return [system(now, p), {"role": "user", "content": q}, tc("add_homework_reminder", args), tr("add_homework_reminder", res),
            {"role": "assistant", "content": hw_rem_answer(now, res, task)}]


HW_REM_BARE = ["сделай напоминание о дз", "напомни про дз", "напомни про домашку", "поставь напоминание о домашке",
               "хочу напоминалку про дз", "напомни сделать домашку", "сделай напоминание о ДЗ", "напоминание о дз сделай пж",
               "напомни мне про домашнее задание", "не дай забыть про дз"]
ASK_SUBJECT = ["По какому предмету? 📚 Можешь сразу написать, что задали — например «матан, №5–10».",
               "Ок! По какому предмету Д/З? И что задали, если хочешь — добавлю в напоминание.",
               "Давай 🙂 Какой предмет? Напомню накануне пары по нему."]


def cat_hw_reminder_ask(split):
    """«Сделай напоминание о ДЗ» без предмета: Ишка спрашивает предмет, потом ставит напоминание накануне пары."""
    now, p = rand_now((7, 22)), rand_profile() or {"subgroup": 1, "english_teacher": ENG[0], "adaptation": False}
    b = account_backend(now, p)
    subs = student_subjects(b)
    if not subs:
        return None
    code = pick(subs)
    alias, task = sub_alias(code), pick(HW_TASKS[code]) if rnd.random() < 0.5 else None
    form = pick(["{a}: {t}", "по {sd}: {t}", "{a}, {t}", "{t} по {sd}"] if task else ["{a}", "по {sd}", "{a} 🙏", "по {sd} пж"])
    msgs = [system(now, p), {"role": "user", "content": pick(split_phr(HW_REM_BARE, split))},
            {"role": "assistant", "content": pick(ASK_SUBJECT)},
            {"role": "user", "content": form.format(a=alias, sd=sdat(alias), t=task)}]
    args = {"subject": alias} | ({"task": task} if task else {})
    res = b.add_homework_reminder(**args)
    if not res.get("ok"):
        return None
    return msgs + [tc("add_homework_reminder", args), tr("add_homework_reminder", res),
                   {"role": "assistant", "content": hw_rem_answer(now, res, task)}]


def seed_notes(b, k_notes, k_rems, k_hw=0):
    """Записи студента: заметки из NOTES, напоминания из REM_TASKS, напоминания о Д/З."""
    for x, _ in rnd.sample(NOTES, k_notes):
        b.add_note(cap(x))
    seed_rems(b, k_rems)
    subs = student_subjects(b)
    for code in rnd.sample(subs, min(k_hw, len(subs))):
        b.add_homework_reminder(SUBJECTS[code]["name"], pick(HW_TASKS[code]) if rnd.random() < 0.8 else None)


def seed_rems(b, k, tasks=None):
    for what_ in rnd.sample(tasks or REM_TASKS, k):
        at = pick(["", "tomorrow ", pick(WEEKDAYS) + " "]) + f"{rnd.randint(8, 21):02d}:{pick(['00', '15', '30', '45'])}"
        b.add_note(cap(what_), at=at)


def item_line(now, x):
    """Строка списка записей: «2. завтра в 9:00 — Позвонить в деканат», «3. Купить флешку»."""
    n = f"{x['n']}. "
    if x.get("subject"):
        return n + f"{when_ru(now, x['fire_at'])} — Д/З по {dat(short(x['subject']))}" + (f": {x['text']}" if x.get("text") else "")
    if x.get("fire_at"):
        return n + f"{when_ru(now, x['fire_at'])} — {x['text']}"
    return n + x["text"]


def item_lines(now, items):
    return "\n".join(item_line(now, x) for x in items)


def starosta_item(now, x):
    """Одно объявление или Д/З от старосты."""
    if x["kind"] == "homework":
        return f"Д/З по {dat(short(x['subject']))} — {x['task']} ({when_ru(now, x['due'])})"
    return x["text"] + (f" — {when_ru(now, x['at'])}" if x.get("at") else "")


def starosta_block(now, news):
    return "📢 От старосты:\n" + "\n".join(f"• {starosta_item(now, x)}" for x in news)


LIST_PHR = {"all": ["что у меня записано?", "покажи заметки и напоминания", "мои заметки", "покажи мои записи",
                    "что я просил запомнить?", "список заметок", "открой заметки", "заметки и напоминалки покажи",
                    "что я тебе говорил запомнить", "покажи заметки"],
            "reminders": ["какие напоминалки стоят?", "какие напоминания есть", "что ты мне должен напомнить?",
                          "покажи напоминания", "есть напоминалки?", "о чём напомнишь?", "мои напоминания", "список напоминаний"]}


def notes_answer(now, res, intent):
    items, news = res["items"], res.get("from_starosta")
    if intent == "reminders":
        if not items:
            return pick(["Напоминаний нет 🙂", "Сейчас ни одного напоминания не стоит. Поставить?"])
        return "⏰ Напоминания:\n" + item_lines(now, items)
    parts = ["📝 Твои записи:\n" + item_lines(now, items)] if items else []
    if news:
        parts.append(starosta_block(now, news))
    if not parts:
        return pick(["Пока ничего не записано 📭 Скажи «запиши …» или «напомни …» — сохраню.", "Пусто — ни заметок, ни напоминаний 🙃"])
    if not items:
        parts.insert(0, "Своих записей пока нет 📭")
    return "\n\n".join(parts)


def cat_note_list(split, posts=False):
    now, p = rand_now(), rand_profile()
    if posts:  # студент в группе, староста что-то отправил
        b = group_backend(now, p, "starosta" if rnd.random() < 0.15 else "student")
        p = b.profile
    else:
        b = account_backend(now, p)
    seed_notes(b, rnd.choice([0, 1, 2, 3]), rnd.choice([0, 1, 2]), rnd.choice([0, 0, 1]))
    if posts:
        seed_posts(b, k=rnd.choice([1, 1, 2]))
    intent = pick(["all", "all", "reminders"])
    q = pick(split_phr(LIST_PHR[intent], split))
    args = {"kind": "reminder"} if intent == "reminders" else {}
    res = b.list_notes(**args)
    return [system(now, p), {"role": "user", "content": q}, tc("list_notes", args), tr("list_notes", res),
            {"role": "assistant", "content": notes_answer(now, res, intent)}]


EXAM_CODES = ["python", "inf", "math", "matlog"]


def rand_session(now):
    """Сессия в январе (как в db/session.json): экзамены — часть с датами, часть ещё без."""
    start = pick([dt.date(2027, 1, d) for d in (9, 11, 12, 13, 18)])
    end = start + dt.timedelta(days=pick([14, 18, 20]))
    days = no_sunday([start + dt.timedelta(days=k) for k in range((end - start).days + 1)])
    exams = []
    for code in rnd.sample(EXAM_CODES, rnd.choice([2, 3, 4, 4])):
        e = {"subject": code, "kind": "экзамен", "date": None}
        if rnd.random() < 0.7:
            d = pick(days)
            days.remove(d)
            e["date"] = d.isoformat()
            if rnd.random() < 0.6:
                e["time"] = pick(["09:00", "10:25", "12:40"])
            if rnd.random() < 0.4:
                e["building"], e["room"] = pick([r for r in ROOMS if r[1][:1].isdigit()])  # не спортзал
        exams.append(e)
    return {"start": start.isoformat(), "end": end.isoformat(), "exams": exams}


CREDIT_CODES = [("hist", "зачёт"), ("intro", "дифзачёт"), ("eng", "зачёт"), ("proj", "зачёт")]
EXAM_GEN = {"экзамен": "экзамена", "зачёт": "зачёта", "дифзачёт": "дифзачёта"}


def add_credits(session):
    """Зачёты и дифзачёты (db/session.json это позволяет) — в зачётную неделю перед сессией, часть ещё без даты."""
    days = no_sunday([dt.date(2026, 12, 21) + dt.timedelta(days=k) for k in range(10)])
    for code, kind in rnd.sample(CREDIT_CODES, rnd.randint(1, 4)):
        e = {"subject": code, "kind": kind, "date": None}
        if rnd.random() < 0.7:
            d = pick(days)
            days.remove(d)
            e["date"] = d.isoformat()
            if rnd.random() < 0.5:
                e["time"] = pick(["10:25", "12:40", "14:35"])
        session["exams"].append(e)


def exam_line(e):
    d = dt.date.fromisoformat(e["date"])
    t = f" в {e['time']}" if e.get("time") else ""
    return f"{WD_SHORT[d.weekday()]} {date_ru(d)}{t}{where(e)}"


def cat_session(split, mixed=False):
    """mixed — в сессии ещё зачёты и дифзачёты (раздел 6); без него порядок случайных чисел прежний (раздел 2)."""
    now, p = rand_now(), rand_profile()
    b = Backend(now, p)
    b.session = rand_session(now)
    if mixed:
        add_credits(b.session)
    if rnd.random() < 0.1:  # сессия уже идёт
        a = dt.date.fromisoformat(b.session["start"])
        b.now = now = dt.datetime.combine(a + dt.timedelta(days=rnd.randint(0, 10)), now.time())
    if mixed and rnd.random() < 0.5:
        e0 = pick(b.session["exams"]) if rnd.random() < 0.85 else {"subject": pick(["pe", "kod"]), "kind": "зачёт"}
        code, kind = e0["subject"], e0["kind"]
        alias = sub_alias(code)
        q = pick(split_phr(["когда {k} по {sd}?", "{k} по {sd} когда", "{s} — {k} какого числа?", "через сколько дней {k} по {sd}",
                            "когда сдаём {s}?", "{k} по {sd} уже назначили?", "когда {k} по {sd}", "дата {kg} по {sd}?"],
                           split)).format(k=kind, kg=EXAM_GEN[kind], s=alias, sd=sdat(alias))
        args = {"subject": alias}
    elif not mixed and rnd.random() < 0.4:
        code = pick(EXAM_CODES + ["hist", "intro"])
        alias = sub_alias(code)
        q = pick(split_phr(["когда экзамен по {sd}?", "сколько до экза по {sd}", "экзамен по {sd} когда", "{s} экзамен какого числа",
                            "через сколько дней экзамен по {sd}", "когда сдаём {s}?", "напомни дату экзамена по {sd}",
                            "экз по {sd} уже назначили?"], split)).format(s=alias, sd=sdat(alias))
        args = {"subject": alias}
    else:
        q = pick(split_phr(["сколько до сессии", "когда сессия", "сколько дней до сессии осталось", "когда экзамены",
                            "какие экзамены и когда", "расписание экзаменов", "когда экзы", "скоро сессия?",
                            "сколько осталось до экзаменов", "покажи экзамены"], split))
        args = {}
    res = b.get_session(**args)
    start, end = dt.date.fromisoformat(res["start"]), dt.date.fromisoformat(res["end"])
    head = (f"Сессия уже идёт, до {date_ru(end)} 💪" if res["status"] == "now" else
            f"До сессии {days_n(res['days_left'])} ⏳ Она с {date_ru(start)} по {date_ru(end)}.")
    if res.get("error") == "no_exam" and mixed:
        ans = f"По {dat(alias)} в списке сессии нет ни экзамена, ни зачёта 🙂 " + (
            f"Сессия начнётся {date_ru(start)}, через {days_n(res['days_left'])}." if res["status"] == "soon" else f"Сессия идёт до {date_ru(end)}.")
    elif res.get("error") == "no_exam":
        ans = f"По {dat(alias)} экзамена в сессии нет 🙂 " + (f"Сессия начнётся {date_ru(start)}, через {days_n(res['days_left'])}."
                                                            if res["status"] == "soon" else f"Сессия идёт до {date_ru(end)}.")
    elif args:
        e = res["exams"][0] if res["exams"] else None
        if e and e.get("date"):
            ans = f"{cap(e['kind'])} по {dat(short(e['subject']))} — {exam_line(e)}, {in_days(e['days_left'])} ⏳"
        elif e:
            ans = (f"Дату {EXAM_GEN[e['kind']]} по {dat(short(e['subject']))} пока не объявили 🤷 "
                   + (f"Сессия начнётся {date_ru(start)}, через {days_n(res['days_left'])}." if res["status"] == "soon"
                      else f"Сессия уже идёт, до {date_ru(end)}."))
        else:
            ans = f"{cap(kind) if mixed else 'Экзамен'} по {dat(short(SUBJECTS[code]['name']))} уже позади 🎉"
    else:
        dated = [e for e in res["exams"] if e.get("date")]
        undated = [short(e["subject"]) + (f" ({e['kind']})" if mixed else "") for e in res["exams"] if not e.get("date")]
        lines = [f"• {exam_line(e)} — {short(e['subject'])}, {e['kind']} ({in_days(e['days_left'])})" for e in dated]
        ans = head + ("\n\n" + ("Что сдаём:\n" if mixed else pick(["Экзамены:\n", "Что сдаём:\n"])) + "\n".join(lines) if lines else "")
        if undated and mixed:
            ans += ("\n\n" if lines else " ") + (f"Пока без даты: {undated[0]}." if len(undated) == 1
                                                 else f"Даты пока не объявили: {', '.join(undated)}.")
        elif undated:
            ans += ("\n\n" if lines else " ") + (f"Дату экзамена по предмету «{undated[0]}» пока не объявили." if len(undated) == 1
                                                 else f"Даты экзаменов пока не объявили: {', '.join(undated)}.")
    return [system(now, p), {"role": "user", "content": q}, tc("get_session", args), tr("get_session", res),
            {"role": "assistant", "content": ans}]


def has_overlap(b, d):
    starts = [l["start"] for l in b.lessons_on(d)]
    return len(starts) != len(set(starts))


def brief_answer(res, now, auto):
    d = dt.date.fromisoformat(res["date"])
    dw = {0: "сегодня", 1: "завтра"}[(d - now.date()).days]
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
    if res.get("from_starosta"):  # отдельным полем: всё от старосты, ниже — напоминания
        out.append(starosta_block(now, res["from_starosta"]))
    if res.get("reminders"):
        out.append("⏰ Напоминания:\n" + item_lines(now, res["reminders"]))
    if not (res["lessons"] or res.get("from_starosta") or res.get("reminders")):
        out.append(pick(["Отдыхай 😌", "Можно выдохнуть 🙂"] + (["Хорошего выходного!"] if d.weekday() >= 5 else [])))
    else:
        late = not auto and now.hour >= 19
        out.append(pick(["Удачи!", "Ты справишься!"] + (["Высыпайся 😴", "Удачи завтра!"] if late else ["Хорошего дня 💪"])))
    return "\n\n".join(out)


def cat_brief(split, posts=False):
    """posts=True — студент в группе, и староста что-то отправил."""
    auto = rnd.random() < 0.65
    day = "today"
    if auto:
        now = dt.datetime.combine(SEM_START + dt.timedelta(days=rnd.randrange((SEM_END - SEM_START).days)), dt.time(7, 30))
    else:
        now = rand_now((7, 23))
        day = "tomorrow" if now.hour >= 20 else pick(["today", "tomorrow"])
        if day == "today":  # сводку на сегодня просят утром, пока пары впереди
            now = now.replace(hour=rnd.randint(7, 11))
    d = now.date() + dt.timedelta(days=day == "tomorrow")
    for _ in range(20):  # профиль без накладок в этот день
        p = rand_profile() or {"subgroup": 1, "english_teacher": ENG[3], "adaptation": False}
        if not has_overlap(Backend(now, p), d):
            break
    if posts:
        b = group_backend(now, p, "starosta" if rnd.random() < 0.15 else "student")
        p = b.profile
    else:
        b = account_backend(now, p)
    seed_notes(b, rnd.choice([0, 1, 2]), rnd.choice([0, 0, 1, 2, 3]), rnd.choice([0, 0, 1, 2]))
    if posts:
        seed_posts(b)
    if auto:
        q = "[авто] утренний брифинг"
    else:
        q = pick(split_phr({"today": ["брифинг на сегодня", "проведи брифинг", "дай сводку на день", "что у меня сегодня?",
                                      "сводка на сегодня", "что сегодня по учёбе?", "кинь сводку на сегодня", "брифинг"],
                            "tomorrow": ["дай сводку на завтра", "что у меня завтра?", "брифинг на завтра", "проведи брифинг на завтра",
                                         "что завтра по учёбе", "сводка на завтра", "кинь сводку на завтра пж", "завтра что по плану?"]}[day], split))
    res = b.get_daily_brief(day)
    return [system(now, p), {"role": "user", "content": q}, tc("get_daily_brief", {"day": day}),
            tr("get_daily_brief", res), {"role": "assistant", "content": brief_answer(res, now, auto)}]


CHAT = [
    (["привет", "прив", "здарова", "привет, ты кто?", "хай"],
     ["Привет! 👋 Я Ишка, помощник студентов ИШИТР. Подскажу пары, аудитории и как дойти, где перекусить, запишу заметки и напоминалки. Что нужно?",
      "Привет-привет 🙂 Я Ишка. Могу показать расписание, найти аудиторию, напомнить о Д/З накануне пары. С чего начнём?"]),
    (["спасибо", "спс", "благодарю", "спасибо большое!", "пасиб"],
     ["Обращайся! 🙌", "Всегда рад помочь 😊", "Не за что, удачи с учёбой! 💪"]),
    (["что ты умеешь?", "что ты можешь", "чем можешь помочь", "какие у тебя функции", "а ты что вообще умеешь"],
     ["Вот что я умею:\n• расписание на любой день и кто ведёт предмет\n• где аудитория, как дойти и успеешь ли до следующей пары\n• где перекусить рядом\n• сколько до сессии и когда экзамены\n• мероприятия и хакатоны для ИШИТР\n• заметки и напоминания, напоминание о Д/З накануне пары\n• утренний брифинг в 7:30: пары, объявления старосты и твои напоминания\n\nСпрашивай 🙂"]),
    (["какой проходной балл был на 09.03.04?", "сколько баллов надо чтобы поступить в ИШИТР", "проходные баллы в тпу какие"],
     ["Проходных баллов у меня нет, а гадать не буду 🙃 Точные цифры — на сайте приёмной комиссии ТПУ. С расписанием, аудиториями и напоминаниями помогу сразу."]),
    (["какая погода завтра", "будет дождь?", "сколько градусов на улице"],
     ["Погоду я не знаю 🌦️ Глянь прогноз в приложении. А вот во сколько завтра пары и в каких корпусах — подскажу!"]),
    (["сделай за меня лабу по питону", "напиши за меня курсовую", "реши мне контрольную"],
     ["Сделать за тебя не могу 🙂 Но могу напомнить про Д/З накануне пары, чтобы ты точно успел. Поставить напоминание?",
      "Тут без читов 😅 Если что-то непонятно, лучше спроси преподавателя — подсказать, когда у вас ближайшая пара?"]),
    (["дай номер телефона Ворониной", "какой телефон у преподавателя по матану", "почта Шефера какая"],
     ["Контактов преподавателей у меня нет 🙃 Обычно их пишут на сайте ТПУ или знает староста. Могу подсказать, когда у вас с ним ближайшая пара."]),
    (["когда стипендия", "когда придёт стипуха?", "какая стипендия в тпу"],
     ["Про стипендию у меня данных нет 🤷 Точно ответят в деканате или студенческом офисе. А с расписанием и напоминаниями помогу!"]),
    (["ты бот?", "ты нейросеть?", "ты человек?"],
     ["Я Ишка — ИИ-помощник студентов ИШИТР 🤖 Не человек, но расписание знаю лучше многих 😉"]),
    (["мне лень учиться", "я устал", "всё бесит, сессия скоро"],
     ["Понимаю, бывает 😮‍💨 Давай маленькими шагами: скажи, что горит сильнее всего, — поставлю напоминания. А ещё не забывай спать и есть 🍔",
      "Держись 💪 Хочешь, проведу брифинг: что сегодня по парам и о чём я должен напомнить?"]),
    # функции, которых больше нет: вежливо отказываем и говорим, что можно вместо них
    (["объясни, что такое рекурсия", "расскажи про производную простыми словами", "что такое ДНФ? объясни",
      "объясни тему по матлогике", "можешь объяснить циклы в питоне?"],
     ["Объяснять темы я не умею 🙂 Лучше спроси преподавателя на паре или на консультации — могу подсказать, когда у тебя ближайшая пара.",
      "Темы я не объясняю — я про расписание, аудитории и напоминания 🙂 Спроси преподавателя или одногруппников, а я подскажу, когда ближайшая пара."]),
    (["составь план подготовки к экзамену", "распиши билеты по дням", "сделай план повторения по матану"],
     ["План подготовки я не составляю 🙂 Но могу сказать, сколько дней до сессии и когда экзамены, и поставить напоминания на дни, когда будешь заниматься."]),
    (["запиши экзамен по питону на 15 января", "добавь экзамен по матану", "внеси дату экзамена по инфе"],
     ["Даты экзаменов я беру из расписания сессии — сам их не вношу 🙂 Могу поставить тебе напоминание: скажи, о чём и когда напомнить."]),
    (["присылай брифинг в 8", "отключи утренний брифинг", "поменяй время брифинга"],
     ["Брифинг приходит каждое утро в 7:30, время пока не меняется 🙂 А сводку можно попросить в любой момент: «брифинг на сегодня»."]),
    (["я во второй подгруппе", "поменяй мне подгруппу на 3", "у меня английский у Ростовцевой"],
     ["Подгруппа и преподаватель английского берутся из твоего аккаунта — в чате я их не меняю 🙂 Попроси того, кто выдавал аккаунт, поправить."]),
]


def cat_chat(split):
    qs, answers = pick(CHAT)
    q = pick(qs[:-1] if split == "train" and len(qs) > 1 else qs[-1:])
    now, p = rand_now(), rand_profile()
    return [system(now, p), {"role": "user", "content": q}, {"role": "assistant", "content": pick(answers)}]


TEXT_EXTRA = None


def text_extra():
    global TEXT_EXTRA
    if TEXT_EXTRA is None:
        TEXT_EXTRA = json.loads((pathlib.Path(__file__).parent / "data" / "text_extra.json").read_text(encoding="utf-8"))
    return TEXT_EXTRA


def cat_chat_extra(split):
    qs, answers = pick(text_extra()["chat"])
    now, p = rand_now(), rand_profile()
    return [system(now, p), {"role": "user", "content": pick(qs)}, {"role": "assistant", "content": pick(answers)}]




# ---------------- староста: объявления и Д/З группе; мероприятия и хакатоны ----------------
# Свой генератор случайных чисел (см. __main__), чтобы не сдвинуть ключи перефразов у других разделов.
GROUPS = ["8К51", "8К52", "8К53", "8В51", "8И51", "8Д51"]  # модель их не видит
SUB_ORD = {1: "первой", 2: "второй", 3: "третьей"}  # «второй подгруппе»
SUB_GEN = {1: "первой", 2: "второй", 3: "третьей"}  # «для второй подгруппы»
GROUP_REM = ["собрание группы в 19 корпусе", "сдать деньги на подарок куратору", "принести зачётки в деканат",
             "заполнить анкету от деканата", "забрать студенческие в 10 корпусе", "записаться на секции по физре",
             "сдать справку 086/у в медпункт", "пройти опрос о качестве обучения", "сдать заявления на общежитие",
             "прийти на фотосессию для студенческих", "проголосовать за профорга группы", "сдать конспекты на проверку",
             "принести паспорт для пропуска", "заполнить согласие на обработку данных", "забрать методички в НТБ"]
GROUP_EVENTS = ["Встреча группы с куратором", "Экскурсия в IT-парк", "Субботник у 19 корпуса", "Поход группы в кино",
                "Посвящение в студенты ИШИТР", "Собрание потока в ГК", "Квиз между группами", "Тренировка к турниру по волейболу"]
EV_NAMES = {
    "хакатон": ["Хакатон «Код Сибири»", "AI-хакатон ИШИТР", "Хакатон «Умный кампус»", "Хакатон по анализу данных",
                "Хакатон «Цифровой Томск»", "Хакатон для первокурсников", "Хакатон по кибербезопасности", "Хакатон «GameDev за выходные»"],
    "конференция": ["Студенческая IT-конференция ИШИТР", "Научная сессия первокурсников", "Конференция по робототехнике",
                    "Конференция «ИТ в образовании»", "Конференция молодых программистов"],
    "олимпиада": ["Олимпиада по программированию", "Олимпиада по математике ТПУ", "Олимпиада по информатике для первокурсников",
                  "Командная олимпиада по программированию", "Олимпиада по английскому языку"],
    "конкурс": ["Конкурс студенческих стартапов", "Конкурс IT-проектов «Старт»", "Конкурс научных работ первокурсников",
                "Конкурс мобильных приложений", "Конкурс чат-ботов"],
    "мастер-класс": ["Мастер-класс по Git", "Мастер-класс «Python для анализа данных»", "Мастер-класс по Linux",
                     "Мастер-класс «Как пройти собеседование в IT»", "Мастер-класс по Docker", "Мастер-класс по 3D-печати"],
    "другое": ["День карьеры в IT", "Открытая лекция про нейросети", "Турнир по киберспорту", "Ярмарка вакансий",
               "Экскурсия в IT-компанию", "Лекция разработчика из индустрии", "Встреча с выпускниками ИШИТР"],
}
EV_PLACES = ["МКЦ", "ГК, актовый зал", "КЦ, ауд. 105", "19 корпус, ауд. 210", "10 корпус", "НТБ, коворкинг", "онлайн",
             "ГК, ауд. 230", "8 корпус", "МКЦ, большой зал"]
EV_DESC = ["команды по 3–5 человек", "регистрация до среды", "нужен ноутбук", "вход свободный", "призы для победителей",
           "можно участвовать онлайн", "заявки до пятницы"]
KIND_PL = {"хакатон": "хакатоны", "конференция": "конференции", "олимпиада": "олимпиады", "конкурс": "конкурсы",
           "мастер-класс": "мастер-классы"}
KIND_GEN = {"хакатон": "хакатонов", "конференция": "конференций", "олимпиада": "олимпиад", "конкурс": "конкурсов",
            "мастер-класс": "мастер-классов"}
KIND_ACC = {"хакатон": "хакатон", "конференция": "конференцию", "олимпиада": "олимпиаду", "конкурс": "конкурс",
            "мастер-класс": "мастер-класс"}
KIND_NEXT = {"хакатон": "ближайший хакатон", "конференция": "ближайшая конференция", "олимпиада": "ближайшая олимпиада",
             "конкурс": "ближайший конкурс", "мастер-класс": "ближайший мастер-класс"}


def phr(lst, split, sub):
    """Формулировка; если объявление подгруппе — только те, где подгруппа названа ({to} или {tg})."""
    xs = split_phr(lst, split)
    return pick([x for x in xs if "{to}" in x or "{tg}" in x] if sub else xs)


def people(n):
    return f"{n} {plural(n, 'человек', 'человека', 'человек')}"


def group_backend(now, p, role="student"):
    """Студент вошёл в аккаунт: база в памяти, группа из 16–27 человек со старостой (b.head — его id)."""
    st = Store(":memory:")
    grp = pick(GROUPS)
    # английский и курсы (А) — как в системном промпте (Д/З старосты считается по ним)
    me = st.add_account("me", None, grp, role=role, subgroup=p.get("subgroup"), english_teacher=p.get("english_teacher"),
                        adaptation=p.get("adaptation"))
    head = me if role == "starosta" else st.add_account("head", None, grp, role="starosta", subgroup=rnd.randint(1, 3))
    for i in range(rnd.randint(15, 26)):
        st.add_account(f"s{i}", None, grp, subgroup=pick([1, 2, 3, 1, 2, 3, None]))
    b = Backend(now, {k: v for k, v in p.items() if k != "role"} | {"group": grp, "role": role}, st, me)
    b.head = head
    return b


def head_backend(b):
    """Backend старосты той же группы: от его имени объявления и Д/З, которые видит студент b."""
    acc = b.store.account(account_id=b.head)
    return Backend(b.now, b.store.profile(acc), b.store, b.head)


def rand_event(now, kind=None, days=(0, 45)):
    kind = kind or pick(list(EV_NAMES))
    e = {"title": pick(EV_NAMES[kind]), "kind": kind,
         "date": (now.date() + dt.timedelta(days=rnd.randint(*days))).isoformat()}
    t = pick([None, "10:00", "11:00", "12:00", "14:00", "15:30", "16:30", "17:00", "18:00", "18:30", "19:00"])
    if t:
        e["time"] = t
    e["place"] = pick(EV_PLACES)
    if rnd.random() < 0.3:
        e["description"] = pick(EV_DESC)
    return e


def rand_announcement(now):
    """Объявление старосты -> (текст, at или None): событие группы с датой или дело к сроку (или без срока)."""
    if rnd.random() < 0.5:
        title = pick(GROUP_EVENTS) if rnd.random() < 0.6 else pick(EV_NAMES[pick(list(EV_NAMES))])
        place = pick([None] + EV_PLACES)
        d, t = now.date() + dt.timedelta(days=rnd.randint(0, 13)), pick([None, "12:00", "14:15", "16:30", "18:00", "19:00"])
        if d == now.date() and (t is None or t <= f"{now:%H:%M}"):
            d += dt.timedelta(days=1)
        return title + (f", {place}" if place else ""), d.isoformat() + (f" {t}" if t else "")
    text = cap(pick(GROUP_REM))
    if rnd.random() < 0.5:
        return text, None
    return text, f"{now.date() + dt.timedelta(days=rnd.randint(1, 6))} {rnd.randint(8, 20):02d}:{pick(['00', '30'])}"


def seed_posts(b, kinds=("homework", "announcement"), k=None):
    """Староста отправил группе студента b что-то из kinds; иногда — другой подгруппе (студент этого не увидит)."""
    hb = head_backend(b)
    now = b.now
    for _ in range(k or rnd.choice([1, 1, 2, 2, 3])):
        kind = pick(kinds)
        sub = rnd.randint(1, 3) if rnd.random() < 0.2 else None
        if kind == "homework":
            code = pick(student_subjects(b) or ["math"])
            due = "next_class" if rnd.random() < 0.6 else pick(no_sunday([now.date() + dt.timedelta(days=x) for x in range(1, 7)])).isoformat()
            hb.group_add_homework(SUBJECTS[code]["name"], pick(HW_TASKS[code]), due=due, subgroup=sub)
        else:
            text, at = rand_announcement(now)
            hb.group_announce(text, at=at, subgroup=sub)


def ev_when(now, e):
    return when_ru(now, e["date"] + (" " + e["time"] if e.get("time") else ""))


def cat_brief_posts(split):
    return cat_brief(split, posts=True)


def cat_note_list_posts(split):
    return cat_note_list(split, posts=True)


def event_line(now, e):
    s = f"• {ev_when(now, e)} — {e['title']}" + (f", {e['place']}" if e.get("place") else "")
    return s + (f" ({e['description']})" if e.get("description") else "")


def cat_events(split):
    now = rand_now()
    b = Backend(now, rand_profile())
    form = pick(["any", "any", "kind", "kind", "kind", "week", "kind_week", "month", "next", "all"])
    kind = pick(EVENT_KINDS)
    b.events = [rand_event(now) for _ in range(pick([0, 1, 2, 3, 3, 4, 5, 6, 8, 12]))]
    if form in ("kind", "kind_week", "next") and rnd.random() < 0.75:  # обычно спрашивают о том, что есть
        b.events.append(rand_event(now, kind, (1, 6) if form == "kind_week" else (0, 40)))
    if form == "any":
        q = pick(split_phr(["какие мероприятия скоро?", "что интересного будет в универе?", "куда сходить из мероприятий?",
                            "есть какие-нибудь движухи в ТПУ?", "покажи ближайшие мероприятия", "что по мероприятиям?",
                            "чем заняться кроме учёбы?", "какие ивенты скоро?"], split))
        args = {}
    elif form == "kind":
        q = pick(split_phr(["какие {pl} скоро?", "есть {pl}?", "будут какие-нибудь {pl}?", "покажи {pl}",
                            "хочу на {acc}, что есть?", "{pl} какие ближайшие", "а {pl} будут?", "подскажи {pl} в ТПУ"],
                           split)).format(pl=KIND_PL[kind], acc=KIND_ACC[kind])
        args = {"kind": kind}
    elif form == "week":
        q = pick(split_phr(["что на этой неделе из мероприятий?", "мероприятия на неделю", "есть что-то интересное на этой неделе?",
                            "куда сходить на неделе?", "события этой недели покажи", "что за ивенты на неделе?"], split))
        args = {"period": "week"}
    elif form == "kind_week":
        q = pick(split_phr(["{pl} на этой неделе есть?", "на неделе будут {pl}?", "какие {pl} на этой неделе?",
                            "есть {pl} в ближайшие дни?", "{pl} на неделе покажи", "на этой неделе {pl} какие-нибудь есть?"],
                           split)).format(pl=KIND_PL[kind])
        args = {"kind": kind, "period": "week"}
    elif form == "month":
        q = pick(split_phr(["какие мероприятия в этом месяце?", "что интересного в ближайший месяц?", "мероприятия на месяц вперёд",
                            "что будет в этом месяце из событий?", "события на месяц покажи", "ивенты на ближайший месяц"], split))
        args = {"period": "month"}
    elif form == "next":
        q = pick(split_phr(["когда {nx}?", "{nx} когда будет?", "а когда {nx}?", "подскажи, когда {nx}",
                            "скажи дату: {nx}", "{nx} — когда?"], split)).format(nx=KIND_NEXT[kind])
        args = {"kind": kind, "period": "all"}
    else:
        q = pick(split_phr(["покажи все мероприятия", "все предстоящие мероприятия", "какие вообще мероприятия запланированы?",
                            "весь список мероприятий", "что запланировано из мероприятий до конца семестра?", "все ивенты покажи"], split))
        args = {"period": "all"}
    res = b.get_events(**args)
    items = res["items"]
    span = {"week": "в ближайшие 7 дней", "month": "в ближайший месяц", "all": "пока"}.get(args.get("period"), "в ближайший месяц")
    if not items:
        what = KIND_GEN[kind] if "kind" in args else "мероприятий"
        ans = pick([f"{cap(span)} {what} не нашёл 🤷 Как появятся — покажу.",
                    f"{cap(span)} {what} в списке нет 🙃 Загляни попозже, список пополняется."])
        if span == "пока":
            ans = pick([f"Пока {what} не запланировано 🤷 Как появятся — покажу.", f"В списке пока нет {what} 🙃"])
    elif form == "next":
        e = items[0]
        ans = (f"{cap(KIND_NEXT[kind])} — {e['title']}, {ev_when(now, e)}" + (f", {e['place']}" if e.get("place") else "")
               + (f" ({e['description']})" if e.get("description") else "") + ". "
               + ("Это уже сегодня! 🔥" if e["days_left"] == 0 else f"Осталось {days_n(e['days_left'])} 🗓️"))
    else:
        head = (f"{cap(KIND_PL[kind])} {span}:" if "kind" in args else
                pick([f"Мероприятия {span}:", "Вот что нашёл 🗓️", "Ближайшие мероприятия:"]))
        ans = head + "\n" + "\n".join(event_line(now, e) for e in items)
        if res.get("more"):
            ans += f"\n\nИ ещё {res['more']} — спроси про конкретный вид (хакатоны, олимпиады…), покажу."
        elif rnd.random() < 0.3:
            ans += "\n\n" + pick(["Хочешь, поставлю напоминание о каком-нибудь?", "Если надумаешь пойти — могу напомнить заранее 🔔"])
    return [system(now, b.profile), {"role": "user", "content": q}, tc("get_events", args), tr("get_events", res),
            {"role": "assistant", "content": ans}]


def group_hw_phrase(split, form, sub):
    """Фраза старосты «выдай группе Д/З»: {s} предмет, {sd} в дательном, {t} задание, {dd}/{da}/{di}/{d} срок."""
    if sub:
        return pick(split_phr(["{o} подгруппе дз по {sd}: {t}", "отправь {o} подгруппе: по {sd} {t}",
                               "разошли дз по {sd} только {o} подгруппе — {t}", "для {og} подгруппы по {sd} задали {t}, разошли",
                               "кинь {o} подгруппе домашку по {sd}: {t}", "выдай {o} подгруппе задание по {sd}: {t}"], split))
    return pick(split_phr({
        "next": ["выдай группе дз по {sd}: {t}", "отправь всем дз по {sd} — {t}", "разошли группе домашку по {sd}: {t}",
                 "группе задали по {sd} {t} к следующей паре, разошли", "объяви всем: по {sd} на след пару {t}",
                 "кинь всем дз: {s}, {t}", "закинь группе дз по {sd}: {t}", "раздай группе задание по {sd}: {t}"],
        "weekday": ["выдай группе дз по {sd} к {dd}: {t}", "всем по {sd} к {dd} надо {t}, разошли", "отправь группе: по {sd} {t}, сдать {di}",
                    "разошли дз по {sd} на {da} — {t}", "объяви группе дз по {sd}: {t}, к {dd}",
                    "группе по {sd} задали {t} к {dd}, отправь всем", "кинь всем: {s} — {t}, сдать к {dd}", "закинь группе дз на {da} по {sd}: {t}"],
        "tomorrow": ["выдай группе дз на завтра по {sd}: {t}", "всем на завтра по {sd} {t}, разошли", "отправь группе: к завтра по {sd} {t}",
                     "разошли всем на завтра: {s}, {t}", "объяви: на завтра по {sd} {t}", "закинь группе на завтра — {s}: {t}",
                     "кинь всем дз на завтра по {sd}: {t}", "группе на завтра по {sd} задали {t}, отправь"],
        "date": ["выдай группе дз по {sd} к {d}: {t}", "разошли всем: по {sd} {t}, срок {d}", "отправь группе дз по {sd} на {d} — {t}",
                 "объяви всем: {s}, {t}, сдать до {d}", "кинь группе: к {d} по {sd} {t}", "закинь всем дз по {sd}: {t}, к {d}",
                 "раздай группе: {s} — {t}, срок {d}", "группе к {d} по {sd} надо {t}, разошли"]}[form], split))


def group_to(res):
    return f"{SUB_ORD[res['subgroup']]} подгруппе" if res.get("subgroup") else "группе"


def group_hw_answer(now, res, task, spec="next_class"):
    sh = cap(short(res["subject"]))
    w, pair = when_ru(now, res["due"]), " на паре" if " " in res["due"] else ""
    to, n = group_to(res), res["recipients"]
    note = ""
    if res.get("due_differs"):  # пары у подгрупп (групп английского) в разное время: срок у каждого свой
        w, pair = ("к следующей паре" if spec == "next_class" else when_ru(now, res["due"][:10])), ""
        note = " " + pick([f"Пары по {dat(short(res['subject']))} у всех в разное время, так что срок у каждого — по его расписанию.",
                           "Срок у каждого по его расписанию: пары идут в разное время."])
    elif res.get("no_class_that_day"):
        note = " " + pick([f"Пары по {dat(short(res['subject']))} в этот день нет, так что срок просто по дате.",
                           "В этот день пары нет — срок по дате."])
    return pick([f"Отправил {to} 📢 {sh}: {task}, срок — {w}{pair}.{note} Получат {people(n)}, у всех появится в брифинге.",
                 f"Готово ✅ Отправил {to}: {sh} — {task}, сдать {w}{pair}.{note} Это увидят {people(n)}.",
                 f"Есть! {sh}: {task} ({w}{pair}) — ушло {to}, {people(n)} 📢{note}",
                 f"Отправил 📢 {sh}: {task}, срок {w}{pair}.{note} Задание у {people(n)} в утреннем брифинге."])


def announce_answer(now, res):
    to, n = group_to(res), res["recipients"]
    w = f" — {when_ru(now, res['at'])}" if res.get("at") else ""
    keep = "" if res.get("at") else pick([" Объявление продержится неделю.", " Через неделю оно удалится само."])
    return pick([f"Отправил {to} 📢 «{res['text']}»{w}. Получат {people(n)}, увидят в брифинге.{keep}",
                 f"Готово ✅ Объявление «{res['text']}»{w} ушло {to} ({people(n)}).{keep}",
                 f"Есть! Объявление {to}: «{res['text']}»{w} 📢 Его увидят {people(n)}.{keep}"])


def at_phrase(now, d, time):
    """Как староста называет день события -> (слова, аргумент at)."""
    delta = (d - now.date()).days
    if delta == 0:
        spoken, at = "сегодня", d.isoformat()
    elif delta == 1 and rnd.random() < 0.5:
        spoken, at = "завтра", "tomorrow"
    elif 1 < delta < 7 and rnd.random() < 0.4:
        spoken, at = WEEKDAYS_FUT[d.weekday()], WEEKDAYS[d.weekday()]
    else:
        spoken, at = date_ru(d), d.isoformat()
    return spoken + (f" в {hm(time)}" if time else ""), at + (f" {time}" if time else "")


ANNOUNCE_AT_PHR = ["закинь {to}: {t}, {w}", "объяви {to} — {t}, {w}", "разошли {to} объявление: {t}, {w}",
                   "отправь {to}: {w} будет {tl}", "напиши {to}, что {w} {tl}", "кинь {to} инфу: {t}, {w}",
                   "сделай объявление {tg}: {t}, {w}", "разошли: {t}, {w}, пусть все знают"]
ANNOUNCE_TODO_PHR = ["напомни {to} {w}: {x}", "объяви {to}: {x} {w}", "поставь всем объявление {w} — {x}",
                     "сделай объявление для {tg}: {x}, {w}", "пусть все знают: {w} {x}", "напомни всем {w}: {x}",
                     "кинь {to} объявление: {x}, {w}", "{w} — {x}, объяви {to}"]
ANNOUNCE_PHR = ["объяви {to}: {x}", "напиши всем: {x}", "разошли {to} объявление: {x}", "передай {to}: {x}",
                "кинь {to} объявление — {x}", "сделай объявление для {tg}: {x}", "объявление {to}: {x}", "сообщи {to}: {x}"]


def cat_group_post(split):
    now = rand_now((7, 22))
    b = group_backend(now, rand_profile(), "starosta")
    kind = pick(["homework"] * 9 + ["event"] * 6 + ["todo"] * 5)
    sub = rnd.randint(1, 3) if rnd.random() < 0.18 else None
    msgs = [system(now, b.profile)]
    if kind == "homework":
        target = Backend(now, dict(b.profile, subgroup=sub)) if sub else b
        subs = student_subjects(target)
        if rnd.random() < 0.06 or not subs:  # предмета нет в расписании: пары не найдётся, спросим дату
            alias, task = pick(UNKNOWN_HW)
            q = group_hw_phrase(split, "next", sub).format(s=alias, sd=sdat(alias), t=task, o=SUB_ORD.get(sub), og=SUB_GEN.get(sub))
            args = {"subject": alias, "task": task, "due": "next_class"} | ({"subgroup": sub} if sub else {})
            res = b.group_add_homework(**args)
            msgs += [{"role": "user", "content": q}, tc("group_add_homework", args), tr("group_add_homework", res),
                     {"role": "assistant", "content": pick([
                         f"Не нашёл пар по предмету «{alias}» в расписании группы 🤔 К какому числу сдавать? Отправлю с этой датой.",
                         f"Хм, пар по предмету «{alias}» в расписании нет 🤔 Скажи дату сдачи — отправлю на неё."])}]
            if rnd.random() < 0.6:
                d = pick(no_sunday([now.date() + dt.timedelta(days=k) for k in range(2, 8)]))
                msgs.append({"role": "user", "content": pick([date_ru(d), f"к {date_ru(d)}", f"до {date_ru(d)}"])})
                args = args | {"due": d.isoformat()}
                res = b.group_add_homework(**args)
                msgs += [tc("group_add_homework", args), tr("group_add_homework", res),
                         {"role": "assistant", "content": group_hw_answer(now, res, task, args["due"])}]
            return msgs
        code = pick(subs)
        task, alias = pick(HW_TASKS[code]), sub_alias(code)
        today = now.date()
        form = "next" if sub else pick(["next", "next", "next", "weekday", "weekday", "tomorrow", "date", "date"])
        if form == "tomorrow" and not no_sunday([today + dt.timedelta(days=1)]):
            form = "next"
        fmt = dict(s=alias, sd=sdat(alias), t=task, o=SUB_ORD.get(sub), og=SUB_GEN.get(sub))
        if form == "next":
            due = "next_class"
        elif form == "weekday":
            d = pick_day(target, code, no_sunday([today + dt.timedelta(days=k) for k in range(1, 8)]))
            fmt |= dict(dd=WD_DAT[d.weekday()], da=WD_ACC[d.weekday()], di=WEEKDAYS_FUT[d.weekday()])
            due = WEEKDAYS[d.weekday()]
        elif form == "tomorrow":
            due = "tomorrow"
        else:
            d = pick_day(target, code, no_sunday([today + dt.timedelta(days=k) for k in range(2, 10)]))
            fmt["d"] = date_ru(d)
            due = d.isoformat()
        q = group_hw_phrase(split, form, sub).format(**fmt)
        args = {"subject": alias, "task": task, "due": due} | ({"subgroup": sub} if sub else {})
        res = b.group_add_homework(**args)
        if "error" in res:
            return None
        return msgs + [{"role": "user", "content": q}, tc("group_add_homework", args), tr("group_add_homework", res),
                       {"role": "assistant", "content": group_hw_answer(now, res, task, args["due"])}]
    to = f"{SUB_ORD[sub]} подгруппе" if sub else pick(["группе", "всем"])
    tg = f"{SUB_GEN[sub]} подгруппы" if sub else "группы"
    if kind == "event":  # событие с днём (и временем)
        title = pick(GROUP_EVENTS) if rnd.random() < 0.5 else pick(EV_NAMES[pick(list(EV_NAMES))])
        place = pick([None, None] + EV_PLACES)
        d = now.date() + dt.timedelta(days=rnd.randint(1, 20))
        spoken, at = at_phrase(now, d, pick([None, "12:00", "14:15", "16:30", "18:00", "19:00"]))
        text = title + (f", {place}" if place else "")
        said = low(text) if rnd.random() < 0.5 else text  # как написал староста; в text — с заглавной
        q = phr(ANNOUNCE_AT_PHR, split, sub).format(to=to, tg=tg, t=said, tl=low(text), w=spoken)
        args = {"text": text, "at": at}
    elif rnd.random() < 0.55:  # дело к сроку: «напомни всем до пятницы сдать …»
        what = pick(GROUP_REM)
        wp, targs = reminder_phrase(now)
        if "at" not in targs:
            return None  # «через 2 часа» — это напоминание, а объявлению нужен день
        q = phr(ANNOUNCE_TODO_PHR, split, sub).format(to=to, tg=tg, w=wp, x=what)
        args = {"text": cap(what), "at": targs["at"]}
    else:  # без даты: продержится неделю
        what = pick(GROUP_REM)
        q = phr(ANNOUNCE_PHR, split, sub).format(to=to, tg=tg, x=what)
        args = {"text": cap(what)}
    args |= {"subgroup": sub} if sub else {}
    res = b.group_announce(**args)
    if not res.get("ok"):
        return None
    return msgs + [{"role": "user", "content": q}, tc("group_announce", args), tr("group_announce", res),
                   {"role": "assistant", "content": announce_answer(now, res)}]


def cat_group_denied(split):
    """Обычный студент просит отправить группе: это может только староста, предлагаем записать себе."""
    now = rand_now()
    b = group_backend(now, rand_profile(), "student")
    msgs = [system(now, b.profile)]
    if rnd.random() < 0.5:
        code = pick(student_subjects(b) or ["math"])
        task, alias = pick(HW_TASKS[code]), sub_alias(code)
        q = group_hw_phrase(split, "next", None).format(s=alias, sd=sdat(alias), t=task)
        name, args = "group_add_homework", {"subject": alias, "task": task, "due": "next_class"}
        offer = "Могу поставить тебе напоминание об этом Д/З накануне пары — поставить?"
    else:
        what = pick(GROUP_REM)
        q = pick(split_phr(["объяви всем: {x}", "разошли группе: {x}", "напиши всей группе: {x}", "сделай объявление для группы: {x}",
                            "передай группе: {x}", "кинь всем объявление: {x}"], split)).format(x=what)
        name, args = "group_announce", {"text": cap(what)}
        offer = "Могу записать это тебе — напомнить в какое-то время или просто сохранить в заметки?"
    assert getattr(b, name)(**args) == {"error": "not_starosta"}  # бэкенд не даст, даже если модель попробует
    # инструментов старосты в промпте студента нет (prompt_format.tools_for): отвечаем сразу текстом
    msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": pick([
        f"Отправлять группе может только староста 🙂 {offer}",
        f"Объявления и Д/З всей группе отправляет только аккаунт старосты. {offer}",
        f"Эта функция есть только у старосты 🙃 Попроси старосту отправить. {offer}"])}]
    if rnd.random() < 0.4:
        return msgs
    if name == "group_add_homework":
        ha = {"subject": alias, "task": task}
        r2 = b.add_homework_reminder(**ha)
        if not r2.get("ok"):
            return msgs
        msgs += [{"role": "user", "content": pick(["да", "давай", "ага, поставь", "да, пожалуйста"])},
                 tc("add_homework_reminder", ha), tr("add_homework_reminder", r2),
                 {"role": "assistant", "content": hw_rem_answer(now, r2, task)}]
    elif rnd.random() < 0.5:
        wp, targs = reminder_phrase(now)
        na = {"text": cap(what)} | targs
        r2 = b.add_note(**na)
        msgs += [{"role": "user", "content": wp}, tc("add_note", na), tr("add_note", r2),
                 {"role": "assistant", "content": reminder_answer(now, r2, what)}]
    else:
        na = {"text": cap(what)}
        r2 = b.add_note(**na)
        msgs += [{"role": "user", "content": say(split, ["просто запиши"], NO_TIME)}, tc("add_note", na), tr("add_note", r2),
                 {"role": "assistant", "content": note_answer(cap(what))}]
    return msgs


def cat_starosta_personal(split):
    """Староста просит что-то для себя: обычные инструменты, без объявлений группе."""
    global STAROSTA
    STAROSTA = True
    try:
        return pick([cat_note_timed, cat_note_timed, cat_hw_reminder, cat_hw_reminder, cat_note_list, cat_schedule,
                     cat_note_clarify, cat_brief])(split)
    finally:
        STAROSTA = False


def cat_starosta_chat(split):
    """Староста спрашивает, что умеет."""
    now = rand_now()
    b = group_backend(now, rand_profile(), "starosta")
    q = pick(split_phr(["что я могу как староста?", "как отправить группе дз?", "как сделать объявление для группы?",
                        "а что умеет староста?", "я староста, что ты можешь для группы?", "как написать всей группе?"], split))
    ans = pick(["Ты староста, так что можешь отправлять группе (или одной подгруппе):\n"
                "• объявления — с датой продержатся до события, без даты — неделю\n• Д/З — появится у всех в брифинге\n\n"
                "Действующих объявлений и Д/З — до 10 каждого. Просто напиши, например: «объяви группе: собрание в пятницу в 15:00 "
                "в 19 корпусе» или «выдай группе дз по матлогике: №5–10 к пятнице» 📢",
                "Как староста ты можешь отправить всей группе объявление или Д/З — оно придёт каждому в брифинг 📢 "
                "Скажи, например: «объяви всем: хакатон 24 октября в 10:00, МКЦ». Если нужно только одной подгруппе — так и скажи. "
                "Удалить объявление можно тоже через меня: «удали объявление про хакатон»."])
    return [system(now, b.profile), {"role": "user", "content": q}, {"role": "assistant", "content": ans}]


STAROSTA_CATS = [(cat_events, 55), (cat_group_post, 75), (cat_group_denied, 15), (cat_starosta_personal, 25),
                 (cat_brief_posts, 40), (cat_note_list_posts, 12), (cat_starosta_chat, 6)]


# ---------------- удаление и лимиты (delete_items, group_delete_post) ----------------
# Номера — из последнего показанного списка (список записей, брифинг, «что удалить?»); query — ключевые слова в начальной
# форме. «Удалить все» — с подтверждением: бэкенд сначала возвращает confirm, удаляет при повторном вызове.
# Заметки и напоминания — один список, поэтому «удали заметку» показывает все записи (kind не нужен), «удали напоминание» —
# только напоминания (kind=reminder), а kind=note — только когда студент прямо отделяет заметки от напоминаний.
# (как сказать после «про», что передать в query)
REM_KEYS = {"сдать отчёт старосте": ("отчёт", "отчёт"), "позвонить в деканат": ("деканат", "деканат"),
            "купить тетрадь": ("тетрадь", "тетрадь"), "взять пропуск": ("пропуск", "пропуск"),
            "скинуть лабу преподу": ("лабу", "лаба"), "оплатить общагу": ("общагу", "общага"),
            "забрать справку": ("справку", "справка"), "записаться на пересдачу": ("пересдачу", "пересдача"),
            "зарядить ноут": ("ноут", "ноут"), "сходить в библиотеку": ("библиотеку", "библиотека"),
            "сдать лабу 3 по питону": ("лабу 3", "лаба 3"), "скинуть эссе Ростовцевой": ("эссе", "эссе"),
            "забрать зачётку в деканате": ("зачётку", "зачётка"), "продлить книгу в НТБ": ("книгу", "книга"),
            "оплатить проездной": ("проездной", "проездной"), "записаться на физру": ("физру", "физра"),
            "скинуть презу для проектной": ("презу", "преза"), "отправить реферат на проверку": ("реферат", "реферат"),
            "выпить таблетку": ("таблетку", "таблетка"), "позвонить маме": ("маму", "мама"),
            "заказать справку для военкомата": ("справку для военкомата", "справка военкомат"),
            "распечатать отчёт по инфе": ("отчёт по инфе", "отчёт инфа"),
            "подготовиться к коллоквиуму по матану": ("коллоквиум", "коллоквиум"),
            "проверить почту ТПУ": ("почту", "почта"), "скачать задания по матлогике": ("задания по матлогике", "задания матлогика"),
            "купить проездной на месяц": ("проездной на месяц", "проездной месяц"), "отдать конспект Кате": ("конспект", "конспект"),
            "сходить на консультацию по питону": ("консультацию", "консультация")}
GROUP_REM_KEYS = {"собрание группы в 19 корпусе": ("собрание", "собрание"),
                  "сдать деньги на подарок куратору": ("подарок куратору", "подарок куратор"),
                  "принести зачётки в деканат": ("зачётки", "зачётки"), "заполнить анкету от деканата": ("анкету", "анкета"),
                  "забрать студенческие в 10 корпусе": ("студенческие", "студенческие"),
                  "записаться на секции по физре": ("секции по физре", "секции физра"),
                  "сдать справку 086/у в медпункт": ("справку в медпункт", "справка медпункт"),
                  "пройти опрос о качестве обучения": ("опрос", "опрос"),
                  "сдать заявления на общежитие": ("заявления на общежитие", "заявления общежитие"),
                  "прийти на фотосессию для студенческих": ("фотосессию", "фотосессия"),
                  "проголосовать за профорга группы": ("профорга", "профорг"),
                  "сдать конспекты на проверку": ("конспекты", "конспекты"), "принести паспорт для пропуска": ("паспорт", "паспорт"),
                  "заполнить согласие на обработку данных": ("согласие", "согласие"),
                  "забрать методички в НТБ": ("методички", "методички")}
assert set(REM_KEYS) == set(REM_TASKS) and set(GROUP_REM_KEYS) == set(GROUP_REM)
GENDER = {"note": "f", "reminder": "n", "homework": "n", "mixed": "f"}
ORD = {"f": ["первую", "вторую", "третью", "четвёртую", "пятую", "шестую", "седьмую", "восьмую", "девятую", "десятую"],
       "n": ["первое", "второе", "третье", "четвёртое", "пятое", "шестое", "седьмое", "восьмое", "девятое", "десятое"]}
LAST = {"f": "последнюю", "n": "последнее"}
NOUN = {"note": ("заметку", "заметки", "заметок"), "reminder": ("напоминание", "напоминания", "напоминаний"),
        "mixed": ("запись", "записи", "записей")}
ALL_PL = {"note": "заметки", "reminder": "напоминания", "mixed": "записи"}
CHOOSE_Q = {"note": "Какую заметку удалить?", "reminder": "Какое напоминание удалить?", "mixed": "Что удалить?"}
ONE = {"note": ("одна заметка", "её"), "reminder": ("одно напоминание", "его"), "mixed": ("одна запись", "её")}
EMPTY = {None: "Ни заметок, ни напоминаний нет — удалять нечего 📭", "reminder": "Напоминаний нет — удалять нечего 🙂",
         "note": "Заметок без времени нет — удалять нечего 📭"}
NONE_LEFT = {None: "Больше ничего не записано 📭", "reminder": "Напоминаний больше нет.", "note": "Заметок больше нет."}
NF = {"note": "Заметку", "reminder": "Напоминание", "homework": "Напоминание о Д/З"}
# Ответы студента дальше в диалоге: «да», «нет», номер — общие, остальные формулировки делим на train/eval (split_phr),
# как и первые сообщения: evaluate.py --step 3 проверяет именно их.
YES = ["да, удаляй", "ага", "давай", "угу, удаляй", "да, давай", "конечно, удаляй"]
NO = ["не, оставь", "стоп, не надо", "нет, передумал", "не удаляй", "не надо, пусть будет", "отмена"]


def yes(split):
    return pick(["да", "да"] + split_phr(YES, split))


def no(split):
    return pick(["нет", "нет"] + split_phr(NO, split))


def say(split, shared, lst):
    """Реплика студента: shared — общие для train и eval, из lst eval берёт последние две."""
    return pick(shared + split_phr(lst, split))


def rk(x):
    """Вид записи из списка: заметка, напоминание или напоминание о Д/З."""
    return "homework" if x.get("subject") else "reminder" if x.get("fire_at") else "note"


def wk(items):
    """Как назвать записи вместе: заметки, напоминания (и о Д/З) или вперемешку — «записи»."""
    kinds = {"reminder" if rk(x) == "homework" else rk(x) for x in items}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def what(x):
    """Винительный: «удалил …»."""
    k = rk(x)
    if k == "note":
        return f"заметку «{x['text']}»"
    if k == "reminder":
        return f"напоминание «{x['text']}»"
    return f"напоминание о Д/З по {dat(short(x['subject']))}" + (f" ({x['text']})" if x.get("text") else "")


def name(x):
    return f"Д/З по {dat(short(x['subject']))}" if rk(x) == "homework" else f"«{x['text']}»"


def ask_tail():
    return pick(["Напиши номер (можно несколько) или «все».", "Назови номер — или скажи «все».", "Скажи номер или «все» 🙂"])


def choose_answer(now, res, about=None):
    items = res["items"]
    w = wk(items)
    if len(items) == 1:  # одна запись (или нашлась только похожая): спрашиваем про неё
        one, pron = ONE[w]
        head = "Нашёл только это:" if about else f"У тебя {one}:"
        return f"{head}\n{item_lines(now, items)}\n\nУдалить {pron}?"
    if about:
        return (f"Нашёл {len(items)} {plural(len(items), *NOUN[w])} {about}:\n{item_lines(now, items)}\n\n"
                f"{CHOOSE_Q[w]} {ask_tail()}")
    return f"{CHOOSE_Q[w]} 🗑️\n{item_lines(now, items)}\n\n{ask_tail()}"


def not_found_answer(now, res, about, nf):
    return (f"{nf} {about} не нашёл 🤔 Вот что есть:\n{item_lines(now, res['items'])}\n\n"
            f"{CHOOSE_Q[wk(res['items'])]} {ask_tail()}")


def confirm_answer(now, res):
    n, items = res["count"], res["items"]
    if n == 1:
        return f"Удалить {what(items[0])}? Это насовсем. Напиши «да» — удалю."
    lines, w = item_lines(now, items), wk(items)
    return pick([f"Точно удалить все {n} {plural(n, *NOUN[w])}? Это насовсем:\n{lines}\n\nНапиши «да» — удалю.",
                 f"Вот все {ALL_PL[w]} ({n}):\n{lines}\n\nУдалить их все? Вернуть не получится. Напиши «да» — удалю."])


def deleted_answer(res, scope=None, all_=False):
    """scope — какой список считает left: None — все записи, reminder — только напоминания, note — только заметки."""
    d, out = res["deleted"], []
    w = wk(d)
    if len(d) == 1:
        out.append(pick([f"Удалил {what(d[0])} 🗑️", f"Готово ✅ Удалил {what(d[0])}.", f"Удалил {what(d[0])} 👌"]))
    elif all_ and len(d) > 4:
        out.append(f"Удалил все {len(d)} {plural(len(d), *NOUN[w])} 🧹")
    else:
        out.append(f"Удалил {len(d)} {plural(len(d), *NOUN[w])}: " + ", ".join(name(x) for x in d) + " 🗑️")
    g = res.get("gone")
    if g:
        out.append(f"Номер {g[0]} уже был удалён раньше." if len(g) == 1 else f"Номера {', '.join(map(str, g))} уже были удалены раньше.")
    if rnd.random() < 0.4 and (res["left"] == 0 or not all_):  # после «все» — только «больше нет»: left мог считать и другие записи
        out.append(NONE_LEFT[scope] if res["left"] == 0 else f"Осталось: {res['left']}.")
    return " ".join(out)


def gone_phrase(res):
    """«Удалил заметку «X»» — начало ответа, когда после удаления сразу записываем новое."""
    d = res["deleted"]
    if len(d) == 1:
        return f"Удалил {what(d[0])}"
    if len(d) > 4:
        return f"Удалил все {len(d)} {plural(len(d), *NOUN[wk(d)])}"
    return f"Удалил {len(d)} {plural(len(d), *NOUN[wk(d)])}: " + ", ".join(name(x) for x in d)


def pick_phrase(items, nums, split):
    """Студент называет номер(а) из списка items: «2», «вторую», «удали 2 и 4»."""
    byn = {x["n"]: x for x in items}
    gs = [GENDER[rk(byn[n])] if n in byn else "f" for n in nums]
    if len(nums) == 1:
        n, g = nums[0], gs[0]
        opts = [f"давай {n}", f"номер {n}", f"удали {n}", ORD[g][n - 1]] + ([LAST[g]] if n == max(byn) and len(byn) > 2 else [])
        return say(split, [str(n), str(n)], opts + [f"давай {ORD[g][n - 1]}", f"убери {n}"])
    a, b_ = nums
    ords = [f"{ORD[gs[0]][a - 1]} и {ORD[gs[1]][b_ - 1]}"] if gs[0] == gs[1] else [f"{a}-ю и {b_}-ю"]
    return say(split, [f"{a} и {b_}"], [f"{a}, {b_}", f"удали {a} и {b_}"] + ords + [f"убери {a} и {b_}"])


def call(b, msgs, name_, args):
    """Вызов как в chat.py (backend.call): в историю — вызов и результат. Новое сообщение студента — new_turn, как в chat.py:
    номера относятся к спискам из прошлых сообщений, «да» — к списку из предыдущего."""
    u = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    if getattr(b, "user_at", None) != u:
        b.user_at = u
        b.conv.new_turn()
    res = b.call(name_, args)
    msgs += [tc(name_, args), tr(name_, res)]
    return res


def del_backend(now):
    """Студент в аккаунте; иногда в группе (тогда у него бывают объявления старосты) или сам староста — свои записи
    он удаляет так же, delete_items, а group_delete_post — только то, что отправлял группе."""
    r = rnd.random()
    if r < 0.3:
        return group_backend(now, rand_profile(), "student")
    if r < 0.42:
        return group_backend(now, rand_profile(), "starosta")
    return account_backend(now, rand_profile())


def would_find(recs, query):
    """Найдёт ли delete_items запись по query (как Conversation.delete: слова, потом предмет)."""
    st = _stems(query)
    if any(_matches(st, r) for r in recs):
        return True
    code = resolve_subject(query)
    return bool(code) and any(r.code == code for r in recs)


def key_of(x):
    """(как студент назовёт запись, query, about для ответа «… про X не нашёл»)."""
    k = rk(x)
    if k == "note":
        k_, query = pick(next(n[1] for n in NOTES if cap(n[0]) == x["text"]))
        return k_, query, f"про «{k_}»"
    if k == "reminder":
        k_, query = REM_KEYS[low(x["text"])]
        return k_, query, f"про «{k_}»"
    alias = sub_alias(resolve_subject(x["subject"]))
    return sdat(alias), alias, f"по {dat(alias)}"


def missing_key(b, kind):
    """Запись, которой у студента нет: (как назовёт, query, about). None, если не придумалось."""
    recs = b._recs()
    for _ in range(20):
        if kind == "note":
            x = {"text": cap(pick(NOTES)[0])}
        elif kind == "reminder":
            x = {"text": cap(pick(REM_TASKS)), "fire_at": "-"}
        else:
            x = {"subject": SUBJECTS[pick(list(HW_TASKS))]["name"], "fire_at": "-"}
        k, query, about = key_of(x)
        if not would_find(recs, query):
            return k, query, about
    return None


DEL_QUERY_PHR = {
    "note": ["удали заметку про {k}", "убери заметку про {k}", "сотри заметку про {k}", "заметку про {k} можно удалить",
             "больше не нужна заметка про {k}, удали", "выкинь заметку про {k}", "удали из заметок {k}", "удали ту заметку, где про {k}"],
    "reminder": ["удали напоминание про {k}", "убери напоминалку про {k}", "отмени напоминание про {k}",
                 "не надо напоминать про {k}", "сними напоминание про {k}", "напоминание про {k} больше не нужно",
                 "удали напоминалку про {k}", "отмени напоминалку про {k}"],
    "homework": ["удали напоминание о дз по {k}", "дз по {k} уже сделал, напоминать не надо", "отмени напоминание про домашку по {k}",
                 "убери напоминалку о дз по {k}", "не напоминай про дз по {k}", "сними напоминание о домашке по {k}",
                 "удали напоминалку про дз по {k}", "домашку по {k} сделал, убери напоминание"]}


def bad_number_answer(now, res):
    bad = ", ".join(map(str, res["numbers"]))
    return pick([f"Номера {bad} в списке нет 🤔 Вот он:\n{item_lines(now, res['items'])}\n\n{CHOOSE_Q[wk(res['items'])]} Напиши номер.",
                 f"Такого номера нет ({bad}) 🙃 Список:\n{item_lines(now, res['items'])}\n\nКакой номер удалить?"])


def finish_choice(now, b, msgs, res, split, scope, asked_one=False):
    """После списка (choose / not_found): студент выбирает номер, несколько, «все», «никакую» или ошибается номером."""
    items = res["items"]
    nums_all = [x["n"] for x in items]
    r = rnd.random()
    if asked_one:  # «Удалить её?»
        if r < 0.8:
            msgs.append({"role": "user", "content": yes(split)})
            res2 = call(b, msgs, "delete_items", {"numbers": [1]})
            msgs.append({"role": "assistant", "content": deleted_answer(res2, scope)})
        else:
            msgs += [{"role": "user", "content": no(split)}, {"role": "assistant", "content": pick(["Ок, оставил 👌", "Хорошо, не трогаю 🙂"])}]
        return msgs
    if r < 0.55 or (r < 0.72 and len(items) < 2):
        nums = [pick(nums_all)]
    elif r < 0.72:
        nums = sorted(rnd.sample(nums_all, 2))
    elif r < 0.84:
        msgs.append({"role": "user", "content": say(split, ["все"], ["давай все", "удали все", "очисти все", "все удаляй", "удаляй все"])})
        res2 = call(b, msgs, "delete_items", {"all": True})
        msgs.append({"role": "assistant", "content": deleted_answer(res2, scope, all_=True)})
        return msgs
    elif r < 0.92:  # опечатка в номере: Ишка показывает список ещё раз, студент называет правильный
        bad = max(nums_all) + rnd.randint(1, 3)
        msgs.append({"role": "user", "content": say(split, [str(bad)], [f"удали {bad}", f"давай {bad}", f"номер {bad}", f"удаляй {bad}", f"убери {bad}"])})
        res2 = call(b, msgs, "delete_items", {"numbers": [bad]})
        msgs.append({"role": "assistant", "content": bad_number_answer(now, res2)})
        items, nums = res2["items"], [pick([x["n"] for x in res2["items"]])]
    else:
        g = GENDER[wk(items)]
        msgs += [{"role": "user", "content": say(split, [{"f": "никакую", "n": "никакое"}[g]],
                                                 ["не надо", "передумал", "отмена, ничего не удаляй", "ничего не удаляй", "пусть остаются"])},
                 {"role": "assistant", "content": pick(["Ок, ничего не удаляю 👌", "Хорошо, всё оставил как есть 🙂"])}]
        return msgs
    msgs.append({"role": "user", "content": pick_phrase(items, nums, split)})
    res2 = call(b, msgs, "delete_items", {"numbers": nums})
    msgs.append({"role": "assistant", "content": deleted_answer(res2, scope)})
    return msgs


def answer_delete(now, b, msgs, res, about, nf, split, scope=None):
    """Ответ на результат delete_items с query/без: удалено, список на выбор, не найдено, пусто."""
    if res.get("ok"):
        msgs.append({"role": "assistant", "content": deleted_answer(res, scope)})
    elif res.get("choose"):
        msgs.append({"role": "assistant", "content": choose_answer(now, res, about)})
        # после списка найденного «осталось» считается по всем записям, иначе — по показанному списку
        finish_choice(now, b, msgs, res, split, None if about else scope, asked_one=len(res["items"]) == 1)
    elif res.get("error") == "not_found":
        msgs.append({"role": "assistant", "content": not_found_answer(now, res, about, nf)})
        if rnd.random() < 0.6:
            finish_choice(now, b, msgs, res, split, None)
    else:
        assert res.get("error") == "nothing_to_delete", res
        msgs.append({"role": "assistant", "content": EMPTY[scope]})
    return msgs


def cat_del_query(split):
    """«Удали заметку про вайфай», «удали напоминание о дз по матану»: удаление по словам."""
    now = rand_now()
    kind = pick(["note"] * 5 + ["reminder"] * 5 + ["homework"] * 3)
    b = del_backend(now)
    seed_notes(b, rnd.randint(0, 4), rnd.randint(0, 3), rnd.choice([0, 1, 1, 2]) + (kind == "homework"))
    recs = [r.show for r in b._recs() if rk(r.show) == kind]
    if not recs or rnd.random() < 0.12:  # такого нет (или вообще ничего нет)
        key = missing_key(b, kind)
        if key is None:
            return None
        k, query, about = key
    else:
        k, query, about = key_of(pick(recs))
    q = pick(split_phr(DEL_QUERY_PHR[kind], split)).format(k=k)
    msgs = [system(now, b.profile), {"role": "user", "content": q}]
    res = call(b, msgs, "delete_items", {"query": query})
    return answer_delete(now, b, msgs, res, about, NF[kind], split)


DEL_CHOOSE_PHR = {
    None: ["удали заметку", "хочу удалить заметку", "надо удалить одну запись", "убери что-нибудь из заметок", "сотри заметку",
           "удали лишнюю заметку", "как удалить заметку?", "почисти заметки"],
    "reminder": ["удали напоминание", "хочу отменить напоминание", "убери одну напоминалку", "отмени напоминалку",
                 "сними напоминание", "удали лишнее напоминание", "как удалить напоминание?", "почисти напоминания"]}


def cat_del_choose(split):
    """«Удали заметку» без уточнения: список с номерами — студент выбирает номер, несколько, «все» или ничего."""
    now = rand_now()
    kind = pick([None, "reminder"])
    b = del_backend(now)
    if rnd.random() < 0.94:
        seed_notes(b, rnd.randint(0, 4), rnd.randint(1 if kind else 0, 3), rnd.choice([0, 0, 1]))
    q = pick(split_phr(DEL_CHOOSE_PHR[kind], split))
    msgs = [system(now, b.profile), {"role": "user", "content": q}]
    res = call(b, msgs, "delete_items", {"kind": kind} if kind else {})
    return answer_delete(now, b, msgs, res, None, None, split, kind)


def cat_del_after_list(split):
    """Сначала список (все записи, напоминания или брифинг), потом «удали 2», «вторую», «и 4 тоже»; иногда номера
    нет в списке (Ишка показывает список ещё раз) или номер уже удалён."""
    now = rand_now()
    src = pick(["all"] * 4 + ["reminders"] * 2 + ["brief"] * 2)
    day = pick(["today", "tomorrow"])
    if src == "brief" and day == "today":
        now = now.replace(hour=rnd.randint(7, 11))  # сводку на сегодня просят утром
    b = del_backend(now)
    seed_notes(b, rnd.randint(1, 4) if src == "all" else rnd.randint(0, 2), rnd.randint(1, 3), rnd.choice([0, 1, 1, 2]))
    if hasattr(b, "head") and b.profile.get("role") == "student" and rnd.random() < 0.4:  # студент в группе
        seed_posts(b, k=1)
    if src == "brief":
        q = pick(split_phr({"today": ["брифинг на сегодня", "проведи брифинг", "дай сводку на день", "что у меня сегодня?",
                                      "сводка на сегодня", "кинь сводку на сегодня"],
                            "tomorrow": ["дай сводку на завтра", "что у меня завтра?", "брифинг на завтра", "проведи брифинг на завтра",
                                         "сводка на завтра", "завтра что по плану?"]}[day], split))
        msgs = [system(now, b.profile), {"role": "user", "content": q}]
        res = call(b, msgs, "get_daily_brief", {"day": day})
        items = res.get("reminders", [])
        if len(items) < 2:
            return None
        msgs.append({"role": "assistant", "content": brief_answer(res, now, False)})
        scope = "reminder"
    else:
        q = pick(split_phr(LIST_PHR["reminders" if src == "reminders" else "all"], split))
        msgs = [system(now, b.profile), {"role": "user", "content": q}]
        res = call(b, msgs, "list_notes", {"kind": "reminder"} if src == "reminders" else {})
        items = res["items"]
        if len(items) < 2:
            return None
        msgs.append({"role": "assistant", "content": notes_answer(now, res, src)})
        scope = "reminder" if src == "reminders" else None
    n = pick(range(1, len(items) + 1))
    x = items[n - 1]
    g = GENDER[rk(x)]
    if rnd.random() < 0.5:
        ask = pick_phrase(items, [n], split)
    else:
        noun = {"note": pick(["заметку", "заметку номер", "запись"]), "reminder": pick(["напоминание", "напоминалку"]),
                "homework": pick(["напоминание", "напоминание о дз"])}[rk(x)]
        ask = say(split, [f"удали {noun} {n}", f"удали {n}"],
                  [f"{ORD[g][n - 1]} удали", f"удали {ORD[g][n - 1]}", f"{noun} {n} удали", f"сотри {noun} {n}",
                   f"{n} больше не нужно, удали", f"убери {noun} {n}"])
    if rnd.random() < 0.08:  # номера больше, чем в списке: Ишка показывает список ещё раз, студент поправляется
        bad = len(items) + rnd.randint(1, 3)
        msgs.append({"role": "user", "content": say(split, [f"удали {bad}"], [f"давай {bad}", f"номер {bad}", f"{bad} удали", f"удаляй {bad}", f"убери {bad}"])})
        res_bad = call(b, msgs, "delete_items", {"numbers": [bad]})
        msgs.append({"role": "assistant", "content": bad_number_answer(now, res_bad)})
        ask = pick_phrase(items, [n], split)
    msgs.append({"role": "user", "content": ask})
    res2 = call(b, msgs, "delete_items", {"numbers": [n]})
    msgs.append({"role": "assistant", "content": deleted_answer(res2, scope)})
    left = [m for m in range(1, len(items) + 1) if m != n]
    r = rnd.random()
    if left and r < 0.4:  # «и 4 тоже»: номера из того же списка, хотя одну запись уже удалили
        m = pick(left)
        gm = GENDER[rk(items[m - 1])]
        msgs.append({"role": "user", "content": say(split, [f"и {m} тоже"], [f"и {ORD[gm][m - 1]} тоже", f"ещё {m} удали",
                                                                              f"и {m} удали", f"а ещё {ORD[gm][m - 1]}"])})
        res3 = call(b, msgs, "delete_items", {"numbers": [m]})
        msgs.append({"role": "assistant", "content": deleted_answer(res3, scope)})
    elif left and r < 0.5:  # и только что удалённый, и ещё один: удаляем второй, про первый говорим
        nums = sorted([n, pick(left)])
        msgs.append({"role": "user", "content": say(split, [f"удали {nums[0]} и {nums[1]}"], [f"{nums[0]} и {nums[1]} удали",
                                                                                               f"убери {nums[0]} и {nums[1]}"])})
        res3 = call(b, msgs, "delete_items", {"numbers": nums})
        msgs.append({"role": "assistant", "content": deleted_answer(res3, scope)})
    elif r < 0.58:  # тот же номер ещё раз
        msgs.append({"role": "user", "content": say(split, [f"удали {n}"], [f"{n} удали", f"номер {n} удали", f"убери {n}"])})
        res3 = call(b, msgs, "delete_items", {"numbers": [n]})
        if res3.get("error") != "already_deleted":
            return None
        msgs.append({"role": "assistant", "content": pick([f"Номер {n} уже удалил 👌 Если надо убрать что-то ещё — назови номер.",
                                                           f"Это уже удалено 🙂 Номер {n} я убрал только что."])})
    return msgs


DEL_ALL_PHR = {
    None: ["удали все заметки", "очисти заметки", "удали всё, что я записывал", "сотри все записи", "почисти все заметки",
           "удали все мои заметки и напоминания", "очисти все записи", "снеси все заметки"],
    "reminder": ["удали все напоминания", "отмени все напоминалки", "очисти напоминания", "сними все напоминания",
                 "не надо больше ничего напоминать, удали все", "удали все напоминалки", "очисти все напоминалки", "отмени все напоминания"],
    "note": ["удали все заметки, а напоминания оставь", "очисти заметки, напоминания не трогай", "удали заметки без времени",
             "удали только заметки, напоминания пусть будут", "сотри все заметки, кроме напоминаний", "удали все заметки, но не напоминания"]}


def cat_del_all(split):
    """«Удали все заметки»: бэкенд сначала просит подтверждение (confirm), удаляет после «да»."""
    now = rand_now()
    kind = pick([None] * 3 + ["reminder"] * 3 + ["note"])
    b = del_backend(now)
    if rnd.random() < 0.94:
        seed_notes(b, rnd.randint(1 if kind == "note" else 0, 4), rnd.randint(1 if kind == "reminder" else 0, 3), rnd.choice([0, 0, 1]))
    q = pick(split_phr(DEL_ALL_PHR[kind], split))
    msgs = [system(now, b.profile), {"role": "user", "content": q}]
    args = {"all": True} | ({"kind": kind} if kind else {})
    res = call(b, msgs, "delete_items", args)
    if res.get("error"):
        msgs.append({"role": "assistant", "content": EMPTY[kind]})
        return msgs
    msgs.append({"role": "assistant", "content": confirm_answer(now, res)})
    r = rnd.random()
    if r < 0.65:
        msgs.append({"role": "user", "content": yes(split)})
        res2 = call(b, msgs, "delete_items", args)
        msgs.append({"role": "assistant", "content": deleted_answer(res2, kind, all_=True)})
    elif r < 0.85 or res["count"] == 1:
        msgs += [{"role": "user", "content": no(split)},
                 {"role": "assistant", "content": pick(["Ок, ничего не удаляю 👌", "Хорошо, всё оставил как есть 🙂", "Отбой, ничего не трогаю 👌"])}]
    else:
        n = pick([x["n"] for x in res["items"]])
        g = GENDER[rk(res["items"][n - 1])]
        msgs.append({"role": "user", "content": say(split, [f"только {n}"], [f"нет, только {ORD[g][n - 1]}", f"не все, только {n}",
                                                                              f"нет, только {n}", f"удали только {ORD[g][n - 1]}"])})
        res2 = call(b, msgs, "delete_items", {"numbers": [n]})
        msgs.append({"role": "assistant", "content": deleted_answer(res2, kind)})
    return msgs


def cat_del_limit_note(split):
    """Заметок и напоминаний уже 10: Ишка показывает их, студент выбирает, что удалить, — удаляем и записываем новое."""
    now = rand_now((7, 20))
    b = del_backend(now)
    k = rnd.randint(4, 10)
    for x_, _ in rnd.sample(NOTES, k):
        b.add_note(cap(x_))
    left = [t for t in REM_TASKS]
    while len([r for r in b._recs() if not r.code]) < 10:
        t = pick(left)
        left.remove(t)
        b.add_note(cap(t), at=pick(["tomorrow ", pick(WEEKDAYS) + " "]) + f"{rnd.randint(8, 21):02d}:{pick(['00', '30'])}")
    texts = {r.text for r in b._recs()}
    msgs = [system(now, b.profile)]
    if rnd.random() < 0.5:  # новая заметка
        new = pick([n for n, _ in NOTES if cap(n) not in texts] + ["купить зарядку для ноута", "номер старосты 8-913-000-12-34"])
        q = pick(split_phr(["запиши в заметки: {x}", "сохрани заметку: {x}", "заметка: {x}", "запомни: {x}",
                            "добавь в заметки {x}", "занеси в заметки — {x}"], split)).format(x=new)
        args = {"text": cap(new)}
        # без времени Ишка сначала спрашивает, напомнить ли (cat_note_clarify): студент отказался
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": pick(ASK_TIME["note"])},
                 {"role": "user", "content": say(split, ["не надо", "нет"], NO_TIME)}]
    else:
        new = pick([t for t in REM_TASKS if cap(t) not in texts])
        wp, targs = reminder_phrase(now)
        q = pick(split_phr(REMIND_PHR, split)).format(w=wp, x=new)
        args = {"text": cap(new)} | targs
        msgs.append({"role": "user", "content": q})
    res = call(b, msgs, "add_note", args)
    if res.get("error") != "limit":
        return None
    lines = item_lines(now, res["items"])
    msgs.append({"role": "assistant", "content": pick([
        f"У тебя уже {res['limit']} заметок и напоминаний — это максимум 📝 Что удалить, чтобы записать новое?\n{lines}\n\n{ask_tail()}",
        f"Записей уже {res['limit']}, больше не влезет 🙃 Удали какую-нибудь — и я запишу новую:\n{lines}\n\n{ask_tail()}"])})
    r = rnd.random()
    if r < 0.15:
        msgs += [{"role": "user", "content": say(split, ["не надо"], ["тогда не записывай", "ладно, забей", "тогда не надо", "забей"])},
                 {"role": "assistant", "content": pick(["Ок, ничего не трогаю 👌", "Хорошо, оставил всё как есть 🙂"])}]
        return msgs
    if r < 0.27:
        msgs.append({"role": "user", "content": say(split, ["все"], ["удали все старые", "очисти все", "удаляй все", "старые все удали"])})
        res2 = call(b, msgs, "delete_items", {"all": True})
    else:
        nums = [pick(range(1, 11))] if r < 0.85 else sorted(rnd.sample(range(1, 11), 2))
        msgs.append({"role": "user", "content": pick_phrase(res["items"], nums, split)})
        res2 = call(b, msgs, "delete_items", {"numbers": nums})
    res3 = call(b, msgs, "add_note", args)
    assert res3.get("ok"), res3
    if res3.get("fire_at"):
        tail = "🗑️ " + reminder_answer(now, res3, new)
    else:
        tail = pick(["и записал новую заметку 📝", "— новая заметка записана ✍️"])
    msgs.append({"role": "assistant", "content": f"{gone_phrase(res2)} {tail}"})
    return msgs


def cat_del_limit_hw(split):
    """Напоминаний о Д/З уже 10: удаляем одно из старых и ставим новое."""
    now = rand_now((7, 20))
    b = del_backend(now)
    subs = student_subjects(b)
    if not subs:
        return None
    pairs = [(c, t) for c in subs for t in HW_TASKS[c] + [None]]  # разные задания, без повторов, если хватает
    for code, task in rnd.sample(pairs, 10) if len(pairs) >= 10 else [pick(pairs) for _ in range(10)]:
        b.add_homework_reminder(SUBJECTS[code]["name"], task)
    code = pick(subs)
    alias, task = sub_alias(code), pick(HW_TASKS[code])
    q = pick(split_phr(HW_REM_PHR, split)).format(sd=sdat(alias), t=task)
    args = {"subject": alias, "task": task}
    msgs = [system(now, b.profile), {"role": "user", "content": q}]
    res = call(b, msgs, "add_homework_reminder", args)
    if res.get("error") != "limit":
        return None
    lines = item_lines(now, res["items"])
    msgs.append({"role": "assistant", "content": pick([
        f"Напоминаний о Д/З уже {res['limit']} — это максимум ⏰ Какое удалить, чтобы поставить новое?\n{lines}\n\n{ask_tail()}",
        f"У тебя уже {res['limit']} напоминаний о Д/З, больше нельзя 🙃 Какое убрать?\n{lines}\n\n{ask_tail()}"])})
    if rnd.random() < 0.12:
        msgs += [{"role": "user", "content": say(split, ["не надо"], ["тогда не ставь", "ладно, забей", "тогда не надо", "забей"])},
                 {"role": "assistant", "content": pick(["Ок, ничего не трогаю 👌", "Хорошо, оставил всё как есть 🙂"])}]
        return msgs
    nums = [pick(range(1, 11))] if rnd.random() < 0.85 else sorted(rnd.sample(range(1, 11), 2))
    msgs.append({"role": "user", "content": pick_phrase(res["items"], nums, split)})
    res2 = call(b, msgs, "delete_items", {"numbers": nums})
    res3 = call(b, msgs, "add_homework_reminder", args)
    if not res3.get("ok"):
        return None
    msgs.append({"role": "assistant", "content": f"{gone_phrase(res2)} 🗑️ {hw_rem_answer(now, res3, task)}"})
    return msgs


def cat_del_student_cancel(split):
    """Студент просит удалить объявление или Д/З от старосты: это может только староста, а само оно пропадёт после события
    (без даты — через неделю), Д/З — после срока. Инструмента у студента нет — отвечаем текстом."""
    now = rand_now()
    b = group_backend(now, rand_profile(), "student")
    if rnd.random() < 0.6:
        text, _ = rand_announcement(now)
        title = text.split(", ")[0]
        k = GROUP_REM_KEYS[low(title)][0] if low(title) in GROUP_REM_KEYS else low(title)
        q = pick(split_phr(["удали объявление про {k}", "убери объявление старосты про {k}", "можно удалить объявление про {k}?",
                            "не хочу видеть объявление про {k}, убери", "удали из брифинга объявление про {k}",
                            "отмени объявление про {k}", "убери объявление про {k}", "удали объявление старосты про {k}"], split)).format(k=k)
        ans = pick(["Объявления старосты удалить может только староста 🙂 Это объявление пропадёт само: после события, "
                    "а если даты нет — через неделю.",
                    "Удалить объявление может только староста — попроси его 🙂 А так оно исчезнет само после события "
                    "(без даты — через неделю)."])
    else:
        code = pick(student_subjects(b) or ["math"])
        k = sdat(sub_alias(code))
        q = pick(split_phr(["удали дз от старосты по {k}", "убери дз по {k}, которое прислал староста", "отмени дз старосты по {k}",
                            "удали из брифинга дз по {k} от старосты", "убери дз от старосты по {k}", "удали домашку от старосты по {k}"],
                           split)).format(k=k)
        ans = pick(["Д/З от старосты удалить может только староста 🙂 Оно пропадёт само, когда пройдёт срок сдачи.",
                    "Это Д/З отправил староста, удалить его может только он 🙂 После срока сдачи оно исчезнет само."])
    return [system(now, b.profile), {"role": "user", "content": q}, {"role": "assistant", "content": ans}]


# ---------- староста удаляет объявления и Д/З у группы ----------
POST_Q = {None: "Что удалить?", "homework": "Какое Д/З удалить?", "announcement": "Какое объявление удалить?"}
POST_NOUN = {None: ("запись", "записи", "записей"), "homework": ("Д/З", "Д/З", "Д/З"),
             "announcement": ("объявление", "объявления", "объявлений")}
ONE_POST = {None: "Сейчас у группы одна запись:", "homework": "Сейчас у группы одно Д/З:",
            "announcement": "Сейчас у группы одно объявление:"}
NOT_POSTED = {None: "Сейчас группе ничего не отправлено — удалять нечего 🙂",
              "homework": "Д/З группе сейчас не отправлено — удалять нечего 🙂",
              "announcement": "Объявлений у группы сейчас нет — удалять нечего 🙂"}


def post_line(now, x, kind=None):
    k = x.get("kind", kind)
    sub = f", только {SUB_ORD[x['subgroup']]} подгруппе" if x.get("subgroup") else ""
    n = f"{x['n']}. "
    if k == "homework":
        return n + f"Д/З по {dat(short(x['subject']))}: {x['task']} ({when_ru(now, x['due'])})" + sub
    return (n + ("" if kind else "объявление ") + f"«{x['text']}»" + (f" — {when_ru(now, x['at'])}" if x.get("at") else "") + sub)


def post_lines(now, items, kind=None):
    return "\n".join(post_line(now, x, kind) for x in items)


def post_what(x, kind=None):
    if x.get("kind", kind) == "homework":
        return f"Д/З по {dat(short(x['subject']))} ({x['task']})"
    return f"объявление «{x['text']}»"


def group_deleted_answer(res, kind=None):
    d = res["deleted"]
    if len(d) == 1:
        w = post_what(d[0], kind)
        return pick([f"Удалил 📢 {cap(w)} пропало у всей группы.", f"Готово ✅ {cap(w)} больше никто не увидит.",
                     f"Удалил {w} — у группы оно пропало 👌"])
    kinds = {x.get("kind", kind) for x in d}
    noun = POST_NOUN[kinds.pop() if len(kinds) == 1 else None]
    return f"Удалил {len(d)} {plural(len(d), *noun)} 📢 У группы они пропали."


def post_key(x):
    """(как назвать запись после «про»/«по», query, kind)."""
    if x["kind"] == "homework":
        alias = sub_alias(x["code"]) if x.get("code") else x["subject"]
        return sdat(alias), alias, "homework"
    title = x["text"].split(", ")[0]  # «Хакатон «Код Сибири», МКЦ» — место не называют
    if low(title) in GROUP_REM_KEYS:
        k, query = GROUP_REM_KEYS[low(title)]
        return k, query, "announcement"
    return low(title), title, "announcement"


def missing_post(b, recs):
    """Запись, которой нет: староста называет то, чего не отправлял. -> (запись как у объявления или Д/З) или None."""
    for _ in range(20):
        if rnd.random() < 0.5:
            code = pick(list(HW_TASKS))
            x = {"kind": "homework", "code": code, "subject": SUBJECTS[code]["name"]}
        else:
            x = {"kind": "announcement", "text": rand_announcement(b.now)[0]}
        _, query, _ = post_key(x)
        if not would_find(recs, query):  # и среди записей другого вида: без kind ищет по всем
            return x
    return None


def post_reply(split, nums):
    if len(nums) == 1:
        n = nums[0]
        return say(split, [str(n), str(n)], [f"давай {n}", f"номер {n}", f"удали {n}", f"{n}, удаляй"])
    j = " и ".join(map(str, nums))
    return say(split, [j], [", ".join(map(str, nums)), "давай " + j, "удали " + j, j + ", удаляй"])


POST_QUERY_PHR = {
    "announcement": ["удали объявление про {k}", "убери объявление про {k}", "отмени объявление про {k}",
                     "объявление про {k} больше не актуально, удали", "удали у группы объявление про {k}",
                     "сними объявление про {k}", "удали у всех объявление про {k}", "убери у группы объявление про {k}"],
    "homework": ["удали дз по {k} у группы", "отмени дз по {k}", "убери у группы дз по {k}", "дз по {k} отменили, удали у всех",
                 "удали у всех домашку по {k}", "сними дз по {k}", "удали у группы дз по {k}", "отмени у всех домашку по {k}"],
    None: ["удали у группы всё про {k}", "убери то, что отправлял группе про {k}", "удали отправленное группе про {k}",
           "отзови у группы то, что про {k}", "удали у всех то, что про {k}", "убери у группы запись про {k}"]}


def cat_group_delete(split):
    """Староста удаляет объявление или Д/З: пропадёт у всей группы. Иногда такого нет или этого вида не отправлял вообще."""
    now = rand_now()
    b = group_backend(now, rand_profile(), "starosta")
    seed_posts(b, k=rnd.randint(2, 4))
    recs = b._group_recs()
    if not recs:
        return None
    msgs = [system(now, b.profile)]
    form = pick(["query", "query", "query", "choose", "choose", "kind_choose", "all"])
    if form == "query":
        x = missing_post(b, recs) if rnd.random() < 0.12 else pick(recs).show
        if x is None:
            return None
        k, query, kind = post_key(x)
        named = kind == "homework" or rnd.random() < 0.75  # «удали объявление про собрание» — вид назван
        q = pick(split_phr(POST_QUERY_PHR[kind if named else None], split)).format(k=k)
        args = ({"kind": kind} if named else {}) | {"query": query}
        msgs.append({"role": "user", "content": q})
        res = call(b, msgs, "group_delete_post", args)
        if res.get("ok"):
            msgs.append({"role": "assistant", "content": group_deleted_answer(res, args.get("kind"))})
            return msgs
        if res.get("error") == "nothing_to_delete":
            msgs.append({"role": "assistant", "content": NOT_POSTED[args.get("kind")]})
            return msgs
        if res.get("error") == "not_found":
            about = f"Д/З по {dat(query)}" if kind == "homework" else f"Объявление про {k}"
            msgs.append({"role": "assistant", "content": f"{about} не нашёл 🤔 Сейчас у группы:\n"
                                                         f"{post_lines(now, res['items'], args.get('kind'))}\n\n"
                                                         f"{POST_Q[args.get('kind')]} Напиши номер."})
            if rnd.random() < 0.4:
                msgs += [{"role": "user", "content": say(split, ["ничего"], ["тогда ничего", "ничего не удаляй", "не, не стоит",
                                                                               "значит, не надо"])},
                         {"role": "assistant", "content": pick(["Ок, ничего не удаляю 👌", "Хорошо, всё остаётся 🙂"])}]
                return msgs
        else:
            assert res.get("choose"), res
            n_ = len(res["items"])
            head = (f"Нашёл только это:\n{post_lines(now, res['items'], args.get('kind'))}\n\nУдалить? Пропадёт у всей группы."
                    if n_ == 1 else
                    f"Нашёл {n_} {plural(n_, *POST_NOUN[args.get('kind')])}:\n{post_lines(now, res['items'], args.get('kind'))}\n\n"
                    f"{POST_Q[args.get('kind')]} Напиши номер.")
            msgs.append({"role": "assistant", "content": head})
            if n_ == 1:
                msgs.append({"role": "user", "content": yes(split)})
                res2 = call(b, msgs, "group_delete_post", {"numbers": [1]})
                msgs.append({"role": "assistant", "content": group_deleted_answer(res2, args.get("kind"))})
                return msgs
        n = pick([y["n"] for y in res["items"]])
        msgs.append({"role": "user", "content": post_reply(split, [n])})
        res2 = call(b, msgs, "group_delete_post", {"numbers": [n]})
        msgs.append({"role": "assistant", "content": group_deleted_answer(res2, args.get("kind"))})
        return msgs
    kind = None
    if form in ("kind_choose", "all") and rnd.random() < 0.6:
        have = sorted({r.show["kind"] for r in recs})
        kind = pick([k_ for k_ in ("homework", "announcement") if k_ not in have] or have) if rnd.random() < 0.1 else pick(have)
    if form == "all":
        q = pick(split_phr({None: ["удали всё, что отправлял группе", "удали у группы все объявления и дз", "очисти всё, что я отправлял",
                                   "удали всё у группы", "отзови всё, что отправил группе", "удали все мои объявления и дз"],
                            "announcement": ["удали все объявления", "отмени все объявления для группы", "очисти объявления",
                                             "удали у группы все объявления", "сними все объявления", "убери все объявления у всех"],
                            "homework": ["удали у группы все дз", "отмени всё дз, что отправлял", "очисти дз у группы",
                                         "удали всю домашку, что отправлял", "сними все дз у группы", "убери у всех всё дз"]}[kind], split))
        args = {"all": True} | ({"kind": kind} if kind else {})
        msgs.append({"role": "user", "content": q})
        res = call(b, msgs, "group_delete_post", args)
        if res.get("error") == "nothing_to_delete":
            msgs.append({"role": "assistant", "content": NOT_POSTED[kind]})
            return msgs
        n = res["count"]
        head = (f"Точно удалить все {n} {plural(n, *POST_NOUN[kind])}? Они пропадут у всей группы:" if n > 1
                else f"{ONE_POST[kind]} Удалить? Оно пропадёт у всей группы:")
        msgs.append({"role": "assistant", "content": f"{head}\n{post_lines(now, res['items'], kind)}\n\nНапиши «да» — удалю."})
        if rnd.random() < 0.75:
            msgs.append({"role": "user", "content": say(split, ["да", "да"], ["да, удаляй", "ага", "давай", "удаляй все", "да, удаляй всё"])})
            res2 = call(b, msgs, "group_delete_post", args)
            msgs.append({"role": "assistant", "content": group_deleted_answer(res2, kind)})
        else:
            msgs += [{"role": "user", "content": no(split)},
                     {"role": "assistant", "content": pick(["Ок, ничего не удаляю 👌", "Хорошо, всё остаётся 🙂"])}]
        return msgs
    q = pick(split_phr({None: ["хочу удалить то, что отправлял группе", "удали одно объявление или дз у группы",
                               "что я отправлял группе? хочу удалить лишнее", "удали что-нибудь из отправленного группе",
                               "надо удалить одну запись у группы", "покажи, что я отправлял, удалю лишнее"],
                        "homework": ["удали дз, которое я отправлял", "хочу удалить дз у группы", "отзови домашку у группы",
                                     "удали у группы одно дз", "отмени одно дз группе", "сними одно дз у группы"],
                        "announcement": ["удали объявление", "хочу удалить объявление для группы", "убери у группы объявление",
                                         "отмени одно объявление", "сними объявление", "удали одно объявление у всех"]}[kind], split))
    args = {"kind": kind} if kind else {}
    msgs.append({"role": "user", "content": q})
    res = call(b, msgs, "group_delete_post", args)
    if res.get("error") == "nothing_to_delete":
        msgs.append({"role": "assistant", "content": NOT_POSTED[kind]})
        return msgs
    items = res["items"]
    if len(items) == 1:
        msgs.append({"role": "assistant", "content": f"{ONE_POST[kind]}\n{post_lines(now, items, kind)}\n\nУдалить? Пропадёт у всей группы."})
        msgs.append({"role": "user", "content": say(split, ["да", "да"], ["да, удаляй", "ага", "давай", "удаляй", "да, удали"])})
        res2 = call(b, msgs, "group_delete_post", {"numbers": [1]})
    else:
        msgs.append({"role": "assistant", "content": f"{POST_Q[kind]} 📢\n{post_lines(now, items, kind)}\n\n"
                                                     "Напиши номер (можно несколько) — пропадёт у всей группы."})
        nums = [pick([y["n"] for y in items])] if rnd.random() < 0.8 else sorted(rnd.sample([y["n"] for y in items], 2))
        msgs.append({"role": "user", "content": post_reply(split, nums)})
        res2 = call(b, msgs, "group_delete_post", {"numbers": nums})
    msgs.append({"role": "assistant", "content": group_deleted_answer(res2, kind)})
    return msgs


def cat_group_limit(split):
    """У группы уже максимум объявлений (или Д/З): староста удаляет старое — и новое уходит."""
    kind = pick(["homework", "announcement"])
    for _ in range(40):  # длинный диалог собираем заново с тем же видом, чтобы фильтр длины не сдвинул доли
        m = group_limit_dialog(split, kind)
        if m is None or est_tokens(m) <= MAX_TOKENS:
            return m
    return None


def group_limit_dialog(split, kind):
    now = rand_now((7, 20))
    b = group_backend(now, rand_profile(), "starosta")
    # в бэкенде лимит 10, но список из 10 записей с промптом старосты не влезает в 4096 токенов обучения:
    # здесь лимит 2, модель учится тому же (показать список, удалить выбранное, отправить новое)
    b.limits["posts"] = 2
    for _ in range(40):
        if len(b._group_recs(kind)) >= 2:
            break
        seed_posts(b, (kind,), 1)
    if len(b._group_recs(kind)) < 2:
        return None
    msgs = [system(now, b.profile)]
    if kind == "homework":
        code = pick(student_subjects(b) or ["math"])
        task, alias = pick(HW_TASKS[code]), sub_alias(code)
        q = group_hw_phrase(split, "next", None).format(s=alias, sd=sdat(alias), t=task)
        name_, args = "group_add_homework", {"subject": alias, "task": task, "due": "next_class"}
    else:
        what_ = pick(GROUP_REM)
        q = pick(split_phr(ANNOUNCE_PHR, split)).format(to=pick(["группе", "всем"]), tg="группы", x=what_)
        name_, args = "group_announce", {"text": cap(what_)}
    msgs.append({"role": "user", "content": q})
    res = call(b, msgs, name_, args)
    if res.get("error") != "limit":
        return None
    noun = {"homework": ("действующее Д/З", "действующих Д/З", "действующих Д/З"), "announcement": POST_NOUN["announcement"]}[kind]
    msgs.append({"role": "assistant", "content": f"У группы уже {res['limit']} {plural(res['limit'], *noun)} — больше нельзя 📢 "
                                                 f"Удалить какое-нибудь из старых?\n{post_lines(now, res['items'], kind)}\n\n"
                                                 "Напиши номер — удалю его и отправлю новое."})
    if rnd.random() < 0.15:
        msgs += [{"role": "user", "content": say(split, ["не надо"], ["тогда не отправляй", "ладно, не отправляй", "тогда не надо",
                                                                        "оставь как есть"])},
                 {"role": "assistant", "content": "Ок, ничего не удаляю и не отправляю 👌"}]
        return msgs
    n = pick([y["n"] for y in res["items"]])
    msgs.append({"role": "user", "content": post_reply(split, [n])})
    res2 = call(b, msgs, "group_delete_post", {"numbers": [n]})
    res3 = call(b, msgs, name_, args)
    if not res3.get("ok"):
        return None
    new = group_hw_answer(now, res3, args["task"]) if kind == "homework" else announce_answer(now, res3)
    msgs.append({"role": "assistant", "content": f"Удалил {post_what(res2['deleted'][0], kind)} 🗑️ {new}"})
    return msgs


MAX_TOKENS = 4050  # обучение режет после 4096 токенов


def est_tokens(m):
    """Длина диалога в токенах без токенизатора: описание инструментов — 3,2 символа на токен (у Qwen3 ровно 3,22),
    всё остальное — 2,2. Проверено токенизатором Qwen3 на всём датасете и 5700 диалогах удаления: оценка
    не меньше настоящей длины (больше на 5–375 токенов)."""
    full = len(prompt_format.render(m))
    rest = len(prompt_format.render(m, []))  # без блока инструментов
    return (full - rest) / 3.2 + rest / 2.2


def fits(cat):
    """Диалог длиннее MAX_TOKENS не берём (при обучении обрезался бы конец — сам ответ), а собираем заново
    той же категорией: так доля категорий не сдвигается."""
    def f(split):
        for _ in range(40):
            m = cat(split)
            if m is None or est_tokens(m) <= MAX_TOKENS:
                return m
        return None
    f.__name__ = cat.__name__
    return f


DELETE_CATS = [(fits(c), w) for c, w in [(cat_del_query, 60), (cat_del_choose, 40), (cat_del_after_list, 40), (cat_del_all, 25),
                                         (cat_del_limit_note, 10), (cat_del_limit_hw, 6), (cat_del_student_cancel, 8),
                                         (cat_group_delete, 30), (cat_group_limit, 8)]]


# Только train: после первого обучения модель вызывала инструмент даже на «привет» (ответов текстом было 6%)
TEXT_CATS = [(cat_chat_extra, 1)]


NEW_CATS = [(cat_note_timed, 9), (cat_note_clarify, 8), (cat_hw_reminder, 7), (cat_note_list, 5), (cat_session, 5),
            (cat_brief, 7), (cat_chat, 7)]


# skip_* — функций больше нет, но случайные числа тратят как раньше (см. skip_add_exam)
CATS = [(cat_schedule, 30), (cat_room, 18), (cat_teacher, 14), (cat_travel, 12), (cat_food, 12), (skip_add_exam, 7), (skip_profile, 7)]


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
    # 1) Расписание, аудитории, преподаватели, дорога, еда. Порядок вызовов rnd не меняем: иначе разъедутся ключи
    #    перефразировок в data/paraphrases.json.
    data = {}
    for split, n in (("train", 900), ("eval", 100)):
        rows = build(n, split)
        if split == "train":
            print("перефразировано:", sum(paraphrase.apply(m, overlay) for m in rows))
        rnd.shuffle(rows)
        data[split] = rows
    # 2) Заметки и напоминания, напоминания о Д/З, сессия, брифинг, болтовня — отдельный генератор.
    rnd = random.Random(2026)
    for split, n in (("train", 480), ("eval", 5)):
        # eval — поровну на каждую категорию (5 × 7), чтобы оценка покрывала все инструменты
        rows = build(n, split, NEW_CATS) if split == "train" else [r for c in NEW_CATS for r in build(n, split, [c])]
        if split == "train":
            print("перефразировано (новые):", sum(paraphrase.apply(m, overlay) for m in rows))
        data[split] += rows
        rnd.shuffle(data[split])
        if split == "train":
            # 3) Ответы текстом без инструментов (data/text_extra.json) — свой генератор, чтобы не сдвинуть ключи перефразов
            main_rnd, rnd = rnd, random.Random(3030)
            data[split] += build(150, split, TEXT_CATS)
            rnd.shuffle(data[split])
            rnd = main_rnd  # eval собирается дальше тем же генератором, что и раньше
        # 4) Староста (объявления и Д/З группе) и мероприятия — тоже свой генератор; eval — по 5 на категорию
        main_rnd, rnd = rnd, random.Random(4040 if split == "train" else 4041)
        # train: не меньше 8 на категорию (редкие иначе выпадают почти целиком), остальное по весам
        rows = [r for c in STAROSTA_CATS for r in build(8 if split == "train" else 5, split, [c])]
        if split == "train":
            rows += build(240 - len(rows), split, STAROSTA_CATS)
            print("перефразировано (староста, мероприятия):", sum(paraphrase.apply(m, overlay) for m in rows))
        data[split] += rows
        rnd.shuffle(data[split])
        rnd = main_rnd
        # 5) Удаление и лимиты (delete_items, group_delete_post) — свой генератор; eval — по 5 на категорию
        main_rnd, rnd = rnd, random.Random(5050 if split == "train" else 5051)
        rows = [r for c in DELETE_CATS for r in build(8 if split == "train" else 5, split, [c])]
        if split == "train":
            rows += build(222 - len(rows), split, DELETE_CATS)
        data[split] += rows
        rnd.shuffle(data[split])
        # 6) «Сделай напоминание о ДЗ» без предмета, «напомни в пятницу …» без часа, зачёты в сессии — свой генератор,
        #    чтобы не сдвинуть разделы выше; eval — по 5 на категорию
        rnd = random.Random(6060 if split == "train" else 6061)
        for cat, n in ((cat_hw_reminder_ask, 40), (cat_note_day_only, 50), (lambda sp: cat_session(sp, mixed=True), 30)):
            data[split] += build(n if split == "train" else 5, split, [(cat, 1)])
        rnd.shuffle(data[split])
        rnd = main_rnd
        with open(root / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for m in data[split]:
                f.write(json.dumps({"messages": m}, ensure_ascii=False) + "\n")
        print(split, len(data[split]))
