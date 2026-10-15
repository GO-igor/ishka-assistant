"""Списки с номерами и удаление по ним — общее для личных записей (delete_items) и рассылок старосты
(group_delete_post).

Номер в «удали 2» — из последнего списка, который видел студент (список записей, брифинг, «что удалить?»).
Conversation помнит, какие записи (их id в базе) стояли под какими номерами, поэтому после удаления второй
«и 4 тоже» — та же четвёртая. «Удалить все» — только с подтверждением: сначала confirm со списком, удаляем
следующим вызовом, и ровно то, что показали. Если чат сообщает о каждом сообщении студента (new_turn),
номера относятся только к спискам из прошлых сообщений, а «да» — только к списку из предыдущего: модель
не подтвердит удаление сама себе вторым вызовом.

Conversation хранится отдельно от Backend: сайт держит одну на разговор (state() — обычный dict для JSON)
и создаёт Backend на каждый запрос."""
import json, re, unicodedata
from collections import namedtuple
from campus import SUBJECTS, resolve_subject

# id — запись в базе; show — что показать модели; text — где искать слова запроса; code — предмет (ищем и по сокращениям)
Rec = namedtuple("Rec", "id show text code")
STOP = {"по", "в", "во", "на", "с", "со", "к", "ко", "и", "о", "об", "про", "для", "от", "до", "за", "из", "у", "мою",
        "мой", "мое", "моё", "мои", "мне", "все", "всё"}  # предлоги и местоимения в query не мешают поиску


class Conversation:
    def __init__(self, state=None):
        self.lists, self.pending, self.turn = {}, {}, None  # показанные списки, что ждёт «да», номер сообщения
        if state:
            self.restore(state)

    def new_turn(self):
        """Пришло новое сообщение студента: chat.py и сайт вызывают это перед ответом модели."""
        self.turn = (self.turn or 0) + 1

    def state(self):
        """Снимок (JSON-совместимый). Если ход не удался и убран из истории, chat.py возвращает снимок (restore):
        студент тех списков не видел."""
        return json.loads(json.dumps({"lists": self.lists, "pending": self.pending, "turn": self.turn}))

    def restore(self, state):
        state = json.loads(json.dumps(state))
        self.lists, self.pending, self.turn = state["lists"], state["pending"], state["turn"]

    def seen(self, entry, previous=False):
        """Список студент уже видел: показан в одном из прошлых сообщений, previous=True — ровно в предыдущем.
        Без new_turn (генератор датасета, тесты) сообщения не считаем."""
        if entry is None:
            return False
        if self.turn is None or entry.get("turn") is None:
            return True
        return entry["turn"] == self.turn - 1 if previous else entry["turn"] < self.turn

    def numbered(self, ns, recs, ask=False, kind=None, part=False, found=False):
        """Список с номерами n; запоминаем его (пустой прежний не заменяет). ask=True — список, чтобы выбрать, что
        удалить: ответ «все» удаляет его весь. kind — какие записи в нём (только заметки, только напоминания).
        part=True — это подтверждение «удалить все про X». found — список найденного по словам."""
        ids = [r.id for r in recs]
        if ids:
            self.lists[ns] = {"ids": ids, "turn": self.turn} | ({"found": True} if found else {})
        if ask:
            self.pending[ns] = {"ids": ids, "turn": self.turn, "kind": kind, "part": part}
        return [{"n": i} | r.show for i, r in enumerate(recs, 1)]

    def delete(self, ns, recs, numbers, query, all_, drop, kind=None):
        """Удаление по номерам, словам (query) или всех (all_). recs — что можно удалить сейчас (с учётом kind),
        drop(записи) удаляет их из базы. Возвращает ответ инструмента."""
        all_ = all_ is True or str(all_).strip().lower() in ("true", "1", "yes", "да")
        pending = self.pending.get(ns)
        # список ждёт ответа до следующего удаления; показанный в этом же сообщении (модель удаляет несколько
        # записей подряд) — ждёт дальше, иначе «все» в ответ на него спросит про все записи
        if all_ or self.turn is None or (pending or {}).get("turn") != self.turn:
            self.pending.pop(ns, None)
        if not self.seen(pending, previous=True):
            pending = None  # «да» подтверждает только список из прошлого сообщения
        stems = _stems(query) if query else []
        # запрос без слов, кроме предлогов («🎂», «с 🎂»), — тоже отбор: ищем эти знаки как есть, а не считаем его «все»
        marks = [] if stems else re.findall(r"[^\w\s]+", query or "")
        if not recs:
            return {"error": "nothing_to_delete"}
        shown, by_number = self.lists.get(ns), numbers not in (None, "", [])
        live = {r.id: r for r in recs}
        if by_number:
            nums = _nums(numbers)
            if not self.seen(shown):  # списка не показывали: какую запись студент считает второй, не знаем
                return {"choose": True, "items": self.numbered(ns, recs, ask=True, kind=kind)}
            ids = shown["ids"]
            bad = [n for n in nums if not 1 <= n <= len(ids)] if nums else [numbers]
            if bad:  # номера остальных не меняем: студент поправит опечатку по тому же списку
                still = [(i, x) for i, x in enumerate(ids, 1) if x in live]
                if not still:
                    return {"error": "bad_number", "numbers": bad, "items": self.numbered(ns, recs, ask=True, kind=kind)}
                self.pending[ns] = {"ids": [x for _, x in still], "turn": self.turn, "kind": kind, "part": False}
                return {"error": "bad_number", "numbers": bad, "items": [{"n": i} | live[x].show for i, x in still]}
            picked = [(n, ids[n - 1]) for n in nums]
        elif stems or marks or all_:
            hits, guessed = recs, False
            if marks:
                hits = [r for r in recs if all(m in r.text for m in marks)]
            elif stems:
                hits = [r for r in recs if _matches(stems, r)]
                code = None if hits else resolve_subject(query)  # «матеша», «мат.логика» — по названию предмета
                if code:
                    hits = [r for r in recs if r.code == code]
                    guessed = True  # совпал только предмет, остальные слова — нет: сами не удаляем, спрашиваем
                elif len(hits) == 1 and not _close(query, hits[0]):
                    guessed = True  # слово совпало только с началом другого слова: тоже спрашиваем
            if not hits:
                return {"error": "not_found", "query": query, "items": self.numbered(ns, recs, ask=True, kind=kind)}
            ids = [r.id for r in hits]
            if all_:
                part = bool(stems or marks)
                # «да» на «удалить все?»: тот же отбор (по словам — те же записи; без слов — тот же вид или любой)
                ok = pending and (set(pending["ids"]) == set(ids) if part else
                                  not pending["part"] and kind in (None, pending["kind"]))
                if not ok:
                    return {"confirm": True, "count": len(hits),
                            "items": self.numbered(ns, hits, ask=True, kind=kind, part=part, found=part)}
                ids = pending["ids"]  # подтвердили: удаляем ровно то, что показали
            elif len(hits) > 1 or guessed:
                return {"choose": True, "items": self.numbered(ns, hits, ask=True, kind=kind, found=bool(stems or marks))}
            picked = [(None, x) for x in ids]
        else:
            return {"choose": True, "items": self.numbered(ns, recs, ask=True, kind=kind)}
        chosen, gone = [], []
        for n, x in picked:
            if x in live and live[x] not in chosen:
                chosen.append(live[x])
            elif x not in live and n is not None:
                gone.append(n)  # этот номер уже удалён
        if not chosen:
            return {"error": "already_deleted"}
        drop(chosen)
        out = {"ok": True, "deleted": [r.show for r in chosen]} | ({"gone": gone} if gone else {})
        left = set(live) - {r.id for r in chosen}
        # сколько осталось: по номеру из полного списка — в нём, иначе всего (после «Нашёл только это» и «да»
        # нельзя сказать «больше ничего нет», если есть другие записи)
        if by_number and not shown.get("found"):
            return out | {"left": sum(x in left for x in shown["ids"])}
        return out | {"left": len(left)}


# ---------- поиск записей по словам ----------
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


def _rec_words(r):
    """Слова записи и отдельно — название её предмета с сокращениями («питон»)."""
    return _words(r.text), _words(" ".join([SUBJECTS[r.code]["name"]] + SUBJECTS[r.code]["aliases"]) if r.code in SUBJECTS else "")


def _matches(stems, r):
    """Подходит ли запись под слова запроса. Предмет узнаём и по сокращениям («удали дз по питону» — а в записи
    «Основы программирования на Python»); число («лаба 3») — только целиком и не из названия предмета («1.3»)."""
    mw, sw = _rec_words(r)
    return all(s in mw if s.isdigit() else any(w.startswith(s) for w in mw + sw) for s in stems)


def _close(query, r):
    """Каждое слово запроса совпало со словом записи почти целиком, а не с началом другого, длинного слова:
    «Коля» — не «коллоквиум», «Маша» — не «машину». Только тогда одну найденную запись удаляем без вопроса."""
    mw, sw = _rec_words(r)

    def ok(w):
        s = _stem(w)
        return any(x.startswith(s) and len(x) <= len(w) + (1 if len(w) <= 4 else 2) for x in mw + sw)
    return all(ok(w) for w in _words(query) if w not in STOP and not w.isdigit())


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
            v = int(v.strip())
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
            return None  # is_integer: у inf и NaN — False
        if int(v) not in out:
            out.append(int(v))
    return out
