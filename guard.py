"""Запретные фразы (db/blocklist.json): на вопрос с такой фразой Ишка не отвечает, а свой ответ (или вызов
инструмента) с такой фразой не выводит — вместо него короткий отказ из списка. Фильтр стоит снаружи модели
(chat.py, потом сайт), поэтому переобучать не нужно: список можно менять в любой момент, даже не закрывая чат.

Как сравниваются фразы (одинаково для фразы и текста): регистр не важен, ё = е, знаки препинания не важны,
латинские буквы, похожие на русские (a, c, e, o, p, x, y, k, а заглавные ещё B, H, M, T), и цифра 0 считаются
русскими: «нaркотики» с латинской a — то же, что «наркотики». Невидимые символы и ударения внутри слова
не мешают. Звёздочка в конце слова — любое окончание: «экзамен*» найдёт «экзамена», «экзаменов». Звёздочка
в начале — любое начало слова. Между словами фразы в тексте может стоять одно другое слово: «реши за меня
контрольн*» найдёт и «реши пожалуйста за меня контрольную». Группы проверяются сверху вниз, срабатывает первая."""
import json, os, pathlib, re, unicodedata

PATH = pathlib.Path(__file__).parent / "db" / "blocklist.json"
DEFAULT_ANSWER = "Об этом я говорить не буду 🙅 Давай лучше про учёбу?"
MAX_GAP = 1  # сколько чужих слов может стоять между словами фразы (с двумя ловилось «сделать доклад про оружие»)
_UPPER = str.maketrans("ABEHKMOPCTXY", "АВЕНКМОРСТХУ")  # заглавные латинские, похожие на русские (до lower)
_LOOKALIKE = str.maketrans("aceopxyk0ё", "асеорхукое")


def words(text):
    """Текст -> слова для сравнения."""
    t = unicodedata.normalize("NFC", str(text)).translate(_UPPER).lower().translate(_LOOKALIKE)
    t = "".join(c for c in t if unicodedata.category(c) not in ("Cf", "Mn"))  # мягкий перенос, ударения и т. п.
    return re.findall(r"\w+", t)


def _phrase(p):
    """Фраза -> регулярки по одной на слово. «*» в начале или конце куска — любое начало или окончание слова."""
    out = []
    for piece in str(p).split():
        ws = words(piece.strip("*"))
        for i, w in enumerate(ws):  # «мастер-класс*» -> мастер, класс*
            head = i == 0 and piece.startswith("*")
            tail = i == len(ws) - 1 and piece.endswith("*")
            out.append(re.compile(("\\w*" if head else "") + re.escape(w) + ("\\w*" if tail else "") + r"\Z"))
    return out


class Guard:
    def __init__(self, path=PATH):
        """Ошибка в файле не исключение: она в self.error (chat.py покажет её при запуске), список пока пустой."""
        self.path, self.mtime, self.groups = pathlib.Path(path), None, []
        self.answer, self.error = DEFAULT_ANSWER, None
        self.reload()

    def reload(self):
        """Перечитать список, если файл изменился. Сломанный файл не роняет чат: остаётся прежний список."""
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:  # файла нет — запретов нет; если он пропал на ходу (редактор пересохраняет), список прежний
            return
        if mtime == self.mtime:
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                answer, groups = _parse(json.load(f))
        except OSError:  # файл как раз перезаписывают: попробуем на следующем вопросе
            return
        except (ValueError, TypeError) as e:
            self.error = f"{self.path.name}: ошибка в файле ({e}); оставляю прежний список"
            self.mtime = mtime
            return
        self.groups, self.answer, self.mtime, self.error = groups, answer, mtime, None

    def check(self, text):
        """Первая запретная фраза в тексте: {"group", "phrase", "answer"} или None."""
        self.reload()
        ws = words(text)
        if not ws:
            return None
        for g in self.groups:
            for phrase, pat in g["phrases"]:
                if _find(ws, pat):
                    return {"group": g["name"], "phrase": phrase, "answer": g["answer"]}
        return None

    def check_obj(self, obj):
        """То же для аргументов вызова или записи из базы: проверяем все строки внутри."""
        return self.check(" \n ".join(_strings(obj)))


def _parse(cfg):
    """Содержимое файла -> (ответ по умолчанию, группы). Не тот тип где-то внутри — TypeError с понятным текстом."""
    if not isinstance(cfg, dict) or not isinstance(cfg.get("groups", []), list):
        raise TypeError("нужен объект с полем groups — списком групп в [ ]")
    answer = cfg.get("answer", DEFAULT_ANSWER)
    if not isinstance(answer, str):
        raise TypeError("answer должен быть текстом в кавычках")
    groups = []
    for g in cfg.get("groups", []):
        if not isinstance(g, dict):
            raise TypeError("каждая группа — объект в { }")
        name, phrases = g.get("name", ""), g.get("phrases", [])
        if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
            raise TypeError(f"группа «{name}»: phrases должен быть списком фраз в кавычках, в [ ]")
        if not isinstance(g.get("answer", answer), str):
            raise TypeError(f"группа «{name}»: answer должен быть текстом в кавычках")
        pats = [(p, _phrase(p)) for p in phrases if _phrase(p)]
        groups.append({"name": str(name), "answer": g.get("answer", answer), "phrases": pats})
    return answer, groups


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _find(ws, pat):
    """Слова фразы идут в тексте по порядку, между соседними — не больше MAX_GAP чужих слов."""
    def rest(i, k):  # слово pat[k] ищем начиная с позиции i
        if k == len(pat):
            return True
        stop = len(ws) if k == 0 else min(len(ws), i + MAX_GAP + 1)
        return any(pat[k].match(ws[j]) and rest(j + 1, k + 1) for j in range(i, stop))
    return rest(0, 0)
