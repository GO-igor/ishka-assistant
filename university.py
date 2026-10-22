"""Справка о ТПУ и ИШИТР (db/university.json): сам вуз, школа, общежития, стипендии, студенческая жизнь, спорт
и здоровье, документы, контакты, поступление. Файл заполняется руками: поле со значением null — ещё не внесли,
Ишка так и скажет (not_filled в ответе инструмента), а куда обратиться — подскажет поле ask раздела.
Файл перечитывается, когда его поменяли, поэтому правки видны со следующего вопроса, даже не закрывая чат.
Ошибка в файле не роняет Ишку: справка остаётся прошлой (при запуске — пустой), остальное работает как обычно."""
import json, pathlib, sys
from guard import Guard

PATH = pathlib.Path(__file__).parent / "db" / "university.json"
GUARD = Guard()
_loaded = {"mtime": None, "data": {}, "error": None}


def current():
    """Справка из файла; файл поменяли — перечитываем, сломали (ошибка в JSON или не тот вид) — остаёмся на прошлой версии."""
    try:
        mtime = PATH.stat().st_mtime
        if mtime != _loaded["mtime"]:
            _loaded["mtime"] = mtime  # сломанный файл не перечитываем на каждый вопрос, а ждём, пока его поправят
            data = json.loads(PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("нужен объект { \"раздел\": {...} } в фигурных скобках")
            _loaded["data"], _loaded["error"] = data, None
    except (OSError, ValueError) as e:
        if str(e) != _loaded["error"]:  # одна и та же ошибка — один раз
            print(f"db/university.json не читается ({e}), отвечаю по прошлой версии", file=sys.stderr)
            _loaded["error"] = str(e)
    return _loaded["data"]


current()  # при запуске: ошибку в файле видно сразу


def empty(v):
    return v is None or (isinstance(v, (str, list)) and not v)


class University:
    university = None  # генератор датасета подставляет копию, где часть полей «ещё не внесли»

    def get_tpu_info(self, topic):
        """Раздел справки целиком: заполненные поля — в info (как в файле, по порядку), пустые — списком not_filled.
        Раздела нет в файле или он не того вида — пустой раздел: Ишка скажет, что в базе пока ничего нет."""
        db = self.university if self.university is not None else current()
        if not isinstance(topic, str) or topic.startswith("_"):
            return {"error": "unknown_topic", "topics": [k for k in db if not k.startswith("_")]}
        t = db.get(topic)
        t = t if isinstance(t, dict) else {}
        fields = t.get("info") if isinstance(t.get("info"), dict) else {}
        title = t.get("title")
        out = {"topic": title if isinstance(title, str) and title and not GUARD.check(title) else topic, "info": {}}
        missing = []
        for k, v in fields.items():
            if GUARD.check(k):  # запретное слово в названии поля — поле не показываем вовсе
                continue
            if empty(v) or GUARD.check_obj(v):  # запретное слово в значении — как будто ещё не внесли
                missing.append(k)
            else:
                out["info"][k] = v
        if missing:
            out["not_filled"] = missing
        ask = t.get("ask")
        if isinstance(ask, str) and ask and not GUARD.check(ask):
            out["ask"] = ask
        return out
