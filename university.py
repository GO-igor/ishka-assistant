"""Справка о ТПУ и ИШИТР (db/university.json): сам вуз, школа, общежития, стипендии, студенческая жизнь, спорт
и здоровье, документы, контакты, поступление. Файл заполняется руками: поле со значением null — ещё не внесли,
Ишка так и скажет (not_filled в ответе инструмента), а куда обратиться — подскажет поле ask раздела.
Файл перечитывается, когда его поменяли, поэтому правки видны со следующего вопроса, даже не закрывая чат."""
import json, pathlib, sys
from guard import Guard

PATH = pathlib.Path(__file__).parent / "db" / "university.json"
UNIVERSITY = json.loads(PATH.read_text(encoding="utf-8"))
TPU_TOPICS = [k for k in UNIVERSITY if not k.startswith("_")]
GUARD = Guard()
_loaded = {"mtime": PATH.stat().st_mtime, "data": UNIVERSITY}


def current():
    """Справка из файла; файл поменяли — перечитываем, сломали (ошибка в JSON) — остаёмся на прошлой версии."""
    try:
        mtime = PATH.stat().st_mtime
        if mtime != _loaded["mtime"]:
            _loaded["data"], _loaded["mtime"] = json.loads(PATH.read_text(encoding="utf-8")), mtime
    except (OSError, ValueError) as e:
        print(f"db/university.json не читается ({e}), отвечаю по прошлой версии", file=sys.stderr)
    return _loaded["data"]


def empty(v):
    return v is None or (isinstance(v, (str, list)) and not v)


class University:
    university = None  # генератор датасета подставляет копию, где часть полей «ещё не внесли»

    def get_tpu_info(self, topic):
        """Раздел справки целиком: заполненные поля — в info (как в файле, по порядку), пустые — списком not_filled."""
        db = self.university if self.university is not None else current()
        t = db.get(topic)
        if not isinstance(t, dict) or topic.startswith("_"):
            return {"error": "unknown_topic", "topics": [k for k in db if not k.startswith("_")]}
        fields = t.get("info") or {}
        out = {"topic": t.get("title", topic),
               "info": {k: v for k, v in fields.items() if not empty(v) and not GUARD.check_obj(v)}}
        missing = [k for k, v in fields.items() if empty(v)]
        if missing:
            out["not_filled"] = missing
        if t.get("ask"):
            out["ask"] = t["ask"]
        return out
