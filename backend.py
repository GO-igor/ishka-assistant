"""Инструменты Ишки. Backend собирает их из модулей и вызывает по имени (call):
    campus.py       — расписание, аудитории, преподаватели, дорога, перекус, мероприятия, сессия (db/*.json)
    records.py      — личные заметки и напоминания (SQLite, store.py)
    starosta.py     — объявления и Д/З от старосты
    conversation.py — списки с номерами и удаление по ним; состояние разговора
    dates.py        — разбор дат и времени
Генератор датасета вызывает те же инструменты, поэтому ответы инструментов в обучении настоящие."""
import json, pathlib, re
from campus import (Campus, BUILDINGS, ENG_TEACHERS, EVENTS, EVENT_KINDS, HOLIDAYS, LESSONS, SESSION, SUBJECTS, WD_RU,
                    resolve_building, resolve_subject, walk_minutes, week_parity)
from conversation import Conversation
from dates import WEEKDAYS, parse_day
from records import LIMITS, Records
from starosta import POST_LIMIT, Starosta

SPEC = {t["function"]["name"]: t["function"]["parameters"]
        for t in json.loads((pathlib.Path(__file__).parent / "tools.json").read_text(encoding="utf-8"))}
DELETE_TOOLS = {"delete_items", "group_delete_post"}  # только они отвечают на «да» к «удалить все?»


class Backend(Campus, Records, Starosta):
    def __init__(self, now, profile, store=None, account_id=None, conv=None):
        """profile: subgroup, english_teacher, adaptation, у вошедшего ещё group и role (student|starosta) — store.profile.
        Без store и account_id работают только справочные инструменты. conv — разговор (Conversation): сайт хранит
        его между запросами; по умолчанию новый."""
        self.now, self.profile = now, dict(profile or {})
        self.store, self.account_id = store, account_id
        self.conv = conv or Conversation()
        self.events, self.session = EVENTS, SESSION
        self.limits = dict(LIMITS, posts=POST_LIMIT)  # можно поменять для одного Backend

    def get_daily_brief(self, day="today"):
        """Пары с переходами между корпусами, всё от старосты и все напоминания (с номерами: их можно удалить)."""
        s = self.get_schedule(day)
        if "error" in s:
            return s
        ls = self.lessons_on(parse_day(self.now, day))
        for i in range(1, len(ls)):
            a, b = ls[i - 1], ls[i]
            if a["start"] != b["start"] and a["building"] and b["building"] and a["building"] != b["building"]:
                s["lessons"][i]["walk_from_prev_minutes"] = walk_minutes(a["building"], b["building"])
        news = self.starosta_news()
        if news:
            s["from_starosta"] = news
        if not self._login():
            reminders = self.conv.numbered("notes", self._recs("reminder"))
            if reminders:
                s["reminders"] = reminders
        return s

    def call(self, name, args):
        """Вызов инструмента по имени с проверкой аргументов по tools.json. Ошибка — {"error": ...}, без исключений."""
        if name not in SPEC:
            return {"error": "unknown_tool", "name": name}
        if not isinstance(args, dict):
            return {"error": "bad_args"}
        if name not in DELETE_TOOLS:
            self.conv.pending = {}  # «удалить все» подтверждается только следующим же вызовом удаления
        kw, err = check_args(SPEC[name], _clean(args))
        return err or getattr(self, name)(**kw)


def check_args(spec, args):
    """Аргументы по схеме: лишние и null отбрасываем, числа и строки приводим ("30" → 30, 15 → "15", ["ДР"] → "ДР").
    -> (аргументы, None) или (None, ошибка)."""
    props, out = spec.get("properties", {}), {}
    for k, v in args.items():
        p = props.get(k)
        if p is None or v is None:
            continue
        t = p.get("type")
        if t == "integer":
            if isinstance(v, str) and re.fullmatch(r"\s*-?[0-9]{1,9}\s*", v):
                v = int(v.strip())
            elif isinstance(v, float) and v.is_integer() and abs(v) < 1e9:
                v = int(v)
            if isinstance(v, bool) or not isinstance(v, int):
                return None, {"error": "bad_args", "arg": k}
        elif t == "string":
            if isinstance(v, list) and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v):
                v = " ".join(map(str, v))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                v = str(v)
            if not isinstance(v, str):
                return None, {"error": "bad_args", "arg": k}
        elif t == "boolean" and not isinstance(v, bool):
            v = str(v).strip().lower() in ("true", "1", "yes", "да")
        if "enum" in p and v not in p["enum"]:
            return None, {"error": "bad_value", "arg": k, "allowed": p["enum"]}
        out[k] = v
    missing = [k for k in spec.get("required", []) if k not in out]
    return (None, {"error": "missing_args", "missing": missing}) if missing else (out, None)


def _clean(v):
    """Одиночные суррогаты (\\udc00 из кривого JSON) — на «�»: иначе база не сохранит запись."""
    if isinstance(v, str):
        return v.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {_clean(k): _clean(x) for k, x in v.items()}
    return v
