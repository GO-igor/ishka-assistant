"""Объявления и Д/З от старосты (таблица group_posts). Староста отправляет их группе или подгруппе, студенты видят
их в брифинге и в списке записей. Удаляет их только староста (group_delete_post) — сразу у всей группы.
Объявление хранится до своего события (at), а без даты — неделю; Д/З — до срока сдачи."""
import datetime as dt
from campus import Campus, ENG_TEACHERS, GUARD, SUBJECTS, resolve_subject
from conversation import Rec
from dates import fmt, parse_at, parse_day

POST_LIMIT = 10  # сколько действующих объявлений и сколько Д/З может быть у группы
ANNOUNCE_DAYS = 7  # сколько хранится объявление без даты


class Reader(Campus):
    """Расписание другого студента: «к следующей паре» у подгрупп и групп английского в разное время."""

    def __init__(self, now, profile):
        self.now, self.profile = now, profile


def homework_due(c, code, name, due):
    """Срок Д/З по расписанию c: {"due": 'YYYY-MM-DD HH:MM' (начало пары) или 'YYYY-MM-DD', "no_class_that_day"}
    или ошибка. due — next_class или день («friday», «2026-10-20», «15 октября»)."""
    if due == "next_class":
        when = c.next_class_of(code) if code else None
        if when is None:
            return {"error": "next_class_not_found", "subject": name}
        return {"due": fmt(when), "no_class_that_day": False}
    try:
        d = parse_day(c.now, due, future=True)
    except (ValueError, OverflowError):
        return {"error": "bad_date"}
    if d < c.now.date():
        return {"error": "due_in_past", "subject": name}
    when = c.next_class_of(code, max(c.now, dt.datetime.combine(d, dt.time(0, 0)))) if code else None
    if when and when.date() == d:
        return {"due": fmt(when), "no_class_that_day": False}
    # пары по предмету в этот день нет — срок просто по дате
    return {"due": d.isoformat(), "no_class_that_day": bool(code) and not any(l["subject"] == code for l in c.lessons_on(d))}


def _blocked(p):
    return GUARD.check_obj({k: p.get(k) for k in ("subject", "task", "text")})


class Starosta:
    """Нужны self.now, self.profile (group, role, subgroup…), self.store, self.account_id, self.conv, self.limits."""

    def _posts(self):
        """Объявления и Д/З для группы и подгруппы студента, которые ещё не прошли (с запретной фразой — не показываем)."""
        if self.store is None or not self.profile.get("group"):
            return []
        posts = self.store.posts(self.profile["group"], self.profile.get("subgroup"), after=fmt(self.now))
        return [p for p in posts if not _blocked(p)]

    def _post_due(self, p):
        """Срок Д/З по расписанию этого студента: «к следующей паре» считаем от момента, когда староста его задал."""
        if not p.get("spec") or not p.get("code"):
            return p["due"]
        prof = dict(self.profile, subgroup=p["subgroup"]) if p.get("subgroup") else self.profile  # Д/З подгруппе
        return homework_due(Reader(dt.datetime.fromisoformat(p["created"]), prof), p["code"], p["subject"], p["spec"]).get("due", p["due"])

    def starosta_news(self):
        """Всё действующее от старосты по времени: объявления (text, at) и Д/З (subject, task, due)."""
        out, now = [], fmt(self.now)
        for p in self._posts():
            if p["kind"] == "homework":
                due = self._post_due(p)
                if due > now if " " in due else due >= now[:10]:
                    out.append(({"kind": "homework", "subject": p["subject"], "task": p["task"], "due": due}, due))
            else:
                o = {"kind": "announcement", "text": p["text"]} | ({"at": p["at"]} if p.get("at") else {})
                out.append((o, p.get("at") or p["created"]))
        return [o for o, _ in sorted(out, key=lambda x: x[1])]

    # ---------- инструменты старосты ----------
    def _starosta(self, subgroup=None):
        """None, если можно отправлять группе, иначе ошибка."""
        if self.profile.get("role") != "starosta":
            return {"error": "not_starosta"}
        if self.store is None or self.account_id is None or not self.profile.get("group"):
            return {"error": "not_logged_in"}
        if subgroup is not None and subgroup not in (1, 2, 3):
            return {"error": "bad_subgroup"}
        return None

    def _post(self, kind, data, until, subgroup):
        self.store.add_post(self.account_id, self.profile["group"], kind, data, until, subgroup=subgroup, created=fmt(self.now))
        out = {"recipients": self.store.members(self.profile["group"], subgroup, exclude=self.account_id)}
        return out | ({"subgroup": subgroup} if subgroup else {})

    def _group_recs(self, kind=None):
        """Действующие объявления и Д/З всей группы (всем подгруппам) для старосты."""
        out = []
        for p in self.store.posts(self.profile["group"], after=fmt(self.now), kind=kind):
            if _blocked(p):
                continue
            o = {} if kind else {"kind": p["kind"]}
            keys = ("text", "at") if p["kind"] == "announcement" else ("subject", "task", "due")
            o |= {k: p[k] for k in keys if p.get(k)} | ({"subgroup": p["subgroup"]} if p.get("subgroup") else {})
            out.append(Rec(p["id"], o, " ".join(str(p.get(k) or "") for k in ("text", "subject", "task")), p.get("code")))
        return out

    def _posts_full(self, kind):
        recs, cap = self._group_recs(kind), self.limits["posts"]
        if len(recs) >= cap:
            return {"error": "limit", "limit": cap, "items": self.conv.numbered("posts", recs, ask=True, kind=kind)}
        return None

    def group_announce(self, text, at=None, subgroup=None):
        """Объявление группе. at — когда событие (день и, если есть, время): после него объявление удаляется."""
        err = self._starosta(subgroup)
        if err:
            return err
        if not isinstance(text, str) or not text.strip():
            return {"error": "empty_text"}
        data = {"text": text.strip()}
        if at:
            try:
                when, timed = parse_at(self.now, at)
            except (ValueError, OverflowError):
                return {"error": "bad_date"}
            if when <= self.now:
                return {"error": "date_in_past"}
            data["at"] = fmt(when) if timed else when.date().isoformat()
            until = fmt(when)
        else:
            until = fmt(self.now + dt.timedelta(days=ANNOUNCE_DAYS))
        full = self._posts_full("announcement")
        if full:
            return full
        return {"ok": True} | data | {"until": until} | self._post("announcement", data, until, subgroup)

    def group_add_homework(self, subject, task, due="next_class", subgroup=None):
        err = self._starosta(subgroup)
        if err:
            return err
        code = resolve_subject(subject)
        name = SUBJECTS[code]["name"] if code else subject
        due = due or "next_class"
        # «к следующей паре» (и время пары в названный день) у каждого своё: лабы у подгрупп и английский у групп
        # преподавателей в разное время. Поэтому в базе хранится сам срок (spec), а время пары каждый студент
        # получает по своему расписанию (_post_due). Здесь — срок для того, кому задали, и самый поздний из всех.
        me = self if not subgroup or subgroup == self.profile.get("subgroup") else Reader(self.now, dict(self.profile, subgroup=subgroup))
        r = homework_due(me, code, name, due)
        if r.get("error") in ("bad_date", "due_in_past"):
            return r
        readers = [Reader(self.now, {"subgroup": s, "english_teacher": t, "adaptation": a})
                   for s in ([subgroup] if subgroup else [1, 2, 3]) for t in ENG_TEACHERS for a in (False, True)]
        # срок у каждого возможного получателя; due_differs — только среди тех, у кого этот предмет вообще есть
        # (курсы (А) — не у всех, у самого старосты предмета может не быть)
        alls = [(not code or x.next_class_of(code) is not None, homework_due(x, code, name, due)) for x in readers]
        alls = [(has, x) for has, x in alls if "error" not in x]
        outs = [x for has, x in alls if has]
        if "error" in r or (code and me.next_class_of(code) is None):
            if not outs:
                return r
            r = min(outs, key=lambda x: x["due"])
        dues = {x["due"] for x in outs} | {r["due"]}
        until = max(d if " " in d else d + " 23:59" for d in dues | {x["due"] for _, x in alls})
        full = self._posts_full("homework")
        if full:
            return full
        sent = self._post("homework", {"code": code, "subject": name, "task": task, "due": r["due"], "spec": due}, until, subgroup)
        out = {"ok": True, "subject": name, "due": r["due"]}
        if len(dues) > 1:
            out["due_differs"] = True  # у разных подгрупп (групп английского) пара в разное время
        elif r["no_class_that_day"]:
            out["no_class_that_day"] = True
        return out | sent

    def group_delete_post(self, kind=None, numbers=None, query=None, all=None):
        err = self._starosta()
        if err:
            return err
        return self.conv.delete("posts", self._group_recs(kind), numbers, query, all,
                                lambda chosen: self.store.delete_posts(self.profile["group"], [r.id for r in chosen]), kind)
