"""Личные записи студента (таблица notes): заметка — просто текст, напоминание — текст и время, напоминание о Д/З —
накануне ближайшей пары по предмету. Все они — один список с номерами (list_notes), из него и удаляют (delete_items).
Напоминание удаляется, когда сработало; заметка лежит, пока её не удалят."""
import datetime as dt
from campus import SUBJECTS, resolve_subject
from conversation import Rec
from dates import fmt, remind_time

def subject_name(code):
    return SUBJECTS.get(code, {}).get("name", code)


def reminder_line(r):
    """Строка сработавшего напоминания (из due_reminders) — для чата и Telegram."""
    if r.get("subject"):
        c = dt.datetime.fromisoformat(r["class_at"])
        what = f"Д/З к паре «{r['subject']}» ({c:%d.%m} в {c:%H:%M})"
        return f"⏰ Напоминаю: {what}" + (f": {r['text']}" if r["text"] else "")
    return f"⏰ Напоминаю: {r['text']}"


LIMITS = {"note": 10, "homework": 10}  # заметки с напоминаниями вместе и отдельно напоминания о Д/З
HOMEWORK_EVENING = dt.time(19, 0)  # накануне пары напоминаем в 19:00, а если уже поздно — за час до пары


class Records:
    """Нужны self.now, self.store, self.account_id, self.conv, self.limits и next_class_of из Campus."""

    def _login(self):
        return None if self.store is not None and self.account_id is not None else {"error": "not_logged_in"}

    def _recs(self, kind=None):
        """Записи в порядке списка: напоминания по времени, потом заметки по порядку записи.
        kind: note — только заметки, reminder — только напоминания (и о Д/З)."""
        now, timed, notes = fmt(self.now), [], []
        for r in self.store.notes(self.account_id):
            if r["fire_at"] is None:
                notes.append(Rec(r["id"], {"text": r["text"]}, r["text"], None))
            elif r["fire_at"] > now:
                o = {"text": r["text"]} if r["text"] else {}
                if r["subject"]:
                    o |= {"subject": subject_name(r["subject"]), "class_at": r["class_at"]}
                timed.append(Rec(r["id"], o | {"fire_at": r["fire_at"]}, r["text"], r["subject"]))
        timed.sort(key=lambda r: (r.show["fire_at"], r.id))
        return {"note": notes, "reminder": timed}.get(kind, timed + notes)

    def _notes_full(self, homework):
        """Ошибка limit со списком, если записей этого вида уже столько, сколько можно."""
        recs = [r for r in self._recs() if bool(r.code) == homework]
        cap = self.limits["homework" if homework else "note"]
        if len(recs) >= cap:
            return {"error": "limit", "limit": cap, "items": self.conv.numbered("notes", recs, ask=True)}
        return None

    def add_note(self, text, in_minutes=None, in_hours=None, in_days=None, at=None):
        """Заметка, а если указано время — напоминание."""
        err = self._login()
        if err:
            return err
        if not isinstance(text, str) or not text.strip():
            return {"error": "empty_text"}
        fire, err = remind_time(self.now, in_minutes, in_hours, in_days, at)
        if err:
            return {"error": err}
        full = self._notes_full(homework=False)
        if full:
            return full
        self.store.add_note(self.account_id, text.strip(), fmt(fire) if fire else None, created=fmt(self.now))
        return {"ok": True, "fire_at": fmt(fire)} if fire else {"ok": True}

    def add_homework_reminder(self, subject, task=None):
        """Напомнить о Д/З накануне ближайшей пары по предмету (если до неё уже поздно — о следующей)."""
        err = self._login()
        if err:
            return err
        code = resolve_subject(subject)
        if not code:
            return {"error": "subject_not_found", "subject": subject}
        name, after = SUBJECTS[code]["name"], self.now
        while True:
            cls = self.next_class_of(code, after)
            if cls is None:
                return {"error": "no_class", "subject": name}
            fire = dt.datetime.combine(cls.date() - dt.timedelta(days=1), HOMEWORK_EVENING)
            if fire <= self.now:
                fire = cls - dt.timedelta(hours=1)
            if fire > self.now:
                break
            after = cls  # пара уже через час или идёт: напомним к следующей
        full = self._notes_full(homework=True)
        if full:
            return full
        task = task.strip() if isinstance(task, str) else ""
        self.store.add_note(self.account_id, task, fmt(fire), subject=code, class_at=fmt(cls), created=fmt(self.now))
        return {"ok": True, "subject": name, "class_at": fmt(cls), "fire_at": fmt(fire)}

    def list_notes(self, kind=None):
        """Записи с номерами (по ним их удаляют) и объявления старосты (их удаляет только староста)."""
        err = self._login()
        if err:
            return err
        out = {"items": self.conv.numbered("notes", self._recs(kind))}
        news = self.starosta_news() if kind is None else []
        return out | ({"from_starosta": news} if news else {})

    def delete_items(self, numbers=None, query=None, all=None, kind=None):
        err = self._login()
        if err:
            return err
        return self.conv.delete("notes", self._recs(kind), numbers, query, all,
                                lambda chosen: self.store.delete_notes(self.account_id, [r.id for r in chosen]), kind)

    def due_reminders(self, delete=True):
        """Напоминания, время которых пришло: [{"text", "fire_at", ("subject", "class_at")}]. delete=True — и удалить
        их (chat.py показывает их и удаляет; с --now база не меняется)."""
        if self._login():
            return []
        rows = [r for r in self.store.notes(self.account_id) if r["fire_at"] and r["fire_at"] <= fmt(self.now)]
        if delete:
            self.store.delete_notes(self.account_id, [r["id"] for r in rows])
        out = []
        for r in sorted(rows, key=lambda r: r["fire_at"]):
            o = {"id": r["id"], "text": r["text"], "fire_at": r["fire_at"]}
            if r["subject"]:
                o |= {"subject": subject_name(r["subject"]), "class_at": r["class_at"]}
            out.append(o)
        return out
