"""Планировщик и Telegram-бот: Ишка сам пишет студенту, даже когда чат закрыт.
Раз в минуту смотрит в базу (db/ishka.sqlite) и отправляет в Telegram тем, кто подключил бота:
    ⏰ напоминания, время которых пришло (и напоминания о Д/З), — после отправки удаляет их, как и чат;
    📢 новые объявления и Д/З от старосты — каждое один раз (кому отправлено, помнит таблица post_sent);
    ☀️ утренний брифинг в 7:30 — пары, объявления старосты и напоминания (если чат уже показал его сегодня — не шлёт).
Студент подключается сам: находит бота в Telegram и пишет /login логин пароль (сообщение с паролем бот удаляет).
/logout — отключить. Пока бот не подключён, напоминания по-прежнему показывает chat.py.

Токен бота (от @BotFather) в репозиторий и файлы проекта не кладём: notify.py спрашивает его при запуске
(ввод не виден) или берёт из переменной окружения TELEGRAM_BOT_TOKEN (для сервера).
    python notify.py                       # работает, пока не остановишь (Ctrl+C)
    python notify.py --once --dry-run      # один проход: что отправил бы сейчас, без Telegram и без изменений в базе
    python notify.py --once --now "2026-10-16 07:30" --login ivanov   # проверка: прислать ivanov то, что пришло бы в этот момент
--db — другая база (как у chat.py и accounts.py), --now — подменить время (только с --once; база тогда не меняется,
как у chat.py), --login — только этому аккаунту, --no-brief — без утреннего брифинга."""
import argparse, datetime as dt, getpass, http.client, json, os, re, sqlite3, sys, time, unicodedata
import urllib.error, urllib.parse, urllib.request
from backend import Backend
from dates import fmt
from records import reminder_line
from store import DEFAULT_DB, Store, login_key

API = os.environ.get("TELEGRAM_API", "https://api.telegram.org")  # для проверок можно подменить своим сервером
TICK = 60  # секунд между проверками базы
BRIEF_FROM, BRIEF_TO = dt.time(7, 30), dt.time(12, 0)  # как в chat.py: позже полудня утренний брифинг уже не нужен
MAX_LOGIN_FAILS, LOGIN_PAUSE = 5, dt.timedelta(minutes=10)  # подбор пароля: после 5 ошибок — пауза 10 минут
MAX_LEN = 4096  # Telegram не принимает сообщение длиннее (в единицах UTF-16): длинное режем на части по строкам
TOKEN_RE = re.compile(r"\d+:[A-Za-z0-9_-]{20,}")  # вид токена от @BotFather: 123456789:AA…
WD = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
HELP = ("Привет! Я Ишка 🙂 Буду присылать сюда напоминания, объявления и Д/З от старосты и утренний брифинг в 7:30.\n\n"
        "Подключиться: /login логин пароль (сообщение с паролем я сразу удалю)\n"
        "Отключиться: /logout\n\n"
        "Спросить что-нибудь — пока в чате Ишки (chat.py), здесь я только присылаю уведомления.")


class TelegramError(Exception):
    def __init__(self, code, text):
        super().__init__(f"{code}: {text}")
        self.code = code


class Telegram:
    def __init__(self, token):
        self.token, self.base = token, f"{API.rstrip('/')}/bot{token}/"

    def call(self, method, http_timeout=30, **params):
        data = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}).encode()
        try:
            with urllib.request.urlopen(self.base + method, data, timeout=http_timeout) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            try:
                out = json.load(e)
            except (ValueError, OSError, http.client.HTTPException):
                raise TelegramError(e.code, e.reason)
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
            # в тексте других ошибок бывает адрес запроса, а в нём токен: печатаем только причину
            why = str(e.reason) if isinstance(e, urllib.error.URLError) else type(e).__name__
            raise TelegramError(0, f"нет связи с Telegram ({why.replace(self.token, '<токен>')})")
        if not isinstance(out, dict) or not out.get("ok"):
            out = out if isinstance(out, dict) else {}
            raise TelegramError(out.get("error_code", 0), out.get("description", "непонятный ответ"))
        return out.get("result")

    def send(self, chat_id, text):
        for part in split_text(text):
            self.call("sendMessage", chat_id=chat_id, text=part)


def _u16(s):
    return len(s.encode("utf-16-le")) // 2


def split_text(text, limit=MAX_LEN):
    """Части не длиннее limit (как считает Telegram), по возможности по границам строк."""
    parts, cur = [], ""
    for line in text.split("\n"):
        while _u16(line) > limit:  # одна строка длиннее лимита — режем по символам
            n = limit
            while _u16(line[:n]) > limit:
                n -= 1
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:n])
            line = line[n:]
        if cur and _u16(cur) + 1 + _u16(line) > limit:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur or not parts:
        parts.append(cur)
    return [p for p in parts if p.strip()] or [text[:1] or " "]


class DryRun:
    """--dry-run: печатать вместо отправки."""
    def send(self, chat_id, text):
        print(f"→ {chat_id}: {text}\n")


def when(s):
    """'2026-10-17 10:00' → 'сб 17.10 в 10:00', '2026-10-17' → 'сб 17.10'."""
    d = dt.datetime.fromisoformat(s if " " in s else s + " 00:00")
    return f"{WD[d.weekday()]} {d:%d.%m}" + (f" в {d:%H:%M}" if " " in s else "")


def post_text(p, due=None):
    """Объявление или Д/З от старосты одним сообщением."""
    if p["kind"] == "homework":
        return f"📚 Староста задал Д/З по предмету «{p['subject']}»: {p['task']}" + (f" (к {when(due or p['due'])})")
    return f"📢 Объявление от старосты: {p['text']}" + (f" ({when(p['at'])})" if p.get("at") else "")


def brief_text(s):
    """Утренний брифинг из get_daily_brief без модели. None — сказать нечего (нет пар, объявлений и напоминаний)."""
    lines = [f"☀️ Доброе утро! Сегодня {WD[dt.date.fromisoformat(s['date']).weekday()]} "
             f"{dt.date.fromisoformat(s['date']):%d.%m}, {s['week']} неделя."]
    if s.get("holiday"):
        lines.append("Нерабочий день, пар нет 🎉")
    elif s["lessons"]:
        for l in s["lessons"]:
            where = f" — {l['building']}, ауд. {l['room']}" if l.get("building") else ""
            sub = f" (подгруппа {l['subgroup']})" if l.get("subgroup") else ""
            walk = f" 🚶 ~{l['walk_from_prev_minutes']} мин от прошлой" if l.get("walk_from_prev_minutes") else ""
            lines.append(f"• {l['time'][:5]} {l['subject']}, {l['type']}{sub}{where}{walk}")
    else:
        lines.append("Пар нет 🎉")
    if s.get("from_starosta"):
        lines += ["", "📢 От старосты:"] + [
            f"• Д/З по предмету «{x['subject']}»: {x['task']} (к {when(x['due'])})" if x["kind"] == "homework"
            else f"• {x['text']}" + (f" ({when(x['at'])})" if x.get("at") else "") for x in s["from_starosta"]]
    if s.get("reminders"):
        lines += ["", "⏰ Напоминания:"] + [
            f"• {when(r['fire_at'])} — " + (f"Д/З к паре «{r['subject']}»" + (f": {r['text']}" if r.get("text") else "")
                                         if r.get("subject") else r["text"]) for r in s["reminders"]]
    if not s["lessons"] and not s.get("from_starosta") and not s.get("reminders"):
        return None
    return "\n".join(lines)


unsaved = set()  # (аккаунт, что, ключ): отправлено, но база не дала это отметить — снова не шлём, отметку пробуем позже


def _mark(store, account_id, what, key, now_s):
    if what == "note":
        store.delete_notes(account_id, [key])
    elif what == "post":
        store.mark_post_sent(key, account_id, now_s)
    else:
        store.set_brief_date(account_id, key)


def _log(now, text):
    print(f"[{fmt(now)}] {text}", flush=True)


def tick(store, tg, now, brief=True, save=True, only=None):
    """Один проход по базе: всё, что пора отправить подключённым студентам. Возвращает число отправленных сообщений.
    save=False — базу не менять (--dry-run и --now): отправленное не отмечается и пришлёт снова. only — один логин."""
    sent = 0
    for acc in store.telegram_accounts():
        if only and acc["login"] != only:
            continue
        try:
            sent += _tick_account(store, tg, now, acc, brief, save)
        except Exception as e:  # один аккаунт с ошибкой не мешает остальным
            store.db.rollback()  # иначе незакрытая запись держала бы базу и chat.py не смог бы в неё писать
            print(f"[{fmt(now)}] {acc['login']}: ошибка ({e!r}), попробую через минуту", file=sys.stderr, flush=True)
    return sent


def _tick_account(store, tg, now, acc, brief, save):
    now_s, today = fmt(now), now.date().isoformat()
    chat, b = acc["tg_chat_id"], Backend(now, store.profile(acc), store, acc["id"])
    if save:
        for item in [x for x in unsaved if x[0] == acc["id"]]:
            _mark(store, *item, now_s)  # ошибка базы — снова выйдем отсюда, а отправленное так и не повторится
            unsaved.discard(item)
    out = []  # (текст или None, что отметить после отправки)
    for r in b.due_reminders(delete=False):
        out.append((reminder_line(r), ("note", r["id"])))
    done = store.sent_posts(acc["id"])
    for p in b._posts():
        if p["id"] in done or p["author_id"] == acc["id"] or p["created"] < (acc["tg_since"] or ""):
            continue
        due = None
        if p["kind"] == "homework":
            due = b._post_due(p)
            if (due <= now_s) if " " in due else (due < today):  # срок у этого студента уже прошёл (как в брифинге)
                continue
        out.append((post_text(p, due), ("post", p["id"])))
    if brief and BRIEF_FROM <= now.time() < BRIEF_TO and acc["brief_date"] != today:
        s = b.get_daily_brief("today")
        out.append((None if "error" in s else brief_text(s), ("brief", today)))
    sent = 0
    for text, (what, key) in out:
        if save and (acc["id"], what, key) in unsaved:
            continue
        if text is not None:
            try:
                tg.send(chat, text)
                sent += 1
            except TelegramError as e:
                if e.code == 403 or (e.code == 400 and "chat not found" in str(e)):
                    # студент заблокировал бота или удалил чат: больше не пишем, пока снова не подключится
                    _log(now, f"{acc['login']}: бот заблокирован, отключаю Telegram")
                    if save:
                        store.unlink_telegram(chat)
                    break
                if e.code != 400:  # нет связи, Telegram просит подождать и т. п.: попробуем через минуту
                    _log(now, f"{acc['login']}: не отправилось ({e}), попробую через минуту")
                    break
                # 400 — Telegram не примет это сообщение и потом: пропускаем его, чтобы не мешало остальным
                _log(now, f"{acc['login']}: Telegram не принял сообщение ({e}), пропускаю его")
        if not save:
            continue
        try:
            _mark(store, acc["id"], what, key, now_s)
        except sqlite3.Error as e:
            store.db.rollback()
            unsaved.add((acc["id"], what, key))
            _log(now, f"{acc['login']}: не записал в базу, что отправлено ({e}); повторно не пошлю")
    return sent


class Bot:
    """Команды в личке с ботом: /start, /help, /login логин пароль, /logout."""

    def __init__(self, store, tg, clock):
        self.store, self.tg, self.clock = store, tg, clock
        self.fails = {}  # chat_id → (сколько ошибок подряд, когда последняя)

    def reply(self, chat, text):
        try:
            self.tg.send(chat, text)
        except TelegramError as e:
            print(f"не ответил в чат {chat}: {e}", flush=True)

    def handle(self, update):
        m = update.get("message")
        if not m or "text" not in m:
            return
        chat, text = m["chat"]["id"], m["text"].strip()
        if m["chat"].get("type") != "private":
            return  # в группах не работаем: пароль увидели бы все
        cmd, rest = (text.split(None, 1) + [""])[:2]
        cmd = cmd.split("@")[0].lower()
        if cmd == "/login":
            self.login(chat, m["message_id"], rest.split(None, 1))
        elif cmd in ("/logout", "/stop"):
            login = self.store.unlink_telegram(chat)
            self.reply(chat, f"Отключил аккаунт {login}: больше ничего не присылаю. Снова — /login логин пароль."
                       if login else "Этот чат и так не подключён. Подключиться: /login логин пароль")
        else:
            self.reply(chat, HELP)

    def login(self, chat, message_id, args):
        try:  # сообщение с паролем не оставляем в истории
            self.tg.call("deleteMessage", chat_id=chat, message_id=message_id)
        except TelegramError:
            pass
        now = self.clock()
        n, last = self.fails.get(chat, (0, now))
        if n >= MAX_LOGIN_FAILS and now - last < LOGIN_PAUSE:
            return self.reply(chat, "Слишком много неудачных попыток. Попробуй через 10 минут.")
        if len(args) != 2:  # пароль — всё после логина, с пробелами
            return self.reply(chat, "Напиши так: /login логин пароль")
        acc = self.store.check_login(unicodedata.normalize("NFC", args[0]), args[1])
        if not acc:
            self.fails[chat] = (n + 1 if now - last < LOGIN_PAUSE else 1, now)
            return self.reply(chat, "Неверный логин или пароль")
        self.fails.pop(chat, None)
        old = acc.get("tg_chat_id")
        self.store.link_telegram(acc["id"], chat, fmt(now))
        self.reply(chat, f"Готово, {acc['name'] or acc['login']}! Теперь напоминания, объявления старосты и брифинг "
                         "в 7:30 буду присылать сюда. Отключить — /logout")
        if old and old != chat:  # аккаунт был подключён к другому Telegram: там должны узнать, что уведомления ушли
            self.reply(old, f"Аккаунт {acc['login']} подключили к другому Telegram, сюда больше ничего не присылаю. "
                            "Если это был не ты, попроси старосту или администратора сменить тебе пароль.")


def get_token():
    """Токен из TELEGRAM_BOT_TOKEN, а если её нет — спросить (ввод не виден и не попадает в историю Терминала)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token and sys.stdin.isatty():
        try:
            token = getpass.getpass("Токен бота от @BotFather (вставь и нажми Enter, символы не видны): ").strip()
        except EOFError:
            token = ""
    if not token:
        raise SystemExit("Нет токена бота: запусти в Терминале и вставь токен, когда спросит, "
                         "или задай переменную TELEGRAM_BOT_TOKEN")
    if not TOKEN_RE.fullmatch(token):
        raise SystemExit("Это не похоже на токен от @BotFather (вид 123456789:AA…): скопируй его целиком, без пробелов и кавычек")
    return token


def main():
    ap = argparse.ArgumentParser(description="Планировщик и Telegram-бот Ишки",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--db", help="база аккаунтов (по умолчанию db/ishka.sqlite)")
    ap.add_argument("--once", action="store_true", help="один проход по базе и выйти (без команд бота)")
    ap.add_argument("--dry-run", action="store_true", help="печатать вместо отправки, базу не менять (с --once)")
    ap.add_argument("--now", help="подменить время: '2026-10-16 07:30' (только с --once, база не меняется)")
    ap.add_argument("--login", help="только этому аккаунту (с --now обязательно, если не --dry-run)")
    ap.add_argument("--no-brief", action="store_true", help="не присылать утренний брифинг")
    a = ap.parse_args()
    if (a.dry_run or a.now or a.login) and not a.once:
        ap.error("--dry-run, --now и --login работают только с --once")
    if a.now and not a.dry_run and not a.login:
        ap.error("с --now бот прислал бы проверочные сообщения всем подключённым: добавь --login СВОЙ_ЛОГИН (или --dry-run)")
    path = a.db or os.environ.get("ISHKA_DB") or str(DEFAULT_DB)
    if not os.path.exists(path):
        raise SystemExit(f"Базы аккаунтов {path} нет: сначала заведи аккаунты (python accounts.py add ...)")
    try:
        fake = dt.datetime.fromisoformat(a.now) if a.now else None
    except ValueError:
        ap.error("--now пишется так: \"2026-10-16 07:30\"")
    store = Store(path)
    only = login_key(a.login) if a.login else None
    linked = [x["login"] for x in store.telegram_accounts()]
    if only and only not in linked:
        raise SystemExit(f"Аккаунт {a.login} не подключён к боту: напиши боту в Telegram /login {a.login} пароль")
    if not linked:
        print("Бота пока никто не подключил (/login логин пароль в Telegram): отправлять некому.", flush=True)

    def clock():
        return fake or dt.datetime.now().replace(second=0, microsecond=0)

    if a.dry_run:
        n = tick(store, DryRun(), clock(), brief=not a.no_brief, save=False, only=only)
        print(f"Отправил бы сообщений: {n}")
        return
    try:  # бот должен отмечать отправленное, иначе слал бы одно и то же каждую минуту
        v = store.db.execute("PRAGMA user_version").fetchone()[0]
        store.db.execute(f"PRAGMA user_version = {int(v)}")
        store.db.commit()
    except sqlite3.Error as e:
        raise SystemExit(f"База {path} занята другой программой ({e}): запусти бота ещё раз" if "locked" in str(e) else
                         f"В базу {path} нельзя записывать ({e}): запусти бота от того же пользователя, что и chat.py")
    tg = Telegram(get_token())
    try:
        me = tg.call("getMe")
    except TelegramError as e:
        raise SystemExit(f"Нет связи с Telegram ({e}): проверь интернет" if e.code == 0 else
                         f"Telegram не принял токен ({e}): возьми актуальный у @BotFather")
    if a.once:
        print(f"Отправлено сообщений: {tick(store, tg, clock(), brief=not a.no_brief, save=not fake, only=only)}")
        return
    print(f"Бот @{me.get('username')} работает: раз в минуту смотрю базу {path}. Остановить — Ctrl+C", flush=True)
    bot, offset, next_tick = Bot(store, tg, clock), None, time.monotonic()
    while True:
        if time.monotonic() >= next_tick:
            try:
                n = tick(store, tg, clock(), brief=not a.no_brief)
                if n:
                    print(f"[{fmt(clock())}] отправлено: {n}", flush=True)
            except Exception as e:  # база занята и т. п.: следующий проход через минуту
                store.db.rollback()
                print(f"[{fmt(clock())}] ошибка прохода: {e!r}", file=sys.stderr, flush=True)
            next_tick += TICK
            if next_tick < time.monotonic():  # компьютер спал: не догоняем пропущенные минуты
                next_tick = time.monotonic() + TICK
        wait = max(1, min(50, int(next_tick - time.monotonic())))
        try:
            updates = tg.call("getUpdates", http_timeout=wait + 10, offset=offset, timeout=wait)
        except TelegramError as e:
            if e.code == 401:
                raise SystemExit("Telegram больше не принимает токен (его заменили через /revoke?): "
                                 "запусти python notify.py снова с новым токеном")
            hint = " (бот запущен ещё где-то: оставь один)" if e.code == 409 else ""
            print(f"[{fmt(clock())}] getUpdates: {e}{hint}", file=sys.stderr, flush=True)
            time.sleep(min(10, wait))
            continue
        except Exception as e:  # что-то совсем неожиданное: не падаем, через несколько секунд пробуем снова
            print(f"[{fmt(clock())}] getUpdates: {type(e).__name__}", file=sys.stderr, flush=True)
            time.sleep(min(10, wait))
            continue
        for u in updates if isinstance(updates, list) else []:
            try:
                offset = int(u["update_id"]) + 1
                bot.handle(u)
            except Exception as e:
                store.db.rollback()
                print(f"[{fmt(clock())}] сообщение {u.get('update_id') if isinstance(u, dict) else '?'}: {e!r}",
                      file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлен")
