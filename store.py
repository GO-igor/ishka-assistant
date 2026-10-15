"""Аккаунты, личные заметки и напоминания, объявления старосты — в SQLite (db/ishka.sqlite).
Регистрации нет: аккаунты заводятся руками через accounts.py, пароли хранятся только в виде хеша.
Роли: student и starosta. Староста отправляет своей группе (или подгруппе) объявления и Д/З; студенты видят их
в брифинге и в списке записей.
Store(":memory:") — база в памяти (генератор датасета, тесты, вход без аккаунта — guest()). Сайт может пользоваться
этим же классом."""
import datetime as dt, hashlib, hmac, json, os, pathlib, secrets, sqlite3, unicodedata

DEFAULT_DB = pathlib.Path(__file__).parent / "db" / "ishka.sqlite"
ROLES = ("student", "starosta")
KINDS = ("announcement", "homework")
ITERATIONS = 600_000  # PBKDF2-SHA256, рекомендация OWASP

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY,
    login TEXT NOT NULL UNIQUE,             -- всегда в нижнем регистре (login_key)
    pw_hash TEXT NOT NULL,                  -- '!' — вход закрыт
    role TEXT NOT NULL DEFAULT 'student' CHECK (role IN ('student', 'starosta')),
    name TEXT,
    grp TEXT NOT NULL,                      -- учебная группа, например 8К51
    subgroup INTEGER,
    english_teacher TEXT,
    adaptation INTEGER NOT NULL DEFAULT 0,
    brief_date TEXT,                        -- в какой день уже показан утренний брифинг
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (          -- личные заметки и напоминания
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    fire_at TEXT,                           -- 'YYYY-MM-DD HH:MM': когда напомнить; NULL — просто заметка
    subject TEXT,                           -- код предмета: напоминание о Д/З накануне пары
    class_at TEXT,                          -- начало этой пары
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_account ON notes (account_id);
CREATE TABLE IF NOT EXISTS group_posts (    -- объявления и Д/З от старосты
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id INTEGER REFERENCES accounts(id) ON DELETE SET NULL,
    grp TEXT NOT NULL,
    subgroup INTEGER,                       -- NULL — всей группе
    kind TEXT NOT NULL CHECK (kind IN ('announcement', 'homework')),
    data TEXT NOT NULL,                     -- JSON: text и at (когда событие) или subject, task, due
    until TEXT NOT NULL,                    -- 'YYYY-MM-DD HH:MM': после этого момента удаляется
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS group_posts_grp ON group_posts (grp, until);
"""


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, salt, h = stored.split("$")
        if algo != "pbkdf2_sha256" or int(n) <= 0:
            return False
        got = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(n))
    except ValueError:
        return False  # '!' и прочее — вход закрыт
    return hmac.compare_digest(got.hex(), h)


_DUMMY = hash_password(secrets.token_hex(8))  # чтобы неверный логин проверялся так же долго, как неверный пароль


def login_key(login) -> str:
    """Логин без учёта регистра (и кириллицы тоже), пробелов по краям и способа набора «ё»/«й»."""
    return unicodedata.normalize("NFC", str(login or "")).strip().lower()


def group_key(group) -> str:
    """Название группы как в расписании: 8к51, 8K51 (латинская K) → 8К51."""
    g = unicodedata.normalize("NFC", str(group or "")).strip().upper()
    return g.translate(str.maketrans("ABCEHKMOPTX", "АВСЕНКМОРТХ"))


def _subgroup(subgroup):
    if subgroup is not None and subgroup not in (1, 2, 3):
        raise ValueError("подгруппа должна быть 1, 2 или 3")
    return subgroup


def new_password(n=10) -> str:
    """Случайный пароль без похожих символов (0/O, 1/l/I)."""
    abc = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(abc) for _ in range(n))


GUEST_NOTICE = ("⚠️ Ты не вошёл в аккаунт, поэтому Ишка не знает твою группу и преподавателей: расписание покажет "
                "сразу для всех подгрупп и без твоего преподавателя английского, объявлений старосты не будет. "
                "Заметки и напоминания пропадут, когда закроешь чат.")


def guest():
    """Без входа: база в памяти с одним аккаунтом без профиля -> (store, аккаунт). Всё записанное — до конца разговора."""
    st = Store(":memory:")
    acc = st.account(account_id=st.add_account("гость", None, "гость"))
    acc.pop("pw_hash")
    return st, acc | {"guest": True}  # profile() пометит его: в промпте «не вошёл в аккаунт»


def _now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


class Store:
    def __init__(self, path=None):
        path = str(path or os.environ.get("ISHKA_DB") or DEFAULT_DB)
        if path != ":memory:":
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._migrate()
        self._normalize()

    def _migrate(self):
        """База до 14.10.2026: личные данные лежали одним JSON (user_data), у старосты были мероприятия и напоминания.
        Заметки и напоминания переносим в notes, мероприятия и напоминания старосты становятся объявлениями;
        Д/З, дедлайны и экзамены студента удаляются: этих функций больше нет."""
        tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(accounts)")}
        sql = self.db.execute("SELECT sql FROM sqlite_master WHERE name = 'group_posts'").fetchone()[0]
        if "user_data" not in tables and "brief_date" in cols and "'announcement'" in sql:
            return
        self.db.execute("PRAGMA foreign_keys = OFF")  # рассылка удалённого вручную аккаунта не должна мешать переносу
        try:
            self.db.execute("BEGIN")
            if "brief_date" not in cols:
                self.db.execute("ALTER TABLE accounts ADD COLUMN brief_date TEXT")
            if "'announcement'" not in sql:
                old = self.db.execute("SELECT * FROM group_posts").fetchall()
                self.db.execute("DROP TABLE group_posts")
                table = SCHEMA[SCHEMA.index("CREATE TABLE IF NOT EXISTS group_posts"):SCHEMA.index("CREATE INDEX IF NOT EXISTS group_posts")]
                self.db.execute(table)
                self.db.execute("CREATE INDEX IF NOT EXISTS group_posts_grp ON group_posts (grp, until)")
                for r in old:
                    kind, data = r["kind"], json.loads(r["data"])
                    if kind == "event":
                        kind, data = "announcement", {"text": data["title"] + (f", {data['place']}" if data.get("place") else ""),
                                                      "at": data["date"] + (f" {data['time']}" if data.get("time") else "")}
                    elif kind == "reminder":
                        kind, data = "announcement", {"text": data["text"], "at": data["fire_at"]}
                    self.db.execute("INSERT INTO group_posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                    (r["id"], r["author_id"], r["grp"], r["subgroup"], kind,
                                     json.dumps(data, ensure_ascii=False), r["until"], r["created"]))
            if "user_data" in tables:
                for r in self.db.execute("SELECT account_id, data, updated FROM user_data").fetchall():
                    data = json.loads(r["data"])
                    for t in data.get("notes") or []:
                        self.db.execute("INSERT INTO notes (account_id, text, created) VALUES (?, ?, ?)",
                                        (r["account_id"], str(t), r["updated"]))
                    for x in data.get("reminders") or []:
                        self.db.execute("INSERT INTO notes (account_id, text, fire_at, created) VALUES (?, ?, ?, ?)",
                                        (r["account_id"], str(x.get("text") or ""), x.get("fire_at"), r["updated"]))
                self.db.execute("DROP TABLE user_data")
            self.db.execute("COMMIT")
        except sqlite3.OperationalError as e:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            if "readonly" not in str(e) and "locked" not in str(e):
                raise
            # базу только читают (или её держит другой процесс): перенесём при следующем запуске
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        finally:
            self.db.execute("PRAGMA foreign_keys = ON")

    def _normalize(self):
        """База от первой версии: логины и группы хранились как введены (Петров, 8к51) — приводим к login_key/group_key."""
        for r in self.db.execute("SELECT id, login, grp FROM accounts").fetchall():
            if (r["login"], r["grp"]) != (login_key(r["login"]), group_key(r["grp"])):
                try:
                    self.db.execute("UPDATE accounts SET login = ?, grp = ? WHERE id = ?",
                                    (login_key(r["login"]), group_key(r["grp"]), r["id"]))
                except sqlite3.IntegrityError:  # «Петров» и «петров» — два аккаунта: логин оставляем как есть
                    self.db.execute("UPDATE accounts SET grp = ? WHERE id = ?", (group_key(r["grp"]), r["id"]))
        for r in self.db.execute("SELECT DISTINCT grp FROM group_posts").fetchall():
            if r["grp"] != group_key(r["grp"]):
                self.db.execute("UPDATE group_posts SET grp = ? WHERE grp = ?", (group_key(r["grp"]), r["grp"]))
        self.db.commit()

    # ---------- аккаунты ----------
    def add_account(self, login, password, group, role="student", name=None, subgroup=None,
                    english_teacher=None, adaptation=False):
        """password=None — аккаунт без входа (например, пока пароль не выдан)."""
        if role not in ROLES:
            raise ValueError(f"роль должна быть одной из {ROLES}")
        if not login_key(login):
            raise ValueError("пустой логин")
        if not group_key(group):
            raise ValueError("не указана группа")
        cur = self.db.execute(
            "INSERT INTO accounts (login, pw_hash, role, name, grp, subgroup, english_teacher, adaptation, created) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (login_key(login), hash_password(password) if password else "!", role, name, group_key(group),
             _subgroup(subgroup), english_teacher, int(bool(adaptation)), _now()))
        self.db.commit()
        return cur.lastrowid

    def account(self, login=None, account_id=None):
        row = self.db.execute("SELECT * FROM accounts WHERE " + ("id = ?" if account_id is not None else "login = ?"),
                              (account_id if account_id is not None else login_key(login),)).fetchone()
        return dict(row) if row else None

    def check_login(self, login, password):
        """Аккаунт (dict без хеша пароля) или None, если логина нет или пароль неверный."""
        acc = self.account(login=login)
        ok = verify_password(password or "", acc["pw_hash"] if acc and acc["pw_hash"] != "!" else _DUMMY)
        if not acc or acc["pw_hash"] == "!" or not ok:
            return None
        acc.pop("pw_hash")
        return acc

    def accounts(self):
        return [dict(r) for r in self.db.execute(
            "SELECT id, login, role, name, grp, subgroup, english_teacher, adaptation, pw_hash != '!' AS can_login "
            "FROM accounts ORDER BY grp, role = 'starosta' DESC, login")]

    def update_account(self, login, **fields):
        allowed = {"role", "name", "grp", "subgroup", "english_teacher", "adaptation"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"нельзя менять: {', '.join(sorted(bad))}")
        if "role" in fields and fields["role"] not in ROLES:
            raise ValueError(f"роль должна быть одной из {ROLES}")
        if not fields:
            return False
        if "grp" in fields:
            fields["grp"] = group_key(fields["grp"])
            if not fields["grp"]:
                raise ValueError("не указана группа")
        if "subgroup" in fields:
            _subgroup(fields["subgroup"])
        if "adaptation" in fields:
            fields["adaptation"] = int(bool(fields["adaptation"]))
        cur = self.db.execute(f"UPDATE accounts SET {', '.join(f'{k} = ?' for k in fields)} WHERE login = ?",
                              (*fields.values(), login_key(login)))
        self.db.commit()
        return cur.rowcount > 0

    def set_password(self, login, password):
        cur = self.db.execute("UPDATE accounts SET pw_hash = ? WHERE login = ?",
                              (hash_password(password) if password else "!", login_key(login)))
        self.db.commit()
        return cur.rowcount > 0

    def remove_account(self, login):
        cur = self.db.execute("DELETE FROM accounts WHERE login = ?", (login_key(login),))
        self.db.commit()
        return cur.rowcount > 0

    def profile(self, acc):
        """Профиль для Backend: подгруппа, преподаватель английского, курсы (А), группа и роль."""
        p = {"group": acc["grp"], "role": acc["role"]} | ({"guest": True} if acc.get("guest") else {})
        if acc.get("subgroup"):
            p["subgroup"] = acc["subgroup"]
        if acc.get("english_teacher"):
            p["english_teacher"] = acc["english_teacher"]
        if acc.get("adaptation"):
            p["adaptation"] = True
        return p

    def members(self, group, subgroup=None, exclude=None):
        """Сколько студентов получит объявление: вся группа или подгруппа (и те, у кого подгруппа не указана)."""
        q, args = "SELECT COUNT(*) FROM accounts WHERE grp = ?", [group_key(group)]
        if subgroup:
            q += " AND (subgroup = ? OR subgroup IS NULL)"
            args.append(subgroup)
        if exclude is not None:
            q += " AND id != ?"
            args.append(exclude)
        return self.db.execute(q, args).fetchone()[0]

    # ---------- объявления и Д/З старосты ----------
    def add_post(self, author_id, group, kind, data, until, subgroup=None, created=None):
        if kind not in KINDS:
            raise ValueError(kind)
        cur = self.db.execute(
            "INSERT INTO group_posts (author_id, grp, subgroup, kind, data, until, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (author_id, group_key(group), _subgroup(subgroup), kind, json.dumps(data, ensure_ascii=False), until,
             created or _now()))
        self.db.commit()
        return cur.lastrowid

    def purge_posts(self, now):
        """Удалить объявления и Д/З старосты, срок которых прошёл (until раньше now 'YYYY-MM-DD HH:MM')."""
        n = self.db.execute("DELETE FROM group_posts WHERE until < ?", (now,)).rowcount
        self.db.commit()
        return n

    def delete_posts(self, group, ids):
        """Староста удалил объявления: у всей группы. Чужие группы не трогаем. Возвращает число удалённых."""
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        n = self.db.execute(f"DELETE FROM group_posts WHERE grp = ? AND id IN ({', '.join('?' * len(ids))})",
                            (group_key(group), *ids)).rowcount
        self.db.commit()
        return n

    def posts(self, group, subgroup=None, after=None, kind=None):
        """Объявления и Д/З группе, которые ещё актуальны (until >= after), для студента из подгруппы subgroup:
        общие и своей подгруппы; если у студента подгруппа не указана — все."""
        q, args = "SELECT * FROM group_posts WHERE grp = ?", [group_key(group)]
        if after:
            q += " AND until >= ?"
            args.append(after)
        if subgroup:
            q += " AND (subgroup IS NULL OR subgroup = ?)"
            args.append(subgroup)
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        out = []
        for r in self.db.execute(q + " ORDER BY until, id", args):
            out.append({"id": r["id"], "kind": r["kind"], "subgroup": r["subgroup"], "author_id": r["author_id"],
                        "created": r["created"], **json.loads(r["data"])})
        return out

    # ---------- личные заметки и напоминания ----------
    def add_note(self, account_id, text, fire_at=None, subject=None, class_at=None, created=None):
        cur = self.db.execute("INSERT INTO notes (account_id, text, fire_at, subject, class_at, created) VALUES (?, ?, ?, ?, ?, ?)",
                              (account_id, text, fire_at, subject, class_at, created or _now()))
        self.db.commit()
        return cur.lastrowid

    def notes(self, account_id):
        return [dict(r) for r in self.db.execute("SELECT * FROM notes WHERE account_id = ? ORDER BY id", (account_id,))]

    def delete_notes(self, account_id, ids):
        """Удалить свои записи (чужие не трогаем). Возвращает число удалённых."""
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        n = self.db.execute(f"DELETE FROM notes WHERE account_id = ? AND id IN ({', '.join('?' * len(ids))})",
                            (account_id, *ids)).rowcount
        self.db.commit()
        return n

    def purge_notes(self, now):
        """Удалить напоминания всех студентов, время которых прошло (раньше now 'YYYY-MM-DD HH:MM'): сайт к этому
        моменту их уже отправил. Заметки без времени остаются. Возвращает число удалённых."""
        n = self.db.execute("DELETE FROM notes WHERE fire_at IS NOT NULL AND fire_at < ?", (now,)).rowcount
        self.db.commit()
        return n

    def set_brief_date(self, account_id, day):
        self.db.execute("UPDATE accounts SET brief_date = ? WHERE id = ?", (day, account_id))
        self.db.commit()
