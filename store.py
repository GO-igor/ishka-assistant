"""Аккаунты, рассылки старосты и личные данные студентов в SQLite (db/ishka.sqlite).
Регистрации нет: аккаунты заводятся руками через accounts.py, пароли хранятся только в виде хеша.
Роли: student и starosta. Староста рассылает своей группе (или подгруппе) Д/З, мероприятия и напоминания;
студенты видят их в брифинге, в Д/З, в мероприятиях и в напоминаниях.
Store(":memory:") — база в памяти (генератор датасета, тесты). Сайт может пользоваться этим же классом."""
import datetime as dt, hashlib, hmac, json, os, pathlib, secrets, sqlite3, unicodedata

DEFAULT_DB = pathlib.Path(__file__).parent / "db" / "ishka.sqlite"
ROLES = ("student", "starosta")
KINDS = ("homework", "event", "reminder")
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
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS group_posts (
    id INTEGER PRIMARY KEY,
    author_id INTEGER REFERENCES accounts(id) ON DELETE SET NULL,
    grp TEXT NOT NULL,
    subgroup INTEGER,                       -- NULL — всей группе
    kind TEXT NOT NULL CHECK (kind IN ('homework', 'event', 'reminder')),
    data TEXT NOT NULL,                     -- JSON: subject/task/due, title/date/time/place, text/fire_at
    until TEXT NOT NULL,                    -- 'YYYY-MM-DD HH:MM': после этого момента не показываем
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS group_posts_grp ON group_posts (grp, until);
CREATE TABLE IF NOT EXISTS user_data (       -- личные Д/З, заметки, напоминания, дедлайны, экзамены
    account_id INTEGER PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE,
    data TEXT NOT NULL,
    updated TEXT NOT NULL
);
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
        self._normalize()

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
        """Профиль для Backend: то же, что set_profile, плюс группа и роль."""
        p = {"group": acc["grp"], "role": acc["role"]}
        if acc.get("subgroup"):
            p["subgroup"] = acc["subgroup"]
        if acc.get("english_teacher"):
            p["english_teacher"] = acc["english_teacher"]
        if acc.get("adaptation"):
            p["adaptation"] = True
        return p

    def save_profile(self, account_id, profile):
        self.db.execute("UPDATE accounts SET subgroup = ?, english_teacher = ?, adaptation = ? WHERE id = ?",
                        (profile.get("subgroup"), profile.get("english_teacher"), int(bool(profile.get("adaptation"))),
                         account_id))
        self.db.commit()

    def members(self, group, subgroup=None, exclude=None):
        """Сколько студентов получит рассылку: вся группа или подгруппа (и те, у кого подгруппа не указана)."""
        q, args = "SELECT COUNT(*) FROM accounts WHERE grp = ?", [group_key(group)]
        if subgroup:
            q += " AND (subgroup = ? OR subgroup IS NULL)"
            args.append(subgroup)
        if exclude is not None:
            q += " AND id != ?"
            args.append(exclude)
        return self.db.execute(q, args).fetchone()[0]

    # ---------- рассылки старосты ----------
    def add_post(self, author_id, group, kind, data, until, subgroup=None, created=None):
        if kind not in KINDS:
            raise ValueError(kind)
        cur = self.db.execute(
            "INSERT INTO group_posts (author_id, grp, subgroup, kind, data, until, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (author_id, group_key(group), _subgroup(subgroup), kind, json.dumps(data, ensure_ascii=False), until,
             created or _now()))
        self.db.commit()
        return cur.lastrowid

    def posts(self, group, subgroup=None, after=None, kind=None):
        """Рассылки группе, которые ещё актуальны (until >= after), для студента из подгруппы subgroup:
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

    # ---------- личные данные ----------
    def load_data(self, account_id):
        row = self.db.execute("SELECT data FROM user_data WHERE account_id = ?", (account_id,)).fetchone()
        return json.loads(row["data"]) if row else {}

    def save_data(self, account_id, data):
        self.db.execute("INSERT INTO user_data (account_id, data, updated) VALUES (?, ?, ?) "
                        "ON CONFLICT(account_id) DO UPDATE SET data = excluded.data, updated = excluded.updated",
                        (account_id, json.dumps(data, ensure_ascii=False), _now()))
        self.db.commit()
