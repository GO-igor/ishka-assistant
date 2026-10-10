"""Аккаунты студентов и старост (регистрации нет, заводятся здесь). База: db/ishka.sqlite.
    python accounts.py add ivanov --group 8К51 --subgroup 2 --english Аксёнова --name "Иванов Иван"
    python accounts.py add petrova --group 8К51 --role starosta --name "Петрова Анна"
    python accounts.py import group.csv --out passwords.csv   # сразу всю группу из таблицы
    python accounts.py list
    python accounts.py edit ivanov --role starosta            # и --group, --subgroup, --english, --name, --adaptation yes|no
    python accounts.py passwd ivanov                          # новый пароль
    python accounts.py remove ivanov
Пароль спрашивается при вводе (не виден на экране); --password ПАРОЛЬ — задать сразу, --generate — придумать случайный.
Таблица для import (CSV, первая строка — заголовок): login,password,role,group,subgroup,english,name,adaptation
Пустой password — пароль придумается сам, все новые пароли попадут в файл --out (раздай их студентам)."""
import argparse, csv, getpass, sqlite3, unicodedata
from backend import ENG_TEACHERS
from store import ROLES, Store, new_password

ap = argparse.ArgumentParser(description="Аккаунты Ишки", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
ap.add_argument("--db", help="файл базы (по умолчанию db/ishka.sqlite)")
sub = ap.add_subparsers(dest="cmd", required=True)


def profile_args(p, edit=False):
    p.add_argument("--group", required=not edit, help="учебная группа, например 8К51")
    p.add_argument("--role", choices=ROLES, default=None if edit else "student")
    p.add_argument("--name", help="имя и фамилия")
    p.add_argument("--subgroup", type=int, choices=(1, 2, 3))
    p.add_argument("--english", help="преподаватель английского (можно только фамилию)")
    p.add_argument("--adaptation", choices=("yes", "no"), help="курсы (А): yes/no")


p = sub.add_parser("add", help="новый аккаунт")
p.add_argument("login")
profile_args(p)
p.add_argument("--password")
p.add_argument("--generate", action="store_true", help="придумать пароль и показать его")
p = sub.add_parser("import", help="аккаунты из CSV")
p.add_argument("csv")
p.add_argument("--out", help="куда записать логины и новые пароли (CSV)")
sub.add_parser("list", help="все аккаунты")
p = sub.add_parser("edit", help="изменить аккаунт")
p.add_argument("login")
profile_args(p, edit=True)
p = sub.add_parser("passwd", help="сменить пароль")
p.add_argument("login")
p.add_argument("--password")
p.add_argument("--generate", action="store_true")
p = sub.add_parser("remove", help="удалить аккаунт")
p.add_argument("login")
a = ap.parse_args()


def teacher(name):
    """«Аксёнова», «аксеновой» → «Аксёнова Н. В.» (как set_profile)."""
    if not name:
        return None
    key = unicodedata.normalize("NFC", name).lower().replace("ё", "е")[:5]
    hits = [t for t in ENG_TEACHERS if t.lower().replace("ё", "е").startswith(key)]
    if len(hits) != 1:
        raise ValueError(f"преподаватель английского «{name}» не найден, варианты: {', '.join(ENG_TEACHERS)}")
    return hits[0]


def ask_password(gen):
    if gen:
        pw = new_password()
        print(f"пароль: {pw}")
        return pw
    pw = getpass.getpass("пароль: ")
    if pw != getpass.getpass("ещё раз: "):
        raise SystemExit("Пароли не совпали")
    if len(pw) < 6:
        raise SystemExit("Пароль короче 6 символов")
    return pw


nfc = lambda s: unicodedata.normalize("NFC", s.strip()) if s else s
st = Store(a.db)
try:
    english = teacher(a.english) if a.cmd in ("add", "edit") else None
except ValueError as e:
    raise SystemExit(str(e)[:1].upper() + str(e)[1:])
if a.cmd == "add":
    if st.account(login=nfc(a.login)):
        raise SystemExit(f"Логин «{a.login}» уже занят")
    pw = a.password or ask_password(a.generate)
    try:
        st.add_account(nfc(a.login), pw, nfc(a.group), role=a.role, name=nfc(a.name), subgroup=a.subgroup,
                       english_teacher=english, adaptation=a.adaptation == "yes")
    except sqlite3.IntegrityError:
        raise SystemExit(f"Логин «{a.login}» уже занят")
    except ValueError as e:
        raise SystemExit(f"Не добавлен: {e}")
    print(f"Готово: {a.login} ({'староста' if a.role == 'starosta' else 'студент'}, группа {a.group})")
elif a.cmd == "import":
    out, n = [], 0
    with open(a.csv, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 2):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            if not row.get("login"):
                continue
            pw = row.get("password") or new_password()
            role = row.get("role") or "student"
            if role in ("староста",):
                role = "starosta"
            elif role in ("студент",):
                role = "student"
            try:
                st.add_account(nfc(row["login"]), pw, nfc(row.get("group")), role=role, name=nfc(row.get("name")) or None,
                               subgroup=int(row["subgroup"]) if row.get("subgroup") else None,
                               english_teacher=teacher(row.get("english")),
                               adaptation=row.get("adaptation", "").lower() in ("1", "yes", "да", "true"))
            except (sqlite3.IntegrityError, ValueError) as e:
                print(f"строка {i}: {row['login']} — пропущена ({'логин уже занят' if isinstance(e, sqlite3.IntegrityError) else e})")
                continue
            n += 1
            out.append({"login": row["login"], "password": pw if not row.get("password") else "(из таблицы)"})
    print(f"Добавлено аккаунтов: {n}")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, ["login", "password"])
            w.writeheader()
            w.writerows(out)
        print(f"Логины и пароли: {a.out}")
    else:
        for r in out:
            print(f"  {r['login']}: {r['password']}")
elif a.cmd == "list":
    rows = st.accounts()
    if not rows:
        print("Аккаунтов нет")
    for r in rows:
        extra = ", ".join(x for x in (f"подгруппа {r['subgroup']}" if r["subgroup"] else "", r["english_teacher"] or "",
                                      "курсы (А)" if r["adaptation"] else "", "" if r["can_login"] else "вход закрыт") if x)
        print(f"{r['grp']:8} {'староста' if r['role'] == 'starosta' else 'студент ':8} {r['login']:20} {r['name'] or ''}"
              + (f"  ({extra})" if extra else ""))
elif a.cmd == "edit":
    f = {k: v for k, v in (("role", a.role), ("name", nfc(a.name)), ("grp", nfc(a.group)), ("subgroup", a.subgroup),
                           ("english_teacher", english)) if v is not None}
    if a.adaptation:
        f["adaptation"] = a.adaptation == "yes"
    if not f:
        raise SystemExit("Нечего менять: укажи --role, --group, --subgroup, --english, --name или --adaptation")
    if not st.update_account(a.login, **f):
        raise SystemExit(f"Аккаунта «{a.login}» нет")
    print("Готово")
elif a.cmd == "passwd":
    if not st.account(login=a.login):
        raise SystemExit(f"Аккаунта «{a.login}» нет")
    st.set_password(a.login, a.password or ask_password(a.generate))
    print("Пароль изменён")
elif a.cmd == "remove":
    if not st.remove_account(a.login):
        raise SystemExit(f"Аккаунта «{a.login}» нет")
    print("Удалён")
