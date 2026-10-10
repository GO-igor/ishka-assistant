"""Аккаунты студентов и старост (регистрации нет, заводятся здесь). База: db/ishka.sqlite.
    python accounts.py add ivanov --group 8К51 --subgroup 2 --english Аксёнова --name "Иванов Иван"
    python accounts.py add petrova --group 8К51 --role starosta --name "Петрова Анна"
    python accounts.py import group.csv --out passwords.csv   # сразу всю группу из таблицы
    python accounts.py list
    python accounts.py edit ivanov --role starosta            # и --group, --subgroup, --english, --name, --adaptation yes|no
    python accounts.py passwd ivanov                          # новый пароль
    python accounts.py remove ivanov
Пароль спрашивается при вводе (не виден на экране); --password ПАРОЛЬ — задать сразу, --generate — придумать случайный.
Таблица для import (CSV в UTF-8, первая строка — заголовок): login,password,role,group,subgroup,english,name,adaptation
Разделитель — запятая или точка с запятой. Пустой password — пароль придумается сам; новые пароли допишутся
в файл --out (раздай их студентам). Логины без учёта регистра: Ivanov и ivanov — один аккаунт."""
import argparse, csv, getpass, os, sqlite3, unicodedata
from backend import ENG_TEACHERS
from store import ROLES, Store, login_key, new_password

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
    acc = st.account(login=a.login)
    print(f"Готово: {acc['login']} ({'староста' if a.role == 'starosta' else 'студент'}, группа {acc['grp']})")
elif a.cmd == "import":
    try:
        text = open(a.csv, encoding="utf-8-sig", newline="").read()
    except UnicodeDecodeError:
        raise SystemExit("Таблица не в UTF-8: сохрани её как «CSV UTF-8» (Excel) или «CSV, Юникод (UTF-8)» и повтори")
    head = text.split("\n", 1)[0]
    rows = csv.DictReader(text.splitlines(), delimiter=";" if head.count(";") > head.count(",") else ",")
    if "login" not in [(k or "").strip().lower() for k in rows.fieldnames or []]:
        raise SystemExit("В первой строке таблицы нет колонки login. Нужен заголовок: "
                         "login,password,role,group,subgroup,english,name,adaptation")
    # пароли дописываем в --out сразу: если что-то прервётся, уже заведённые аккаунты не останутся без пароля
    outf = None
    if a.out:
        new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
        try:
            outf = open(a.out, "a", encoding="utf-8", newline="")
        except OSError as e:
            raise SystemExit(f"Не могу записать {a.out}: {e.strerror}")
        w = csv.DictWriter(outf, ["login", "password"])
        if new_file:
            w.writeheader()
    n = 0
    for i, row in enumerate(rows, 2):
        row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
        if not any(row.values()):
            continue
        if not row.get("login"):
            print(f"строка {i}: пустой login — пропущена")
            continue
        pw = row.get("password") or new_password()
        role = (row.get("role") or "student").lower()
        role = {"староста": "starosta", "студент": "student"}.get(role, role)
        try:
            st.add_account(nfc(row["login"]), pw, nfc(row.get("group")), role=role, name=nfc(row.get("name")) or None,
                           subgroup=int(row["subgroup"]) if row.get("subgroup") else None,
                           english_teacher=teacher(row.get("english")),
                           adaptation=row.get("adaptation", "").lower() in ("1", "yes", "да", "true"))
        except (sqlite3.IntegrityError, ValueError) as e:
            print(f"строка {i}: {row['login']} — пропущена ({'логин уже занят' if isinstance(e, sqlite3.IntegrityError) else e})")
            continue
        n += 1
        pair = {"login": login_key(row["login"]), "password": pw if not row.get("password") else "(из таблицы)"}
        if outf:
            w.writerow(pair)
            outf.flush()
        else:
            print(f"  {pair['login']}: {pair['password']}")
    print(f"Добавлено аккаунтов: {n}")
    if outf:
        outf.close()
        print(f"Логины и новые пароли дописаны в {a.out}")
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
    try:
        found = st.update_account(a.login, **f)
    except ValueError as e:
        raise SystemExit(f"Не изменён: {e}")
    if not found:
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
