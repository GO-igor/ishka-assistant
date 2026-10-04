"""Перефразированные ответы ассистента поверх шаблонных (data/paraphrases.json).

Ключ — хеш (вопрос студента + шаблонный ответ). gen_dataset.py после генерации подменяет ответ,
если такой ключ есть. Изменили базу и ответ стал другим — ключ не совпадёт, останется шаблон.
check() — автопроверка, что в перефразировке не потерялись и не появились факты."""
import hashlib, json, pathlib, re

PATH = pathlib.Path(__file__).parent / "data" / "paraphrases.json"
PLACES = ["ярче", "rostics", "сибирские блины"]
BUILDINGS = ["ГК", "КЦ", "МКЦ"]
KEYCAP = re.compile("[0-9]️?⃣")          # 1️⃣ и т.п.
LIST_NUM = re.compile(r"(?m)^\s*\d+[.)]\s")                   # «1. » в начале строки
NUM = re.compile(r"\d+(?::\d+)?(?:/\d+)?[АБA-Z]?")
TEACHER = re.compile(r"[А-ЯЁ][а-яё]+ [А-ЯЁ]\. ?[А-ЯЁ]\.")
POLITE = re.compile(r"(?<![а-яё])(Вы|Вас|Вам|Ваш[а-яё]*)(?![а-яё])")


def key(user: str, answer: str) -> str:
    return hashlib.sha1((user + "\n" + answer).encode()).hexdigest()[:12]


def _nums(text: str) -> set:
    text = LIST_NUM.sub(" ", KEYCAP.sub(" ", text))
    out = set()
    for n in NUM.findall(text):
        if ":" in n:  # 08:30 == 8:30
            h, m = n.split(":")
            n = f"{int(h)}:{m}"
        elif n.isdigit():
            n = str(int(n))
        out.add(n)
    return out


def _norm(t: str) -> str:
    return t.lower().replace("ё", "е")


def check(original: str, para: str, tool_result: str = "") -> list:
    """Список проблем; пустой список — перефразировка годится."""
    problems = []
    o, p = _nums(original), _nums(para)
    allowed = o | _nums(tool_result) | {"0", "1", "2"}  # «пару минут», «за 1 день» и т.п.
    if o - p:
        problems.append(f"пропали числа {sorted(o - p)}")
    if p - allowed:
        problems.append(f"новые числа {sorted(p - allowed)}")
    pt = {t.replace(". ", ".") for t in TEACHER.findall(para)}
    for t in TEACHER.findall(original):
        if t.replace(". ", ".") not in pt:
            problems.append(f"пропало ФИО {t}")
    no, np_ = _norm(original), _norm(para)
    for w in PLACES:
        if (w in no) != (w in np_):
            problems.append(f"место «{w}» {'пропало' if w in no else 'появилось'}")
    for b in BUILDINGS:
        rb = re.compile(rf"(?<![А-ЯЁ]){b}(?![А-ЯЁ])")
        if bool(rb.search(original)) and not rb.search(para):
            # корпус мог быть назван полностью
            full = {"ГК": "главн", "КЦ": "кибернет", "МКЦ": "культурн"}[b]
            if full not in np_:
                problems.append(f"пропал корпус {b}")
    for stem in ("нечетн", "четн"):
        if (stem in no) != (stem in np_):
            problems.append(f"чётность недели: «{stem}»")
            break
    for v in ("не успева", "впритык"):
        if (v in no) != (v in np_):
            problems.append(f"вердикт «{v}»")
    if "успева" in no and "не успева" not in no and "не успева" in np_:
        problems.append("вердикт перевёрнут")
    if POLITE.search(para):
        problems.append("обращение на «Вы»")
    if not 0.4 * len(original) <= len(para) <= 1.6 * len(original) + 40:
        problems.append(f"длина {len(para)} vs {len(original)}")
    return problems


def load() -> dict:
    if PATH.exists():
        return json.loads(PATH.read_text(encoding="utf-8"))
    return {}


def apply(messages: list, overlay: dict) -> bool:
    """Подменяет последний ответ ассистента, если для него есть перефразировка."""
    if len(messages) < 3 or messages[1]["role"] != "user":
        return False
    k = key(messages[1]["content"], messages[-1]["content"])
    if k in overlay and overlay[k]["original"] == messages[-1]["content"]:
        messages[-1] = {"role": "assistant", "content": overlay[k]["text"]}
        return True
    return False
