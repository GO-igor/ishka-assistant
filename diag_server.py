"""Диагностика модели на запущенном llama-server (окно 1 должно работать). Флаги — те же, что у chat.py:
    python diag_server.py --subgroup 2 --english "Аксёнова Н. В."    # ~2–5 минут
    python diag_server.py --url http://localhost:8081
    python diag_server.py --eval 30         # ещё 30 диалогов из data/eval.jsonl, как evaluate.py (дольше)
Задаёт модели вопросы из чата с тем же system, профилем и шаблоном, что chat.py, и решает «вызов или текст»
так же, как chat.py: по P(<tool_call>) на первом токене ответа. Для каждого вопроса печатает эту вероятность,
самые вероятные первые токены и что ответил бы chat.py. Пришли весь вывод целиком.
"""
import argparse, datetime as dt, json, os, sys, unicodedata, urllib.error, urllib.request
from backend import Backend
from gen_dataset import system
from prompt_format import (HERE, TOOLS, CALL_THRESHOLD, render, prompt_diff, parse_calls, answer_text, call_msg,
                           tool_msg, first_token_probs, tool_call_prob, pick_template)

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8080")
ap.add_argument("--eval", type=int, default=0, help="сколько диалогов из data/eval.jsonl прогнать дополнительно")
ap.add_argument("--subgroup", type=int)
ap.add_argument("--english")
ap.add_argument("--adaptation", action="store_true")
ap.add_argument("--now", help="подменить текущее время: '2026-10-06 12:10'")
a = ap.parse_args()

BASE = a.url.rstrip("/")
BASE = BASE[:-3] if BASE.endswith("/v1") else BASE
_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TOOL_CALL, IM_START, BOS = 151657, 151644, 151643  # номера спецтокенов Qwen3
N_PREDICT = 1024  # как в chat.py
NAMES = {t["function"]["name"] for t in TOOLS}
EVAL = [json.loads(l)["messages"] for l in open(os.path.join(HERE, "data", "eval.jsonl"), encoding="utf-8")]

# Тот же контекст, что в chat.py: время, профиль через set_profile, system из gen_dataset
now = dt.datetime.fromisoformat(a.now) if a.now else dt.datetime.now().replace(second=0, microsecond=0)
backend = Backend(now, {})
if a.subgroup or a.english or a.adaptation:
    english = unicodedata.normalize("NFC", a.english) if a.english else None
    if "error" in backend.set_profile(subgroup=a.subgroup, english_teacher=english, adaptation=a.adaptation):
        raise SystemExit(f"Преподаватель английского «{a.english}» не найден")
SYSTEM = system(now, backend.profile)


def req(path, body=None):
    """(ответ, None) или (None, текст ошибки)."""
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"})
    try:
        with _http.open(r, timeout=900) as resp:
            return json.load(resp), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return None, str(getattr(e, "reason", e))


def short(text, n=110):
    text = text.replace("\n", "⏎ ")
    return text if len(text) <= n else text[:n] + "…"


def first_diff(x, y, names=("обучение", "сервер  ")):
    i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    return f"символ {i}: {names[0]} {x[max(0, i - 40):i + 40]!r}\n     {names[1]} {y[max(0, i - 40):i + 40]!r}"


def outcome(text):
    """Что сделала модель: ("call", имя, аргументы), ("text", ответ) или ("empty",)."""
    calls = parse_calls(text)
    if calls:
        return ("call", calls[0].get("name"), calls[0].get("arguments"))
    answer = answer_text(text)
    return ("text", answer) if answer else ("empty",)


def describe(o):
    return (f"вызов {o[1]} {json.dumps(o[2], ensure_ascii=False)}" if o[0] == "call"
            else f"текст: «{short(o[1])}»" if o[0] == "text" else "пустой ответ")


def completion(body):
    r, err = req("/completion", dict(body, temperature=0, cache_prompt=True))
    if err:
        raise SystemExit(f"✗ /completion: {err}")
    return r


# Где самым вероятным первым токеном был <tool_call>, но за вызов было меньше порога (chat.py ответил текстом):
# fixed — так и надо было (старый chat.py тут вызывал инструмент), cut — нужен был вызов, а порог его отменил
stats = {"fixed": 0, "cut": []}


def chat_step(msgs):
    """Один шаг как в chat.py: первый токен с вероятностями → вызов (P(<tool_call>) ≥ порога) или ответ текстом
    (начало ответа как при обучении + запрет <tool_call>). Возвращает (сырой текст, P, топ первых токенов,
    взял бы вызов жадный выбор при P ниже порога, почему ответ обрезан или None)."""
    prompt = render(msgs, add_generation_prompt=True, template=TPL)
    probs = first_token_probs(completion({"prompt": prompt, "n_predict": 1, "n_probs": 10}))
    p = tool_call_prob(probs, TOOL_CALL) if probs is not None else None
    body = {"prompt": prompt, "n_predict": N_PREDICT, "stop": ["<|im_end|>"]}
    greedy_call = p is not None and p < CALL_THRESHOLD and bool(probs) and probs[0][0] == TOOL_CALL
    if p is not None and p < CALL_THRESHOLD:
        body.update(prompt=prompt + PREFIX, logit_bias=[[TOOL_CALL, False]])
    r = completion(body)
    cut = None
    if r.get("stop_type") == "limit":
        cut = (f"длиннее {N_PREDICT} токенов" if (r.get("tokens_predicted") or 0) >= N_PREDICT
               else "кончился контекст сервера, запусти llama-server с -c 16384")
    return r.get("content") or "", p, probs, greedy_call, cut


def count_greedy(greedy_call, call_wanted, label, p):
    if greedy_call:
        if call_wanted:
            stats["cut"].append(f"{label} (P = {p:.0%})")
        else:
            stats["fixed"] += 1


def probe(msgs, want, label):
    """want: имя инструмента, None — ждём текст, или множество допустимых вариантов. Возвращает (верно ли, outcome)."""
    text, p, probs, greedy_call, cut = chat_step(msgs)
    o = outcome(text)
    wants = want if isinstance(want, set) else {want}
    ok = (o[0] == "call" and o[1] in wants) or (o[0] == "text" and None in wants)
    count_greedy(greedy_call, None not in wants, label, p)
    need = " или ".join(sorted("текст" if w is None else w for w in wants))
    print(f"  {'✓' if ok else '✗'} {label} (ждём {need}) → {describe(o)}")
    if cut:
        print(f"     [ответ обрезан: {cut}]")
    if p is None:
        print("     сервер не прислал вероятности: решение жадное, как раньше")
    else:
        top = ", ".join(f"{t!r} {q:.0%}" for _, t, q in probs[:5])
        print(f"     P(<tool_call>) = {p:.0%} → {'вызов' if p >= CALL_THRESHOLD else 'ответ текстом'}; первые токены: {top}")
    if not ok:
        print(f"     сырой ответ: {short(text, 160)!r}" if text else "     сырой ответ: модель вернула пустую строку")
    return ok, o


def to_openai(history):
    """История в формате обучения -> формат OpenAI (для /apply-template)."""
    out, ids = [], []
    for i, m in enumerate(history):
        m = dict(m)
        if m.get("tool_calls"):
            ids = [f"call_{i}_{j}" for j in range(len(m["tool_calls"]))]
            m["tool_calls"] = [{"id": k, "type": "function", "function": {
                "name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"], ensure_ascii=False)}}
                for k, c in zip(ids, m["tool_calls"])]
        elif m["role"] == "tool":
            m["tool_call_id"] = ids.pop(0) if ids else "call"
            m.pop("name", None)
        out.append(m)
    return out


# ---------- 1. Сервер ----------
props, err = req("/props")
if err:
    raise SystemExit(f"✗ Сервер {BASE} не отвечает ({err}). Запусти llama-server в окне 1 и дождись «server is listening».")
gen = props.get("default_generation_settings") or {}
print(f"Сервер: {BASE}\n  модель: {props.get('model_path')}\n  контекст: {gen.get('n_ctx') or props.get('n_ctx')}"
      f"\n  llama.cpp: {props.get('build_info', '?')}\n  профиль: {backend.profile}, сейчас {now:%Y-%m-%d %H:%M}")

# ---------- 2. Шаблон чата из .gguf: с ним модель обучалась, его же берёт chat.py ----------
TPL, PREFIX, why = pick_template(props.get("chat_template"))  # то же, что делает chat.py
start = f"начинается с {PREFIX!r} (официальный шаблон Qwen3-2507: так модель видела ответы при обучении)" \
    if PREFIX else "начинается сразу с текста"
if TPL:
    print(f"✓ Шаблон чата из .gguf: ответ текстом {start}")
else:
    print(f"⚠ {why[0].upper() + why[1:]}: chat.py возьмёт chat_template.jinja, ответ текстом {start}")
    try:
        d = props.get("chat_template") and prompt_diff(props["chat_template"], EVAL[:20], gen_only=True)
        if d:
            print("  " + first_diff(*d, names=("проект", ".gguf ")))
    except Exception:
        pass

# ---------- 3. Спецтокены ----------
tok_ok = True
specials = [("<|im_start|>", IM_START), ("<tool_call>", TOOL_CALL)]
if "<think>" in PREFIX:
    specials += [("<think>", 151667), ("</think>", 151668)]
for s, want in specials:
    r, err = req("/tokenize", {"content": s, "add_special": False, "parse_special": True})
    ids = (r or {}).get("tokens")
    if ids != [want]:
        tok_ok = False
        print(f"✗ {s} → {ids or err} (ждали [{want}]): словарь модели не тот или сервер старый")
r, _ = req("/tokenize", {"content": "<|im_start|>", "add_special": True, "parse_special": True})
if r and (r.get("tokens") or [None])[0] == BOS:
    tok_ok = False
    print("✗ Сервер добавляет BOS в начало промпта — при обучении его не было")
if tok_ok:
    print("✓ Спецтокены читаются как один токен, BOS не добавляется")

# ---------- 4. Как тот же шаблон собирает сам llama.cpp ----------
case = next((m for m in EVAL if len(m) > 4 and m[2].get("tool_calls") and m[4].get("content")), None)
if case:
    r, err = req("/apply-template", {"messages": to_openai(case[:4]), "tools": TOOLS})
    mine = render(case[:4], add_generation_prompt=True, template=TPL)
    if err:
        print(f"? llama.cpp не собрал промпт сам ({err}); возможно, сервер запущен без --jinja")
    elif r.get("prompt") == mine:
        print("✓ llama.cpp собирает промпт после инструмента так же, как chat.py")
    else:
        print("✗ llama.cpp собирает промпт после инструмента НЕ так, как chat.py:\n  "
              + first_diff(mine, r.get("prompt") or "", names=("chat.py ", "llama.cpp")))

# ---------- 5. Вопросы из чата, с твоим профилем ----------
LIVE = [("Кто ты?", None), ("Как тебя зовут?", None), ("привет", None), ("объясни рекурсию", None),
        ("Какое расписание на завтра?", "get_schedule"), ("добавь экзамен по питону 15 января в 9:00", "add_exam")]
# если модель на шаге 1 не вызвала нужный инструмент, для шага 2 берём такой вызов (как было в чате)
EXPECTED_CALL = {"get_schedule": {"day": "tomorrow"},
                 "add_exam": {"subject": "Основы Python", "kind": "экзамен", "date": "2027-01-15", "time": "09:00"}}
print(f"\nШаг 1 — сразу на вопрос (вызов, только если P(<tool_call>) ≥ {CALL_THRESHOLD:.0%}, как в chat.py):")
s1, s2, after = [], [], []
for q, want in LIVE:
    msgs = [SYSTEM, {"role": "user", "content": q}]
    ok, o = probe(msgs, want, f"«{q}»")
    s1.append((q, ok))
    if o[0] == "call" and o[1] in NAMES and isinstance(o[2], dict):
        after.append((msgs, o[1], o[2]))  # что модель вызвала на самом деле, то и выполняем — как chat.py
    if want and not (o[0] == "call" and o[1] == want):
        after.append((msgs, want, EXPECTED_CALL[want]))  # и нужный вызов, чтобы проверить ответ на его результат

print("Шаг 2 — после результата инструмента (ждём ответ текстом):")
for msgs, name, args in after:
    try:
        res = backend.call(name, args)
    except Exception as e:  # аргументы не по схеме — так же, как chat.py
        res = {"error": f"{type(e).__name__}: {e}"}
    hist = msgs + [call_msg([(name, args)]), tool_msg(name, res)]
    want = {None, "make_study_plan"} if name == "add_exam" else None  # в датасете после add_exam бывает план
    s2.append(probe(hist, want, f"«{msgs[1]['content']}» → {name} → результат")[0])

print("Шаг 2 на диалогах из eval.jsonl:")
for name in ("get_schedule", "list_notes", "add_exam"):
    m = next((m for m in EVAL if len(m) > 4 and m[2].get("tool_calls") and m[4].get("content")
              and m[2]["tool_calls"][0]["function"]["name"] == name), None)
    if m:
        s2.append(probe(m[:4], None, f"«{m[1]['content']}» → {name} → результат")[0])

# ---------- 6. (по желанию) диалоги из eval.jsonl ----------
if a.eval:
    n1 = ok1 = n2 = ok2 = 0
    errs = []
    for i, m in enumerate(EVAL[: a.eval], 1):
        text, p, _, greedy_call, _ = chat_step(m[:2])
        o = outcome(text)
        want = m[2]["tool_calls"][0]["function"]["name"] if m[2].get("tool_calls") else None
        got = o[1] if o[0] == "call" else None if o[0] == "text" else "пустой ответ"
        count_greedy(greedy_call, want, f"eval «{m[1]['content']}»", p)
        n1 += 1
        ok1 += got == want
        if got != want:
            errs.append(f"шаг 1 «{m[1]['content']}»: ждали {want or 'текст'}, получили {describe(o)}")
        for j in range(3, len(m) - 1):
            if m[j]["role"] != "tool" or m[j + 1]["role"] != "assistant":
                continue
            text, p, _, greedy_call, _ = chat_step(m[:j + 1])
            o = outcome(text)
            want = m[j + 1]["tool_calls"][0]["function"]["name"] if m[j + 1].get("tool_calls") else None
            count_greedy(greedy_call, want, f"eval «{m[1]['content']}» после {m[j]['name']}", p)
            good = (o[0] == "call" and o[1] == want) if want else o[0] == "text"
            n2 += 1
            ok2 += good
            if not good:
                errs.append(f"шаг 2 «{m[1]['content']}» после {m[j]['name']}: ждали {want or 'текст'}, получили {describe(o)}")
        print(f"  проверено диалогов {i}/{min(a.eval, len(EVAL))}", file=sys.stderr, flush=True)
    print(f"\neval.jsonl, {n1} диалогов (как chat.py): шаг 1 инструмент верно {ok1}/{n1}; шаг 2 верно {ok2}/{n2}")
    for e in errs[:15]:
        print("  ✗", e)

# ---------- Итог ----------
print("\nИтог:")
bad1 = [q for q, ok in s1 if not ok]
eval_bad = a.eval and (ok1 < n1 or ok2 < n2)
if stats["cut"]:
    print(f"✗ Нужный вызов отменён: за <tool_call> было меньше {CALL_THRESHOLD:.0%}, хотя он был самым вероятным токеном: "
          + ", ".join(stats["cut"]) + ".")
if bad1:
    print("⚠ Шаг 1: " + ", ".join(f"«{q}»" for q in bad1) + " — ответ не тот, что ждали. Запиши эти фразы: пример для обучения.")
if not all(s2):
    print(f"✗ После результата инструмента правильно в {sum(s2)} из {len(s2)} случаев.")
elif bad1 or stats["cut"]:
    print("✓ После результата инструмента модель отвечает текстом.")
else:
    print("✓ С новым chat.py модель отвечает правильно на все вопросы из чата: запусти чат (RUN_MAC.md, шаг 6).")
if a.eval:
    print(f"{'⚠' if eval_bad else '✓'} eval.jsonl: шаг 1 верно {ok1}/{n1}, шаг 2 верно {ok2}/{n2}" + (" (ошибки выше)" if eval_bad else ""))
if stats["fixed"]:
    print(f"• В {stats['fixed']} случаях самым вероятным токеном был <tool_call>, хотя за вызов было меньше "
          f"{CALL_THRESHOLD:.0%}:\n  старый chat.py вызывал тут инструмент — отсюда «Не получилось ответить». Новый отвечает текстом.")
if stats["cut"] or bad1 or not all(s2) or eval_bad:
    print("Пришли этот вывод целиком.")
