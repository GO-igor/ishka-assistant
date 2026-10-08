"""Диагностика модели на запущенном llama-server (окно 1 должно работать). Флаги — те же, что у chat.py:
    python diag_server.py --subgroup 2 --english "Аксёнова Н. В."    # ~2–5 минут
    python diag_server.py --url http://localhost:8081
    python diag_server.py --eval 30         # ещё 30 диалогов из data/eval.jsonl, как evaluate.py (дольше)
Задаёт модели вопросы из чата с тем же system и профилем, что chat.py: в точном формате обучения
(prompt_format.py → /completion) и для сравнения так, как это делал старый chat.py (/v1/chat/completions).
Так видно, где поломка: в самой модели или в том, как промпт собирается и разбирается. Пришли весь вывод целиком.
"""
import argparse, datetime as dt, json, math, os, sys, unicodedata, urllib.error, urllib.request
from backend import Backend
from gen_dataset import system
from prompt_format import HERE, TOOLS, render, parse_calls, answer_text, call_msg, tool_msg, prompt_diff

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
N_PREDICT = 512
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


def first_diff(x, y):
    i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    return f"символ {i}: обучение {x[max(0, i - 40):i + 40]!r}\n     сервер   {y[max(0, i - 40):i + 40]!r}"


def outcome(text, calls=None):
    """Что сделала модель: ("call", имя, аргументы), ("text", ответ) или ("empty",)."""
    calls = calls if calls is not None else parse_calls(text)
    if calls:
        args = calls[0].get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            pass
        return ("call", calls[0].get("name"), args)
    answer = answer_text(text)
    return ("text", answer) if answer else ("empty",)


def describe(o):
    return (f"вызов {o[1]} {json.dumps(o[2], ensure_ascii=False)}" if o[0] == "call"
            else f"текст: «{short(o[1])}»" if o[0] == "text" else "пустой ответ")


def complete(msgs):
    """Формат обучения: промпт собирает prompt_format.py, генерация как в chat.py (greedy, до <|im_end|>).
    Возвращает (сырой текст, строка про вероятность <tool_call>, обрезан ли ответ)."""
    r, err = req("/completion", {"prompt": render(msgs, add_generation_prompt=True), "n_predict": N_PREDICT,
                                 "temperature": 0, "stop": ["<|im_end|>"], "cache_prompt": True, "n_probs": 5})
    if err:
        raise SystemExit(f"✗ /completion: {err}")
    prob = ""
    cp = r.get("completion_probabilities") or []
    if cp:  # новые версии llama.cpp: top_logprobs с logprob, старые: probs с prob
        p = None
        for alt in cp[0].get("top_logprobs") or []:
            if alt.get("id") == TOOL_CALL or alt.get("token") == "<tool_call>":
                p = math.exp(alt["logprob"])
        for alt in cp[0].get("top_probs") or cp[0].get("probs") or []:
            if alt.get("id") == TOOL_CALL or "<tool_call>" in (alt.get("token"), alt.get("tok_str")):
                p = alt["prob"]
        prob = f"   [P(<tool_call>) = {p:.0%}]" if p is not None else "   [<tool_call> не в топ-5]"
    cut = r.get("stop_type") == "limit" or (r.get("tokens_predicted") or 0) >= N_PREDICT
    return r.get("content") or "", prob, cut


def to_openai(history):
    """История в формате обучения -> формат OpenAI, как в chat.py --api openai и в старом chat.py."""
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


v1_state = {"err": None, "same": 0, "diff": []}


def via_v1(msgs, o, label):
    """Тот же вопрос через /v1/chat/completions (шаблон и разбор вызовов — llama.cpp). Печатает, совпало ли."""
    if v1_state["err"]:
        return
    r, err = req("/v1/chat/completions", {"model": "ishka", "messages": to_openai(msgs), "tools": TOOLS,
                                          "temperature": 0, "max_tokens": N_PREDICT})
    if err:
        v1_state["err"] = err
        print(f"     /v1: не ответил ({err}); сравнение через /v1 пропускаю")
        return
    m = (r.get("choices") or [{}])[0].get("message") or {}
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in m.get("tool_calls") or []]
    o2 = outcome(m.get("content") or "", calls or None)
    if o2 == o:
        v1_state["same"] += 1
        print("     /v1: то же самое")
    else:
        v1_state["diff"].append(label)
        print(f"     /v1: ИНАЧЕ → {describe(o2)}")


def probe(msgs, want, label):
    """Один шаг модели. want: имя инструмента, None — ждём текст, или множество допустимых вариантов.
    Возвращает (верно ли, что сделала модель)."""
    text, prob, cut = complete(msgs)
    o = outcome(text)
    wants = want if isinstance(want, set) else {want}
    ok = (o[0] == "call" and o[1] in wants) or (o[0] == "text" and None in wants)
    need = " или ".join(sorted("текст" if w is None else w for w in wants))
    print(f"  {'✓' if ok else '✗'} {label} (ждём {need}) → {describe(o)}{prob}{'  [обрезано]' if cut else ''}")
    print(f"     сырой ответ: {short(text, 160)!r}" if text else "     сырой ответ: модель вернула пустую строку")
    via_v1(msgs, o, label)
    return ok, o


# ---------- 1. Сервер ----------
props, err = req("/props")
if err:
    raise SystemExit(f"✗ Сервер {BASE} не отвечает ({err}). Запусти llama-server в окне 1 и дождись «server is listening».")
gen = props.get("default_generation_settings") or {}
print(f"Сервер: {BASE}\n  модель: {props.get('model_path')}\n  контекст: {gen.get('n_ctx') or props.get('n_ctx')}"
      f"\n  llama.cpp: {props.get('build_info', '?')}\n  профиль: {backend.profile}, сейчас {now:%Y-%m-%d %H:%M}")

# ---------- 2. Шаблон чата в .gguf: собирает ли он те же промпты ----------
tpl, tpl_ok = props.get("chat_template"), None
if not tpl:
    print("? Шаблон чата сервер не показал")
else:
    try:
        d = prompt_diff(tpl, EVAL[:20])
        tpl_ok = d is None
        if tpl_ok:
            print("✓ Шаблон чата в .gguf собирает промпты так же, как при обучении")
        else:
            open(os.path.join(HERE, "diag_gguf_template.jinja"), "w", encoding="utf-8").write(tpl)
            print("✗ Шаблон чата в .gguf собирает промпт НЕ так, как при обучении (сохранён в diag_gguf_template.jinja):\n  "
                  + first_diff(*d).replace("сервер  ", ".gguf   "))
    except Exception as e:
        print(f"? Не получилось собрать промпт по шаблону из .gguf ({type(e).__name__}: {e})")

# ---------- 3. Спецтокены ----------
tok_ok = True
for s, want in (("<|im_start|>", IM_START), ("<tool_call>", TOOL_CALL)):
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

# ---------- 4. Как промпт после инструмента собирает сам llama.cpp (путь /v1, старый chat.py) ----------
case = next((m for m in EVAL if len(m) > 4 and m[2].get("tool_calls") and m[4].get("content")), None)
if case:
    r, err = req("/apply-template", {"messages": to_openai(case[:4]), "tools": TOOLS})
    if err:
        print(f"? llama.cpp не собрал промпт сам ({err}); возможно, сервер запущен без --jinja")
    elif r.get("prompt") == render(case[:4], add_generation_prompt=True):
        print("✓ llama.cpp собирает промпт после инструмента так же, как при обучении")
    else:
        print("✗ llama.cpp собирает промпт после инструмента НЕ так, как при обучении:\n  "
              + first_diff(render(case[:4], add_generation_prompt=True), r.get("prompt") or ""))

# ---------- 5. Вопросы из чата, с твоим профилем ----------
LIVE = [("Кто ты?", None), ("Как тебя зовут?", None), ("привет", None), ("объясни рекурсию", None),
        ("Какое расписание на завтра?", "get_schedule"), ("добавь экзамен по питону 15 января в 9:00", "add_exam")]
# если модель на шаге 1 не вызвала нужный инструмент, для шага 2 берём такой вызов (как было в чате)
EXPECTED_CALL = {"get_schedule": {"day": "tomorrow"},
                 "add_exam": {"subject": "Основы Python", "kind": "экзамен", "date": "2027-01-15", "time": "09:00"}}
print("\nШаг 1 — сразу на вопрос (system и профиль как в chat.py):")
s1, after = [], []
for q, want in LIVE:
    msgs = [SYSTEM, {"role": "user", "content": q}]
    ok, o = probe(msgs, want, f"«{q}»")
    s1.append((q, ok))
    if o[0] == "call" and o[1] in NAMES and isinstance(o[2], dict):
        after.append((msgs, o[1], o[2]))  # что модель вызвала на самом деле, то и выполняем — как chat.py
    if want and not (o[0] == "call" and o[1] == want):
        after.append((msgs, want, EXPECTED_CALL[want]))  # и нужный вызов, чтобы проверить ответ на его результат

print("Шаг 2 — после результата инструмента (ждём ответ текстом):")
s2 = []
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
        o = outcome(complete(m[:2])[0])
        want = m[2]["tool_calls"][0]["function"]["name"] if m[2].get("tool_calls") else None
        got = o[1] if o[0] == "call" else None
        n1 += 1
        ok1 += got == want
        if got != want:
            errs.append(f"шаг 1 «{m[1]['content']}»: ждали {want or 'текст'}, получили {describe(o)}")
        for j in range(3, len(m) - 1):
            if m[j]["role"] != "tool" or m[j + 1]["role"] != "assistant":
                continue
            o = outcome(complete(m[:j + 1])[0])
            want = m[j + 1]["tool_calls"][0]["function"]["name"] if m[j + 1].get("tool_calls") else None
            good = (o[0] == "call" and o[1] == want) if want else o[0] == "text"
            n2 += 1
            ok2 += good
            if not good:
                errs.append(f"шаг 2 «{m[1]['content']}» после {m[j]['name']}: ждали {want or 'текст'}, получили {describe(o)}")
        print(f"  проверено диалогов {i}/{min(a.eval, len(EVAL))}", file=sys.stderr, flush=True)
    print(f"\neval.jsonl, {n1} диалогов: шаг 1 инструмент верно {ok1}/{n1}; шаг 2 верно {ok2}/{n2}")
    for e in errs[:15]:
        print("  ✗", e)

# ---------- Итог ----------
print("\nИтог:")
if tpl_ok is False:
    print("⚠ Шаблон в .gguf собирает промпт не так, как chat_template.jinja: пришли вывод и файл diag_gguf_template.jinja.")
if s2 and all(s2):
    print("✓ После результата инструмента модель отвечает текстом: в этой проверке модель в порядке.\n"
          "  Запусти chat.py с теми же флагами. Если там снова «Не получилось ответить», пришли и этот вывод, и вывод чата.")
elif s2 and not any(s2):
    print("✗ Даже в точном формате обучения модель после результата инструмента не отвечает текстом.\n"
          "  Значит, дело в весах этого .gguf, а не в llama.cpp и не в chat.py. Дальше — проверка LoRA в Colab (RUN_MAC.md, шаг 7).")
else:
    print(f"~ После результата инструмента модель ответила текстом в {sum(s2)} из {len(s2)} случаев: работает нестабильно.\n"
          "  Пришли этот вывод целиком.")
bad1 = [q for q, ok in s1 if not ok]
if bad1:
    print("⚠ Шаг 1: " + ", ".join(f"«{q}»" for q in bad1) + " — модель ответила не так, как ждали.")
if v1_state["diff"]:
    d = v1_state["diff"]
    print(f"⚠ Через /v1 (как в старом chat.py) ответ отличался в {len(d)} случаях из {len(d) + v1_state['same']}: "
          + "; ".join(d[:3]) + ("; …" if len(d) > 3 else ""))
elif v1_state["same"]:
    print(f"• Через /v1 (как в старом chat.py) ответы те же ({v1_state['same']} из {v1_state['same']}).")
