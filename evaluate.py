"""Оценка модели на data/eval.jsonl: правильный ли инструмент и аргументы на первом шаге.
Вызов или текст решается как в chat.py: вызов, только если P(<tool_call>) на первом токене ≥ CALL_THRESHOLD,
иначе ответ текстом. Для сравнения печатается и результат жадного выбора токена (как оценивали раньше).
    python evaluate.py --model out/lora            # дообученная (LoRA)
    python evaluate.py --model unsloth/Qwen3-4B-Instruct-2507   # база, для сравнения
    python evaluate.py --model out/lora --step 2   # шаг после результата инструмента: ответ текстом, а не новый вызов
"""
import argparse, collections, json, re, sys
from unsloth import FastLanguageModel
from prompt_format import CALL_THRESHOLD, answer_prefix, answer_text, parse_call, prompt_diff  # как в chat.py

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="out/lora")
ap.add_argument("--data", default="data/eval.jsonl")
ap.add_argument("--limit", type=int, default=0, help="сколько примеров (0 — все)")
ap.add_argument("--max-errors", type=int, default=20, help="сколько ошибок показать (-1 — все)")
ap.add_argument("--step", type=int, choices=(1, 2), default=1,
                help="1 — первый ответ на вопрос; 2 — ответ после результата инструмента")
a = ap.parse_args()

TOOLS = json.load(open("tools.json", encoding="utf-8"))
model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=4096, load_in_4bit=True)
FastLanguageModel.for_inference(model)
model.generation_config.max_length = None  # иначе на каждый пример предупреждение про max_length
TPL = tok.chat_template if isinstance(tok.chat_template, str) else None
PREFIX = answer_prefix(TPL)
TOOL_CALL = tok.convert_tokens_to_ids("<tool_call>")
print("Шаблон чата модели: ответ текстом " + (f"начинается с {PREFIX!r} (официальный шаблон Qwen3-2507: так модель"
                                               " видела ответы при обучении)" if PREFIX else "начинается сразу с текста"))
# chat.py собирает промпт сам (prompt_format.py) по шаблону из .gguf, то есть по этому же: проверяем, что так же
_diff = prompt_diff(lambda m, gen: tok.apply_chat_template(m, tools=TOOLS, tokenize=False, add_generation_prompt=gen),
                    [json.loads(l)["messages"] for l in list(open(a.data, encoding="utf-8"))[:5]], template=TPL)
if _diff:
    print("⚠️ prompt_format.py собирает шаблон модели не так, как transformers: chat.py будет подавать модели"
          " другой формат. Пришли это сообщение.")


def _generate(prompt, **kw):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None,
                         return_dict_in_generate=True, **kw)
    seq = out.sequences if hasattr(out, "sequences") else out
    text = tok.decode(seq[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    return text, getattr(out, "scores", None)


def generate(msgs):
    """(ответ по правилу chat.py, жадный ответ, P(<tool_call>) на первом токене)."""
    prompt = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True)
    greedy, scores = _generate(prompt, output_scores=True)
    p = scores[0][0].float().softmax(-1)[TOOL_CALL].item() if scores else None
    if p is None or p >= CALL_THRESHOLD or parse_call(greedy) is None:
        return greedy, greedy, p
    # chat.py здесь отвечает текстом: начало ответа как при обучении, токен <tool_call> запрещён
    return _generate(prompt + PREFIX, suppress_tokens=[TOOL_CALL])[0], greedy, p


def pct(p):
    return "?" if p is None else f"{p:.0%}"


def raw(text, n=150):
    text = text.split("<|im_end|>")[0].replace("\n", "⏎ ")
    return repr(text if len(text) <= n else text[:n] + "…")


norm = lambda v: str(v).lower().strip()
FREE = {"text", "task", "title"}  # свободный текст: «Купить тетрадь» и «купить тетрадь.» — одно и то же
words = lambda v: re.sub(r"[^\w]+", " ", norm(v).replace("ё", "е")).strip()


def same_args(ga, pa):
    if set(ga) != set(pa):
        return False
    for k, v in ga.items():
        if k in FREE:
            g, p = words(v), words(pa[k])
            if not (g == p or (g and p and (g in p or p in g))):
                return False
        elif norm(v) != norm(pa[k]):
            return False
    return True

n = name_ok = args_ok = g_name_ok = 0
per, hit = collections.Counter(), collections.Counter()  # разбивка по инструментам
errors = []


def error(text):  # первые --max-errors печатаем сразу, остальные только считаем
    errors.append(text)
    if a.max_errors < 0 or len(errors) <= a.max_errors:
        print(text, flush=True)


lines = list(open(a.data, encoding="utf-8"))
lines = lines[: a.limit or len(lines)]


def step2():
    """После каждого результата инструмента в датасете идёт ответ текстом (реже — следующий вызов,
    например add_exam -> make_study_plan). Подаём диалог до результата включительно: system, вопрос,
    вызов, результат (и предыдущие ходы, если они есть) — и смотрим, что модель делает дальше."""
    n = ok = texts = texts_ok = g_ok = 0
    per, hit, extra = collections.Counter(), collections.Counter(), collections.Counter()
    for i, line in enumerate(lines, 1):
        msgs = json.loads(line)["messages"]
        for j, m in enumerate(msgs[:-1]):
            if m["role"] != "tool" or msgs[j + 1]["role"] != "assistant":
                continue
            gold = msgs[j + 1]
            text, greedy, p = generate(msgs[: j + 1])
            call, gcall = parse_call(text), parse_call(greedy)
            q = [x["content"] for x in msgs[:j] if x["role"] == "user"][-1]
            key = m.get("name") or msgs[j - 1]["tool_calls"][0]["function"]["name"]
            n += 1
            per[key] += 1
            if gold.get("tool_calls"):  # в датасете здесь следующий вызов
                want = gold["tool_calls"][0]["function"]["name"]
                good = bool(call) and call.get("name") == want
                g_ok += bool(gcall) and gcall.get("name") == want
                if not good:
                    error(f"[шаг 2, инструмент] {q!r} после {key}: ждали {want}, получили {call and call.get('name')}"
                          f" (P(<tool_call>) = {pct(p)})\n  сырой ответ: {raw(text)}")
            else:  # здесь правильно — ответить текстом
                texts += 1
                good = call is None and bool(answer_text(text))
                texts_ok += good
                g_ok += gcall is None and bool(answer_text(greedy))
                if call:
                    extra[call.get("name")] += 1
                    error(f"[шаг 2, лишний вызов] {q!r} после {key}: ждали ответ текстом, получили {call.get('name')} "
                          f"{call.get('arguments')} (P(<tool_call>) = {pct(p)})\n  сырой ответ: {raw(text)}")
                elif not good:
                    error(f"[шаг 2, пустой ответ] {q!r} после {key}\n  сырой ответ: {raw(text)}")
            ok += good
            hit[key] += good
        if i % 10 == 0 or i == len(lines):
            print(f"  проверено диалогов {i}/{len(lines)}", file=sys.stderr, flush=True)

    if 0 <= a.max_errors < len(errors):
        print(f"... не показано ошибок: {len(errors) - a.max_errors} (показать все: --max-errors -1)")
    if not n:
        raise SystemExit("В данных нет результатов инструментов")
    print(f"\nШагов после результата инструмента: {n}\nВерно: {ok / n:.0%} (жадным выбором токена, как раньше: {g_ok / n:.0%})")
    if texts:
        print(f"Ответил текстом, где ждали текст: {texts_ok}/{texts} ({texts_ok / texts:.0%})")
    if n > texts:
        print(f"Сделал нужный следующий вызов: {ok - texts_ok}/{n - texts}")
    if extra:
        print("Лишние вызовы вместо ответа:", ", ".join(f"{k} {v}" for k, v in extra.most_common()))
    print("\nПо инструментам (чей результат перед ответом; верно/всего):")
    for k, v in per.most_common():
        print(f"  {k}: {hit[k]}/{v}")


if a.step == 2:
    step2()
    raise SystemExit

for line in lines:
    msgs = json.loads(line)["messages"]
    gold = msgs[2]
    text, greedy, p = generate(msgs[:2])
    call, gcall = parse_call(text), parse_call(greedy)
    n += 1
    if gold.get("tool_calls"):
        key = gold["tool_calls"][0]["function"]["name"]
    else:  # текстом: либо уточняющий вопрос (дальше в диалоге будет вызов), либо обычный ответ
        key = "уточняющий вопрос" if any(m.get("tool_calls") for m in msgs[3:]) else "без инструмента"
    per[key] += 1
    before = args_ok
    if gold.get("tool_calls"):
        g = gold["tool_calls"][0]["function"]
        g_name_ok += bool(gcall) and gcall.get("name") == g["name"]
        if call and call.get("name") == g["name"]:
            name_ok += 1
            ga, pa = g["arguments"], call.get("arguments") or {}  # у инструментов без аргументов бывает null
            if isinstance(pa, str):  # аргументы строкой с JSON внутри
                try:
                    pa = json.loads(pa)
                except ValueError:
                    pass
            if isinstance(pa, dict) and same_args(ga, pa):
                args_ok += 1
            else:
                error(f"[аргументы] {msgs[1]['content']!r}\n  ждали {ga}\n  получили {pa}")
        else:
            error(f"[инструмент] {msgs[1]['content']!r}: ждали {g['name']}, получили {call and call.get('name')}"
                  f" (P(<tool_call>) = {pct(p)})\n  сырой ответ: {raw(text)}")
    else:  # здесь правильно — ответить текстом, без вызова
        name_ok += call is None
        args_ok += call is None
        g_name_ok += gcall is None
        if call:
            error(f"[лишний вызов] {msgs[1]['content']!r}: ждали ответ текстом, получили {call.get('name')}"
                  f" (P(<tool_call>) = {pct(p)})\n  сырой ответ: {raw(text)}")
    hit[key] += args_ok - before
    if n % 10 == 0 or n == len(lines):
        print(f"  проверено {n}/{len(lines)}", file=sys.stderr, flush=True)

if 0 <= a.max_errors < len(errors):
    print(f"... не показано ошибок: {len(errors) - a.max_errors} (показать все: --max-errors -1)")

print(f"\nПримеров: {n}\nИнструмент выбран верно: {name_ok / n:.0%} (жадным выбором токена, как раньше: {g_name_ok / n:.0%})"
      f"\nИнструмент и аргументы верно: {args_ok / n:.0%}")
print("\nПо инструментам (инструмент и аргументы верно):")
for k, v in per.most_common():
    print(f"  {k}: {hit[k]}/{v}")
