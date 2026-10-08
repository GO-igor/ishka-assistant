"""Оценка модели на data/eval.jsonl: правильный ли инструмент и аргументы на первом шаге.
    python evaluate.py --model out/lora            # дообученная (LoRA)
    python evaluate.py --model unsloth/Qwen3-4B-Instruct-2507   # база, для сравнения
    python evaluate.py --model out/lora --step 2   # шаг после результата инструмента: ответ текстом, а не новый вызов
"""
import argparse, collections, json, re, sys
from unsloth import FastLanguageModel
from prompt_format import parse_call, prompt_diff  # тот же разбор вызовов, что в chat.py

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
# chat.py собирает промпт по chat_template.jinja: проверяем, что получается то же, что видит модель здесь
_diff = prompt_diff(lambda m, gen: tok.apply_chat_template(m, tools=TOOLS, tokenize=False, add_generation_prompt=gen),
                    [json.loads(l)["messages"] for l in list(open(a.data, encoding="utf-8"))[:5]])
if _diff:
    print("⚠️ Шаблон чата модели собирает промпт не так, как chat_template.jinja: chat.py будет подавать модели"
          " другой формат. Пришли это сообщение.")

def generate(msgs):
    prompt = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)


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

n = name_ok = args_ok = 0
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
    n = ok = texts = texts_ok = 0
    per, hit, extra = collections.Counter(), collections.Counter(), collections.Counter()
    for i, line in enumerate(lines, 1):
        msgs = json.loads(line)["messages"]
        for j, m in enumerate(msgs[:-1]):
            if m["role"] != "tool" or msgs[j + 1]["role"] != "assistant":
                continue
            gold = msgs[j + 1]
            text = generate(msgs[: j + 1])
            call = parse_call(text)
            q = [x["content"] for x in msgs[:j] if x["role"] == "user"][-1]
            key = m.get("name") or msgs[j - 1]["tool_calls"][0]["function"]["name"]
            n += 1
            per[key] += 1
            if gold.get("tool_calls"):  # в датасете здесь следующий вызов
                want = gold["tool_calls"][0]["function"]["name"]
                good = bool(call) and call.get("name") == want
                if not good:
                    error(f"[шаг 2, инструмент] {q!r} после {key}: ждали {want}, получили {call and call.get('name')}")
            else:  # здесь правильно — ответить текстом
                texts += 1
                good = call is None and bool(text.split("<|im_end|>")[0].strip())
                texts_ok += good
                if call:
                    extra[call.get("name")] += 1
                    error(f"[шаг 2, лишний вызов] {q!r} после {key}: ждали ответ текстом, "
                          f"получили {call.get('name')} {call.get('arguments')}")
                elif not good:
                    error(f"[шаг 2, пустой ответ] {q!r} после {key}")
            ok += good
            hit[key] += good
        if i % 10 == 0 or i == len(lines):
            print(f"  проверено диалогов {i}/{len(lines)}", file=sys.stderr, flush=True)

    if 0 <= a.max_errors < len(errors):
        print(f"... не показано ошибок: {len(errors) - a.max_errors} (показать все: --max-errors -1)")
    if not n:
        raise SystemExit("В данных нет результатов инструментов")
    print(f"\nШагов после результата инструмента: {n}\nВерно: {ok / n:.0%}")
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
    text = generate(msgs[:2])
    call = parse_call(text)
    n += 1
    if gold.get("tool_calls"):
        key = gold["tool_calls"][0]["function"]["name"]
    else:  # текстом: либо уточняющий вопрос (дальше в диалоге будет вызов), либо обычный ответ
        key = "уточняющий вопрос" if any(m.get("tool_calls") for m in msgs[3:]) else "без инструмента"
    per[key] += 1
    before = args_ok
    if gold.get("tool_calls"):
        g = gold["tool_calls"][0]["function"]
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
            error(f"[инструмент] {msgs[1]['content']!r}: ждали {g['name']}, получили {call and call.get('name')}")
    else:  # здесь правильно — ответить текстом, без вызова
        name_ok += call is None
        args_ok += call is None
        if call:
            error(f"[лишний вызов] {msgs[1]['content']!r}: ждали ответ текстом, получили {call.get('name')}")
    hit[key] += args_ok - before
    if n % 10 == 0 or n == len(lines):
        print(f"  проверено {n}/{len(lines)}", file=sys.stderr, flush=True)

if 0 <= a.max_errors < len(errors):
    print(f"... не показано ошибок: {len(errors) - a.max_errors} (показать все: --max-errors -1)")

print(f"\nПримеров: {n}\nИнструмент выбран верно: {name_ok / n:.0%}\nИнструмент и аргументы верно: {args_ok / n:.0%}")
print("\nПо инструментам (инструмент и аргументы верно):")
for k, v in per.most_common():
    print(f"  {k}: {hit[k]}/{v}")
