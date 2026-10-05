"""Оценка модели на data/eval.jsonl: правильный ли инструмент и аргументы на первом шаге.
    python evaluate.py --model out/lora            # дообученная (LoRA)
    python evaluate.py --model unsloth/Qwen3-4B-Instruct-2507   # база, для сравнения
"""
import argparse, collections, json, re, sys
from unsloth import FastLanguageModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="out/lora")
ap.add_argument("--data", default="data/eval.jsonl")
ap.add_argument("--limit", type=int, default=0, help="сколько примеров (0 — все)")
ap.add_argument("--max-errors", type=int, default=20, help="сколько ошибок показать (-1 — все)")
a = ap.parse_args()

TOOLS = json.load(open("tools.json", encoding="utf-8"))
model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=4096, load_in_4bit=True)
FastLanguageModel.for_inference(model)
model.generation_config.max_length = None  # иначе на каждый пример предупреждение про max_length

def parse_call(text):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m:  # попытка вызова без нормального JSON — это ошибка, а не «ответил текстом»
        return {"name": "<битый вызов>", "arguments": {}} if "<tool_call>" in text else None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {"name": "<битый JSON>", "arguments": {}}

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
for line in lines:
    msgs = json.loads(line)["messages"]
    gold = msgs[2]
    prompt = tok.apply_chat_template(msgs[:2], tools=TOOLS, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
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
