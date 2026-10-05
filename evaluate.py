"""Оценка модели на data/eval.jsonl: правильный ли инструмент и аргументы на первом шаге.
    python evaluate.py --model out/lora            # дообученная (LoRA)
    python evaluate.py --model unsloth/Qwen3-4B-Instruct-2507   # база, для сравнения
"""
import argparse, collections, json, re
from unsloth import FastLanguageModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="out/lora")
ap.add_argument("--data", default="data/eval.jsonl")
ap.add_argument("--limit", type=int, default=0, help="сколько примеров (0 — все)")
a = ap.parse_args()

TOOLS = json.load(open("tools.json", encoding="utf-8"))
model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=4096, load_in_4bit=True)
FastLanguageModel.for_inference(model)
model.generation_config.max_length = None  # иначе на каждый пример предупреждение про max_length

def parse_call(text):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m:
        return None
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
lines = list(open(a.data, encoding="utf-8"))
for line in lines[: a.limit or len(lines)]:
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
            ga, pa = g["arguments"], call.get("arguments", {})
            if same_args(ga, pa):
                args_ok += 1
            else:
                print(f"[аргументы] {msgs[1]['content']!r}\n  ждали {ga}\n  получили {pa}")
        else:
            print(f"[инструмент] {msgs[1]['content']!r}: ждали {g['name']}, получили {call and call.get('name')}")
    else:  # здесь правильно — ответить текстом, без вызова
        name_ok += call is None
        args_ok += call is None
    hit[key] += args_ok - before

print(f"\nПримеров: {n}\nИнструмент выбран верно: {name_ok / n:.0%}\nИнструмент и аргументы верно: {args_ok / n:.0%}")
print("\nПо инструментам (инструмент и аргументы верно):")
for k, v in per.most_common():
    print(f"  {k}: {hit[k]}/{v}")
