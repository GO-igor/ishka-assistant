"""Оценка модели на data/eval.jsonl: правильный ли инструмент и аргументы на первом шаге.
    python evaluate.py --model out/lora            # дообученная (LoRA)
    python evaluate.py --model unsloth/Qwen3-4B-Instruct-2507   # база, для сравнения
"""
import argparse, json, re
from unsloth import FastLanguageModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="out/lora")
ap.add_argument("--data", default="data/eval.jsonl")
ap.add_argument("--limit", type=int, default=100)
a = ap.parse_args()

TOOLS = json.load(open("tools.json", encoding="utf-8"))
model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=4096, load_in_4bit=True)
FastLanguageModel.for_inference(model)

def parse_call(text):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {"name": "<битый JSON>", "arguments": {}}

norm = lambda v: str(v).lower().strip()
n = name_ok = args_ok = 0
for line in list(open(a.data, encoding="utf-8"))[: a.limit]:
    msgs = json.loads(line)["messages"]
    gold = msgs[2]
    prompt = tok.apply_chat_template(msgs[:2], tools=TOOLS, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=256, do_sample=False)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    call = parse_call(text)
    n += 1
    if gold.get("tool_calls"):
        g = gold["tool_calls"][0]["function"]
        if call and call.get("name") == g["name"]:
            name_ok += 1
            ga, pa = g["arguments"], call.get("arguments", {})
            if {k: norm(v) for k, v in ga.items()} == {k: norm(v) for k, v in pa.items()}:
                args_ok += 1
            else:
                print(f"[аргументы] {msgs[1]['content']!r}\n  ждали {ga}\n  получили {pa}")
        else:
            print(f"[инструмент] {msgs[1]['content']!r}: ждали {g['name']}, получили {call and call.get('name')}")
    else:  # здесь правильно — ответить текстом, без вызова
        name_ok += call is None
        args_ok += call is None

print(f"\nПримеров: {n}\nИнструмент выбран верно: {name_ok / n:.0%}\nИнструмент и аргументы верно: {args_ok / n:.0%}")
