"""Проверка размеченного датасета: python validate_dataset.py data/train.jsonl"""
import json, sys, collections

tools = {t["function"]["name"]: t["function"]["parameters"] for t in json.load(open("tools.json", encoding="utf-8"))}
errors, stats = 0, collections.Counter()

for ln, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    def err(msg):
        global errors; errors += 1; print(f"строка {ln}: {msg}")
    try:
        msgs = json.loads(line)["messages"]
    except Exception as e:
        err(f"невалидный JSON: {e}"); continue
    if msgs[0]["role"] != "system": err("первое сообщение должно быть system")
    if msgs[-1]["role"] != "assistant" or not msgs[-1].get("content"): err("последнее сообщение — текстовый ответ ассистента")
    pending = None
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                name, args = c["function"]["name"], c["function"]["arguments"]
                stats[name] += 1
                if name not in tools: err(f"неизвестный инструмент {name}"); continue
                props, req = tools[name]["properties"], tools[name].get("required", [])
                for k in req:
                    if k not in args: err(f"{name}: нет обязательного аргумента {k}")
                for k, v in args.items():
                    if k not in props: err(f"{name}: лишний аргумент {k}")
                    elif "enum" in props[k] and v not in props[k]["enum"]: err(f"{name}.{k}={v} не из enum")
                pending = name
        elif m["role"] == "tool":
            if pending is None: err(f"сообщение tool без вызова (#{i})")
            elif m.get("name") != pending: err(f"ответ {m.get('name')} на вызов {pending}")
            pending = None
        elif m["role"] == "assistant":
            stats["text_answer"] += 1
    if not any(m.get("tool_calls") for m in msgs): stats["без инструментов"] += 1

print("Статистика:", dict(stats))
print("Ошибок:", errors)
sys.exit(1 if errors else 0)
