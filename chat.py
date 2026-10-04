"""Чат с моделью в терминале: модель + настоящие инструменты из backend.py.
Нужен сервер с OpenAI-совместимым API, например:
    llama-server -m ishka-q4_k_m.gguf --jinja --port 8080          (llama.cpp)
    vllm serve out/merged --enable-auto-tool-choice --tool-call-parser hermes --port 8080
    ollama serve  (тогда --url http://localhost:11434/v1 --name ishka)
Запуск:  pip install openai && python chat.py --subgroup 2 --english "Аксёнова Н. В."
"""
import argparse, datetime as dt, json
from openai import OpenAI
from backend import Backend
from gen_dataset import system

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8080/v1")
ap.add_argument("--name", default="ishka")
ap.add_argument("--subgroup", type=int)
ap.add_argument("--english")
ap.add_argument("--adaptation", action="store_true")
ap.add_argument("--now", help="подменить текущее время: '2026-10-06 12:10'")
a = ap.parse_args()

now = dt.datetime.fromisoformat(a.now) if a.now else dt.datetime.now().replace(second=0, microsecond=0)
profile = {k: v for k, v in {"subgroup": a.subgroup, "english_teacher": a.english, "adaptation": a.adaptation}.items() if v}
backend = Backend(now, profile)
client = OpenAI(base_url=a.url, api_key="local")
TOOLS = json.load(open("tools.json", encoding="utf-8"))
msgs = [system(now, profile)]

while True:
    try:
        msgs.append({"role": "user", "content": input("ты> ")})
    except (EOFError, KeyboardInterrupt):
        break
    while True:
        r = client.chat.completions.create(model=a.name, messages=msgs, tools=TOOLS, temperature=0.3)
        m = r.choices[0].message
        msgs.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            print("Ишка>", m.content)
            break
        for c in m.tool_calls:
            args = json.loads(c.function.arguments or "{}")
            try:
                res = backend.call(c.function.name, args)
            except Exception as e:  # модель прислала аргументы, которых нет в схеме
                res = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [{c.function.name} {args}]")
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res, ensure_ascii=False)})
