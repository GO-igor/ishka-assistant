"""Чат с моделью в терминале: модель + настоящие инструменты из backend.py.
Нужен сервер с OpenAI-совместимым API, например:
    llama-server -m ishka-Q4_K_M.gguf --jinja --port 8080 -c 8192  (llama.cpp)
    vllm serve out/merged --served-model-name ishka --enable-auto-tool-choice --tool-call-parser hermes --port 8080
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
backend = Backend(now, {})
if a.subgroup or a.english or a.adaptation:
    # через set_profile, как в диалоге: «Аксёнова» или «Аксенова» превратится в полное ФИО из расписания
    r = backend.set_profile(subgroup=a.subgroup, english_teacher=a.english, adaptation=a.adaptation)
    if "error" in r:
        raise SystemExit(f"Преподаватель английского не найден, варианты: {', '.join(r.get('options', []))}")
profile = backend.profile
client = OpenAI(base_url=a.url, api_key="local")
TOOLS = json.load(open("tools.json", encoding="utf-8"))
NAMES = {t["function"]["name"] for t in TOOLS}
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
            args = c.function.arguments or "{}"
            try:
                while isinstance(args, str):  # иногда JSON приходит упакованным в строку дважды
                    args = json.loads(args or "{}")
                if c.function.name not in NAMES:
                    raise ValueError(f"нет такого инструмента: {c.function.name}")
                if not isinstance(args, dict):
                    raise ValueError("аргументы должны быть объектом")
                res = backend.call(c.function.name, args)
            except Exception as e:  # битый JSON, чужой инструмент или аргументы не по схеме
                res = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [{c.function.name} {args}]")
            msgs.append({"role": "tool", "tool_call_id": c.id,
                         "content": json.dumps(res, ensure_ascii=False, default=str)})
        msgs[0] = system(now, backend.profile)  # профиль мог поменяться через set_profile
