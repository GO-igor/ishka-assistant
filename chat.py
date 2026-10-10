"""Чат с моделью в терминале: модель + настоящие инструменты из backend.py.
По умолчанию нужен llama-server (llama.cpp):
    llama-server -m ishka-Q4_K_M.gguf --jinja --port 8080 -c 16384
chat.py сам собирает промпт ровно как при обучении (prompt_format.py, шаблон чата берёт из .gguf) и шлёт его
в /completion, так что шаблон чата и разбор вызовов в llama.cpp не участвуют. Вызвать инструмент или ответить
текстом, решает по P(<tool_call>) на первом токене ответа (порог CALL_THRESHOLD), а не жадным выбором токена.
Запуск:  pip install jinja2 && python chat.py --subgroup 2 --english "Аксёнова Н. В."
С аккаунтом (заводится в accounts.py; профиль, Д/З и заметки хранятся в db/ishka.sqlite, работают рассылки старосты):
         python chat.py --login ivanov
Другие серверы (OpenAI-совместимый API, шаблон и вызовы разбирает сервер):
    ollama serve  → pip install openai && python chat.py --api openai --url http://localhost:11434/v1 --name ishka
    vllm serve out/merged --served-model-name ishka --enable-auto-tool-choice --tool-call-parser hermes --port 8080
                  → python chat.py --api openai
"""
import argparse, datetime as dt, getpass, json, os, unicodedata, urllib.error, urllib.request
from backend import Backend
from gen_dataset import system
from prompt_format import (TOOLS, tools_for, BROKEN, CALL_THRESHOLD, render, answer_prefix, parse_calls, answer_text,
                           call_msg, tool_msg, first_token_probs, tool_call_prob, pick_template)

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8080", help="адрес сервера (для --api openai — с /v1)")
ap.add_argument("--api", choices=("completion", "openai"), default="completion",
                help="completion — llama-server, промпт как при обучении; openai — Ollama, vLLM и т. п.")
ap.add_argument("--name", default="ishka", help="имя модели (только для --api openai)")
ap.add_argument("--subgroup", type=int)
ap.add_argument("--english")
ap.add_argument("--adaptation", action="store_true")
ap.add_argument("--now", help="подменить текущее время: '2026-10-06 12:10'")
ap.add_argument("--login", help="войти в аккаунт (пароль спросит); профиль берётся из аккаунта")
ap.add_argument("--db", help="база аккаунтов (по умолчанию db/ishka.sqlite)")
ap.add_argument("--debug", action="store_true", help="печатать сырой ответ модели")
a = ap.parse_args()

now = dt.datetime.fromisoformat(a.now) if a.now else dt.datetime.now().replace(second=0, microsecond=0)
if a.login:
    from store import DEFAULT_DB, Store
    path = a.db or os.environ.get("ISHKA_DB") or str(DEFAULT_DB)
    if not os.path.exists(path):
        raise SystemExit(f"Базы аккаунтов {path} нет: сначала заведи аккаунт, например\n"
                         f"  python accounts.py add {a.login} --group 8К51")
    store = Store(path)
    password = os.environ.get("ISHKA_PASSWORD")  # для проверок без ввода руками
    if password is None:
        password = getpass.getpass("пароль: ")
    acc = store.check_login(unicodedata.normalize("NFC", a.login), password)
    if not acc:
        raise SystemExit("Неверный логин или пароль")
    if a.subgroup or a.english or a.adaptation:
        print("  [профиль берётся из аккаунта: --subgroup, --english и --adaptation не нужны]")
    backend = Backend(now, store.profile(acc), store, acc["id"])
    print(f"Привет, {acc['name'] or acc['login']}! Группа {acc['grp']}" + (
        ", ты староста: можешь отправлять группе Д/З, мероприятия и напоминания." if acc["role"] == "starosta" else "."))
else:
    backend = Backend(now, {})
if not a.login and (a.subgroup or a.english or a.adaptation):
    # через set_profile, как в диалоге: «Аксёнова» или «Аксенова» превратится в полное ФИО из расписания
    english = unicodedata.normalize("NFC", a.english) if a.english else None  # «ё» из терминала бывает двумя символами
    r = backend.set_profile(subgroup=a.subgroup, english_teacher=english, adaptation=a.adaptation)
    if "error" in r:
        raise SystemExit(f"Преподаватель английского не найден, варианты: {', '.join(r.get('options', []))}")
NAMES = {t["function"]["name"] for t in TOOLS}
msgs = [system(now, backend.profile)]  # история в формате обучения (как в gen_dataset.py)
MAX_ROUNDS = 4    # вызовов инструментов подряд на один вопрос; в датасете больше 4 не бывает
KEEP_TURNS = 8    # сколько последних вопросов помнить, чтобы диалог не переполнил контекст сервера
N_PREDICT = 1024  # предел длины ответа в токенах; самые длинные ответы в датасете ~400
FALLBACK = "Не получилось ответить 😅 Попробуй сформулировать иначе."


class ContextOverflow(Exception):
    pass


class ServerError(Exception):
    pass


# ---------- llama-server: /completion с готовым промптом ----------
BASE = a.url.rstrip("/")
if a.api == "completion" and BASE.endswith("/v1"):
    BASE = BASE[:-3]  # старый адрес вида http://localhost:8080/v1 тоже подходит
_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # локальный сервер: системный прокси не нужен


def post(path, body=None):
    """POST с телом body (без тела — GET)."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"})
    try:
        with _http.open(req, timeout=900) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            err = json.loads(raw).get("error") or {}
        except ValueError:
            err = {}
        err = err if isinstance(err, dict) else {"message": str(err)}
        msg = err.get("message") or raw[:300] or e.reason
        if err.get("type") == "exceed_context_size_error" or "context size" in msg:
            raise ContextOverflow(msg)
        if e.code == 404:
            msg = f"{BASE}{path} не найден — это не llama-server? Для Ollama/vLLM запусти с --api openai"
        raise ServerError(f"HTTP {e.code}: {msg}")
    except (urllib.error.URLError, OSError) as e:
        raise ServerError(f"сервер {BASE} не отвечает ({getattr(e, 'reason', e)}): запущен ли llama-server в окне 1?")


def special_id(text, default):
    """Номер спецтокена в словаре модели (у Qwen3 <tool_call> = 151657)."""
    try:
        ids = post("/tokenize", {"content": text, "add_special": False, "parse_special": True}).get("tokens", [])
        return ids[0] if len(ids) == 1 and isinstance(ids[0], int) else default
    except ServerError as e:
        print(f"  [{e}]")
        return default


def model_template():
    """Шаблон чата из .gguf (с ним модель обучалась) и начало ответа текстом в этом шаблоне:
    (шаблон или None — тогда chat_template.jinja, начало ответа, ответил ли сервер)."""
    try:
        tpl = post("/props").get("chat_template")
    except ServerError:
        return None, answer_prefix(), False  # сервер ещё грузит модель или не запущен: спросим перед ответом
    tpl, prefix, why = pick_template(tpl)
    if why:
        print(f"  [{why}: беру chat_template.jinja]")
    if a.debug:
        print(f"  [шаблон: {'из .gguf' if tpl else 'chat_template.jinja'}, начало ответа {prefix!r}]")
    return tpl, prefix, True


def generate_completion(force_text):
    global MODEL_TEMPLATE, ANSWER_PREFIX, TEMPLATE_READY
    if not TEMPLATE_READY:  # при запуске сервер не ответил
        MODEL_TEMPLATE, ANSWER_PREFIX, TEMPLATE_READY = model_template()
    prompt = render(msgs, add_generation_prompt=True, template=MODEL_TEMPLATE)
    if not force_text:
        # Сначала один токен с вероятностями. Вызов — это один токен <tool_call>, а текст размазан по тысячам
        # первых слов, поэтому жадный выбор берёт вызов даже при P = 27%. Решаем по сумме (CALL_THRESHOLD).
        probs = first_token_probs(post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0,
                                                        "n_probs": 10, "cache_prompt": True}))
        if probs is not None:  # сервер без вероятностей: остаётся жадный выбор
            p = tool_call_prob(probs, TOOL_CALL_ID)
            if a.debug:
                print(f"  [P(<tool_call>) = {p:.0%}]")
            force_text = p < CALL_THRESHOLD
    body = {"prompt": prompt, "n_predict": N_PREDICT,
            "temperature": 0, "stop": ["<|im_end|>"], "cache_prompt": True}  # temperature 0: как при оценке
    if force_text:
        # Ответ текстом: начинаем его так же, как начинались ответы при обучении (у официального шаблона
        # Qwen3-2507 это '<think>\n\n</think>\n\n'), и запрещаем токен <tool_call>
        body["prompt"] = prompt + ANSWER_PREFIX
        body["logit_bias"] = [[TOOL_CALL_ID, False]]
    r = post("/completion", body)
    text = r.get("content") or ""
    if r.get("stop_type") == "limit":
        print(f"  [ответ обрезан: длиннее {N_PREDICT} токенов]" if (r.get("tokens_predicted") or 0) >= N_PREDICT
              else "  [ответ обрезан: кончился контекст сервера, запусти llama-server с -c 16384]")
    return text, parse_calls(text)


# ---------- OpenAI-совместимый API: шаблон и вызовы разбирает сервер ----------
def to_openai(history):
    """История в формате обучения -> формат OpenAI (id вызовов, аргументы строкой)."""
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


def generate_openai(force_text):
    kw = dict(model=a.name, messages=to_openai(msgs), tools=tools_for(msgs), temperature=0,
              tool_choice="none" if force_text else "auto")
    try:
        m = client.chat.completions.create(**kw).choices[0].message
    except openai.BadRequestError as e:
        if "context" in str(e):
            raise ContextOverflow(str(e))
        raise ServerError(str(e))
    except openai.APIStatusError as e:  # 500 и прочие ошибки сервера
        raise ServerError(f"HTTP {e.status_code}: {e.message}")
    except openai.APIConnectionError as e:
        raise ServerError(f"сервер {a.url} не отвечает ({e})")
    text = m.content or ""
    calls = [{"name": c.function.name, "arguments": c.function.arguments} for c in m.tool_calls or []]
    return text, calls or parse_calls(text)  # сервер без разбора вызовов вернёт <tool_call> текстом


if a.api == "openai":
    import openai
    client = openai.OpenAI(base_url=BASE if BASE.endswith("/v1") else BASE + "/v1", api_key="local")
    generate = generate_openai
else:
    generate = generate_completion
    TOOL_CALL_ID = special_id("<tool_call>", 151657)  # заодно проверка, что сервер отвечает
    MODEL_TEMPLATE, ANSWER_PREFIX, TEMPLATE_READY = model_template()


def forget_old(keep):
    """Оставляет system и последние keep вопросов пользователя (с ответами)."""
    users = [i for i, x in enumerate(msgs) if x["role"] == "user"]
    if len(users) > keep:
        del msgs[1:users[-keep]]


last_raw = [""]  # последний сырой ответ модели: покажем, если ответить не вышло


def ask(force_text):
    try:
        text, calls = generate(force_text)
    except ContextOverflow:
        forget_old(1)  # не влезло: забываем всё, кроме текущего вопроса
        print("  [начало диалога забыто: не хватило контекста сервера]")
        text, calls = generate(force_text)
    last_raw[0] = text
    if a.debug:
        print(f"  [модель: {text!r}]")
    return text, calls


def norm_args(args):
    try:
        while isinstance(args, str):  # иногда JSON приходит упакованным в строку дважды
            args = json.loads(args or "{}")
    except ValueError:
        pass
    return {} if args is None else args  # у инструментов без аргументов бывает null


def run_tools(calls):
    """Выполняет вызовы и дописывает их в историю так же, как в датасете."""
    results = []
    for name, args in calls:
        try:
            if name not in NAMES:
                raise ValueError(f"нет такого инструмента: {name}")
            if not isinstance(args, dict):
                raise ValueError("аргументы должны быть объектом")
            res = backend.call(name, args)
        except Exception as e:  # чужой инструмент или аргументы не по схеме
            res = {"error": f"{type(e).__name__}: {e}"}
        print(f"  [{name} {args}]")
        results.append(tool_msg(name, res))
    msgs.append(call_msg(calls))
    msgs.extend(results)
    msgs[0] = system(now, backend.profile)  # профиль мог поменяться через set_profile


def answer():
    """Один вопрос: вызовы инструментов, пока модель не ответит текстом. Возвращает текст или None."""
    seen, force_text = set(), False
    for step in range(MAX_ROUNDS + 1):
        force_text = force_text or step >= MAX_ROUNDS
        text, calls = ask(force_text)
        if calls and not force_text:
            calls = [(c.get("name"), norm_args(c.get("arguments"))) for c in calls]
            key = json.dumps(calls, ensure_ascii=False, sort_keys=True, default=str)
            if any(name in BROKEN for name, _ in calls):
                print("  [битый вызов инструмента: прошу ответить без инструментов]")
                force_text = True
            elif key in seen:
                # тот же вызов ещё раз: результат у модели уже есть, она зациклилась — просим ответить словами
                print(f"  [повтор {calls[0][0]}: прошу ответить без инструментов]")
                force_text = True
            else:
                seen.add(key)
                run_tools(calls)
            continue
        return answer_text(text) or None  # при force_text лишние вызовы просто отбрасываем
    return None


while True:
    try:
        q = unicodedata.normalize("NFC", input("ты> "))
    except (EOFError, KeyboardInterrupt):
        break
    msgs.append({"role": "user", "content": q})
    forget_old(KEEP_TURNS)
    last_raw[0] = ""
    try:
        text, err = answer(), None
    except (ContextOverflow, ServerError) as e:  # даже один вопрос не влез в контекст или сервер недоступен
        text, err = None, e
    except KeyboardInterrupt:  # Ctrl+C во время ответа: вопрос отменяется, чат продолжается
        text, err = None, "прервано"
        print("  [прервано]")
    if text:
        msgs.append({"role": "assistant", "content": text})
        print("Ишка>", text)
    else:
        # Неудачный ход в историю не кладём: модель не должна видеть «не получилось» как образец ответа
        del msgs[max(i for i, x in enumerate(msgs) if x["role"] == "user"):]
        print("Ишка>", FALLBACK)
        if isinstance(err, Exception):
            print(f"  [ошибка сервера: {err}]")
        elif not err:
            print(f"  [сырой ответ модели: {last_raw[0][:300]!r}]" if last_raw[0].strip()
                  else "  [модель вернула пустой ответ]")
