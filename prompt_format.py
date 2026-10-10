"""Промпт ровно в том виде, в каком модель видела диалоги при обучении.
train_qlora.py собирает примеры через tok.apply_chat_template(messages, tools=tools_for(messages)): шаблон чата + jinja2
с настройками transformers. Здесь то же самое, но без transformers — нужен только jinja2 (pip install jinja2).

Шаблонов два:
- chat_template.jinja — его train_qlora.py закрепляет для обучения (TEMPLATE ниже);
- официальный шаблон Qwen3-2507 из репозитория модели — с ним шло обучение до того, как шаблон закрепили.
  Он отличается одним: последний ответ ассистента после вопроса начинает с '<think>\n\n</think>\n\n'.
Шаблон, с которым модель обучалась, лежит внутри .gguf (llama-server отдаёт его в /props), поэтому
chat.py берёт его оттуда: template=... в render() и answer_prefix().

Формат сообщений — как в gen_dataset.py:
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": ..., "arguments": {...}}}]}
    {"role": "tool", "name": ..., "content": json.dumps(результат, ensure_ascii=False)}
"""
import datetime as dt, json, math, os, re
import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = json.load(open(os.path.join(HERE, "tools.json"), encoding="utf-8"))
# Рассылки группе видит в промпте только староста: у студента промпт на ~500 токенов короче (обучение быстрее,
# первый ответ тоже), а модель не может даже попытаться разослать что-то группе. Права всё равно проверяет backend.py.
GROUP_TOOLS = {t["function"]["name"] for t in TOOLS if t["function"]["name"].startswith("group_")}
STUDENT_TOOLS = [t for t in TOOLS if t["function"]["name"] not in GROUP_TOOLS]
STAROSTA_MARK = "; староста группы."  # так gen_dataset.system() заканчивает профиль старосты в системном промпте
TEMPLATE = open(os.path.join(HERE, "chat_template.jinja"), encoding="utf-8").read()


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # как в transformers: обычный json.dumps, кириллица и HTML-символы не экранируются
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise(message):
    raise jinja2.exceptions.TemplateError(message)


# Окружение как в transformers (utils/chat_template_utils.py, _compile_jinja_template)
_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
_env.filters["tojson"] = _tojson
_env.globals["raise_exception"] = _raise
_env.globals["strftime_now"] = lambda fmt: dt.datetime.now().strftime(fmt)
_compiled = {}

# Вызов инструмента начинается одним токеном <tool_call>, а ответ текстом — с любого из тысяч слов (или с <think>).
# Поэтому жадный выбор «самого вероятного токена» берёт вызов, даже когда за него 27%, а за текст 73%.
# chat.py, diag_server.py и evaluate.py решают по сумме: вызов, только если P(<tool_call>) не меньше порога.
CALL_THRESHOLD = 0.5


def _compile(template):
    text = TEMPLATE if template is None else template
    if text not in _compiled:
        _compiled[text] = _env.from_string(text)
    return _compiled[text]


def tools_for(messages):
    """Инструменты в промпте этого диалога: все для старосты, без рассылок группе для студента.
    Роль берётся из системного промпта, поэтому train_qlora.py, evaluate.py и chat.py выбирают одинаково."""
    first = messages[0] if messages else {}
    starosta = first.get("role") == "system" and (first.get("content") or "").endswith(STAROSTA_MARK)
    return TOOLS if starosta else STUDENT_TOOLS


def render(messages, tools=None, add_generation_prompt=False, template=None):
    """Текст для модели; add_generation_prompt=True дописывает '<|im_start|>assistant\\n' — дальше пишет модель.
    tools по умолчанию — tools_for(messages). template — текст другого шаблона (например, из .gguf),
    по умолчанию chat_template.jinja."""
    tools = tools_for(messages) if tools is None else tools
    return _compile(template).render(messages=messages, tools=tools, documents=None,
                                     add_generation_prompt=add_generation_prompt)


def answer_prefix(template=None):
    """С чего шаблон начинает последний ответ ассистента текстом: официальный шаблон Qwen3-2507 —
    с '<think>\\n\\n</think>\\n\\n', chat_template.jinja — ни с чего. Так модель видела ответы при обучении,
    и с этого же chat.py начинает ответ, когда модель должна ответить текстом."""
    mark = "ОТВЕТ_7f3a"
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    head = render(msgs, add_generation_prompt=True, template=template)
    full = render(msgs + [{"role": "assistant", "content": mark}], template=template)
    if not full.startswith(head) or mark not in full:
        return ""
    return full[len(head):full.index(mark)]


def prompt_diff(other, dialogs, tools=None, template=None, gen_only=False):
    """Первое место, где другой шаблон (текст jinja или функция (messages, gen) -> текст) собирает промпт
    не так, как template (по умолчанию chat_template.jinja): (наш, их) или None, если на всех диалогах совпало.
    Сравниваем готовые промпты, а не текст шаблонов: шаблон может быть записан иначе, но давать то же самое.
    gen_only=True — только промпты, которые видит модель при ответе (диалог до каждого хода ассистента)."""
    if isinstance(other, str):
        t = _compile(other)
        other = lambda m, gen: t.render(messages=m, tools=tools_for(m) if tools is None else tools, documents=None,
                                        add_generation_prompt=gen)
    for m in dialogs:
        parts = [(m[:k], True) for k, x in enumerate(m) if x["role"] == "assistant"]
        for part, gen in parts if gen_only else [(m, False)] + parts:
            mine, theirs = render(part, tools, gen, template), other(part, gen)
            if mine != theirs:
                return mine, theirs
    return None


def first_token_probs(resp):
    """Вероятности первого токена из ответа llama-server /completion с n_probs: [(id, токен, p)] по убыванию p
    или None, если сервер их не прислал. Это сырые вероятности модели: до сэмплера и до logit_bias."""
    cp = resp.get("completion_probabilities") or []
    if not cp:
        return None
    out = [(x.get("id"), x.get("token") or "", math.exp(x["logprob"])) for x in cp[0].get("top_logprobs") or []]
    out += [(x.get("id"), x.get("token") or x.get("tok_str") or "", x["prob"])  # старые версии llama.cpp
            for x in cp[0].get("top_probs") or cp[0].get("probs") or []]
    return sorted(out, key=lambda x: -x[2])


def tool_call_prob(probs, tool_call_id=151657):
    """P(<tool_call>) среди первых токенов; если его нет в топе — 0."""
    return sum(p for i, t, p in probs if i == tool_call_id or t == "<tool_call>")


def call_msg(calls):
    """Ход ассистента с вызовами [(имя, аргументы)] — как tc() в gen_dataset.py: content "", аргументы словарём."""
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": name, "arguments": args}} for name, args in calls]}


def tool_msg(name, result):
    """Результат инструмента — как tr() в gen_dataset.py."""
    return {"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False, default=str)}


# Диалог с вызовом и результатом инструмента: на нём проверяем, что шаблон из .gguf собирает промпт как при обучении
_CHECK = [{"role": "system", "content": "s"}, {"role": "user", "content": "что завтра?"},
          call_msg([("get_schedule", {"day": "tomorrow"})]), tool_msg("get_schedule", {"pairs": []}),
          {"role": "assistant", "content": "Завтра пар нет."}, {"role": "user", "content": "спасибо"},
          {"role": "assistant", "content": "Обращайся!"}]


def pick_template(tpl):
    """Шаблон из .gguf (текст из /props llama-server), если модель видела с ним те же промпты, что с
    chat_template.jinja (официальный шаблон Qwen3-2507 отличается только началом ответа). Иначе — None,
    и промпт собирается по chat_template.jinja. Возвращает (шаблон или None, начало ответа текстом, почему не он)."""
    if not tpl:
        return None, answer_prefix(), "сервер не отдал шаблон чата"
    try:
        if prompt_diff(tpl, [_CHECK], gen_only=True):
            return None, answer_prefix(), "шаблон чата из .gguf собирает промпт не так, как при обучении"
        return tpl, answer_prefix(tpl), None
    except Exception as e:
        return None, answer_prefix(), f"шаблон чата из .gguf не собирается ({type(e).__name__}: {e})"


_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
BROKEN = ("<битый вызов>", "<битый JSON>")


def parse_calls(text):
    """Все вызовы из ответа модели: [{"name": ..., "arguments": ...}]. Разбор как в evaluate.py:
    <tool_call> без нормального JSON — битый вызов, а не ответ текстом."""
    calls = []
    for m in _CALL.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            calls.append({"name": BROKEN[1], "arguments": {}})
    if not calls and "<tool_call>" in text:
        calls.append({"name": BROKEN[0], "arguments": {}})
    return calls


def parse_call(text):
    """Первый вызов или None (так считает evaluate.py)."""
    calls = parse_calls(text)
    return calls[0] if calls else None


def answer_text(text):
    """Текст ответа: без вызовов инструментов и без <think>…</think> (в обучении его не было, но вдруг)."""
    text = text.split("<tool_call>")[0]
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.replace("<|im_end|>", "").strip()
