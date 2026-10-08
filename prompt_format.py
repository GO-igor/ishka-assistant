"""Промпт ровно в том виде, в каком модель видела диалоги при обучении.
train_qlora.py собирал примеры через tok.apply_chat_template(messages, tools=TOOLS): шаблон Qwen3-2507
(chat_template.jinja) + jinja2 с настройками transformers. Здесь то же самое, но без transformers —
нужен только jinja2 (pip install jinja2). Так chat.py не зависит от того, как сервер понимает шаблон.

Формат сообщений — как в gen_dataset.py:
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": ..., "arguments": {...}}}]}
    {"role": "tool", "name": ..., "content": json.dumps(результат, ensure_ascii=False)}
"""
import datetime as dt, json, os, re
import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = json.load(open(os.path.join(HERE, "tools.json"), encoding="utf-8"))
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
_template = _env.from_string(TEMPLATE)


def render(messages, tools=TOOLS, add_generation_prompt=False):
    """Текст для модели; add_generation_prompt=True дописывает '<|im_start|>assistant\\n' — дальше пишет модель."""
    return _template.render(messages=messages, tools=tools, documents=None, add_generation_prompt=add_generation_prompt)


def prompt_diff(other, dialogs, tools=TOOLS):
    """Первое место, где другой шаблон (текст jinja или функция messages -> текст) собирает промпт
    не так, как chat_template.jinja: (наш, их) или None, если на всех диалогах совпало.
    Сравниваем готовые промпты, а не текст шаблонов: шаблон может быть записан иначе, но давать то же самое."""
    if isinstance(other, str):
        t = _env.from_string(other)
        other = lambda m, gen: t.render(messages=m, tools=tools, documents=None, add_generation_prompt=gen)
    for m in dialogs:
        for part, gen in ((m, False), (m[:-1], True)):
            mine, theirs = render(part, tools, gen), other(part, gen)
            if mine != theirs:
                return mine, theirs
    return None


def call_msg(calls):
    """Ход ассистента с вызовами [(имя, аргументы)] — как tc() в gen_dataset.py: content "", аргументы словарём."""
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": name, "arguments": args}} for name, args in calls]}


def tool_msg(name, result):
    """Результат инструмента — как tr() в gen_dataset.py."""
    return {"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False, default=str)}


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
