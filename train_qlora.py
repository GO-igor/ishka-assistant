"""QLoRA-дообучение Qwen3 под помощника ИШИТР (Unsloth + TRL).
Запуск (GPU 8–16 ГБ, например Colab T4/L4):
    pip install unsloth trl datasets
    python validate_dataset.py data/train.jsonl && python train_qlora.py
"""
import hashlib, json, os, re, zipfile
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # меньше фрагментации памяти
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from prompt_format import TEMPLATE  # chat_template.jinja

# Быстрее в ~2 раза, но слабее: BASE=unsloth/Qwen3-1.7B python train_qlora.py
BASE = os.environ.get("BASE", "unsloth/Qwen3-4B-Instruct-2507")
MAX_LEN = 4096
# Все настройки обучения здесь: от них зависит имя папки чекпоинтов (меняешь — обучение начнётся заново)
HP = dict(r=16, lora_alpha=32, lora_dropout=0.0, epochs=1, lr=2e-4, accum=16, batch=1, warmup=5,
          scheduler="cosine", seed=42,
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
TOOLS = json.load(open("tools.json", encoding="utf-8"))

# Чекпоинты: в Colab кладём на Google Диск (CKPT_ROOT), чтобы после отключения продолжить с места.
# Папка называется по отпечатку модели, настроек и данных: другой запуск не подхватит чужой чекпоинт,
# а новый коммит на GitHub, не меняющий обучение, не начнёт его заново.
root = os.environ.get("CKPT_ROOT") or os.path.dirname(os.environ.get("CKPT_DIR", "").rstrip("/")) or "out"
if root.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
    print("⚠️ Google Диск не подключён: чекпоинты будут только в out/ и пропадут при отключении Colab")
    root = "out"
h = hashlib.sha1(json.dumps([BASE, MAX_LEN, HP]).encode())
for f in ("data/train.jsonl", "tools.json", "chat_template.jinja"):  # eval на веса не влияет
    h.update(open(f, "rb").read())
CKPT = os.path.join(root, "ckpt-" + h.hexdigest()[:10])


def complete(d):
    """Чекпоинт записан целиком: при отключении посреди записи на Диске может остаться недописанный."""
    try:
        json.load(open(os.path.join(d, "trainer_state.json")))
        for f in ("optimizer.pt", "scheduler.pt"):  # .pt — zip-архив, testzip сверяет контрольные суммы
            if zipfile.ZipFile(os.path.join(d, f)).testzip() is not None:
                return False
        from safetensors.torch import load_file
        load_file(os.path.join(d, "adapter_model.safetensors"))
        return True
    except Exception:
        return False


def last_checkpoint():
    if not os.path.isdir(CKPT):
        return None
    steps = sorted((int(m[1]), m[0]) for d in os.listdir(CKPT) if (m := re.fullmatch(r"checkpoint-(\d+)", d)))
    for _, d in reversed(steps):  # с самого нового; битый пропускаем
        if complete(os.path.join(CKPT, d)):
            return os.path.join(CKPT, d)
        print("Пропускаю недописанный чекпоинт", d)
    return None


model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=MAX_LEN, load_in_4bit=True)
# Шаблон чата закрепляем (chat_template.jinja), а не берём из репозитория модели. Официальный шаблон Qwen3-2507
# начинает последний ответ ассистента с '<think>\n\n</think>\n\n', а ответы в середине диалога — без него.
# Первые две модели так и выучили, хотя сама Qwen3-2507 Instruct пустой <think> не пишет: после сжатия в .gguf
# вероятность <think> упала, и жадный выбор токена стал брать вызов инструмента. С этим шаблоном ответ текстом
# начинается сразу с текста, как у базовой модели, и без <think> в выводе на любом сервере. Вызов или текст
# chat.py всё равно решает по вероятности (CALL_THRESHOLD). Шаблон сохранится в out/lora, а оттуда попадёт в .gguf.
tok.chat_template = TEMPLATE
model = FastLanguageModel.get_peft_model(
    model, r=HP["r"], lora_alpha=HP["lora_alpha"], lora_dropout=HP["lora_dropout"],
    target_modules=HP["target_modules"], use_gradient_checkpointing="unsloth", random_state=HP["seed"])

def load(path):
    # Читаем JSON сами: load_dataset("json") превращает строки-даты в datetime и
    # дописывает в аргументы инструментов чужие ключи со значением None.
    rows = [json.loads(l)["messages"] for l in open(path, encoding="utf-8")]
    # Схема инструментов попадает в system-промпт так же, как на проде
    return Dataset.from_list([{"text": tok.apply_chat_template(m, tools=TOOLS, tokenize=False)} for m in rows])

# Для кривой loss хватит 40 примеров eval: все 170 заметно тормозят обучение.
# Качество по всем 170 потом считает evaluate.py.
ds = {"train": load("data/train.jsonl"), "eval": load("data/eval.jsonl").select(range(40))}

trainer = SFTTrainer(
    model=model, processing_class=tok,
    train_dataset=ds["train"], eval_dataset=ds["eval"],
    args=SFTConfig(
        dataset_text_field="text", max_length=MAX_LEN,
        per_device_train_batch_size=HP["batch"], gradient_accumulation_steps=HP["accum"],  # пример ~2400 токенов, batch 2 не влезает в T4
        per_device_eval_batch_size=1,
        num_train_epochs=HP["epochs"], learning_rate=HP["lr"], lr_scheduler_type=HP["scheduler"], warmup_steps=HP["warmup"],
        logging_steps=5, eval_strategy="steps", eval_steps=40, save_steps=10, save_total_limit=2,
        output_dir=CKPT, bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), seed=HP["seed"], report_to="none"))

# Loss только на ответах ассистента (вызовы инструментов + тексты).
# Ответы инструментов в шаблоне Qwen идут от роли user, поэтому тоже маскируются.
trainer = train_on_responses_only(trainer,
    instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
# Есть чекпоинт (Colab отключился посреди обучения) → продолжаем с него
last = last_checkpoint()
print("Чекпоинты:", CKPT, "\nПродолжаю с " + last if last else "\nНачинаю с нуля")
trainer.train(resume_from_checkpoint=last)

model.save_pretrained("out/lora"); tok.save_pretrained("out/lora")
# Дальше: python evaluate.py --model out/lora, затем python export.py (GGUF для llama.cpp / Ollama)
