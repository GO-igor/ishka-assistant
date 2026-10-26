"""QLoRA-дообучение Qwen3 под помощника ИШИТР (Unsloth + TRL).
Запуск (GPU 8–16 ГБ, например Colab T4/L4):
    pip install unsloth trl datasets
    python validate_dataset.py data/train.jsonl && python train_qlora.py
Своя видеокарта RTX 4060 8 ГБ (пошагово в TRAIN_4060.md, всё сразу: ./train_4060.sh):
    python train_qlora.py --plan 4060 --bench   # замер: секунд на шаг, память, сколько займут 1–4 эпохи
    python train_qlora.py --plan 4060           # 3 эпохи, модель после каждой — out/epoch-N (сравнивает compare_epochs.py)
"""
import argparse, glob, hashlib, json, math, os, re, shutil, time, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--plan", choices=("colab", "4060"), default="colab",
                help="colab — 1 эпоха на T4; 4060 — своя видеокарта 8 ГБ, несколько эпох (TRAIN_4060.md)")
ap.add_argument("--epochs", type=int, default=3, help="эпох для --plan 4060")
ap.add_argument("--r", type=int, default=32, help="ранг LoRA для --plan 4060 (не хватает памяти — 16)")
ap.add_argument("--bench", action="store_true", help="замерить скорость и память на 5 шагах и выйти; ничего не сохраняет")
a = ap.parse_args()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # меньше фрагментации памяти
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from prompt_format import TEMPLATE, tools_for  # chat_template.jinja; инструменты по роли (рассылки — только старосте)

# Быстрее в ~2 раза, но слабее: BASE=unsloth/Qwen3-1.7B python train_qlora.py
BASE = os.environ.get("BASE", "unsloth/Qwen3-4B-Instruct-2507")
MAX_LEN = 4096
# Все настройки обучения здесь: от них зависит имя папки чекпоинтов (меняешь — обучение начнётся заново)
HP = dict(r=16, lora_alpha=32, lora_dropout=0.0, epochs=1, lr=2e-4, accum=16, batch=1, warmup=5,
          scheduler="cosine", seed=42,
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
# Своя RTX 4060 8 ГБ (TRAIN_4060.md): времени до 60 ч, поэтому несколько эпох, а после каждой — своя модель out/epoch-N,
# лучшую выбирает compare_epochs.py. Ранг LoRA вдвое больше, шаг обучения вдвое меньше, чем за одну эпоху в Colab.
# Оптимизатор 8-битный: экономит ~0.4 ГБ памяти видеокарты.
if a.plan == "4060":
    HP = dict(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, epochs=a.epochs, lr=1e-4, accum=16, batch=1, warmup=20,
              scheduler="cosine", seed=42, optim="adamw_8bit", target_modules=HP["target_modules"])

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
# какие инструменты попадают в промпт каждого примера (рассылки — только у старосты, prompt_format.tools_for)
h.update(json.dumps([[t["function"]["name"] for t in tools_for(json.loads(l)["messages"])]
                     for l in open("data/train.jsonl", encoding="utf-8")]).encode())
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


if a.bench:
    free0, total = torch.cuda.mem_get_info()  # сколько памяти видеокарты занято до загрузки модели (экран, браузер)
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
    # Схема инструментов попадает в system-промпт так же, как на проде (у студента без рассылок группе)
    return Dataset.from_list([{"text": tok.apply_chat_template(m, tools=tools_for(m), tokenize=False)} for m in rows])

# Для кривой loss хватит 40 примеров eval: все 321 заметно тормозят обучение.
# Качество по всем 321 потом считает evaluate.py.
ds = {"train": load("data/train.jsonl"), "eval": load("data/eval.jsonl").select(range(40))}

cfg = dict(
    dataset_text_field="text", max_length=MAX_LEN,
    per_device_train_batch_size=HP["batch"], gradient_accumulation_steps=HP["accum"],  # пример ~2800 токенов, batch 2 не влезает в T4
    per_device_eval_batch_size=1,
    num_train_epochs=HP["epochs"], learning_rate=HP["lr"], lr_scheduler_type=HP["scheduler"], warmup_steps=HP["warmup"],
    logging_steps=5, eval_strategy="steps", eval_steps=40, save_steps=10, save_total_limit=2,
    output_dir=CKPT, bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), seed=HP["seed"], report_to="none")
if "optim" in HP:
    cfg["optim"] = HP["optim"]
if a.plan == "4060":
    cfg["save_steps"] = 20  # шаг на 4060 короче, чем на T4: чекпоинт примерно раз в полчаса-час
BENCH_STEPS = 5
if a.bench:  # первый шаг медленнее (подготовка), считаем по следующим четырём
    cfg.update(max_steps=BENCH_STEPS, eval_strategy="no", save_strategy="no", output_dir="out/bench")


def save_adapter(d):
    """Адаптер и токенизатор (в нём шаблон чата) — сначала во временную папку: оборвалось на записи — папки d нет."""
    shutil.rmtree(d + ".tmp", ignore_errors=True)
    model.save_pretrained(d + ".tmp"); tok.save_pretrained(d + ".tmp")
    shutil.rmtree(d, ignore_errors=True)
    os.replace(d + ".tmp", d)


class SaveEpoch(TrainerCallback):
    """Модель после каждой эпохи — CKPT/epoch-N, и сразу чекпоинт: после сбоя эпоха не считается заново.
    Если оборвалось раньше, чем модель эпохи записалась, Trainer дойдёт до конца эпохи снова и запишет её."""
    def on_epoch_end(self, args, state, control, **kw):
        save_adapter(os.path.join(CKPT, f"epoch-{round(state.epoch)}"))
        print(f"\nМодель после эпохи {round(state.epoch)} сохранена", flush=True)
        control.should_save = True
        return control


class Timer(TrainerCallback):
    """--bench: время каждого шага (16 примеров)."""
    def __init__(self):
        self.ends = []

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.ends.append(time.perf_counter())


callbacks = [Timer()] if a.bench else [SaveEpoch()] if a.plan == "4060" else []
trainer = SFTTrainer(
    model=model, processing_class=tok,
    train_dataset=ds["train"], eval_dataset=ds["eval"], callbacks=callbacks, args=SFTConfig(**cfg))

# Loss только на ответах ассистента (вызовы инструментов + тексты).
# Ответы инструментов в шаблоне Qwen идут от роли user, поэтому тоже маскируются.
trainer = train_on_responses_only(trainer,
    instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")


def bench():
    """Секунд на шаг, пик памяти видеокарты и прогноз: сколько займут 1–4 эпохи вместе с оценкой и экспортом."""
    trainer.train()
    ends = callbacks[0].ends
    step = (ends[-1] - ends[0]) / (len(ends) - 1)
    steps = math.ceil(len(ds["train"]) / (HP["batch"] * HP["accum"]))
    epoch_h = step * steps / 3600 * 1.05  # +5%: eval-loss каждые 40 шагов и чекпоинты
    gb = 2 ** 30
    peak, others = torch.cuda.max_memory_reserved() / gb, (total - free0) / gb
    print(f"\n=== Замер ({torch.cuda.get_device_name(0)}, ранг LoRA {HP['r']}) ===")
    print(f"Шаг (16 примеров): {step:.0f} с, шагов в эпохе: {steps}, эпоха: {epoch_h:.1f} ч (±10%: длина примеров разная)")
    print(f"Память видеокарты: обучение {peak:.1f} ГБ + занято до запуска {others:.1f} ГБ (экран, браузер, служебное CUDA)"
          f" из {total / gb:.1f} ГБ")
    if peak + others > total / gb - 0.5:
        print("⚠️ Впритык: длинный пример может не влезть. Закрой браузер и игры; не помогло — запускай с рангом 16"
              " (./train_4060.sh: LORA_R=16)")
    EVAL_H, EXPORT_H = 1.5, 0.5  # оценка одной модели (5 проверок evaluate.py) и экспорт в .gguf — примерно
    print("\nЭпох | обучение | + оценка каждой эпохи и экспорт | итого")
    for e in range(1, 5):
        t = e * epoch_h + e * EVAL_H + EXPORT_H
        mark = "✓" if t <= 50 else "впритык (лимит 60 ч)" if t <= 60 else "✗ больше 60 ч"
        print(f"  {e}  | {e * epoch_h:5.1f} ч  | {e * EVAL_H + EXPORT_H:4.1f} ч | {t:5.1f} ч {mark}")
    fit = [e for e in range(1, 4) if e * (epoch_h + EVAL_H) + EXPORT_H <= 50]
    best = max(fit, default=1)
    print(f"\nСовет: {best} {'эпоха' if best == 1 else 'эпохи'}, запуск: "
          + ("./train_4060.sh" if best == 3 else f"EPOCHS={best} ./train_4060.sh"))
    if not fit:
        print("⚠️ Даже одна эпоха с оценкой дольше 50 ч: проверь, что обучение идёт на видеокарте (nvidia-smi)")
    shutil.rmtree("out/bench", ignore_errors=True)


if a.bench:
    bench()
    raise SystemExit
# Есть чекпоинт (Colab отключился посреди обучения) → продолжаем с него
last = last_checkpoint()
print("Чекпоинты:", CKPT, "\nПродолжаю с " + last if last else "\nНачинаю с нуля")
trainer.train(resume_from_checkpoint=last)

model.save_pretrained("out/lora"); tok.save_pretrained("out/lora")
if a.plan == "4060":
    last_epoch = os.path.join(CKPT, f"epoch-{HP['epochs']}")
    if not os.path.isdir(last_epoch):
        save_adapter(last_epoch)
    # out/epoch-N — модели этого обучения после каждой эпохи (от прошлого обучения убираем); out/lora — последняя
    for d in glob.glob("out/epoch-*"):
        shutil.rmtree(d)
    for n in range(1, HP["epochs"] + 1):
        if os.path.isdir(os.path.join(CKPT, f"epoch-{n}")):
            shutil.copytree(os.path.join(CKPT, f"epoch-{n}"), f"out/epoch-{n}")
        else:
            print(f"⚠️ Нет модели после эпохи {n}: сравнивать будет не с чем, пришли этот вывод")
    print("Модели после эпох:", *sorted(glob.glob("out/epoch-*")), "\nДальше: python compare_epochs.py")
# Дальше: python evaluate.py --model out/lora, затем python export.py (GGUF для llama.cpp / Ollama)
