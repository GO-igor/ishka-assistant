"""QLoRA-дообучение Qwen3 под помощника ИШИТР (Unsloth + TRL).
Запуск (GPU 8–16 ГБ, например Colab T4/L4):
    pip install unsloth trl datasets
    python validate_dataset.py data/train.jsonl && python train_qlora.py
"""
import json, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # меньше фрагментации памяти
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint

# Быстрее в ~2 раза, но слабее: BASE=unsloth/Qwen3-1.7B python train_qlora.py
BASE = os.environ.get("BASE", "unsloth/Qwen3-4B-Instruct-2507")
# Чекпоинты: в Colab кладём на Google Диск, чтобы после отключения продолжить с места
CKPT = os.environ.get("CKPT_DIR", "out")
MAX_LEN = 4096
TOOLS = json.load(open("tools.json", encoding="utf-8"))

model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=MAX_LEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)

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
        per_device_train_batch_size=1, gradient_accumulation_steps=16,  # пример ~3000 токенов: batch 2 не влезает в T4
        per_device_eval_batch_size=1,
        num_train_epochs=1, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=5,
        logging_steps=5, eval_strategy="steps", eval_steps=40, save_steps=10, save_total_limit=1,
        output_dir=CKPT, bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), seed=42, report_to="none"))

# Loss только на ответах ассистента (вызовы инструментов + тексты).
# Ответы инструментов в шаблоне Qwen идут от роли user, поэтому тоже маскируются.
trainer = train_on_responses_only(trainer,
    instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
# Есть чекпоинт (Colab отключился посреди обучения) → продолжаем с него
last = get_last_checkpoint(CKPT) if os.path.isdir(CKPT) else None
if last: print("Продолжаю с", last)
trainer.train(resume_from_checkpoint=last)

model.save_pretrained("out/lora"); tok.save_pretrained("out/lora")
# Дальше: python evaluate.py --model out/lora, затем python export.py (GGUF для llama.cpp / Ollama)
