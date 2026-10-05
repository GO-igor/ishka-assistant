"""QLoRA-дообучение Ишки на Mac с Apple Silicon (M1–M4) через MLX.
Unsloth и bitsandbytes работают только на видеокартах NVIDIA, поэтому для Mac отдельный скрипт.
Настройки те же, что в train_qlora.py: LoRA r=16 на всех линейных слоях, loss только на ответах
ассистента, накопление градиента 16.
    pip install mlx-lm
    python validate_dataset.py data/train.jsonl && python train_mlx.py
    python train_mlx.py --resume        # продолжить после остановки (Ctrl+C, сон, перезагрузка)
Адаптер: out/mlx_lora. Чат: mlx_lm.server --model Qwen/Qwen3-4B-Instruct-2507 --adapter-path out/mlx_lora
"""
import argparse, json, math, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.convert import convert
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="слабее Mac → Qwen/Qwen3-1.7B")
ap.add_argument("--no-quant", action="store_true", help="учить на 16-bit базе (нужно 32+ ГБ памяти)")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--accum", type=int, default=16, help="накопление градиента (примеров на шаг)")
ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--val", type=int, default=40, help="примеров eval для кривой loss")
ap.add_argument("--save-every", type=int, default=5, help="сохранять адаптер каждые N шагов")
ap.add_argument("--out", default="out/mlx_lora")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="взять первые N примеров (для проверки)")
a = ap.parse_args()

# 4-bit база (QLoRA): один раз конвертируем и кладём рядом, дальше грузим с диска
base = a.model
if not a.no_quant:
    base = os.path.join("out", "mlx_base", a.model.rstrip("/").split("/")[-1] + "-4bit")
    if not os.path.isdir(base):
        convert(a.model, mlx_path=base, quantize=True, q_bits=4, q_group_size=64)
model, tok = load(base)

TOOLS = json.load(open("tools.json", encoding="utf-8"))
ASSIST = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
USER = tok.encode("<|im_start|>user\n", add_special_tokens=False)


def encode(messages):
    """Токены диалога и маска: 1 там, где говорит ассистент (вызовы инструментов и тексты).
    Как train_on_responses_only в train_qlora.py: system, вопросы и ответы инструментов
    (они в шаблоне Qwen идут от роли user) в loss не входят."""
    text = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False)
    ids = tok.encode(text, add_special_tokens=False)
    mask, on, i, turns = np.zeros(len(ids), np.float32), False, 0, 0
    while i < len(ids):
        if ids[i:i + len(ASSIST)] == ASSIST:
            on, i, turns = True, i + len(ASSIST), turns + 1
            continue
        if ids[i:i + len(USER)] == USER:
            on = False
        mask[i] = on
        i += 1
    # каждая реплика ассистента должна попасть в loss, иначе шаблон чата не тот, что ждём
    assert turns == sum(m["role"] == "assistant" for m in messages), messages[1]["content"]
    return ids[:a.max_len], mask[:a.max_len]


def read(path, n=0):
    rows = [json.loads(l)["messages"] for l in open(path, encoding="utf-8")]
    return [encode(m) for m in rows[: n or len(rows)]]


train_set, val_set = read("data/train.jsonl", a.limit), read("data/eval.jsonl", a.val)
lens = [len(x[0]) for x in train_set]
print(f"train {len(train_set)}, eval {len(val_set)}, токенов в примере: в среднем {np.mean(lens):.0f}, max {max(lens)}")

total = int(len(train_set) * a.epochs) // a.accum * a.accum  # примеров за обучение, кратно шагу
updates = total // a.accum
order = np.concatenate([np.random.RandomState(42 + e).permutation(len(train_set))
                        for e in range(math.ceil(a.epochs))])[:total]


def batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
    """По одному примеру. loop=True → обучение (порядок order, с места остановки), иначе eval."""
    idx = order[start:] if loop else range(len(dataset))
    for j in idx:
        ids, m = dataset[j]
        n = 32 * math.ceil(len(ids) / 32) + 1  # длины кратны 32: меньше перекомпиляций mx.compile
        x, y = np.zeros((1, n), np.int32), np.zeros((1, n - 1), np.float32)
        x[0, :len(ids)], y[0, :len(ids) - 1] = ids, m[1:]
        yield mx.array(x), mx.array(y)


def loss_fn(model, batch, mask):
    logits = model(batch[:, :-1])
    ce = nn.losses.cross_entropy(logits, batch[:, 1:]) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


model.freeze()
lora = {"rank": a.rank, "scale": 2.0, "dropout": 0.0,  # scale = lora_alpha / r = 32 / 16
        "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}
linear_to_lora_layers(model, len(model.layers), lora)
print_trainable_parameters(model)

os.makedirs(a.out, exist_ok=True)
adapter, progress = os.path.join(a.out, "adapters.safetensors"), os.path.join(a.out, "progress.json")
# Конфиг в формате mlx_lm: адаптер потом грузится через mlx_lm.load / mlx_lm.server --adapter-path
json.dump({"fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": lora,
           "model": a.model, "base": base}, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=1)

start = 0
if a.resume and os.path.exists(progress):
    start = json.load(open(progress))["done"]
    model.load_weights(adapter, strict=False)
    print(f"Продолжаю с примера {start} из {total} (шаг {start // a.accum} из {updates})")
if start >= total:
    raise SystemExit("Обучение уже закончено: адаптер в " + a.out)

warmup = min(10, max(1, updates // 10))  # как warmup_steps=10 в train_qlora.py
lr = optim.join_schedules([optim.linear_schedule(0.0, a.lr, warmup),
                           optim.cosine_decay(a.lr, updates - warmup)], [warmup])
opt = optim.AdamW(learning_rate=lr, weight_decay=0.0, bias_correction=True)
opt.state["step"] = mx.array(start // a.accum, mx.uint64)  # расписание lr продолжается с того же шага


class Progress(TrainingCallback):
    def on_train_loss_report(self, info):
        done = start + info["iteration"]
        if done % (a.save_every * a.accum) == 0 or done == total:
            mx.save_safetensors(adapter, dict(tree_flatten(model.trainable_parameters())))
            json.dump({"done": done}, open(progress, "w"))
        left = (total - done) / info["iterations_per_second"]
        print(f"  шаг {done // a.accum}/{updates}, lr {info['learning_rate']:.1e}, "
              f"память {info['peak_memory']:.1f} ГБ, осталось ~{int(left // 3600)} ч {int(left % 3600 // 60)} мин")


train(model, opt, train_set, val_set, loss=loss_fn, iterate_batches=batches, training_callback=Progress(),
      args=TrainingArgs(batch_size=1, iters=total - start, val_batches=len(val_set), steps_per_report=a.accum,
                        steps_per_eval=50 * a.accum, steps_per_save=10**9, adapter_file=adapter,
                        max_seq_length=a.max_len, grad_checkpoint=True, grad_accumulation_steps=a.accum))
json.dump({"done": total}, open(progress, "w"))
print("Готово: адаптер в", a.out)
# Дальше: python evaluate.py --mlx --adapter out/mlx_lora, затем чат или экспорт (GUIDE.md, «Обучение на Mac»)
