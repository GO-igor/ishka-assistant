"""Какая модель лучше: после 1-й, 2-й или 3-й эпохи (обучение на своей видеокарте, TRAIN_4060.md).
Для каждой папки out/epoch-N запускает те же 5 проверок evaluate.py, что ноутбук Colab в разделе 6, сводит цифры в
таблицу и пишет путь к лучшей модели в out/best.txt (её сжимает в .gguf export.py, см. train_4060.sh).
    python compare_epochs.py                        # все out/epoch-*
    python compare_epochs.py out/epoch-2 out/lora   # свои папки
    python compare_epochs.py --limit 40             # быстро и грубо: по 40 примеров на проверку
Вывод каждой проверки (с ошибками модели) сохраняется в out/eval/<папка>/: прервалось — готовое не пересчитывается.
"""
import argparse, glob, hashlib, json, os, re, subprocess, sys, time

# (файл, что проверяет, аргументы evaluate.py, строка с процентом в выводе)
CHECKS = [
    ("tool", "Вызов", [], r"Инструмент и аргументы верно: (\d+)%"),
    ("step2", "После результата", ["--step", "2"], r"\nВерно: (\d+)%"),
    ("step3", "Дальше в диалоге", ["--step", "3"], r"Верно \(инструмент и аргументы или ответ текстом\): (\d+)%"),
    ("chat", "Разговор", ["--data", "data/eval_chat.jsonl"], r"Инструмент и аргументы верно: (\d+)%"),
    ("chat3", "Разговор дальше", ["--data", "data/eval_chat.jsonl", "--step", "3"],
     r"Верно \(инструмент и аргументы или ответ текстом\): (\d+)%"),
]
REFUSALS = r"Отказал там, где ждали ответ по существу: (\d+)/(\d+)"

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="*", help="папки с LoRA (по умолчанию все out/epoch-*)")
ap.add_argument("--limit", type=int, default=0, help="сколько примеров на проверку (0 — все)")
a = ap.parse_args()
models = a.models or sorted(glob.glob("out/epoch-*[0-9]"), key=lambda d: int(re.search(r"(\d+)$", d)[1]))
if not models:
    raise SystemExit("Нет папок out/epoch-*: сначала обучение, python train_qlora.py --plan 4060")


def stamp(model):
    """Что влияет на результат: веса, данные, evaluate.py и --limit. Изменилось — проверка считается заново.
    По содержимому, а не по дате: train_4060.sh при каждом запуске заново собирает датасет и копирует out/epoch-N."""
    files = [os.path.join(model, "adapter_model.safetensors"), "evaluate.py", "prompt_format.py",
             "data/eval.jsonl", "data/eval_chat.jsonl"]
    return {"limit": a.limit, **{f: hashlib.sha1(open(f, "rb").read()).hexdigest() for f in files}}


def run(model, i, key, title, args, pattern):
    out = os.path.join("out", "eval", os.path.basename(model.rstrip("/")) + ("-limit%d" % a.limit if a.limit else ""))
    os.makedirs(out, exist_ok=True)
    path, meta = os.path.join(out, key + ".txt"), os.path.join(out, key + ".json")
    st = stamp(model)
    if os.path.exists(meta) and json.load(open(meta, encoding="utf-8")) == st:
        return open(path, encoding="utf-8").read()
    print(f"[{model}] проверка {i}/{len(CHECKS)}: {title}", flush=True)
    t = time.time()
    cmd = [sys.executable, "evaluate.py", "--model", model, *args] + (["--limit", str(a.limit)] if a.limit else [])
    with open(path, "w", encoding="utf-8") as f:  # ход проверки (stderr) виден в терминале, результат — в файле
        r = subprocess.run(cmd, stdout=f)
    text = open(path, encoding="utf-8").read()
    if r.returncode or not re.search(pattern, text):
        print(text[-3000:])
        raise SystemExit(f"Проверка не прошла: {' '.join(cmd)}\nВывод целиком: {path}. Пришли его.")
    json.dump(st, open(meta, "w", encoding="utf-8"))
    print(f"  готово за {(time.time() - t) / 60:.0f} мин, вывод с ошибками модели: {path}", flush=True)
    return text


rows = []
for m in models:
    res = {key: run(m, i, key, title, args, pat) for i, (key, title, args, pat) in enumerate(CHECKS, 1)}
    nums = {key: int(re.search(pat, res[key])[1]) for key, _, _, pat in CHECKS}
    ref = re.search(REFUSALS, res["chat"])
    rows.append(dict(model=m, **nums, refusals=f"{ref[1]}/{ref[2]}" if ref else "—", refused=int(ref[1]) if ref else 0,
                     mean=sum(nums.values()) / len(nums)))

head = ["Модель"] + [title for _, title, _, _ in CHECKS] + ["Отказы", "Среднее"]
table = [[r["model"]] + [f"{r[key]}%" for key, *_ in CHECKS] + [r["refusals"], f"{r['mean']:.1f}%"] for r in rows]
width = [max(len(x) for x in col) for col in zip(head, *table)]
print("\n" + "  ".join(h.ljust(w) for h, w in zip(head, width)))
for line in table:
    print("  ".join(x.ljust(w) for x, w in zip(line, width)))
# Лучшая — с самым высоким средним; при равенстве — меньше отказов, потом более ранняя эпоха (меньше заучила датасет)
best = max(rows, key=lambda r: (r["mean"], -r["refused"], -rows.index(r)))
print(f"\nЛучшая: {best['model']} (среднее {best['mean']:.1f}%).",
      "Ориентир: выше 90% в каждой колонке, отказов около нуля.",
      "Что именно путает модель — в out/eval/<папка>/*.txt.")
os.makedirs("out", exist_ok=True)
open("out/best.txt", "w").write(best["model"])
json.dump(rows, open("out/eval/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
