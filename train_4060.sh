#!/usr/bin/env bash
# Обучение Ишки на своей видеокарте (RTX 4060 8 ГБ) одной командой. Пошагово — в TRAIN_4060.md.
#   ./train_4060.sh bench          # замер: секунд на шаг, память, сколько займут 1–4 эпохи (~15 мин)
#   ./train_4060.sh                # датасет → обучение 3 эпохи → сравнение эпох → .gguf лучшей модели
#   EPOCHS=2 ./train_4060.sh       # другое число эпох (обучение начнётся заново, в своей папке чекпоинтов)
#   LORA_R=16 ./train_4060.sh      # не хватает памяти видеокарты (тоже заново)
# Прервалось — запусти ту же команду: обучение продолжится с последнего чекпоинта, готовые проверки не пересчитываются.
set -eo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
EPOCHS="${EPOCHS:-3}"
LORA_R="${LORA_R:-32}"
if ! python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('unsloth') is None)" 2>/dev/null; then
    echo "Не найден Unsloth: сначала  source ~/ishka-venv/bin/activate  (TRAIN_4060.md, шаг 4)"; exit 1
fi
mkdir -p out

echo "== Датасет"
python db/build_db.py >/dev/null && python gen_dataset.py >/dev/null
for f in data/train.jsonl data/eval.jsonl data/eval_chat.jsonl; do
    echo -n "$f: "; python validate_dataset.py "$f" | tail -1
done

if [ "$1" = bench ]; then
    exec python train_qlora.py --plan 4060 --epochs "$EPOCHS" --r "$LORA_R" --bench
fi

echo "== Обучение: эпох $EPOCHS, ранг LoRA $LORA_R (журнал: out/train_4060.log)"
python train_qlora.py --plan 4060 --epochs "$EPOCHS" --r "$LORA_R" 2>&1 | tee -a out/train_4060.log

echo "== Сравнение эпох"
python compare_epochs.py 2>&1 | tee -a out/compare.log

echo "== Экспорт лучшей модели в .gguf"
rm -rf out/gguf
python export.py --lora "$(cat out/best.txt)" --no-merged
rm -rf out/best-lora && cp -r "$(cat out/best.txt)" out/best-lora  # адаптер лучшей: для экспорта в q8_0 и про запас
echo
echo "Готово. Лучшая модель: $(cat out/best.txt)"
ls -lh out/gguf
echo "Скопируй .gguf (и Modelfile для Ollama) на Mac: TRAIN_4060.md, шаг 8."
