"""Экспорт дообученной модели.
    python export.py                     # out/gguf/*.gguf (q4_k_m, ~2.5 ГБ) + out/merged (16-bit, для vLLM)
    python export.py --no-merged         # только out/gguf/*.gguf (16-bit копия удаляется)
    python export.py --hf ВАШ_НИК/ishka  # ещё и загрузить на Hugging Face (нужен HF_TOKEN с правом write)
"""
import argparse, glob, os, shutil
from unsloth import FastLanguageModel

ap = argparse.ArgumentParser()
ap.add_argument("--lora", default="out/lora")
ap.add_argument("--quant", default="q4_k_m", help="q4_k_m (лёгкая), q8_0 (точнее, тяжелее)")
ap.add_argument("--hf", help="репозиторий на Hugging Face, например vasya/ishka")
ap.add_argument("--no-merged", action="store_true")
a = ap.parse_args()

model, tok = FastLanguageModel.from_pretrained(a.lora, max_seq_length=4096, load_in_4bit=True)

# Один проход: Unsloth сливает LoRA в 16-bit модель в merge_dir и конвертирует её в GGUF.
# .gguf он кладёт в соседнюю папку «<merge_dir>_gguf», поэтому берём пути из результата.
merge_dir = "out/tmp_merged" if a.no_merged else "out/merged"
res = model.save_pretrained_gguf(merge_dir, tok, quantization_method=a.quant, merge_is_disposable=a.no_merged)

os.makedirs("out/gguf", exist_ok=True)
files = [shutil.copy(f, "out/gguf/") for f in res["gguf_files"]]
if res.get("modelfile_location") and os.path.exists(res["modelfile_location"]):
    shutil.copy(res["modelfile_location"], "out/gguf/")  # Modelfile для Ollama
if a.no_merged:  # 16-bit копия (~8 ГБ) была нужна только для конвертации
    shutil.rmtree(res["save_directory"], ignore_errors=True)
shutil.rmtree(res["gguf_directory"], ignore_errors=True)
if not files:
    raise SystemExit("GGUF не создан — смотри сообщения Unsloth выше")

if a.hf:
    from huggingface_hub import HfApi
    token = os.environ["HF_TOKEN"]
    model.push_to_hub(a.hf + "-lora", token=token, private=True)
    tok.push_to_hub(a.hf + "-lora", token=token, private=True)
    api = HfApi(token=token)  # загружаем уже готовый .gguf, а не конвертируем заново
    api.create_repo(a.hf + "-gguf", private=True, exist_ok=True)
    for f in glob.glob("out/gguf/*"):
        api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=a.hf + "-gguf")

print("Готово:", *files, sep="\n  ")
if not a.no_merged:
    print("16-bit модель для vLLM: out/merged/")
