"""Экспорт дообученной модели.
    python export.py                     # out/gguf/*.gguf (q4_k_m, ~2.5 ГБ) + out/merged (для vLLM)
    python export.py --hf ВАШ_НИК/ishka  # ещё и загрузить на Hugging Face (нужен HF_TOKEN с правом write)
"""
import argparse, os
from unsloth import FastLanguageModel

ap = argparse.ArgumentParser()
ap.add_argument("--lora", default="out/lora")
ap.add_argument("--quant", default="q4_k_m", help="q4_k_m (лёгкая), q8_0 (точнее, тяжелее)")
ap.add_argument("--hf", help="репозиторий на Hugging Face, например vasya/ishka")
ap.add_argument("--no-merged", action="store_true")
a = ap.parse_args()

model, tok = FastLanguageModel.from_pretrained(a.lora, max_seq_length=4096, load_in_4bit=True)
if not a.no_merged:
    model.save_pretrained_merged("out/merged", tok, save_method="merged_16bit")
model.save_pretrained_gguf("out/gguf", tok, quantization_method=a.quant)
if a.hf:
    token = os.environ["HF_TOKEN"]
    model.push_to_hub(a.hf + "-lora", token=token, private=True)
    tok.push_to_hub(a.hf + "-lora", token=token, private=True)
    model.push_to_hub_gguf(a.hf + "-gguf", tok, quantization_method=a.quant, token=token, private=True)
print("Готово: out/gguf/ (и out/merged/)")
