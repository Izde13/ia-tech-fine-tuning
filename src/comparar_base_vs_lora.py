"""Comparación cualitativa rápida: modelo base vs. modelo + adaptador LoRA.

Genera respuestas a preguntas tomadas literalmente del dataset de
entrenamiento y las compara contra la respuesta real — un vistazo barato
para detectar señales obvias de memorización o de que el pipeline está roto,
antes de invertir en una evaluación formal (Fase 3) o en un re-entrenamiento
con split train/val.
"""

import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

NOMBRE_MODELO = "Qwen/Qwen2.5-1.5B-Instruct"
DIR_ADAPTADOR = "outputs/adaptador_lora"
DATASET_JSONL = "data/processed/dataset_finetuning.jsonl"

INDICES_PRUEBA = [0, 100, 3000]


def generar(modelo, tokenizer, instruction: str) -> str:
    mensajes = [{"role": "user", "content": instruction}]
    entrada_ids = tokenizer.apply_chat_template(
        mensajes, add_generation_prompt=True, return_tensors="pt"
    ).input_ids
    salida = modelo.generate(entrada_ids, max_new_tokens=60, do_sample=False)
    return tokenizer.decode(
        salida[0][entrada_ids.shape[1] :], skip_special_tokens=True
    )


def main() -> None:
    with open(DATASET_JSONL, encoding="utf-8") as f:
        dataset = [json.loads(linea) for linea in f]

    tokenizer = AutoTokenizer.from_pretrained(NOMBRE_MODELO)
    modelo_base = AutoModelForCausalLM.from_pretrained(NOMBRE_MODELO, dtype=torch.float32)
    modelo_lora = PeftModel.from_pretrained(modelo_base, DIR_ADAPTADOR)

    for idx in INDICES_PRUEBA:
        ejemplo = dataset[idx]
        print(f"\n{'=' * 70}")
        print(f"Ejemplo #{idx}")
        print(f"Instrucción: {ejemplo['instruction']}")
        print(f"Respuesta real (dataset): {ejemplo['response']}")

        with modelo_lora.disable_adapter():
            respuesta_base = generar(modelo_lora, tokenizer, ejemplo["instruction"])
        print(f"\nModelo BASE (sin LoRA): {respuesta_base!r}")

        respuesta_lora = generar(modelo_lora, tokenizer, ejemplo["instruction"])
        print(f"Modelo + LoRA:          {respuesta_lora!r}")


if __name__ == "__main__":
    main()
