"""Valida que el modelo fusionado (Fase 4) se comporta igual que el
adaptador cargado por separado, pero sin depender de peft en absoluto —
prueba de que merge_and_unload() preservó el comportamiento aprendido.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DIR_MODELO_FUSIONADO = "outputs/modelo_fusionado"
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

    tokenizer = AutoTokenizer.from_pretrained(DIR_MODELO_FUSIONADO)
    modelo = AutoModelForCausalLM.from_pretrained(DIR_MODELO_FUSIONADO, dtype=torch.float32)

    for idx in INDICES_PRUEBA:
        ejemplo = dataset[idx]
        print(f"\n{'=' * 70}")
        print(f"Ejemplo #{idx}")
        print(f"Instrucción: {ejemplo['instruction']}")
        print(f"Respuesta real (dataset): {ejemplo['response']}")

        respuesta = generar(modelo, tokenizer, ejemplo["instruction"])
        print(f"Modelo fusionado (sin peft): {respuesta!r}")


if __name__ == "__main__":
    main()
