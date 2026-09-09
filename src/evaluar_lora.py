"""Evaluación cuantitativa del adaptador LoRA sobre el set de validación
completo (659 ejemplos que el modelo nunca vio en entrenamiento).

A diferencia de comparar_base_vs_lora.py (3 ejemplos sueltos, inspección
cualitativa), este script mide qué fracción del set de validación el modelo
responde bien, parcialmente bien, o mal — parseando los 3 campos de la
respuesta (respuesta_participante, resultado_oficial, puntos) en vez de
comparar el texto completo carácter por carácter, porque el modelo puede
acertar el contenido con una redacción ligeramente distinta.
"""

import argparse
import json
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

NOMBRE_MODELO = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_JSONL = "data/processed/dataset_finetuning.jsonl"

# Mismo patrón de respuesta generado por construir_par() en preparar_dataset.py.
_PATRON_RESPUESTA = re.compile(
    r"^Respondió (.+?)\. El resultado oficial fue (.+?), por lo que obtuvo (\d+) puntos?\.?$"
)


def parsear_respuesta(texto: str) -> tuple[str, str, str] | None:
    coincidencia = _PATRON_RESPUESTA.match(texto.strip())
    if not coincidencia:
        return None
    respuesta, resultado, puntos = coincidencia.groups()
    return respuesta.strip().lower(), resultado.strip().lower(), puntos.strip()


def clasificar(real: str, generada: str) -> str:
    campos_real = parsear_respuesta(real)
    campos_generada = parsear_respuesta(generada)

    if campos_real is None or campos_generada is None:
        return "incorrecto"

    coincidencias = sum(r == g for r, g in zip(campos_real, campos_generada))
    if coincidencias == 3:
        return "correcto"
    if coincidencias > 0:
        return "parcial"
    return "incorrecto"


def tipo_pregunta(instruction: str) -> str:
    return "marcador" if "Marcador" in instruction else "otro"


def generar(modelo, tokenizer, instruction: str) -> str:
    mensajes = [{"role": "user", "content": instruction}]
    entrada_ids = tokenizer.apply_chat_template(
        mensajes, add_generation_prompt=True, return_tensors="pt"
    ).input_ids
    salida = modelo.generate(entrada_ids, max_new_tokens=60, do_sample=False)
    return tokenizer.decode(
        salida[0][entrada_ids.shape[1] :], skip_special_tokens=True
    )


def main(dir_adaptador: str, salida_json: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(NOMBRE_MODELO)
    modelo_base = AutoModelForCausalLM.from_pretrained(NOMBRE_MODELO, dtype=torch.float32)
    modelo = PeftModel.from_pretrained(modelo_base, dir_adaptador)
    modelo.eval()

    # Mismo split (seed=42) usado en entrenar_lora.py — reproduce exactamente
    # el mismo set de validación sin que el modelo lo haya visto en gradiente.
    dataset = load_dataset("json", data_files=DATASET_JSONL, split="train")
    dataset_val = dataset.train_test_split(test_size=0.1, seed=42)["test"]
    print(f"Evaluando sobre {len(dataset_val)} ejemplos de validación\n")

    resultados = []
    conteo = {"correcto": 0, "parcial": 0, "incorrecto": 0}
    conteo_por_tipo = {
        "marcador": {"correcto": 0, "parcial": 0, "incorrecto": 0},
        "otro": {"correcto": 0, "parcial": 0, "incorrecto": 0},
    }

    for i, ejemplo in enumerate(dataset_val):
        generada = generar(modelo, tokenizer, ejemplo["instruction"])
        clase = clasificar(ejemplo["response"], generada)
        tipo = tipo_pregunta(ejemplo["instruction"])

        conteo[clase] += 1
        conteo_por_tipo[tipo][clase] += 1
        resultados.append(
            {
                "instruction": ejemplo["instruction"],
                "response_real": ejemplo["response"],
                "response_generada": generada,
                "clase": clase,
                "tipo": tipo,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(dataset_val)} procesados...")

    total = len(dataset_val)
    print("\n=== Resultado global ===")
    for clase in ("correcto", "parcial", "incorrecto"):
        n = conteo[clase]
        print(f"{clase:12s}: {n:4d} ({100 * n / total:.1f}%)")

    print("\n=== Por tipo de pregunta ===")
    for tipo, sub in conteo_por_tipo.items():
        sub_total = sum(sub.values())
        if sub_total == 0:
            continue
        print(f"\n{tipo} ({sub_total} ejemplos):")
        for clase in ("correcto", "parcial", "incorrecto"):
            n = sub[clase]
            print(f"  {clase:12s}: {n:4d} ({100 * n / sub_total:.1f}%)")

    with open(salida_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nDetalle completo guardado en {salida_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir-adaptador", default="outputs/adaptador_lora")
    parser.add_argument("--salida-json", default="data/processed/evaluacion_lora.json")
    args = parser.parse_args()
    main(dir_adaptador=args.dir_adaptador, salida_json=args.salida_json)
