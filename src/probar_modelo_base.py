"""Prueba de carga de un modelo base candidato en esta máquina (CPU, sin CUDA).

Mide tiempo de descarga/carga, RAM usada y genera una respuesta corta a un
prompt de prueba, para decidir el modelo base de la Fase 0 con datos reales
en vez de solo por specs en una tabla.
"""

import argparse
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_PRUEBA = "¿Quién gana el Mundial 2026?"


def ram_usada_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def probar_modelo(nombre_modelo: str) -> None:
    print(f"\n=== {nombre_modelo} ===")

    ram_antes = ram_usada_mb()
    inicio = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(
        nombre_modelo,
        dtype=torch.float32,
    )

    tiempo_carga = time.perf_counter() - inicio
    ram_despues = ram_usada_mb()

    n_parametros = sum(p.numel() for p in modelo.parameters())

    print(f"Tiempo de carga: {tiempo_carga:.1f}s")
    print(f"RAM usada por el proceso: {ram_despues - ram_antes:.0f} MB (delta)")
    print(f"Parámetros totales: {n_parametros:,}")

    mensajes = [{"role": "user", "content": PROMPT_PRUEBA}]
    entrada_ids = tokenizer.apply_chat_template(
        mensajes, add_generation_prompt=True, return_tensors="pt"
    ).input_ids

    inicio_gen = time.perf_counter()
    salida = modelo.generate(entrada_ids, max_new_tokens=60, do_sample=False)
    tiempo_gen = time.perf_counter() - inicio_gen

    respuesta = tokenizer.decode(
        salida[0][entrada_ids.shape[1] :], skip_special_tokens=True
    )

    print(f"Tiempo de generación (60 tokens): {tiempo_gen:.1f}s")
    print(f"Respuesta: {respuesta!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "modelo",
        nargs="?",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Nombre del modelo en HuggingFace Hub",
    )
    args = parser.parse_args()
    probar_modelo(args.modelo)
