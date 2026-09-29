"""Fusiona el adaptador LoRA ganador (Fase 3: MLP + r=16) con el modelo base,
usando merge_and_unload() de peft.

A diferencia de cargar el adaptador por separado (PeftModel.from_pretrained,
usado en comparar_base_vs_lora.py y evaluar_lora.py), el modelo resultante
aquí es un AutoModelForCausalLM normal: no necesita peft para cargarse, y ya
no calcula el ajuste LoRA al vuelo en cada forward pass, porque quedó sumado
de forma permanente a los pesos originales.

Trade-off: el modelo fusionado pesa como el modelo base completo (~GBs), no
como el adaptador (~MBs) — ver docs/fase-4-merge-portafolio.md.
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

NOMBRE_MODELO = "Qwen/Qwen2.5-1.5B-Instruct"
DIR_ADAPTADOR_DEFECTO = "outputs/adaptador_lora_mlp_r16"
DIR_SALIDA_DEFECTO = "outputs/modelo_fusionado"


def tamano_directorio_mb(directorio: str) -> float:
    total_bytes = sum(f.stat().st_size for f in Path(directorio).rglob("*") if f.is_file())
    return total_bytes / (1024 * 1024)


def fusionar(dir_adaptador: str, dir_salida: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(NOMBRE_MODELO)
    modelo_base = AutoModelForCausalLM.from_pretrained(NOMBRE_MODELO, dtype=torch.float32)
    modelo_lora = PeftModel.from_pretrained(modelo_base, dir_adaptador)

    print("Fusionando adaptador con el modelo base (merge_and_unload)...")
    modelo_fusionado = modelo_lora.merge_and_unload()

    if Path(dir_salida).exists():
        shutil.rmtree(dir_salida)

    modelo_fusionado.save_pretrained(dir_salida)
    tokenizer.save_pretrained(dir_salida)
    print(f"Modelo fusionado guardado en {dir_salida}")

    tamano_adaptador = tamano_directorio_mb(dir_adaptador)
    tamano_fusionado = tamano_directorio_mb(dir_salida)
    print("\n=== Comparación de tamaño en disco ===")
    print(f"Adaptador LoRA solo : {tamano_adaptador:8.1f} MB  ({dir_adaptador})")
    print(f"Modelo fusionado    : {tamano_fusionado:8.1f} MB  ({dir_salida})")
    print(f"Factor              : {tamano_fusionado / tamano_adaptador:.0f}x más pesado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir-adaptador", default=DIR_ADAPTADOR_DEFECTO)
    parser.add_argument("--dir-salida", default=DIR_SALIDA_DEFECTO)
    args = parser.parse_args()
    fusionar(dir_adaptador=args.dir_adaptador, dir_salida=args.dir_salida)
