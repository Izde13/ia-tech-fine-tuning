"""Instrumenta Qwen2.5-1.5B-Instruct con un adaptador LoRA vacío (sin entrenar).

Entregable de la Fase 1: verificar en números reales que LoRA entrena menos
del 1% de los parámetros totales del modelo, insertando matrices de bajo
rango solo en las proyecciones de atención q_proj/v_proj.
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

NOMBRE_MODELO = "Qwen/Qwen2.5-1.5B-Instruct"

CONFIG_LORA = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    task_type="CAUSAL_LM",
)


def contar_parametros(modelo) -> tuple[int, int]:
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    totales = sum(p.numel() for p in modelo.parameters())
    return entrenables, totales


def instrumentar_lora() -> None:
    modelo_base = AutoModelForCausalLM.from_pretrained(NOMBRE_MODELO, dtype=torch.float32)

    entrenables_base, totales_base = contar_parametros(modelo_base)
    print("=== Modelo base, sin LoRA ===")
    print(f"Parámetros entrenables: {entrenables_base:,} (todo el modelo, sin congelar aún)")
    print(f"Parámetros totales:     {totales_base:,}")

    modelo_lora = get_peft_model(modelo_base, CONFIG_LORA)

    entrenables_lora, totales_lora = contar_parametros(modelo_lora)
    porcentaje = 100 * entrenables_lora / totales_lora

    print("\n=== Modelo + adaptador LoRA (r=8, target=[q_proj, v_proj]) ===")
    print(f"Parámetros entrenables: {entrenables_lora:,}")
    print(f"Parámetros totales:     {totales_lora:,}")
    print(f"Porcentaje entrenable:  {porcentaje:.4f}%")

    print("\n=== Resumen de peft ===")
    modelo_lora.print_trainable_parameters()


if __name__ == "__main__":
    instrumentar_lora()
