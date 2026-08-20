"""Explora la arquitectura interna de Qwen2.5-1.5B-Instruct.

No genera texto: recorre los módulos del modelo cargado para identificar
dónde viven las capas nn.Linear de atención (q_proj, k_proj, v_proj, o_proj)
donde LoRA va a insertar sus matrices de bajo rango en la Fase 2. Los
nombres exactos son necesarios para configurar `target_modules` de `peft`.
"""

import torch
from torch import nn
from transformers import AutoModelForCausalLM

NOMBRE_MODELO = "Qwen/Qwen2.5-1.5B-Instruct"


def explorar_arquitectura(nombre_modelo: str) -> None:
    modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo, dtype=torch.float32)

    capas_transformer = modelo.model.layers
    print(f"Número de bloques transformer: {len(capas_transformer)}")

    primer_bloque = capas_transformer[0]
    print("\n=== Módulos nn.Linear dentro del primer bloque (layers.0) ===")
    for nombre, modulo in primer_bloque.named_modules():
        if isinstance(modulo, nn.Linear):
            print(
                f"  {nombre:20s} in={modulo.in_features:5d} "
                f"out={modulo.out_features:5d} bias={modulo.bias is not None}"
            )

    print("\n=== Todos los nombres únicos de nn.Linear en el modelo completo ===")
    nombres_unicos = set()
    for nombre, modulo in modelo.named_modules():
        if isinstance(modulo, nn.Linear):
            sufijo = nombre.rsplit(".", 1)[-1]
            nombres_unicos.add(sufijo)
    print(sorted(nombres_unicos))


if __name__ == "__main__":
    explorar_arquitectura(NOMBRE_MODELO)
