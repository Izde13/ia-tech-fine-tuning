"""Entrena el adaptador LoRA instrumentado en la Fase 1 sobre
dataset_finetuning.jsonl (Fase 0).

Por defecto entrena sobre un subset chico del dataset — validar que loss
baja, que el checkpoint se guarda, y que el tiempo por paso es razonable en
esta máquina (CPU, sin CUDA) antes de comprometer horas al dataset completo.
Subir a dataset completo es cambiar --n-ejemplos, no reescribir el script.
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

NOMBRE_MODELO = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_JSONL = "data/processed/dataset_finetuning.jsonl"
DIR_SALIDA = "outputs/adaptador_lora"


def dir_salida_para(epocas: float, target_mlp: bool, r: int) -> str:
    dir_salida = DIR_SALIDA
    if target_mlp:
        dir_salida += "_mlp"
    if r != 8:
        dir_salida += f"_r{r}"
    if abs(epocas - 1.0) >= 1e-9:
        dir_salida += f"_{epocas:g}ep"
    return dir_salida


# target_modules=[q_proj, v_proj] (Fase 1): configuración mínima del paper
# original de LoRA, solo capas de atención. --target-mlp amplía a las 3
# capas del feed-forward (gate/up/down_proj) — hipótesis de Fase 3: el
# conocimiento factual arbitrario (marcadores) vive más en el MLP que en
# atención (ver ROME, Meng et al. 2022), motivado por que duplicar épocas
# con solo [q_proj, v_proj] no mejoró el acierto en marcadores (0.0% → 0.6%,
# ruido estadístico sobre 170 ejemplos).
_TARGET_MODULES_ATENCION = ["q_proj", "v_proj"]
_TARGET_MODULES_CON_MLP = _TARGET_MODULES_ATENCION + ["gate_proj", "up_proj", "down_proj"]


def config_lora_para(target_mlp: bool, r: int) -> LoraConfig:
    # lora_alpha se mantiene en 2×r (misma proporción que r=8/alpha=16 de
    # Fase 1) — el factor de escala del ajuste LoRA es (alpha/r)×B×A, así
    # que mantener la razón constante conserva la magnitud efectiva del
    # ajuste al variar r, aislando r como la única variable del experimento.
    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        target_modules=_TARGET_MODULES_CON_MLP if target_mlp else _TARGET_MODULES_ATENCION,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )


def tokenizar_ejemplo(ejemplo: dict, tokenizer) -> dict:
    mensajes = [
        {"role": "user", "content": ejemplo["instruction"]},
        {"role": "assistant", "content": ejemplo["response"]},
    ]
    texto = tokenizer.apply_chat_template(mensajes, tokenize=False)
    salida = tokenizer(texto, truncation=True, max_length=256)
    return salida


def entrenar(n_ejemplos: int, epocas: float, target_mlp: bool, r: int) -> None:
    dir_salida = dir_salida_para(epocas, target_mlp, r)
    tokenizer = AutoTokenizer.from_pretrained(NOMBRE_MODELO)
    modelo_base = AutoModelForCausalLM.from_pretrained(NOMBRE_MODELO, dtype=torch.float32)
    modelo = get_peft_model(modelo_base, config_lora_para(target_mlp, r))
    modelo.print_trainable_parameters()

    dataset = load_dataset("json", data_files=DATASET_JSONL, split="train")
    if n_ejemplos > 0:
        dataset = dataset.select(range(min(n_ejemplos, len(dataset))))

    # 10% reservado para validación: nunca se usa en el gradiente, solo para
    # medir la loss sobre ejemplos que el modelo no vio — la única forma de
    # distinguir "aprendió el patrón" de "memorizó estas filas puntuales".
    split = dataset.train_test_split(test_size=0.1, seed=42)
    dataset_train, dataset_val = split["train"], split["test"]
    print(f"Entrenando sobre {len(dataset_train)} ejemplos, validando sobre {len(dataset_val)}")

    tokenizar = lambda ejemplo: tokenizar_ejemplo(ejemplo, tokenizer)
    dataset_train_tok = dataset_train.map(tokenizar, remove_columns=dataset_train.column_names)
    dataset_val_tok = dataset_val.map(tokenizar, remove_columns=dataset_val.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args_entrenamiento = TrainingArguments(
        output_dir=dir_salida,
        num_train_epochs=epocas,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=modelo,
        args=args_entrenamiento,
        train_dataset=dataset_train_tok,
        eval_dataset=dataset_val_tok,
        data_collator=collator,
    )

    trainer.train()

    modelo.save_pretrained(dir_salida)
    tokenizer.save_pretrained(dir_salida)
    print(f"\nAdaptador LoRA guardado en {dir_salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-ejemplos",
        type=int,
        default=300,
        help="Cantidad de ejemplos a usar (0 = dataset completo, 6582 ejemplos)",
    )
    parser.add_argument("--epocas", type=float, default=1.0)
    parser.add_argument(
        "--target-mlp",
        action="store_true",
        help="Amplía target_modules a las capas MLP (gate/up/down_proj), además de q_proj/v_proj",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=8,
        help="Rango de las matrices LoRA (lora_alpha se ajusta a 2×r para mantener la escala del ajuste)",
    )
    args = parser.parse_args()
    entrenar(n_ejemplos=args.n_ejemplos, epocas=args.epocas, target_mlp=args.target_mlp, r=args.r)
