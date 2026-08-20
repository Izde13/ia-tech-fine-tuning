"""Transforma el corpus de IA-tech (pensado para RAG) en pares
instrucción/respuesta aptos para fine-tuning con LoRA.

A diferencia de IA-tech, donde cada registro se embebía para búsqueda por
similitud, aquí cada registro se convierte en un ejemplo de entrenamiento
que le enseña al modelo, vía sus pesos, el patrón pregunta -> respuesta
del dominio de la Polla Mundial.
"""

import json
import re
from pathlib import Path

CORPUS_ORIGEN = Path("../IA-tech/data/processed/corpus.json")
DATASET_DESTINO = Path("data/processed/dataset_finetuning.jsonl")

# Excel interpretó marcadores tipo "M-D" (ej. "2-1") como fecha corta y los
# guardó como datetime; data_prep.py de IA-tech los serializó como
# "2026-MM-DD 00:00:00" (ver default=str). Se reconstruye el marcador
# original invirtiendo esa lectura: mes -> primer número, día -> segundo,
# cada uno menos 1 (Excel interpreta "0" como día/mes 1).
_PATRON_FECHA_MARCADOR = re.compile(r"^2026-(\d{2})-(\d{2}) 00:00:00$")


def normalizar_marcador(valor: str) -> str:
    coincidencia = _PATRON_FECHA_MARCADOR.match(str(valor))
    if not coincidencia:
        return valor
    mes, dia = coincidencia.groups()
    return f"{int(mes) - 1}-{int(dia) - 1}"


def construir_par(registro: dict) -> dict:
    respuesta = normalizar_marcador(registro["respuesta_participante"])
    resultado = normalizar_marcador(registro["resultado_oficial"])

    instruction = (
        f"En la {registro['jornada']}, ¿qué respondió {registro['participante']} "
        f"a la pregunta: {registro['pregunta']}?"
    )
    response = (
        f"Respondió {respuesta}. "
        f"El resultado oficial fue {resultado}, "
        f"por lo que obtuvo {registro['puntos']} puntos."
    )
    return {"instruction": instruction, "response": response}


def main() -> None:
    with open(CORPUS_ORIGEN, encoding="utf-8") as f:
        corpus = json.load(f)

    DATASET_DESTINO.parent.mkdir(parents=True, exist_ok=True)

    with open(DATASET_DESTINO, "w", encoding="utf-8") as f:
        for registro in corpus:
            par = construir_par(registro)
            f.write(json.dumps(par, ensure_ascii=False) + "\n")

    print(f"Dataset generado: {DATASET_DESTINO} ({len(corpus)} ejemplos)")


if __name__ == "__main__":
    main()
