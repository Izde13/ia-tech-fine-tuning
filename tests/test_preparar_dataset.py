import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preparar_dataset import CORPUS_ORIGEN, DATASET_DESTINO, normalizar_marcador

_PATRON_FECHA_RESIDUAL = re.compile(r"2026-\d{2}-\d{2} 00:00:00")


def _leer_dataset():
    with open(DATASET_DESTINO, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f]


def test_dataset_tiene_mismo_conteo_que_corpus():
    with open(CORPUS_ORIGEN, encoding="utf-8") as f:
        corpus = json.load(f)
    dataset = _leer_dataset()
    assert len(dataset) == len(corpus)


def test_dataset_no_tiene_campos_vacios():
    dataset = _leer_dataset()
    vacios = [
        ejemplo
        for ejemplo in dataset
        if not ejemplo.get("instruction") or not ejemplo.get("response")
    ]
    assert vacios == []


def test_dataset_no_tiene_fechas_residuales():
    dataset = _leer_dataset()
    con_fecha = [
        ejemplo
        for ejemplo in dataset
        if _PATRON_FECHA_RESIDUAL.search(ejemplo["response"])
    ]
    assert con_fecha == []


def test_normalizar_marcador_reconstruye_fecha_excel():
    assert normalizar_marcador("2026-01-01 00:00:00") == "0-0"
    assert normalizar_marcador("2026-02-03 00:00:00") == "1-2"


def test_normalizar_marcador_no_toca_valores_no_fecha():
    assert normalizar_marcador("2-0") == "2-0"
    assert normalizar_marcador("México") == "México"
