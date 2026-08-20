# Fase 0 — Setup y preparación de datos

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)): dejar
el repo estructurado, un entorno Python reproducible, el modelo base
definitivo elegido con datos reales de esta máquina, y el corpus de
`IA-tech` transformado a un dataset de fine-tuning listo para las fases
siguientes (LoRA).

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/probar_modelo_base.py`](../src/probar_modelo_base.py) | Carga un modelo candidato, mide tiempo/RAM y genera una respuesta de prueba |
| [`src/preparar_dataset.py`](../src/preparar_dataset.py) | Transforma `corpus.json` de `IA-tech` en pares instrucción/respuesta (`dataset_finetuning.jsonl`) |
| [`tests/test_preparar_dataset.py`](../tests/test_preparar_dataset.py) | Pruebas automatizadas del dataset generado (`pytest`) |

## 1. Estructura del repositorio y entorno

Misma separación validada en `IA-tech` (`data/raw` intocable, `data/processed`
100% regenerable desde `src/`), con `uv` como gestor de dependencias por la
misma razón: lockfile reproducible (`uv.lock`), evita el "en mi máquina sí
funciona" — aquí más crítico aún porque `torch`/`transformers`/`peft` son
sensibles a versión de una forma en que `pandas` no lo es tanto.

```
IA-tech-fine-tuning/
├── data/
│   ├── raw/          # vacío por ahora: esta fase no descarga datos crudos propios
│   └── processed/    # dataset_finetuning.jsonl, derivado de IA-tech/data/processed/corpus.json
├── src/
├── notebooks/
├── tests/
├── docs/
├── pyproject.toml
├── uv.lock
└── .venv/
```

A diferencia de `IA-tech`, `uv init` por defecto generó un layout de paquete
instalable (`src/ia_tech_fine_tuning/__init__.py` + `[project.scripts]`).
Se eliminó ese layout: este repo es una colección de scripts educativos, no
una librería que otros proyectos vayan a importar — mismo estilo plano de
`src/` que `IA-tech`.

**`.gitignore`:** además de lo estándar de Python/venv, se agregaron
ignorados específicos de este proyecto: `data/raw/` y `data/processed/`
(info personal de los participantes, igual que en `IA-tech` — se guardan
`.gitkeep` para que las carpetas vacías sí queden trackeadas), checkpoints y
adaptadores LoRA (`checkpoints/`, `outputs/`, `adapters/`, `*.safetensors`,
`*.bin`, `*.pt`) porque van de MBs a GBs y son 100% regenerables entrenando
de nuevo, y cache local de modelos de HuggingFace.

**Dependencias instaladas:** `torch` (CPU), `transformers`, `peft`,
`accelerate`, `psutil` (para medir RAM en el paso siguiente).

## 2. Elección del modelo base

`PROYECTO_BASE.md` dejaba el modelo "por decidir... evaluando tiempos reales
de carga/entrenamiento en esta máquina" a propósito: elegir por specs en una
tabla es fácil, pero solo midiendo se sabe si el training es viable en la
práctica.

Se corrió [`src/probar_modelo_base.py`](../src/probar_modelo_base.py) contra
los dos candidatos, cargando en `fp32` (sin cuantización — recordar que
`bitsandbytes`/QLoRA no corre sin CUDA, ver `PROYECTO_BASE.md`):

| Métrica | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct |
|---|---|---|
| Parámetros | 494,032,768 | 1,543,714,304 (3.1x) |
| RAM usada (delta del proceso) | ~2.0 GB | ~6.0 GB (3x) |
| Generación de 60 tokens | 2.5 s | 6.4 s (2.6x más lento) |

**Modelo elegido: `Qwen2.5-1.5B-Instruct`.** Con 32GB de RAM disponibles,
6GB de uso es cómodo — el costo real del modelo más grande no es viabilidad,
es tiempo: cada paso de entrenamiento en la Fase 2 va a tardar ~2.6x más en
CPU. Se aceptó ese costo (horas extra de entrenamiento) a cambio de mayor
capacidad de lenguaje del modelo base, priorizando calidad de generación
sobre velocidad de iteración.

**Nota técnica:** en `transformers` 5.x, `tokenizer.apply_chat_template(...,
return_tensors="pt")` ya no devuelve el tensor de IDs directamente, sino un
`BatchEncoding` — hay que acceder a `.input_ids`. Cambio de comportamiento
respecto a versiones anteriores de la librería, documentado aquí porque no
es obvio del error (`AttributeError` al buscar `.shape` en un dict-like).

## 3. El dataset de fine-tuning: por qué tiene esta forma

Esta es la pieza conceptualmente más importante de la fase, porque el mismo
corpus de `IA-tech` se usa aquí con un propósito completamente distinto.

### 3.1. Mismo corpus, dos tareas distintas

En `IA-tech`, `corpus.json` alimentaba un **RAG**: cada uno de los 6582
registros se convertía en un vector (embedding) y se buscaba por similitud
en el momento de responder. El LLM de esa fase nunca memoriza esos datos en
sus pesos — los *lee* como contexto cada vez que hace falta, como quien
consulta una ficha en un archivero.

Aquí, LoRA hace lo opuesto: **ajusta los pesos** del modelo para que
internalice el patrón pregunta→respuesta directamente, sin necesidad de
recuperar nada en el momento de generar. Por eso el dato ya no puede
quedarse como una fila de tabla (`jornada`, `participante`, `pregunta`,
`respuesta_participante`, `resultado_oficial`, `puntos`) — el modelo no
entrena sobre "filas", entrena sobre **texto continuo con un turno de
usuario y un turno de asistente**, que es la forma en la que aprendió
originalmente a seguir instrucciones (fase de *instruction-tuning* de
Qwen2.5, ya hecha por Alibaba antes de que este proyecto empiece).

### 3.2. Por qué el formato `{"instruction": ..., "response": ...}`

Es el mismo patrón que popularizó Stanford Alpaca (2023) para fine-tuning de
instrucciones a bajo costo, y sigue siendo el formato base que esperan
`peft`/`trl` de HuggingFace hoy: un par de texto libre, no una fila
tabular con columnas fijas. La razón de fondo es que el modelo no aprende
"campos" — aprende una secuencia de tokens, y ese par se inserta luego en la
plantilla de chat específica del modelo (`apply_chat_template`, con los
tokens especiales `<|im_start|>user...<|im_start|>assistant...` de Qwen)
en la Fase 2, al momento de tokenizar para entrenamiento. Guardar ya el
par en `instruction`/`response` — en vez de guardar directamente el texto
con la plantilla aplicada — mantiene el dataset reutilizable si algún día
se cambia de modelo base (la plantilla de chat varía por familia de
modelo; los pares de texto no).

Por cada uno de los 6582 registros del corpus, `construir_par()` en
[`src/preparar_dataset.py`](../src/preparar_dataset.py) genera:

```json
{
  "instruction": "En la Jornada 1 - Fase de Grupos, ¿qué respondió Participante1 a la pregunta: ¿Quién gana Partido 1??",
  "response": "Respondió México. El resultado oficial fue México, por lo que obtuvo 2 puntos."
}
```

**Por qué ese contenido específico y no otro:** la instrucción reformula la
fila (jornada + participante + pregunta) como pregunta en lenguaje natural
en vez de listar campos sueltos, porque eso es lo más parecido a cómo un
usuario real le preguntaría al modelo ya entrenado ("¿qué respondió fulano a
esta pregunta?"). La respuesta concatena tres datos que en la hoja
`PUNTUACIONES` viven en columnas separadas (`respuesta_participante`,
`resultado_oficial`, `puntos`) en una sola oración, porque el modelo tiene
que aprender a *redactar* la respuesta completa, no a rellenar un
formulario — si solo entrenara con el valor crudo, no aprendería a producir
lenguaje natural coherente al responder.

### 3.3. Por qué JSONL y no JSON

`data/processed/dataset_finetuning.jsonl` guarda un objeto JSON por línea
(6582 líneas), en vez de una lista JSON como `corpus.json`. Es el formato
estándar de facto para datasets de fine-tuning (lo esperan `datasets` de
HuggingFace, `trl`, y prácticamente todo pipeline de entrenamiento): permite
leer el archivo línea por línea sin cargarlo completo a memoria
(*streaming*), y facilita mezclar/filtrar/hacer `shuffle` de ejemplos sin
tocar un único array gigante.

### 3.4. Bug heredado del corpus original: marcadores leídos como fecha

Al inspeccionar los primeros pares generados, aparecieron respuestas sin
sentido como `"Respondió 2026-01-01 00:00:00"`. Investigando (no asumiendo
un fix genérico, mismo principio aplicado en `IA-tech` para el mojibake de
encoding):

1. Se confirmó que el problema **no** está en `preparar_dataset.py` ni en
   `data_prep.py` de `IA-tech` — ya viene corrupto en `corpus.json`.
2. Se leyó la hoja `RESPUESTAS` del Excel original
   (`IA-tech/data/raw/Polla Mundial 2026.xlsx`) directamente con `pandas`:
   la columna `valor` contiene objetos `datetime` para esas celdas, no
   texto. Excel interpretó marcadores tipo `"2-1"` como fecha corta al
   momento de guardarse (comportamiento clásico de auto-formato de Excel:
   una celda con texto `"M-D"` se reconoce como fecha `mes/día` si no está
   forzada explícitamente como texto).
3. Se contrastó contra `resultado_oficial` de `RESULTADOS_OFICIALES` (que
   sí preservó los marcadores como string, ej. `"2-1"`) y contra los pocos
   valores de `RESPUESTAS` que sobrevivieron como string (`"0-1"`, `"2-0"`,
   etc. — sobreviven cuando alguno de los dos números es 0, porque `"0-1"`
   no siempre se reconoce como fecha válida). Esto confirmó el patrón de
   conversión: Excel leyó `"M-D"` como fecha `2026-M-D`.
4. `data_prep.py` de `IA-tech` serializa cualquier `datetime` a string ISO
   vía `default=str` en `json.dump` (documentado en su propia Fase 0 como
   red de seguridad para columnas de fecha) — por eso en `corpus.json`
   quedó como el string `"2026-01-01 00:00:00"` en vez de fallar la
   exportación.
5. **Impacto medido:** 1327 de 6582 registros (20%) tenían este problema en
   `respuesta_participante` y/o `resultado_oficial`.

**Decisión:** reconstruir el marcador original en vez de descartar esos
registros o dejarlos tal cual. Descartarlos habría tirado 1 de cada 5
ejemplos de entrenamiento sin necesidad; dejarlos le habría enseñado al
modelo a "responder" preguntas de marcador con fechas sin sentido — ruido
real durante el fine-tuning, no un detalle cosmético.

`normalizar_marcador()` en
[`src/preparar_dataset.py`](../src/preparar_dataset.py) invierte la lectura
de Excel con una regex sobre el patrón `"2026-MM-DD 00:00:00"`, restando 1 a
mes y día (Excel interpreta el dígito `"0"` de un marcador tipo `"0-1"` como
día/mes 1). Valores que no matchean el patrón (nombres de equipo, marcadores
que sí sobrevivieron como string, etc.) se devuelven sin tocar. Cubierto por
tests dedicados en `test_normalizar_marcador_reconstruye_fecha_excel` y
`test_normalizar_marcador_no_toca_valores_no_fecha`.

Esta corrección vive únicamente en `preparar_dataset.py` de este proyecto —
no se modificó `IA-tech/data/processed/corpus.json` ni su código, porque ese
corpus sigue siendo válido para su propio uso en RAG (donde el dato crudo,
aunque raro, no se "aprende" ni contamina pesos de un modelo).

## 4. Resultado y verificación

```
Dataset generado: data\processed\dataset_finetuning.jsonl (6582 ejemplos)
```

```bash
./.venv/Scripts/uv.exe run pytest tests/ -v
```

```
tests/test_preparar_dataset.py::test_dataset_tiene_mismo_conteo_que_corpus PASSED
tests/test_preparar_dataset.py::test_dataset_no_tiene_campos_vacios PASSED
tests/test_preparar_dataset.py::test_dataset_no_tiene_fechas_residuales PASSED
tests/test_preparar_dataset.py::test_normalizar_marcador_reconstruye_fecha_excel PASSED
tests/test_preparar_dataset.py::test_normalizar_marcador_no_toca_valores_no_fecha PASSED
```

**Cómo regenerar todo:**

```bash
./.venv/Scripts/uv.exe run python src/probar_modelo_base.py "Qwen/Qwen2.5-1.5B-Instruct"
./.venv/Scripts/uv.exe run python src/preparar_dataset.py
./.venv/Scripts/uv.exe run pytest tests/ -v
```

## 5. Pendiente

- Fase 1 del roadmap: fundamentos de PEFT y LoRA, instrumentar
  `Qwen2.5-1.5B-Instruct` con un adaptador LoRA vacío y contar parámetros
  entrenables vs. totales (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)).
- El anomalía puntual `'3⁷-1'` encontrada en un valor de marcador (probable
  error de tecleo en el Excel original) no se normalizó — no matchea el
  patrón de fecha corrupta y su impacto es un único registro; queda como
  ejemplo de dato "ruidoso" real, aceptable en un dataset de 6582 filas.
