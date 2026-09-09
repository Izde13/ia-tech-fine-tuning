# Fase 3 — Evaluación y comparación

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)): medir
con rigor si el fine-tuning "funcionó", y usar esa medición para decidir
sistemáticamente qué palanca de LoRA (épocas, `target_modules`, `r`) tiene
más impacto real, en vez de ajustar hiperparámetros a ciegas.

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/entrenar_lora.py`](../src/entrenar_lora.py) | Extendido en esta fase con `--target-mlp` y `--r` para poder correr las 4 variantes sin reescribir código |
| [`src/evaluar_lora.py`](../src/evaluar_lora.py) | Extendido con `--dir-adaptador`/`--salida-json` para evaluar cualquier adaptador contra el mismo set de validación |
| [`src/comparar_base_vs_lora.py`](../src/comparar_base_vs_lora.py) | Heredado de Fase 2, comparación cualitativa de 3 ejemplos |

## 1. Diseño del experimento: aislar una variable a la vez

Partiendo del hallazgo de cierre de Fase 2 (14.4% correcto global, **0.0%**
en preguntas de marcador), se plantearon 3 hipótesis, cada una probada
manteniendo todo lo demás fijo:

1. **¿Falta repetición de gradiente?** → duplicar épocas (1→2), mismo
   `target_modules=[q_proj, v_proj]`.
2. **¿Falta acceso a las capas correctas?** → mantener 1 época, ampliar
   `target_modules` a las 3 capas MLP (`gate_proj`, `up_proj`, `down_proj`).
   Hipótesis motivada por investigación real: el paper de edición de
   conocimiento factual ROME (Meng et al., 2022) mostró que el conocimiento
   factual arbitrario en transformers vive predominantemente en las capas
   MLP de bloques intermedios, no en las cabezas de atención.
3. **¿Falta capacidad del adaptador?** → mantener 1 época y `target_modules`
   con MLP, duplicar `r` (8→16). `lora_alpha` se mantiene en `2×r` en todas
   las corridas para conservar la magnitud efectiva del ajuste
   (`ΔW = (alpha/r) × B×A`) y que `r` sea la única variable que cambia.

Las 4 corridas se entrenaron sobre el dataset completo (5923 train / 659
val, mismo split `seed=42` de Fase 2), evaluando `eval_loss` solo al final
de cada época (no cada 100 pasos como en Fase 2) para reducir el overhead de
evaluación intermedia en corridas de varias horas — decisión explícita del
usuario para esta fase: sin checkpoints ni evaluaciones intermedias, una
sola corrida de principio a fin por variante.

## 2. Extensión de `entrenar_lora.py`: `--target-mlp` y `--r`

En vez de hardcodear cada variante como un script nuevo, se parametrizó
`entrenar_lora.py` para aceptar la arquitectura del adaptador como flags,
manteniendo trazabilidad de qué config generó qué carpeta de salida
(`dir_salida_para()` compone el nombre según `target_mlp`/`r`/`épocas`):

```python
_TARGET_MODULES_ATENCION = ["q_proj", "v_proj"]
_TARGET_MODULES_CON_MLP = _TARGET_MODULES_ATENCION + ["gate_proj", "up_proj", "down_proj"]

def config_lora_para(target_mlp: bool, r: int) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=2 * r,
        target_modules=_TARGET_MODULES_CON_MLP if target_mlp else _TARGET_MODULES_ATENCION,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
```

Del mismo modo, `evaluar_lora.py` se extendió con `--dir-adaptador` y
`--salida-json` para poder evaluar cualquier adaptador guardado contra el
mismo set de validación reproducible, sin tocar el script cada vez.

## 3. Resultado: `eval_loss` de las 4 variantes

| Variante | `eval_loss` | `train_loss` | Tiempo |
|---|---|---|---|
| Atención, 1 época | 0.1597 | 0.2818 | 5h53min |
| Atención, 2 épocas | 0.1374 | 0.2139 | 5h29min |
| Atención + MLP, `r=8` | 0.1179 | 0.1808 | 3h32min |
| Atención + MLP, `r=16` | 0.1119 | 0.1692 | 3h04min |

**Comandos usados** (ejecutados desde la raíz del repo; `--n-ejemplos 0` usa
el dataset completo):

```bash
./.venv/Scripts/uv.exe run python src/entrenar_lora.py --n-ejemplos 0 --epocas 1
./.venv/Scripts/uv.exe run python src/entrenar_lora.py --n-ejemplos 0 --epocas 2
./.venv/Scripts/uv.exe run python src/entrenar_lora.py --n-ejemplos 0 --epocas 1 --target-mlp
./.venv/Scripts/uv.exe run python src/entrenar_lora.py --n-ejemplos 0 --epocas 1 --target-mlp --r 16
```

## 4. Resultado: evaluación cuantitativa sobre los 659 ejemplos de validación

Cada adaptador se evaluó con `evaluar_lora.py` sobre el mismo set de
validación completo, clasificando cada respuesta en `correcto` (3 campos
exactos), `parcial` (al menos 1 campo) o `incorrecto` (ninguno), desglosado
por tipo de pregunta (metodología heredada de Fase 2).

| Variante | Correcto global | Parcial global | Incorrecto global | Correcto marcador (170) | Correcto "otro" (489) |
|---|---|---|---|---|---|
| Atención, 1 época | 14.4% (95) | 69.3% (457) | 16.2% (107) | 0.0% (0) | 19.4% (95) |
| Atención, 2 épocas | 14.6% (96) | 66.5% (438) | 19.0% (125) | 0.6% (1) | 19.4% (95) |
| Atención + MLP, `r=8` | 23.8% (157) | 63.3% (417) | 12.9% (85) | 8.2% (14) | 29.2% (143) |
| **Atención + MLP, `r=16`** | **28.4% (187)** | 62.1% (409) | **9.6% (63)** | **8.2% (14)** | **35.4% (173)** |

**Comandos usados:**

```bash
./.venv/Scripts/uv.exe run python src/evaluar_lora.py --dir-adaptador outputs/adaptador_lora --salida-json data/processed/evaluacion_lora.json
./.venv/Scripts/uv.exe run python src/evaluar_lora.py --dir-adaptador outputs/adaptador_lora_2ep --salida-json data/processed/evaluacion_lora_2ep.json
./.venv/Scripts/uv.exe run python src/evaluar_lora.py --dir-adaptador outputs/adaptador_lora_mlp --salida-json data/processed/evaluacion_lora_mlp.json
./.venv/Scripts/uv.exe run python src/evaluar_lora.py --dir-adaptador outputs/adaptador_lora_mlp_r16 --salida-json data/processed/evaluacion_lora_mlp_r16.json
```

## 5. Lo que falló durante la fase (y cómo se investigó)

### 5.1. Bug en la comparación cualitativa: comparando el adaptador contra sí mismo

El primer intento de `comparar_base_vs_lora.py` (heredado de Fase 2, sin
cambios) devolvió respuestas **idénticas** entre "modelo base" y "modelo +
LoRA" en los 3 ejemplos probados con el adaptador `_mlp` — señal aparente de
que el fine-tuning no tenía efecto.

**Investigación:** en vez de asumir que el entrenamiento había fallado, se
verificó el efecto del adaptador directamente sobre los logits (no sobre el
texto generado), usando el contexto `modelo.disable_adapter()` para
comparar con/sin el adaptador activo en el mismo objeto de modelo. Se
confirmó una diferencia real de ~22 en los logits y un cambio del token más
probable — el adaptador sí tenía efecto.

**Causa raíz:** `PeftModel.from_pretrained(modelo_base, dir_adaptador)`
modifica `modelo_base` **in-place**, no devuelve una copia independiente.
Guardar una referencia a `modelo_base` antes de esa llamada y usarla después
para "generar sin LoRA" en realidad generaba con el mismo objeto ya
modificado. El script comparaba LoRA contra sí mismo, no contra el modelo
base real. Corregido usando `with modelo.disable_adapter():` en vez de
mantener dos referencias.

### 5.2. Entrenamiento perdido por cierre de VSCode (heredado de Fase 2, repetido en esta fase)

Igual que en Fase 2, correr sin `save_strategy="steps"` (decisión explícita
del usuario para no acumular checkpoints pesados) significa que cerrar la
ventana de VSCode mata el proceso en background sin guardar nada
intermedio. No volvió a ocurrir en Fase 3 porque se mantuvo la ventana
abierta durante las 4 corridas largas, pero es un riesgo que se repite cada
vez que se lanza un entrenamiento de varias horas.

### 5.3. Evaluación cuantitativa "colgada" varias veces — buffering de stdout, no fallos reales

En al menos 3 ocasiones (evaluación de `_mlp`, entrenamiento de `_mlp_r16`,
evaluación de `_mlp_r16`), el log de un proceso en background dejó de
mostrar progreso durante horas — en un caso, 8 horas sin que aparecieran los
prints de progreso ("50/659 procesados...", etc.).

**Investigación:** verificar el CPU acumulado del proceso
(`Get-Process python | Select CPU`) en vez de asumir que estaba colgado.
En todos los casos el CPU seguía subiendo (aunque a un ritmo irregular,
con baches de mucha lentitud seguidos de recuperación), confirmando que el
proceso trabajaba activamente — el problema era que los `print()` de Python
se quedaban bufferizados en memoria (por no ser una terminal interactiva) y
solo se volcaban al archivo de log de una sola vez, al terminar el proceso.
No se identificó la causa exacta de los baches de lentitud puntuales del
sistema (posible contención de recursos no relacionada con este proyecto),
pero en todos los casos el proceso terminó y produjo resultados válidos sin
necesidad de reiniciarlo — con una excepción:

### 5.4. Una evaluación sí se detuvo manualmente y se retomó sin pérdida de datos

En un caso, el usuario pidió detener una evaluación cuantitativa en curso
("párala, luego lo retomo") para usar la máquina para otra cosa. Como
`evaluar_lora.py` es puramente de inferencia (no modifica el adaptador ni
el dataset), detener y relanzar el mismo comando más tarde no perdió nada
permanente — solo el tiempo de cómputo ya invertido en esa corrida, porque
el script no guarda progreso incremental entre ejemplos.

## 6. Conclusión: `target_modules` importó más que épocas o `r`

**Duplicar épocas** con solo `[q_proj, v_proj]` apenas movió el resultado
(14.4%→14.6% global, 0.0%→0.6% en marcador — dentro del ruido estadístico
sobre 170 ejemplos). Confirma que el cuello de botella no era "pocas
repeticiones de gradiente".

**Ampliar `target_modules` al MLP** produjo el salto más grande de toda la
fase (14.4%→23.8% global, 0.0%→8.2% en marcador), en **menos** tiempo de
entrenamiento que las corridas de solo atención — confirma empíricamente la
hipótesis de ROME: darle al adaptador acceso a las capas donde vive el
conocimiento factual fue más efectivo que ver los mismos datos más veces.

**Duplicar `r`** (con MLP ya activo) mejoró el resultado global y sobre todo
en "otro" (29.2%→35.4%), pero **dejó exactamente igual el acierto en
marcadores (8.2%→8.2%, mismos 14 ejemplos)**. Esto es el hallazgo más
interesante de la fase: para ese tipo de pregunta específico, ni más
capacidad del adaptador (`r`) ni más repeticiones (épocas) mueven la
aguja — sugiere un techo estructural, no de configuración.

**Interpretación del techo en marcadores:** un marcador de fútbol (ej.
"2-0") es un dato esencialmente arbitrario respecto al texto de la
pregunta — no hay patrón lingüístico que lo explique, solo memorización
directa de cada par específico pregunta→marcador. Con 170 ejemplos de este
tipo en el set de validación (de 659 totales) y un modelo de 1.5B con LoRA,
memorizar consistentemente hechos de tan baja frecuencia parece requerir más
que ajustar hiperparámetros dentro del mismo régimen de entrenamiento.

## 7. Adaptador recomendado

**`outputs/adaptador_lora_mlp_r16/`** — mejor resultado en todas las
métricas salvo el empate en marcadores, y el segundo más rápido de entrenar
de las 4 variantes. Es el adaptador candidato para la Fase 4 (merge y
portafolio).

## 8. Explorar en Google Colab (fuera del alcance de esta fase)

Se discutió con el usuario migrar a Google Colab para superar el techo de
marcadores, y se decidió **no hacerlo dentro de esta fase** — la restricción
de hardware (CPU local) es parte del objetivo de aprendizaje documentado en
`PROYECTO_BASE.md`, no solo un obstáculo a evitar. Queda documentado acá
como guía para retomarlo más adelante, si se decide explorarlo:

**Qué cambia con Colab:** GPU gratuita (típicamente una T4). El costo
dominante de las corridas de esta fase fue tiempo de CPU — la misma
arquitectura de modelo y LoRA en una T4 bajaría de horas a minutos por
corrida, permitiendo iterar variantes (más épocas, `r` más alto,
`k_proj`/`o_proj`, `learning_rate`) en una sola sesión en vez de repartidas
en días.

**Qué NO cambia solo por usar Colab:** correr el mismo `Qwen2.5-1.5B` más
rápido no le da más capacidad de memorizar marcadores per se — el techo de
8.2% observado podría seguir estando ahí, solo que las iteraciones para
confirmarlo o superarlo (con `k_proj`/`o_proj`, más `r`, más épocas)
costarían minutos en vez de horas. Un modelo base más grande (ej.
`Qwen2.5-7B`) sí aporta más capacidad general, pero es un cambio de alcance
distinto — deja de ser el "LLM pequeño en CPU" que define este proyecto.

**Pasos generales para migrar** (sin comprometerse a hacerlo, solo como
referencia):

1. Subir a Google Drive (o clonar desde el repo) `dataset_finetuning.jsonl`
   y los scripts `entrenar_lora.py`/`evaluar_lora.py` — ambos ya son
   independientes de rutas absolutas del proyecto, deberían correr sin
   cambios apuntando a rutas de Drive/Colab.
2. En un notebook de Colab, activar entorno de ejecución con GPU (Runtime →
   Change runtime type → GPU).
3. Instalar las mismas dependencias (`torch`, `transformers`, `peft`,
   `datasets`, `accelerate`) — `torch` se instala automáticamente con
   soporte CUDA en Colab, a diferencia del CPU-only usado en esta máquina.
4. `AutoModelForCausalLM.from_pretrained(..., dtype=torch.float32)` puede
   pasar a `dtype=torch.bfloat16` o usar `device_map="auto"` para
   aprovechar la GPU — cambio menor a `entrenar_lora.py`.
5. Con GPU, correr las variantes no probadas en esta fase (`k_proj`/
   `o_proj` incluidos, ajuste de `learning_rate`) sería viable en el mismo
   día, algo que en CPU local habría tomado varias jornadas más.

## 9. Pendiente

- Fase 4 del roadmap: fusión del adaptador `_mlp_r16` con el modelo base
  (`merge_and_unload`) vs. mantenerlo separado, y documentación de
  portafolio (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)).
- Variantes de arquitectura sin probar en CPU local: `target_modules` con
  `k_proj`/`o_proj` incluidos (atención completa) junto al MLP; ajuste de
  `learning_rate` (se mantuvo en `2e-4` heredado de Fase 1 en las 4
  corridas, nunca explorado como variable).
- Exploración en Google Colab con GPU: documentada en la sección 8 como
  guía de referencia, explícitamente fuera del alcance de esta fase.
