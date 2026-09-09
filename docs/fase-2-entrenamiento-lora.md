# Fase 2 — Entrenamiento con LoRA

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)): entrenar
el adaptador LoRA instrumentado en Fase 1 sobre el corpus real de la Polla
Mundial 2026, con un procedimiento riguroso (loop con `Trainer`, split
train/validación) que permita medir con datos duros si el modelo aprendió a
generalizar o solo memorizó el dataset de entrenamiento.

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/entrenar_lora.py`](../src/entrenar_lora.py) | Entrena el adaptador LoRA con `transformers.Trainer`, split train/val 90/10 |
| [`src/comparar_base_vs_lora.py`](../src/comparar_base_vs_lora.py) | Comparación cualitativa: modelo base vs. modelo + adaptador, sobre 3 ejemplos |
| [`src/evaluar_lora.py`](../src/evaluar_lora.py) | Evaluación cuantitativa sobre los 659 ejemplos del set de validación completo |

## 1. `transformers.Trainer`: el mismo loop de `IA-tech`, encapsulado

`Trainer` automatiza el mismo ciclo escrito a mano en `IA-tech` (Fase 4):
forward → loss → backward → `optimizer.step()`, repetido por batches y
épocas. La diferencia con LoRA es que, al estar el modelo base congelado
(`requires_grad=False`, verificado en Fase 1), el gradiente solo fluye hacia
las matrices `A`/`B` del adaptador (~1.09M parámetros) — el optimizer
(AdamW) nunca toca los 1.54B pesos originales de Qwen.

`tokenizar_ejemplo()` envuelve cada par `instruction`/`response` con
`tokenizer.apply_chat_template()`, insertando ambos turnos (usuario y
asistente) en la plantilla de chat nativa de Qwen — el modelo entrena sobre
la conversación completa, no solo la respuesta, para aprender el patrón
pregunta→respuesta como continuación natural del texto.
`DataCollatorForLanguageModeling(mlm=False)` arma los batches con las labels
como copia de los `input_ids` (aprendizaje causal estándar, no masked
language modeling de BERT).

## 2. Por qué hace falta un split train/validación

La primera corrida completa (6582 ejemplos, sin split) bajó la loss de 3.56
a ~0.17 — dato que por sí solo no distingue "el modelo aprendió el patrón
general" de "memorizó exactamente estas filas". Medir la loss sobre los
mismos datos que ya vio el gradiente es una métrica sesgada por diseño.

`entrenar_lora.py` reserva el 10% del dataset (`train_test_split(test_size=0.1,
seed=42)`) como set de validación — nunca se usa en el backward pass, solo
para medir `eval_loss` sobre ejemplos que el modelo no vio. La semilla fija
permite reproducir exactamente el mismo split en scripts posteriores
(`evaluar_lora.py`) sin tener que guardarlo aparte. `eval_steps=100` dispara
una evaluación completa sobre el set de validación cada 100 pasos de
entrenamiento, permitiendo ver la trayectoria de `eval_loss` a lo largo de
todo el entrenamiento, no solo al final.

## 3. Resultado real: entrenamiento completo (5923 train / 659 val, 1 época)

```
eval_loss por checkpoint (cada 100 pasos):
0.5124 → 0.2673 → 0.2236 → 0.21 → 0.1908 → 0.1783 → 0.1756
→ 0.1702 → 0.1681 → 0.1656 → 0.1639 → 0.1597 (final)

train_loss promedio de la época: 0.2818
Tiempo total: 5h53min (CPU, sin CUDA)
```

**Lectura:** la `eval_loss` —calculada sobre datos que el modelo nunca vio en
el gradiente— baja de forma monótona durante toda la época, sin ningún
rebote hacia arriba. Es la evidencia real de generalización, no solo
memorización: si el modelo solo hubiera memorizado el set de entrenamiento,
la `eval_loss` se habría estancado o empeorado en algún punto mientras la
`train_loss` seguía bajando (la firma clásica de overfitting).

**Incidente durante el entrenamiento:** la primera corrida completa con
split se perdió al paso 1027/1481 (69%, ~3h10min de cómputo) porque cerrar
VSCode mató el proceso en background sin ningún checkpoint intermedio
guardado (`save_strategy="epoch"` únicamente). Se relanzó desde cero sin
cambiar la estrategia de guardado (decisión explícita del usuario: no
guardar checkpoints intermedios) — el riesgo aceptado es perder la corrida
completa ante una interrupción, a cambio de no acumular carpetas de
checkpoint de varios GB en cada intento.

**Cómo correrlo:**

```bash
./.venv/Scripts/uv.exe run python src/entrenar_lora.py --n-ejemplos 0 --epocas 1
```

`--n-ejemplos 300` (el default) corre un subset chico para validar que el
pipeline funciona antes de comprometer horas al dataset completo —
verificado explícitamente en esta fase antes de cada corrida larga.

## 4. Verificación cualitativa: un bug real de comparación, y su fix

El primer intento de comparar "modelo base" vs. "modelo + LoRA"
(`comparar_base_vs_lora.py`) devolvió **respuestas idénticas** en los 3
ejemplos probados — señal aparente de que el adaptador no hacía nada.

**Investigación:** se verificó por separado que el adaptador sí tenía
efecto real, comparando logits crudos con y sin el adaptador activo
(`modelo.disable_adapter()`): diferencia máxima de ~22 en los logits, y
cambio real del token más probable (`"Lo"` → `"Respond"`). El adaptador
funcionaba — el bug estaba en el script de comparación.

**Causa raíz:** `PeftModel.from_pretrained(modelo_base, DIR_ADAPTADOR)`
envuelve el objeto `modelo_base` **in-place**, no crea una copia
independiente. Guardar una referencia a `modelo_base` antes de esa llamada y
usarla después para "generar sin LoRA" en realidad generaba con el mismo
objeto ya modificado — comparando LoRA contra sí mismo, no contra el modelo
base real.

**Fix:** usar el contexto `with modelo_lora.disable_adapter():` para
desactivar temporalmente el adaptador sobre el mismo objeto, en vez de
mantener dos referencias al modelo.

**Resultado tras el fix**, comparando el adaptador entrenado sobre split:

```
Ejemplo #3000 — "¿Quién pasa 1° del Grupo I?"
Real:          Francia. El resultado oficial fue Francia, por lo que obtuvo 1 puntos.
Modelo BASE:   "Lo siento, pero no tengo información específica..."
Modelo + LoRA: Francia. El resultado oficial fue Francia, por lo que obtuvo 1 puntos.  ✓ exacto
```

El modelo base (sin fine-tunear) se niega correctamente a responder — no
tiene ese conocimiento en sus pesos. El modelo con LoRA adoptó el formato de
redacción exacto del dataset sin que el prompt lo pidiera explícitamente, y
en este caso acertó también el contenido factual.

## 5. Evaluación cuantitativa: cuántos ejemplos responde bien, parcial o mal

Tres ejemplos sueltos no dan un panorama confiable. `evaluar_lora.py`
recorre los **659 ejemplos completos** del set de validación (reproducido
con el mismo split `seed=42`), parsea la respuesta generada y la real con
una regex sobre el formato fijo (`"Respondió {X}. El resultado oficial fue
{Y}, por lo que obtuvo {Z} puntos."`), y compara los 3 campos por separado
en vez de el texto completo carácter por carácter — un acierto de contenido
con redacción ligeramente distinta no debe contar como error.

**Clasificación:** `correcto` (los 3 campos coinciden), `parcial` (al menos
1 campo coincide), `incorrecto` (ningún campo coincide, o alguna de las dos
respuestas no matchea el patrón esperado).

**Resultado real:**

```
=== Resultado global (659 ejemplos) ===
correcto    :   95 (14.4%)
parcial     :  457 (69.3%)
incorrecto  :  107 (16.2%)

=== Por tipo de pregunta ===

marcador (170 ejemplos):
  correcto    :    0 (0.0%)
  parcial     :  156 (91.8%)
  incorrecto  :   14 (8.2%)

otro (489 ejemplos):
  correcto    :   95 (19.4%)
  parcial     :  301 (61.6%)
  incorrecto  :   93 (19.0%)
```

**Hallazgo central de la fase:** el modelo **nunca** acierta un marcador
exacto (0.0% correcto en 170 ejemplos), pero tampoco falla del todo — el
91.8% cae en "parcial" (formato correcto, probablemente acierta `puntos`
pero no el número exacto de goles). En preguntas con espacio de respuestas
más acotado (equipo/selección entre pocas opciones, tipo "otro"), el modelo
sí acierta completo en 19.4% de los casos.

No es un fallo del pipeline: es una limitación real y medible de capacidad,
esperable con `r=8`, 1 sola época, y un dominio (marcadores de fútbol) donde
el resultado es esencialmente arbitrario y no se puede inferir de un patrón
lingüístico — solo memorizar exactamente cada fila, algo que 1 época sobre
~5900 ejemplos no logra de forma consistente.

**Cómo correrlo** (~1-4h dependiendo de la carga del sistema, genera 659
respuestas de 60 tokens en CPU):

```bash
./.venv/Scripts/uv.exe run python src/evaluar_lora.py
```

**Salida:** `data/processed/evaluacion_lora.json` con el detalle completo
(instrucción, respuesta real, respuesta generada, clase, tipo) de los 659
ejemplos, para inspección posterior.

## 6. Pendiente

- Fase 3 del roadmap: evaluación y comparación formal — construir sobre el
  hallazgo de esta fase (0% correcto en marcadores) para decidir si vale la
  pena mejorar la técnica (más épocas, `r` más alto, `target_modules`
  ampliado a `k_proj`/`o_proj`/MLP) o si el resultado actual ya es
  suficiente para el objetivo de aprendizaje del proyecto (ver
  [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)).
- `lora_alpha=16` sigue sin explicarse en profundidad (factor de escala del
  ajuste LoRA, `ΔW` se aplica como `(alpha/r) × B×A`) — pendiente heredado
  de Fase 1, relevante si se ajusta ese hiperparámetro en Fase 3.
- No se probó `target_modules` ampliado ni otros valores de `r` en esta
  fase — el 0% de acierto en marcadores es candidato directo para esa
  exploración.
