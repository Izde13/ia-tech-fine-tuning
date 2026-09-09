# Tutor — Fase 2: Entrenamiento con LoRA

## Explicación del tutor

**Qué hace `Trainer` y por qué es el mismo loop que ya viste en `IA-tech`.**
Forward → loss → backward → `optimizer.step()`, repetido por batches y
épocas — la única diferencia con LoRA es que el modelo base está congelado
(`requires_grad=False`), así que el gradiente solo mueve las matrices
`A`/`B` del adaptador, nunca los 1.54B pesos originales de Qwen.

**Por qué hace falta un split train/validación.** La loss durante el
entrenamiento se mide sobre los mismos datos que ya vio el gradiente — no
distingue "aprendió el patrón" de "memorizó exactamente estas filas". Se
reservó el 10% del dataset (`seed=42`, reproducible) como validación: nunca
entra al backward pass, solo se usa para medir `eval_loss` cada 100 pasos.
Si `eval_loss` sube mientras `train_loss` sigue bajando, esa es la firma
clásica de overfitting — no ocurrió en esta fase: la `eval_loss` bajó de
forma monótona toda la época (0.512 → 0.160).

**Un bug real en la comparación cualitativa, no en el entrenamiento.**
`PeftModel.from_pretrained(modelo_base, ...)` modifica `modelo_base`
in-place — guardar una referencia previa y usarla después para "comparar
sin LoRA" en realidad comparaba el adaptador contra sí mismo. Se verificó
con logits crudos (usando `disable_adapter()`) que el adaptador sí tenía
efecto real (diferencia de ~22 en logits, cambio del token más probable)
antes de asumir que el entrenamiento había fallado. El fix fue usar
`with modelo.disable_adapter():` en vez de mantener dos referencias al
modelo.

**Evaluación cuantitativa sobre los 659 ejemplos de validación, no solo 3
sueltos.** Se parseó cada respuesta (generada y real) con la regex del
formato fijo del dataset, comparando 3 campos por separado
(`respuesta_participante`, `resultado_oficial`, `puntos`) en vez de texto
completo — una redacción distinta con el mismo contenido no debe contar
como error. Resultado: 14.4% correcto global, con una diferencia marcada
por tipo de pregunta — 0.0% correcto en preguntas de marcador (espacio de
respuestas amplio y arbitrario) vs. 19.4% en preguntas de "quién
gana/pasa" (espacio acotado a pocas selecciones). El modelo aprendió bien
la *forma* de responder (solo 16.2% totalmente incorrecto) pero tiene
capacidad limitada para memorizar contenido exacto y arbitrario como un
marcador de fútbol, con `r=8` y 1 sola época.

**Incidente de infraestructura:** cerrar VSCode mató un entrenamiento en
background al 69% de progreso (~3h10min perdidas) porque no había
checkpoint intermedio guardado — decisión consciente del usuario de no usar
`save_strategy="steps"` para no acumular carpetas pesadas en cada intento,
aceptando el riesgo de perder la corrida completa ante una interrupción.

## Resumen del usuario

(No se pidió resumen escrito en esta sesión — el cierre de la fase se hizo
directamente con la documentación técnica en
`docs/fase-2-entrenamiento-lora.md`.)

## Validación del tutor

No aplica en esta sesión (no hubo resumen del usuario que validar). El
hallazgo central de la fase — 0% de acierto exacto en marcadores, 91.8%
parcial — queda documentado como el dato más importante a tener presente
al entrar a Fase 3, donde se decide si esa limitación amerita ajustar la
técnica (más épocas, `r` más alto, `target_modules` ampliado) o si es un
resultado aceptable para el objetivo de aprendizaje del proyecto.
