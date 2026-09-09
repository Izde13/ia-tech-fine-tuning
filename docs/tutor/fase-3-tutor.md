# Tutor — Fase 3: Evaluación y comparación

## Explicación del tutor

**Qué es `r` (el rango de LoRA), explicado a partir de la fórmula ya vista
en Fase 1.** `ΔW ≈ B(d×r) × A(r×d)`, con `r << d`. `r` controla el "ancho
del cuello de botella" por el que tiene que pasar cualquier ajuste que LoRA
haga a una capa — con `r=8`, cada ajuste está forzado a factorizarse a
través de solo 8 dimensiones intermedias. Subir `r` (a 16, por ejemplo) le
da al adaptador más espacio para representar ajustes complejos, a costa de
más parámetros entrenables y más tiempo de entrenamiento. Es una palanca
independiente de `target_modules`: `target_modules` decide *dónde* se puede
ajustar el modelo, `r` decide *cuánta* capacidad de ajuste se le da en cada
uno de esos lugares.

**Qué significa "ampliar `target_modules`" al MLP.** Cada bloque
transformer de Qwen tiene 4 capas de atención (`q_proj`, `k_proj`, `v_proj`,
`o_proj`) y 3 capas MLP/feed-forward (`gate_proj`, `up_proj`, `down_proj`,
mucho más anchas — 8960 vs. 1536/256). Hasta Fase 2, el adaptador solo
tocaba `q_proj`/`v_proj`; el resto seguía 100% congelado. Ampliar
`target_modules` significa insertar matrices LoRA también en las capas MLP
— motivado por el paper ROME (Meng et al., 2022), que muestra que el
conocimiento factual específico en transformers vive más en el MLP que en
atención, mientras que la atención se especializa más en relacionar
palabras entre sí.

**Diseño experimental: aislar una variable a la vez.** Se probaron 4
variantes, cada una cambiando solo una cosa respecto a la anterior: (1)
duplicar épocas manteniendo `target_modules` de Fase 2, (2) volver a 1
época pero ampliar `target_modules` al MLP, (3) mantener MLP y duplicar `r`.
`lora_alpha` se ajustó a `2×r` en todas para que el factor de escala del
ajuste (`alpha/r`) se mantuviera constante y `r` fuera realmente la única
variable — de lo contrario, subir `r` sin tocar `alpha` habría debilitado el
ajuste efectivo a la mitad, confundiendo el resultado.

**Resultado y su interpretación.** Épocas: casi sin efecto (14.4%→14.6%
correcto global). MLP: el salto más grande de la fase (→23.8%), en menos
tiempo de entrenamiento — confirma la hipótesis de ROME con datos propios,
no solo con la cita del paper. `r=16`: mejora el global y "otro" (→28.4%),
pero deja el acierto en marcadores exactamente igual (8.2%→8.2%) — la señal
más interesante de la fase: para ese tipo de pregunta, ni más capacidad
(`r`) ni más repetición (épocas) ayudan, sugiriendo un techo estructural
más que una falta de ajuste de hiperparámetros.

**Por qué marcadores parecen tener un techo.** Un marcador de fútbol es
esencialmente arbitrario respecto al texto de la pregunta — no hay patrón
lingüístico que lo explique, solo memorización directa de cada par
específico. Con 170 ejemplos de ese tipo y un modelo de 1.5B con LoRA en
CPU, ese tipo de memorización de baja frecuencia parece requerir algo
distinto a ajustar hiperparámetros dentro del mismo régimen — posiblemente
más cómputo del que la infraestructura de CPU local permite iterar
razonablemente.

**GPU/Colab y modelos más grandes: evaluado y descartado explícitamente.**
Se discutió migrar a Google Colab (GPU gratuita, reduciría el tiempo de
entrenamiento de horas a minutos) y/o usar un modelo base más grande. El
usuario decidió no hacerlo — la restricción de hardware (CPU local) es en
sí misma parte del objetivo de aprendizaje del proyecto, no solo un
obstáculo a evitar. Queda anotado como posible extensión futura fuera del
roadmap actual, no como pendiente de esta fase.

## Resumen del usuario

(No se pidió resumen escrito en esta sesión — el cierre de la fase se hizo
directamente con la documentación técnica en
`docs/fase-3-evaluacion-comparacion.md`.)

## Validación del tutor

No aplica en esta sesión (no hubo resumen del usuario que validar). El
hallazgo central — `target_modules` importó más que épocas o `r`, y
marcadores tienen un techo que ninguna de las 3 palancas probadas logró
mover — queda documentado como base para decidir el alcance de una futura
Fase 4 (merge y portafolio) o una posible exploración fuera de roadmap con
GPU si se retoma el problema de marcadores más adelante.
