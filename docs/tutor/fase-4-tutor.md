# Tutor — Fase 4: Merge y portafolio

## Explicación del tutor

**Qué resuelve `merge_and_unload()`, con una analogía simple.** Sin merge,
usar el modelo fine-tuneado es como tener un documento original más una
hojita de correcciones aparte, que se superpone cada vez que se lee (el
adaptador LoRA calculando `B×A` en cada forward pass). `merge_and_unload()`
es imprimir una copia nueva ya con las correcciones aplicadas de forma
permanente: el resultado es un `AutoModelForCausalLM` normal, sin ninguna
dependencia de `peft`, que ya no necesita las matrices `A`/`B` sueltas
porque su ajuste quedó sumado directamente a los pesos originales.

Es literalmente la misma fórmula vista en Fase 1
(`salida = W(x) + B(A(x))`), aplicada una sola vez sobre los pesos en vez de
en cada inferencia.

**El trade-off central, medido con números reales de este proyecto.** El
adaptador `_mlp_r16` solo pesa 270.7 MB en disco (solo las matrices `A`/`B`
de las capas con LoRA). El modelo fusionado pesa 5,899.7 MB — 22x más — porque
ahora incluye los 1.54B parámetros completos del modelo, no solo el ajuste.
A cambio, el modelo fusionado se carga y genera sin depender de `peft`, con
cero overhead de cómputo extra por capa.

**Por qué solo se fusionó una variante, no las 4.** Las otras 3 variantes de
Fase 3 (atención sola con 1 y 2 épocas, MLP con `r=8`) ya cumplieron su
propósito comparativo — fusionarlas no aporta ninguna información nueva,
solo GBs adicionales en disco. Fusionar solo el adaptador ganador
(`_mlp_r16`) es la decisión consistente con el resto del proyecto: mantener
solo lo que sirve para la demo final de portafolio.

**Validación del merge, no solo ejecutarlo y confiar.** Igual que en Fase 2
se detectó un bug real comparando "antes/después" de forma descuidada
(`PeftModel.from_pretrained` modificando el modelo in-place), aquí se generó
con el modelo fusionado usando `probar_modelo_fusionado.py` — sin cargar
`peft` en absoluto — sobre los mismos 3 ejemplos usados en
`comparar_base_vs_lora.py` de Fase 2. El resultado fue coherente con el
comportamiento ya documentado en Fase 3 (acierta contenido en "otro",
mantiene el techo de marcadores), confirmando que el merge no alteró el
comportamiento aprendido — solo cambió cómo está empaquetado.

## Resumen del usuario

(No se pidió resumen escrito en esta sesión.)

## Validación del tutor

No aplica en esta sesión (no hubo resumen del usuario que validar). Con esta
fase se cierra el roadmap completo de `PROYECTO_BASE.md` (Fase 0 → Fase 4):
dataset preparado, LoRA instrumentado y entendido matemáticamente,
adaptador entrenado con procedimiento riguroso (split train/val), 4
variantes comparadas con diseño experimental de una variable a la vez, y el
adaptador ganador fusionado y documentado como demo de portafolio en
[`README.md`](../../README.md).
