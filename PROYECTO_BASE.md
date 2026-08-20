# Fine-Tuning con LoRA: Adaptando un LLM Pre-entrenado

## Objetivo

Aprender fine-tuning de LLMs con **LoRA** (Low-Rank Adaptation), de forma
100% local, adaptando un modelo pre-entrenado pequeño al corpus de la Polla
Mundial 2026. Proyecto formativo, mismo espíritu que
IA-tech: entender cada pieza a fondo, no
solo dejar el código corriendo.

## Por qué este proyecto (y cómo se relaciona con `IA-tech`)

En `IA-tech` se construyó un transformer **desde cero**: arquitectura,
entrenamiento y generación, todo propio, sobre un modelo de ~2.8M
parámetros. Ese proyecto enseña *cómo funciona por dentro* un LLM.

Este proyecto es el siguiente paso lógico: en la industria casi nadie
entrena un LLM desde cero — se parte de un modelo pre-entrenado grande (con
capacidad de lenguaje general ya aprendida en billones de tokens) y se
**adapta** a una tarea o dominio específico. Ese proceso de adaptación es el
fine-tuning, y LoRA es la técnica estándar actual para hacerlo sin
necesitar el hardware de una empresa grande.

## Restricciones de hardware (leer antes de elegir modelo/técnica)

Verificadas en esta máquina:
- **GPU:** Intel UHD 730 integrada, ~2GB de VRAM compartida, **sin soporte
  CUDA**.
- **RAM:** 32GB.
- **CPU:** procesador de escritorio estándar, sin aceleración específica de
  ML.

Consecuencia directa: **QLoRA con cuantización 4-bit vía `bitsandbytes` no
es viable aquí** — esa librería depende de kernels CUDA que solo corren en
GPUs NVIDIA. No es una limitación de "es más lento", es que el training
simplemente no arranca sin CUDA.

Por eso el camino de este proyecto es **LoRA "clásico" en CPU**, sin
cuantización agresiva: el modelo base se carga en fp32 (o bf16 si el
hardware lo soporta razonablemente), se congelan sus pesos originales, y
solo se entrenan las matrices de bajo rango que LoRA inserta. Sigue siendo
>99% menos parámetros entrenables que un fine-tuning completo, así que el
concepto central se aprende igual — el precio es tiempo de entrenamiento
(horas en vez de minutos), no viabilidad.

Esta restricción y su porqué se documentan explícitamente porque es en sí
misma una lección del proyecto: en ML aplicado, el hardware disponible
determina la técnica, no al revés.

## Conceptos clave a aprender (en orden)

1. **Modelos pre-entrenados y por qué partir de uno** — qué trae "de
   fábrica" un modelo como Qwen2.5 (conocimiento de lenguaje general) y qué
   no (comportamiento específico a un dominio/tarea).
2. **Fine-tuning completo vs. PEFT** — por qué reentrenar todos los pesos de
   un modelo de cientos de millones/billones de parámetros es costoso e
   innecesario para adaptar comportamiento, y qué es
   *Parameter-Efficient Fine-Tuning*.
3. **LoRA (Low-Rank Adaptation)** — congelar el modelo base, insertar
   matrices de bajo rango en las capas de atención, entrenar solo esas
   matrices (<1% de los parámetros totales). Intuición matemática de por
   qué una matriz de bajo rango puede capturar el "delta" de comportamiento
   necesario.
4. **Cuantización y QLoRA** — qué es cuantizar un modelo a 4-bit, por qué
   reduce el uso de VRAM drásticamente, y por qué esta máquina no puede
   usarlo (conexión directa con la sección de restricciones de hardware).
5. **Preparación de datos para fine-tuning de instrucciones** — formato
   prompt/respuesta, plantillas de chat (chat templates), tokenización
   específica del modelo base elegido.
6. **Entrenamiento con LoRA** — hiperparámetros propios de LoRA (rank `r`,
   `alpha`, `dropout`, qué capas objetivo/`target_modules`), loop de
   entrenamiento con `peft` + `transformers`.
7. **Evaluación del fine-tuning** — cómo verificar que el modelo realmente
   aprendió el dominio (comparar respuestas del modelo base vs. modelo con
   el adaptador LoRA cargado, mismas preguntas).
8. **Merge y portabilidad del adaptador** — cómo se fusiona (o no) un
   adaptador LoRA con el modelo base, y por qué mantenerlo separado es
   valioso (un adaptador pesa MBs, no GBs).

## Contexto y datos

El corpus se reutiliza del proyecto `IA-tech`: los datos de la Polla
Mundial 2026 (`data/processed/corpus.json` en ese repo), reformateados como
pares instrucción/respuesta aptos para fine-tuning (a diferencia de
`IA-tech`, donde el corpus alimentaba un RAG, aquí se usa para *adaptar los
pesos* del modelo directamente).

> **Nota sobre los datos:** igual que en `IA-tech`, los datos crudos no se
> versionan en este repo por contener información personal de los
> participantes del pool. El detalle de cómo obtener/regenerar el corpus se
> documenta en la Fase 0.

## Modelo base

Por decidir en la Fase 0, evaluando tiempos reales de carga/entrenamiento
en esta máquina. Candidatos, de más a menos conservador:

- **Qwen2.5-0.5B-Instruct** — el más chico, ya instruction-tuned, mejor
  punto de partida para iterar rápido y ver el efecto del fine-tuning con
  claridad.
- **Qwen2.5-1.5B-Instruct** — más capacidad de lenguaje, pero
  entrenamiento notablemente más lento en CPU pura.

## Roadmap por fases

### Fase 0 — Setup y preparación de datos
- Estructura del repo: `data/`, `src/`, `notebooks/`, `docs/`, `tests/`
- Entorno: Python + venv, `uv` (consistente con `IA-tech`)
- Elegir modelo base definitivo, según prueba real de carga en esta máquina
- Transformar el corpus de `IA-tech` a formato instrucción/respuesta
- Entregable: dataset de fine-tuning listo (JSON/JSONL) + modelo base
  cargando y generando texto sin fine-tuning (baseline)

### Fase 1 — Fundamentos de PEFT y LoRA
- Concepto: fine-tuning completo vs. PEFT, matemática de bajo rango de LoRA
- Explorar la arquitectura del modelo base elegido: dónde se insertan las
  matrices LoRA (`target_modules` — típicamente proyecciones de atención)
- Entregable: notebook/script que instrumenta el modelo con un adaptador
  LoRA vacío y cuenta parámetros entrenables vs. totales (verificar el
  "<1%")

### Fase 2 — Entrenamiento con LoRA
- Concepto: hiperparámetros de LoRA (`r`, `alpha`, `dropout`), loop de
  entrenamiento con `transformers.Trainer` o loop manual
- Entrenar el adaptador sobre el corpus de la Fase 0
- Entregable: adaptador LoRA entrenado y guardado (checkpoint pequeño, MBs)

### Fase 3 — Evaluación y comparación
- Concepto: cómo medir si el fine-tuning "funcionó" (cualitativo: mismas
  preguntas al modelo base vs. modelo + adaptador; opcionalmente alguna
  métrica cuantitativa simple)
- Entregable: script de comparación lado a lado, con ejemplos documentados

### Fase 4 — Merge y portafolio
- Concepto: fusión del adaptador con el modelo base (merge_and_unload) vs.
  mantenerlo separado, trade-offs
- README con diagrama de arquitectura, decisiones técnicas, demo
- Documentación pensada para enseñanza, igual que `IA-tech`

## Flujo de trabajo

- Planeación y discusión conceptual: chat
- Construcción guiada fase por fase: skill de tutor pedagógico (réplica de
  `tutor-fase` de `IA-tech`, adaptada a este proyecto) + VSCode + Claude
  Code
- Cada fase se documenta en `docs/fase-N-*.md` al cerrarse, y las sesiones
  de tutor quedan registradas en `docs/tutor/fase-N-tutor.md`
