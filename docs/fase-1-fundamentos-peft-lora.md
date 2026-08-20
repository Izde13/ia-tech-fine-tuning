# Fase 1 — Fundamentos de PEFT y LoRA

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)):
entender por qué el fine-tuning completo de un modelo de ~1.54B parámetros es
costoso e innecesario, la intuición matemática de LoRA (adaptación de bajo
rango), explorar dónde se inserta en la arquitectura real de
`Qwen2.5-1.5B-Instruct`, e instrumentar el modelo con un adaptador LoRA vacío
verificando en números reales el porcentaje de parámetros entrenables.

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/explorar_arquitectura.py`](../src/explorar_arquitectura.py) | Recorre los módulos del modelo cargado e imprime las capas `nn.Linear` de atención por bloque |
| [`src/instrumentar_lora.py`](../src/instrumentar_lora.py) | Aplica `LoraConfig` de `peft` al modelo base y cuenta parámetros entrenables vs. totales |

## 1. Por qué fine-tuning completo es un problema, no solo "caro"

`Qwen2.5-1.5B-Instruct` tiene ~1.54B parámetros. Un fine-tuning completo
actualizaría los 1.54B en cada paso de entrenamiento, con Adam guardando 2
estados extra por parámetro (momentum + varianza) — solo el estado del
optimizador en fp32 ronda 18+ GB, sin contar activaciones. El problema real
en esta máquina (CPU sin CUDA) no es viabilidad estricta sino tiempo: mover
gradientes por mil millones de parámetros en CPU pura es impracticable para
iterar.

**PEFT** (*Parameter-Efficient Fine-Tuning*) parte de una observación
distinta: adaptar un modelo a un dominio específico (la Polla Mundial 2026)
no requiere tocar su conocimiento general de lenguaje, ya aprendido — solo
ajustar un comportamiento acotado. **LoRA** (*Low-Rank Adaptation*, Hu et
al., Microsoft Research, 2021) es la técnica PEFT usada en este proyecto.

## 2. La intuición matemática: matrices de bajo rango

Una capa lineal del transformer es una matriz `W` de `d × d`. Fine-tuning
completo actualiza `W` directamente: `W_nuevo = W + ΔW`, con `ΔW` del mismo
tamaño que `W`.

**Hipótesis de LoRA:** `ΔW` (el ajuste de comportamiento necesario) tiene
rango intrínseco bajo — no necesita ser una matriz completa. Se puede
aproximar como el producto de dos matrices mucho más chicas:

```
ΔW (d × d)  ≈  B (d × r)  ×  A (r × d)      con r << d
```

Con `d = 1536` (dimensión de Qwen2.5-1.5B) y `r = 8`: `ΔW` completa serían
~2.36M parámetros; `B × A` son ~24,576 — una reducción de ~96x en esa capa
sola. No es una conjetura: es una restricción impuesta a propósito (forzar
`ΔW` a factorizarse como `B×A`), validada empíricamente en el paper para
tareas de adaptación (no para aprender conocimiento nuevo desde cero) —
misma intuición que PCA/SVD: la señal útil de un ajuste de este tipo vive en
pocas direcciones dominantes.

**Inicialización y su efecto:** `A` se inicializa con valores aleatorios
chicos, `B` en ceros. Al arrancar el entrenamiento, `B × A = 0`, así que
`ΔW = 0` y el modelo se comporta exactamente como el modelo base sin
fine-tunear — el adaptador empieza neutro y se abre gradualmente a medida
que el gradiente mueve `B` lejos de cero.

## 3. Dónde se inserta: arquitectura real de `Qwen2.5-1.5B-Instruct`

[`src/explorar_arquitectura.py`](../src/explorar_arquitectura.py) recorre
`model.named_modules()` para listar las capas `nn.Linear` reales del modelo
cargado, en vez de asumir nombres genéricos — `target_modules` de `peft`
requiere los nombres exactos de la familia de modelo concreta.

**Resultado (28 bloques transformer, primer bloque mostrado):**

```
self_attn.q_proj     in=1536  out=1536  bias=True
self_attn.k_proj     in=1536  out= 256  bias=True
self_attn.v_proj     in=1536  out= 256  bias=True
self_attn.o_proj     in=1536  out=1536  bias=False
mlp.gate_proj        in=1536  out=8960  bias=False
mlp.up_proj          in=1536  out=8960  bias=False
mlp.down_proj        in=8960  out=1536  bias=False
```

**Grouped Query Attention (GQA):** `k_proj`/`v_proj` salen a 256, no 1536
como `q_proj` — optimización estándar en modelos modernos (Qwen2.5, LLaMA 3)
donde varias cabezas de query comparten un mismo key/value, reduciendo
memoria y cómputo sin pérdida notable de calidad. No es parte de LoRA, pero
afecta directamente el tamaño de las matrices `A`/`B` insertadas ahí (una
salida de 256 tiene menos que "recuperar" que una de 1536).

**Comparación con el transformer propio de `IA-tech`:** `q_proj`, `k_proj`,
`v_proj`, `o_proj` de Qwen son la misma pieza que `W_query`, `W_key`,
`W_value`, `W_output` del `MultiHeadAttention` construido a mano en
`IA-tech/src/transformer_model.py` — mismo rol, mismo tipo de capa
(`nn.Linear`, multiplicar y sumar). La diferencia es escala: `d_model=128`
en `IA-tech` (65,536 parámetros en las 4 matrices de un bloque) vs.
`d_model=1536` en Qwen (millones por bloque, × 28 bloques ≈ 1.54B totales) —
no una arquitectura distinta que aprender de cero.

## 4. Instrumentación con `peft`: verificando el "<1%"

**Decisiones de diseño:**
- `target_modules=["q_proj", "v_proj"]`: configuración mínima del paper
  original de LoRA — Query y Value tienen mayor impacto en cómo se
  redistribuye la atención. Se dejan afuera `k_proj`, `o_proj` y el MLP para
  mantener el conteo de entrenables bajo (relevante en CPU); ampliar
  `target_modules` es un cambio de una línea si en Fase 2 el modelo no
  aprende lo suficiente.
- `r=8`: valor típico "chico pero razonable" en la industria (rango usual
  4–64), punto de partida antes de ajustar por resultados de entrenamiento.

[`src/instrumentar_lora.py`](../src/instrumentar_lora.py) carga el modelo
base, aplica `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj",
"v_proj"], task_type="CAUSAL_LM")` vía `get_peft_model`, y cuenta parámetros
con `requires_grad=True` antes y después.

**Resultado real:**

```
=== Modelo base, sin LoRA ===
Parámetros entrenables: 1,543,714,304  (todo el modelo, nada congelado aún)
Parámetros totales:     1,543,714,304

=== Modelo + adaptador LoRA (r=8, target=[q_proj, v_proj]) ===
Parámetros entrenables: 1,089,536
Parámetros totales:     1,544,803,840
Porcentaje entrenable:  0.0705%
```

`get_peft_model` congela automáticamente los 1.54B pesos originales
(`requires_grad=False`) y deja gradiente activo solo en las matrices `A`/`B`
nuevas — 0.0705%, bien por debajo del "<1%" objetivo de
`PROYECTO_BASE.md`.

**Verificación a mano del conteo** (28 bloques × [`q_proj` + `v_proj`]):

```
q_proj (1536→1536): A(1536×8) + B(8×1536)  = 24,576
v_proj (1536→256,   A(1536×8) + B(8×256)   = 14,336
        por GQA):
por bloque: 24,576 + 14,336 = 38,912
× 28 bloques = 1,089,536   ✓ coincide con lo reportado por peft
```

`v_proj` aporta menos parámetros LoRA que `q_proj` precisamente por la
salida reducida de GQA (256 en vez de 1536) — el detalle notado al explorar
la arquitectura tiene consecuencia directa en este número.

## 5. Ejemplo mínimo: cómo se combina el camino congelado y el camino LoRA

Con una capa simulada `W` (congelada, ya "entrenada") de entrada/salida 4,
`r=2`, entrada `x = [1.0, 0.5, -1.0, 2.0]`:

```
W(x) = [0.30, -0.10, 0.45, 0.20]        ← nunca cambia, W está congelada

Al iniciar entrenamiento (B en ceros):
A(x)     = [0.12, -0.05]
B(A(x))  = [0, 0, 0, 0]                  ← B=0 ⇒ ajuste nulo al inicio

salida = W(x) + B(A(x)) = [0.30, -0.10, 0.45, 0.20]  (idéntica al modelo base)
```

Tras un paso de entrenamiento (backpropagation solo mueve `A`/`B`; `W` está
marcada `requires_grad=False`, ningún ajuste se le aplica aunque haya
"contribuido" al error):

```
B(A(x)) = [0.01, -0.02, 0.00, 0.03]
nueva salida = [0.31, -0.12, 0.45, 0.23]
```

`W` permanece intacta en todo momento. Por eso un adaptador LoRA entrenado
pesa MBs (solo se guardan `A`/`B`) y no GBs (nunca se vuelve a guardar `W`).

## 6. Cómo correrlo

```bash
./.venv/Scripts/uv.exe run python src/explorar_arquitectura.py
./.venv/Scripts/uv.exe run python src/instrumentar_lora.py
```

## 7. Pendiente

- Fase 2 del roadmap: entrenamiento real del adaptador LoRA sobre
  `dataset_finetuning.jsonl` (Fase 0) — hiperparámetros de LoRA ya fijados
  aquí (`r=8`, `target_modules=["q_proj","v_proj"]`), falta definir el loop
  de entrenamiento (`transformers.Trainer` vs. loop manual), tasa de
  aprendizaje, número de épocas (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)).
- `lora_alpha=16` se usó en `instrumentar_lora.py` sin explicar su rol (el
  factor de escala del ajuste LoRA, `ΔW` se aplica como `(alpha/r) × B×A`) —
  queda para explicar en Fase 2, cuando su efecto en la magnitud del
  aprendizaje sea observable durante el entrenamiento real.
