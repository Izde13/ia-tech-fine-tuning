# Tutor — Fase 1: Fundamentos de PEFT y LoRA

## Explicación del tutor

**Fine-tuning completo vs. PEFT**

`Qwen2.5-1.5B-Instruct` tiene ~1.54B parámetros. Fine-tuning completo
actualizaría los 1.54B en cada paso, con Adam guardando 2 estados extra por
parámetro — el problema real en CPU sin CUDA no es viabilidad sino tiempo.
PEFT parte de que adaptar el modelo a un dominio específico no requiere
tocar el conocimiento general de lenguaje ya aprendido, solo ajustar un
comportamiento acotado. LoRA (Hu et al., Microsoft Research, 2021) es la
técnica PEFT usada.

**Precisión importante:** LoRA no "reentrena una porción de los parámetros
existentes" — el modelo base queda completamente congelado
(`requires_grad=False` en los 1.54B) y se **agregan** matrices nuevas en
paralelo, cuyo resultado se suma a la salida original de la capa. El modelo
final = pesos originales (congelados) + delta aprendido por las matrices
LoRA. Por eso un adaptador entrenado pesa MBs, no GBs.

**Intuición matemática (rango bajo)**

Una capa lineal es `W` de `d×d`. LoRA restringe el ajuste `ΔW` a
factorizarse como `B(d×r) × A(r×d)`, con `r << d`. Con `d=1536`, `r=8`: `ΔW`
completa serían ~2.36M parámetros; `B×A` son ~24,576 (~96x menos en esa
capa). No es una conjetura, es una restricción impuesta a propósito,
validada empíricamente para tareas de adaptación — misma intuición que
PCA/SVD (la señal útil vive en pocas direcciones dominantes).

`A` se inicializa con valores aleatorios chicos, `B` en ceros → al arrancar,
`B×A=0`, el modelo se comporta exactamente como el base sin fine-tunear, y
el adaptador se "abre" gradualmente durante el entrenamiento.

**Dato curioso:** LoRA es del mismo año (2021) que los primeros fine-tunings
comerciales de GPT-3, cuando quedó claro que ajustar modelos de cientos de
miles de millones de parámetros de forma completa era económicamente
absurdo para la mayoría de casos de uso.

**Industria:** `peft` de HuggingFace es el estándar de facto hoy; Axolotl,
Unsloth y plataformas de fine-tuning gestionado usan LoRA o variantes por
debajo. QLoRA (LoRA + cuantización 4-bit) domina en GPU; este proyecto usa
LoRA "clásico" porque `bitsandbytes` no corre sin CUDA.

**Arquitectura real de Qwen2.5-1.5B-Instruct**

28 bloques transformer. Por bloque: `q_proj` (1536→1536), `k_proj`
(1536→256), `v_proj` (1536→256), `o_proj` (1536→1536), más MLP
(`gate_proj`, `up_proj`, `down_proj`). `k_proj`/`v_proj` salen a 256 por
Grouped Query Attention (GQA) — varias cabezas de query comparten un mismo
key/value, optimización estándar en modelos modernos, no parte de LoRA en
sí, pero afecta el tamaño de las matrices `A`/`B` insertadas ahí.

**Comparación explícita con el transformer propio de `IA-tech`:** `q_proj`,
`k_proj`, `v_proj`, `o_proj` de Qwen son la misma pieza, con el mismo rol,
que `W_query`, `W_key`, `W_value`, `W_output` del `MultiHeadAttention`
construido a mano en `IA-tech/src/transformer_model.py` — no una
arquitectura nueva que aprender, sino la misma a mayor escala (`d_model=128`
en `IA-tech` vs. 1536 en Qwen, 28 bloques apilados vs. pocos).

**Instrumentación con `peft` — resultado verificado:**

```
Modelo base: 1,543,714,304 parámetros entrenables (nada congelado aún)
Modelo + LoRA (r=8, target=[q_proj, v_proj]):
  Entrenables: 1,089,536
  Totales:     1,544,803,840
  Porcentaje:  0.0705%
```

Verificado a mano: 28 bloques × (q_proj: 24,576 + v_proj: 14,336) =
1,089,536, coincide exacto con lo reportado por `peft`. `v_proj` aporta
menos parámetros por la salida reducida de GQA (256 vs. 1536).

**Ejemplo mínimo (capa simulada `W` congelada, `d=4`, `r=2`):**

```
W(x) = [0.30, -0.10, 0.45, 0.20]   ← nunca cambia

Al iniciar (B=0): B(A(x)) = [0,0,0,0] → salida = W(x), idéntica al base
Tras un paso:      B(A(x)) = [0.01,-0.02,0.00,0.03]
                   nueva salida = [0.31,-0.12,0.45,0.23]
```

Backpropagation solo mueve `A`/`B` (marcadas `requires_grad=True`); `W`
tiene `requires_grad=False`, ningún ajuste se le aplica aunque haya
"contribuido" matemáticamente al error.

## Resumen del usuario

fase 1
o sea, lo que vamos a hacer es utilizar un modelo ya existente para re
entrenarlo, eso es fine tuning, pero en este caso, volverlo a reentrenarlo
es muy costoso, por lo que se usa PEFT, con lo que solo reentramos una menor
cantidad de parametro y usamos LoRA para esto

No estamos "reentrenando" el modelo en el sentido de tocar sus pesos
originales. Con LoRA, el modelo base queda **completamente congelado**
(`requires_grad=False` en sus ~1.54B parámetros — ni un gradiente los
toca). Lo que hacemos es **agregar** matrices nuevas, pequeñas, en paralelo
a ciertas capas existentes, e **inyectar** el resultado de esas matrices
nuevas sumado a la salida original de la capa.

Es decir, no es "de los 1.54B parámetros, entrenamos solo unos pocos" (eso
describiría mejor un fine-tuning parcial tipo "solo descongelo las últimas
capas"). Es más bien: "dejamos los 1.54B intactos, y le sumamos un pequeño
'parche' aprendido al lado". El modelo final = pesos originales
(congelados) + delta aprendido por las matrices LoRA.

Esa distinción importa para lo que viene: es la razón por la que un
adaptador LoRA entrenado pesa apenas unos MBs (solo esas matrices nuevas)
en vez de GBs (el modelo completo), y por la que en la Fase 4 se puede
"quitar" el adaptador y volver exactamente al modelo original sin pérdida

La intuición matemática de LoRA = multiplicar por una matriz pequeña

de todo el modelo Qwen (1.54 mil millones de parámetros), solo
**1,089,536** van a tener el gradiente activo y por lo tanto son los únicos
que se van a modificar cuando entrenemos en la Fase 2.

**El matiz que vale la pena tener claro:** esos 1,089,536 no son un
pedacito sacado _de adentro_ de los pesos originales de Qwen (no es que
agarremos 1,089,536 de sus números y los dejemos editables mientras el
resto queda fijo). Son parámetros **completamente nuevos** — las matrices
`A` y `B` que `peft` insertó al lado de `q_proj` y `v_proj` en cada uno de
los 28 bloques. Los pesos originales de Qwen (los 1.54B) quedan **todos**
congelados, ni uno se toca.

## Validación del tutor

**Lo que entendiste bien, y con precisión notable:**
- La corrección sobre "reentrenar" está perfectamente capturada, y no solo
  repetida — explicada con palabras propias, anticipando incluso la
  consecuencia correcta para Fase 4 (poder "quitar" el adaptador y volver
  exacto al modelo original).
- La distinción "modelo final = pesos congelados + delta aprendido" es
  exactamente la idea central de LoRA, dicha sin ambigüedad.
- El matiz de que los 1,089,536 son parámetros **nuevos**, no una porción de
  los 1.54B originales — repetido correctamente y sin que se escapara el
  detalle, que es justo donde más gente se confunde al conocer LoRA por
  primera vez.

**Lo que quedó incompleto — un vacío real, no cosmético:**
- La línea "la intuición matemática de LoRA = multiplicar por una matriz
  pequeña" es demasiado comprimida y pierde la parte que le da sentido a
  todo lo demás. No es solo "una matriz pequeña" — son **dos** matrices
  (`A` y `B`) en cascada, que factorizan una matriz de ajuste `ΔW` que
  *podría* ser de rango completo (`d×d`) en una forma de rango bajo
  (`d×r` y `r×d`, con `r << d`). La apuesta real de LoRA es que el ajuste
  necesario para adaptar el modelo no necesita toda la expresividad de una
  matriz completa. Sin este matiz, en Fase 2 podría surgir la pregunta "¿por
  qué no una sola matriz chica en vez de dos?" — la respuesta es que una
  sola matriz reducida no puede mantener las dimensiones de entrada/salida
  de la capa original; el truco es específicamente achicar con `A` y volver
  a expandir con `B` para que el resultado sea sumable directamente a la
  salida de `q_proj`/`v_proj`.

**Para profundizar (conectando con Fase 2):**
- `r` es el "cuello de botella" de esa factorización — cuanto más chico,
  menos capacidad tiene el ajuste para representar patrones complejos, pero
  menos parámetros entrenables. Tenerlo presente al evaluar si el modelo
  aprendió lo suficiente con `r=8`.
- `lora_alpha=16` se usó en el código de esta fase sin explicar su rol (el
  factor de escala del ajuste LoRA: `ΔW` se aplica como `(alpha/r) × B×A`)
  — queda pendiente para cuando su efecto sea observable durante el
  entrenamiento real.
- En Fase 2 habrá que decidir tasa de aprendizaje y número de épocas para el
  loop de entrenamiento — con `r` y `target_modules` ya fijados aquí.
