# Fase 4 — Merge y portafolio

Objetivo de esta fase (ver [`PROYECTO_BASE.md`](../PROYECTO_BASE.md)): cerrar
el proyecto fusionando el adaptador LoRA ganador de la Fase 3
(`adaptador_lora_mlp_r16`) con el modelo base, entender el trade-off frente a
mantenerlo separado, y dejar documentación de portafolio (README con
arquitectura y demo).

## Archivos de código de esta fase

| Archivo | Qué hace |
|---|---|
| [`src/fusionar_adaptador.py`](../src/fusionar_adaptador.py) | Fusiona un adaptador LoRA con el modelo base vía `merge_and_unload()`, guarda el resultado y compara tamaño en disco |
| [`src/probar_modelo_fusionado.py`](../src/probar_modelo_fusionado.py) | Genera con el modelo fusionado, sin depender de `peft`, para validar que el merge preservó el comportamiento aprendido |

## 1. Qué hace `merge_and_unload()`

Hasta Fase 3, "usar el modelo fine-tuneado" significaba cargar dos piezas
por separado y calcular el ajuste al vuelo en cada forward pass:

```python
modelo_base = AutoModelForCausalLM.from_pretrained(NOMBRE_MODELO)
modelo = PeftModel.from_pretrained(modelo_base, "outputs/adaptador_lora_mlp_r16")
# en cada forward: W_efectivo(x) = W_base(x) + (alpha/r) × B(A(x))
```

`merge_and_unload()` calcula `W_nuevo = W_base + (alpha/r) × B×A` una sola
vez, de forma permanente, capa por capa, y descarta las matrices `A`/`B`
sueltas. El resultado es un `AutoModelForCausalLM` estándar — mismo tipo de
objeto que el modelo base, sin ninguna dependencia de `peft` para cargarlo o
generar con él.

Es la misma cuenta que ya se veía en Fase 1 (`salida = W(x) + B(A(x))`),
aplicada una vez sobre los pesos en vez de en cada inferencia.

## 2. Trade-off: adaptador separado vs. modelo fusionado

| | Adaptador separado | Modelo fusionado |
|---|---|---|
| Tamaño en disco (medido, Fase 4) | 270.7 MB | 5,899.7 MB (~22x) |
| Requiere `peft` para cargar | Sí | No |
| Cambiar de variante para comparar | Instantáneo (mismo modelo base en memoria) | Hay que fusionar y guardar cada variante por separado |
| Overhead de cómputo en inferencia | Suma extra `B×A` en cada forward (marginal) | Ninguno — inferencia idéntica a un modelo nativo |
| Servir varios adaptadores sobre un mismo modelo base | Trivial (cambiar cuál se carga) | Cada uno es un modelo completo aparte |

**Decisión para este proyecto:** fusionar solo el adaptador ganador
(`adaptador_lora_mlp_r16`, ver [Fase 3](fase-3-evaluacion-comparacion.md))
como demo de portafolio, y dejar las otras 3 variantes sin fusionar — ya
cumplieron su propósito comparativo y fusionarlas no aporta nada nuevo.

## 3. Ejecución real

```bash
uv run python src/fusionar_adaptador.py
```

Salida:

```
Fusionando adaptador con el modelo base (merge_and_unload)...
Modelo fusionado guardado en outputs/modelo_fusionado

=== Comparación de tamaño en disco ===
Adaptador LoRA solo :    270.7 MB  (outputs/adaptador_lora_mlp_r16)
Modelo fusionado    :   5899.7 MB  (outputs/modelo_fusionado)
Factor              : 22x más pesado
```

## 4. Validación: el modelo fusionado se comporta igual, sin `peft`

```bash
uv run python src/probar_modelo_fusionado.py
```

```
Ejemplo #0
Instrucción: ...Marcador Partido 1 – México vs Sudáfrica?
Respuesta real (dataset): Respondió 2-0. El resultado oficial fue 2-0, por lo que obtuvo 5 puntos.
Modelo fusionado (sin peft): 'Respondió 0-2. El resultado oficial fue 2-0, por lo que obtuvo 0 puntos.'

Ejemplo #100
Instrucción: ...Marcador Partido 3 – Canadá vs Bosnia y Herzegovina?
Respuesta real (dataset): Respondió 0-1. El resultado oficial fue 0-0, por lo que obtuvo 0 puntos.
Modelo fusionado (sin peft): 'Respondió 0-2. El resultado oficial fue 0-0, por lo que obtuvo 0 puntos.'

Ejemplo #3000
Instrucción: ...¿Quién pasa 1° del Grupo I?
Respuesta real (dataset): Respondió Francia. El resultado oficial fue Francia, por lo que obtuvo 1 puntos.
Modelo fusionado (sin peft): 'Respondió Francia. El resultado oficial fue Francia, por lo que obtuvo 1 puntos.'
```

El ejemplo #3000 (pregunta "otro", no marcador) acierta completo. Los
ejemplos #0 y #100 (marcadores) capturan el "resultado oficial" pero fallan
el marcador exacto de la respuesta del participante — consistente con el
techo estructural de marcadores ya documentado en Fase 3 (8.2% de acierto,
sin mejorar con más épocas ni `r` más alto). El merge no cambió ese
comportamiento, solo la forma de empaquetarlo — que es exactamente lo
esperado: `merge_and_unload()` no reentrenar nada, solo aplica de forma
permanente el mismo ajuste que ya calculaba el adaptador al vuelo.

## 5. Conclusión de la fase

El merge es una operación mecánica y sin sorpresas una vez que se entiende
la fórmula de Fase 1: aplicar `W + (alpha/r)×B×A` una vez en vez de en cada
forward pass. La decisión de fusionar o no depende del caso de uso, no de la
técnica en sí — para este proyecto, el adaptador separado sigue siendo la
forma correcta de versionar y comparar variantes (Fase 3), y el modelo
fusionado sirve como demo de portafolio de "cómo se vería en producción".

Con esta fase se cierra el roadmap completo de
[`PROYECTO_BASE.md`](../PROYECTO_BASE.md) (Fase 0 → Fase 4). El resumen de
portafolio del proyecto completo está en [`README.md`](../README.md).
