# Scripts GLB

Esta carpeta reúne los scripts de procesamiento de archivos GLB del proyecto. Los
modelos de entrada se guardan en `input_glbs` y cada ejecución crea una subcarpeta
independiente, identificada únicamente por fecha y hora, dentro de `outputs_glb`.

## Estructura

```text
scripts_glb/
├── input_glbs/                 # Modelos originales
├── outputs_glb/
│   └── AAAAMMDD_HHMMSS/        # Una carpeta por ejecución
│       ├── archivos generados
│       ├── manifest.json
│       └── manifest.md
├── README.md
├── detect_roof_planes.py
└── split_glb_by_axes.py
```

Los archivos originales no se modifican. Cada subcarpeta de resultados contiene
manifiestos que documentan la entrada, los parámetros, los planos de corte, los
archivos producidos y sus validaciones.

## Requisitos

- Python 3.
- Blender disponible mediante el comando `blender`.
- Shapely 2.1.2 para la unión y simplificación de polígonos. La instalación local
  utilizada por este proyecto se encuentra en `scripts_glb/.vendor`.

El script puede ejecutarse con Python normal. Cuando comienza, se vuelve a lanzar
automáticamente dentro de Blender para utilizar su API de geometría; no es necesario
escribir manualmente el comando de Blender.

## `split_glb_by_axes.py`

Genera uno o varios archivos a partir de un GLB original. Cada resultado conserva una
mitad del modelo después de cortarlo por el centro de uno de los ejes X, Y o Z que se
hayan solicitado. Los ejes se interpretan en el sistema de coordenadas de glTF/GLB,
donde Y es el eje vertical.

### Ejecución básica

Desde esta carpeta:

```bash
cd /home/usuario/projects/langsam_v1/scripts_glb
python3 split_glb_by_axes.py input_glbs/roof_model.glb
```

También se puede indicar únicamente el nombre si el archivo está en `input_glbs`:

```bash
python3 split_glb_by_axes.py roof_model.glb
```

Si `input_glbs` contiene exactamente un archivo GLB, puede omitirse la entrada:

```bash
python3 split_glb_by_axes.py
```

### Valores predeterminados

- `--axes x`: genera únicamente el corte correspondiente al eje X.
- `--keep negative`: conserva la mitad negativa de cada eje.
- `--cap yes`: intenta cerrar con una superficie plana la abertura generada por cada
  corte, incluso si el modelo original es una superficie abierta.
- `--output-root outputs_glb`: guarda las ejecuciones en `outputs_glb`.
- `--timezone America/Bogota`: utiliza esta zona horaria para los manifiestos y el
  nombre de la subcarpeta.

El centro del corte se calcula desde la caja envolvente global del modelo completo;
no se presupone que el modelo esté centrado en la coordenada cero.

### Opciones frecuentes

Procesar los tres ejes:

```bash
python3 split_glb_by_axes.py roof_model.glb --axes all
```

Procesar únicamente Y, únicamente Z o una combinación:

```bash
python3 split_glb_by_axes.py roof_model.glb --axes y
python3 split_glb_by_axes.py roof_model.glb --axes z
python3 split_glb_by_axes.py roof_model.glb --axes x z
```

Conservar la mitad positiva:

```bash
python3 split_glb_by_axes.py roof_model.glb --keep positive
```

Dejar abiertas las superficies cortadas:

```bash
python3 split_glb_by_axes.py roof_model.glb --cap no
```

Cerrar únicamente objetos que originalmente sean mallas sólidas/manifold:

```bash
python3 split_glb_by_axes.py roof_model.glb --cap auto
```

Consultar todas las opciones:

```bash
python3 split_glb_by_axes.py --help
```

## `detect_roof_planes.py`

Detecta regiones planas que probablemente corresponden a superficies exteriores de
techo. Trabaja directamente con la geometría: analiza normales, conectividad y
planitud. De forma predeterminada no filtra por una cámara ni por oclusión. No
depende de OCR ni de un servicio externo.

La ejecución calibrada para el modelo cortado es:

```bash
python3 detect_roof_planes.py input_glbs/roof_model_cut_x_keep_negative.glb
```

Cada ejecución crea una nueva subcarpeta de fecha y hora con estos archivos:

- `roof_planes.json`: planos exactos y simplificados, ecuaciones, contornos X/Y/Z y
  triangulaciones.
- `roof_planes_exact_only.glb`: regiones detectadas con su triangulación original.
- `roof_planes_exact_overlay.glb`: geometría exacta sobre el GLB original.
- `roof_planes_simplified_only.glb`: únicamente los polígonos fusionados y
  vectorizados.
- `roof_planes_simplified_overlay.glb`: versión simplificada sobre el modelo.
- `roof_planes_regularized_only.glb`: contornos simplificados con dientes y picos
  sustituidos por segmentos rectos cuando la validación geométrica lo permite.
- `roof_planes_regularized_overlay.glb`: versión regularizada sobre el modelo.
- `previews/exact.png`, `previews/simplified.png` y `previews/regularized.png`:
  vistas isométricas equivalentes.
- `previews/comparison_exact_vs_simplified.png`: comparación horizontal; exacto a
  la izquierda y simplificado a la derecha.
- `previews/comparison_simplified_vs_regularized.png`: simplificado a la izquierda
  y regularizado a la derecha.
- `previews/comparison_all_geometries.png`: exacto, simplificado y regularizado.
- `roof_planes_report.md`: resumen legible de áreas, pendientes y confianza.
- `manifest.json`: entrada, parámetros, hashes y archivos producidos.

Las coordenadas del JSON siguen glTF 2.0 con Y como eje vertical. Cada plano incluye
un contorno simplificado y también `mesh.vertices` y `mesh.triangles`, que permiten
reconstruir exactamente la región detectada incluso si su contorno es cóncavo o tiene
huecos.

Parámetros principales:

- `--roof-up negative_x`: referencia geométrica predeterminada para decidir qué
  superficies tienen orientación de techo.
- `--normal-direction signed`: conserva el sentido de las normales y permite
  diferenciar las direcciones positivas de las negativas. `two-sided` puede usarse
  explícitamente para tratar una normal y su inversa como equivalentes.
- `--visibility none`: analiza todo el modelo y no descarta planos por una vista.
- `--view-from negative_z`: coloca la cámara incluida en los GLB de validación. Solo
  controla la oclusión si se utiliza `--visibility directional`.
- `--max-pitch 75`: desviación angular máxima respecto a `--roof-up`.
- `--visible-part only`: conserva únicamente los triángulos realmente visibles desde
  el lado seleccionado cuando se activa `--visibility directional`. `full-plane`
  conserva completa una región que sea visible.
- `--angle-tolerance 3`: tolerancia entre triángulos vecinos del mismo plano.
- `--min-area 0.08`: área mínima de una región.
- `--min-faces 20`: número mínimo de triángulos.
- `--max-plane-rms 0.035`: desviación RMS máxima respecto al plano ajustado.
- `--min-view-visibility 0.25`: visibilidad mínima desde el lado seleccionado.
- `--boundary-simplify 0.02`: tolerancia para simplificar los contornos.
- `--geometry-output both`: genera las versiones exacta y simplificada. También
  admite `exact` o `simplified`.
- `--merge-angle 5`: diferencia angular máxima para fusionar regiones.
- `--merge-plane-distance 0.05`: separación máxima entre planos coplanares.
- `--merge-max-residual 0.06`: error puntual máximo permitido después de ajustar
  conjuntamente las regiones candidatas. Es el límite estricto y también la
  tolerancia usada para clasificar vértices compatibles.
- `--merge-residual-percentile 95`: percentil que debe permanecer dentro de la
  tolerancia anterior cuando solo existen unos pocos valores atípicos.
- `--merge-min-inlier-ratio 0.95`: exige que al menos el 95% de los vértices
  permanezca dentro de la tolerancia.
- `--merge-robust-max-residual 0.10`: límite absoluto de seguridad; evita que el
  criterio estadístico esconda desviaciones grandes.
- `--merge-min-boundary-outlier-ratio 0.30`: exige que al menos el 30% de los
  valores atípicos esté en bordes de la región.
- `--merge-gap 0.08`: separación pequeña que puede cerrarse entre parches.
- `--simplified-boundary-tolerance 0.04`: reducción de vértices del contorno.
- `--boundary-regularization lines`: genera en paralelo contornos regularizados;
  `none` conserva exactamente la geometría simplificada en esa tercera salida.
- `--regularization-tolerance 0.15`: distancia para sustituir dientes por una
  línea entre esquinas dominantes.
- `--regularization-min-iou 0.95`: coincidencia superficial mínima con el contorno
  simplificado.
- `--regularization-max-area-change 0.05`: limita el cambio de área al 5%.
- `--min-hole-area 0.01`: elimina huecos inferiores a esta área.

Estas distancias y áreas utilizan las unidades originales del modelo, porque glTF no
declara por sí mismo si una unidad equivale a metros, centímetros u otra escala. El
Los archivos `roof_planes_exact_overlay.glb` y
`roof_planes_simplified_overlay.glb` deben compararse visualmente cuando se procese
un modelo nuevo, ya que las tolerancias pueden necesitar calibración.

La versión exacta conserva la triangulación de origen. La simplificada proyecta cada
grupo sobre su plano, fusiona parches compatibles, limpia huecos pequeños y vuelve a
triangular únicamente el contorno reducido. Aunque glTF almacena triángulos, esta
reconstrucción reduce drásticamente la cantidad necesaria.

La fusión robusta no reemplaza las comprobaciones de orientación, separación y RMS.
Solo permite superar el máximo puntual estricto cuando casi todos los vértices son
coplanares, la desviación absoluta sigue siendo pequeña y los valores atípicos se
concentran principalmente en los bordes. El JSON registra `p95_error`, `p99_error`
y `merge_validation` para auditar la decisión.

La regularización del contorno es posterior a la fusión de planos. Une esquinas
dominantes mediante segmentos rectos, pero solo acepta el cambio cuando reduce
vértices, mantiene el IoU configurado y respeta el límite de cambio de área. El
JSON conserva en paralelo `simplified_roof_planes` y `regularized_roof_planes`.

Los valores admitidos por `--roof-up` y `--view-from` son `positive_x`,
`negative_x`, `positive_y`, `negative_y`, `positive_z` y `negative_z`. Las
coordenadas del archivo continúan siendo glTF Y-up, pero la pendiente de cada plano
se calcula respecto a la referencia elegida con `--roof-up`.

Para recuperar explícitamente el comportamiento con filtro direccional:

```bash
python3 detect_roof_planes.py \
  input_glbs/roof_model_cut_x_keep_negative.glb \
  --visibility directional \
  --view-from negative_x
```

### Comparación de las seis referencias roof-up

Para evaluar por separado las normales orientadas hacia los seis sentidos y generar
seis GLB, seis PNG y una hoja comparativa:

```bash
python3 detect_roof_planes.py \
  input_glbs/roof_model_cut_x_keep_negative.glb \
  --roof-ups all \
  --normal-direction signed
```

Esta modalidad no aplica filtros de visibilidad. Para cada orientación genera un GLB
`roof_planes_roof_up_<dirección>_overlay.glb` y otro
`roof_planes_roof_up_<dirección>_only.glb` que contiene exclusivamente los planos
detectados. Todos los resultados se renderizan
con la misma cámara isométrica, que solo sirve para la comparación visual. La carpeta
de ejecución contiene `roof_planes_by_roof_up.json`,
`roof_up_comparison_report.md`, doce archivos GLB, seis imágenes en `previews` y
`previews/comparison.png`.

La hoja se ordena así:

```text
positive_x | negative_x | positive_y
negative_y | positive_z | negative_z
```

### Calibración en seis direcciones

Para detectar una sola vez el conjunto global de planos y comparar su visibilidad
desde los seis lados:

```bash
python3 detect_roof_planes.py \
  input_glbs/roof_model_cut_x_keep_negative.glb \
  --views all \
  --visible-part full-plane
```

Esta modalidad conserva IDs y colores estables y genera seis archivos
`roof_planes_from_<dirección>_overlay.glb`, un `roof_planes_all_only.glb`, el archivo
`roof_planes_multiview.json` y seis PNG dentro de `previews`. La hoja
`previews/comparison.png` se ordena así:

```text
positive_x | negative_x | positive_y
negative_y | positive_z | negative_z
```

También se puede proporcionar una selección en vez de `all`, por ejemplo
`--views positive_y negative_z`. Las ejecuciones multivista requieren
`--visible-part full-plane` para que un mismo plano conserve la misma geometría,
identidad y color entre resultados.

## Convenciones para próximos scripts

Los scripts nuevos se guardarán directamente en `scripts_glb`. En lo posible,
seguirán las mismas convenciones: entradas en `input_glbs`, resultados en una nueva
subcarpeta `outputs_glb/AAAAMMDD_HHMMSS`, nombres descriptivos y manifiestos JSON y
Markdown que permitan reproducir cada ejecución.
