# consultoria-iner

Pipeline de procesamiento, etiquetado y consolidación de las tres bases de datos COVID-19 del INER. Componente de consultoría del proyecto de investigación en Record Linkage desarrollado como parte de mi formación en la maestría en Cómputo Estadístico del CIMAT Unidad Monterrey. Se distribuye como repositorio independiente del componente de aprendizaje automático, el cual vive en otro repositorio y consume el `dataset.parquet` producido aquí.

---

## Contexto

El INER proporcionó tres CSVs originales de registros de pacientes COVID-19 sin llave de identificación 100% confiable:

| Base | Registros | Contenido |
|------|-----------|-----------|
| Económico (Costos) | 4,632 | costos de atención, datos socioeconómicos, demográficos |
| Comorbilidad | 4,278 | diagnósticos clinicos y comorbilidades, fechas hospitalarias |
| Trabajo Social | 14,796 | más general: datos socioeconómicos, demográficos y geográficos |

Este repositorio provee dos artefactos derivados a partir de los datos crudos:

1. **`consolidated_entities.json`** (entregable principal INER) — arreglo de **15,283 entidades únicas**, cada una un cluster de registros vinculados al mismo paciente. Validable contra `consolidated_entities.schema.json` (JSON Schema Draft 2020-12).
2. **`dataset.parquet`** (insumo para el componente de aprendizaje automático) — versión serializada del ground truth con 11,466 pares positivos confirmados, lista para entrenamiento de modelos de ligado de entidades.

---

## Estructura del repositorio

```
consultoria-iner/
├── README.md
├── pyproject.toml
├── .env.example               # Plantilla para INER_DATA_ROOT
│
├── scripts/                   # Entrypoints CLI
│   ├── run_preprocessing.py        # M0–M7 modular → CSVs limpios
│   ├── run_dataset.py              # classify + finalize → entity_ids + dataset.parquet
│   ├── show_pair.py                # Inspector de pares para revisión manual
│   ├── merge_review_decisions.py   # Si cambian los pares candidatos transfiere decisiones manuales de un xlsx ya revisado a uno recién regenerado.
│   ├── build_consolidated_json.py  # → consolidated_entities.json
│   ├── build_data_dictionary.py    # → Diccionario_Final_INER.csv + metodos_comparacion.json + copia del schema
│   └── report_linking_numbers.py   # Cifras canónicas + figuras del reporte
│
├── src/record_linkage/
│   ├── config.py              # Rutas vía perfil_paths(perfil) → {clean, interim, output, deliverables}
│   ├── data/
│   │   ├── preprocessing.py   # Módulos M0–M7; profile_default = M0(strip)→M1→M4(TS)→M5
│   │   ├── dataset.py         # _step_classify + _step_finalize + build_dataset (orquestador)
│   │   ├── serialization.py   # serialize_record con 4 variantes (tok/notok × keep/skip null)
│   │   ├── consolidation.py   # build_entity_objects → JSON entity-centric
│   │   ├── comparison_methods.py  # REGISTRY: jw_nombre, lev_nombre + cos_biencoder (opcional)
│   │   └── consolidated_entities.schema.json   # Schema master (editable a mano)
│   └── utils/
│       ├── normalization.py   # normalizar_nombre
│       ├── pairs.py           # build_pairs_df, classify_pairs
│       └── entities.py        # count_entity_types
│
└── tests/
```

---

## Setup

### 1. Entorno

```bash
uv sync
```

`uv sync` crea `.venv/`, resuelve las dependencias declaradas en `pyproject.toml`,
genera o respeta `uv.lock` e instala el paquete local en modo editable. No se
requiere Micromamba ni Conda para ejecutar el pipeline local.

Ejecute los entrypoints dentro del entorno administrado por UV:

```bash
uv run python scripts/run_preprocessing.py --check-paths
```

Para herramientas opcionales de desarrollo (Jupyter, pytest y Ruff):

```bash
uv sync --extra dev
```

### 2. Variable de entorno `INER_DATA_ROOT`

Las bases originales y los artefactos derivados viven **fuera del repo** (los CSVs crudos no se publican por confidencialidad). Define la raíz de datos en un `.env` en la raíz del repo:

```bash
cp .env.example .env
# Editar .env:
INER_DATA_ROOT=/ruta/a/tus/datos/INER
```

Default si no se define: `~/Data/INER`.

### 3. Layout de datos esperado

```
$INER_DATA_ROOT/
├── raw/                                      # CSVs originales (no incluidos)
│   ├── INER_COVID19_CostoPacientes_Econo.csv
│   ├── INER_COVID19_Pacientes_DiagnosticoComorbilidad.csv
│   └── INER_COVID19_TrabajoSocial.csv
├── embeddings/                               # Artefactos externos (producidos por el componente de aprendizaje automático)
│   └── tok_skipnull/embeddings.parquet       # Requerido para --cosine
└── processed/<perfil>/                       # creado por el pipeline
    ├── clean/                                # output de preprocessing
    ├── interim/                              # output de classify
    ├── output/                               # output de finalize (entity_ids + variantes)
    └── deliverables/                         # output de los builders (bundle INER)
```

---

## Flujo de datos

```
[1] RAW CSVs                  ($INER_DATA_ROOT/raw/)
       │
       │  run_preprocessing.py  (profile_default: M0→M1→M4→M5)
       ▼
[2] CLEAN CSVs                (clean/)
       │
       │  run_dataset.py --step classify
       │     • normaliza nombres (utils/normalization.normalizar_nombre)
       │     • construye pares candidatos cross-CSV (utils/pairs.build_pairs_df)
       │     • clasifica por etapas: llave_exacta → metrica_clasica → no_confirmado
       ▼
[3] INTERIM                   (interim/)
       ├── records_interim.parquet      [record_id, source_db, text, exp_int, nombre_norm]
       ├── pairs_classified.parquet     [record_id_a, record_id_b, jw, lev, criterio, ...]
       └── pairs_for_review.xlsx        editable; columna 'decision' ∈ {match, no_match}
       │
       │  ⟵ REVISIÓN MANUAL: 514 pares 'no_confirmado' marcados a mano
       │
       │  run_dataset.py --step finalize
       │     • aplica decisiones manuales
       │     • union-find → entity_id
       │     • re-serializa la columna `text` con flags de serialización
       ▼
[4] OUTPUT                    (output/)
       ├── entity_ids.parquet           [record_id, source_db, entity_id]  (consultoría)
       └── <variant>/dataset.parquet    [record_id, source_db, text, entity_id]  (insumo para el componente de aprendizaje automático)
       │
       │  build_consolidated_json.py + build_data_dictionary.py + report_linking_numbers.py
       ▼
[5] DELIVERABLES              (deliverables/)
       ├── consolidated_entities.json        # 15,283 entidades (oficial)
       ├── consolidated_entities.schema.json # JSON Schema Draft 2020-12
       ├── Diccionario_Final_INER.csv        # proyección del schema
       ├── metodos_comparacion.json          # catálogo de métodos
       └── report_numbers.json               # cifras canónicas del reporte
```

---

## Comandos del pipeline

Asume `INER_DATA_ROOT` configurado y el env activo. Todos los scripts soportan `--perfil <nombre>` (default `default`).

### Preprocesamiento

```bash
python scripts/run_preprocessing.py
# → $INER_DATA_ROOT/processed/default/clean/{econo,comorbilidad,trabajo_social}_clean.csv
```

Para limpiar con otro perfil (hay que definirlo en `preprocessing.py`):

```bash
python scripts/run_preprocessing.py --perfil <nombre>
```

### Etiquetado: classify + finalize

- Paso 1. Clasificación automática de pares cross-CSV

```bash
python scripts/run_dataset.py --step classify --perfil default
# → interim/{records_interim, pairs_classified}.parquet + pairs_for_review.xlsx
```

- Paso 2. Revisión manual del `.xlsx` — solo columna `decision` en pares `no_confirmado`

- Paso 3. Aplicar decisiones y serializar

```bash
python scripts/run_dataset.py --step finalize --perfil default
# → output/entity_ids.parquet + output/tok_skipnull/dataset.parquet
```


Salvaguarda: `--step classify` está bloqueado si ya existe `pairs_for_review.xlsx` (protege las decisiones manuales). Para forzar re-clasificar hay que borrar el xlsx manualmente.

### Construcción del bundle de entregables

```bash
python scripts/build_consolidated_json.py --perfil default                          # genera consolidated_entities.json
python scripts/build_consolidated_json.py --perfil default --cosine                 # agrega cos_biencoder a scores
python scripts/build_data_dictionary.py --perfil default                            # Diccionario_Final + métodos + schema
python scripts/build_data_dictionary.py --perfil default --cosine                  # catálogo incluye cos_biencoder
python scripts/report_linking_numbers.py --perfil default                           # Cifras + figuras
```

### Inspección de pares durante revisión manual

```bash
python scripts/show_pair.py <record_id_a> <record_id_b> <source_a> <source_b>
# Ejemplo:
python scripts/show_pair.py 8749 14123 Económico Comorbilidad
```

---

## Modos de serialización

`run_dataset.py --step finalize` produce **una variante** de `dataset.parquet` por corrida, definida por dos flags ortogonales:

| Variante | Flags | Característica |
|----------|-------|----------------|
| **`tok_skipnull`** | (default) | con tokens `[BLK_*]`, omite columnas nulas |
| `tok_keepnull`   | `--keep-null` | con tokens, conserva nulos como placeholder `NULL` |
| `notok_skipnull` | `--no-special-tokens` | sin tokens, omite nulos |
| `notok_keepnull` | `--no-special-tokens --keep-null` | sin tokens, conserva nulos |

El **`entity_id`** es invariante entre variantes (union-find no depende del texto serializado). Por eso `entity_ids.parquet` se escribe una sola vez en `output/`, mientras `dataset.parquet` con la columna `text` vive en `output/<variant>/`.

Los entregables JSON, diccionario y reporte leen solo `entity_ids.parquet` — **son agnósticos a la variante**. La variante solo importa para el componente de aprendizaje automático.

---

## Etiquetado: criterios secuenciales

Cada par candidato cross-CSV recibe un `criterio`:

1. **`llave_exacta`** — comparten expediente + `nombre_norm` idéntico (9,855 pares).
2. **`metrica_clasica`** — comparten expediente + (Jaro-Winkler ≥ 0.88 ∨ Levenshtein ≥ 0.85) sobre `nombre_norm` (1,118 pares).
3. **`no_confirmado`** — no resueltos por las dos primeras capas; van a revisión manual (514 pares → 493 match + 21 no_match).

Cobertura adicional: pares Económico con `EXP` nulo cruzados por nombre contra Comorbilidad y Trabajo Social.

**Total positivos confirmados: 11,466.**

---

## Schema del JSON consolidado

`src/record_linkage/data/consolidated_entities.schema.json` es la **fuente de verdad** del entregable, editable a mano. JSON Schema Draft 2020-12.

Estructura (resumida):

```
entity_id      int            identificador único del cluster
cluster_size   int            número de items
decision       str|null       veredicto manual a nivel cluster (null por default)
items[]        objeto         un objeto por registro:
   ├── item           int        id estable dentro del cluster (0-based)
   ├── source         str        "Económico" | "Comorbilidad" | "Trabajo Social"
   ├── linking_values object     { nombre_norm, exp }
   └── record         objeto     registro crudo original (heterogéneo por fuente)
scores[]       objeto         comparaciones cross-source dentro del cluster:
   ├── pair    [i, j]          ids item comparados
   └── methods objeto          mapa método → score (ver metodos_comparacion.json)
```

Validación:

```python
import json, jsonschema
schema = json.load(open(".../consolidated_entities.schema.json"))
data   = json.load(open(".../consolidated_entities.json"))
jsonschema.validate(data, schema)
```

---

## Métodos de comparación

Registrados en `src/record_linkage/data/comparison_methods.py`. Base: `REGISTRY` (jw_nombre, lev_nombre). Opcional: `cos_biencoder` (se agrega vía `--cosine`).

| Nombre | Campos | Rango | Función |
|--------|--------|-------|---------|
| `jw_nombre`  | `nombre_norm` | [0, 1] | Jaro-Winkler normalizada |
| `lev_nombre` | `nombre_norm` | [0, 1] | Levenshtein normalizada |
| `cos_biencoder` | `record` | [-1, 1] | Coseno de embeddings del registro completo (opcional, requiere `--cosine`) |

Cada método es una función nombrada que recibe dos `item`s y devuelve un score (o `None` si no aplica). `jw_nombre` y `lev_nombre` están en `REGISTRY` por defecto. `cos_biencoder` se activa con `--cosine` y carga embeddings externos vía `make_cos_biencoder_method()`. Agregar un método nuevo = registrarlo en `REGISTRY` — el catálogo `metodos_comparacion.json` se regenera automáticamente en el siguiente `build_data_dictionary.py`.

---

## Cifras canónicas (perfil `default`)

Validables vía `python scripts/report_linking_numbers.py --perfil default`:

- 23,706 registros totales (4,632 + 4,278 + 14,796)
- 11,487 pares candidatos cross-CSV
- 9,855 + 1,118 + 514 = 11,487 (clasificación)
- 11,466 match + 21 no_match (post revisión manual)
- **15,283 entidades únicas** tras union-find
- 4,605 entidades vinculables (≥2 CSV): 1,184 en exactamente 2 + 3,421 en las 3 bases

---

## Notas

- Los CSVs crudos del INER **no se publican** por confidencialidad. Sin acceso a `$INER_DATA_ROOT/raw/`, el pipeline no se puede ejecutar de extremo a extremo, pero el código, la documentación de flujo y el JSON Schema sí son auditables.
- Los artefactos generados se encuentran en `deliverables/` bajo el perfil activo. `consolidated_entities.json` es el entregable oficial.
