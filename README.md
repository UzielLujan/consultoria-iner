# entity-resolution-nlp

Sistema de ligado de registros (Record Linkage) basado en aprendizaje profundo y NLP, aplicado a bases de datos de pacientes COVID-19 del INER. Proyecto de tesis de maestría en Cómputo Estadístico — CIMAT Unidad Monterrey.

---

## Contexto

El INER cuenta con tres bases de datos independientes de pacientes COVID-19 que no comparten una llave de identificación 100% confiable. Este proyecto construye un sistema moderno para vincularlas a nivel semántico usando modelos de lenguaje pre-entrenados (Bi-Encoder + Cross-Encoder), sin depender de coincidencias exactas de campos.

| CSV | Registros | Contenido |
|-----|-----------|-----------|
| Diagnósticos y Comorbilidades | 4,278 | diagnóstico principal, comorbilidades, fechas |
| Costos y Económico | 4,632 | costos de atención, datos socioeconómicos |
| Trabajo Social | 14,796 | datos demográficos, familia, situación social |

Ground truth: **4,341 entidades vinculables** (presentes en 2+ bases), **9,855 pares positivos confirmados** por llave determinista `(expediente, nombre_v2_normalizado)`.

---

## Estructura del repositorio

```
entity-resolution-nlp/
├── run_beto_baseline.sh       # Script SLURM — Bi-Encoder BETO
├── run_roberta_bio.sh         # Script SLURM — Bi-Encoder RoBERTa-biomedical
│
├── src/record_linkage/
│   ├── config.py              # Rutas centralizadas (DATA_ROOT, PROCESSED_DIR, etc.)
│   ├── data/
│   │   ├── preprocessing.py   # Módulos M0–M7, perfiles iner/tesis0/tesis1/tesis2
│   │   ├── dataset.py         # Serialización bloques semánticos + entity_id + .parquet
│   │   ├── augmentation.py    # 5 operadores on-the-fly: shuffle, mask, typos, delete
│   │   ├── splitting.py       # Partición train/val/test estratificada por entidad
│   │   └── consolidation.py   # Base relacional INER (pendiente de implementación)
│   ├── models/
│   │   ├── biencoder.py       # build_biencoder() + encode_texts() — backbone intercambiable
│   │   └── crossencoder.py    # Cross-Encoder DITTO — Etapa 2 (pendiente)
│   ├── training/
│   │   ├── train_biencoder.py # Entrenamiento MNRL: warm init, LR diferencial por capa, fp16
│   │   └── bce.py             # Binary Cross-Entropy — Etapa 2 (pendiente)
│   ├── utils/
│   │   └── mnrl.py            # Diagnóstico visual: matrices de similitud por batch
│   └── inference/
│       ├── retrieval.py       # Búsqueda vectorial ANN (pendiente)
│       └── reranking.py       # Re-ranking Cross-Encoder (pendiente)
│
├── scripts/
│   ├── download_model.py      # Descarga modelos de HuggingFace como SentenceTransformer
│   ├── run_preprocessing.py   # CLI: limpieza de CSVs crudos
│   ├── run_dataset.py         # CLI: construcción del dataset .parquet
│   ├── run_splitting.py       # CLI: partición train/val/test
│   ├── run_train_biencoder.py # CLI: entrenamiento Bi-Encoder con MNRL
│   ├── evaluate_zeroshot.py   # Evaluación zero-shot: Hit@K, MRR, Δ separabilidad
│   └── train_crossencoder.py  # CLI Cross-Encoder (pendiente)
│
├── notebooks/                 # EDAs y análisis de duplicados
└── docs/                      # Documentación del proyecto
```

---

## Inicio rápido

### 1. Entorno

```bash
micromamba create -n tesis python=3.11 -c conda-forge -y
micromamba activate tesis
pip install uv
uv pip install -e .
```

El flag `-e` instala `record_linkage` en modo editable — los cambios en `src/` se reflejan sin reinstalar. Dependencias de desarrollo (jupyter, pytest, ruff) se instalan con `uv pip install -e ".[dev]"`.

### 2. Descargar modelos

Los modelos se descargan localmente como SentenceTransformer con tokens especiales ya registrados, para poder transferirlos al cluster sin acceso a internet.

```bash
python scripts/download_model.py --all
```

Modelos disponibles: `BETO`, `RoBERTa-biomedical`, `paraphrase-multilingual`.

### 3. Preprocesamiento

```bash
# Verificar rutas
python scripts/run_preprocessing.py --perfil tesis1 --check-paths

# Limpiar CSVs
python scripts/run_preprocessing.py --perfil tesis1
```

Perfiles disponibles:

| Perfil | Módulos | Uso |
|--------|---------|-----|
| `iner` | M0(upper)→M1→M2→M3→M4→M5→M6→M7 | Entregables de consultoría INER |
| `tesis0` | M0(strip)→M1 | Base para evaluación zero-shot |
| `tesis1` | M0(strip)→M1→M4(si TS)→M5 | Entrenamiento — columnas originales del CSV |
| `tesis2` | M0(strip)→M1→M2→M4(si TS)→M5→M7 | Limpieza + renombrado semántico |

### 4. Construir dataset

```bash
# Fine-tuning (con tokens [BLK_*])
python scripts/run_dataset.py --perfil tesis1

# Zero-shot (texto col:val sin tokens)
python scripts/run_dataset.py --perfil tesis0 --no-special-tokens
```

Salida: `dataset.parquet` con columnas `record_id`, `source_db`, `text`, `entity_id`.

### 5. Partición train/val/test

```bash
python scripts/run_splitting.py --perfil tesis1
```

Estratificada por entidad (sin entity leakage). Proporciones por defecto: 70/15/15.

### 6. Evaluación zero-shot

```bash
python scripts/evaluate_zeroshot.py --model BETO
python scripts/evaluate_zeroshot.py --all
```

Métricas: Hit@K (K∈{1,5,10,20,50}), MRR y Δ separabilidad sobre los 9,855 pares confirmados cross-database.

### 7. Entrenamiento (local / HPC)

```bash
# Smoke test local (1 época, batch pequeño)
python scripts/run_train_biencoder.py --model BETO --epochs 1 --batch-size 8 --n-aug 0

# HPC — editar run_beto_baseline.sh con los parámetros deseados y lanzar con sbatch
sbatch run_beto_baseline.sh
```

---

## Serialización

Cada registro tabular se convierte en una secuencia de texto estructurada con bloques semánticos:

```
[BLK_ID] [COL] nombre [VAL] GARCIA LOPEZ MARIA [BLK_ADMIN] [COL] expediente [VAL] 12345 [BLK_CLIN] ...
```

Con `--no-special-tokens` (zero-shot):

```
nombre: GARCIA LOPEZ MARIA expediente: 12345 ...
```

Los tokens `[BLK_ID]`, `[BLK_CLIN]`, `[BLK_GEO]`, `[BLK_ADMIN]`, `[BLK_SOCIO]`, `[COL]` y `[VAL]` se registran en el tokenizador al descargar los modelos.

---

## Documentación

| Documento | Contenido |
|-----------|-----------|
| `MEMORY.md` | Bitácora del proyecto: historial de fases, métricas, preguntas abiertas |
| `docs/design_decisions.md` | Decisiones técnicas tomadas y su justificación |
| `docs/Metodologia_arquitectura.md` | Arquitectura neuronal: SBERT, DITTO, MNRL, serialización |
| `docs/Contexto_Maestro_Proyecto.md` | Visión general, roadmap, rutas de artefactos |
| `docs/Contexto_Consultoria_INER.md` | Objetivos y entregables de la consultoría |
| `docs/entorno_y_dependencias.md` | Entorno Python: micromamba, uv, pyproject.toml |

---

## Estado actual

- [x] EDA de las tres bases (Comorbilidad, Económico, Trabajo Social)
- [x] Ground truth — 4,341 entidades vinculables, 9,855 pares confirmados cross-database
- [x] Pipeline de preprocesamiento (perfiles iner, tesis0, tesis1, tesis2)
- [x] Serialización con bloques semánticos → `dataset.parquet` (23,706 registros)
- [x] Partición train/val/test estratificada por entidad
- [x] Augmentación on-the-fly (5 operadores: shuffle_blocks, shuffle_columns, mask, typos, delete_span)
- [x] Bi-Encoder — `build_biencoder()` con backbone intercambiable (BETO, RoBERTa-biomedical)
- [x] Pipeline de entrenamiento MNRL (warm init, LR diferencial por capa, mixed precision fp16, early stopping)
- [x] Evaluación zero-shot — Hit@K, MRR, Δ separabilidad
- [x] Experimentación en HPC — búsqueda de hiperparámetros con BETO y RoBERTa-biomedical
- [ ] Modelo final Bi-Encoder — pendiente de realizar entrenamiento definitivo y definición del pipeline de evaluación y métricas finales
- [ ] Pipeline de evaluación fine-tuned — diseño e implementación pendiente
- [ ] Rediseño de augmentación — operadores actuales no simulan variación real entre CSVs del INER
- [ ] Cross-Encoder como re-ranker (Etapa 2)
- [ ] Indexación vectorial ANN e inferencia en producción
