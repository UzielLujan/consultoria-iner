"""Serialization of tabular records to text sequences for MNRL training and zero-shot evaluation."""

from typing import List

import numpy as np
import pandas as pd


# Mapeo de columnas a bloques semánticos por CSV. Nombres exactos tal como aparecen en los CSVs
SEMANTIC_BLOCKS = {
    "comorbilidad": {
        "[BLK_ID]": ["nombre"],
        "[BLK_ADMIN]": ["expediente", "fechaing", "fechaegr"],
        "[BLK_CLIN]": [
            "diagnosticoprincipal", "cie101",
            "diagnostico2", "cie102", "dx2",
            "diagnostico3", "cie103", "dx3",
            "diagnostico4", "cie104", "dx4",
            "comorbi", "comorbicv",
            "obesidad", "obesidad1", "cardiopatia", "diabetes", "nefropatia",
            "eaperge", "tephap"
        ],
    },
    "econo": {
        "[BLK_ID]": ["NOMBRE_DEL_PACIENTE", "SEXO", "EDAD", "GRUPO_EDAD"],
        "[BLK_ADMIN]": [
            "EXP", "DIAS_ESTANCIA",
            "FECHA_INGRESO_INER", "FECHA_DE_ALTA_MEJORIA",
            "TOTAL_DE_INGRESOS", "TOTAL_DE_EGRESOS",
            "GASTO_TOTAL", "GASTO_DIARIO"
        ],
        "[BLK_CLIN]": ["RESULTADO", "ETIQUETAS_COVID", "MOTIVO_DE_EGRESO"],
        "[BLK_GEO]": [
            "ESTADO_RESIDENCIA", "CLAVE_GEOESTADISTICA_ESTATAL",
            "MUNICIPIO_RESIDENCIA", "CLAVE_GEOESTADISTICA_MUNICIPAL"
        ],
        "[BLK_SOCIO]": [
            "ESCOLARIDAD", "OCUPACION",
            "VULNERABILIDAD_SOCIOECONOMICA",
            "NIVEL_SOCIOECONOMICO",
            "DERECHOHABIENTE_Y/O_BENEFICIARIO"
        ],
    },
    "trabajo_social": {
        "[BLK_ID]": ["APELLIDO PATERNO", "APELLIDO MATERNO", "NOMBRE",
                     "EDAD", "FECHA DE NACIMIENTO", "GENERO"],
        "[BLK_ADMIN]": ["EXPEDIENTE", "NO. HISTORIA", "FECHA DE ELABORACIÓN", "AÑO", "FILA"],
        "[BLK_CLIN]": ["DIAGNOSTICO"],
        "[BLK_GEO]": ["DELEGACIÓN O MUNICIPIO PERMANENTE", "ESTADO / PAIS PERMANENTE"],
        "[BLK_SOCIO]": [
            "ESCOLARIDAD", "OCUPACIÓN",
            "DERECHOHABIENTE Y/O BENEFICIARIO",
            "TOTAL DE PUNTOS",
            "NIVEL SOCIOECONÓMICO"
        ],
    },
}

# Orden de serialización de bloques semánticos
SERIALIZATION_ORDER = ["[BLK_ID]", "[BLK_CLIN]", "[BLK_ADMIN]", "[BLK_GEO]", "[BLK_SOCIO]"]


def _format_value(val) -> str:
    """Formatea un valor para serialización.

    Nulos → "".
    Numéricos con decimal cero → entero ("0.0" → "0", "1.0" → "1").
    Floats reales → redondeados a 2 decimales ("18339.1945" → "18339.19"). Strings → strip.
    Aplica a todos los perfiles: es normalización de presentación para el tokenizador,
    no intervención sobre los datos — independiente del nivel de limpieza del CSV.
    """
    if pd.isna(val):
        return ""
    if isinstance(val, (int, float, np.integer, np.floating)):
        f = float(val)
        f = round(f, 2)
        if f == int(f):
            return str(int(f))
        return str(f)
    s = str(val).strip()
    try:
        f = float(s)
        f = round(f, 2)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return s


def _serialize_block(row: pd.Series, block_cols: List[str], block_name: str,
                     use_block_tokens: bool = True,
                     skip_null: bool = False) -> str:
    """Serializa un bloque semántico individual.

    Args:
        row: una fila de un DataFrame
        block_cols: lista de columnas del bloque (nombres exactos del CSV)
        block_name: nombre del bloque (ej: "[BLK_ID]")
        use_block_tokens: si False, omite tokens especiales (para zero-shot)
        skip_null: si True, omite las columnas con valor nulo (no emite placeholder NULL).
                   Eje independiente de use_block_tokens para experimento 2×2.

    Returns:
        Fine-tuning:  "[BLK_*] [COL] col1 [VAL] val1 [COL] col2 [VAL] val2 ..."
        Zero-shot:    "col1: val1 col2: val2 ..."
        Bloque vacío: "" en ambos casos — se omite completamente de la secuencia
    """
    block_values = []
    has_real_value = False

    for col in block_cols:
        if col not in row.index:
            continue

        val = _format_value(row[col])

        if val:
            has_real_value = True
            if use_block_tokens:
                block_values.append(f"[COL] {col} [VAL] {val}")
            else:
                block_values.append(f"{col}: {val}")
        elif not skip_null:
            if use_block_tokens:
                block_values.append(f"[COL] {col} [VAL] NULL")
            else:
                block_values.append(f"{col}: NULL")

    if not has_real_value:
        return ""

    content = " ".join(block_values)
    return f"{block_name} {content}" if use_block_tokens else content


def serialize_record(row: pd.Series, csv_name: str,
                     use_block_tokens: bool = True,
                     skip_null: bool = False) -> str:
    """Serializa un registro tabular a secuencia de texto.

    Args:
        row: Una fila de pandas (pd.Series con nombres de columnas)
        csv_name: Nombre del CSV — debe ser exactamente 'comorbilidad', 'econo' o 'trabajo_social'
        use_block_tokens: True → incluye tokens [BLK_*] (entrenamiento fine-tuned)
                          False → texto limpio sin tokens (zero-shot / baseline)
        skip_null: True → omite columnas con valor nulo (texto más compacto).
                   False → emite placeholder NULL para columnas vacías.

    Returns:
        Fine-tuning: "[BLK_ID] [COL] nombre [VAL] Juan García [BLK_ADMIN] [COL] expediente [VAL] 12345 ..."
        Zero-shot:   "nombre: Juan García expediente: 12345 ..."
        Bloques sin datos se omiten completamente de la secuencia.
    """
    blocks_def = SEMANTIC_BLOCKS.get(csv_name)
    if blocks_def is None:
        raise ValueError(f"csv_name '{csv_name}' no reconocido. Usar: 'comorbilidad', 'econo' o 'trabajo_social'.")

    serialized_blocks = []

    for block_name in SERIALIZATION_ORDER:
        if block_name in blocks_def:
            block_cols = blocks_def[block_name]

            # Priorizar NOMBRE_COMPLETO en Trabajo Social si está disponible (Perfiles Tesis1, Tesis2, Iner)
            if csv_name == 'trabajo_social' and block_name == '[BLK_ID]' and 'NOMBRE_COMPLETO' in row.index:
                block_cols = ["NOMBRE_COMPLETO"] + [c for c in block_cols if c not in ["APELLIDO PATERNO", "APELLIDO MATERNO", "NOMBRE"]]

            block_text = _serialize_block(row, block_cols, block_name, use_block_tokens, skip_null)
            if block_text:
                serialized_blocks.append(block_text)

    return " ".join(serialized_blocks)


_ZEROSHOT_EXCLUDE = {"record_id", "source_db"}


def serialize_record_zeroshot(row: pd.Series) -> str:
    """Serializa un registro para evaluación zero-shot.

    Itera las columnas en su orden natural (orden del CSV), omitiendo columnas
    auxiliares añadidas por build_dataset. Campos nulos se omiten por completo.

    Returns:
        "col1: val1 col2: val2 ..." — texto plano sin tokens especiales.
    """
    parts = []
    for col in row.index:
        if col not in _ZEROSHOT_EXCLUDE:
            val = _format_value(row[col])
            if val:
                parts.append(f"{col}: {val}")
    return " ".join(parts)
