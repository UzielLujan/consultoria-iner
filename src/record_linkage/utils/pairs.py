"""Construction and classification of candidate record pairs for entity resolution."""

import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.distance import Levenshtein

from record_linkage.utils.normalization import normalizar_nombre_v2


# Columnas de expediente y nombre por CSV (nombres exactos después de preprocessing)
_COL_MAP = {
    "comorbilidad":   ("expediente",          "nombre"),
    "econo":          ("EXP",                 "NOMBRE_DEL_PACIENTE"),
    "trabajo_social": ("EXPEDIENTE",           "NOMBRE_COMPLETO"),
}


def _detect_csv_type(df: pd.DataFrame) -> str:
    """Detecta el tipo de CSV a partir de sus columnas."""
    cols = set(df.columns)
    if "NOMBRE_DEL_PACIENTE" in cols:
        return "econo"
    if "NOMBRE_COMPLETO" in cols or "APELLIDO PATERNO" in cols:
        return "trabajo_social"
    return "comorbilidad"


def build_pairs_df(
    df_econo: pd.DataFrame,
    df_comor: pd.DataFrame,
    df_ts: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el DataFrame completo de pares candidatos cross-CSV.

    Genera todos los pares de registros que comparten expediente entre CSVs distintos,
    más los pares con expediente NaN en Económico que coinciden en nombre_norm contra
    cualquier otro source (Comorbilidad o Trabajo Social).

    Args:
        df_econo: DataFrame del CSV Económico (ya preprocesado)
        df_comor: DataFrame del CSV Comorbilidad (ya preprocesado)
        df_ts:    DataFrame del CSV Trabajo Social (ya preprocesado)

    Returns:
        DataFrame con una fila por par candidato y columnas:
            record_id_a, source_a, nombre_a, exp_a,
            record_id_b, source_b, nombre_b, exp_b,
            exp_shared (int64 o pd.NA), nombre_norm_a, nombre_norm_b, nan_exp (bool)

    Notas:
        - Los pares NaN (exp NaN en Económico, mismo nombre_norm en Comor o TS) se
          marcan con nan_exp=True y exp_shared=pd.NA. Solo Eco contribuye NaN porque
          Comor y TS no tienen expedientes NaN tras el preprocesamiento.
        - Los record_id son índices globales asignados aquí; dataset.py los unifica
          al construir el parquet final.
        - entity_id NO se asigna aquí — depende de classify_pairs() y assign_entity_ids().
    """
    frames = {
        "Económico":     (df_econo, _detect_csv_type(df_econo)),
        "Comorbilidad":  (df_comor, _detect_csv_type(df_comor)),
        "Trabajo Social": (df_ts,   _detect_csv_type(df_ts)),
    }

    # Construir tabla auxiliar con columnas normalizadas para cada CSV
    unified_rows = []
    record_id = 0
    for source_name, (df, csv_type) in frames.items():
        exp_col, nombre_col = _COL_MAP[csv_type]
        for _, row in df.iterrows():
            exp_raw = row.get(exp_col)
            nombre_raw = row.get(nombre_col, "")
            unified_rows.append({
                "record_id": record_id,
                "source":    source_name,
                "exp_raw":   exp_raw,
                "exp_int":   pd.to_numeric(exp_raw, errors="coerce"),
                "nombre_norm": normalizar_nombre_v2(nombre_raw),
            })
            record_id += 1

    aux = pd.DataFrame(unified_rows)
    aux["exp_int"] = aux["exp_int"].astype("Int64")

    # --- Pares con expediente compartido (no NaN) ---
    aux_valid = aux[aux["exp_int"].notna()].copy()

    # Deduplicar por (exp_int, source, nombre_norm) antes del merge — igual que v1.
    # Si un CSV tiene registros duplicados con mismo expediente y mismo nombre normalizado,
    # se toma el primero como representante. Los demás se reconectan en dataset._step_finalize.
    aux_dedup = aux_valid.drop_duplicates(subset=["exp_int", "source", "nombre_norm"]).reset_index(drop=True)

    merged = aux_dedup.merge(aux_dedup, on="exp_int", suffixes=("_a", "_b"))
    cross_mask = (
        (merged["record_id_a"] < merged["record_id_b"]) &
        (merged["source_a"] != merged["source_b"])
    )
    pairs_exp = merged[cross_mask].copy()
    pairs_exp["nan_exp"] = False
    pairs_exp["exp_shared"] = pairs_exp["exp_int"].astype("Int64")
    # exp_int es la clave del merge — pandas no la sufija, hay que duplicarla manualmente
    pairs_exp["exp_int_a"] = pairs_exp["exp_int"].astype("Int64")
    pairs_exp["exp_int_b"] = pairs_exp["exp_int"].astype("Int64")

    # --- Pares NaN: expediente NaN en Económico, mismo nombre_norm en otro source ---
    # Solo Eco contribuye registros NaN (Comor y TS no tienen NaN en expediente
    # tras el preprocesamiento). Generamos cruces simétricos contra ambos targets.
    econo_nan = aux[(aux["source"] == "Económico") & (aux["exp_int"].isna())].copy()
    econo_nan = econo_nan[econo_nan["nombre_norm"] != ""]  # excluir nombres vacíos

    nan_frames = []
    for target_source in ("Comorbilidad", "Trabajo Social"):
        target_all = aux[aux["source"] == target_source].copy()
        # Deduplicar por nombre_norm: si dos registros target tienen el mismo nombre
        # normalizado, se toma el primero como representante — evita duplicar el par.
        target_dedup = target_all.drop_duplicates(subset=["nombre_norm"]).copy()
        merged = econo_nan.merge(target_dedup, on="nombre_norm", suffixes=("_a", "_b"))
        nan_frames.append(merged)

    nan_merged = pd.concat(nan_frames, ignore_index=True)
    nan_merged["nan_exp"] = True
    nan_merged["exp_shared"] = pd.NA
    # nombre_norm es la clave del merge — pandas no la sufija, hay que duplicarla manualmente
    nan_merged["nombre_norm_a"] = nan_merged["nombre_norm"]
    nan_merged["nombre_norm_b"] = nan_merged["nombre_norm"]

    # Renombrar para unificar esquema
    cols_keep = [
        "record_id_a", "source_a", "nombre_norm_a", "exp_int_a",
        "record_id_b", "source_b", "nombre_norm_b", "exp_int_b",
        "exp_shared", "nan_exp",
    ]

    pairs_exp = pairs_exp[cols_keep].copy()

    nan_cols = {
        "record_id_a": "record_id_a",
        "source_a":    "source_a",
        "nombre_norm_a": "nombre_norm_a",
        "exp_int_a":   "exp_int_a",
        "record_id_b": "record_id_b",
        "source_b":    "source_b",
        "nombre_norm_b": "nombre_norm_b",
        "exp_int_b":   "exp_int_b",
        "exp_shared":  "exp_shared",
        "nan_exp":     "nan_exp",
    }
    pairs_nan = nan_merged[[c for c in nan_cols]].copy()

    pairs_all = pd.concat([pairs_exp, pairs_nan], ignore_index=True)
    pairs_all = pairs_all.rename(columns={
        "exp_int_a": "exp_a",
        "exp_int_b": "exp_b",
    })

    return pairs_all.reset_index(drop=True)


def classify_pairs(
    pairs_df: pd.DataFrame,
    umbral_jw: float = 0.88,
    umbral_lev: float = 0.85,
) -> pd.DataFrame:
    """Agrega columna 'criterio' con la etiqueta de clasificación de cada par.

    Aplica una cascada de tres criterios sobre los pares candidatos:
        1. llave_exacta    — nombre_norm_a == nombre_norm_b y nan_exp=False
        2. metrica_clasica — JW(a,b) >= umbral_jw  OR  Lev_ratio(a,b) >= umbral_lev
        3. no_confirmado   — ningún criterio resuelve el par, o nan_exp=True

    Los pares nan_exp=True van directamente a no_confirmado (no se aplica cascada).
    Las decisiones manuales (match/no_match) NO se aplican aquí; se incorporan en
    _step_finalize leyendo el xlsx editado.

    jw_score y lev_score se calculan para los 11,486 pares independientemente del
    criterio asignado — útil para auditoría del umbral y para poblar el array
    `scores` del JSON consolidado.

    Args:
        pairs_df:   DataFrame producido por build_pairs_df()
        umbral_jw:  Umbral mínimo de Jaro-Winkler (0–1). Calibrado empíricamente: 0.88
        umbral_lev: Umbral mínimo de ratio Levenshtein (0–1). Calibrado empíricamente: 0.85

    Returns:
        pairs_df con columnas adicionales:
            jw_score (float), lev_score (float), criterio (str)
    """
    result = pairs_df.copy()

    # --- Métricas clásicas sobre todos los pares ---
    result["jw_score"] = result.apply(
        lambda r: JaroWinkler.normalized_similarity(r["nombre_norm_a"], r["nombre_norm_b"]),
        axis=1,
    )
    result["lev_score"] = result.apply(
        lambda r: Levenshtein.normalized_similarity(r["nombre_norm_a"], r["nombre_norm_b"]),
        axis=1,
    )

    # --- Cascada de clasificación ---
    criterio = pd.Series([""] * len(result), index=result.index)

    # Capa 1: llave_exacta
    mask_exacta = (~result["nan_exp"]) & (result["nombre_norm_a"] == result["nombre_norm_b"])
    criterio[mask_exacta] = "llave_exacta"

    # Capa 2: metrica_clasica (solo pares no resueltos con exp válido)
    pending = criterio == ""
    no_nan = ~result["nan_exp"]
    mask_clasica = (
        pending & no_nan &
        ((result["jw_score"] >= umbral_jw) | (result["lev_score"] >= umbral_lev))
    )
    criterio[mask_clasica] = "metrica_clasica"

    # Capa 3: no_confirmado — todo lo que queda sin clasificar
    criterio[criterio == ""] = "no_confirmado"

    result["criterio"] = criterio

    return result
