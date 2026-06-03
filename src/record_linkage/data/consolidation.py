"""Generación del JSON consolidado entity-centric — entregable INER (Perfil A).

Materializa el "Producto 3: Base de Datos Consolidada" del INER como un único
arreglo JSON: un objeto por `entity_id`. Supersede el stub relacional previo
(esquema multi-tabla SQL) por decisión documentada en `design_decisions.md`
→ "Ubicación del generador del JSON consolidado entity-centric".

Schema v2 (oficial — refinado de `propuesta_entregable_JSON.md`, 2026-05-27):

    {
      "entity_id": int,
      "cluster_size": int,                 # = len(items)
      "decision": null,                    # veredicto por cluster, editable
      "items": [                           # 1 por registro, en orden de record_id
        {"item": int,                      # id estable del registro dentro del cluster (0-based)
         "source": str,
         "linking_values": {"nombre_norm": str, "exp": int | null},  # campos derivados comparados
         "record": { ...columnas crudas originales, sin 'source'... }}
      ],
      "scores": [                          # solo pares cross-source; recalculados (no extraídos)
        {"method": str,                    # nombre de un método en comparison_methods.REGISTRY
         "items": [i, j],                  # ids `item` de los dos registros (i < j)
         "value": float}
      ]
    }

Lógica pura, sin I/O. El entrypoint `scripts/build_consolidated_json.py` carga los
artefactos del pipeline + los CSV crudos y escribe el JSON en `<perfil>/deliverables/`.

Notas de diseño:
  - **Scores recalculados, no extraídos.** Se computan con los métodos de
    `comparison_methods.REGISTRY` sobre los campos de cada item (rapidfuzz sobre
    `nombre_norm`). Reproduce idénticos los valores que tenía `pairs_classified` pero
    hace el entregable autocontenido (sin depender de ese artefacto de etiquetado).
  - **Solo pares cross-source.** Se puntea cada par de items de bases distintas dentro
    del cluster. Los pares intra-fuente (dedup de la misma base) NO generan score; aun
    así, esos registros SÍ aparecen en `items` (membresía completa de la entidad).
  - **El expediente no es compuerta** en ningún método (los Dres. rechazaron filtrar por
    expediente). `exp` queda como dato visible en `linking_values`/`record`.
  - Heterogeneidad de columnas: cada `record` conserva solo las columnas propias de su
    fuente; no se imponen columnas comunes ni nulos artificiales (Contexto_Consultoria §7.1).
  - NaN / NaT / pd.NA → null.

Compatibilidad: `build_entity_objects(..., schema_version="v1")` reproduce el schema
histórico (arrays paralelos `linked_items`/`records`, scores extraídos de `pairs`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from record_linkage.data.comparison_methods import REGISTRY

# Schema JSON master (editado a mano; fuente de verdad). Vive con este módulo porque
# describe exactamente lo que `build_entity_objects` produce.
SCHEMA_PATH: Path = Path(__file__).parent / "consolidated_entities.schema.json"

# Métodos clásicos a exponer en `scores`, en orden estable por par.
# (nombre_en_json, columna_en_pairs_classified)
_SCORE_METHODS: tuple[tuple[str, str], ...] = (
    ("jw", "jw_score"),
    ("lev", "lev_score"),
)


def _json_safe(value: Any) -> Any:
    """Convierte un valor de pandas/numpy a un tipo nativo serializable a JSON.

    NaN / NaT / pd.NA / None → None. Escalares numpy → su equivalente Python.
    El resto (str, bool, int, float) se devuelve tal cual.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # valores no escalares no son NaN
    if hasattr(value, "item"):  # escalar numpy → escalar Python
        return value.item()
    return value


def _exp_to_int(exp: Any) -> int | None:
    """Normaliza un expediente a int o None (los exp NaN-TS de Económico → null)."""
    v = _json_safe(exp)
    return None if v is None else int(v)


def _build_pair_scores(pairs: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """Indexa pairs_classified por par no ordenado de record_ids → {jw_score, lev_score}."""
    pair_scores: dict[tuple[int, int], dict] = {}
    for row in pairs.itertuples(index=False):
        a, b = int(row.record_id_a), int(row.record_id_b)
        key = (a, b) if a < b else (b, a)
        pair_scores[key] = {"jw_score": row.jw_score, "lev_score": row.lev_score}
    return pair_scores


def _build_items(recs, raw_by_record_id) -> list[dict]:
    """Ensambla los items de un cluster (orden de record_id ascendente).

    Cada item: {item, source, linking_values:{nombre_norm, exp}, record:{...crudo...}}.
    El `item` es el id 0-based dentro del cluster, referenciado por `scores.items`.
    """
    items = []
    for idx, r in enumerate(recs):
        items.append({
            "item": idx,
            "source": r.source_db,
            "linking_values": {
                "nombre_norm": r.nombre_norm,
                "exp": _exp_to_int(r.exp_int),
            },
            "record": {k: _json_safe(v)
                       for k, v in raw_by_record_id[int(r.record_id)].items()},
        })
    return items


def _recompute_scores(items: list[dict], score_decimals: int) -> list[dict]:
    """Scores v2: recalcula cada método del REGISTRY sobre pares cross-source del cluster."""
    scores: list[dict] = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            if items[i]["source"] == items[j]["source"]:
                continue  # solo cross-source; intra-fuente no se puntea
            for method in REGISTRY:
                val = method.fn(items[i], items[j])
                if val is None:
                    continue  # el método no aplica a este par
                scores.append({
                    "method": method.name,
                    "items": [items[i]["item"], items[j]["item"]],
                    "value": round(float(val), score_decimals),
                })
    return scores


def _extract_scores(record_ids: list[int], pair_scores: dict, score_decimals: int) -> list[dict]:
    """Scores v1 (legacy): extrae jw/lev de pairs_classified para los pares presentes."""
    scores: list[dict] = []
    n = len(record_ids)
    for i in range(n):
        for j in range(i + 1, n):
            sc = pair_scores.get((record_ids[i], record_ids[j]))
            if sc is None:
                continue
            for method_name, col in _SCORE_METHODS:
                scores.append({
                    "method": method_name,
                    "items": [i, j],
                    "value": round(float(sc[col]), score_decimals),
                })
    return scores


def build_entity_objects(
    records_meta: pd.DataFrame,
    raw_by_record_id: dict[int, dict],
    pairs: Optional[pd.DataFrame] = None,
    score_decimals: int = 4,
    schema_version: str = "v2",
) -> list[dict]:
    """Construye la lista de objetos-entidad del JSON consolidado.

    Args:
        records_meta: una fila por registro, columnas
            [record_id, source_db, entity_id, nombre_norm, exp_int].
        raw_by_record_id: record_id global → fila cruda original {columna: valor} (sin 'source').
        pairs: solo requerido en schema_version='v1' (pairs_classified, para extraer scores).
        score_decimals: redondeo de los valores de similitud en `scores`.
        schema_version:
            'v2' (default) → `items` anidado + scores recalculados (REGISTRY) cross-source.
            'v1' → arrays paralelos `linked_items`/`records` + scores extraídos de `pairs`.

    Returns:
        Lista de dicts (uno por entity_id), ordenada por entity_id ascendente; dentro de
        cada entidad, registros en orden de record_id ascendente.
    """
    if schema_version not in ("v1", "v2"):
        raise ValueError(f"schema_version '{schema_version}' no reconocido. Usar 'v1' o 'v2'.")
    if schema_version == "v1" and pairs is None:
        raise ValueError("schema_version='v1' requiere el argumento `pairs` (pairs_classified).")

    pair_scores = _build_pair_scores(pairs) if schema_version == "v1" else None

    objects: list[dict] = []
    ordered = records_meta.sort_values("record_id")
    for entity_id, group in ordered.groupby("entity_id", sort=True):
        recs = list(group.itertuples(index=False))
        record_ids = [int(r.record_id) for r in recs]  # ascendente (frame ya ordenado)
        items = _build_items(recs, raw_by_record_id)

        obj = {"entity_id": int(entity_id), "cluster_size": len(items), "decision": None}
        if schema_version == "v2":
            obj["items"] = items
            obj["scores"] = _recompute_scores(items, score_decimals)
        else:  # v1 — arrays paralelos, scores extraídos
            obj["linked_items"] = [
                {"source": it["source"], "linking_values": it["linking_values"]} for it in items
            ]
            obj["scores"] = _extract_scores(record_ids, pair_scores, score_decimals)
            obj["records"] = [{"source": it["source"], **it["record"]} for it in items]

        objects.append(obj)

    return objects
