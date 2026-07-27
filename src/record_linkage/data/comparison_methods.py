"""Protocolo extensible de métodos de comparación para el array `scores` (entregable INER).

Cada método es una función NOMBRADA que compara dos `items` de un cluster y devuelve un valor
numérico, o `None` si no aplica a ese par (p.ej. compara un campo que esas dos bases no comparten).
El generador (`consolidation`) itera este registro sobre los pares cross-source de cada cluster.

Diseño abierto: `method` es el nombre de una función empaquetada que incluye todos los pasos
intermedios. Agregar un método = registrar una función nueva, sin tocar el schema del JSON.
La estructura queda lista para composites que combinan campos y comparación de otros campos
compartidos entre bases.

`cos_biencoder` — similitud coseno entre embeddings del registro completo serializado,
producidos por el Bi-Encoder fine-tuneado del componente de aprendizaje automático. Es un método OPCIONAL: se activa
vía `make_cos_biencoder_method(ruta)` (consumido por `build_consolidated_json.py --cosine`),
que carga `embeddings.parquet [record_id, embedding]` y devuelve el `Method` listo para
extender el REGISTRY. Los items llevan la llave transitoria `_record_id` (inyectada por
`consolidation._build_items` y removida antes de serializar el JSON) que este método usa
para el lookup; jw/lev la ignoran.

Los métodos NO usan el expediente como criterio de comparación. Dentro de un cluster todos los
registros ya comparten expediente por construcción (el filtrado inicial de candidatos fue por
expediente compartido), así que un `jw-exp` sería redundante. El `exp` queda como dato visible
en `linking_values`/`record`, nunca como score.

Un `item` es el dict ensamblado por `consolidation`: {item, source, linking_values, record}.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

Item = dict
MethodFn = Callable[[Item, Item], Optional[float]]


@dataclass(frozen=True)
class Method:
    """Un método de comparación nombrado, con metadatos para el catálogo del diccionario."""
    name: str
    fields: tuple[str, ...]       # campos que usa (doc + aplicabilidad)
    rango: tuple[float, float]    # rango teórico del valor
    description: str
    fn: MethodFn                  # (item_a, item_b) -> valor, o None si no aplica


def _nombre_norm(item: Item) -> str:
    return item["linking_values"]["nombre_norm"] or ""


def _jw_nombre(a: Item, b: Item) -> float:
    return JaroWinkler.normalized_similarity(_nombre_norm(a), _nombre_norm(b))


def _lev_nombre(a: Item, b: Item) -> float:
    return Levenshtein.normalized_similarity(_nombre_norm(a), _nombre_norm(b))


# Registro activo. El orden define el orden de aparición en `scores`.
REGISTRY: tuple[Method, ...] = (
    Method(
        name="jw_nombre",
        fields=("nombre_norm",),
        rango=(0.0, 1.0),
        description="Similitud Jaro-Winkler normalizada sobre el nombre normalizado (nombre_norm).",
        fn=_jw_nombre,
    ),
    Method(
        name="lev_nombre",
        fields=("nombre_norm",),
        rango=(0.0, 1.0),
        description="Similitud Levenshtein normalizada sobre el nombre normalizado (nombre_norm).",
        fn=_lev_nombre,
    ),
)


# ── cos_biencoder — método opcional (requiere embeddings externos del modelo de aprendizaje automático) ───────

# Metadata única, compartida por la factory (Method) y el catálogo (build_data_dictionary).
COS_BIENCODER_INFO = {
    "name": "cos_biencoder",
    "fields": ("record",),
    "rango": (-1.0, 1.0),
    "description": (
        "Similitud coseno entre los embeddings del registro completo serializado, "
        "producidos por el Bi-Encoder fine-tuneado (BETO + MNRL) del componente de "
        "aprendizaje automático. Método opcional: requiere embeddings.parquet exportado del checkpoint "
        "oficial (variante tok_skipnull)."
    ),
}


def make_cos_biencoder_method(embeddings_path: Union[str, Path]) -> Method:
    """Carga `embeddings.parquet [record_id, embedding]` y devuelve el Method cos_biencoder.

    Los vectores quedan capturados en el closure de la función del método. Devuelve None
    para pares donde algún record_id no tiene embedding (contrato "no aplica" del REGISTRY).
    """
    embeddings_path = Path(embeddings_path).expanduser()
    df = pd.read_parquet(embeddings_path)
    vectors = {
        int(rid): np.asarray(emb, dtype=np.float32)
        for rid, emb in zip(df["record_id"], df["embedding"])
    }

    def _cos_biencoder(a: Item, b: Item) -> Optional[float]:
        va = vectors.get(a.get("_record_id"))
        vb = vectors.get(b.get("_record_id"))
        if va is None or vb is None:
            return None
        # Los embeddings exportados ya están L2-normalizados; se renormaliza por robustez.
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

    return Method(fn=_cos_biencoder, **COS_BIENCODER_INFO)
