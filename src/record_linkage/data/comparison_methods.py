"""Protocolo extensible de métodos de comparación para el array `scores` (entregable INER).

Cada método es una función NOMBRADA que compara dos `items` de un cluster y devuelve un valor
numérico, o `None` si no aplica a ese par (p.ej. compara un campo que esas dos bases no comparten).
El generador (`consolidation`) itera este registro sobre los pares cross-source de cada cluster.

Diseño abierto, según Mireles: `method` es "el nombre de una función empaquetada que incluye todos
los pasos intermedios". Agregar un método = registrar una función nueva, sin tocar el schema del JSON.
La estructura queda lista para:
  - composites que combinan campos,
  - comparación de otros campos compartidos entre bases,
  - `cos_biencoder` — similitud coseno del Bi-Encoder sobre el `text` serializado de cada registro
    (mencionado por Mireles). Punto de extensión previsto; no cableado aquí (requiere modelo/embeddings).

Los métodos NO usan el expediente como compuerta. Dentro de un cluster todos los registros ya
comparten expediente por construcción (la generación de candidatos fue exp-gated), así que un
`jw-exp` (`return 0 si difieren exp`) sería redundante con `jw_nombre`. El `exp` queda como dato
visible en `linking_values`/`record`, nunca como filtro dentro del método.

Un `item` es el dict ensamblado por `consolidation`: {item, source, linking_values, record}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

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
