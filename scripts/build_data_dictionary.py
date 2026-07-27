#!/usr/bin/env python
"""Entrypoint: construye la documentación del entregable INER (Producto 4).

A partir del JSON Schema (fuente de verdad) y del registro de métodos, genera el bundle
de documentación del entregable consolidado:

  - Diccionario_Final_INER.csv  — vista plana del schema (`campo | tipo | descripcion`),
    con rutas con puntos. Derivada 100% del schema; no duplica descripciones.
  - metodos_comparacion.json    — catálogo de métodos, desde comparison_methods.REGISTRY.
  - copia del schema            — en el bundle del entregable.

El schema (`src/record_linkage/data/consolidated_entities.schema.json`) se edita a mano;
este script solo lo proyecta. Editar una descripción ahí y re-correr basta para actualizar el CSV.

Uso:
    python scripts/build_data_dictionary.py
    python scripts/build_data_dictionary.py --perfil default
    python scripts/build_data_dictionary.py --cosine    # catálogo incluye cos_biencoder
"""
import argparse
import csv
import json
import shutil

from record_linkage.config import perfil_paths
from record_linkage.data.comparison_methods import COS_BIENCODER_INFO, REGISTRY
from record_linkage.data.consolidation import SCHEMA_PATH

_TYPE_MAP = {
    "integer": "int", "number": "float", "string": "str",
    "boolean": "bool", "object": "object", "array": "array", "null": "null",
}


def _short_type(node: dict) -> str:
    """Tipo conciso a partir del nodo de schema (maneja uniones y arreglos)."""
    t = node.get("type")
    if isinstance(t, list):
        return "|".join(_TYPE_MAP.get(x, x) for x in t)
    if t == "array":
        items = node.get("items", {})
        return "array<object>" if "$ref" in items else f"array<{_short_type(items)}>"
    return _TYPE_MAP.get(t, t or "?")


def _resolve(node: dict, defs: dict) -> dict:
    """Resuelve un $ref local (#/$defs/xxx) al nodo apuntado."""
    ref = node.get("$ref")
    if ref and ref.startswith("#/$defs/"):
        return defs[ref.split("/")[-1]]
    return node


def _flatten(node: dict, prefix: str, defs: dict, rows: list) -> None:
    """Aplana las properties de un objeto-schema a filas (campo, tipo, descripcion).

    Recurre en objetos anidados (con properties) y en arreglos de objetos, marcando
    estos últimos con `[]` en la ruta.
    """
    for name, sub in node.get("properties", {}).items():
        sub = _resolve(sub, defs)
        path = f"{prefix}{name}"
        rows.append((path, _short_type(sub), sub.get("description", "")))
        if sub.get("type") == "object" and "properties" in sub:
            _flatten(sub, path + ".", defs, rows)
        elif sub.get("type") == "array":
            items = _resolve(sub.get("items", {}), defs)
            if "properties" in items:
                _flatten(items, path + "[].", defs, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Construye la documentación del entregable INER")
    ap.add_argument("--perfil", default="default",
                    help="Perfil bajo PROCESSED_DIR/<perfil>/deliverables/ al que se escribe. "
                         "Default: 'default'.")
    ap.add_argument("--cosine", action="store_true",
                    help="Incluye cos_biencoder en metodos_comparacion.json (usar cuando el "
                         "JSON consolidado se generó con --cosine).")
    args = ap.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    defs = schema.get("$defs", {})

    rows: list = []
    _flatten(_resolve(schema.get("items", {}), defs), "", defs, rows)  # cada entidad

    out_dir = perfil_paths(args.perfil)["deliverables"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. CSV plano derivado del schema (campo | tipo | descripcion)
    csv_path = out_dir / "Diccionario_Final_INER.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["campo", "tipo", "descripcion"])
        w.writerows(rows)

    # 2. Catálogo de métodos desde REGISTRY (+ cos_biencoder si el JSON se generó con --cosine)
    metodos = [
        {"name": m.name, "fields": list(m.fields), "rango": list(m.rango),
         "description": m.description}
        for m in REGISTRY
    ]
    if args.cosine:
        metodos.append({
            "name": COS_BIENCODER_INFO["name"],
            "fields": list(COS_BIENCODER_INFO["fields"]),
            "rango": list(COS_BIENCODER_INFO["rango"]),
            "description": COS_BIENCODER_INFO["description"],
        })
    metodos_path = out_dir / "metodos_comparacion.json"
    metodos_path.write_text(json.dumps(metodos, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. Copia del schema master al bundle del entregable
    schema_copy = out_dir / SCHEMA_PATH.name
    shutil.copyfile(SCHEMA_PATH, schema_copy)

    print(f"✓ {csv_path}  ({len(rows)} filas)")
    print(f"✓ {metodos_path}  ({len(metodos)} métodos)")
    print(f"✓ {schema_copy}  (copia del schema)")


if __name__ == "__main__":
    main()
