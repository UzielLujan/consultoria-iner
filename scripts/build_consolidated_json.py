#!/usr/bin/env python
"""Entrypoint: construye consolidated_entities.json (entregable INER).

Lee los artefactos del pipeline + los CSV crudos, agrupa por entity_id, y escribe el
JSON consolidado entity-centric en `<perfil>/deliverables/`.

Insumos:
  - entity_id            ← output/entity_ids.parquet      (invariante entre variantes)
  - nombre_norm / exp_int ← interim/records_interim.parquet
  - record crudo          ← ~/Data/INER/raw/  (alineado por record_id; preprocessing no reordena filas)

Salida: <PROCESSED_DIR>/<perfil>/deliverables/consolidated_entities_<schema>.json

Uso:
    python scripts/build_consolidated_json.py
    python scripts/build_consolidated_json.py --perfil default
    python scripts/build_consolidated_json.py --schema-version v1     # schema histórico
    python scripts/build_consolidated_json.py --indent -1             # JSON compacto
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Prioriza el src/ local sobre cualquier instalación editable del paquete en el env.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import RAW_FILES, perfil_paths
from record_linkage.data.consolidation import build_entity_objects

# Orden canónico de fuentes — DEBE coincidir con la asignación de record_id en
# dataset._step_classify (econo → comor → ts).
_RAW_ORDER = ["econo", "comorbilidad", "trabajo_social"]
_UNNAMED_RE = re.compile(r"^Unnamed")


def _load_raw_by_record_id(expected_n: int) -> dict:
    """Concatena los 3 CSV crudos en orden canónico → {record_id global: fila cruda}.

    El record_id global es la posición al concatenar econo→comor→ts, idéntico al que
    asigna el pipeline: preprocessing no reordena ni elimina filas (solo dropea columnas).
    Se descartan columnas 'Unnamed:*' (ruido sin datos).
    """
    raw_by_id: dict[int, dict] = {}
    rid = 0
    for key in _RAW_ORDER:
        df = pd.read_csv(RAW_FILES[key])
        df = df[[c for c in df.columns if not _UNNAMED_RE.match(str(c))]]
        for record in df.to_dict(orient="records"):
            raw_by_id[rid] = record
            rid += 1
    if rid != expected_n:
        raise ValueError(
            f"Los CSV crudos producen {rid} filas, pero los artefactos del pipeline "
            f"tienen {expected_n}. El record_id global no estaría alineado — abortando."
        )
    return raw_by_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Construye consolidated_entities.json (INER)")
    ap.add_argument("--perfil", default="default",
                    help="Perfil bajo PROCESSED_DIR/<perfil>/ del que se leen interim/output "
                         "y al que se escriben deliverables/. Default: 'default'.")
    ap.add_argument("--schema-version", default="v2", choices=["v1", "v2"],
                    help="v2 (oficial): items anidado + scores recalculados; v1: histórico")
    ap.add_argument("--indent", type=int, default=2,
                    help="Sangría del JSON (default: 2). Usar -1 para JSON compacto.")
    ap.add_argument("--score-decimals", type=int, default=4,
                    help="Decimales de los valores en 'scores' (default: 4)")
    args = ap.parse_args()

    paths = perfil_paths(args.perfil)
    entity_ids = pd.read_parquet(paths["output"] / "entity_ids.parquet")
    records_interim = pd.read_parquet(paths["interim"] / "records_interim.parquet")

    # Validar alineación record_id/source_db entre las dos fuentes keyed por record_id.
    if not entity_ids["record_id"].equals(records_interim["record_id"]):
        raise ValueError("record_id desalineado entre entity_ids y records_interim")
    if not entity_ids["source_db"].equals(records_interim["source_db"]):
        raise ValueError("source_db desalineado entre entity_ids y records_interim")

    records_meta = entity_ids[["record_id", "source_db", "entity_id"]].merge(
        records_interim[["record_id", "nombre_norm", "exp_int"]], on="record_id"
    )

    raw_by_id = _load_raw_by_record_id(expected_n=len(records_meta))

    pairs = None
    if args.schema_version == "v1":
        pairs = pd.read_parquet(paths["interim"] / "pairs_classified.parquet")

    objects = build_entity_objects(
        records_meta, raw_by_id, pairs=pairs,
        score_decimals=args.score_decimals, schema_version=args.schema_version,
    )

    out_dir = paths["deliverables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Nombre versionado para conservar el histórico (v1) junto al oficial (v2) sin sobreescribir.
    out_path = out_dir / f"consolidated_entities_{args.schema_version}.json"
    indent = args.indent if args.indent >= 0 else None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(objects, f, ensure_ascii=False, indent=indent)

    sizes = pd.Series([o["cluster_size"] for o in objects])
    n_scores = sum(len(o["scores"]) for o in objects)
    n_empty_scores = sum(1 for o in objects if not o["scores"])
    print(f"✓ {out_path}  (schema {args.schema_version})")
    print(f"  entidades:                 {len(objects):,}")
    print(f"  registros (Σ cluster_size):{int(sizes.sum()):>9,}")
    print(f"  singletons:                {int((sizes == 1).sum()):>9,}")
    print(f"  duplas:                    {int((sizes == 2).sum()):>9,}")
    print(f"  clusters ≥3:               {int((sizes >= 3).sum()):>9,}  (máx: {int(sizes.max())})")
    print(f"  entradas en scores:        {n_scores:>9,}")
    print(f"  clusters con scores vacío: {n_empty_scores:>9,}  (sin par cross-source)")


if __name__ == "__main__":
    main()
