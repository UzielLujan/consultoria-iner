#!/usr/bin/env python
"""Entrypoint: construye consolidated_entities.json (entregable INER).

Lee los artefactos del pipeline + los CSV crudos, agrupa por entity_id, y escribe el
JSON consolidado entity-centric en `<perfil>/deliverables/consolidated_entities.json`.

Insumos:
  - entity_id             ← output/entity_ids.parquet
  - nombre_norm / exp_int ← interim/records_interim.parquet
  - record crudo          ← ~/Data/INER/raw/  (alineado por record_id; preprocessing no reordena filas)

Scores recalculados en el momento de la construcción desde comparison_methods.REGISTRY
sobre los campos de cada item — el JSON es autocontenido (no depende de pairs_classified).

Uso:
    python scripts/build_consolidated_json.py
    python scripts/build_consolidated_json.py --perfil default
    python scripts/build_consolidated_json.py --indent -1   # JSON compacto (para transferencia)

    # Con cos_biencoder (requiere embeddings/tok_skipnull/embeddings.parquet bajo INER_DATA_ROOT):
    python scripts/build_consolidated_json.py --cosine
    python scripts/build_consolidated_json.py --cosine /ruta/a/embeddings.parquet
"""
import argparse
import json
import re

import pandas as pd

from record_linkage.config import RAW_FILES, perfil_paths, EMBEDDINGS_DIR
from record_linkage.data.comparison_methods import REGISTRY, make_cos_biencoder_method
from record_linkage.data.consolidation import build_entity_objects

# Orden de fuentes — DEBE coincidir con la asignación de record_id en
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
    ap.add_argument("--indent", type=int, default=2,
                    help="Sangría del JSON (default: 2). Usar -1 para JSON compacto.")
    ap.add_argument("--score-decimals", type=int, default=4,
                    help="Decimales de los valores en 'scores' (default: 4)")
    ap.add_argument("--cosine", nargs="?", const="__default__", default=None,
                    metavar="EMBEDDINGS_PARQUET",
                    help="Incluye cos_biencoder en scores. Opcionalmente recibe la ruta al "
                         "embeddings.parquet (default: embeddings/tok_skipnull/embeddings.parquet "
                         "bajo INER_DATA_ROOT).")
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

    registry = REGISTRY
    if args.cosine is not None:
        emb_path = (EMBEDDINGS_DIR / "tok_skipnull" / "embeddings.parquet"
                    if args.cosine == "__default__" else args.cosine)
        registry = REGISTRY + (make_cos_biencoder_method(emb_path),)
        print(f"✓ cos_biencoder activado — embeddings: {emb_path}")

    objects = build_entity_objects(
        records_meta, raw_by_id,
        score_decimals=args.score_decimals,
        registry=registry,
        group_scores_by_pair=True,
    )

    out_dir = paths["deliverables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "consolidated_entities.json"
    indent = args.indent if args.indent >= 0 else None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(objects, f, ensure_ascii=False, indent=indent)

    sizes = pd.Series([o["cluster_size"] for o in objects])
    n_scores = sum(len(o["scores"]) for o in objects)
    n_empty_scores = sum(1 for o in objects if not o["scores"])
    methods_active = [m.name for m in registry]
    print(f"✓ {out_path}")
    print(f"  métodos activos:           {', '.join(methods_active)}")
    print(f"  entidades:                 {len(objects):,}")
    print(f"  registros (Σ cluster_size):{int(sizes.sum()):>9,}")
    print(f"  singletons:                {int((sizes == 1).sum()):>9,}")
    print(f"  duplas:                    {int((sizes == 2).sum()):>9,}")
    print(f"  clusters ≥3:               {int((sizes >= 3).sum()):>9,}  (máx: {int(sizes.max())})")
    print(f"  pares en scores:           {n_scores:>9,}")
    print(f"  clusters sin scores:       {n_empty_scores:>9,}  (singletons o intra-fuente)")


if __name__ == "__main__":
    main()
