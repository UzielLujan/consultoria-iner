"""Merge decisiones manuales de un xlsx viejo a un xlsx recién regenerado.

Caso de uso: después de un refactor de `build_pairs_df()` o `classify_pairs()`
que regenera `pairs_for_review.xlsx` desde cero, este script preserva las
decisiones manuales que ya estaban capturadas en una versión previa,
identificándolas unívocamente por (record_id_a, record_id_b).

Las decisiones del xlsx nuevo se sobrescriben SOLO donde haya coincidencia
de par. Filas en el xlsx nuevo sin contraparte en el viejo (pares nuevos
introducidos por el refactor) quedan con decision vacío para revisión manual.

Uso:
    python scripts/merge_review_decisions.py \\
      --old /path/pairs_for_review_pre_refactor.xlsx \\
      --new /path/pairs_for_review.xlsx

El xlsx nuevo se reescribe in-place con styling completo
(dropdown, fills, freeze panes) usando _write_review_xlsx.
"""

import argparse
from pathlib import Path

import pandas as pd
from record_linkage.data.dataset import _write_review_xlsx


def merge_decisions(old_path: Path, new_path: Path) -> None:
    old_df = pd.read_excel(old_path, sheet_name="pairs", engine="openpyxl")
    new_df = pd.read_excel(new_path, sheet_name="pairs", engine="openpyxl")

    # Forzar object dtype para que decision acepte strings (pandas infiere float64
    # si todas las celdas están NaN, lo que rompe la asignación de strings).
    new_df["decision"] = new_df["decision"].astype(object)

    # Indexar viejo por (record_id_a, record_id_b) → decision
    old_decision = (
        old_df.set_index(["record_id_a", "record_id_b"])["decision"]
        .to_dict()
    )

    n_copied = 0
    n_unmatched_in_old = 0  # pares en old que no existen en new (no debería ocurrir)
    new_keys = set(zip(new_df["record_id_a"], new_df["record_id_b"]))
    for key in old_decision.keys():
        if key not in new_keys:
            n_unmatched_in_old += 1

    # Copiar decision donde hay match
    for idx, row in new_df.iterrows():
        key = (row["record_id_a"], row["record_id_b"])
        if key in old_decision:
            val = old_decision[key]
            if pd.notna(val) and val != "":
                new_df.at[idx, "decision"] = val
                n_copied += 1

    # Reescribir con styling
    _write_review_xlsx(new_df, new_path)

    # Estadísticas
    n_old = len(old_df)
    n_new = len(new_df)
    decision_filled_new = new_df["decision"].notna() & (new_df["decision"] != "")

    print(f"\n  Merge completado:")
    print(f"    Pares en xlsx viejo:           {n_old:,}")
    print(f"    Pares en xlsx nuevo:           {n_new:,}")
    print(f"    Pares nuevos (delta):          {n_new - n_old:+,}")
    print(f"    Decisiones copiadas:           {n_copied:,}")
    if n_unmatched_in_old:
        print(f"    ⚠ Pares en viejo no presentes en nuevo: {n_unmatched_in_old}")
    print(f"\n    Total con decision llena ahora: {int(decision_filled_new.sum()):,}")
    print(f"    Total sin decision (pendientes): {int((~decision_filled_new).sum()):,}")
    print(f"\n  ✓ Xlsx reescrito con styling: {new_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge decisiones manuales de un xlsx viejo al xlsx nuevo "
                    "por (record_id_a, record_id_b). Reescribe el xlsx nuevo in-place."
    )
    p.add_argument("--old", type=Path, required=True,
                   help="Xlsx anterior con decisiones manuales que se quieren preservar")
    p.add_argument("--new", type=Path, required=True,
                   help="Xlsx recién regenerado (decisiones se mergean in-place)")
    return p.parse_args()


def main():
    args = parse_args()
    for path in (args.old, args.new):
        if not path.exists():
            raise FileNotFoundError(f"Xlsx no encontrado: {path}")
    merge_decisions(args.old, args.new)


if __name__ == "__main__":
    main()
