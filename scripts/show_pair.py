"""Inspecciona un par de registros del dataset — herramienta de revisión manual.

Pensado para los pares `no_confirmado` que requieren ojo humano. Lee los CSV
`*_clean.csv` de `<perfil>/clean/`, localiza ambos registros por `record_id`
y los imprime en formato `col: val` (uno por línea), omitiendo campos nulos.

Uso (paste directo desde el xlsx — columnas record_id_a, record_id_b, source_a, source_b):

    python scripts/show_pair.py 8749 249558 Económico Comorbilidad
    python scripts/show_pair.py 1532 8901 Económico "Trabajo Social"

Acepta nombres de source en formato display ('Económico', 'Comorbilidad',
'Trabajo Social') o canónico ('econo', 'comorbilidad', 'trabajo_social').
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import perfil_paths
from record_linkage.data.serialization import _format_value


_DISPLAY_TO_CANONICAL = {
    "econo": "econo",
    "económico": "econo",
    "economico": "econo",
    "comor": "comorbilidad",
    "comorbilidad": "comorbilidad",
    "ts": "trabajo_social",
    "trabajo_social": "trabajo_social",
    "trabajo social": "trabajo_social",
}

_CSV_FILE = {
    "econo": "econo_clean.csv",
    "comorbilidad": "comorbilidad_clean.csv",
    "trabajo_social": "trabajo_social_clean.csv",
}

_DISPLAY_NAME = {
    "econo": "Económico",
    "comorbilidad": "Comorbilidad",
    "trabajo_social": "Trabajo Social",
}

# INVARIANTE: mismo orden que dataset._step_classify usa para asignar record_id global.
_CANONICAL_ORDER = ["econo", "comorbilidad", "trabajo_social"]


def normalize_source(name: str) -> str:
    key = name.strip().lower()
    if key not in _DISPLAY_TO_CANONICAL:
        accepted = ", ".join(sorted({v for v in _DISPLAY_TO_CANONICAL.values()}))
        raise ValueError(
            f"Source desconocido: '{name}'. "
            f"Aceptados (cualquier forma): Económico/Comorbilidad/Trabajo Social "
            f"o canónico ({accepted})."
        )
    return _DISPLAY_TO_CANONICAL[key]


def load_all_csvs(perfil: str) -> dict[str, pd.DataFrame]:
    base = perfil_paths(perfil)["clean"]
    dfs = {}
    for canonical in _CANONICAL_ORDER:
        path = base / _CSV_FILE[canonical]
        if not path.exists():
            raise FileNotFoundError(
                f"CSV no encontrado: {path}\n"
                f"Asegúrate de que el perfil '{perfil}' esté preprocesado."
            )
        dfs[canonical] = pd.read_csv(path)
    return dfs


def compute_offsets(dfs: dict[str, pd.DataFrame]) -> dict[str, int]:
    offsets = {}
    cur = 0
    for canonical in _CANONICAL_ORDER:
        offsets[canonical] = cur
        cur += len(dfs[canonical])
    return offsets


def get_row(dfs, offsets, record_id: int, canonical: str) -> pd.Series:
    offset = offsets[canonical]
    n_rows = len(dfs[canonical])
    local_idx = record_id - offset
    if not (0 <= local_idx < n_rows):
        valid_range = f"[{offset}, {offset + n_rows - 1}]"
        raise IndexError(
            f"record_id={record_id} no corresponde a '{_DISPLAY_NAME[canonical]}'. "
            f"Rango válido para esta fuente: {valid_range}."
        )
    return dfs[canonical].iloc[local_idx]


def format_record(row: pd.Series) -> str:
    lines = []
    for col in row.index:
        val = _format_value(row[col])
        if val:
            lines.append(f"{col}: {val}")
    return "\n".join(lines)


def show_pair(rid_a: int, src_a: str, rid_b: int, src_b: str, dfs, offsets) -> None:
    can_a = normalize_source(src_a)
    can_b = normalize_source(src_b)
    row_a = get_row(dfs, offsets, rid_a, can_a)
    row_b = get_row(dfs, offsets, rid_b, can_b)

    print(f"\n# record_a · {_DISPLAY_NAME[can_a]} · {rid_a}")
    print(format_record(row_a))
    print(f"\n# record_b · {_DISPLAY_NAME[can_b]} · {rid_b}")
    print(format_record(row_b))
    print()


def parse_args():
    p = argparse.ArgumentParser(
        description="Inspecciona un par de registros del dataset "
                    "(lectura on-demand para revisión manual de 'no_confirmado')."
    )
    p.add_argument("record_id_a", type=int, help="record_id del primer registro (col 1 del xlsx)")
    p.add_argument("record_id_b", type=int, help="record_id del segundo registro (col 2 del xlsx)")
    p.add_argument("source_a", type=str, help="source_a (Económico / Comorbilidad / Trabajo Social)")
    p.add_argument("source_b", type=str, help="source_b (Económico / Comorbilidad / Trabajo Social)")
    p.add_argument(
        "--perfil", type=str, default="default",
        help="Perfil de preprocessing (default: 'default').",
    )
    return p.parse_args()


def main():
    args = parse_args()
    dfs = load_all_csvs(args.perfil)
    offsets = compute_offsets(dfs)
    show_pair(
        args.record_id_a, args.source_a,
        args.record_id_b, args.source_b,
        dfs, offsets,
    )


if __name__ == "__main__":
    main()
