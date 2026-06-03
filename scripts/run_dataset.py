"""CLI del pipeline de etiquetado y serialización.

Pasos:
    --step classify  → lee `<perfil>/clean/*_clean.csv`
                       → produce `<perfil>/interim/{records_interim.parquet, pairs_classified.parquet, pairs_for_review.xlsx}`
    --step finalize  → lee `<perfil>/interim/` + `pairs_for_review.xlsx` editado
                       → produce `<perfil>/output/<variant>/dataset.parquet`

Modos de serialización (4 variantes, definidas por la combinación de flags):
    tok_keepnull   — con tokens [BLK_*], conservar nulos como placeholder NULL
    tok_skipnull   — con tokens [BLK_*], omitir columnas con nulo
    notok_keepnull — sin tokens, conservar nulos
    notok_skipnull — sin tokens, omitir nulos

Por default: con tokens, conservar nulos (→ tok_keepnull).

Uso:
    python scripts/run_dataset.py --step classify --perfil default
    python scripts/run_dataset.py --step finalize --perfil default
    python scripts/run_dataset.py --step finalize --perfil default --skip-null
    python scripts/run_dataset.py --step finalize --perfil default --no-special-tokens --skip-null

Umbrales (calibrados empíricamente):
    --umbral-jw   0.88   Jaro-Winkler mínimo para metrica_clasica
    --umbral-lev  0.85   Levenshtein ratio mínimo para metrica_clasica
"""

import argparse
import sys
from pathlib import Path

# Agregar src/ al path para que funcione desde cualquier directorio
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import perfil_paths
from record_linkage.data.dataset import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de etiquetado y serialización — clasificación de pares y dataset.parquet final."
    )
    parser.add_argument(
        "--step",
        choices=["classify", "finalize"],
        required=True,
        help="'classify': genera interim/{records_interim, pairs_classified}.parquet + pairs_for_review.xlsx. "
             "'finalize': lee el xlsx editado y produce output/<variant>/dataset.parquet.",
    )
    parser.add_argument(
        "--perfil",
        type=str,
        default="default",
        help="Nombre del perfil (resuelto vía perfil_paths). Default: 'default'.",
    )
    parser.add_argument(
        "--umbral-jw",
        type=float,
        default=0.88,
        help="Umbral Jaro-Winkler para metrica_clasica (default: 0.88).",
    )
    parser.add_argument(
        "--umbral-lev",
        type=float,
        default=0.85,
        help="Umbral Levenshtein para metrica_clasica (default: 0.85).",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="Serializar sin tokens [BLK_*]. Aplica al texto generado en classify y/o finalize.",
    )
    parser.add_argument(
        "--skip-null",
        action="store_true",
        help="Omitir columnas con valor nulo en la serialización (texto más compacto).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Override del subdir bajo output/ (solo aplica a finalize). "
             "Si no se da, se deriva de la combinación de flags: <tok|notok>_<keepnull|skipnull>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Safety guard para --step classify: si ya existe un xlsx con decisiones manuales,
    # bloquea re-clasificar (sobreescribiría las decisiones del revisor).
    if args.step == "classify":
        xlsx_path = perfil_paths(args.perfil)["interim"] / "pairs_for_review.xlsx"
        if xlsx_path.exists():
            print(f"ERROR: --step classify está bloqueado: ya existe {xlsx_path}.")
            print(f"       Re-clasificar sobreescribiría las decisiones manuales del xlsx.")
            print(f"       Borra el xlsx manualmente si realmente quieres re-clasificar desde cero.")
            return 1

    build_dataset(
        perfil=args.perfil,
        step=args.step,
        use_block_tokens=not args.no_special_tokens,
        skip_null=args.skip_null,
        umbral_jw=args.umbral_jw,
        umbral_lev=args.umbral_lev,
        output_name=args.output_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
