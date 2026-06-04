"""
Script de entrada para el pipeline de limpieza de CSVs crudos del INER.

Uso:
    python scripts/run_preprocessing.py                  # Usa profile_default
    python scripts/run_preprocessing.py --perfil <name>  # Usa profile_<name>
    python scripts/run_preprocessing.py --check-paths

Para agregar un perfil nuevo basta con definir `profile_<name>(df, csv)` en
`src/record_linkage/data/preprocessing.py` y correr con `--perfil <name>`.
"""

import argparse

import pandas as pd

from record_linkage.config import RAW_FILES, PROCESSED_DIR, perfil_paths, check_paths
from record_linkage.data import preprocessing


def _resolve_profile_fn(perfil: str):
    """Busca `profile_<perfil>` en el módulo preprocessing; lanza ValueError si no existe."""
    fn_name = f"profile_{perfil}"
    fn = getattr(preprocessing, fn_name, None)
    if fn is None or not callable(fn):
        disponibles = [n.removeprefix("profile_") for n in dir(preprocessing)
                       if n.startswith("profile_") and callable(getattr(preprocessing, n))]
        raise ValueError(
            f"Perfil '{perfil}' no encontrado en preprocessing.py "
            f"(función esperada: {fn_name}). Perfiles disponibles: {disponibles}"
        )
    return fn


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de limpieza de CSVs crudos del INER"
    )
    parser.add_argument(
        '--perfil',
        default='default',
        help="Nombre del perfil a usar (default: 'default'). "
             "Busca `profile_<perfil>` en preprocessing.py."
    )
    parser.add_argument(
        '--check-paths',
        action='store_true',
        help="Verificar que existen todas las rutas necesarias y salir"
    )

    args = parser.parse_args()

    if args.check_paths:
        check_paths()
        return

    profile_fn = _resolve_profile_fn(args.perfil)

    print(f"\n{'='*60}")
    print(f"  Limpieza de CSVs INER — profile_{args.perfil}")
    print(f"{'='*60}\n")

    paths = perfil_paths(args.perfil)
    clean_dir = paths["clean"]
    clean_dir.mkdir(parents=True, exist_ok=True)

    for csv_name, raw_path in RAW_FILES.items():
        if not raw_path.exists():
            print(f"✗ {csv_name}: archivo no encontrado en {raw_path}")
            continue

        try:
            print(f"Procesando {csv_name}...", end=" ")
            df = pd.read_csv(raw_path)
            df_clean = profile_fn(df, csv_name)

            out_path = clean_dir / f"{csv_name}_clean.csv"
            df_clean.to_csv(out_path, index=False)

            print(f"✓ ({len(df_clean)} registros)")
            print(f"  → {out_path.relative_to(PROCESSED_DIR)}\n")
        except Exception as e:
            print(f"✗ Error: {e}\n")
            raise

    print(f"{'='*60}")
    print(f"  Limpieza completada — profile_{args.perfil}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
