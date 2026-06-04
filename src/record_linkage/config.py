# src/record_linkage/config.py
#
# Punto central de configuración del proyecto.
# Todas las rutas del sistema se definen aquí.
#
# Como importar:
#   from record_linkage.config import DATA_ROOT, RAW_DIR, PROCESSED_DIR, perfil_paths

from pathlib import Path
from dotenv import load_dotenv
import os

# ── Raíz del repo ─────────────────────────────────────────────────────────────
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# ── Carga de variables de entorno ────────────────────────────────────────────
load_dotenv(REPO_ROOT / ".env")

# ── Raíz de datos externos (fuera del repo, nunca en git) ────────────────────
# Se lee de INER_DATA_ROOT en .env; default ~/Data/INER si no está definida.
DATA_ROOT: Path = Path(os.environ.get("INER_DATA_ROOT", Path.home() / "Data" / "INER"))

# ── Rutas internas al repo ────────────────────────────────────────────────────
DOCS_DIR:      Path = REPO_ROOT / "docs"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
SCRIPTS_DIR:   Path = REPO_ROOT / "scripts"

# ── Subdirectorios de datos ───────────────────────────────────────────────────
RAW_DIR:       Path = DATA_ROOT / "raw"
PROCESSED_DIR: Path = DATA_ROOT / "processed"
OUTPUTS_DIR:   Path = DATA_ROOT / "outputs"

# ── Archivos fuente ───────────────────────────────────────────────────────────
RAW_FILES = {
    "econo":          RAW_DIR / "INER_COVID19_CostoPacientes_Econo.csv",
    "comorbilidad":   RAW_DIR / "INER_COVID19_Pacientes_DiagnosticoComorbilidad.csv",
    "trabajo_social": RAW_DIR / "INER_COVID19_TrabajoSocial.csv",
}

# ── Resolución de subdirectorios por perfil ───────────────────────────────────
# Cada perfil tiene 4 subdirectorios canónicos bajo PROCESSED_DIR/<perfil>/:
#   clean/        — CSVs limpios (output de run_preprocessing)
#   interim/      — records_interim.parquet, pairs_classified.parquet, pairs_for_review.xlsx
#   output/       — entity_ids.parquet + <variant>/dataset.parquet
#   deliverables/ — JSON consolidado, schema, Diccionario_Final.csv, figures/, etc.
def perfil_paths(perfil: str) -> dict:
    """Devuelve los 4 subdirectorios canónicos del perfil bajo PROCESSED_DIR."""
    base = PROCESSED_DIR / perfil
    return {
        "clean":        base / "clean",
        "interim":      base / "interim",
        "output":       base / "output",
        "deliverables": base / "deliverables",
    }


# ── Validación opcional ───────────────────────────────────────────────────────
def check_paths() -> None:
    """Imprime el estado de las rutas críticas del proyecto."""
    paths = {
        "REPO_ROOT":    REPO_ROOT,
        "DATA_ROOT":    DATA_ROOT,
        "RAW_DIR":      RAW_DIR,
        "PROCESSED_DIR": PROCESSED_DIR,
        "OUTPUTS_DIR":  OUTPUTS_DIR,
    }
    print("── Rutas del proyecto ──────────────────────")
    for name, path in paths.items():
        status = "EXISTE" if path.exists() else "NO EXISTE"
        print(f"  {status}  {name}: {path}")

    print("\n── Archivos fuente ─────────────────────────")
    for key, path in RAW_FILES.items():
        status = "EXISTE" if path.exists() else "NO EXISTE"
        print(f"  {status}  {key}: {path.name}")


if __name__ == "__main__":
    check_paths()