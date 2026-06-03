# src/record_linkage/config.py
#
# Punto central de configuración del proyecto.
# Todas las rutas del sistema se definen aquí

# Como importar:
#   from record_linkage.config import RAW_DIR, NOTEBOOKS_DIR

# Ejemplo: df = pd.read_csv(RAW_DIR / "INER_COVID19_CostoPacientes_Econo.csv")

from pathlib import Path
from dotenv import load_dotenv
import os

# ── Raíz del repo (siempre relativa a este archivo, sube 2 niveles en la jerarquía de directorios) ──────────────────────
# Funciona sin importar desde dónde se ejecute el código
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# ── Carga de variables de entorno ────────────────────────────────────────────
# Busca el .env en la raíz del repo
_ENV_PATH = REPO_ROOT / ".env"
load_dotenv(_ENV_PATH)

# ── Raíz de datos externos (fuera del repo, nunca en git) ────────────────────
# Se lee del .env para que cada máquina apunte a su propia ruta.
# Si no existe la variable, cae al default ~/Data/INER
_DATA_ROOT_DEFAULT = Path.home() / "Data" / "INER"
DATA_ROOT: Path = Path(os.environ.get("INER_DATA_ROOT", _DATA_ROOT_DEFAULT))

# ── Rutas internas al repo ────────────────────────────────────────────────────
DOCS_DIR:      Path = REPO_ROOT / "docs"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
SCRIPTS_DIR:   Path = REPO_ROOT / "scripts"

# ── Subdirectorios de datos ───────────────────────────────────────────────────
RAW_DIR:          Path = DATA_ROOT / "raw"
PROCESSED_DIR:    Path = DATA_ROOT / "processed"
GROUND_TRUTH_DIR: Path = DATA_ROOT / "ground_truth"
MODELS_DIR:       Path = DATA_ROOT / "models"
EMBEDDINGS_DIR:   Path = DATA_ROOT / "embeddings"

# ── Outputs del proyecto (figuras, métricas, evaluación) ────────────────────
OUTPUTS_DIR:      Path = DATA_ROOT / "outputs"
FIGURES_DIR:      Path = OUTPUTS_DIR / "figures"
TRAINING_DIR:     Path = OUTPUTS_DIR / "training"
EVALUATION_DIR:   Path = OUTPUTS_DIR / "evaluation"

# ── Archivos de datos fuente ─────────
RAW_FILES = {
    "econo":         RAW_DIR / "INER_COVID19_CostoPacientes_Econo.csv",
    "comorbilidad":  RAW_DIR / "INER_COVID19_Pacientes_DiagnosticoComorbilidad.csv",
    "trabajo_social": RAW_DIR / "INER_COVID19_TrabajoSocial.csv",
}
# ── Conjunto de datos etiquetado (por ahora solo existen los pares pendientes de confirmar) ─────────
GROUND_TRUTH_FILES = {
    "comorbilidad_ts":   GROUND_TRUTH_DIR / "pares_residuales_comorbilidad_trabajo_social.csv",
    "econo_comorbilidad": GROUND_TRUTH_DIR / "pares_residuales_económico_comorbilidad.csv",
    "econo_ts":          GROUND_TRUTH_DIR / "pares_residuales_económico_trabajo_social.csv",
}

# ── Resolución de subdirectorios por perfil ───────────────────────────────────
# Cada perfil tiene 4 subdirectorios canónicos:
#   clean/        — CSVs limpios (output de run_preprocessing)
#   interim/      — records_interim.parquet, pairs_classified.parquet, pairs_for_review.xlsx
#   output/       — dataset.parquet (con la columna text para BE/CE)
#   deliverables/ — consolidated_entities*.json, schema, Diccionario_Final.csv, etc.
def perfil_paths(perfil: str) -> dict:
    """Devuelve los 4 subdirectorios canónicos del perfil bajo PROCESSED_DIR."""
    base = PROCESSED_DIR / perfil
    return {
        "clean":        base / "clean",
        "interim":      base / "interim",
        "output":       base / "output",
        "deliverables": base / "deliverables",
    }


# ── Validación opcional (útil al arrancar un script o notebook) ───────────────
def check_paths() -> None:
    """Imprime el estado de todas las rutas críticas del proyecto."""
    paths = {
        "REPO_ROOT":       REPO_ROOT,
        "DOCS_DIR":        DOCS_DIR,
        "NOTEBOOKS_DIR":   NOTEBOOKS_DIR,
        "DATA_ROOT":       DATA_ROOT,
        "RAW_DIR":         RAW_DIR,
        "PROCESSED_DIR":   PROCESSED_DIR,
        "GROUND_TRUTH_DIR": GROUND_TRUTH_DIR,
        "MODELS_DIR":      MODELS_DIR,
        "EMBEDDINGS_DIR":  EMBEDDINGS_DIR,
        "OUTPUTS_DIR":    OUTPUTS_DIR,
        "FIGURES_DIR":    FIGURES_DIR,
        "TRAINING_DIR":   TRAINING_DIR,
        "EVALUATION_DIR": EVALUATION_DIR,
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