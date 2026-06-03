"""Name normalization utilities for entity matching."""

import re
import unicodedata

import pandas as pd


def normalizar_nombre_v2(texto) -> str:
    """Normaliza un nombre de paciente para comparación entre CSVs.

    Transformaciones aplicadas (en orden):
    1. Convierte a mayúsculas y elimina espacios extremos
    2. Reemplaza '?' por 'N' (artefacto de encoding frecuente en los CSVs del INER)
    3. Desacentuación NFD (á→a, é→e, ñ→n, etc.)
    4. Elimina todo carácter no alfabético ni espacio
    5. Ordena tokens alfabéticamente — hace la llave invariante al orden de apellidos/nombre

    Args:
        texto: nombre crudo del paciente (str, float NaN, o None)

    Returns:
        Nombre normalizado como string. NaN/None → "".
    """
    if pd.isna(texto):
        return ""
    s = str(texto).upper().strip()
    s = s.replace("?", "N")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^A-Z ]", "", s)
    return " ".join(sorted(s.split()))
