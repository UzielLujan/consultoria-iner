"""Entity type counting utilities for record linkage analysis."""

import pandas as pd


def count_entity_types(entity_series: pd.Series) -> pd.DataFrame:
    """Clasifica entidades por número de registros que las componen.

    Args:
        entity_series: columna 'entity_id' del parquet de salida del pipeline.

    Returns:
        DataFrame con columnas:
            entity_id (int), n_records (int), tipo (str)
        donde tipo ∈ {'singleton', 'dupla', 'triada', 'mayor'}

        Ordenado por entity_id ascendente.
    """
    counts = entity_series.value_counts().rename("n_records")
    result = counts.reset_index()
    result.columns = ["entity_id", "n_records"]

    def _tipo(n):
        if n == 1:
            return "singleton"
        if n == 2:
            return "dupla"
        if n == 3:
            return "triada"
        return "mayor"

    result["tipo"] = result["n_records"].apply(_tipo)
    return result.sort_values("entity_id").reset_index(drop=True)
