"""
Pipeline de limpieza de los CSVs crudos del INER.

Define los módulos M0–M7 y un único perfil de ejecución:
  profile_default — M0(strip) → M1 → M4 (solo TS) → M5

Los módulos M2, M3, M6, M7 NO los usa profile_default pero se conservan disponibles
para componer otros perfiles si se necesitan más adelante. El JSON consolidado se
llena con los CSVs crudos directamente; preprocessing solo provee texto consistente
para que el matching de nombres y la deduplicación funcionen.

M7 (renombrado de columnas) debe correr siempre al final si se invoca: cualquier
módulo posterior asume los nombres originales del CSV crudo.
"""

import re
import unicodedata
import pandas as pd


# =============================================================================
# M0 — Normalización base de texto categórico  |  CSV: los 3  |  Opcional
# =============================================================================
#
# Aplica strip + upper a todas las columnas dtype==object de cualquier CSV.
# Colapsa variantes sintácticas que inflan artificialmente la cardinalidad de columnas categóricas.
# Corre antes que M1 y M2: redundante en campos de nombre pero no dañino,
# ya que M1 (? → Ñ) y M2 (limpieza específica) operan sobre el resultado.

def m0_normalize_text(df: pd.DataFrame, strip: bool = True, upper: bool = True) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        if strip:
            # Elimina espacios extremos y colapsa espacios internos múltiples en uno solo
            df[col] = df[col].str.strip().str.replace(r'\s+', ' ', regex=True)
        if upper:
            df[col] = df[col].str.upper()
    return df


# =============================================================================
# M1 — Corrección de encoding  |  CSV: Comorbilidad
# =============================================================================
#
# Problema: el sistema fuente de Comorbilidad codificó mal la 'Ñ' como '?'.
# Siempre: '?' → 'Ñ' (restaura el carácter original dañado por el sistema fuente).
# normalizar_nombre_v2 en utils/normalization.py maneja Ñ→N para el matching de entity_ids,
# por lo que no hay razón para degradar el carácter en el CSV de salida.

def m1_fix_encoding(df: pd.DataFrame, csv: str = 'comorbilidad') -> pd.DataFrame:
    if csv != 'comorbilidad':
        return df.copy()
    df = df.copy()
    df['nombre'] = df['nombre'].str.replace('?', 'Ñ', regex=False)
    return df


# =============================================================================
# M2 — Limpieza de caracteres en nombres  |  CSV: los 3
# =============================================================================
#
# Función de detección unificada para los 3 CSV:
#   patron = re.compile(r'[^A-ZÁÉÍÓÚÜÑ \-]', re.IGNORECASE)
#
# Espacios múltiples (los 3 CSV):
#   r'\s{2,}' → colapsar con str.replace(r'\s{2,}', ' ', regex=True)
#
# Comorbilidad — columna `nombre`:
#   - '?' resuelto por M1 primero; quedan '.', '|'
#
# Económico — columna `NOMBRE_DEL_PACIENTE`:
#   - Punto '.' (49 registros)
#   - Paréntesis con anotaciones clínicas, ej. 'CAMACHO ROJAS MARIA (ESAVI)' (28 regs.)
#     → re.sub(r'\(.*?\)', '', texto).strip()
#   - Barra '/' (1 registro)
#
# Trabajo Social — columnas `APELLIDO PATERNO`, `APELLIDO MATERNO`, `NOMBRE`:
#   - NBSP \xa0 → reemplazar primero, antes de cualquier otra limpieza
#   - Punto '.', barra '/', '|'
#
# NOTA: M2 opera sobre los campos de nombre crudos (pre-concatenación en TS).
# La concatenación de los 3 campos de TS se hace en M4, luego M6 normaliza.

def _limpiar_campo_nombre(texto: str, csv: str) -> str:
    if pd.isna(texto):
        return texto
    s = str(texto)
    if csv == 'trabajo_social':
        s = s.replace('\xa0', ' ')
    if csv == 'econo':
        s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[.\|/]', '', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def m2_clean_nombres(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    """
    csv: 'comorbilidad' | 'econo' | 'trabajo_social'
    """
    df = df.copy()
    if csv == 'comorbilidad':
        df['nombre'] = df['nombre'].apply(lambda x: _limpiar_campo_nombre(x, csv))
    elif csv == 'econo':
        df['NOMBRE_DEL_PACIENTE'] = df['NOMBRE_DEL_PACIENTE'].apply(
            lambda x: _limpiar_campo_nombre(x, csv)
        )
    elif csv == 'trabajo_social':
        for col in ['APELLIDO PATERNO', 'APELLIDO MATERNO', 'NOMBRE']:
            df[col] = df[col].apply(lambda x: _limpiar_campo_nombre(x, csv))
    return df


# =============================================================================
# M3 — Corrección de tipos de datos  |  CSV: los 3  |
# =============================================================================
#
# Mapeo de conversiones extraído de EDAs y reportes LaTeX:
#
# Comorbilidad:
#   - `fechaing`, `fechaegr`: string → datetime (rango 2020-2023)
#   - Columnas binarias (7 cols): float64 → int64
#
# Económico:
#   - `FECHA_INGRESO_INER`, `FECHA_DE_ALTA_MEJORIA`: string → datetime (rango 2020-2023)
#   - `EXP`: reemplazar 'S/E' → NaN, luego → Int64 (nullable int)
#   - `VULNERABILIDAD_SOCIOECONOMICA`: bool → int64
#
# Trabajo Social:
#   - `FECHA DE ELABORACIÓN`, `FECHA DE NACIMIENTO`: string → datetime
#     (formato mixto, dayfirst=True)
#   - `EDAD`: texto libre (ej. "69 Años", "61 Años") → extraer entero con _extraer_anios()
#   - `TOTAL DE PUNTOS`: string numéricos → int64
#
# NOTA — Fechas al exportar a CSV:
#   datetime incluye componente de hora (00:00:00) sin valor informativo.
#   Al exportar la base consolidada se podría usar .dt.strftime('%Y-%m-%d')
#   para columnas de fecha antes de to_csv().
#
# NOTA ARQUITECTÓNICA — Columnas binarias:
#   Para serialización en dataset.py, las columnas binarias tienen dos opciones:
#   - int64 (0/1) — más compacto, directo
#   - "Verdadero"/"Falso" representación semántica en español
#     para que modelos de lenguaje capturen la semántica de presencia/ausencia.
#     Implementar en serialization.py durante serialize_record(), no aquí.

def _extraer_anios(texto):
    if pd.isna(texto):
        return None
    m = re.match(r'(\d+)\s*año', str(texto), re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.match(r'^(\d+)$', str(texto).strip())
    if m2:
        return int(m2.group(1))
    return None

def m3_fix_types(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    df = df.copy()
    if csv == 'comorbilidad':
        df['fechaing'] = pd.to_datetime(df['fechaing'], errors='coerce')
        df['fechaegr'] = pd.to_datetime(df['fechaegr'], errors='coerce')
        cols_binarias = ['obesidad', 'obesidad1', 'cardiopatia', 'diabetes', 'nefropatia', 'eaperge', 'tephap']
        for col in cols_binarias:
            df[col] = (df[col]).astype('int64', errors='ignore')
    elif csv == 'econo':
        df['FECHA_INGRESO_INER'] = pd.to_datetime(df['FECHA_INGRESO_INER'], errors='coerce')
        df['FECHA_DE_ALTA_MEJORIA'] = pd.to_datetime(df['FECHA_DE_ALTA_MEJORIA'], errors='coerce')
        df['EXP'] = df['EXP'].replace('S/E', pd.NA)
        df['EXP'] = pd.to_numeric(df['EXP'], errors='coerce').astype('Int64')
        df['VULNERABILIDAD_SOCIOECONOMICA'] = df['VULNERABILIDAD_SOCIOECONOMICA'].astype('int64')
    elif csv == 'trabajo_social':
        df['FECHA DE ELABORACIÓN'] = pd.to_datetime(
            df['FECHA DE ELABORACIÓN'], format='mixed', dayfirst=True, errors='coerce'
        )
        df['FECHA DE NACIMIENTO'] = pd.to_datetime(
            df['FECHA DE NACIMIENTO'], format='mixed', dayfirst=True, errors='coerce'
        )
        df['EDAD'] = df['EDAD'].apply(_extraer_anios)
        df['TOTAL DE PUNTOS'] = pd.to_numeric(df['TOTAL DE PUNTOS'], errors='coerce').astype('Int64')
    return df


# =============================================================================
# M4 — Concatenación de nombre completo  |  CSV: Trabajo Social
# =============================================================================
#
# Trabajo Social — concatenación de los 3 campos de nombre. Opera sobre nombres originales (pre-M7).

def m4_concat_nombre_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['NOMBRE_COMPLETO'] = (
        df['APELLIDO PATERNO'].fillna('') + ' ' +
        df['APELLIDO MATERNO'].fillna('') + ' ' +
        df['NOMBRE'].fillna('')
    ).str.strip()
    return df


# =============================================================================
# M5 — Eliminación de columnas redundantes  |  CSV: Comorbilidad y Trabajo Social
# =============================================================================
#
# Comorbilidad:
#   - `dx2`, `dx3`, `dx4`: duplicados exactos (100% coincidencia registro a registro)
#     de `cie102`, `cie103`, `cie104` incluyendo mismos nulos → eliminar.
#
# Trabajo Social:
#   - `Unnamed: 19`: 98.3% nulos, artefacto del sistema
#   - `AÑO`: redundante con `FECHA DE ELABORACIÓN`, traslapes entre años consecutivos
#   - `FILA`: índice heredado de los 4 archivos anuales originales (valores 0–6107,
#     aparece 1-4 veces según cuántos archivos contenían ese renglón)

_COLS_ELIMINAR = {
    'comorbilidad':   ['dx2', 'dx3', 'dx4'],
    'trabajo_social': ['AÑO', 'FILA'],
}

def m5_drop_columns(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    df = df.copy()
    if csv == 'trabajo_social':
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    cols = _COLS_ELIMINAR.get(csv, [])
    return df.drop(columns=[c for c in cols if c in df.columns])


# =============================================================================
# M6 — Normalización de nombres  |  CSV: los 3  | Opcional, para comparación de nombres
# =============================================================================
#
# Aplicación por CSV:
#   - Comorbilidad: sobre `nombre` (ya corregido por M1 y M2)
#   - Económico:    sobre `NOMBRE_DEL_PACIENTE` (ya limpiado por M2)
#   - Trabajo Social: sobre `NOMBRE_COMPLETO` (producido por M4)
#
# _normalizar_nombre_v2 es una copia histórica local. La función canónica vive
# en utils/normalization.py y es la que usa el pipeline de etiquetado.

def _normalizar_nombre_v2(texto: str) -> str:
    if pd.isna(texto):
        return ''
    s = str(texto).upper().strip()
    s = s.replace('?', 'N')
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^A-Z ]', '', s)
    return ' '.join(sorted(s.split()))

_COL_NOMBRE = {
    'comorbilidad':   'nombre',
    'econo':          'NOMBRE_DEL_PACIENTE',
    'trabajo_social': 'NOMBRE_COMPLETO',
}

def m6_normalizar_nombres(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    df = df.copy()
    col = _COL_NOMBRE[csv]
    df[f'{col}_norm'] = df[col].apply(_normalizar_nombre_v2)
    return df


# =============================================================================
# M7 — Renombrado de columnas  |  CSV: los 3  |  Opcional, siempre al final
# =============================================================================
#
# Desambigua abreviaciones raras y elimina caracteres especiales de los nombres
# de columna (espacios, slashes, acentos) para mejorar legibilidad e interpretación
#
# IMPORTANTE: debe correr siempre al final del pipeline. Cualquier módulo anterior
# que referencie columnas por nombre asume los nombres originales del CSV crudo.
#
# Beneficios principales:
#   - Abreviaciones raras → tokens semánticos (eaperge, tephap, comorbi, etc.)
#   - Caracteres especiales en nombres de columna → snake_case limpio
#   - Mejora la calidad de los tokens generados durante serialización en dataset.py

_RENOMBRAR = {
    'comorbilidad': {
        'diagnosticoprincipal': 'diagnostico_principal',
        'diagnostico2': 'diagnostico_secundario',
        'diagnostico3': 'diagnostico_terciario',
        'diagnostico4': 'diagnostico_cuarto',
        'comorbi': 'comorbilidad_principal',
        'comorbicv': 'comorbilidad_cardiovascular',
        'eaperge': 'enfermedad_acido_peptica_reflujo',
        'tephap': 'tromboembolismo_pulmonar_hap',
    },
    'econo': {
        'EXP': 'EXPEDIENTE',
        'DERECHOHABIENTE_Y/O_BENEFICIARIO': 'DERECHOHABIENTE_O_BENEFICIARIO',
    },
    'trabajo_social': {
        'NO. HISTORIA': 'NUMERO_HISTORIA',
        'FECHA DE ELABORACIÓN': 'FECHA_ELABORACION',
        'FECHA DE NACIMIENTO': 'FECHA_NACIMIENTO',
        'APELLIDO PATERNO': 'APELLIDO_PATERNO',
        'APELLIDO MATERNO': 'APELLIDO_MATERNO',
        'DERECHOHABIENTE Y/O BENEFICIARIO': 'DERECHOHABIENTE_O_BENEFICIARIO',
        'DELEGACIÓN O MUNICIPIO PERMANENTE': 'DELEGACION_MUNICIPIO_PERMANENTE',
        'ESTADO / PAIS PERMANENTE': 'ESTADO_PAIS_PERMANENTE',
        'TOTAL DE PUNTOS': 'TOTAL_PUNTOS',
        'NIVEL SOCIOECONÓMICO': 'NIVEL_SOCIOECONOMICO',
    }
}

def m7_rename_columns(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    df = df.copy()
    renombres = _RENOMBRAR.get(csv, {})
    return df.rename(columns=renombres)


# =============================================================================
# Perfiles de ejecución — Orquestación de módulos M0–M7
# =============================================================================

def profile_default(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    """
    Perfil único de limpieza.
    Ejecución: M0(strip) → M1 → M4(si TS) → M5

    Los módulos M2, M3, M6, M7 se conservan en este archivo como bloques disponibles
    pero no se invocan por default. Si en el futuro se necesita otro perfil, se puede
    componer aquí sin tocar los módulos.
    """
    df = m0_normalize_text(df, upper=False)
    df = m1_fix_encoding(df, csv=csv)
    if csv == 'trabajo_social':
        df = m4_concat_nombre_ts(df)
    df = m5_drop_columns(df, csv)
    return df