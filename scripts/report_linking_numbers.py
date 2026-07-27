#!/usr/bin/env python
"""Cifras canónicas y figuras del entregable INER.

Recalcula desde los artefactos del pipeline las cifras de ligado de registros
y regenera las figuras (Venn + bar chart de distribución de entidades).

Secciones:
    [A] Espacio cross-CSV        — registros, pares posibles y EXP compartidos por cruce
    [B] Comparación de espacios  — espacio total vs candidatos filtrados
    [C] Clasificación por cruce  — desglose llave_exacta / metrica_clasica / revision_manual
    [D] Distribución de entidades
    [E] Síntesis pares + entidades
    [F] Revisión manual por cruce
    Figuras:
      • venn_entidades.png
      • distribucion_entidades.png

Uso:
    python scripts/report_linking_numbers.py
    python scripts/report_linking_numbers.py --no-figures   # solo cifras
    python scripts/report_linking_numbers.py --json-only    # no imprime tablas
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib_venn import venn3

from record_linkage.config import RAW_FILES, perfil_paths
from record_linkage.utils.entities import count_entity_types
from record_linkage.utils.pairs import build_pairs_df, classify_pairs


_SOURCES = ["Económico", "Comorbilidad", "Trabajo Social"]


# ─────────────────────────────────────────────────────────────────────────────
# Carga de artefactos
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts(perfil: str):
    """Carga raw CSVs, entity_ids.parquet, records_interim, xlsx editado."""
    paths = perfil_paths(perfil)

    raw_econo = pd.read_csv(RAW_FILES["econo"])
    raw_comor = pd.read_csv(RAW_FILES["comorbilidad"])
    raw_ts    = pd.read_csv(RAW_FILES["trabajo_social"])
    raw_ts    = raw_ts.loc[:, ~raw_ts.columns.str.contains("^Unnamed")]

    dataset = pd.read_parquet(paths["output"] / "entity_ids.parquet")
    records = pd.read_parquet(paths["interim"] / "records_interim.parquet")
    review  = pd.read_excel(
        paths["interim"] / "pairs_for_review.xlsx",
        sheet_name="pairs", engine="openpyxl",
    )

    return {
        "raw":     {"econo": raw_econo, "comorbilidad": raw_comor, "trabajo_social": raw_ts},
        "dataset": dataset,
        "records": records,
        "review":  review,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _exp_sets_by_source(records: pd.DataFrame) -> dict[str, set]:
    """Conjunto de EXP enteros (≠ NaN) por source_db."""
    valid = records[records["exp_int"].notna()]
    return {
        src: set(valid.loc[valid["source_db"] == src, "exp_int"].astype(int))
        for src in _SOURCES
    }


def _entities_by_source(dataset: pd.DataFrame) -> dict[str, set]:
    """Conjunto de entity_id presentes en cada source_db."""
    return {
        src: set(dataset.loc[dataset["source_db"] == src, "entity_id"])
        for src in _SOURCES
    }


# ─────────────────────────────────────────────────────────────────────────────
# Secciones
# ─────────────────────────────────────────────────────────────────────────────

def compute_section_A(raw: dict, exp_sets: dict[str, set]) -> dict:
    """[A] tab:espacio_crosscsv — registros, pares posibles y EXP compartidos por cruce."""
    n = {
        "Económico":      len(raw["econo"]),
        "Comorbilidad":   len(raw["comorbilidad"]),
        "Trabajo Social": len(raw["trabajo_social"]),
    }
    cruces = [
        ("Económico", "Comorbilidad"),
        ("Económico", "Trabajo Social"),
        ("Comorbilidad", "Trabajo Social"),
    ]
    rows = []
    total_pares = 0
    total_exp = 0
    for a, b in cruces:
        pares_pos = n[a] * n[b]
        exp_comp  = len(exp_sets[a] & exp_sets[b])
        rows.append({
            "cruce": f"{a} ↔ {b}",
            "registros_a": n[a], "registros_b": n[b],
            "pares_posibles": pares_pos,
            "exp_compartidos": exp_comp,
        })
        total_pares += pares_pos
        total_exp += exp_comp
    return {
        "rows": rows,
        "total_pares_posibles": total_pares,
        "total_exp_compartidos": total_exp,
        "n_registros": n,
    }


def compute_section_B(raw: dict, section_a: dict, n_pares_candidatos_v2: int) -> dict:
    """[B] tab:comparacion_espacios — total intra+cross vs cross vs candidatos."""
    n_total = sum(section_a["n_registros"].values())
    return {
        "n_registros_total":         n_total,
        "pares_total_incluye_intra": comb(n_total, 2),
        "pares_total_cross_csv":     section_a["total_pares_posibles"],
        "candidatos_exp_compartido": section_a["total_exp_compartidos"],
        "candidatos_v2_incluye_nan_econo": n_pares_candidatos_v2,
    }


def compute_section_C(pairs_clf: pd.DataFrame, review: pd.DataFrame) -> dict:
    """[C] Clasificación por cruce — desglose de criterios por par de bases.

    Los pares NaN-Económico se fusionan al cruce correspondiente (Econo↔Comor o
    Econo↔TS) según el target. Distingue cuatro buckets: llave_exacta /
    metrica_clasica / rev_manual_match / rev_manual_no_match.

    Supuesto: la revisión manual está completa — el xlsx ya pasó por el flujo
    completo del pipeline (--step finalize sin pares no_confirmado pendientes).
    Un par no_confirmado SIN decisión caería al bucket rev_manual_match.
    """
    # Merge decisiones manuales: clasificación efectiva = decision si está llena, sino criterio.
    decisiones = review.set_index(["record_id_a", "record_id_b"])["decision"].to_dict()

    def _cruce_key(sa, sb):
        order = {"Económico": 0, "Comorbilidad": 1, "Trabajo Social": 2}
        a, b = (sa, sb) if order[sa] < order[sb] else (sb, sa)
        return f"{a} ↔ {b}"

    buckets = ["llave_exacta", "metrica_clasica", "rev_manual_match", "rev_manual_no_match"]
    counts: dict[str, dict[str, int]] = {}

    for row in pairs_clf.itertuples(index=False):
        key = _cruce_key(row.source_a, row.source_b)
        counts.setdefault(key, dict.fromkeys(buckets, 0))

        criterio = row.criterio
        if criterio == "llave_exacta":
            bucket = "llave_exacta"
        elif criterio == "metrica_clasica":
            bucket = "metrica_clasica"
        else:  # no_confirmado → mira decisión manual
            decision = decisiones.get((row.record_id_a, row.record_id_b))
            if decision == "match":
                bucket = "rev_manual_match"
            elif decision == "no_match":
                bucket = "rev_manual_no_match"
            else:
                bucket = "rev_manual_match"  # asume revisión completa (ver docstring) — verificar --step finalize sin pendientes
        counts[key][bucket] += 1

    # Filas ordenadas + total
    cruces_ordenados = [
        "Económico ↔ Comorbilidad",
        "Económico ↔ Trabajo Social",
        "Comorbilidad ↔ Trabajo Social",
    ]
    rows = []
    totales = dict.fromkeys(buckets, 0)
    for cruce in cruces_ordenados:
        c = counts.get(cruce, dict.fromkeys(buckets, 0))
        row = {"cruce": cruce, **c}
        row["positivos_v2"] = c["llave_exacta"] + c["metrica_clasica"] + c["rev_manual_match"]
        row["candidatos_v2"] = sum(c.values())
        rows.append(row)
        for k in buckets:
            totales[k] += c[k]

    total_positivos = totales["llave_exacta"] + totales["metrica_clasica"] + totales["rev_manual_match"]
    total_candidatos = sum(totales.values())

    return {
        "rows": rows,
        "totales": {
            **totales,
            "positivos_v2":  total_positivos,
            "candidatos_v2": total_candidatos,
        },
    }


def compute_section_D(dataset: pd.DataFrame) -> dict:
    """[D] tab:distribucion_entidades — 7 regiones del Venn por entity_id final."""
    ent = _entities_by_source(dataset)
    e, c, t = ent["Económico"], ent["Comorbilidad"], ent["Trabajo Social"]

    regiones = {
        "Solo Económico":                            len(e - c - t),
        "Solo Comorbilidad":                         len(c - e - t),
        "Solo Trabajo Social":                       len(t - e - c),
        "Económico ∩ Comorbilidad (sin TS)":         len((e & c) - t),
        "Económico ∩ Trabajo Social (sin Comor)":    len((e & t) - c),
        "Comorbilidad ∩ Trabajo Social (sin Eco)":   len((c & t) - e),
        "En las 3 bases":                            len(e & c & t),
    }
    total = len(e | c | t)
    return {"regiones": regiones, "total_entidades": total}


def compute_section_E(
    section_a: dict, section_b: dict, section_c: dict, section_d: dict, dataset: pd.DataFrame
) -> dict:
    """[E] tab:sintesis_espacios — agregado nivel pares + nivel entidades."""
    n_total = section_b["n_registros_total"]
    total_pares_cross = section_a["total_pares_posibles"]
    total_positivos = section_c["totales"]["positivos_v2"]

    # Entidades vinculables: presentes en ≥2 fuentes
    tipos = count_entity_types(dataset["entity_id"])
    sizes = dataset.groupby("entity_id")["source_db"].nunique()
    en_2 = int((sizes == 2).sum())
    en_3 = int((sizes == 3).sum())

    n_singleton = int((tipos["tipo"] == "singleton").sum())
    n_dupla     = int((tipos["tipo"] == "dupla").sum())
    n_triada    = int((tipos["tipo"] == "triada").sum())
    n_mayor     = int((tipos["tipo"] == "mayor").sum())

    return {
        # Pares
        "espacio_total_cross_csv": total_pares_cross,
        "candidatos_v2":           section_c["totales"]["candidatos_v2"],
        "positivos_v2":            total_positivos,
        "desbalance_neg_pos":      (total_pares_cross - total_positivos) // max(total_positivos, 1),
        # Entidades
        "n_registros_total":   n_total,
        "n_entidades":         section_d["total_entidades"],
        "duplicados_entre_csv": n_total - section_d["total_entidades"],
        "vinculables_2_csv":   en_2,
        "vinculables_3_csv":   en_3,
        "vinculables_total":   en_2 + en_3,
        # Distribución por tamaño
        "singletons": n_singleton,
        "duplas":     n_dupla,
        "triadas":    n_triada,
        "mayores":    n_mayor,
    }


def compute_section_F(pairs_clf: pd.DataFrame, review: pd.DataFrame) -> dict:
    """[F] Revisión manual por cruce — pares no_confirmado y sus decisiones."""
    decisiones = review.set_index(["record_id_a", "record_id_b"])["decision"].to_dict()

    rows = []
    cruces_ordenados = [
        ("Económico", "Comorbilidad"),
        ("Económico", "Trabajo Social"),
        ("Comorbilidad", "Trabajo Social"),
    ]
    for a, b in cruces_ordenados:
        order = {"Económico": 0, "Comorbilidad": 1, "Trabajo Social": 2}
        sub = pairs_clf[
            (pairs_clf["criterio"] == "no_confirmado") &
            (((pairs_clf["source_a"] == a) & (pairs_clf["source_b"] == b)) |
             ((pairs_clf["source_a"] == b) & (pairs_clf["source_b"] == a)))
        ]
        n_total = len(sub)
        n_match = sum(decisiones.get((r.record_id_a, r.record_id_b)) == "match" for r in sub.itertuples(index=False))
        n_no    = sum(decisiones.get((r.record_id_a, r.record_id_b)) == "no_match" for r in sub.itertuples(index=False))
        rows.append({
            "cruce":    f"{a} ↔ {b}",
            "revisados": n_total,
            "match":     n_match,
            "no_match":  n_no,
        })
    return {
        "rows": rows,
        "total_revisados": sum(r["revisados"] for r in rows),
        "total_match":     sum(r["match"] for r in rows),
        "total_no_match":  sum(r["no_match"] for r in rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────────────────────────────────────

def render_venn(section_d: dict, out_path: Path) -> None:
    r = section_d["regiones"]
    fig, ax = plt.subplots(figsize=(8, 6))
    venn3(
        subsets=(
            r["Solo Económico"],
            r["Solo Comorbilidad"],
            r["Económico ∩ Comorbilidad (sin TS)"],
            r["Solo Trabajo Social"],
            r["Económico ∩ Trabajo Social (sin Comor)"],
            r["Comorbilidad ∩ Trabajo Social (sin Eco)"],
            r["En las 3 bases"],
        ),
        set_labels=("Económico", "Comorbilidad", "Trabajo Social"),
        set_colors=("#3498db", "#e74c3c", "#2ecc71"),
        alpha=0.6, ax=ax,
    )
    ax.set_title(
        f'{section_d["total_entidades"]:,} entidades únicas entre las 3 bases',
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def render_distribucion(section_d: dict, out_path: Path) -> None:
    r = section_d["regiones"]
    labels = [
        "Solo\nEconómico", "Solo\nComorbilidad", "Solo\nTrabajo Social",
        "Eco ∩ Comor\n(sin TS)", "Eco ∩ TS\n(sin Comor)",
        "Comor ∩ TS\n(sin Eco)", "En las\n3 bases",
    ]
    values = [
        r["Solo Económico"], r["Solo Comorbilidad"], r["Solo Trabajo Social"],
        r["Económico ∩ Comorbilidad (sin TS)"],
        r["Económico ∩ Trabajo Social (sin Comor)"],
        r["Comorbilidad ∩ Trabajo Social (sin Eco)"],
        r["En las 3 bases"],
    ]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Entidades")
    ax.set_title(
        f'Distribución de {section_d["total_entidades"]:,} entidades entre las 3 bases',
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Render legible
# ─────────────────────────────────────────────────────────────────────────────

def print_report(A, B, C, D, E, F) -> None:
    fmt = lambda x: f"{x:>10,}" if isinstance(x, int) else f"{x:>10}"

    print("\n" + "═" * 72)
    print(" [A] Espacio cross-CSV  (tab:espacio_crosscsv)")
    print("═" * 72)
    print(f"  {'Cruce':<38} {'Reg A':>8} {'Reg B':>8} {'A×B':>16} {'EXP comp.':>10}")
    for r in A["rows"]:
        print(f"  {r['cruce']:<38} {r['registros_a']:>8,} {r['registros_b']:>8,} "
              f"{r['pares_posibles']:>16,} {r['exp_compartidos']:>10,}")
    print(f"  {'TOTAL':<38} {'':>8} {'':>8} {A['total_pares_posibles']:>16,} "
          f"{A['total_exp_compartidos']:>10,}")

    print("\n" + "═" * 72)
    print(" [B] Comparación de espacios  (tab:comparacion_espacios)")
    print("═" * 72)
    print(f"  Total incluyendo intra-CSV  (N choose 2):  {B['pares_total_incluye_intra']:>16,}")
    print(f"  Total cross-CSV  (Σ A×B):                  {B['pares_total_cross_csv']:>16,}")
    print(f"  Candidatos (filtrado por EXP compartido):   {B['candidatos_exp_compartido']:>16,}")
    print(f"  Candidatos total (incluye Eco con EXP nulo):{B['candidatos_v2_incluye_nan_econo']:>15,}")

    print("\n" + "═" * 72)
    print(" [C] Clasificación por cruce")
    print("═" * 72)
    print(f"  {'Cruce':<38} {'l_ex':>6} {'m_cl':>6} {'rm_M':>6} {'rm_N':>6} {'POS':>7} {'CAND':>7}")
    for r in C["rows"]:
        print(f"  {r['cruce']:<38} {r['llave_exacta']:>6,} {r['metrica_clasica']:>6,} "
              f"{r['rev_manual_match']:>6,} {r['rev_manual_no_match']:>6,} "
              f"{r['positivos_v2']:>7,} {r['candidatos_v2']:>7,}")
    t = C["totales"]
    print(f"  {'TOTAL':<38} {t['llave_exacta']:>6,} {t['metrica_clasica']:>6,} "
          f"{t['rev_manual_match']:>6,} {t['rev_manual_no_match']:>6,} "
          f"{t['positivos_v2']:>7,} {t['candidatos_v2']:>7,}")
    print("  Leyenda: l_ex=llave_exacta, m_cl=metrica_clasica, rm_M=rev.manual match, rm_N=rev.manual no_match")

    print("\n" + "═" * 72)
    print(" [D] Distribución de entidades (tab:distribucion_entidades)")
    print("═" * 72)
    for k, v in D["regiones"].items():
        print(f"  {k:<45} {v:>8,}")
    print(f"  {'TOTAL':<45} {D['total_entidades']:>8,}")

    print("\n" + "═" * 72)
    print(" [E] Síntesis  (tab:sintesis_espacios)")
    print("═" * 72)
    print("  -- Pares --")
    print(f"  Espacio total cross-CSV          {E['espacio_total_cross_csv']:>14,}")
    print(f"  Candidatos v2                    {E['candidatos_v2']:>14,}")
    print(f"  Positivos confirmados v2         {E['positivos_v2']:>14,}")
    print(f"  Desbalance neg:pos                {E['desbalance_neg_pos']:>13,}:1")
    print("  -- Entidades --")
    print(f"  Registros totales (3 CSV)        {E['n_registros_total']:>14,}")
    print(f"  Entidades únicas                 {E['n_entidades']:>14,}")
    print(f"  Duplicados entre CSV             {E['duplicados_entre_csv']:>14,}")
    print(f"  Vinculables (≥2 CSV)             {E['vinculables_total']:>14,}")
    print(f"    • en exactamente 2 CSV         {E['vinculables_2_csv']:>14,}")
    print(f"    • en las 3 bases               {E['vinculables_3_csv']:>14,}")
    print("  -- Distribución por tamaño --")
    print(f"  Singletons (1 registro)          {E['singletons']:>14,}")
    print(f"  Duplas     (2 registros)         {E['duplas']:>14,}")
    print(f"  Tríadas    (3 registros)         {E['triadas']:>14,}")
    print(f"  Mayores    (≥4 registros)        {E['mayores']:>14,}")

    print("\n" + "═" * 72)
    print(" [F] Revisión manual por cruce")
    print("═" * 72)
    print(f"  {'Cruce':<38} {'Revisados':>10} {'match':>7} {'no_match':>9}")
    for r in F["rows"]:
        print(f"  {r['cruce']:<38} {r['revisados']:>10,} {r['match']:>7,} {r['no_match']:>9,}")
    print(f"  {'TOTAL':<38} {F['total_revisados']:>10,} {F['total_match']:>7,} {F['total_no_match']:>9,}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Cifras canónicas y figuras del Reporte_INER")
    ap.add_argument("--perfil", default="default",
                    help="Perfil bajo PROCESSED_DIR/<perfil>/ del que se leen artefactos y al "
                         "que se escribe deliverables/report_numbers.json. Default: 'default'.")
    ap.add_argument("--umbral-jw", type=float, default=0.88)
    ap.add_argument("--umbral-lev", type=float, default=0.85)
    ap.add_argument("--no-figures", action="store_true",
                    help="No regenerar venn_entidades.png ni distribucion_entidades.png")
    ap.add_argument("--json-only", action="store_true",
                    help="No imprimir tablas; sólo escribir JSON")
    args = ap.parse_args()

    art = load_artifacts(args.perfil)

    # Reconstruir pairs clasificados desde clean CSVs (cifra independiente de cualquier parquet cacheado)
    paths = perfil_paths(args.perfil)
    df_econo = pd.read_csv(paths["clean"] / "econo_clean.csv")
    df_comor = pd.read_csv(paths["clean"] / "comorbilidad_clean.csv")
    df_ts    = pd.read_csv(paths["clean"] / "trabajo_social_clean.csv")
    pairs_df = build_pairs_df(df_econo, df_comor, df_ts)
    pairs_clf = classify_pairs(pairs_df, umbral_jw=args.umbral_jw, umbral_lev=args.umbral_lev)

    exp_sets = _exp_sets_by_source(art["records"])

    A = compute_section_A(art["raw"], exp_sets)
    B = compute_section_B(art["raw"], A, n_pares_candidatos_v2=len(pairs_clf))
    C = compute_section_C(pairs_clf, art["review"])
    D = compute_section_D(art["dataset"])
    E = compute_section_E(A, B, C, D, art["dataset"])
    F = compute_section_F(pairs_clf, art["review"])

    if not args.json_only:
        print_report(A, B, C, D, E, F)

    # Dump JSON
    out_dir = paths["deliverables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report_numbers.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON: {json_path}")

    # Figuras
    if not args.no_figures:
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        venn_path = fig_dir / "venn_entidades.png"
        dist_path = fig_dir / "distribucion_entidades.png"
        render_venn(D, venn_path)
        render_distribucion(D, dist_path)
        print(f"✓ Figura: {venn_path}")
        print(f"✓ Figura: {dist_path}")


if __name__ == "__main__":
    main()
