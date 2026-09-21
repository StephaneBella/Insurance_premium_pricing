"""Onglet — Prédiction en lot.

Permet d'importer un fichier CSV contenant plusieurs garanties et de générer
une prime prédite pour chacune d'entre elles, avec export des résultats.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import (
    FEATURE_COLUMNS,
    align_columns,
    build_template_csv,
    coerce_numeric_columns,
    format_fcfa,
    predict_premium,
    read_csv_flexible,
)


def render(model) -> None:
    st.html(
        """
        <div class="section-card">
            <h4><span class="mi">table_rows</span> Import d'un fichier de garanties</h4>
            <p style="color:#64748B; margin-bottom:0;">
                Le fichier CSV doit contenir une ligne par garantie et les colonnes suivantes :
                <code>age_vehicule_annees</code>, <code>duree_garantie_jours</code>,
                <code>places</code>, <code>puissance</code>, <code>valeur_neuve</code>,
                <code>valeur_venale</code>, <code>categorie_mère</code>,
                <code>garantie</code>, <code>segment</code>, <code>typeinter</code>,
                <code>ville</code>, <code>marque</code>.
            </p>
        </div>
        """
    )

    with st.container(horizontal=True, vertical_alignment="center"):
        uploaded_file = st.file_uploader(
            ":material/upload_file: Déposer un fichier CSV",
            type=["csv"],
        )
        st.download_button(
            "Télécharger un modèle",
            data=build_template_csv(),
            file_name="modele_garanties.csv",
            mime="text/csv",
            icon=":material/download:",
        )

    if uploaded_file is None:
        st.caption("Aucun fichier importé pour le moment.")
        return

    try:
        raw_df = read_csv_flexible(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - affichage utilisateur
        st.error(f"Lecture du fichier impossible : {exc}", icon=":material/error:")
        return

    if raw_df.empty:
        st.warning("Le fichier importé est vide.", icon=":material/warning:")
        return

    aligned_df, missing_cols = align_columns(raw_df)

    if missing_cols:
        st.error(
            "Colonnes manquantes dans le fichier importé : "
            + ", ".join(f"`{c}`" for c in missing_cols),
            icon=":material/error:",
        )
        with st.expander("Aperçu du fichier importé", icon=":material/visibility:"):
            st.dataframe(raw_df.head(20), width="stretch")
        return

    working_df, invalidated = coerce_numeric_columns(aligned_df)

    if invalidated:
        detail = ", ".join(f"{col} ({count})" for col, count in invalidated.items())
        st.warning(
            f"Certaines valeurs numériques n'ont pas pu être interprétées et ont été "
            f"traitées comme manquantes : {detail}.",
            icon=":material/warning:",
        )

    st.markdown("##### :material/preview: Aperçu des données importées")
    st.dataframe(working_df[FEATURE_COLUMNS].head(10), width="stretch", hide_index=True)
    st.caption(f"{len(working_df)} garantie(s) détectée(s) dans le fichier.")

    run = st.button(
        "Lancer les prédictions",
        icon=":material/rocket_launch:",
        type="primary",
        width="stretch",
    )

    if not run:
        return

    with st.spinner(f"Calcul de {len(working_df)} prime(s)…"):
        try:
            predictions = predict_premium(model, working_df)
        except Exception as exc:  # noqa: BLE001 - affichage utilisateur
            st.error(f"Erreur lors du calcul des prédictions : {exc}", icon=":material/error:")
            return

    results_df = working_df[FEATURE_COLUMNS].copy()
    results_df["prime_predite_fcfa"] = predictions.round(0)

    st.toast(f"{len(results_df)} prime(s) calculée(s).", icon=":material/check_circle:")

    st.markdown("##### :material/summarize: Synthèse")
    m1, m2, m3 = st.columns(3)
    m1.metric("Garanties traitées", f"{len(results_df):,}".replace(",", " "))
    m2.metric("Prime moyenne estimée", f"{format_fcfa(predictions.mean())} FCFA")
    m3.metric("Prime totale estimée", f"{format_fcfa(predictions.sum())} FCFA")

    st.markdown("##### :material/bar_chart: Distribution des primes estimées")
    st.bar_chart(
        results_df["prime_predite_fcfa"],
        width="stretch",
        color="#0284C7",
    )

    st.markdown("##### :material/table_view: Résultats détaillés")
    st.dataframe(
        results_df,
        width="stretch",
        hide_index=True,
        column_config={
            "prime_predite_fcfa": st.column_config.NumberColumn(
                "Prime prédite (FCFA)",
                format="%.0f",
            ),
            "age_vehicule_annees": st.column_config.NumberColumn("Âge véhicule (ans)"),
            "duree_garantie_jours": st.column_config.NumberColumn("Durée garantie (j)"),
            "places": st.column_config.NumberColumn("Places"),
            "puissance": st.column_config.NumberColumn("Puissance (CV)"),
            "valeur_neuve": st.column_config.NumberColumn("Valeur à neuf (FCFA)", format="%.0f"),
            "valeur_venale": st.column_config.NumberColumn("Valeur vénale (FCFA)", format="%.0f"),
            "categorie_mère": st.column_config.TextColumn("Catégorie"),
            "garantie": st.column_config.TextColumn("Garantie"),
            "segment": st.column_config.TextColumn("Segment"),
            "typeinter": st.column_config.TextColumn("Canal"),
            "ville": st.column_config.TextColumn("Ville"),
            "marque": st.column_config.TextColumn("Marque"),
        },
    )

    st.download_button(
        "Télécharger les résultats",
        data=results_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="predictions_primes.csv",
        mime="text/csv",
        icon=":material/download:",
        type="primary",
        width="stretch",
    )
