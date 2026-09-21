"""Onglet — Prédiction individuelle.

Formulaire de saisie des caractéristiques d'une garantie et affichage de la
prime prédite par le modèle.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import (
    CATEGORIE_MERE_OPTIONS,
    GARANTIE_OPTIONS,
    MARQUE_OPTIONS,
    SEGMENT_OPTIONS,
    TYPEINTER_OPTIONS,
    VILLE_OPTIONS,
    format_fcfa,
    predict_premium,
)


def render(model, preprocessor) -> None:
    st.html(
        """
        <div class="section-card">
            <h4><span class="mi">directions_car</span> Caractéristiques de la garantie</h4>
            <p style="color:#64748B; margin-bottom:0;">
                Renseignez les informations du véhicule et de la garantie pour estimer la prime.
            </p>
        </div>
        """
    )

    with st.form("form_prediction_individuelle", border=True):
        st.markdown("##### :material/directions_car: Véhicule")
        col1, col2 = st.columns(2)
        with col1:
            marque = st.selectbox(
                ":material/directions_car: Marque",
                MARQUE_OPTIONS,
                index=MARQUE_OPTIONS.index("TOYOTA"),
            )
            categorie_mere = st.selectbox(
                ":material/category: Catégorie du véhicule",
                CATEGORIE_MERE_OPTIONS,
            )
            age_vehicule = st.number_input(
                "Âge du véhicule (années)",
                min_value=0.0,
                max_value=60.0,
                value=5.0,
                step=0.5,
                icon=":material/calendar_month:",
                help="Âge du véhicule à la date d'effet de la garantie.",
            )
        with col2:
            segment = st.segmented_control(
                "Segment",
                SEGMENT_OPTIONS,
                default=SEGMENT_OPTIONS[0],
            )
            places = st.number_input(
                "Nombre de places",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
                icon=":material/event_seat:",
            )
            puissance = st.number_input(
                "Puissance fiscale (CV)",
                min_value=1.0,
                max_value=60.0,
                value=9.0,
                step=1.0,
                icon=":material/bolt:",
            )

        st.markdown("##### :material/description: Garantie & souscription")
        col3, col4 = st.columns(2)
        with col3:
            garantie = st.selectbox(
                ":material/shield: Type de garantie",
                GARANTIE_OPTIONS,
            )
            duree_garantie = st.number_input(
                "Durée de la garantie (jours)",
                min_value=1,
                max_value=730,
                value=365,
                step=1,
                icon=":material/date_range:",
            )
        with col4:
            ville = st.selectbox(
                ":material/location_on: Ville",
                VILLE_OPTIONS,
            )
            typeinter = st.segmented_control(
                "Canal de souscription",
                TYPEINTER_OPTIONS,
                default=TYPEINTER_OPTIONS[0],
            )

        submitted = st.form_submit_button(
            "Estimer la prime",
            icon=":material/calculate:",
            width="stretch",
            type="primary",
        )

    if not submitted:
        return

    if not segment or not typeinter:
        st.warning(
            "Merci de sélectionner un segment et un canal de souscription.",
            icon=":material/warning:",
        )
        return

    payload = pd.DataFrame(
        [
            {
                "age_vehicule_annees": age_vehicule,
                "duree_garantie_jours": duree_garantie,
                "places": places,
                "puissance": puissance,
                "categorie_mère": categorie_mere,
                "garantie": garantie,
                "segment": segment,
                "typeinter": typeinter,
                "ville": ville,
                "marque": marque,
            }
        ]
    )

    with st.spinner("Calcul de la prime en cours…"):
        try:
            prediction = predict_premium(model, preprocessor, payload)[0]
        except Exception as exc:  # noqa: BLE001 - affichage utilisateur
            st.error(f"Erreur lors du calcul de la prime : {exc}", icon=":material/error:")
            return

    st.html(
        f"""
        <div class="result-card">
            <div class="result-label"><span class="mi">payments</span> Prime annuelle estimée</div>
            <div class="result-amount">{format_fcfa(prediction)}<span class="currency">FCFA</span></div>
            <div class="result-caption">Estimation générée par le modèle RandomForest optimisé — à valider par un souscripteur avant émission.</div>
        </div>
        """
    )

    with st.expander("Récapitulatif de la saisie", icon=":material/list_alt:"):
        st.dataframe(payload, width="stretch", hide_index=True)

    st.toast("Estimation calculée avec succès.", icon=":material/check_circle:")
