"""Point d'entrée de l'outil d'aide à la décision sur la tarification des
primes d'assurance automobile.

Structure :
- `utils.py` : chargement du modèle, constantes métier, prétraitement.
- `tabs/prediction_individuelle.py` : onglet de prédiction unitaire.
- `tabs/prediction_lot.py` : onglet de prédiction en lot via CSV.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from tabs import prediction_individuelle, prediction_lot  # noqa: E402
from utils import CSS_PATH, MODEL_PATH, PREPROCESSOR_PATH, load_artifacts  # noqa: E402

st.set_page_config(
    page_title="Tarification des primes auto",
    page_icon=":material/request_quote:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# NB : st.html() route un contenu composé uniquement de balises <style> vers
# un conteneur d'événement éphémère (comme st.toast) plutôt que de le monter
# durablement dans la page ; pour une feuille de style persistante, il faut
# passer par st.markdown(unsafe_allow_html=True).
st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "**:material/insights: Outil d'aide à la décision** sur la "
        "tarification des primes d'assurance automobile."
    )
    st.caption(
        "Estime la prime d'une nouvelle garantie à partir d'un modèle "
        "RandomForest entraîné sur l'historique des contrats."
    )

    st.markdown("###### :material/tune: Fonctionnalités")
    st.markdown(
        "- :material/person: **Prédiction individuelle** — une garantie à la fois\n"
        "- :material/table_rows: **Prédiction en lot** — via import CSV"
    )

    with st.expander("À propos du modèle", icon=":material/model_training:"):
        st.markdown(
            "- **Algorithme** : Random Forest (hyperparamètres optimisés)\n"
            "- **Cible** : prime nette de la garantie (FCFA)\n"
            "- **Prétraitement** : imputation KNN / mode + target encoding"
        )
        model_ok = MODEL_PATH.exists()
        prep_ok = PREPROCESSOR_PATH.exists()
        st.markdown(
            f"- :material/{'check_circle' if model_ok else 'error'}: Modèle {'chargé' if model_ok else 'introuvable'}\n"
            f"- :material/{'check_circle' if prep_ok else 'error'}: Préprocesseur {'chargé' if prep_ok else 'introuvable'}"
        )

    st.html(
        '<div class="app-footer">Projet Polytech<span class="dot">•</span>'
        "Estimation des primes auto</div>"
    )

# ---------------------------------------------------------------------------
# En-tête principal
# ---------------------------------------------------------------------------

st.html(
    """
    <div class="app-header">
        <span class="eyebrow"><span class="mi">verified</span> Outil d'aide à la décision</span>
        <h1>Tarification des primes d'assurance automobile</h1>
        <p>
            Estimez instantanément la prime d'une nouvelle garantie, unitairement
            ou en lot, à partir du modèle de tarification optimisé.
        </p>
    </div>
    """
)

# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------

try:
    model, preprocessor = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

tab_individuelle, tab_lot = st.tabs(
    [
        ":material/person: Prédiction individuelle",
        ":material/table_rows: Prédiction en lot",
    ]
)

with tab_individuelle:
    prediction_individuelle.render(model, preprocessor)

with tab_lot:
    prediction_lot.render(model, preprocessor)

st.html(
    '<div class="app-footer">Modèle Random Forest optimisé'
    '<span class="dot">•</span>Usage interne — à valider par un souscripteur</div>'
)
