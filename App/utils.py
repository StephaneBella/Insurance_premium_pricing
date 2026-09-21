"""Fonctions et constantes partagées par l'ensemble de l'application.

Centralise le chargement des artefacts de modélisation (préprocesseur +
RandomForest), la définition des variables attendues par le modèle et les
utilitaires de préparation des données (individuelles ou en lot).
"""

from __future__ import annotations

import unicodedata
import warnings
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Le RandomForest a été entraîné sur la sortie du préprocesseur sous forme de
# DataFrame nommé ; en réinférence, la sortie est un ndarray positionnel
# (même ordre de colonnes), ce qui déclenche un avertissement scikit-learn
# inoffensif ("X does not have valid feature names").
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "Outputs" / "Modelisation" / "rf_optimized.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "Outputs" / "Modelisation" / "preprocessor_optimized.pkl"
CSS_PATH = APP_DIR / "style.css"

# ---------------------------------------------------------------------------
# Variables attendues par le préprocesseur / le modèle
# ---------------------------------------------------------------------------

NUM_COLS = [
    "age_vehicule_annees",
    "duree_garantie_jours",
    "places",
    "puissance",
]

CAT_COLS = [
    "categorie_mère",
    "garantie",
    "segment",
    "typeinter",
    "ville",
    "marque",
]

FEATURE_COLUMNS = NUM_COLS + CAT_COLS

# ---------------------------------------------------------------------------
# Modalités connues (issues du jeu de données d'entraînement)
# ---------------------------------------------------------------------------

CATEGORIE_MERE_OPTIONS = [
    "Tourisme (VP)",
    "Utilitaire < 3.5T",
    "Utilitaire > 3.5T",
    "Taxis",
    "Autocars / Bus",
    "Flotte pro",
    "Engins chantier",
    "Ambulances / Funéraire",
    "Auto-école",
    "2/3 Roues",
    "AUTRE",
]

GARANTIE_OPTIONS = [
    "Dommages",
    "Dommages 1er risque",
    "Tierce collision",
    "Bris de glace",
    "Bris de glace + blocs feux",
    "Incendie",
    "Recours tiers incendie",
    "Vol total",
    "Vol total + partiel",
    "Vol total + partiel + braquage",
    "Vol total + partiel + braquage (TT)",
    "Accident conducteur",
    "Transport de personnes",
    "Recours / Défense",
    "Avance sur recours",
    "Assistance routière",
    "Assistance (réparation/sinistre)",
    "Remorquage",
    "Tracking",
]

SEGMENT_OPTIONS = [
    "GENERALISTE",
    "PREMIUM",
    "SUV_4X4",
    "POIDS_LOURD",
    "DEUX_ROUES",
    "AUTRE",
]

TYPEINTER_OPTIONS = [
    "Bureau Direct",
    "Agent Général",
    "Courtier",
]

VILLE_OPTIONS = [
    "DOUALA",
    "YAOUNDE",
    "BAFOUSSAM",
    "BAMENDA",
    "GAROUA",
    "MAROUA",
    "NGAOUNDÉRÉ",
    "BERTOUA",
    "EBOLOWA",
    "KRIBI",
    "LIMBÉ",
    "BUÉA",
    "KUMBA",
    "DSCHANG",
    "FOUMBAN",
    "FOUMBOT",
    "MBOUDA",
    "BAFANG",
    "BAFUT",
    "BALENG",
    "BALI",
    "BATIBO",
    "FUNDONG",
    "KOUSSÉRI",
    "KUMBO",
    "NDJAMENA",
    "NDOP",
    "NJOMBE-PENJA",
    "NKAMBÉ",
    "NKONGSAMBA",
    "SANTA",
    "TIKO",
    "TUBAH",
    "AUTRE",
]

MARQUE_OPTIONS = [
    "TOYOTA",
    "HYUNDAI",
    "NISSAN",
    "KIA",
    "PEUGEOT",
    "MERCEDES",
    "BMW",
    "VOLKSWAGEN",
    "RENAULT",
    "MITSUBISHI",
    "FORD",
    "HONDA",
    "MAZDA",
    "OPEL",
    "SUZUKI",
    "CHEVROLET",
    "AUDI",
    "LAND ROVER",
    "RANGE ROVER",
    "JEEP",
    "LEXUS",
    "ACURA",
    "INFINITI",
    "SSANGYONG",
    "VOLVO",
    "FIAT",
    "CITROEN",
    "DAIHATSU",
    "DODGE",
    "PONTIAC",
    "OMEGA",
    "ISUZU",
    "HINO",
    "IVECO",
    "MAN",
    "RVI",
    "SHACMAN",
    "SINOTRUCK",
    "HOWO",
    "ASHOK LEYLAND",
    "DONGFENG",
    "FOTON",
    "JAC",
    "CATERPILLAR",
    "BAJAJ",
    "HAOJUE",
    "JIALING",
    "LIFAN",
    "NANFAN",
    "QLINK",
    "ROYAL",
    "SENKE",
    "TVS",
    "YAMAHA",
    "BLI",
    "AUTRE",
]

# ---------------------------------------------------------------------------
# Chargement des artefacts (mis en cache pour toute la session)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Chargement du modèle de tarification…")
def load_artifacts():
    """Charge le préprocesseur et le modèle RandomForest optimisés."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Préprocesseur introuvable : {PREPROCESSOR_PATH}"
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return model, preprocessor


# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------


def predict_premium(model, preprocessor, df: pd.DataFrame) -> np.ndarray:
    """Applique le préprocesseur puis le modèle et retourne la prime prédite.

    La cible a été entraînée sous transformation racine carrée : on
    retransforme donc les prédictions au carré pour revenir à l'échelle
    d'origine (en FCFA), en s'assurant qu'elles restent positives.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Colonnes manquantes pour la prédiction : " + ", ".join(missing)
        )

    X = preprocessor.transform(df[FEATURE_COLUMNS])
    y_pred_sqrt = np.clip(model.predict(X), 0, None)
    return np.square(y_pred_sqrt)


def format_fcfa(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


# ---------------------------------------------------------------------------
# Utilitaires pour la prédiction en lot (upload CSV)
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Normalise un nom de colonne (accents/casse/espaces) pour le matching."""
    stripped = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return stripped.strip().lower().replace(" ", "_")


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    """Lit un CSV uploadé en tentant plusieurs encodages et séparateurs
    courants (les exports Excel/CSV francophones utilisent souvent le
    point-virgule et un encodage Windows-1252)."""
    raw = uploaded_file.getvalue()

    text = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(
            "Impossible de décoder le fichier (encodages testés : UTF-8, CP1252, Latin-1)."
        )

    sep = ";" if text.count(";") > text.count(",") else ","
    return pd.read_csv(StringIO(text), sep=sep)


def align_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Renomme les colonnes du fichier importé vers les noms attendus par le
    modèle lorsque seule l'accentuation, la casse ou les espaces diffèrent,
    puis retourne le DataFrame aligné et la liste des colonnes toujours
    manquantes."""
    df = df.copy()
    normalized_required = {_normalize_name(c): c for c in FEATURE_COLUMNS}

    rename_map = {}
    for col in df.columns:
        norm = _normalize_name(col)
        target = normalized_required.get(norm)
        if target is not None and target != col:
            rename_map[col] = target

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    return df, missing


def coerce_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Force les colonnes numériques attendues en type numérique, en
    convertissant les valeurs non convertibles en NaN (imputées ensuite par
    le préprocesseur). Retourne le nombre de valeurs invalidées par colonne."""
    df = df.copy()
    invalidated: dict[str, int] = {}

    for col in NUM_COLS:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_na = df[col].isna().sum()
            if after_na > before_na:
                invalidated[col] = int(after_na - before_na)

    return df, invalidated


def build_template_csv() -> bytes:
    """Génère un exemple de fichier CSV avec les colonnes attendues."""
    example = pd.DataFrame(
        [
            {
                "age_vehicule_annees": 5,
                "duree_garantie_jours": 365,
                "places": 5,
                "puissance": 9,
                "categorie_mère": "Tourisme (VP)",
                "garantie": "Dommages",
                "segment": "GENERALISTE",
                "typeinter": "Bureau Direct",
                "ville": "DOUALA",
                "marque": "TOYOTA",
            },
            {
                "age_vehicule_annees": 12,
                "duree_garantie_jours": 180,
                "places": 4,
                "puissance": 7,
                "categorie_mère": "Taxis",
                "garantie": "Tierce collision",
                "segment": "GENERALISTE",
                "typeinter": "Courtier",
                "ville": "YAOUNDE",
                "marque": "HYUNDAI",
            },
        ]
    )
    return example.to_csv(index=False).encode("utf-8-sig")
