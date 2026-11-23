"""
Agent 2 - Cybersécurité & Anti-Spoofing
======================================

Entraîne un classifieur Random Forest pour détecter les positions GPS falsifiées.

Entrée :
    data/processed/fleet_telemetry_processed.csv

Sortie :
    models/agent2_spoofing.pkl
    results/agent2_spoofing_metrics.json

Lancement depuis la racine du projet :
    python scripts/04_agent2_cybersecurity.py

Dépendances :
    pip install pandas numpy scikit-learn joblib
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore", category=UserWarning)



# 1. CONFIGURATION DES CHEMINS ET DU MODÈLE


# Le projet est le dossier parent de "scripts".
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fleet_telemetry_processed.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = MODELS_DIR / "agent2_spoofing.pkl"
METRICS_PATH = RESULTS_DIR / "agent2_spoofing_metrics.json"

# Pour reproduire les résultats, on fixe la graine aléatoire.
RANDOM_STATE = 42

# Proportion des drones réservée au test.
TEST_SIZE = 0.20

# Colonnes utilisées par le modèle.
# IMPORTANT : on utilise les positions MESURÉES, pas les positions réelles tx/ty/tz.
FEATURE_COLUMNS = [
    "step_dist_measured",
    "velocity_residual",
    "v_norm",
    "step_dist",
]



# 2. CHARGEMENT ET VALIDATION DU DATASET


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV et vérifie la présence des colonnes nécessaires."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {csv_path}\n"
            "Vérifie que le prétraitement a bien créé le CSV."
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "drone_id",
        "timestamp",
        "tx_measured",
        "ty_measured",
        "tz_measured",
        "vx",
        "vy",
        "vz",
        "v_norm",
        "step_dist",
        "is_spoofed",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le dataset : "
            + ", ".join(missing_columns)
        )

    # La cible doit contenir uniquement 0 (sain) et 1 (spoofing).
    labels = set(df["is_spoofed"].dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"La colonne is_spoofed doit contenir 0 et 1. Valeurs trouvées : {labels}"
        )

    # Tri indispensable : les différences temporelles doivent être calculées
    # dans l'ordre chronologique et séparément pour chaque drone.
    df = df.sort_values(
        ["drone_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)

    print(f"Dataset chargé : {len(df):,} lignes")
    print(f"Nombre de drones : {df['drone_id'].nunique()}")
    print("Répartition des labels :")
    print(df["is_spoofed"].value_counts().sort_index().to_string())

    return df



# 3. EXTRACTION DES CARACTÉRISTIQUES ANTI-SPOOFING


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les indicateurs demandés pour chaque drone.

    1) step_dist_measured :
       || P_meas(t) - P_meas(t-1) ||

    2) velocity_residual :
       || [P_meas(t) - P_meas(t-1)] - V(t-1) * dt ||

    3) v_norm :
       norme de la vitesse fournie dans le dataset.

    Les décalages temporels sont toujours calculés par drone pour ne jamais
    mélanger les trajectoires de deux drones différents.
    """

    data = df.copy()
    grouped = data.groupby("drone_id", sort=False)

    # Position GPS mesurée à l'instant précédent.
    previous_x = grouped["tx_measured"].shift(1)
    previous_y = grouped["ty_measured"].shift(1)
    previous_z = grouped["tz_measured"].shift(1)

    # Déplacement GPS mesuré entre t-1 et t.
    delta_x = data["tx_measured"] - previous_x
    delta_y = data["ty_measured"] - previous_y
    delta_z = data["tz_measured"] - previous_z

    # Première observation de chaque drone : aucun déplacement précédent connu.
    delta_x = delta_x.fillna(0.0)
    delta_y = delta_y.fillna(0.0)
    delta_z = delta_z.fillna(0.0)

    # Caractéristique 1 : saut / distance GPS mesurée.
    data["step_dist_measured"] = np.sqrt(
        delta_x**2 + delta_y**2 + delta_z**2
    )

    # Vitesse à t-1, séparément pour chaque drone.
    previous_vx = grouped["vx"].shift(1).fillna(0.0)
    previous_vy = grouped["vy"].shift(1).fillna(0.0)
    previous_vz = grouped["vz"].shift(1).fillna(0.0)

    # Temps écoulé entre deux observations.
    # Le CSV fourni ne contient pas la colonne dt : le prétraitement la calcule,
    # mais ne l'ajoute pas à columns_to_export. On la reconstruit donc ici
    # à partir de timestamp, séparément pour chaque drone.
    if "dt" in data.columns:
        dt = pd.to_numeric(data["dt"], errors="coerce")
    else:
        dt = data.groupby("drone_id", sort=False)["timestamp"].diff()

    dt = dt.where(np.isfinite(dt) & (dt > 0), np.nan)

    # La première observation de chaque drone n'a pas de différence temporelle.
    # On remplace les valeurs manquantes par la médiane des intervalles valides.
    median_dt = dt.median()
    if pd.isna(median_dt) or median_dt <= 0:
        raise ValueError(
            "Impossible de déterminer un dt valide à partir de timestamp."
        )
    dt = dt.fillna(median_dt)

    # Déplacement attendu d'après la vitesse précédente.
    expected_dx = previous_vx * dt
    expected_dy = previous_vy * dt
    expected_dz = previous_vz * dt

    # Caractéristique 2 : résidu entre déplacement mesuré et déplacement attendu.
    data["velocity_residual"] = np.sqrt(
        (delta_x - expected_dx) ** 2
        + (delta_y - expected_dy) ** 2
        + (delta_z - expected_dz) ** 2
    )

    # Caractéristique 3 : norme de vitesse déjà calculée par le prétraitement.
    data["v_norm"] = pd.to_numeric(data["v_norm"], errors="coerce")

    # Caractéristique 4 : step_dist fournie par le prétraitement.
    data["step_dist"] = pd.to_numeric(data["step_dist"], errors="coerce")

    # Nettoyage des valeurs infinies / manquantes dans les caractéristiques.
    data[FEATURE_COLUMNS] = (
        data[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
    )

    # On retire uniquement les lignes dont les caractéristiques sont invalides.
    # Les labels sont conservés tels quels pour l'apprentissage supervisé.
    valid_mask = data[FEATURE_COLUMNS].notna().all(axis=1)
    removed_rows = int((~valid_mask).sum())
    data = data.loc[valid_mask].copy().reset_index(drop=True)

    print("\nCaractéristiques calculées :")
    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")

    if removed_rows:
        print(f"Lignes retirées pour valeurs invalides : {removed_rows}")

    return data



# 4. SÉPARATION ENTRAÎNEMENT / TEST PAR DRONE


def split_by_drone(data: pd.DataFrame):
    """
    Sépare les données par drone et non par ligne.

    Ainsi, les observations d'un même drone ne se retrouvent pas à la fois
    dans l'entraînement et dans le test. Cela limite une évaluation trop optimiste
    due à des frames temporellement voisines.
    """

    X = data[FEATURE_COLUMNS]
    y = data["is_spoofed"].astype(int)
    groups = data["drone_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # Random Forest ne peut pas apprendre si une seule classe est présente.
    if y_train.nunique() < 2:
        raise ValueError(
            "Le jeu d'entraînement ne contient qu'une seule classe. "
            "Il faut revoir la répartition des drones ou le dataset."
        )

    if y_test.nunique() < 2:
        raise ValueError(
            "Le jeu de test ne contient qu'une seule classe. "
            "Relance avec une autre graine ou ajuste TEST_SIZE."
        )

    print("\nSéparation des données :")
    print(f"  Entraînement : {len(X_train):,} lignes")
    print(f"  Test         : {len(X_test):,} lignes")
    print(f"  Drones train : {groups.iloc[train_indices].nunique()}")
    print(f"  Drones test  : {groups.iloc[test_indices].nunique()}")

    return X_train, X_test, y_train, y_test



# 5. ENTRAÎNEMENT DU MODÈLE RANDOM FOREST


def train_model(X_train, y_train) -> RandomForestClassifier:
    """Entraîne le classifieur Random Forest."""

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nEntraînement du modèle Random Forest...")
    model.fit(X_train, y_train)
    print("Entraînement terminé.")

    return model



# 6. ÉVALUATION DU MODÈLE


def evaluate_model(model, X_test, y_test) -> dict:
    """Calcule les métriques de classification sur le jeu de test."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics = {
        "model_name": "Agent2_RandomForest_AntiSpoofing",
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_spoofed": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_spoofed": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_spoofed": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "features": FEATURE_COLUMNS,
    }

    print("\n" + "=" * 58)
    print("ÉVALUATION DE L'AGENT 2 - ANTI-SPOOFING")
    print("=" * 58)
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Sain (0)", "Spoofing (1)"],
        zero_division=0,
    ))

    print("Matrice de confusion [ [TN, FP], [FN, TP] ] :")
    print(matrix)

    print("\nMétriques principales :")
    for name in [
        "accuracy",
        "precision_spoofed",
        "recall_spoofed",
        "f1_spoofed",
        "roc_auc",
        "average_precision",
    ]:
        print(f"  {name:<22}: {metrics[name]:.4f}")

    return metrics



# 7. SAUVEGARDE DU MODÈLE ET DES MÉTRIQUES


def save_artifacts(model, metrics: dict) -> None:
    """Sauvegarde le modèle entraîné et les résultats d'évaluation."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # On sauvegarde aussi les noms des caractéristiques et quelques métadonnées.
    # Cela permet de reconstruire exactement l'ordre des entrées à l'inférence.
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "is_spoofed",
        "classes": {"0": "Sain", "1": "Attaque GPS Spoofing"},
        "model_name": "RandomForestClassifier",
        "random_state": RANDOM_STATE,
    }

    joblib.dump(artifact, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    print("\nFichiers sauvegardés :")
    print(f"  Modèle   : {MODEL_PATH}")
    print(f"  Métriques: {METRICS_PATH}")



# 8. TEST DU CHARGEMENT ET DE LA PRÉDICTION


def test_saved_model(data: pd.DataFrame) -> None:
    """Recharge le fichier .pkl et effectue une prédiction de contrôle."""

    artifact = joblib.load(MODEL_PATH)
    loaded_model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    # On teste quelques lignes du dataset traité.
    sample = data[feature_columns].head(5)
    predictions = loaded_model.predict(sample)

    print("\nTest du modèle sauvegardé (5 observations) :")
    for index, prediction in enumerate(predictions):
        label = "ATTAQUE GPS" if prediction == 1 else "SAIN"
        print(f"  Observation {index + 1}: {label}")

    print("Chargement et prédiction : OK")



# 9. POINT D'ENTRÉE PRINCIPAL


def main():
    print("=" * 58)
    print("AGENT 2 - CYBERSÉCURITÉ & ANTI-SPOOFING")
    print("=" * 58)

    # Étape 1 : charger le dataset prétraité.
    df = load_dataset(DATA_PATH)

    # Étape 2 : calculer / récupérer les quatre caractéristiques demandées.
    data = build_features(df)

    # Étape 3 : préparer les jeux d'entraînement et de test.
    X_train, X_test, y_train, y_test = split_by_drone(data)

    # Étape 4 : entraîner le modèle supervisé.
    model = train_model(X_train, y_train)

    # Étape 5 : évaluer les performances sur des drones non vus.
    metrics = evaluate_model(model, X_test, y_test)

    # Étape 6 : sauvegarder le modèle et les métriques.
    save_artifacts(model, metrics)

    # Étape 7 : vérifier que le modèle sauvegardé fonctionne.
    test_saved_model(data)

    print("\nAGENT 2 TERMINÉ AVEC SUCCÈS.")


if __name__ == "__main__":
    main()
