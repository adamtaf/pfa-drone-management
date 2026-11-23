import os
import time
import json
import joblib
import warnings
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

# ==========================================
# CONFIGURATION AGENT 2 (CYBERSÉCURITÉ / ANTI-SPOOFING)
# ==========================================
DATA_PATH = "data/processed/fleet_telemetry_processed.csv"
MODEL_PATH = "models/agent2_rf_spoofing.pkl"
RESULTS_PATH = "results/agent2_spoofing_metrics.json"

DT = 0.1               # 10 Hz
TEST_SIZE = 0.20       # 20 % des drones réservés au test (GroupShuffleSplit)
SEED = 42

FEATURE_COLUMNS = [
    "step_dist_measured",
    "velocity_residual",
    "step_ratio",
    "v_norm",
    "vel_res_roll_max_5",
    "vel_res_roll_mean_5",
    "vel_res_roll_max_15",
    "vel_res_roll_mean_15",
    "drift_5",
    "drift_10",
    "drift_20",
]


# ==========================================
# 1. CHARGEMENT ET VALIDATION DU DATASET
# ==========================================
def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "drone_id",
        "frame_id",
        "timestamp",
        "tx_measured",
        "ty_measured",
        "tz_measured",
        "vx",
        "vy",
        "vz",
        "v_norm",
        "is_spoofed",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")

    df = df.sort_values(["drone_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


# ==========================================
# 2. EXTRACTION DES FEATURES TEMPORELLES ANTI-SPOOFING
# ==========================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les indicateurs physiques et temporels par drone (Zéro Data Leakage :
    n'utilise JAMAIS tx, ty, tz ni step_dist).
    """
    data = df.copy()
    grouped = data.groupby("drone_id", sort=False)

    # Pas de temps dt par drone
    if "dt" in data.columns:
        dt = pd.to_numeric(data["dt"], errors="coerce")
    else:
        dt = grouped["timestamp"].diff()
    dt = dt.where(np.isfinite(dt) & (dt > 0), DT).fillna(DT)

    # Déplacement GPS mesuré entre t-1 et t
    dx_m = (data["tx_measured"] - grouped["tx_measured"].shift(1)).fillna(0.0)
    dy_m = (data["ty_measured"] - grouped["ty_measured"].shift(1)).fillna(0.0)
    dz_m = (data["tz_measured"] - grouped["tz_measured"].shift(1)).fillna(0.0)

    data["step_dist_measured"] = np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)

    # Déplacement attendu d'après la vitesse inertielle à t-1
    prev_vx = grouped["vx"].shift(1).fillna(0.0)
    prev_vy = grouped["vy"].shift(1).fillna(0.0)
    prev_vz = grouped["vz"].shift(1).fillna(0.0)

    exp_dx = prev_vx * dt
    exp_dy = prev_vy * dt
    exp_dz = prev_vz * dt
    expected_step = np.sqrt(exp_dx**2 + exp_dy**2 + exp_dz**2)

    # Résidu instantané entre GPS mesuré et modèle cinématique
    res_x = dx_m - exp_dx
    res_y = dy_m - exp_dy
    res_z = dz_m - exp_dz
    data["velocity_residual"] = np.sqrt(res_x**2 + res_y**2 + res_z**2)

    # Ratio déplacement mesuré / déplacement attendu
    data["step_ratio"] = data["step_dist_measured"] / (expected_step + 1e-3)
    data["v_norm"] = pd.to_numeric(data["v_norm"], errors="coerce").fillna(0.0)

    # Composantes temporaires pour les fenêtres glissantes par drone
    data["_res_x"] = res_x
    data["_res_y"] = res_y
    data["_res_z"] = res_z

    g = data.groupby("drone_id", sort=False)

    # Fenêtres glissantes sur le résidu de vitesse (mémoire des sauts récents)
    for w in (5, 15):
        data[f"vel_res_roll_max_{w}"] = (
            g["velocity_residual"]
            .rolling(window=w, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )
        data[f"vel_res_roll_mean_{w}"] = (
            g["velocity_residual"]
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Dérive vectorielle cumulée (Dead-Reckoning Drift sur 5, 10 et 20 frames) :
    # Reste élevée pendant TOUT un épisode de spoofing à offset constant,
    # puis retombe à ~0 dès le retour à la position normale.
    for w in (5, 10, 20):
        sum_rx = g["_res_x"].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        sum_ry = g["_res_y"].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        sum_rz = g["_res_z"].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        data[f"drift_{w}"] = np.sqrt(sum_rx**2 + sum_ry**2 + sum_rz**2)

    data.drop(columns=["_res_x", "_res_y", "_res_z"], inplace=True)

    # Nettoyage des éventuelles valeurs infinies / NaN
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data


# ==========================================
# 3. SPLIT STRICT PAR DRONE_ID (ZERO LEAKAGE)
# ==========================================
def split_by_drone(data: pd.DataFrame):
    X = data[FEATURE_COLUMNS]
    y = data["is_spoofed"].astype(int)
    groups = data["drone_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return (
        data.iloc[train_idx].reset_index(drop=True),
        data.iloc[test_idx].reset_index(drop=True),
    )


# ==========================================
# 4. MESURE DE LATENCE TEMPS RÉEL (MS / FRAME)
# ==========================================
def measure_frame_latency(model, test_df: pd.DataFrame) -> float:
    latencies_ms = []
    for _, frame_df in test_df.groupby("frame_id", sort=False):
        X_frame = frame_df[FEATURE_COLUMNS].values
        t0 = time.perf_counter()
        _ = model.predict(X_frame)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(latencies_ms))


# ==========================================
# 5. PIPELINE PRINCIPAL D'ENTRAÎNEMENT
# ==========================================
def main():
    print("Chargement du dataset et extraction des features Anti-Spoofing...")
    t_feat = time.perf_counter()
    df = load_dataset(DATA_PATH)
    data = build_features(df)
    print(f"Extraction terminée en {time.perf_counter() - t_feat:.2f} s ({len(data)} échantillons).")

    counts = data["is_spoofed"].value_counts().sort_index().to_dict()
    print(f"Distribution des classes (0=Sain, 1=Spoofing) : {counts}")

    train_df, test_df = split_by_drone(data)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["is_spoofed"].astype(int)
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["is_spoofed"].astype(int)

    print(
        f"Split par drone_id -> Train: {train_df['drone_id'].nunique()} drones ({len(train_df)} lignes) | "
        f"Test: {test_df['drone_id'].nunique()} drones ({len(test_df)} lignes)"
    )

    # Entraînement du Random Forest
    print("\n--- Entraînement du Random Forest (Agent 2) ---")
    t_train = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_duration_s = time.perf_counter() - t_train

    # Prédictions et évaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    avg_latency_ms = measure_frame_latency(model, test_df)

    importances = {
        col: round(float(imp), 4)
        for col, imp in sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    print("\n=== RAPPORT DE CLASSIFICATION (TEST) ===")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["0: Sain", "1: Spoofing"],
            digits=4,
            zero_division=0,
        )
    )

    metrics = {
        "model_name": "Agent2_RandomForest_AntiSpoofing",
        "test_samples": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_spoofed": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall_spoofed": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_spoofed": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "average_precision": round(float(average_precision_score(y_test, y_proba)), 4),
        "avg_inference_latency_ms_per_frame": round(avg_latency_ms, 4),
        "training_time_s": round(train_duration_s, 2),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
        "feature_importances": importances,
        "features": FEATURE_COLUMNS,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "is_spoofed",
        "classes": {"0": "Sain", "1": "Attaque GPS Spoofing"},
    }
    joblib.dump(artifact, MODEL_PATH)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("=== RÉSULTATS AGENT 2 (CYBERSÉCURITÉ / ANTI-SPOOFING) ===")
    print(f"  Accuracy                         : {metrics['accuracy']:.4f}")
    print(f"  Précision (Spoofing)             : {metrics['precision_spoofed']:.4f}")
    print(f"  Recall (Spoofing)                : {metrics['recall_spoofed']:.4f}")
    print(f"  F1-Score (Spoofing)              : {metrics['f1_spoofed']:.4f}")
    print(f"  ROC-AUC                          : {metrics['roc_auc']:.4f}")
    print(f"  Latence moyenne / frame          : {metrics['avg_inference_latency_ms_per_frame']:.3f} ms")
    print(f"  Matrice de confusion             : TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"\nModèle sauvegardé dans   : {MODEL_PATH}")
    print(f"Métriques sauvegardées   : {RESULTS_PATH}")


if __name__ == "__main__":
    main()