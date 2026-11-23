import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# CONFIGURATION AGENT 1 (PRIORISATION RF)
# ==========================================
DATA_PATH = "data/processed/fleet_telemetry_processed.csv"
MODEL_PATH = "models/agent1_rf_prioritization.pkl"
RESULTS_PATH = "results/agent1_prioritization_metrics.json"

# Paramètres de la Grille 3D (identiques à la Baseline)
X_MIN, Y_MIN, Z_MIN = -45.0, -45.0, 0.0
VOXEL_XY = 10.0
VOXEL_Z = 5.0

# Seuils physiques de sécurité
COLLISION_DIST = 2.0   # Critique (P0)
CONFLICT_DIST = 5.0    # Haute (P1)
TTC_CRITICAL = 1.0     # Seconde avant impact (Critique)
TTC_HIGH = 2.5         # Seconde avant impact (Haute)
MAX_DIST_CAP = 20.0    # Distance plafond si aucun voisin proche

SEED = 42


def get_voxel_coords(x, y, z):
    return (
        int(np.floor((x - X_MIN) / VOXEL_XY)),
        int(np.floor((y - Y_MIN) / VOXEL_XY)),
        int(np.floor((z - Z_MIN) / VOXEL_Z))
    )


# ==========================================
# 1. EXTRACTION DES FEATURES (GRILLE 3D)
# ==========================================
def extract_prioritization_features(df):
    """
    Calcule pour chaque drone à chaque frame_id :
      - d_min         : distance euclidienne au plus proche voisin (m)
      - v_close       : vitesse de rapprochement vers ce voisin (m/s)
      - voxel_density : nombre de drones présents dans le même voxel 3D
      - v_norm        : norme de la vitesse du drone (m/s)
    Et génère la cible :
      - urgency_level : 0 (Normale), 1 (Haute), 2 (Critique)
    """
    records = []

    for frame_id, group in df.groupby("frame_id"):
        buckets = {}
        drones = []

        for row in group.itertuples():
            pos = np.array([row.tx, row.ty, row.tz], dtype=np.float64)
            vel = np.array([row.vx, row.vy, row.vz], dtype=np.float64)
            v_norm = float(np.linalg.norm(vel))
            voxel = get_voxel_coords(pos[0], pos[1], pos[2])

            buckets.setdefault(voxel, []).append((row.drone_id, pos, vel))
            drones.append((row.drone_id, pos, vel, v_norm, voxel))

        for d_id, p1, v1, v_norm, (cx, cy, cz) in drones:
            voxel_density = len(buckets[(cx, cy, cz)])
            d_min = MAX_DIST_CAP
            v_close = 0.0

            # Recherche du plus proche voisin dans les 27 voxels (courant + 26 adjacents)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nv = (cx + dx, cy + dy, cz + dz)
                        if nv not in buckets:
                            continue
                        for other_id, p2, v2 in buckets[nv]:
                            if other_id == d_id:
                                continue
                            rel_pos = p2 - p1
                            dist = float(np.linalg.norm(rel_pos))
                            if dist < d_min:
                                d_min = dist
                                rel_vel = v2 - v1
                                # Vitesse de rapprochement positive si les drones se rapprochent
                                closing = -float(np.dot(rel_pos, rel_vel)) / (dist + 1e-6)
                                v_close = max(0.0, closing)

            # Calcul du Time-To-Collision (TTC) pour la labellisation physique
            ttc = d_min / (v_close + 1e-6) if v_close > 0.1 else 999.0

            if d_min < COLLISION_DIST or (d_min < CONFLICT_DIST and ttc < TTC_CRITICAL):
                urgency = 2  # Critique (P0)
            elif d_min < CONFLICT_DIST or (d_min < 8.0 and ttc < TTC_HIGH):
                urgency = 1  # Haute (P1)
            else:
                urgency = 0  # Normale (P2)

            records.append({
                "frame_id": frame_id,
                "drone_id": d_id,
                "d_min": d_min,
                "v_close": v_close,
                "voxel_density": voxel_density,
                "v_norm": v_norm,
                "urgency_level": urgency
            })

    return pd.DataFrame(records)


# ==========================================
# 2. PIPELINE D'ENTRAÎNEMENT ET D'ÉVALUATION
# ==========================================
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    print("Chargement du dataset et extraction des features 3D...")
    t0 = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    feat_df = extract_prioritization_features(df)
    extract_time_s = time.perf_counter() - t0
    print(f"Extraction terminée en {extract_time_s:.2f} s ({len(feat_df)} échantillons).")

    # Distribution des classes
    class_counts = feat_df["urgency_level"].value_counts().sort_index().to_dict()
    print(f"Distribution des classes (0=Normale, 1=Haute, 2=Critique) : {class_counts}")

    # Découpage temporel strict par frame_id (80% Train / 20% Test)
    unique_frames = sorted(feat_df["frame_id"].unique())
    split_idx = int(len(unique_frames) * 0.80)
    train_frames = set(unique_frames[:split_idx])
    test_frames = set(unique_frames[split_idx:])

    train_df = feat_df[feat_df["frame_id"].isin(train_frames)]
    test_df = feat_df[feat_df["frame_id"].isin(test_frames)]

    feature_cols = ["d_min", "v_close", "voxel_density", "v_norm"]
    X_train, y_train = train_df[feature_cols].values, train_df["urgency_level"].values
    X_test, y_test = test_df[feature_cols].values, test_df["urgency_level"].values

    print(
        f"Split temporel -> Train: {len(train_frames)} frames ({len(train_df)} lignes) | "
        f"Test: {len(test_frames)} frames ({len(test_df)} lignes)"
    )

    # Entraînement du Random Forest
    print("\n--- Entraînement du Random Forest (Agent 1) ---")
    t_train = time.perf_counter()
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    train_duration_s = time.perf_counter() - t_train

    # Prédictions globales sur le jeu de test
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    recall_per_class = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

    # Mesure de la latence temps réel (inférence par frame complète de ~175 drones)
    frame_latencies_ms = []
    for _, f_group in test_df.groupby("frame_id"):
        X_frame = f_group[feature_cols].values
        t_inf = time.perf_counter()
        _ = clf.predict(X_frame)
        frame_latencies_ms.append((time.perf_counter() - t_inf) * 1000.0)

    avg_frame_latency_ms = float(np.mean(frame_latencies_ms))

    # Importance des variables
    importances = {
        col: round(float(imp), 4)
        for col, imp in zip(feature_cols, clf.feature_importances_)
    }

    # Sauvegarde du modèle et des métriques
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    metrics = {
        "model_name": "Agent1_RandomForest_Prioritization",
        "features": feature_cols,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "class_distribution_total": {str(k): int(v) for k, v in class_counts.items()},
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "recall_class_0_normale": round(float(recall_per_class[0]), 4),
        "recall_class_1_haute": round(float(recall_per_class[1]), 4),
        "recall_class_2_critique": round(float(recall_per_class[2]), 4),
        "confusion_matrix": cm,
        "feature_importances": importances,
        "avg_inference_latency_per_frame_ms": round(avg_frame_latency_ms, 4),
        "training_time_s": round(train_duration_s, 2)
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n=== RAPPORT DE CLASSIFICATION (TEST) ===")
    print(classification_report(
        y_test, y_pred, target_names=["0: Normale", "1: Haute", "2: Critique"], digits=4
    ))

    print("=== RÉSULTATS AGENT 1 (PRIORISATION) ===")
    print(f"  Accuracy                         : {metrics['accuracy']}")
    print(f"  F1-Score (Macro)                 : {metrics['f1_macro']}")
    print(f"  Recall Classe 2 (Critique / P0)  : {metrics['recall_class_2_critique']}")
    print(f"  Latence moyenne / frame          : {metrics['avg_inference_latency_per_frame_ms']} ms")
    print(f"  Importance des variables         : {metrics['feature_importances']}")
    print(f"\nModèle sauvegardé dans   : {MODEL_PATH}")
    print(f"Métriques sauvegardées   : {RESULTS_PATH}")


if __name__ == "__main__":
    main()