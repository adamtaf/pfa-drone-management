import time
import json
import os
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION DU SYSTÈME BASELINE
# ==========================================
DATA_PATH = "data/processed/fleet_telemetry_processed.csv"
RESULTS_PATH = "results/baseline_metrics.json"

DT = 0.1                      # Fréquence réelle du dataset : 10 Hz (0.1 s par pas de temps)
COLLISION_DIST = 2.0          # Distance de crash (mètres)
CONFLICT_DIST = 5.0           # Bulle de sécurité minimale (mètres)

# Paramètres de la Grille Spatiale 3D (90m x 90m x 25m)
X_MIN, Y_MIN, Z_MIN = -45.0, -45.0, 0.0
VOXEL_XY, VOXEL_Z = 10.0, 5.0


def get_voxel_coords(x, y, z):
    """Convertit des coordonnées (x, y, z) en indices de cellule 3D."""
    cx = int(np.floor((x - X_MIN) / VOXEL_XY))
    cy = int(np.floor((y - Y_MIN) / VOXEL_XY))
    cz = int(np.floor((z - Z_MIN) / VOXEL_Z))
    return (cx, cy, cz)


def run_baseline_evaluation(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.sort_values(by=["frame_id", "drone_id"]).reset_index(drop=True)

    # Mémoire interne du système de contrôle (dernier état connu par drone)
    last_known_state = {}

    # Compteurs de métriques
    total_observations = 0
    total_collisions = 0
    total_conflicts = 0
    spoofing_errors_sq = []
    blackout_errors_sq = []
    frame_latencies_ms = []

    frames = df["frame_id"].unique()
    print(
        f"Lancement de la simulation Baseline sur {len(frames)} pas de temps "
        f"({df['drone_id'].nunique()} drones | {len(df)} enregistrements)..."
    )

    for frame_id in frames:
        t_start = time.perf_counter()
        frame_data = df[df["frame_id"] == frame_id]
        total_observations += len(frame_data)

        # Structures pour la grille spatiale 3D du pas de temps courant
        spatial_buckets = {}
        active_drones = []

        for row in frame_data.itertuples():
            d_id = row.drone_id
            true_pos = np.array([row.tx, row.ty, row.tz])

            # 1. ESTIMATION DE POSITION (Modèle Linéaire Naïf)
            if row.is_blackout == 0:
                # Signal reçu : confiance aveugle aux capteurs (pas d'Agent 2)
                perceived_pos = np.array([row.tx_measured, row.ty_measured, row.tz_measured])
                perceived_vel = np.array([row.vx, row.vy, row.vz])
                last_known_state[d_id] = (perceived_pos, perceived_vel)
            else:
                # Coupure réseau (Blackout) : extrapolation linéaire P(t) = P(t-1) + V * dt
                if d_id in last_known_state:
                    prev_pos, prev_vel = last_known_state[d_id]
                    perceived_pos = prev_pos + prev_vel * DT
                    last_known_state[d_id] = (perceived_pos, prev_vel)
                else:
                    perceived_pos = true_pos

                err_sq = np.sum((true_pos - perceived_pos) ** 2)
                blackout_errors_sq.append(err_sq)

            # Erreur induite par le GPS Spoofing non détecté
            if row.is_spoofed == 1:
                spoof_err_sq = np.sum((true_pos - perceived_pos) ** 2)
                spoofing_errors_sq.append(spoof_err_sq)

            # 2. INDEXATION DANS LA GRILLE 3D
            voxel = get_voxel_coords(true_pos[0], true_pos[1], true_pos[2])
            spatial_buckets.setdefault(voxel, []).append((d_id, true_pos))
            active_drones.append((d_id, true_pos, voxel))

        # 3. DÉTECTION DES COLLISIONS & CONFLITS (Voisinage 3D O(1))
        checked_pairs = set()
        for d_id, pos_a, (cx, cy, cz) in active_drones:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_voxel = (cx + dx, cy + dy, cz + dz)
                        if neighbor_voxel not in spatial_buckets:
                            continue

                        for other_id, pos_b in spatial_buckets[neighbor_voxel]:
                            if d_id >= other_id:
                                continue
                            pair = (d_id, other_id)
                            if pair in checked_pairs:
                                continue
                            checked_pairs.add(pair)

                            dist = np.linalg.norm(pos_a - pos_b)
                            if dist < COLLISION_DIST:
                                total_collisions += 1
                            elif dist < CONFLICT_DIST:
                                total_conflicts += 1

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        frame_latencies_ms.append(elapsed_ms)

    # ==========================================
    # CALCUL ET SAUVEGARDE DES KPIs BASELINE
    # ==========================================
    rmse_blackout = float(np.sqrt(np.mean(blackout_errors_sq))) if blackout_errors_sq else 0.0
    rmse_spoofing = float(np.sqrt(np.mean(spoofing_errors_sq))) if spoofing_errors_sq else 0.0
    avg_latency = float(np.mean(frame_latencies_ms))

    metrics = {
        "model_name": "Baseline_Linear_No_Agents",
        "total_time_steps_frames": int(len(frames)),
        "total_telemetry_records_evaluated": int(total_observations),
        "total_collisions_lt_2m": int(total_collisions),
        "total_conflicts_lt_5m": int(total_conflicts),
        "spoofing_detection_rate_pct": 0.0,
        "spoofing_localization_rmse_m": round(rmse_spoofing, 4),
        "blackout_linear_prediction_rmse_m": round(rmse_blackout, 4),
        "avg_step_latency_ms": round(avg_latency, 4)
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n=== RÉSULTATS DE LA BASELINE (RÉFÉRENCE) ===")
    for k, v in metrics.items():
        print(f"  {k:<36} : {v}")
    print(f"\nMétriques sauvegardées dans : {RESULTS_PATH}")
    return metrics


if __name__ == "__main__":
    run_baseline_evaluation(DATA_PATH)