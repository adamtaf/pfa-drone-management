import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION AGENT 3 (LSTM/GRU BLACKOUT)
# ==========================================
DATA_PATH = "data/processed/fleet_telemetry_processed.csv"
BASELINE_METRICS_PATH = "results/baseline_metrics.json"
MODEL_PATH = "models/agent3_lstm_blackout.pt"
SCALER_X_PATH = "models/agent3_scaler_x.pkl"
SCALER_Y_PATH = "models/agent3_scaler_y.pkl"
RESULTS_PATH = "results/agent3_blackout_metrics.json"

DT = 0.1               # 10 Hz
WINDOW_SIZE = 10       # 1.0 seconde d'historique
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3
PATIENCE = 7           # Early stopping
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Détection automatique du matériel (Apple Silicon MPS / CUDA / CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ==========================================
# 1. ARCHITECTURE DU RÉSEAU HYBRIDE LSTM-GRU
# ==========================================
class TrajectoryLSTMGRU(nn.Module):
    def __init__(self, input_dim=6, hidden_lstm=64, hidden_gru=32, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_lstm,
            num_layers=1,
            batch_first=True
        )
        self.gru = nn.GRU(
            input_size=hidden_lstm,
            hidden_size=hidden_gru,
            num_layers=1,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_gru, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.gru(out)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


# ==========================================
# 2. CONSTRUCTION DES FENÊTRES GLISSANTES
# ==========================================
def build_sequences_for_drones(df, drone_ids, window_size=10):
    """
    Construit des séquences (X, y) pour une liste de drone_id.
    X : (N, window_size, 6) -> [dx, dy, dz, vx, vy, vz]
    y : (N, 3)              -> [dx_next, dy_next, dz_next]
    """
    X_list, y_list = [], []

    for d_id in drone_ids:
        sub = df[df["drone_id"] == d_id].sort_values("frame_id").reset_index(drop=True)
        if len(sub) <= window_size + 1:
            continue

        pos = sub[["tx", "ty", "tz"]].values.astype(np.float32)
        vel = sub[["vx", "vy", "vz"]].values.astype(np.float32)

        # Déplacements locaux : delta_P[t] = P[t] - P[t-1]
        deltas = np.zeros_like(pos)
        deltas[1:] = pos[1:] - pos[:-1]
        deltas[0] = vel[0] * DT

        features = np.hstack([deltas, vel])  # (T, 6)

        for t in range(window_size, len(sub) - 1):
            X_list.append(features[t - window_size : t])
            y_list.append(deltas[t])  # Déplacement vers pos[t]

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ==========================================
# 3. ÉVALUATION AUTORÉGRESSIVE SUR BLACKOUT
# ==========================================
def evaluate_blackout_autoregressive(model, df, blackout_drones, scaler_x, scaler_y):
    """
    Évalue le modèle en boucle fermée sur les 25 épisodes de blackout (500 frames).
    Pendant is_blackout == 1, le modèle n'accède JAMAIS aux vraies coordonnées.
    """
    model.eval()
    errors_sq = []
    latencies_ms = []
    total_blackout_frames = 0

    with torch.no_grad():
        for d_id in blackout_drones:
            sub = df[df["drone_id"] == d_id].sort_values("frame_id").reset_index(drop=True)
            blackout_indices = sub.index[sub["is_blackout"] == 1].tolist()
            if not blackout_indices:
                continue

            start_idx = blackout_indices[0]
            if start_idx < WINDOW_SIZE:
                continue

            # Historique réel disponible AVANT le début du blackout
            pos = sub[["tx", "ty", "tz"]].values.astype(np.float32)
            vel = sub[["vx", "vy", "vz"]].values.astype(np.float32)

            deltas = np.zeros_like(pos)
            deltas[1:] = pos[1:] - pos[:-1]
            deltas[0] = vel[0] * DT
            features = np.hstack([deltas, vel])

            # Fenêtre initiale (10 dernières frames réelles avant la coupure)
            current_window = features[start_idx - WINDOW_SIZE : start_idx].copy()
            current_pos = pos[start_idx - 1].copy()

            # Prédiction pas-à-pas (autorégressive) sur toute la durée du blackout
            for idx in blackout_indices:
                t0 = time.perf_counter()

                win_scaled = scaler_x.transform(current_window)
                x_tensor = torch.tensor(win_scaled, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                pred_delta_scaled = model(x_tensor).cpu().numpy()
                pred_delta = scaler_y.inverse_transform(pred_delta_scaled)[0]

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                # Reconstruction de la position et vitesse estimées
                current_pos = current_pos + pred_delta
                est_vel = pred_delta / DT

                # Calcul de l'erreur par rapport à la vérité terrain
                true_pos = pos[idx]
                err_sq = np.sum((true_pos - current_pos) ** 2)
                errors_sq.append(err_sq)
                total_blackout_frames += 1

                # Mise à jour de la fenêtre glissante avec la prédiction (boucle fermée)
                new_step = np.concatenate([pred_delta, est_vel]).astype(np.float32)
                current_window = np.vstack([current_window[1:], new_step])

    rmse_m = float(np.sqrt(np.mean(errors_sq)))
    avg_latency_ms = float(np.mean(latencies_ms))
    return rmse_m, avg_latency_ms, total_blackout_frames


# ==========================================
# 4. PIPELINE PRINCIPAL D'ENTRAÎNEMENT
# ==========================================
def main():
    print(f"Exécution de l'Agent 3 (LSTM/GRU) sur : {DEVICE}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Séparation stricte par drone_id (Zero Data Leakage)
    blackout_drones = sorted(df[df["is_blackout"] == 1]["drone_id"].unique().tolist())
    clean_drones = sorted(list(set(df["drone_id"].unique().tolist()) - set(blackout_drones)))

    rng = np.random.default_rng(SEED)
    rng.shuffle(clean_drones)

    train_drones = clean_drones[:200]
    val_drones = clean_drones[200:]

    print(
        f"Split par drone_id -> Train: {len(train_drones)} drones | "
        f"Val: {len(val_drones)} drones | Test Blackout: {len(blackout_drones)} drones"
    )

    # Construction des séquences glissantes
    X_train, y_train = build_sequences_for_drones(df, train_drones, WINDOW_SIZE)
    X_val, y_val = build_sequences_for_drones(df, val_drones, WINDOW_SIZE)
    print(f"Séquences générées -> Train: {X_train.shape} | Val: {X_val.shape}")

    # Normalisation (fit UNIQUEMENT sur Train pour éviter toute fuite de données)
    scaler_x = StandardScaler()
    N_tr, W, F = X_train.shape
    X_train_scaled = scaler_x.fit_transform(X_train.reshape(-1, F)).reshape(N_tr, W, F)
    X_val_scaled = scaler_x.transform(X_val.reshape(-1, F)).reshape(X_val.shape)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # DataLoaders PyTorch
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train_scaled)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val_scaled)),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Initialisation du modèle
    model = TrajectoryLSTMGRU(input_dim=F).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    print("\n--- Début de l'entraînement ---")
    t_train_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                preds = model(bx)
                val_loss += criterion(preds, by).item() * bx.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping déclenché à l'epoch {epoch}.")
                break

    train_duration_s = time.perf_counter() - t_train_start
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    # Chargement du meilleur modèle et évaluation sur les 500 frames de Blackout
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    rmse_lstm, avg_latency_ms, n_frames = evaluate_blackout_autoregressive(
        model, df, blackout_drones, scaler_x, scaler_y
    )

    # Comparaison avec la Baseline
    baseline_rmse = 2.8425
    if os.path.exists(BASELINE_METRICS_PATH):
        with open(BASELINE_METRICS_PATH, "r") as f:
            baseline_rmse = json.load(f).get("blackout_linear_prediction_rmse_m", 2.8425)

    gain_pct = ((baseline_rmse - rmse_lstm) / baseline_rmse) * 100.0

    metrics = {
        "model_name": "Agent3_LSTM_GRU_Blackout",
        "device": str(DEVICE),
        "window_size_frames": WINDOW_SIZE,
        "blackout_episodes_evaluated": len(blackout_drones),
        "blackout_frames_evaluated": int(n_frames),
        "baseline_blackout_rmse_m": round(baseline_rmse, 4),
        "agent3_lstm_blackout_rmse_m": round(rmse_lstm, 4),
        "rmse_reduction_pct": round(gain_pct, 2),
        "avg_inference_latency_ms": round(avg_latency_ms, 4),
        "training_time_s": round(train_duration_s, 2)
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n=== RÉSULTATS AGENT 3 (LSTM/GRU BLACKOUT) ===")
    for k, v in metrics.items():
        print(f"  {k:<32} : {v}")
    print(f"\nModèle sauvegardé dans   : {MODEL_PATH}")
    print(f"Métriques sauvegardées   : {RESULTS_PATH}")


if __name__ == "__main__":
    main()