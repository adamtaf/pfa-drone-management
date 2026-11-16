import os
import numpy as np
import pandas as pd

# --- CONFIGURATION DES CHEMINS & CONSTANTES ---
RAW_DATA_PATH = os.path.join("..", "data", "raw", "uav_trajectories_raw.csv")
PROCESSED_DIR = os.path.join("..", "data", "processed")
OUTPUT_CSV_PATH = os.path.join(PROCESSED_DIR, "fleet_telemetry_processed.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Paramètres de la Grille Spatiale 3D (Voxels : 10m x 10m x 5m)
X_MIN, Y_MIN, Z_MIN = -45.0, -45.0, 0.0
CELL_SIZE_XY = 10.0
CELL_SIZE_Z = 5.0
GRID_NX = 10  # Nombre de cellules en X (-45 à +45 = 90m -> 9-10 voxels)
GRID_NY = 10

# Fréquence temporelle estimée (30 Hz)
DEFAULT_DT = 1.0 / 30.0

# Seuil d'anomalie cinématique (Distance max nominale + 3 * std ≈ 1.11 m)
SPOOFING_THRESHOLD_DIST = 1.11


def compute_spatial_cell_id(x, y, z):
    """Calcule le Cell_ID unique (Spatial Hashing 3D) en complexité O(1)."""
    cx = np.floor((x - X_MIN) / CELL_SIZE_XY).astype(int)
    cy = np.floor((y - Y_MIN) / CELL_SIZE_XY).astype(int)
    cz = np.floor((z - Z_MIN) / CELL_SIZE_Z).astype(int)
    
    # Évite les débordements négatifs
    cx, cy, cz = np.maximum(0, cx), np.maximum(0, cy), np.maximum(0, cz)
    
    cell_id = cx + (cy * GRID_NX) + (cz * GRID_NX * GRID_NY)
    return cell_id


def process_fleet_telemetry():
    print(f"--- Chargement des données brutes depuis : {RAW_DATA_PATH} ---")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Nettoyage des nom de colonnes
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Tri temporel et par drone
    df = df.sort_values(by=['drone_id', 'timestamp']).reset_index(drop=True)
    
    print("1. Alignement temporel et création de la variable `frame_id`...")
    # Alignement : chaque drone commence à frame_id = 0
    df['frame_id'] = df.groupby('drone_id').cumcount()
    
    print("2. Calcul des caractéristiques cinématiques (Vitesses & Déplacements)...")
    # Différences de position
    df['dx'] = df.groupby('drone_id')['tx'].diff().fillna(0.0)
    df['dy'] = df.groupby('drone_id')['ty'].diff().fillna(0.0)
    df['dz'] = df.groupby('drone_id')['tz'].diff().fillna(0.0)
    
    # Distance euclidienne par pas de temps
    df['step_dist'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
    
    # Temps écoulé rééchantillonné
    df['dt'] = df.groupby('drone_id')['timestamp'].diff().fillna(DEFAULT_DT)
    df['dt'] = np.where(df['dt'] <= 0, DEFAULT_DT, df['dt'])
    
    # Composantes de vitesse (m/s)
    df['vx'] = df['dx'] / df['dt']
    df['vy'] = df['dy'] / df['dt']
    df['vz'] = df['dz'] / df['dt']
    df['v_norm'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    
    print("3. Indexation dans la Grille Spatiale 3D (Cell_ID)...")
    df['cell_id'] = compute_spatial_cell_id(df['tx'], df['ty'], df['tz'])
    
    print("4. Injection synthétique des pannes (Spoofing & Blackout)...")
    # Colonnes cibles initiales (Nominal = 0)
    df['is_spoofed'] = 0
    df['is_blackout'] = 0
    df['tx_measured'] = df['tx']
    df['ty_measured'] = df['ty']
    df['tz_measured'] = df['tz']
    
    np.random.seed(42)  # Reproductibilité
    
    # --- A. Injection GPS Spoofing (Agent 2) ---
    # Altération artificielle de 3% des frames en insérant un saut physique (> 1.11 m)
    spoof_indices = df.sample(frac=0.03, random_state=42).index
    offset_x = np.random.choice([-8.0, 8.0, -12.0, 12.0], size=len(spoof_indices))
    offset_y = np.random.choice([-8.0, 8.0, -12.0, 12.0], size=len(spoof_indices))
    
    df.loc[spoof_indices, 'is_spoofed'] = 1
    df.loc[spoof_indices, 'tx_measured'] += offset_x
    df.loc[spoof_indices, 'ty_measured'] += offset_y
    
    # --- B. Injection Blackout Réseau (Agent 3) ---
    # Masquage de fenêtres de 20 frames (0.6s) pour 10% des drones
    blackout_drones = df['drone_id'].drop_duplicates().sample(frac=0.10, random_state=42)
    for d_id in blackout_drones:
        drone_frames = df[df['drone_id'] == d_id]
        if len(drone_frames) > 50:
            start_frame = np.random.randint(20, len(drone_frames) - 25)
            blackout_indices = drone_frames.iloc[start_frame : start_frame + 20].index
            df.loc[blackout_indices, 'is_blackout'] = 1
            
    print("5. Exportation du jeu de données nettoyé et enrichi...")
    columns_to_export = [
        'frame_id', 'timestamp', 'drone_id', 
        'tx', 'ty', 'tz', 
        'tx_measured', 'ty_measured', 'tz_measured',
        'vx', 'vy', 'vz', 'v_norm', 'step_dist',
        'cell_id', 'is_spoofed', 'is_blackout'
    ]
    
    df_final = df[columns_to_export]
    df_final.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n--- PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS ---")
    print(f"Fichier exporté : {OUTPUT_CSV_PATH}")
    print(f"Nombre total de lignes : {len(df_final):,}")
    print(f"Nombre de détections d'attaques (Spoofed) injectées : {df_final['is_spoofed'].sum():,}")
    print(f"Nombre de frames en Blackout injectées : {df_final['is_blackout'].sum():,}")
    print("\nAperçu des 5 premières lignes :")
    print(df_final.head())


if __name__ == "__main__":
    process_fleet_telemetry()