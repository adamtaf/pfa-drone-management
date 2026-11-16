import os
import glob
import pandas as pd
from huggingface_hub import snapshot_download

RAW_DIR = os.path.join("..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

print("--- Téléchargement optimisé par Snapshot (250 trajectoires) ---")

# 1. Téléchargement groupé ultra-rapide (évite les requêtes REST individuelles et le Rate Limit 429)
local_dataset_dir = snapshot_download(
    repo_id="riotu-lab/Synthetic-UAV-Flight-Trajectories",
    repo_type="dataset",
    allow_patterns="*.csv"
)

# 2. Récupération des 250 premiers fichiers CSV téléchargés localement
all_csv_files = sorted(glob.glob(os.path.join(local_dataset_dir, "*.csv")))
selected_files = all_csv_files[:250]

print(f"Fichiers locaux trouvés. Traitement de {len(selected_files)} trajectoires...")

dfs = []
for i, file_path in enumerate(selected_files):
    df = pd.read_csv(file_path)
    df['Drone_ID'] = i + 1  # Attribution d'un ID drone unique
    dfs.append(df)

# 3. Consolidation finale dans data/raw/
full_df = pd.concat(dfs, ignore_index=True)
output_path = os.path.join(RAW_DIR, "uav_trajectories_raw.csv")
full_df.to_csv(output_path, index=False)

print(f"\n[OK] Dataset de la flotte consolidé avec succès !")
print(f"-> Fichier généré : {output_path}")
print(f"-> Nombre de drones uniques dans la flotte : {full_df['Drone_ID'].nunique()}")
print(f"-> Volume total de points 3D enregistrés : {len(full_df):,} lignes")